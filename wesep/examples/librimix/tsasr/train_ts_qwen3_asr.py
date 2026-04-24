from __future__ import annotations

import argparse
import logging
import math
import os
import re
import shutil
import sys
from pprint import pformat
from typing import Optional

import torch
from qwen_asr import Qwen3ASRModel
from torch.optim import AdamW
from torch.utils.data import Subset
from transformers import GenerationConfig, Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler, LengthGroupedSampler
import yaml

from dynatar_qwen.data import DataCollatorForDynaTaRQwen, TSASRShardDataset
from dynatar_qwen.modeling_ts_qwen3_asr import TargetAwareEncoderConfig, upgrade_qwen3_asr_model

MAX_NUM_LOG_FILES = 100
LOGGER: Optional[logging.Logger] = None


def distributed_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def distributed_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return distributed_rank() == 0


def configure_logger(output_dir: str) -> logging.Logger:
    rank = distributed_rank()
    logger_name = f"tsasr.train.rank{rank}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "train.log")
        for i in range(MAX_NUM_LOG_FILES - 1, -1, -1):
            if i == 0:
                src = log_path
                dst = os.path.join(output_dir, "train.1.log")
            else:
                src = os.path.join(output_dir, f"train.{i}.log")
                dst = os.path.join(output_dir, f"train.{i + 1}.log")
            if os.path.exists(src):
                if i == MAX_NUM_LOG_FILES - 1:
                    os.remove(src)
                else:
                    shutil.move(src, dst)

        formatter = logging.Formatter("[ %(levelname)s : %(asctime)s ] - %(message)s")
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    else:
        logger.addHandler(logging.NullHandler())

    return logger


def main_print(message: str = ""):
    if is_main_process():
        if LOGGER is not None:
            LOGGER.info(message)
        else:
            print(message, flush=True)


def stage_print(title: str):
    main_print(title)


def write_run_config(output_dir: str, args, extra_fields: dict):
    if not is_main_process():
        return
    config_payload = dict(vars(args))
    config_payload.update(extra_fields)
    run_config_path = os.path.join(output_dir, "run_config.yaml")
    with open(run_config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config_payload, handle, sort_keys=False, allow_unicode=True)
    main_print(f"saved run config to: {run_config_path}")


def maybe_dataset_len(dataset) -> int:
    if dataset is None:
        return 0
    return len(dataset)


def describe_dataset(
    name: str,
    dataset,
    per_device_batch_size: int,
    world_size: int,
    grad_acc: int,
):
    if dataset is None:
        main_print(f"{name}: disabled")
        return

    sample_count = len(dataset)
    global_micro_batch = max(1, per_device_batch_size * max(1, world_size))
    optimizer_batch = max(1, global_micro_batch * max(1, grad_acc))
    micro_steps = math.ceil(sample_count / global_micro_batch)
    optimizer_steps = math.ceil(sample_count / optimizer_batch)
    main_print(f"{name}: {sample_count} samples")
    if hasattr(dataset, "index_cache_path"):
        cache_path = getattr(dataset, "index_cache_path", "")
        cache_source = getattr(dataset, "index_source", "unknown")
        main_print(f"{name} shard index: {cache_path} ({cache_source})")
    main_print(f"{name} micro-batches/epoch: {micro_steps}")
    if name.lower().startswith("train"):
        main_print(f"{name} optimizer-steps/epoch: {optimizer_steps}")


def log_model_summary(model):
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen_params = total_params - trainable_params
    added_prefixes = (
        "thinker.audio_tower.mhfa_backend",
        "thinker.audio_tower.prototype_projector",
        "thinker.audio_tower.refinement_blocks",
        "thinker.audio_tower.mix_memory_proj",
        "thinker.audio_tower.memory_to_hidden",
        "thinker.audio_tower.mix_summary_attn",
        "thinker.audio_tower.mix_summary_proj",
        "thinker.audio_tower.overlap_head",
    )
    added_param_breakdown = {prefix: 0 for prefix in added_prefixes}
    for name, parameter in model.named_parameters():
        for prefix in added_prefixes:
            if name.startswith(prefix):
                added_param_breakdown[prefix] += parameter.numel()
                break
    added_params = sum(added_param_breakdown.values())
    main_print(f"model size: {total_params / 1e6:.2f} M parameters")
    main_print(f"trainable parameters: {trainable_params / 1e6:.2f} M")
    main_print(f"frozen parameters: {frozen_params / 1e6:.2f} M")
    main_print(f"added target-aware parameters: {added_params / 1e6:.2f} M")
    main_print(
        "added target-aware breakdown: "
        f"mhfa={added_param_breakdown['thinker.audio_tower.mhfa_backend'] / 1e6:.2f} M, "
        f"prototype={added_param_breakdown['thinker.audio_tower.prototype_projector'] / 1e6:.2f} M, "
        f"refinement={added_param_breakdown['thinker.audio_tower.refinement_blocks'] / 1e6:.2f} M, "
        f"mix_memory_proj={added_param_breakdown['thinker.audio_tower.mix_memory_proj'] / 1e6:.2f} M, "
        f"memory_to_hidden={added_param_breakdown['thinker.audio_tower.memory_to_hidden'] / 1e6:.2f} M"
    )


def log_runtime_overview(
    args,
    use_bf16: bool,
    world_size: int,
    requested_gradient_checkpointing: bool,
    effective_gradient_checkpointing: bool,
    gradient_checkpointing_note: str = "",
):
    global_micro_batch = args.batch_size * world_size
    effective_batch = global_micro_batch * args.grad_acc
    main_print(f"output_dir: {args.output_dir}")
    main_print(f"model_path: {args.model_path}")
    main_print(f"world_size: {world_size}")
    main_print(f"precision: {'bf16' if use_bf16 else 'fp16'}")
    main_print(f"per_device_train_batch_size: {args.batch_size}")
    main_print(f"per_device_eval_batch_size: {args.eval_batch_size}")
    main_print(f"global_micro_batch_size: {global_micro_batch}")
    main_print(f"effective_batch_size: {effective_batch}")
    main_print(f"gradient_accumulation_steps: {args.grad_acc}")
    main_print(f"requested_gradient_checkpointing: {requested_gradient_checkpointing}")
    main_print(f"effective_gradient_checkpointing: {effective_gradient_checkpointing}")
    if gradient_checkpointing_note:
        main_print(f"gradient_checkpointing_note: {gradient_checkpointing_note}")
    main_print(f"group_by_length: {bool(args.group_by_length)}")
    main_print(f"enroll_crop_seconds: {args.enroll_crop_seconds}")
    main_print(f"silence_seconds: {args.silence_seconds}")
    main_print(f"enable_overlap_head: {bool(args.enable_overlap_head)}")
    main_print(f"enable_target_consistency: {bool(args.enable_target_consistency)}")
    main_print(f"enable_router_supervision: {bool(args.enable_router_supervision)}")


def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


def dataset_length_hints(dataset) -> Optional[list[int]]:
    if dataset is None:
        return None
    if isinstance(dataset, Subset):
        base_lengths = dataset_length_hints(dataset.dataset)
        if base_lengths is None:
            return None
        return [int(base_lengths[index]) for index in dataset.indices]
    lengths = getattr(dataset, "lengths", None)
    if lengths is None:
        return None
    return [int(length) for length in lengths]


class CastFloatInputsTrainer(Trainer):
    def __init__(
        self,
        *args,
        use_custom_optimizer: bool = False,
        high_lr_prefixes: Optional[list[str]] = None,
        high_lr_multiplier: float = 1.0,
        **kwargs,
    ):
        self.use_custom_optimizer = use_custom_optimizer
        self.high_lr_prefixes = tuple(high_lr_prefixes or [])
        self.high_lr_multiplier = high_lr_multiplier
        self._logged_eval_sampler_note = False
        super().__init__(*args, **kwargs)

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for key, value in list(inputs.items()):
                if torch.is_tensor(value) and value.is_floating_point():
                    inputs[key] = value.to(dtype=model_dtype)
        return inputs

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if not getattr(self.args, "group_by_length", False):
            return super()._get_train_sampler(train_dataset)
        if dataset is None:
            return None

        lengths = dataset_length_hints(dataset)
        if not lengths:
            if self.args.process_index == 0:
                main_print(
                    "[sampler] group_by_length requested but no dataset length hints were found; fallback to default sampler."
                )
            return super()._get_train_sampler(train_dataset)

        batch_size = self._train_batch_size or self.args.train_batch_size
        if self.args.world_size <= 1:
            if self.args.process_index == 0:
                main_print(f"[sampler] length bucketing enabled for training (batch_size={batch_size}).")
            return LengthGroupedSampler(
                batch_size=batch_size,
                dataset=dataset,
                lengths=lengths,
                model_input_name=None,
            )

        if self.args.process_index == 0:
            main_print(
                "[sampler] distributed length bucketing enabled for training "
                f"(batch_size={batch_size}, world_size={self.args.world_size})."
            )
        return DistributedLengthGroupedSampler(
            batch_size=batch_size,
            dataset=dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            seed=self.args.seed,
            drop_last=self.args.dataloader_drop_last,
            lengths=lengths,
            model_input_name=None,
        )

    def _get_eval_sampler(self, eval_dataset):
        if getattr(self.args, "group_by_length", False) and not self._logged_eval_sampler_note:
            if self.args.process_index == 0:
                main_print("[sampler] eval uses the default sampler; length bucketing is training-only.")
            self._logged_eval_sampler_note = True
        original = getattr(self.args, "group_by_length", False)
        try:
            self.args.group_by_length = False
            return super()._get_eval_sampler(eval_dataset)
        finally:
            self.args.group_by_length = original

    def _get_test_sampler(self, test_dataset):
        original = getattr(self.args, "group_by_length", False)
        try:
            self.args.group_by_length = False
            return super()._get_test_sampler(test_dataset)
        finally:
            self.args.group_by_length = original

    def _get_unwrapped_model(self):
        model = self.model
        while hasattr(model, "module"):
            model = model.module
        return model

    def _collect_tsasr_debug_logs(self):
        model = self._get_unwrapped_model()
        thinker = getattr(model, "thinker", model)
        audio_tower = getattr(thinker, "audio_tower", None)
        debug = getattr(audio_tower, "_last_debug", {}) or {}
        logs = {}

        token_lengths = debug.get("token_lengths")
        if token_lengths is not None and hasattr(token_lengths, "tolist"):
            token_lengths = token_lengths.tolist()
            if len(token_lengths) >= 3:
                logs["tsasr/enroll_tokens"] = float(token_lengths[0])
                logs["tsasr/silence_tokens"] = float(token_lengths[1])
                logs["tsasr/mix_tokens"] = float(token_lengths[2])

        gate_mean = debug.get("gate_mean")
        if gate_mean is not None:
            logs["tsasr/gate_mean"] = float(gate_mean)

        retrieve_entropy = debug.get("retrieve_entropy")
        if retrieve_entropy is not None:
            logs["tsasr/retrieve_entropy"] = float(retrieve_entropy)

        retrieve_max_prob = debug.get("retrieve_max_prob")
        if retrieve_max_prob is not None:
            logs["tsasr/retrieve_max_prob"] = float(retrieve_max_prob)

        router_probs = debug.get("router_probs")
        if router_probs is not None and hasattr(router_probs, "numel") and router_probs.numel() > 0:
            router_probs = router_probs.float()
            mean_probs = router_probs.mean(dim=tuple(range(router_probs.ndim - 1)))
            router_entropy = -(
                router_probs.clamp_min(1.0e-8).log() * router_probs
            ).sum(dim=-1).mean()
            logs["tsasr/router_entropy"] = float(router_entropy.item())
            for idx, prob in enumerate(mean_probs.tolist()):
                logs[f"tsasr/router_expert_{idx}_prob"] = float(prob)

        aux_logs = getattr(thinker, "_last_aux_logs", {}) or {}
        for key, value in aux_logs.items():
            logs[key] = float(value)

        return logs

    def log(self, logs, start_time=None):
        merged_logs = dict(logs)
        merged_logs.update(self._collect_tsasr_debug_logs())
        if self.args.process_index == 0:
            main_print(f"[trainer.log] {merged_logs}")
        return super().log(merged_logs, start_time=start_time)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        if (
            not self.use_custom_optimizer
            or not self.high_lr_prefixes
            or self.high_lr_multiplier <= 1.0
        ):
            return super().create_optimizer()

        low_lr_params = []
        high_lr_params = []
        low_lr_names = []
        high_lr_names = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if any(name.startswith(prefix) for prefix in self.high_lr_prefixes):
                high_lr_params.append(parameter)
                high_lr_names.append(name)
            else:
                low_lr_params.append(parameter)
                low_lr_names.append(name)

        if not high_lr_params:
            main_print(
                "[optimizer] no parameters matched the high-lr prefixes; "
                "falling back to the default optimizer."
            )
            return super().create_optimizer()

        param_groups = []
        if low_lr_params:
            param_groups.append(
                {
                    "params": low_lr_params,
                    "lr": self.args.learning_rate,
                    "weight_decay": self.args.weight_decay,
                }
            )
        param_groups.append(
            {
                "params": high_lr_params,
                "lr": self.args.learning_rate * self.high_lr_multiplier,
                "weight_decay": 0.0,
            }
        )
        self.optimizer = AdamW(
            param_groups,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        main_print(
            "[optimizer] custom dual-lr enabled: "
            f"base_lr={self.args.learning_rate:g}, "
            f"high_lr={self.args.learning_rate * self.high_lr_multiplier:g}, "
            f"high_lr_multiplier={self.high_lr_multiplier:g}"
        )
        main_print(
            "[optimizer] param groups: "
            f"low_lr={len(low_lr_names)} tensors / {sum(p.numel() for p in low_lr_params)} params, "
            f"high_lr={len(high_lr_names)} tensors / {sum(p.numel() for p in high_lr_params)} params"
        )
        main_print(f"[optimizer] high-lr prefixes: {', '.join(self.high_lr_prefixes)}")
        return self.optimizer


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        match = _CKPT_RE.match(name)
        if not match:
            continue
        step = int(match.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


def copy_required_hf_files_for_qwen_asr(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]
    for file_name in required:
        src = os.path.join(src_dir, file_name)
        dst = os.path.join(dst_dir, file_name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        copy_required_hf_files_for_qwen_asr(self.base_model_path, checkpoint_dir)
        return control


def maybe_subset_dataset(dataset, max_samples: int):
    if dataset is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, range(max_samples))


def parse_prefix_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def freeze_text_decoder(model) -> int:
    frozen = 0
    for name, parameter in model.named_parameters():
        if (
            ".thinker.model." in name
            or name.startswith("thinker.model.")
            or ".thinker.lm_head." in name
            or name.startswith("thinker.lm_head.")
        ):
            parameter.requires_grad = False
            frozen += parameter.numel()
    return frozen


def parse_args():
    parser = argparse.ArgumentParser("Train DynaTaR-Qwen TS-ASR")
    parser.add_argument("--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--local-rank", dest="local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--data_type", type=str, default="shard", choices=["shard"])
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, default="")
    parser.add_argument("--train_spk2utt", type=str, default="")
    parser.add_argument("--train_single_wav_scp", type=str, default="")
    parser.add_argument("--eval_spk1_enroll", type=str, default="")
    parser.add_argument("--eval_spk2_enroll", type=str, default="")
    parser.add_argument("--eval_enroll_paths_json", type=str, default="")
    parser.add_argument("--eval_spk2utt", type=str, default="")
    parser.add_argument("--train_aux_label_manifest", type=str, default="")
    parser.add_argument("--eval_aux_label_manifest", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./dynatar-qwen-out")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--silence_seconds", type=float, default=1.0)
    parser.add_argument("--enroll_crop_seconds", type=float, default=0.0)
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--role_layers", type=int, default=5)
    parser.add_argument("--num_prototypes", type=int, default=8)
    parser.add_argument("--mhfa_heads", type=int, default=4)
    parser.add_argument("--mhfa_compression_dim", type=int, default=128)
    parser.add_argument("--router_hidden_multiplier", type=int, default=2)
    parser.add_argument("--refinement_variant", type=str, default="full")
    parser.add_argument("--shared_refinement", type=int, default=0)
    parser.add_argument("--memory_dim", type=int, default=256)
    parser.add_argument("--expert_rank", type=int, default=64)
    parser.add_argument("--num_experts", type=int, default=3)
    parser.add_argument("--refinement_layer_strategy", type=str, default="post_role_all")
    parser.add_argument("--enable_overlap_head", type=int, default=0)
    parser.add_argument("--overlap_num_classes", type=int, default=4)
    parser.add_argument("--overlap_loss_weight", type=float, default=0.10)
    parser.add_argument("--enable_target_consistency", type=int, default=0)
    parser.add_argument("--target_consistency_weight", type=float, default=0.05)
    parser.add_argument("--target_consistency_mode", type=str, default="hybrid")
    parser.add_argument("--target_consistency_temperature", type=float, default=0.07)
    parser.add_argument("--target_consistency_detach_target", type=int, default=1)
    parser.add_argument("--enable_router_supervision", type=int, default=0)
    parser.add_argument("--router_loss_weight", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=12)
    parser.add_argument("--lr", type=float, default=4e-6)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--pin_memory", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--group_by_length", type=int, default=1)
    parser.add_argument("--use_custom_optimizer", type=int, default=1)
    parser.add_argument("--new_module_lr_multiplier", type=float, default=50.0)
    parser.add_argument(
        "--new_module_prefixes",
        type=str,
        default=(
            "thinker.audio_tower.mhfa_backend,"
            "thinker.audio_tower.prototype_projector,"
            "thinker.audio_tower.refinement_blocks,"
            "thinker.audio_tower.mix_summary_attn,"
            "thinker.audio_tower.mix_summary_proj,"
            "thinker.audio_tower.overlap_head"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument(
        "--freeze_text_decoder",
        type=int,
        default=1,
        help="Deprecated compatibility flag. The text decoder is always frozen during TS-ASR training.",
    )
    parser.add_argument("--gradient_checkpointing", type=int, default=1)
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval_strategy", type=str, default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--logging_dir", type=str, default="")
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--resume", type=int, default=0)
    return parser.parse_args()


def build_dataset(args, split: str):
    data_file = args.train_file if split == "train" else args.eval_file
    if not data_file:
        return None
    if split == "train":
        if not args.train_spk2utt:
            raise ValueError("--train_spk2utt is required when --data_type shard")
        return TSASRShardDataset(
            data_file,
            split="train",
            sampling_rate=args.sr,
            language=args.language,
            prompt=args.prompt,
            enroll_crop_seconds=args.enroll_crop_seconds,
            train_spk2utt=args.train_spk2utt,
            train_single_wav_scp=args.train_single_wav_scp,
            aux_label_manifest=args.train_aux_label_manifest,
        )
    if not args.eval_enroll_paths_json and not (args.eval_spk1_enroll and args.eval_spk2_enroll and args.eval_spk2utt):
        raise ValueError(
            "--eval_enroll_paths_json or (--eval_spk1_enroll, --eval_spk2_enroll, --eval_spk2utt) "
            "is required when --data_type shard"
        )
    return TSASRShardDataset(
        data_file,
        split="eval",
        sampling_rate=args.sr,
        language=args.language,
        prompt=args.prompt,
        enroll_crop_seconds=args.enroll_crop_seconds,
        eval_spk1_enroll=args.eval_spk1_enroll,
        eval_spk2_enroll=args.eval_spk2_enroll,
        eval_enroll_paths_json=args.eval_enroll_paths_json,
        eval_spk2utt=args.eval_spk2utt,
        aux_label_manifest=args.eval_aux_label_manifest,
    )


def main():
    global LOGGER
    args = parse_args()
    if (args.enable_overlap_head == 1 or args.enable_router_supervision == 1) and not args.train_aux_label_manifest:
        main_process_hint = " --train_aux_label_manifest is required when overlap/router supervision is enabled."
        raise ValueError(main_process_hint.strip())
    os.makedirs(args.output_dir, exist_ok=True)
    LOGGER = configure_logger(args.output_dir)

    stage_print("<== Passed Arguments ==>")
    for line in pformat(vars(args)).split("\n"):
        main_print(line)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    world_size = distributed_world_size()
    requested_gradient_checkpointing = args.gradient_checkpointing == 1
    effective_gradient_checkpointing = requested_gradient_checkpointing and world_size == 1
    gradient_checkpointing_note = ""
    if requested_gradient_checkpointing and not effective_gradient_checkpointing:
        gradient_checkpointing_note = (
            "auto-disabled because multi-GPU DDP with reentrant checkpoint backward "
            f"is unstable in this recipe (world_size={world_size})"
        )
    stage_print("<== Runtime ==>")
    log_runtime_overview(
        args,
        use_bf16,
        world_size=world_size,
        requested_gradient_checkpointing=requested_gradient_checkpointing,
        effective_gradient_checkpointing=effective_gradient_checkpointing,
        gradient_checkpointing_note=gradient_checkpointing_note,
    )

    stage_print("<== Model ==>")
    main_print("loading base Qwen3-ASR model ...")
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        attn_implementation="eager",
        device_map=None,
    )
    model = upgrade_qwen3_asr_model(
        asr_wrapper.model,
        TargetAwareEncoderConfig(
            num_role_layers=args.role_layers,
            num_prototypes=args.num_prototypes,
            mhfa_heads=args.mhfa_heads,
            mhfa_compression_dim=args.mhfa_compression_dim,
            router_hidden_multiplier=args.router_hidden_multiplier,
            refinement_variant=args.refinement_variant,
            shared_refinement=args.shared_refinement == 1,
            memory_dim=args.memory_dim,
            expert_rank=args.expert_rank,
            num_experts=args.num_experts,
            refinement_layer_strategy=args.refinement_layer_strategy,
            enable_overlap_head=args.enable_overlap_head == 1,
            overlap_num_classes=args.overlap_num_classes,
            overlap_loss_weight=args.overlap_loss_weight,
            enable_target_consistency=args.enable_target_consistency == 1,
            target_consistency_weight=args.target_consistency_weight,
            target_consistency_mode=args.target_consistency_mode,
            target_consistency_temperature=args.target_consistency_temperature,
            target_consistency_detach_target=args.target_consistency_detach_target == 1,
            enable_router_supervision=args.enable_router_supervision == 1,
            router_loss_weight=args.router_loss_weight,
        ),
    )
    processor = asr_wrapper.processor
    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    if args.freeze_text_decoder != 1:
        main_print("[freeze] --freeze_text_decoder is deprecated and ignored; decoder stays frozen.")
    frozen = freeze_text_decoder(model)
    main_print(f"[freeze] text decoder frozen parameters: {frozen}")
    if effective_gradient_checkpointing:
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if hasattr(model.thinker.config, "use_cache"):
            model.thinker.config.use_cache = False
    elif gradient_checkpointing_note:
        main_print(f"[checkpointing] {gradient_checkpointing_note}")
    log_model_summary(model)

    stage_print("<== Datasets ==>")
    main_print(f"building train dataset from: {args.train_file}")
    train_dataset = build_dataset(args, split="train")
    if args.eval_file:
        main_print(f"building eval dataset from: {args.eval_file}")
    eval_dataset = build_dataset(args, split="eval") if args.eval_file else None
    train_dataset = maybe_subset_dataset(train_dataset, args.max_train_samples)
    eval_dataset = maybe_subset_dataset(eval_dataset, args.max_eval_samples)
    describe_dataset(
        "train dataset",
        train_dataset,
        per_device_batch_size=args.batch_size,
        world_size=distributed_world_size(),
        grad_acc=args.grad_acc,
    )
    describe_dataset(
        "eval dataset",
        eval_dataset,
        per_device_batch_size=args.eval_batch_size,
        world_size=distributed_world_size(),
        grad_acc=1,
    )
    main_print(f"train enrollment pool: {args.train_spk2utt}")
    if args.train_single_wav_scp:
        main_print(f"train single.wav.scp: {args.train_single_wav_scp}")
    if args.train_aux_label_manifest:
        main_print(f"train aux labels: {args.train_aux_label_manifest}")
    if args.eval_file:
        if args.eval_enroll_paths_json:
            main_print(f"eval enroll map json: {args.eval_enroll_paths_json}")
        else:
            main_print(f"eval spk1 enroll map: {args.eval_spk1_enroll}")
            main_print(f"eval spk2 enroll map: {args.eval_spk2_enroll}")
        main_print(f"eval single.wav.scp: {args.eval_spk2utt}")
        if args.eval_aux_label_manifest:
            main_print(f"eval aux labels: {args.eval_aux_label_manifest}")

    stage_print("<== Collator ==>")
    collator = DataCollatorForDynaTaRQwen(
        processor=processor,
        sampling_rate=args.sr,
        silence_seconds=args.silence_seconds,
        default_prompt=args.prompt,
        include_overlap_labels=args.enable_overlap_head == 1 or args.enable_router_supervision == 1,
        include_target_audio=args.enable_target_consistency == 1,
    )
    logging_dir = args.logging_dir.strip() or args.output_dir
    main_print(f"tensorboard dir: {logging_dir}")

    stage_print("<== Training Arguments ==>")
    training_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=args.log_steps,
        logging_first_step=True,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy,
        do_eval=eval_dataset is not None and args.eval_strategy != "no",
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=args.pin_memory == 1,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        data_seed=args.seed,
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to=args.report_to,
        logging_dir=logging_dir,
        gradient_checkpointing=effective_gradient_checkpointing,
        group_by_length=args.group_by_length == 1,
    )
    if args.num_workers > 0:
        training_kwargs["dataloader_prefetch_factor"] = args.prefetch_factor
    training_args = TrainingArguments(**training_kwargs)
    for line in pformat(training_kwargs).split("\n"):
        main_print(line)

    stage_print("<== Trainer ==>")
    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        use_custom_optimizer=args.use_custom_optimizer == 1,
        high_lr_prefixes=parse_prefix_list(args.new_module_prefixes),
        high_lr_multiplier=args.new_module_lr_multiplier,
        callbacks=[MakeEveryCheckpointInferableCallback(base_model_path=args.model_path)],
    )
    main_print("trainer created")
    main_print(f"custom dual-lr optimizer: {bool(args.use_custom_optimizer)}")
    if args.use_custom_optimizer == 1:
        main_print(f"high-lr multiplier: {args.new_module_lr_multiplier}")
        main_print(f"high-lr prefixes: {args.new_module_prefixes}")

    resume_from = (args.resume_from or "").strip()
    if not resume_from and args.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    global_micro_batch_size = args.batch_size * world_size
    effective_batch_size = global_micro_batch_size * args.grad_acc
    write_run_config(
        args.output_dir,
        args,
        {
            "world_size": world_size,
            "rank": distributed_rank(),
            "global_micro_batch_size": global_micro_batch_size,
            "effective_batch_size": effective_batch_size,
            "precision": "bf16" if use_bf16 else "fp16",
            "requested_gradient_checkpointing": requested_gradient_checkpointing,
            "effective_gradient_checkpointing": effective_gradient_checkpointing,
            "gradient_checkpointing_note": gradient_checkpointing_note,
            "resolved_logging_dir": logging_dir,
            "resolved_resume_from": resume_from,
            "train_num_samples": maybe_dataset_len(train_dataset),
            "eval_num_samples": maybe_dataset_len(eval_dataset),
            "do_eval": eval_dataset is not None and args.eval_strategy != "no",
            "training_arguments": training_kwargs,
        },
    )

    stage_print("<========== Training process ==========>")
    if resume_from:
        main_print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        main_print("train from scratch ...")
        trainer.train()


if __name__ == "__main__":
    main()
