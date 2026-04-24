from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path("/scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr")
WESEP_ROOT = Path("/scratch/project_465002316/junyi/tse/ts_asr/wesep")
QWEN_ROOT = Path("/scratch/project_465002316/junyi/tse/ts_asr/Qwen3-ASR")
for path in (str(ROOT), str(WESEP_ROOT), str(QWEN_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from qwen_asr import Qwen3ASRModel

from dynatar_qwen.data import DataCollatorForDynaTaRQwen, TSASRShardDataset
from dynatar_qwen.modeling_ts_qwen3_asr import TargetAwareEncoderConfig, upgrade_qwen3_asr_model
from dynatar_qwen.utils import dump_jsonl
from train_ts_qwen3_asr import freeze_text_decoder, patch_outer_forward


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def parse_args():
    parser = argparse.ArgumentParser("Infer DynaTaR-Qwen TS-ASR hypotheses")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--data_type", type=str, default="shard", choices=["shard"])
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--eval_spk1_enroll", type=str, default="")
    parser.add_argument("--eval_spk2_enroll", type=str, default="")
    parser.add_argument("--eval_enroll_paths_json", type=str, default="")
    parser.add_argument("--eval_spk2utt", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
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
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args()


def build_dataset(args):
    if not args.eval_enroll_paths_json and not (args.eval_spk1_enroll and args.eval_spk2_enroll and args.eval_spk2utt):
        raise ValueError(
            "--eval_enroll_paths_json or (--eval_spk1_enroll, --eval_spk2_enroll, --eval_spk2utt) "
            "is required for shard inference"
        )
    return TSASRShardDataset(
        args.eval_file,
        split="eval",
        sampling_rate=args.sr,
        language=args.language,
        prompt=args.prompt,
        enroll_crop_seconds=args.enroll_crop_seconds,
        eval_spk1_enroll=args.eval_spk1_enroll,
        eval_spk2_enroll=args.eval_spk2_enroll,
        eval_enroll_paths_json=args.eval_enroll_paths_json,
        eval_spk2utt=args.eval_spk2utt,
    )


def maybe_subset_dataset(dataset, max_samples: int):
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    return [dataset[index] for index in range(max_samples)]


def collate_with_meta(collator, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = collator(samples)
    batch["samples"] = samples
    return batch


def first_supervised_index(labels: torch.Tensor) -> int:
    valid = torch.nonzero(labels != -100, as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
        return labels.numel()
    return int(valid[0].item())


def extract_generated_text(decoded: str) -> str:
    marker = "<asr_text>"
    text = decoded.split(marker, 1)[1] if marker in decoded else decoded
    for stop_marker in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        if stop_marker in text:
            text = text.split(stop_marker, 1)[0]
    return text.strip()


def resolve_checkpoint_path(path: str) -> Path | None:
    if not path:
        return None
    checkpoint_path = Path(path)
    if checkpoint_path.is_file():
        return checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    if (checkpoint_path / "model.safetensors").is_file():
        return checkpoint_path / "model.safetensors"
    if (checkpoint_path / "pytorch_model.bin").is_file():
        return checkpoint_path / "pytorch_model.bin"

    best_step = None
    best_checkpoint = None
    for child in checkpoint_path.iterdir():
        match = _CKPT_RE.match(child.name)
        if child.is_dir() and match:
            step = int(match.group(1))
            if best_step is None or step > best_step:
                best_step = step
                best_checkpoint = child
    if best_checkpoint is None:
        raise FileNotFoundError(f"No checkpoint-* directory found under {checkpoint_path}")
    if (best_checkpoint / "model.safetensors").is_file():
        return best_checkpoint / "model.safetensors"
    if (best_checkpoint / "pytorch_model.bin").is_file():
        return best_checkpoint / "pytorch_model.bin"
    raise FileNotFoundError(f"No model weights found in {best_checkpoint}")


def build_ts_config_from_args(args) -> TargetAwareEncoderConfig:
    return TargetAwareEncoderConfig(
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
    )


def resolve_ts_config(args, checkpoint_path: Path | None) -> tuple[TargetAwareEncoderConfig, bool, str]:
    if checkpoint_path is not None:
        checkpoint_dir = checkpoint_path.parent
        config_path = checkpoint_dir / "config.json"
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8") as handle:
                config_payload = json.load(handle)
            ts_payload = config_payload.get("tsasr")
            if isinstance(ts_payload, dict):
                return TargetAwareEncoderConfig.from_any(ts_payload), True, str(config_path)

        run_config_path = checkpoint_dir.parent / "run_config.yaml"
        if run_config_path.is_file():
            with run_config_path.open("r", encoding="utf-8") as handle:
                run_config = yaml.safe_load(handle) or {}
            ts_keys = set(TargetAwareEncoderConfig().__dict__.keys())
            ts_payload = {key: run_config[key] for key in ts_keys if key in run_config}
            if ts_payload:
                return TargetAwareEncoderConfig.from_any(ts_payload), False, str(run_config_path)

    return build_ts_config_from_args(args), False, "cli"


def load_checkpoint(model, checkpoint_path: Path | None, strict: bool):
    if checkpoint_path is None:
        print("[checkpoint] skip: using base model weights only")
        return
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint_path))
    else:
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    print(f"[checkpoint] loaded from {checkpoint_path}")
    print(f"[checkpoint] strict={strict} missing={len(missing)} unexpected={len(unexpected)}")
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Strict checkpoint loading failed for {checkpoint_path}: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )


def cast_single_example(batch: Dict[str, Any], index: int, model) -> Dict[str, torch.Tensor]:
    cast_batch = {}
    for key, value in batch.items():
        if key == "samples":
            continue
        if torch.is_tensor(value):
            value = value[index : index + 1].to(model.device)
            if value.is_floating_point():
                value = value.to(dtype=model.dtype)
        cast_batch[key] = value
    return cast_batch


def save_text_file(path: Path, entries: List[tuple[str, str]]):
    with path.open("w", encoding="utf-8") as handle:
        for key, text in entries:
            handle.write(f"{key} {text}\n")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)
    dataset = maybe_subset_dataset(dataset, args.max_samples)

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    ts_config, strict_checkpoint_load, ts_config_source = resolve_ts_config(args, checkpoint_path)
    print(f"[config] ts_config_source={ts_config_source}")

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    wrapper = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        attn_implementation="eager",
        device_map=None,
    )
    model = upgrade_qwen3_asr_model(
        wrapper.model,
        ts_config,
    )
    patch_outer_forward(model)
    freeze_text_decoder(model)
    model = model.cuda().eval()

    load_checkpoint(model, checkpoint_path, strict=strict_checkpoint_load)

    collator = DataCollatorForDynaTaRQwen(
        processor=wrapper.processor,
        sampling_rate=args.sr,
        silence_seconds=args.silence_seconds,
        default_prompt=args.prompt,
        include_overlap_labels=False,
        include_target_audio=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda samples: collate_with_meta(collator, samples),
    )

    tokenizer = wrapper.processor.tokenizer
    results = []
    ref_entries = []
    hyp_entries = []

    for batch_idx, batch in enumerate(loader, start=1):
        samples = batch["samples"]
        labels = batch["labels"]
        for sample_idx, sample in enumerate(samples):
            cast_batch = cast_single_example(batch, sample_idx, model)
            prefix_len = first_supervised_index(labels[sample_idx])
            generation_inputs = {
                "input_ids": cast_batch["input_ids"][:, :prefix_len],
                "attention_mask": cast_batch["attention_mask"][:, :prefix_len],
                "input_features": cast_batch["input_features"],
                "feature_attention_mask": cast_batch["feature_attention_mask"],
                "segment_feature_lengths": cast_batch["segment_feature_lengths"],
                "segment_token_lengths": cast_batch["segment_token_lengths"],
                "audio_feature_lengths": cast_batch["audio_feature_lengths"],
                "max_new_tokens": args.max_new_tokens,
            }

            with torch.no_grad():
                generated = model.thinker.generate(**generation_inputs)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)[0]
            hypothesis = extract_generated_text(decoded)

            result = {
                "key": sample["key"],
                "mix_id": sample.get("mix_id", ""),
                "target_role": sample.get("target_role", ""),
                "target_spk": sample.get("target_spk", ""),
                "reference": sample["target_text"],
                "hypothesis": hypothesis,
                "decoded_raw": decoded,
                "checkpoint": str(checkpoint_path) if checkpoint_path is not None else "",
            }
            results.append(result)
            ref_entries.append((sample["key"], sample["target_text"]))
            hyp_entries.append((sample["key"], hypothesis))

        print(f"[decode] processed {len(results)} samples")

    results_jsonl = output_dir / "results.jsonl"
    ref_path = output_dir / "ref.text"
    hyp_path = output_dir / "hyp.text"
    summary_path = output_dir / "decode_summary.json"

    dump_jsonl(results_jsonl, results)
    save_text_file(ref_path, ref_entries)
    save_text_file(hyp_path, hyp_entries)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_samples": len(results),
                "checkpoint": str(checkpoint_path) if str(checkpoint_path) else "",
                "model_path": args.model_path,
                "data_type": args.data_type,
                "eval_file": args.eval_file,
                "output_dir": str(output_dir),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[decode] results_jsonl={results_jsonl}")
    print(f"[decode] ref_path={ref_path}")
    print(f"[decode] hyp_path={hyp_path}")
    print(f"[decode] summary_path={summary_path}")


if __name__ == "__main__":
    main()
