from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path("/scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr")
WESEP_ROOT = Path("/scratch/project_465002316/junyi/tse/ts_asr/wesep")
QWEN_ROOT = Path("/scratch/project_465002316/junyi/tse/ts_asr/Qwen3-ASR")
for path in (str(ROOT), str(WESEP_ROOT), str(QWEN_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from qwen_asr import Qwen3ASRModel

from dynatar_qwen.data import DataCollatorForDynaTaRQwen, TSASRShardDataset
from dynatar_qwen.modeling_ts_qwen3_asr import TargetAwareEncoderConfig, upgrade_qwen3_asr_model
from train_ts_qwen3_asr import freeze_text_decoder, patch_outer_forward


DEFAULT_DATA_ROOT = ROOT / "data" / "clean"
DEFAULT_CHECKPOINT = ROOT / "exp" / "full_chain_qwen_tsasr_smoke" / "checkpoint-2" / "model.safetensors"


def parse_args():
    parser = argparse.ArgumentParser("Inspect one train-100 TS-ASR sample in Qwen3-ASR fine-tuning format")
    parser.add_argument("--data_type", choices=["shard"], default="shard")
    parser.add_argument("--train_file", default=str(DEFAULT_DATA_ROOT / "train-100" / "shard.list"))
    parser.add_argument("--train_spk2utt", default=str(DEFAULT_DATA_ROOT / "train-100" / "spk2enroll.json"))
    parser.add_argument("--model_path", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--language", default="English")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--silence_seconds", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    return parser.parse_args()


def build_dataset(args):
    return TSASRShardDataset(
        args.train_file,
        split="train",
        sampling_rate=16000,
        language=args.language,
        prompt=args.prompt,
        train_spk2utt=args.train_spk2utt,
    )


def load_checkpoint_if_needed(model, checkpoint_path: str):
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        print(f"[checkpoint] skip: {checkpoint} not found")
        return
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required to load checkpoint weights") from exc
    state_dict = load_file(str(checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[checkpoint] loaded from {checkpoint}")
    print(f"[checkpoint] missing={len(missing)} unexpected={len(unexpected)}")


def decode_ids(tokenizer, ids: torch.Tensor) -> str:
    if ids.numel() == 0:
        return ""
    return tokenizer.decode(ids.tolist(), skip_special_tokens=False)


def first_supervised_index(labels: torch.Tensor) -> int:
    valid = torch.nonzero(labels != -100, as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
        return labels.numel()
    return int(valid[0].item())


def extract_generated_text(decoded: str) -> str:
    marker = "<asr_text>"
    if marker not in decoded:
        return decoded
    text = decoded.split(marker, 1)[1]
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    return text.strip()


def main():
    args = parse_args()
    dataset = build_dataset(args)
    sample = dataset[args.sample_index]

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    wrapper = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        attn_implementation="eager",
        device_map=None,
    )
    model = upgrade_qwen3_asr_model(
        wrapper.model,
        TargetAwareEncoderConfig(
            num_role_layers=5,
            num_prototypes=8,
            mhfa_heads=4,
            mhfa_compression_dim=128,
            router_hidden_multiplier=2,
        ),
    )
    patch_outer_forward(model)
    freeze_text_decoder(model)
    load_checkpoint_if_needed(model, args.checkpoint)
    model = model.cuda().eval()

    collator = DataCollatorForDynaTaRQwen(
        processor=wrapper.processor,
        sampling_rate=16000,
        silence_seconds=args.silence_seconds,
        default_prompt=args.prompt,
    )
    batch = collator([sample])

    tokenizer = wrapper.processor.tokenizer
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    prefix_len = first_supervised_index(labels)
    prefix_ids = input_ids[:prefix_len]
    supervised_ids = labels[labels != -100]
    placeholder_count = int((input_ids == tokenizer.audio_token_id).sum().item())

    print("=== Sample Meta ===")
    for key in ("key", "mix_id", "target_role", "target_spk", "target_text"):
        if key in sample:
            print(f"{key}: {sample[key]}")
    if "mix_wav" in sample:
        print(f"mix_wav: {sample['mix_wav']}")
    if "enroll_wav" in sample:
        print(f"enroll_wav: {sample['enroll_wav']}")

    print("\n=== Qwen3-ASR Fine-tuning Format ===")
    print("prefix_text = chat_template(system + user(audio)) with generation prompt")
    print("full_text   = prefix_text + target_text + eos")
    print("labels      = input_ids clone, then prefix span and pad positions -> -100")

    print("\n=== Current TS-ASR Mapping ===")
    print("audio input to encoder: [enroll ; sil ; mix]")
    print("text side placeholders: only refined mix token count")
    print("assistant target text :", decode_ids(tokenizer, supervised_ids))

    print("\n=== Batch Tensors ===")
    print(f"input_ids.shape: {tuple(batch['input_ids'].shape)}")
    print(f"input_features.shape: {tuple(batch['input_features'].shape)}")
    print(f"feature_attention_mask.shape: {tuple(batch['feature_attention_mask'].shape)}")
    print(f"segment_feature_lengths: {batch['segment_feature_lengths'][0].tolist()}")
    print(f"segment_token_lengths: {batch['segment_token_lengths'][0].tolist()}")
    print(f"audio_feature_lengths: {batch['audio_feature_lengths'][0].item()}")
    print(f"audio placeholder count in input_ids: {placeholder_count}")
    print(f"prefix token length: {prefix_len}")
    print(f"supervised token count: {int((labels != -100).sum().item())}")

    print("\n=== Prefix Decoded ===")
    print(decode_ids(tokenizer, prefix_ids))

    print("\n=== Supervised Target Decoded ===")
    print(decode_ids(tokenizer, supervised_ids))

    cast_batch = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.cuda()
            if value.is_floating_point():
                value = value.to(dtype=model.dtype)
        cast_batch[key] = value

    with torch.no_grad():
        outputs = model.thinker(**cast_batch)
    print("\n=== Teacher-Forced Loss ===")
    print(float(outputs.loss.detach().cpu()))

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

    print("\n=== Generated Output (raw decoded) ===")
    print(decoded)

    print("\n=== Generated Output (assistant text after <asr_text>) ===")
    print(extract_generated_text(decoded))


if __name__ == "__main__":
    main()
