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


def parse_args():
    data_root = ROOT / "data" / "clean" / "train-3mix-custom"
    parser = argparse.ArgumentParser("Check train-3mix-custom training contract")
    parser.add_argument("--data_root", default=str(data_root))
    parser.add_argument("--shard_list", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    shard_list = args.shard_list or str(data_root / "shard.list")
    dataset = TSASRShardDataset(
        shard_list,
        split="train",
        sampling_rate=16000,
        language="English",
        prompt="",
        enroll_crop_seconds=4.0,
        train_spk2utt=str(data_root / "spk2enroll.json"),
        train_single_wav_scp=str(data_root / "single.wav.scp"),
    )

    print(f"dataset_len={len(dataset)}")

    candidate_probe_indices = [0, 1, 2, 100, 1000, 10000, 50000, len(dataset) - 1]
    probe_indices = []
    for idx in candidate_probe_indices:
        if 0 <= idx < len(dataset) and idx not in probe_indices:
            probe_indices.append(idx)
    found_roles = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        role = sample["target_role"]
        if role in {"spk4", "spk5", "spk6"} and role not in found_roles:
            found_roles[role] = (idx, sample)
        if len(found_roles) == 3:
            break

    for idx in probe_indices:
        sample = dataset[idx]
        print(f"idx={idx}")
        print(f"key={sample['key']}")
        print(f"role={sample['target_role']} spk={sample['target_spk']}")
        print(f"enroll={sample['enroll_wav']}")
        print(
            "shapes:",
            f"enroll={tuple(sample['enroll_audio'].shape)}",
            f"mix={tuple(sample['mix_audio'].shape)}",
            f"target={tuple(sample['target_audio'].shape)}",
        )

    for role in sorted(found_roles):
        idx, sample = found_roles[role]
        print(f"found_{role}: idx={idx} key={sample['key']} spk={sample['target_spk']}")
        print(f"found_{role}_enroll={sample['enroll_wav']}")
        print(f"found_{role}_shape={tuple(sample['enroll_audio'].shape)}")

    target_sample = found_roles.get("spk6", next(iter(found_roles.values())))[1] if found_roles else dataset[0]

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    model_path = str(ROOT / "qwen_models" / "Qwen3-ASR-0.6B")
    wrapper = Qwen3ASRModel.from_pretrained(
        model_path,
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
    model = model.cuda().eval()

    collator = DataCollatorForDynaTaRQwen(
        processor=wrapper.processor,
        sampling_rate=16000,
        silence_seconds=1.0,
        default_prompt="",
    )
    batch = collator([target_sample])
    cast_batch = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.cuda()
            if value.is_floating_point():
                value = value.to(dtype=model.dtype)
        cast_batch[key] = value
    with torch.no_grad():
        outputs = model(**cast_batch)
    print(f"forward_loss={float(outputs.loss):.6f}")
    print(f"logits_shape={tuple(outputs.logits.shape)}")
    print(f"segment_token_lengths={batch['segment_token_lengths'][0].tolist()}")


if __name__ == "__main__":
    main()
