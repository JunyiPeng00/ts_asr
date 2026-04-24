from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr")
WESEP_ROOT = Path("/scratch/project_465002316/junyi/tse/ts_asr/wesep")
QWEN_ROOT = Path("/scratch/project_465002316/junyi/tse/ts_asr/Qwen3-ASR")
for path in (str(ROOT), str(WESEP_ROOT), str(QWEN_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dynatar_qwen.data import TSASRShardDataset


def parse_args():
    parser = argparse.ArgumentParser("Check one train-360 shard sample")
    data_root = ROOT / "data" / "clean" / "train-360"
    parser.add_argument("--shard_list", default=str(data_root / "shard.list"))
    parser.add_argument("--spk2utt", default=str(data_root / "spk2enroll.json"))
    parser.add_argument("--single_wav_scp", default=str(data_root / "single.wav.scp"))
    return parser.parse_args()


def main():
    args = parse_args()
    train_ds = TSASRShardDataset(
        args.shard_list,
        split="train",
        sampling_rate=16000,
        language="English",
        prompt="",
        enroll_crop_seconds=4.0,
        train_spk2utt=args.spk2utt,
        train_single_wav_scp=args.single_wav_scp,
    )
    sample = train_ds[0]
    print(f"key={sample['key']}")
    print(f"role={sample['target_role']} spk={sample['target_spk']}")
    print(f"enroll_shape={tuple(sample['enroll_audio'].shape)}")
    print(f"mix_shape={tuple(sample['mix_audio'].shape)}")
    print(f"target_shape={tuple(sample['target_audio'].shape)}")


if __name__ == "__main__":
    main()
