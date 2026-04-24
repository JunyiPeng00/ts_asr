#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


KNOWN_SPLITS = (
    "train-clean-100",
    "train-clean-360",
    "dev-clean",
    "test-clean",
    "train-other-500",
    "dev-other",
    "test-other",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite LibriMix-style single.wav.scp entries to original LibriSpeech audio paths.",
    )
    parser.add_argument("--input_scp", required=True, help="Existing single.wav.scp file.")
    parser.add_argument("--output_scp", required=True, help="Output single.wav.scp file.")
    parser.add_argument(
        "--librispeech_root",
        required=True,
        help="Root directory of original LibriSpeech, e.g. /path/to/LibriSpeech",
    )
    parser.add_argument(
        "--prefer_split",
        action="append",
        default=[],
        help="Preferred LibriSpeech split(s) to search first. Can be repeated.",
    )
    return parser.parse_args()


def key_to_utt_id(key: str) -> str:
    role, filename = key.split("/", 1)
    stem = Path(filename).stem
    utt_ids = stem.split("_")
    if not role.startswith("s") or not role[1:].isdigit():
        raise ValueError(f"Unsupported enrollment key format: {key}")
    index = int(role[1:]) - 1
    if index < 0 or index >= len(utt_ids):
        raise ValueError(f"Role index out of range for key: {key}")
    return utt_ids[index]


def resolve_librispeech_path(root: Path, utt_id: str, prefer_splits: list[str]) -> Path:
    speaker_id, chapter_id, _ = utt_id.split("-", 2)
    searched_splits: list[str] = []
    for split in [*prefer_splits, *KNOWN_SPLITS]:
        if split in searched_splits:
            continue
        searched_splits.append(split)
        candidate = root / split / speaker_id / chapter_id / f"{utt_id}.flac"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Failed to locate original LibriSpeech audio for {utt_id} under {root}",
    )


def main() -> None:
    args = parse_args()
    input_scp = Path(args.input_scp)
    output_scp = Path(args.output_scp)
    librispeech_root = Path(args.librispeech_root)
    prefer_splits = list(args.prefer_split)

    lines_out: list[str] = []
    with input_scp.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            key, _ = line.split(maxsplit=1)
            utt_id = key_to_utt_id(key)
            wav_path = resolve_librispeech_path(librispeech_root, utt_id, prefer_splits)
            lines_out.append(f"{key} {wav_path}")

    output_scp.parent.mkdir(parents=True, exist_ok=True)
    output_scp.write_text("\n".join(lines_out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
