#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path


def read_2col(path: Path) -> OrderedDict[str, str]:
    data: OrderedDict[str, str] = OrderedDict()
    if not path.is_file():
        return data
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            key, value = line.split(maxsplit=1)
            if key in data and data[key] != value:
                raise ValueError(f"Conflict for key {key!r} in {path}: {data[key]!r} vs {value!r}")
            data[key] = value
    return data


def write_2col(path: Path, data: OrderedDict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for key, value in data.items():
            handle.write(f"{key} {value}\n")


def read_list(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def write_list(path: Path, items: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(f"{item}\n")


def merge_spk2enroll(paths: list[Path]) -> dict[str, list[list[str]]]:
    merged: dict[str, list[list[str]]] = OrderedDict()
    seen: dict[str, set[tuple[str, str]]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for speaker, entries in payload.items():
            if speaker not in merged:
                merged[speaker] = []
                seen[speaker] = set()
            for entry in entries:
                if not isinstance(entry, list) or len(entry) != 2:
                    raise ValueError(f"Unexpected enrollment entry for speaker {speaker!r} in {path}: {entry!r}")
                pair = (entry[0], entry[1])
                if pair in seen[speaker]:
                    continue
                seen[speaker].add(pair)
                merged[speaker].append([entry[0], entry[1]])
    return merged


def main() -> None:
    parser = argparse.ArgumentParser("Merge multiple TS-ASR train splits into one shard-based split")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    source_dirs = [Path(src).resolve() for src in args.sources]
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_shards: list[str] = []
    seen_shards: set[str] = set()
    merged_single_wav = OrderedDict()
    merged_single_utt2spk = OrderedDict()
    merged_wav = OrderedDict()
    merged_utt2spk = OrderedDict()
    spk2enroll_paths: list[Path] = []
    summary = {"sources": [str(src) for src in source_dirs], "splits": OrderedDict()}

    for src in source_dirs:
        shard_list = read_list(src / "shard.list")
        for item in shard_list:
            if item not in seen_shards:
                seen_shards.add(item)
                merged_shards.append(item)

        for key, value in read_2col(src / "single.wav.scp").items():
            if key in merged_single_wav and merged_single_wav[key] != value:
                raise ValueError(f"single.wav.scp conflict for {key!r}: {merged_single_wav[key]!r} vs {value!r}")
            merged_single_wav[key] = value

        for key, value in read_2col(src / "single.utt2spk").items():
            if key in merged_single_utt2spk and merged_single_utt2spk[key] != value:
                raise ValueError(f"single.utt2spk conflict for {key!r}: {merged_single_utt2spk[key]!r} vs {value!r}")
            merged_single_utt2spk[key] = value

        for key, value in read_2col(src / "wav.scp").items():
            if key in merged_wav and merged_wav[key] != value:
                raise ValueError(f"wav.scp conflict for {key!r}: {merged_wav[key]!r} vs {value!r}")
            merged_wav[key] = value

        for key, value in read_2col(src / "utt2spk").items():
            if key in merged_utt2spk and merged_utt2spk[key] != value:
                raise ValueError(f"utt2spk conflict for {key!r}: {merged_utt2spk[key]!r} vs {value!r}")
            merged_utt2spk[key] = value

        spk2enroll_path = src / "spk2enroll.json"
        if not spk2enroll_path.is_file():
            raise FileNotFoundError(f"Missing spk2enroll.json in {src}")
        spk2enroll_paths.append(spk2enroll_path)

        summary["splits"][src.name] = {
            "num_shards": len(shard_list),
            "num_single_wav": len(read_2col(src / "single.wav.scp")),
            "num_wav": len(read_2col(src / "wav.scp")),
        }

    merged_spk2enroll = merge_spk2enroll(spk2enroll_paths)

    write_list(output_dir / "shard.list", merged_shards)
    write_2col(output_dir / "single.wav.scp", merged_single_wav)
    write_2col(output_dir / "single.utt2spk", merged_single_utt2spk)
    write_2col(output_dir / "wav.scp", merged_wav)
    write_2col(output_dir / "utt2spk", merged_utt2spk)
    with (output_dir / "spk2enroll.json").open("w", encoding="utf-8") as handle:
        json.dump(merged_spk2enroll, handle, indent=2, ensure_ascii=False)

    summary["merged"] = {
        "num_shards": len(merged_shards),
        "num_single_wav": len(merged_single_wav),
        "num_wav": len(merged_wav),
        "num_speakers_in_pool": len(merged_spk2enroll),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"[done] merged split written to {output_dir}")
    print(f"[done] shard.list entries: {len(merged_shards)}")
    print(f"[done] speakers in pool: {len(merged_spk2enroll)}")
    print(f"[done] single.wav.scp entries: {len(merged_single_wav)}")
    print(f"[done] wav.scp entries: {len(merged_wav)}")


if __name__ == "__main__":
    main()
