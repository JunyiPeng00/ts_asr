from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_SPLIT_SEARCH_ORDER = [
    "train-clean-360",
    "train-clean-100",
    "train-other-500",
    "dev-clean",
    "test-clean",
]


def parse_args():
    parser = argparse.ArgumentParser("Prepare Libri3Mix train-360 metadata in TS-ASR style")
    parser.add_argument(
        "--mix-root",
        required=True,
        help="Root directory like .../Libri3Mix/wav16k/max/train-360",
    )
    parser.add_argument(
        "--librispeech-root",
        required=True,
        help="LibriSpeech root, e.g. /scratch/.../librispeech/LibriSpeech",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mix-subdir", default="mix_clean")
    parser.add_argument(
        "--split-search-order",
        default=",".join(DEFAULT_SPLIT_SEARCH_ORDER),
        help="Comma-separated LibriSpeech splits to search when resolving utt ids",
    )
    parser.add_argument(
        "--enroll-split",
        default="train-clean-360",
        help="LibriSpeech split used to build spk2enroll.json",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def load_chapter_transcripts(chapter_dir: Path, cache: dict[Path, dict[str, str]]) -> dict[str, str]:
    if chapter_dir in cache:
        return cache[chapter_dir]
    trans_path = chapter_dir / f"{chapter_dir.parent.name}-{chapter_dir.name}.trans.txt"
    if not trans_path.is_file():
        raise FileNotFoundError(f"Missing transcript file: {trans_path}")
    mapping: dict[str, str] = {}
    with trans_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            utt_id, text = line.split(maxsplit=1)
            mapping[utt_id] = normalize_text(text)
    cache[chapter_dir] = mapping
    return mapping


def resolve_utt(
    utt_id: str,
    librispeech_root: Path,
    split_order: list[str],
    transcript_cache: dict[Path, dict[str, str]],
) -> tuple[str, str]:
    spk, chapter, _ = utt_id.split("-", 2)
    for split in split_order:
        chapter_dir = librispeech_root / split / spk / chapter
        wav_path = chapter_dir / f"{utt_id}.flac"
        if not wav_path.is_file():
            continue
        text_map = load_chapter_transcripts(chapter_dir, transcript_cache)
        if utt_id not in text_map:
            raise KeyError(f"Transcript missing for {utt_id} in {chapter_dir}")
        return str(wav_path), text_map[utt_id]
    raise FileNotFoundError(f"Could not resolve utterance {utt_id} under {librispeech_root}")


def build_spk2enroll(librispeech_root: Path, enroll_split: str) -> dict[str, list[list[str]]]:
    enroll_root = librispeech_root / enroll_split
    if not enroll_root.is_dir():
        raise FileNotFoundError(f"Missing enroll split directory: {enroll_root}")
    spk2enroll: dict[str, list[list[str]]] = defaultdict(list)
    for audio in sorted(enroll_root.rglob("*.flac")):
        spk = audio.parent.parent.name
        utt_id = audio.stem
        spk2enroll[spk].append([utt_id, str(audio)])
    return dict(spk2enroll)


def main():
    args = parse_args()
    mix_root = Path(args.mix_root)
    librispeech_root = Path(args.librispeech_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mix_dir = mix_root / args.mix_subdir
    if not mix_dir.is_dir():
        raise FileNotFoundError(f"Missing mix directory: {mix_dir}")

    split_order = [item.strip() for item in args.split_search_order.split(",") if item.strip()]
    transcript_cache: dict[Path, dict[str, str]] = {}

    wav_scp_lines = []
    utt2spk_lines = []
    single_wav_lines = []
    single_utt2spk_lines = []
    spk_texts: dict[int, list[str]] = defaultdict(list)

    mix_count = 0
    missing_mix = 0
    num_speakers_set = set()

    for mix_wav in sorted(mix_dir.glob("*.wav")):
        mix_id = mix_wav.stem
        utt_ids = mix_id.split("_")
        if len(utt_ids) != 3:
            continue

        source_wavs = []
        speakers = []
        for idx, utt_id in enumerate(utt_ids, start=1):
            spk = utt_id.split("-", 1)[0]
            source_mix_wav = mix_root / f"s{idx}" / f"{mix_id}.wav"
            if not source_mix_wav.is_file():
                raise FileNotFoundError(f"Missing source wav for {mix_id}: {source_mix_wav}")
            enroll_wav, text = resolve_utt(utt_id, librispeech_root, split_order, transcript_cache)
            source_wavs.append(str(source_mix_wav))
            speakers.append(spk)
            single_key = f"s{idx}/{mix_id}.wav"
            single_wav_lines.append(f"{single_key} {enroll_wav}")
            single_utt2spk_lines.append(f"{single_key} {spk}")
            spk_texts[idx].append(f"{mix_id} {text}")

        wav_scp_lines.append(" ".join([mix_id, str(mix_wav), *source_wavs]))
        utt2spk_lines.append(" ".join([mix_id, *speakers]))
        mix_count += 1
        num_speakers_set.update(speakers)

    if mix_count == 0:
        missing_mix += 1

    (output_dir / "wav.scp").write_text("\n".join(wav_scp_lines) + "\n", encoding="utf-8")
    (output_dir / "utt2spk").write_text("\n".join(utt2spk_lines) + "\n", encoding="utf-8")
    (output_dir / "single.wav.scp").write_text("\n".join(single_wav_lines) + "\n", encoding="utf-8")
    (output_dir / "single.utt2spk").write_text("\n".join(single_utt2spk_lines) + "\n", encoding="utf-8")

    for idx, lines in sorted(spk_texts.items()):
        (output_dir / f"spk{idx}_text").write_text("\n".join(lines) + "\n", encoding="utf-8")

    spk2enroll = build_spk2enroll(librispeech_root, args.enroll_split)
    with (output_dir / "spk2enroll.json").open("w", encoding="utf-8") as handle:
        json.dump(spk2enroll, handle, indent=2, ensure_ascii=False)

    summary = {
        "mix_count": mix_count,
        "missing_mix_wavs": missing_mix,
        "num_target_roles": len(spk_texts),
        "num_speakers_in_mixes": len(num_speakers_set),
        "num_speakers_in_pool": len(spk2enroll),
        "mix_root": str(mix_root),
        "librispeech_root": str(librispeech_root),
        "mix_subdir": args.mix_subdir,
        "enroll_split": args.enroll_split,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
