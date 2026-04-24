from __future__ import annotations

import argparse
import gzip
import json
import urllib.request
from collections import defaultdict
from pathlib import Path


DEFAULT_DOWNLOAD_URL = (
    "https://nextcloud.fit.vutbr.cz/public.php/dav/files/ddRxWNZJ4Aw6gwy/?accept=zip"
)


def parse_args():
    parser = argparse.ArgumentParser("Prepare train-3mix-custom metadata in WeSep/TS-ASR style")
    parser.add_argument("--mix-wav-dir", required=True, help="Directory containing generated mixed wav files")
    parser.add_argument("--output-dir", required=True, help="Output metadata directory")
    parser.add_argument(
        "--manifest",
        default="",
        help="Path to custom train-3mix cutset (.jsonl or .jsonl.gz). Download if missing.",
    )
    parser.add_argument(
        "--download-url",
        default=DEFAULT_DOWNLOAD_URL,
        help="Download URL for the official custom train-3mix manifest",
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="Actual LibriSpeech root, e.g. /tmp/librispeech/LibriSpeech",
    )
    return parser.parse_args()


def ensure_manifest(path: Path, download_url: str):
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading manifest to {path}", flush=True)
    urllib.request.urlretrieve(download_url, path)
    return path


def open_manifest(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def normalize_source(source: str, source_root: str) -> str:
    source_root = source_root.rstrip("/")
    prefixes = [
        "PREFIX/librispeech/LibriSpeech/",
        "/tmp/librispeechmix/librispeech/LibriSpeech/",
        "/tmp/librispeech/LibriSpeech/",
    ]
    for prefix in prefixes:
        if source.startswith(prefix):
            return source_root + "/" + source[len(prefix) :]
    marker = "/librispeech/LibriSpeech/"
    if marker in source:
        return source_root + "/" + source.split(marker, 1)[1]
    marker = "/LibriSpeech/"
    if marker in source:
        return source_root + "/" + source.split(marker, 1)[1]
    return source


def main():
    args = parse_args()
    mix_wav_dir = Path(args.mix_wav_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest) if args.manifest else output_dir / "librispeechmix_custom_cutset_train-3mix.jsonl.gz"
    manifest_path = ensure_manifest(manifest_path, args.download_url)

    wav_scp_lines = []
    utt2spk_lines = []
    single_wav_lines = []
    single_utt2spk_lines = []
    spk_texts = defaultdict(list)
    spk2enroll = defaultdict(list)
    spk2enroll_seen = set()

    mixed_count = 0
    missing_mix = 0
    mono_count = 0

    with open_manifest(manifest_path) as handle:
        for line in handle:
            obj = json.loads(line)
            if obj.get("type") != "MixedCut":
                mono_count += 1
                continue

            mix_id = obj["id"]
            mix_wav = mix_wav_dir / f"{mix_id}.wav"
            if not mix_wav.is_file():
                missing_mix += 1
                continue

            src_paths = []
            speakers = []
            for idx, track in enumerate(obj["tracks"], start=1):
                cut = track["cut"]
                supervision = cut["supervisions"][0]
                source = cut["recording"]["sources"][0]["source"]
                source = normalize_source(source, args.source_root)

                utt_id = supervision["id"]
                spk = supervision["speaker"]
                text = " ".join(supervision["text"].strip().split())

                src_paths.append(source)
                speakers.append(spk)

                role_key = f"s{idx}/{mix_id}.wav"
                single_wav_lines.append(f"{role_key} {source}")
                single_utt2spk_lines.append(f"{role_key} {spk}")
                spk_texts[idx].append(f"{mix_id} {text}")

                enroll_key = (spk, utt_id, source)
                if enroll_key not in spk2enroll_seen:
                    spk2enroll_seen.add(enroll_key)
                    spk2enroll[spk].append([utt_id, source])

            wav_scp_lines.append(" ".join([mix_id, str(mix_wav), *src_paths]))
            utt2spk_lines.append(" ".join([mix_id, *speakers]))
            mixed_count += 1

    (output_dir / "wav.scp").write_text("\n".join(wav_scp_lines) + "\n", encoding="utf-8")
    (output_dir / "utt2spk").write_text("\n".join(utt2spk_lines) + "\n", encoding="utf-8")
    (output_dir / "single.wav.scp").write_text("\n".join(single_wav_lines) + "\n", encoding="utf-8")
    (output_dir / "single.utt2spk").write_text("\n".join(single_utt2spk_lines) + "\n", encoding="utf-8")
    with (output_dir / "spk2enroll.json").open("w", encoding="utf-8") as f:
        json.dump(spk2enroll, f, indent=2, ensure_ascii=False)

    for idx, lines in sorted(spk_texts.items()):
        (output_dir / f"spk{idx}_text").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "mixed_cut_count": mixed_count,
        "missing_mix_wavs": missing_mix,
        "non_mixed_entries_skipped": mono_count,
        "num_speakers_in_pool": len(spk2enroll),
        "manifest": str(manifest_path),
        "mix_wav_dir": str(mix_wav_dir),
        "source_root": args.source_root,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
