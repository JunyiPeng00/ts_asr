from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynatar_qwen.utils import dump_jsonl, read_2column_text


def read_mix_scp(path: str) -> dict[str, str]:
    mapping = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid wav.scp line: {line}")
            mapping[parts[0]] = parts[1]
    return mapping


def resolve_audio_path(mix_wav: str, maybe_relative_path: str) -> str:
    candidate = Path(maybe_relative_path)
    if candidate.is_absolute():
        return str(candidate)
    mix_root = Path(mix_wav).resolve().parent.parent
    return str((mix_root / candidate).resolve())


def read_mix_speakers(path: str) -> dict[str, tuple[str, str]]:
    mapping = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Invalid utt2spk line: {line}")
            mapping[parts[0]] = (parts[1], parts[2])
    return mapping


def load_spk2enroll(path: str) -> dict[str, list[list[str]]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_enrollment_path(
    speaker_id: str,
    target_utt: str,
    speaker_pool: dict[str, list[list[str]]],
) -> str:
    candidates = speaker_pool.get(speaker_id)
    if not candidates:
        raise KeyError(f"No enrollment candidates found for speaker {speaker_id}")
    for candidate_utt, candidate_path in candidates:
        if candidate_utt != target_utt:
            return candidate_path
    return candidates[0][1]


def parse_args():
    parser = argparse.ArgumentParser("Prepare LibriMix TS-ASR manifest")
    parser.add_argument("--mix-scp", required=True, help="wav.scp for mixture audio")
    parser.add_argument("--spk1-enroll", default="", help="spk1.enroll from WeSep prep")
    parser.add_argument("--spk2-enroll", default="", help="spk2.enroll from WeSep prep")
    parser.add_argument(
        "--single-wav-scp",
        default="",
        help="single.wav.scp for resolving fixed enrollment keys to audio paths",
    )
    parser.add_argument("--utt2spk", default="", help="3-column mixture speaker mapping for train splits")
    parser.add_argument("--spk2enroll-json", default="", help="speaker->enrollment pool JSON for train splits")
    parser.add_argument("--spk1-text", required=True, help="Two-column target transcript file for speaker 1")
    parser.add_argument("--spk2-text", required=True, help="Two-column target transcript file for speaker 2")
    parser.add_argument("--output", required=True, help="Output JSONL manifest path")
    parser.add_argument("--language", default="English")
    parser.add_argument("--prompt", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    mix_map = read_mix_scp(args.mix_scp)
    spk1_text = read_2column_text(args.spk1_text)
    spk2_text = read_2column_text(args.spk2_text)

    use_fixed_enroll = bool(args.spk1_enroll and args.spk2_enroll)
    use_pool_enroll = bool(args.utt2spk and args.spk2enroll_json)
    if use_fixed_enroll == use_pool_enroll:
        raise ValueError(
            "Specify either (--spk1-enroll and --spk2-enroll) or (--utt2spk and --spk2enroll-json)"
        )

    spk1_enroll = read_2column_text(args.spk1_enroll) if use_fixed_enroll else {}
    spk2_enroll = read_2column_text(args.spk2_enroll) if use_fixed_enroll else {}
    single_wav_map = read_2column_text(args.single_wav_scp) if use_fixed_enroll and args.single_wav_scp else {}
    mix_speakers = read_mix_speakers(args.utt2spk) if use_pool_enroll else {}
    speaker_pool = load_spk2enroll(args.spk2enroll_json) if use_pool_enroll else {}

    records = []
    for mix_id, mix_wav in mix_map.items():
        required = [("spk1_text", mix_id, spk1_text), ("spk2_text", mix_id, spk2_text)]
        missing = [name for name, key, mapping in required if key not in mapping]
        if missing:
            raise KeyError(f"Missing {missing} for mixture id {mix_id}")

        if use_fixed_enroll:
            fixed_required = [("spk1_enroll", mix_id, spk1_enroll), ("spk2_enroll", mix_id, spk2_enroll)]
            missing_fixed = [name for name, key, mapping in fixed_required if key not in mapping]
            if missing_fixed:
                raise KeyError(f"Missing {missing_fixed} for mixture id {mix_id}")
            spk1_entry = spk1_enroll[mix_id]
            spk2_entry = spk2_enroll[mix_id]
            if single_wav_map:
                if spk1_entry not in single_wav_map or spk2_entry not in single_wav_map:
                    raise KeyError(f"Missing enrollment key in single.wav.scp for mixture id {mix_id}")
                spk1_enroll_path = single_wav_map[spk1_entry]
                spk2_enroll_path = single_wav_map[spk2_entry]
            else:
                spk1_enroll_path = resolve_audio_path(mix_wav, spk1_entry)
                spk2_enroll_path = resolve_audio_path(mix_wav, spk2_entry)
        else:
            if mix_id not in mix_speakers:
                raise KeyError(f"Missing speaker mapping for mixture id {mix_id}")
            spk1_id, spk2_id = mix_speakers[mix_id]
            spk1_utt, spk2_utt = mix_id.split("_", 1)
            spk1_enroll_path = select_enrollment_path(spk1_id, spk1_utt, speaker_pool)
            spk2_enroll_path = select_enrollment_path(spk2_id, spk2_utt, speaker_pool)

        records.append(
            {
                "key": f"{mix_id}/spk1",
                "mix_id": mix_id,
                "target_role": "spk1",
                "mix_wav": mix_wav,
                "enroll_wav": spk1_enroll_path,
                "target_text": spk1_text[mix_id],
                "language": args.language,
                "prompt": args.prompt,
            }
        )
        records.append(
            {
                "key": f"{mix_id}/spk2",
                "mix_id": mix_id,
                "target_role": "spk2",
                "mix_wav": mix_wav,
                "enroll_wav": spk2_enroll_path,
                "target_text": spk2_text[mix_id],
                "language": args.language,
                "prompt": args.prompt,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(output_path, records)
    print(f"Wrote {len(records)} TS-ASR samples to {output_path}")


if __name__ == "__main__":
    main()
