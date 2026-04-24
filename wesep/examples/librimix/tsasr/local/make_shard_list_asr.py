import argparse
import io
import logging
import multiprocessing
import os
import random
import sys
import tarfile
import time
from pathlib import Path

from wesep.dataset.target_speaker.io import read_2column_text
from wesep.modules.target_speaker.role_utils import role_to_index


AUDIO_FORMAT_SETS = {"flac", "mp3", "m4a", "ogg", "opus", "wav", "wma"}


def read_mix_scp(path):
    mapping = {}
    with Path(path).open("r", encoding="utf8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid wav.scp line in {path}: {line}")
            mapping[parts[0]] = {
                "mix_wav": parts[1],
                "source_wavs": parts[2:],
            }
    return mapping


def read_mix_speakers(path):
    mapping = {}
    with Path(path).open("r", encoding="utf8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Invalid utt2spk line in {path}: {line}")
            mapping[parts[0]] = parts[1:]
    return mapping


def role_index(role: str) -> int:
    return role_to_index(role)


def write_text_file(tar, name, value):
    payload = value.encode("utf8")
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def write_audio_file(tar, member_name, wav_path):
    suffix = wav_path.split(".")[-1]
    assert suffix in AUDIO_FORMAT_SETS
    with open(wav_path, "rb") as handle:
        data = handle.read()
    info = tarfile.TarInfo(member_name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def write_tar_file(data_list, tar_file, index=0, total=1):
    logging.info("Processing %s %s/%s", tar_file, index, total)
    read_time = 0.0
    write_time = 0.0
    with tarfile.open(tar_file, "w") as tar:
        for item in data_list:
            record_key = item["record_key"]
            ts = time.time()
            write_text_file(tar, f"{record_key}.spk", item["speaker"])
            write_text_file(tar, f"{record_key}.txt", item["target_text"])
            write_text_file(tar, f"{record_key}.role", item["target_role"])
            write_text_file(tar, f"{record_key}.mixid", item["mix_id"])
            write_time += time.time() - ts

            ts = time.time()
            try:
                write_audio_file(tar, f"{record_key}.wav", item["mix_wav"])
                write_audio_file(tar, f"{record_key}.targetwav", item["target_wav"])
            except FileNotFoundError as ex:
                print(ex)
                sys.exit(1)
            read_time += time.time() - ts
    logging.info("read %s write %s", read_time, write_time)


def get_args():
    parser = argparse.ArgumentParser(description="Create ASR-aware TS-ASR shards")
    parser.add_argument("--num_utts_per_shard", type=int, default=1000)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--prefix", default="shards")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--spk1_text", default="")
    parser.add_argument("--spk2_text", default="")
    parser.add_argument(
        "--text_dir",
        default="",
        help="Directory containing spk*_text files such as spk1_text, spk2_text, ...",
    )
    parser.add_argument(
        "--text_file",
        action="append",
        default=[],
        help="Repeatable role mapping in the form spkN=/path/to/text_file",
    )
    parser.add_argument("wav_file")
    parser.add_argument("utt2spk_file")
    parser.add_argument("shards_dir")
    parser.add_argument("shards_list")
    return parser.parse_args()


def collect_role_text_maps(args):
    role_text_paths = {}
    if args.spk1_text:
        role_text_paths["spk1"] = Path(args.spk1_text)
    if args.spk2_text:
        role_text_paths["spk2"] = Path(args.spk2_text)

    if args.text_dir:
        for path in sorted(Path(args.text_dir).glob("spk*_text")):
            stem = path.stem
            if not stem.endswith("_text"):
                continue
            role = stem[:-5]
            role_index(role)
            role_text_paths[role] = path

    for item in args.text_file:
        if "=" not in item:
            raise ValueError(f"--text_file must be role=path, got: {item}")
        role, path = item.split("=", 1)
        role = role.strip()
        role_index(role)
        role_text_paths[role] = Path(path.strip())

    if not role_text_paths:
        raise ValueError("No role text files were provided")

    return {
        role: read_2column_text(path)
        for role, path in sorted(role_text_paths.items(), key=lambda item: role_index(item[0]))
    }


def build_records(args):
    mix_map = read_mix_scp(args.wav_file)
    mix_speakers = read_mix_speakers(args.utt2spk_file)
    role_texts = collect_role_text_maps(args)

    records = []
    for mix_id, mix_info in mix_map.items():
        if mix_id not in mix_speakers:
            raise KeyError(f"Missing speaker ids for {mix_id}")
        speakers = mix_speakers[mix_id]
        mix_wav = mix_info["mix_wav"]
        source_wavs = mix_info.get("source_wavs", [])
        for role, text_map in role_texts.items():
            if mix_id not in text_map:
                continue
            speaker_idx = role_index(role) - 1
            if speaker_idx >= len(speakers):
                raise IndexError(
                    f"{mix_id} has text for {role} but only {len(speakers)} speaker ids in {args.utt2spk_file}"
                )
            if speaker_idx >= len(source_wavs):
                raise IndexError(
                    f"{mix_id} has text for {role} but only {len(source_wavs)} clean source paths in {args.wav_file}"
                )
            records.append(
                {
                    "record_key": f"{mix_id}__{role}",
                    "mix_id": mix_id,
                    "mix_wav": mix_wav,
                    "target_wav": source_wavs[speaker_idx],
                    "speaker": speakers[speaker_idx],
                    "target_role": role,
                    "target_text": text_map[mix_id],
                }
            )
    return records


def main():
    args = get_args()
    random.seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data = build_records(args)
    if args.shuffle:
        random.shuffle(data)

    num = args.num_utts_per_shard
    chunks = [data[i : i + num] for i in range(0, len(data), num)]
    os.makedirs(args.shards_dir, exist_ok=True)

    pool = multiprocessing.Pool(processes=args.num_threads)
    shards_list = []
    num_chunks = len(chunks)
    for index, chunk in enumerate(chunks):
        tar_file = os.path.join(args.shards_dir, f"{args.prefix}_{index:09d}.tar")
        shards_list.append(tar_file)
        pool.apply_async(write_tar_file, (chunk, tar_file, index, num_chunks))

    pool.close()
    pool.join()

    with open(args.shards_list, "w", encoding="utf8") as fout:
        for name in shards_list:
            fout.write(name + "\n")


if __name__ == "__main__":
    main()
