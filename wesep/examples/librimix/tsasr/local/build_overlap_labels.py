from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import math
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TSASR_ROOT = Path(__file__).resolve().parents[1]
if str(TSASR_ROOT) not in sys.path:
    sys.path.insert(0, str(TSASR_ROOT))

from dynatar_qwen.aux_lmdb import (
    DEFAULT_COMMIT_INTERVAL,
    DEFAULT_LMDB_MAP_SIZE,
    LMDBWriter,
    encode_aux_payload,
    remove_lmdb_path,
)


def read_wav_scp(path: str) -> list[tuple[str, str, list[str]]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"wav.scp line must contain key mix_wav src1 [...]: {line}")
            key = parts[0]
            mix_wav = parts[1]
            source_wavs = parts[2:]
            records.append((key, mix_wav, source_wavs))
    return records


def read_utt2spk(path: str) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"utt2spk line must contain key spk1 [...]: {line}")
            mapping[parts[0]] = parts[1:]
    return mapping


def read_cutset_offsets(path: str) -> dict[str, list[float]]:
    offsets: dict[str, list[float]] = {}
    open_fn = gzip.open if str(path).endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = record.get("id") or record.get("cut_id")
            tracks = record.get("tracks", [])
            if not key or not isinstance(tracks, list):
                continue
            offsets[str(key)] = [float(track.get("offset", 0.0)) for track in tracks]
    return offsets


def load_audio(path: str, sr: int) -> np.ndarray:
    wav, source_sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if int(source_sr) != int(sr):
        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=sr)
    return np.asarray(wav, dtype=np.float32)


def estimate_num_samples(path: str, target_sr: int) -> int:
    info = sf.info(path)
    if int(info.samplerate) == int(target_sr):
        return int(info.frames)
    return max(int(round(float(info.frames) * float(target_sr) / float(info.samplerate))), 0)


def num_frames(num_samples: int, frame_len: int, hop_len: int) -> int:
    if num_samples <= 0:
        return 0
    if num_samples <= frame_len:
        return 1
    return 1 + math.ceil((num_samples - frame_len) / hop_len)


def compute_source_activity(
    wav: np.ndarray,
    frame_len: int,
    hop_len: int,
    threshold_db: float,
) -> np.ndarray:
    total_frames = num_frames(int(wav.shape[0]), frame_len, hop_len)
    if total_frames <= 0:
        return np.zeros((0,), dtype=np.uint8)
    padded_len = frame_len + max(total_frames - 1, 0) * hop_len
    padded = np.zeros((padded_len,), dtype=np.float32)
    padded[: wav.shape[0]] = wav
    energies = np.zeros((total_frames,), dtype=np.float32)
    for frame_idx in range(total_frames):
        start = frame_idx * hop_len
        frame = padded[start : start + frame_len]
        energies[frame_idx] = float(np.mean(frame * frame))
    ref_db = 10.0 * np.log10(max(float(energies.max()), 1.0e-8))
    frame_db = 10.0 * np.log10(np.maximum(energies, 1.0e-8))
    return (frame_db >= (ref_db - threshold_db)).astype(np.uint8)


def place_activity_on_mix_timeline(
    source_activity: np.ndarray,
    mix_frames: int,
    offset_seconds: float,
    sr: int,
    hop_len: int,
) -> np.ndarray:
    placed = np.zeros((mix_frames,), dtype=np.uint8)
    if mix_frames <= 0 or source_activity.size == 0:
        return placed
    shift_frames = max(int(round(offset_seconds * sr / hop_len)), 0)
    end = min(mix_frames, shift_frames + source_activity.shape[0])
    if end > shift_frames:
        placed[shift_frames:end] = source_activity[: end - shift_frames]
    return placed


def build_target_labels(source_activity: np.ndarray, target_index: int) -> dict[str, np.ndarray | int]:
    target_active = source_activity[:, target_index].astype(np.uint8)
    interferer_count = (source_activity.sum(axis=1) - target_active).clip(min=0).astype(np.uint8)
    active_speaker_count = source_activity.sum(axis=1).astype(np.uint8)

    router_label = np.zeros((source_activity.shape[0],), dtype=np.int8)
    router_label[(target_active == 1) & (interferer_count == 0)] = 1
    router_label[(target_active == 1) & (interferer_count > 0)] = 2
    router_label[(target_active == 0) & (interferer_count > 0)] = 3

    return {
        "router_label": router_label,
        "target_active": target_active,
        "interferer_count": interferer_count,
        "active_speaker_count": active_speaker_count,
        "source_activity": source_activity.astype(np.uint8),
        "target_index": np.asarray(target_index, dtype=np.int64),
        "num_sources": np.asarray(source_activity.shape[1], dtype=np.int64),
    }


def _build_mix_records(task):
    (
        mix_id,
        mix_wav,
        source_wavs,
        speaker_ids,
        offsets,
        sr,
        frame_len,
        hop_len,
        threshold_db,
        lmdb_name,
    ) = task

    mix_frames = num_frames(estimate_num_samples(mix_wav, target_sr=sr), frame_len, hop_len)
    activities = []
    for source_wav, offset_seconds in zip(source_wavs, offsets):
        source_audio = load_audio(source_wav, sr=sr)
        source_activity = compute_source_activity(
            source_audio,
            frame_len=frame_len,
            hop_len=hop_len,
            threshold_db=threshold_db,
        )
        activities.append(
            place_activity_on_mix_timeline(
                source_activity,
                mix_frames=mix_frames,
                offset_seconds=float(offset_seconds),
                sr=sr,
                hop_len=hop_len,
            )
        )

    source_activity = np.stack(activities, axis=-1) if activities else np.zeros((0, 0), dtype=np.uint8)
    results = []
    for target_index, speaker_id in enumerate(speaker_ids):
        target_role = f"spk{target_index + 1}"
        lmdb_key = f"{mix_id}__{target_role}"
        label_payload = build_target_labels(source_activity, target_index=target_index)
        manifest_record = {
            "storage": "lmdb",
            "key": lmdb_key,
            "mix_id": mix_id,
            "target_role": target_role,
            "target_index": target_index,
            "num_sources": len(source_wavs),
            "mix_wav": mix_wav,
            "source_wavs": source_wavs,
            "speaker_ids": speaker_ids,
            "target_spk": speaker_id,
            "lmdb_path": lmdb_name,
            "lmdb_key": lmdb_key,
        }
        results.append((lmdb_key, encode_aux_payload(label_payload), manifest_record))
    return results


def parse_args():
    parser = argparse.ArgumentParser("Build overlap sidecar labels for TS-ASR")
    parser.add_argument("--wav_scp", type=str, required=True)
    parser.add_argument("--utt2spk", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cutset_jsonl_gz", type=str, default="")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--frame_ms", type=float, default=10.0)
    parser.add_argument("--window_ms", type=float, default=25.0)
    parser.add_argument("--energy_db_threshold", type=float, default=40.0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--map_size", type=int, default=DEFAULT_LMDB_MAP_SIZE)
    parser.add_argument("--commit_interval", type=int, default=DEFAULT_COMMIT_INTERVAL)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_lmdb = output_dir / "aux_labels.lmdb"
    manifest_path = output_dir / "aux_label_manifest.jsonl"
    temp_labels_lmdb = output_dir / "aux_labels.lmdb.tmp"
    temp_manifest_path = output_dir / "aux_label_manifest.jsonl.tmp"
    remove_lmdb_path(temp_labels_lmdb)
    temp_manifest_path.unlink(missing_ok=True)

    frame_len = max(int(round(args.window_ms * args.sr / 1000.0)), 1)
    hop_len = max(int(round(args.frame_ms * args.sr / 1000.0)), 1)
    wav_records = read_wav_scp(args.wav_scp)
    utt2spk = read_utt2spk(args.utt2spk)
    cutset_offsets = read_cutset_offsets(args.cutset_jsonl_gz) if args.cutset_jsonl_gz else {}
    mix_tasks = []
    for mix_id, mix_wav, source_wavs in wav_records:
        speaker_ids = utt2spk.get(mix_id, [])
        if len(speaker_ids) != len(source_wavs):
            raise ValueError(
                f"Speaker/source mismatch for {mix_id}: {len(speaker_ids)} speakers vs {len(source_wavs)} sources"
            )
        offsets = cutset_offsets.get(mix_id, [0.0] * len(source_wavs))
        if len(offsets) < len(source_wavs):
            offsets = offsets + [0.0] * (len(source_wavs) - len(offsets))
        mix_tasks.append(
            (
                mix_id,
                mix_wav,
                source_wavs,
                speaker_ids,
                offsets,
                int(args.sr),
                frame_len,
                hop_len,
                float(args.energy_db_threshold),
                labels_lmdb.name,
            )
        )

    num_records = 0
    with LMDBWriter(
        temp_labels_lmdb,
        map_size=args.map_size,
        commit_interval=args.commit_interval,
        overwrite=True,
    ) as lmdb_writer, temp_manifest_path.open("w", encoding="utf-8") as handle:
        worker_count = max(int(args.num_workers), 1)
        chunksize = max(len(mix_tasks) // (worker_count * 8), 1) if mix_tasks else 1
        if worker_count == 1:
            mix_results_iter = map(_build_mix_records, mix_tasks)
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=worker_count)
            mix_results_iter = executor.map(_build_mix_records, mix_tasks, chunksize=chunksize)
        try:
            for mix_results in mix_results_iter:
                for lmdb_key, payload_bytes, manifest_record in mix_results:
                    lmdb_writer.put(lmdb_key, payload_bytes)
                    handle.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")
                    num_records += 1
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)

    remove_lmdb_path(labels_lmdb)
    temp_labels_lmdb.replace(labels_lmdb)
    manifest_path.unlink(missing_ok=True)
    temp_manifest_path.replace(manifest_path)

    print(f"[overlap-labels] wrote {num_records} target labels")
    print(f"[overlap-labels] lmdb={labels_lmdb}")
    print(f"[overlap-labels] manifest={manifest_path}")


if __name__ == "__main__":
    main()
