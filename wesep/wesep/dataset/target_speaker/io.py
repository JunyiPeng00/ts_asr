from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import soundfile as sf


_PATH_REWRITES = [
    (
        "/tmp/libri2mix_min_100_360/Libri2Mix/wav16k/max/",
        "/flash/project_465002316/junyi/Libri2Mix/wav16k/min/",
    ),
]


def read_list_file(path: str | Path) -> List[str]:
    list_path = Path(path)
    base_dir = list_path.parent
    entries: List[str] = []
    with list_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = Path(line)
            if not entry.is_absolute():
                entry = base_dir / entry
            entries.append(str(entry.resolve(strict=False)))
    return entries


def read_2column_text(path: str | Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(maxsplit=1)
            mapping[key] = value
    return mapping


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_audio_path(path: str) -> str:
    if not path:
        return path
    candidate_paths = [path]
    for source_prefix, target_prefix in _PATH_REWRITES:
        if path.startswith(source_prefix):
            candidate_paths.append(path.replace(source_prefix, target_prefix, 1))
    if "/Libri2Mix/wav16k/max/" in path:
        candidate_paths.append(path.replace("/Libri2Mix/wav16k/max/", "/Libri2Mix/wav16k/min/", 1))
    for candidate in candidate_paths:
        if Path(candidate).exists():
            return candidate
    return candidate_paths[-1]


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    resolved = resolve_audio_path(path)
    wav, _ = librosa.load(resolved, sr=sr, mono=True)
    return wav.astype(np.float32)


def load_audio_bytes(audio_bytes: bytes, sr: int = 16000) -> np.ndarray:
    wav, source_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if int(source_sr) != int(sr):
        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=sr)
    return wav.astype(np.float32)


def crop_or_pad_audio(
    audio: np.ndarray,
    target_num_samples: int,
    random_crop: bool = False,
) -> np.ndarray:
    if target_num_samples <= 0:
        return np.asarray(audio, dtype=np.float32)

    audio = np.asarray(audio, dtype=np.float32)
    current_num_samples = int(audio.shape[0])
    if current_num_samples == target_num_samples:
        return audio
    if current_num_samples > target_num_samples:
        max_start = current_num_samples - target_num_samples
        if random_crop and max_start > 0:
            start = int(np.random.randint(0, max_start + 1))
        else:
            start = max_start // 2
        return audio[start : start + target_num_samples].astype(np.float32)

    padded = np.zeros(target_num_samples, dtype=np.float32)
    padded[:current_num_samples] = audio
    return padded
