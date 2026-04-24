from __future__ import annotations

from typing import Tuple

import numpy as np


def concat_enroll_silence_mix(
    enroll_audio: np.ndarray,
    mix_audio: np.ndarray,
    sample_rate: int,
    silence_seconds: float = 1.0,
) -> Tuple[np.ndarray, tuple[int, int, int]]:
    enroll_audio = np.asarray(enroll_audio, dtype=np.float32)
    mix_audio = np.asarray(mix_audio, dtype=np.float32)
    silence_length = max(int(round(float(sample_rate) * float(silence_seconds))), 1)
    silence_audio = np.zeros(silence_length, dtype=np.float32)
    concat_audio = np.concatenate([enroll_audio, silence_audio, mix_audio], axis=0).astype(np.float32)
    return concat_audio, (int(enroll_audio.shape[0]), silence_length, int(mix_audio.shape[0]))
