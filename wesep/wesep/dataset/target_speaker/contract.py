from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CanonicalTargetSpeakerSample:
    example_id: str
    mix_id: str
    target_role: str
    target_spk: str
    mix_audio: np.ndarray
    target_audio: np.ndarray | None
    enroll_audio: np.ndarray | None
    sample_rate: int
    target_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
