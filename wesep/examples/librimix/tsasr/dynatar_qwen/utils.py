from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

from wesep.dataset.target_speaker.io import read_2column_text
from wesep.modules.target_speaker.segment_utils import qwen_audio_token_length, reconcile_three_lengths


def load_jsonl(path: str | Path) -> List[dict]:
    records: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def dump_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_target_text(language: str | None, transcript: str) -> str:
    transcript = transcript.strip()
    if "<asr_text>" in transcript:
        return transcript
    language = (language or "None").strip() or "None"
    return f"language {language}<asr_text>{transcript}"


def expand_audio_token(prefix_text: str, audio_token: str, audio_token_count: int) -> str:
    if audio_token_count <= 0:
        raise ValueError(f"audio_token_count must be positive, got {audio_token_count}")
    if audio_token not in prefix_text:
        raise ValueError("prefix_text does not contain the audio special token")
    return prefix_text.replace(audio_token, audio_token * audio_token_count, 1)
