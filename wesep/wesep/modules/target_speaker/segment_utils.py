from __future__ import annotations

from typing import Sequence

import torch


def qwen_audio_token_length(input_lengths: int | torch.Tensor) -> int | torch.Tensor:
    if torch.is_tensor(input_lengths):
        input_lengths = input_lengths.long()
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths.long()

    value = int(input_lengths)
    value_leave = value % 100
    feat_lengths = (value_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (value // 100) * 13
    return int(output_lengths)


def validate_three_segment_lengths(lengths: Sequence[int | torch.Tensor]) -> None:
    if len(lengths) != 3:
        raise ValueError(f"Expected 3 segment lengths [enroll, silence, mix], got {len(lengths)}")


def reconcile_three_lengths(total_length: int, lengths: Sequence[int | torch.Tensor]) -> torch.LongTensor:
    validate_three_segment_lengths(lengths)
    total = max(int(total_length), 0)
    raw = [max(int(x), 0) for x in lengths]
    if total == 0:
        return torch.zeros(3, dtype=torch.long)

    first = min(raw[0], total)
    second = min(raw[1], total - first)
    third = max(total - first - second, 0)
    return torch.tensor([first, second, third], dtype=torch.long)
