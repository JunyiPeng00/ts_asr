from __future__ import annotations

import tarfile
from typing import Any

from .contract import CanonicalTargetSpeakerSample
from .io import load_audio_bytes


def index_tsasr_shards(shard_paths: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    required_fields = {"wav", "targetwav", "spk", "txt", "role", "mixid"}
    for shard_path in shard_paths:
        grouped: dict[str, dict[str, Any]] = {}
        with tarfile.open(shard_path, "r") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                pos = member.name.rfind(".")
                if pos <= 0:
                    continue
                prefix = member.name[:pos]
                postfix = member.name[pos + 1 :]
                grouped.setdefault(prefix, {"members": {}, "sizes": {}})
                grouped[prefix]["members"][postfix] = member.name
                grouped[prefix]["sizes"][postfix] = member.size
        for prefix, record_info in grouped.items():
            member_map = record_info["members"]
            missing = required_fields.difference(member_map)
            if missing:
                raise ValueError(f"Shard record {prefix} in {shard_path} is missing fields: {sorted(missing)}")
            records.append(
                {
                    "shard_path": shard_path,
                    "prefix": prefix,
                    "members": member_map,
                    "length_hint": int(record_info["sizes"].get("wav", 0)),
                }
            )
    return records


def load_tsasr_shard_sample(record: dict[str, Any], sampling_rate: int) -> CanonicalTargetSpeakerSample:
    payload: dict[str, Any] = {}
    with tarfile.open(record["shard_path"], "r") as tar:
        for postfix, member_name in record["members"].items():
            extracted = tar.extractfile(member_name)
            if extracted is None:
                raise FileNotFoundError(f"Missing {member_name} in {record['shard_path']}")
            data = extracted.read()
            if postfix == "wav":
                payload["mix_audio"] = load_audio_bytes(data, sr=sampling_rate)
            elif postfix == "targetwav":
                payload["target_audio"] = load_audio_bytes(data, sr=sampling_rate)
            else:
                payload[postfix] = data.decode("utf-8").strip()

    return CanonicalTargetSpeakerSample(
        example_id=record["prefix"],
        mix_id=payload["mixid"],
        target_role=payload["role"],
        target_spk=payload["spk"],
        mix_audio=payload["mix_audio"],
        target_audio=payload.get("target_audio"),
        enroll_audio=None,
        sample_rate=int(sampling_rate),
        target_text=payload.get("txt"),
        metadata={
            "shard_path": record["shard_path"],
            "members": dict(record["members"]),
        },
    )
