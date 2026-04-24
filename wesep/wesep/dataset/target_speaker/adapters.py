from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as tf

from wesep.modules.target_speaker.role_utils import iter_standard_roles, role_to_index, sorted_roles, validate_role

from .contract import CanonicalTargetSpeakerSample


def _to_numpy_audio(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        array = value
    elif torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    return np.asarray(array, dtype=np.float32)


def parse_tse_shard_record(
    mix_record: Mapping[str, Any],
    role_texts: Mapping[str, str] | None = None,
) -> list[CanonicalTargetSpeakerSample]:
    if "wav_mix" not in mix_record:
        raise KeyError("mix_record must contain 'wav_mix'")

    sample_rate = int(mix_record.get("sample_rate", 16000))
    roles = [
        key
        for key in mix_record.keys()
        if key.startswith("spk") and key[3:].isdigit() and "_" not in key
    ]
    roles = sorted_roles(roles[: int(mix_record.get("num_speaker", len(roles)))])

    samples: list[CanonicalTargetSpeakerSample] = []
    mix_audio = _to_numpy_audio(mix_record["wav_mix"])
    if mix_audio is None:
        raise ValueError("mix_record['wav_mix'] must not be None")
    for role in roles:
        validate_role(role)
        target_audio = _to_numpy_audio(mix_record.get(f"wav_{role}"))
        if target_audio is None:
            target_audio = _to_numpy_audio(mix_record.get(f"wav_spk{role_to_index(role)}"))
        enroll_audio = _to_numpy_audio(mix_record.get(f"enroll_audio_{role}"))
        target_text = role_texts.get(role) if role_texts is not None else None
        metadata = {
            "legacy_key": mix_record.get("key", ""),
            "legacy_tse_enroll": mix_record.get(f"embed_{role}"),
            "speaker_label": mix_record.get(f"{role}_label"),
            "enroll_path": mix_record.get(f"enroll_path_{role}", ""),
        }
        samples.append(
            CanonicalTargetSpeakerSample(
                example_id=f"{mix_record.get('key', '')}__{role}",
                mix_id=str(mix_record.get("key", "")),
                target_role=role,
                target_spk=str(mix_record[role]),
                mix_audio=mix_audio.copy(),
                target_audio=target_audio,
                enroll_audio=enroll_audio,
                sample_rate=sample_rate,
                target_text=target_text,
                metadata=metadata,
            )
        )
    return samples


def expand_target_speaker_examples(
    mix_record: Mapping[str, Any],
    role_texts: Mapping[str, str] | None = None,
) -> list[CanonicalTargetSpeakerSample]:
    return parse_tse_shard_record(mix_record, role_texts=role_texts)


def _adjust_tse_enroll_lengths(spk_embeds: list[torch.Tensor], mode: str) -> list[int]:
    lengths = [tensor.shape[1] for tensor in spk_embeds]
    if len(set(lengths)) == 1:
        return lengths
    if mode == "max":
        target_len = max(lengths)
        for index, tensor in enumerate(spk_embeds):
            if tensor.ndim == 2:
                spk_embeds[index] = tf.pad(tensor, (0, target_len - tensor.shape[1]), "constant", 0)
            else:
                spk_embeds[index] = tf.pad(tensor, (0, 0, 0, target_len - tensor.shape[1]), "constant", 0)
    else:
        target_len = min(lengths)
        for index, tensor in enumerate(spk_embeds):
            if tensor.ndim == 2:
                spk_embeds[index] = tensor[:, :target_len]
            else:
                spk_embeds[index] = tensor[:, :target_len, :]
    return [tensor.shape[1] for tensor in spk_embeds]


def canonical_samples_to_tse_batch(samples: Iterable[CanonicalTargetSpeakerSample], mode: str = "min") -> dict[str, Any]:
    wav_mix = []
    wav_targets = []
    spk_embeds = []
    spk = []
    key = []
    spk_label = []
    target_role = []
    metadata = []

    for sample in samples:
        wav_mix.append(torch.from_numpy(sample.mix_audio.copy()).unsqueeze(0))
        if sample.target_audio is None:
            raise ValueError("TSE adapter requires target_audio for every canonical sample")
        wav_targets.append(torch.from_numpy(sample.target_audio.copy()).unsqueeze(0))
        spk.append(sample.target_spk)
        key.append(sample.metadata.get("legacy_key", sample.mix_id))
        target_role.append(sample.target_role)
        metadata.append(dict(sample.metadata))

        enroll_feature = sample.metadata.get("legacy_tse_enroll")
        if enroll_feature is None and sample.enroll_audio is not None:
            enroll_feature = np.expand_dims(sample.enroll_audio, axis=0)
        if enroll_feature is None:
            raise ValueError("TSE adapter requires legacy_tse_enroll or enroll_audio")
        if torch.is_tensor(enroll_feature):
            enroll_tensor = enroll_feature.detach().clone()
        else:
            enroll_tensor = torch.from_numpy(np.asarray(enroll_feature).copy())
        spk_embeds.append(enroll_tensor)
        if sample.metadata.get("speaker_label") is not None:
            spk_label.append(sample.metadata["speaker_label"])

    length_spk_embeds = _adjust_tse_enroll_lengths(spk_embeds, mode=mode)
    batch = {
        "wav_mix": torch.concat(wav_mix),
        "wav_targets": torch.concat(wav_targets),
        "spk_embeds": torch.concat(spk_embeds),
        "length_spk_embeds": length_spk_embeds,
        "spk": spk,
        "key": key,
        "spk_label": torch.as_tensor(spk_label),
        "target_role": target_role,
        "metadata": metadata,
    }
    return batch


def canonical_sample_to_tsasr_item(
    sample: CanonicalTargetSpeakerSample,
    *,
    enroll_wav: str,
    language: str,
    prompt: str,
) -> dict[str, Any]:
    if sample.enroll_audio is None:
        raise ValueError("TS-ASR item conversion requires enroll_audio")
    if sample.target_text is None or not str(sample.target_text).strip():
        raise ValueError("TS-ASR item conversion requires non-empty target_text")
    return {
        "key": sample.example_id,
        "mix_id": sample.mix_id,
        "target_role": sample.target_role,
        "target_spk": sample.target_spk,
        "mix_audio": sample.mix_audio,
        "target_audio": sample.target_audio,
        "enroll_wav": enroll_wav,
        "enroll_audio": sample.enroll_audio,
        "target_text": sample.target_text or "",
        "language": language,
        "prompt": prompt,
    }
