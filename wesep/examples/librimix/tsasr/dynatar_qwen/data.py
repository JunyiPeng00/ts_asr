from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from wesep.dataset.target_speaker import (
    EnrollmentResolver,
    canonical_sample_to_tsasr_item,
    crop_or_pad_audio,
    index_tsasr_shards,
    load_audio,
    load_tsasr_shard_sample,
    read_list_file,
)
from wesep.modules.target_speaker import (
    concat_enroll_silence_mix,
    qwen_audio_token_length,
    reconcile_three_lengths,
)

from .aux_lmdb import decode_aux_payload_bytes, open_readonly_lmdb, resolve_sidecar_path
from .utils import build_target_text, expand_audio_token


@dataclass(frozen=True)
class AuxLabelLMDBRecord:
    lmdb_path: str
    lmdb_key: str


class AuxLabelLMDBCache:
    def __init__(self) -> None:
        self._envs: dict[str, lmdb.Environment] = {}

    def _get_env(self, lmdb_path: str) -> lmdb.Environment:
        env = self._envs.get(lmdb_path)
        if env is None:
            env = open_readonly_lmdb(lmdb_path)
            self._envs[lmdb_path] = env
        return env

    def load(self, ref: AuxLabelLMDBRecord) -> dict[str, np.ndarray]:
        env = self._get_env(ref.lmdb_path)
        with env.begin(write=False) as txn:
            payload_bytes = txn.get(ref.lmdb_key.encode("utf-8"))
        if payload_bytes is None:
            raise KeyError(f"Missing aux-label LMDB key {ref.lmdb_key} in {ref.lmdb_path}")
        return decode_aux_payload_bytes(bytes(payload_bytes))


def _load_aux_label_manifest(
    manifest_path: str,
) -> tuple[dict[str, AuxLabelLMDBRecord], dict[tuple[str, str], AuxLabelLMDBRecord]]:
    by_key: dict[str, AuxLabelLMDBRecord] = {}
    by_mix_role: dict[tuple[str, str], AuxLabelLMDBRecord] = {}
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            storage = str(record.get("storage", "")).strip().lower()
            if storage != "lmdb":
                raise ValueError(f"Expected storage=lmdb in {manifest_path}:{line_number}")
            lmdb_path_value = record.get("lmdb_path")
            lmdb_key = str(record.get("lmdb_key", "")).strip()
            if not lmdb_path_value:
                raise KeyError(f"Missing lmdb_path in {manifest_path}:{line_number}")
            if not lmdb_key:
                raise KeyError(f"Missing lmdb_key in {manifest_path}:{line_number}")
            ref = AuxLabelLMDBRecord(
                lmdb_path=resolve_sidecar_path(path, str(lmdb_path_value)),
                lmdb_key=lmdb_key,
            )
            key = str(record.get("key", "")).strip() or lmdb_key
            mix_id = str(record.get("mix_id", "")).strip()
            target_role = str(record.get("target_role", "")).strip()
            if key:
                by_key[key] = ref
            if mix_id and target_role:
                by_mix_role[(mix_id, target_role)] = ref
    return by_key, by_mix_role


def _pad_audio_batch(audios: List[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    if not audios:
        raise ValueError("Expected at least one audio array to pad")
    lengths = torch.tensor([int(audio.shape[0]) for audio in audios], dtype=torch.long)
    max_len = int(lengths.max().item())
    batch = torch.zeros(len(audios), max_len, dtype=torch.float32)
    for idx, audio in enumerate(audios):
        length = int(audio.shape[0])
        if length > 0:
            batch[idx, :length] = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    return batch, lengths


def _resample_discrete_labels(labels: np.ndarray, target_len: int) -> tuple[np.ndarray, bool]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if target_len <= 0:
        return np.zeros((0,), dtype=np.int64), labels.shape[0] != 0
    if labels.shape[0] == target_len:
        return labels.astype(np.int64, copy=False), False
    if labels.shape[0] == 0:
        return np.full((target_len,), -100, dtype=np.int64), True
    positions = np.linspace(0, labels.shape[0] - 1, num=target_len)
    indices = np.rint(positions).astype(np.int64)
    indices = np.clip(indices, 0, labels.shape[0] - 1)
    return labels[indices].astype(np.int64, copy=False), True


class TSASRShardDataset(Dataset):
    def __init__(
        self,
        shard_list_path: str,
        split: str,
        sampling_rate: int = 16000,
        language: str = "English",
        prompt: str = "",
        enroll_crop_seconds: float = 0.0,
        train_spk2utt: str = "",
        train_single_wav_scp: str = "",
        eval_spk1_enroll: str = "",
        eval_spk2_enroll: str = "",
        eval_enroll_paths_json: str = "",
        eval_spk2utt: str = "",
        aux_label_manifest: str = "",
    ):
        self.shard_list_path = Path(shard_list_path)
        self.shard_paths = read_list_file(self.shard_list_path)
        self.index_cache_path = self.shard_list_path.with_name("shard.index.jsonl")
        self.index_source = "scan"
        self.split = split
        self.sampling_rate = sampling_rate
        self.language = language
        self.prompt = prompt
        self.use_pool_enroll = split == "train"
        self.enroll_crop_seconds = max(float(enroll_crop_seconds), 0.0)
        self.enroll_crop_samples = int(round(self.enroll_crop_seconds * self.sampling_rate))
        self.enrollment_resolver = EnrollmentResolver.from_paths(
            mode="train" if self.use_pool_enroll else "eval",
            train_spk2utt=train_spk2utt,
            train_single_wav_scp=train_single_wav_scp,
            eval_enroll_paths={
                "spk1": eval_spk1_enroll,
                "spk2": eval_spk2_enroll,
            },
            eval_enroll_paths_json=eval_enroll_paths_json,
            eval_single_wav_scp=eval_spk2utt,
        )
        self.aux_label_manifest = str(aux_label_manifest).strip()
        self.aux_labels_by_key: dict[str, AuxLabelLMDBRecord] = {}
        self.aux_labels_by_mix_role: dict[tuple[str, str], AuxLabelLMDBRecord] = {}
        self.aux_label_lmdb_cache = AuxLabelLMDBCache() if self.aux_label_manifest else None
        if self.aux_label_manifest:
            self.aux_labels_by_key, self.aux_labels_by_mix_role = _load_aux_label_manifest(self.aux_label_manifest)

        self.records = self._load_or_build_index()
        self.lengths = [int(record.get("length_hint", 0)) for record in self.records]

    def _load_or_build_index(self) -> List[Dict[str, Any]]:
        records = self._load_index_cache()
        if records is not None:
            self.index_source = "cache"
            return records
        records = index_tsasr_shards(self.shard_paths)
        self._save_index_cache(records)
        self.index_source = "scan"
        return records

    def _load_index_cache(self) -> List[Dict[str, Any]] | None:
        if not self.index_cache_path.is_file():
            return None

        shard_path_set = set(self.shard_paths)
        seen_shards = set()
        records: List[Dict[str, Any]] = []
        required_fields = {"shard_path", "prefix", "members", "length_hint"}
        try:
            with self.index_cache_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if not required_fields.issubset(record):
                        return None
                    shard_path = record["shard_path"]
                    if shard_path not in shard_path_set:
                        return None
                    members = record.get("members", {})
                    if not isinstance(members, dict):
                        return None
                    record["length_hint"] = int(record.get("length_hint", 0))
                    records.append(record)
                    seen_shards.add(shard_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

        if not records:
            return None
        if seen_shards != shard_path_set:
            return None
        return records

    def _save_index_cache(self, records: List[Dict[str, Any]]) -> None:
        tmp_path = self.index_cache_path.with_suffix(self.index_cache_path.suffix + ".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            tmp_path.replace(self.index_cache_path)
        except OSError:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def __len__(self) -> int:
        return len(self.records)

    def _lookup_enrollment(self, mix_id: str, target_role: str, speaker_id: str) -> str:
        return self.enrollment_resolver.resolve(
            target_spk=speaker_id,
            target_role=target_role,
            mix_id=mix_id,
        )

    def _lookup_aux_label_ref(self, key: str, mix_id: str, target_role: str) -> AuxLabelLMDBRecord | None:
        if key in self.aux_labels_by_key:
            return self.aux_labels_by_key[key]
        return self.aux_labels_by_mix_role.get((mix_id, target_role))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        sample = load_tsasr_shard_sample(record, sampling_rate=self.sampling_rate)
        enroll_path = self._lookup_enrollment(
            mix_id=sample.mix_id,
            target_role=sample.target_role,
            speaker_id=sample.target_spk,
        )
        enroll_audio = load_audio(enroll_path, sr=self.sampling_rate)
        if self.enroll_crop_samples > 0:
            enroll_audio = crop_or_pad_audio(
                enroll_audio,
                self.enroll_crop_samples,
                random_crop=self.use_pool_enroll,
            )
        sample.enroll_audio = enroll_audio
        sample.metadata["enroll_wav"] = enroll_path
        item = canonical_sample_to_tsasr_item(
            sample,
            enroll_wav=enroll_path,
            language=self.language,
            prompt=self.prompt,
        )

        aux_label_ref = self._lookup_aux_label_ref(
            key=item["key"],
            mix_id=item["mix_id"],
            target_role=item["target_role"],
        )
        if aux_label_ref:
            if self.aux_label_lmdb_cache is None:
                raise RuntimeError("Aux-label LMDB cache is not initialized")
            aux_payload = self.aux_label_lmdb_cache.load(aux_label_ref)
            if "router_label" not in aux_payload:
                raise KeyError(
                    f"Missing router_label in aux label LMDB record: {aux_label_ref.lmdb_path}:{aux_label_ref.lmdb_key}"
                )
            item["overlap_labels"] = np.asarray(aux_payload["router_label"], dtype=np.int64)
            if "target_active" in aux_payload:
                item["target_active"] = np.asarray(aux_payload["target_active"], dtype=np.int64)
            if "interferer_count" in aux_payload:
                item["interferer_count"] = np.asarray(aux_payload["interferer_count"], dtype=np.int64)
            item["aux_label_lmdb_path"] = aux_label_ref.lmdb_path
            item["aux_label_lmdb_key"] = aux_label_ref.lmdb_key
        return item


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


@dataclass
class DataCollatorForDynaTaRQwen:
    processor: Any
    sampling_rate: int = 16000
    silence_seconds: float = 1.0
    default_prompt: str = ""
    include_overlap_labels: bool = False
    include_target_audio: bool = False
    _prefix_cache: dict[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._prefix_cache = {}

    def _get_prefix_text(self, prompt: str) -> str:
        if prompt in self._prefix_cache:
            return self._prefix_cache[prompt]
        messages = build_prefix_messages(prompt, None)
        prefix = self.processor.apply_chat_template([messages], add_generation_prompt=True, tokenize=False)[0]
        self._prefix_cache[prompt] = prefix
        return prefix

    def _feature_length(self, audio: np.ndarray) -> int:
        features = self.processor.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return int(features["attention_mask"].sum().item())

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        eos = self.processor.tokenizer.eos_token or ""
        concat_audios: List[np.ndarray] = []
        segment_feature_lengths: List[List[int]] = []
        prompts: List[str] = []
        targets: List[str] = []
        target_audios: List[np.ndarray] = []
        raw_overlap_labels: List[np.ndarray | None] = []

        for item in features:
            enroll_audio = np.asarray(item["enroll_audio"], dtype=np.float32)
            mix_audio = np.asarray(item["mix_audio"], dtype=np.float32)
            concat_audio, segment_sample_lengths = concat_enroll_silence_mix(
                enroll_audio=enroll_audio,
                mix_audio=mix_audio,
                sample_rate=self.sampling_rate,
                silence_seconds=self.silence_seconds,
            )
            concat_audios.append(concat_audio)

            silence_audio = np.zeros(segment_sample_lengths[1], dtype=np.float32)
            segment_feature_lengths.append(
                [
                    self._feature_length(enroll_audio),
                    self._feature_length(silence_audio),
                    self._feature_length(mix_audio),
                ]
            )
            prompts.append(item.get("prompt", self.default_prompt))
            targets.append(build_target_text(item.get("language"), item["target_text"]))

            if self.include_target_audio:
                target_audio = item.get("target_audio")
                if target_audio is None:
                    raise ValueError("target_audio is required when include_target_audio=True")
                target_audios.append(np.asarray(target_audio, dtype=np.float32))

            if self.include_overlap_labels:
                overlap_labels = item.get("overlap_labels")
                raw_overlap_labels.append(
                    None if overlap_labels is None else np.asarray(overlap_labels, dtype=np.int64)
                )

        audio_inputs = self.processor.feature_extractor(
            concat_audios,
            sampling_rate=self.sampling_rate,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_features = audio_inputs["input_features"]
        feature_attention_mask = audio_inputs["attention_mask"].long()
        total_feature_lengths = feature_attention_mask.sum(dim=1).long()
        total_token_lengths = qwen_audio_token_length(total_feature_lengths)

        segment_feature_lengths_tensor = []
        segment_token_lengths_tensor = []
        expanded_prefixes = []
        full_texts = []
        shared_mix_token_lengths = []
        overlap_label_tensors = []
        overlap_label_resample_count = 0

        for sample_idx, (prompt, target, seg_feat_raw, total_feat_len, total_tok_len) in enumerate(
            zip(
                prompts,
                targets,
                segment_feature_lengths,
                total_feature_lengths.tolist(),
                total_token_lengths.tolist(),
            )
        ):
            seg_feat = reconcile_three_lengths(total_feat_len, seg_feat_raw)
            seg_tok_raw = qwen_audio_token_length(seg_feat)
            seg_tok = reconcile_three_lengths(total_tok_len, seg_tok_raw.tolist())
            segment_feature_lengths_tensor.append(seg_feat)
            segment_token_lengths_tensor.append(seg_tok)
            mix_token_len = int(seg_tok[-1].item())
            shared_mix_token_lengths.append(mix_token_len)

            prefix_text = self._get_prefix_text(prompt)
            expanded_prefix = expand_audio_token(prefix_text, self.processor.audio_token, mix_token_len)
            expanded_prefixes.append(expanded_prefix)
            full_texts.append(expanded_prefix + target + eos)

            if self.include_overlap_labels:
                labels = raw_overlap_labels[sample_idx]
                if labels is None:
                    resampled = np.full((mix_token_len,), -100, dtype=np.int64)
                    did_resample = False
                else:
                    resampled, did_resample = _resample_discrete_labels(labels, mix_token_len)
                if did_resample:
                    overlap_label_resample_count += 1
                overlap_label_tensors.append(torch.from_numpy(resampled).long())

        text_inputs = self.processor.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor.tokenizer(
            expanded_prefixes,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        labels = text_inputs["input_ids"].clone()
        prefix_lengths = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        for idx, prefix_len in enumerate(prefix_lengths):
            labels[idx, :prefix_len] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        batch = {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "segment_feature_lengths": torch.stack(segment_feature_lengths_tensor, dim=0).long(),
            "segment_token_lengths": torch.stack(segment_token_lengths_tensor, dim=0).long(),
            # Compatibility alias: the runtime still expects audio_feature_lengths,
            # but the underlying semantic is "mix token lengths".
            "audio_feature_lengths": torch.tensor(shared_mix_token_lengths, dtype=torch.long),
            "labels": labels,
        }

        if self.include_overlap_labels:
            padded_overlap = torch.nn.utils.rnn.pad_sequence(
                overlap_label_tensors,
                batch_first=True,
                padding_value=-100,
            )
            batch["overlap_labels"] = padded_overlap.long()
            # Keep this value 1-D so DataParallel scatter works when multiple
            # GPUs are visible in single-process smoke runs.
            batch["overlap_label_resample_count"] = torch.tensor(
                [overlap_label_resample_count],
                dtype=torch.long,
            )

        if self.include_target_audio:
            target_waveforms, target_wave_lengths = _pad_audio_batch(target_audios)
            target_audio_inputs = self.processor.feature_extractor(
                target_audios,
                sampling_rate=self.sampling_rate,
                padding=True,
                truncation=False,
                return_attention_mask=True,
                return_tensors="pt",
            )
            batch["target_waveforms"] = target_waveforms
            batch["target_wave_lengths"] = target_wave_lengths
            batch["target_input_features"] = target_audio_inputs["input_features"]
            batch["target_feature_attention_mask"] = target_audio_inputs["attention_mask"].long()

        return batch
