from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from wesep.modules.target_speaker.role_utils import role_to_index, validate_role

from .io import load_json, read_2column_text, resolve_audio_path


def _select_enrollment_path(
    speaker_id: str,
    target_utt: str | None,
    speaker_pool: Mapping[str, list[list[str]]],
    exclude_path: str | None = None,
) -> str:
    candidates = speaker_pool.get(speaker_id)
    if not candidates:
        raise KeyError(f"No enrollment candidates found for speaker {speaker_id}")
    for candidate_utt, candidate_path in candidates:
        resolved_path = resolve_audio_path(candidate_path)
        if target_utt is not None and candidate_utt == target_utt:
            continue
        if exclude_path is not None and resolved_path == exclude_path:
            continue
        return resolved_path
    return resolve_audio_path(candidates[0][1])


def _resolve_fixed_enrollment(
    enroll_key_or_path: str,
    single_wav_map: Mapping[str, str],
) -> str:
    if enroll_key_or_path in single_wav_map:
        return resolve_audio_path(single_wav_map[enroll_key_or_path])
    return resolve_audio_path(enroll_key_or_path)


@dataclass
class EnrollmentResolver:
    mode: str
    speaker_pool: Mapping[str, list[list[str]]] = field(default_factory=dict)
    train_single_wav_map: Mapping[str, str] = field(default_factory=dict)
    eval_enroll_maps: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    eval_single_wav_map: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_paths(
        cls,
        mode: str,
        train_spk2utt: str = "",
        train_single_wav_scp: str = "",
        eval_enroll_paths: Mapping[str, str] | None = None,
        eval_enroll_paths_json: str = "",
        eval_single_wav_scp: str = "",
    ) -> "EnrollmentResolver":
        speaker_pool = load_json(train_spk2utt) if mode == "train" and train_spk2utt else {}
        train_single_wav_map = read_2column_text(train_single_wav_scp) if mode == "train" and train_single_wav_scp else {}
        eval_enroll_maps = {}
        if mode != "train" and eval_enroll_paths:
            eval_enroll_maps = {
                validate_role(role): read_2column_text(path)
                for role, path in eval_enroll_paths.items()
                if path
            }
        if mode != "train" and eval_enroll_paths_json:
            json_payload = load_json(eval_enroll_paths_json)
            if not isinstance(json_payload, Mapping):
                raise TypeError(
                    "eval_enroll_paths_json must point to a JSON object that maps roles to enrollment map paths"
                )
            eval_enroll_maps.update(
                {
                    validate_role(str(role)): read_2column_text(str(path))
                    for role, path in json_payload.items()
                    if path
                }
            )
        eval_single_wav_map = read_2column_text(eval_single_wav_scp) if mode != "train" and eval_single_wav_scp else {}
        return cls(
            mode=mode,
            speaker_pool=speaker_pool,
            train_single_wav_map=train_single_wav_map,
            eval_enroll_maps=eval_enroll_maps,
            eval_single_wav_map=eval_single_wav_map,
        )

    def resolve_train(self, target_spk: str, target_role: str, mix_id: str) -> str:
        target_role = validate_role(target_role)
        if "_" in mix_id and target_role in {"spk1", "spk2"}:
            spk1_utt, spk2_utt = mix_id.split("_", 1)
            target_utt = spk1_utt if target_role == "spk1" else spk2_utt
            return _select_enrollment_path(target_spk, target_utt, self.speaker_pool)

        exclude_path = None
        single_key = f"s{role_to_index(target_role)}/{mix_id}.wav"
        if single_key in self.train_single_wav_map:
            exclude_path = resolve_audio_path(self.train_single_wav_map[single_key])
        return _select_enrollment_path(target_spk, None, self.speaker_pool, exclude_path=exclude_path)

    def resolve_eval(self, target_role: str, mix_id: str) -> str:
        target_role = validate_role(target_role)
        enroll_map = self.eval_enroll_maps.get(target_role)
        if not enroll_map:
            raise KeyError(f"Missing fixed enrollment map for role {target_role}")
        if mix_id not in enroll_map:
            raise KeyError(f"Missing enrollment mapping for {target_role} of {mix_id}")
        return _resolve_fixed_enrollment(enroll_map[mix_id], self.eval_single_wav_map)

    def resolve(self, target_spk: str, target_role: str, mix_id: str) -> str:
        if self.mode == "train":
            return self.resolve_train(target_spk=target_spk, target_role=target_role, mix_id=mix_id)
        return self.resolve_eval(target_role=target_role, mix_id=mix_id)
