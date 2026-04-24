from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

ROOT = Path("/scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr")
WESEP_ROOT = Path("/scratch/project_465002316/junyi/tse/ts_asr/wesep")
QWEN_ROOT = Path("/scratch/project_465002316/junyi/tse/ts_asr/Qwen3-ASR")

for path in (str(ROOT), str(WESEP_ROOT), str(QWEN_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


class TensorDictDataset(Dataset):
    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        item = {}
        for key, value in sample.items():
            if torch.is_tensor(value):
                item[key] = value.clone()
            else:
                item[key] = value
        return item


def pythonpath_env():
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(ROOT), str(WESEP_ROOT), str(QWEN_ROOT), os.environ.get("PYTHONPATH", "")]
        ).strip(os.pathsep),
    }


def collate_tensor_dicts(samples):
    batch = {}
    for key in samples[0].keys():
        values = [sample[key] for sample in samples]
        if torch.is_tensor(values[0]):
            if key == "overlap_label_resample_count":
                batch[key] = torch.stack(values, dim=0).sum()
            else:
                batch[key] = torch.stack(values, dim=0)
        else:
            batch[key] = values
    return batch


def lazy_model_imports():
    from dynatar_qwen.modeling_ts_qwen3_asr import TargetAwareEncoderConfig, upgrade_qwen3_asr_model
    from dynatar_qwen.modules import RoleConstrainedMaskBuilder
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
        Qwen3ASRAudioEncoderConfig,
        Qwen3ASRConfig,
        Qwen3ASRTextConfig,
        Qwen3ASRThinkerConfig,
    )
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration

    return (
        TargetAwareEncoderConfig,
        upgrade_qwen3_asr_model,
        RoleConstrainedMaskBuilder,
        Qwen3ASRAudioEncoderConfig,
        Qwen3ASRConfig,
        Qwen3ASRTextConfig,
        Qwen3ASRThinkerConfig,
        Qwen3ASRForConditionalGeneration,
    )


def build_tiny_model(ts_config_kwargs=None):
    torch.manual_seed(0)
    (
        TargetAwareEncoderConfig,
        upgrade_qwen3_asr_model,
        _RoleConstrainedMaskBuilder,
        Qwen3ASRAudioEncoderConfig,
        Qwen3ASRConfig,
        Qwen3ASRTextConfig,
        Qwen3ASRThinkerConfig,
        Qwen3ASRForConditionalGeneration,
    ) = lazy_model_imports()
    audio_config = Qwen3ASRAudioEncoderConfig(
        num_mel_bins=16,
        encoder_layers=8,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        d_model=32,
        output_dim=48,
        max_source_positions=512,
        downsample_hidden_size=8,
        conv_chunksize=64,
    )
    text_config = Qwen3ASRTextConfig(
        vocab_size=128,
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=12,
        max_position_embeddings=256,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_scaling={"rope_type": "default", "mrope_section": [4, 4, 4]},
    )
    thinker_config = Qwen3ASRThinkerConfig(
        audio_config=audio_config,
        text_config=text_config,
        audio_token_id=127,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config = Qwen3ASRConfig(
        thinker_config=thinker_config,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    base_model = Qwen3ASRForConditionalGeneration(config)
    target_aware_defaults = {
        "num_role_layers": 5,
        "num_prototypes": 4,
        "mhfa_heads": 2,
        "mhfa_compression_dim": 16,
        "router_hidden_multiplier": 2,
    }
    if ts_config_kwargs:
        target_aware_defaults.update(ts_config_kwargs)
    model = upgrade_qwen3_asr_model(
        base_model,
        TargetAwareEncoderConfig(**target_aware_defaults),
    )
    return model


def build_batch(model, include_aux: bool = False):
    from dynatar_qwen.utils import qwen_audio_token_length, reconcile_three_lengths

    torch.manual_seed(1)
    batch_size = 2
    num_mel_bins = model.config.thinker_config.audio_config.num_mel_bins
    audio_token_id = model.config.thinker_config.audio_token_id

    segment_feature_lengths = torch.tensor(
        [
            [64, 12, 80],
            [56, 12, 72],
        ],
        dtype=torch.long,
    )
    total_feature_lengths = segment_feature_lengths.sum(dim=1)
    max_feature_length = int(total_feature_lengths.max().item())
    input_features = torch.randn(batch_size, num_mel_bins, max_feature_length)
    feature_attention_mask = torch.zeros(batch_size, max_feature_length, dtype=torch.long)
    for index, total_len in enumerate(total_feature_lengths.tolist()):
        feature_attention_mask[index, :total_len] = 1

    segment_token_lengths = []
    mix_token_lengths = []
    total_token_lengths = qwen_audio_token_length(total_feature_lengths)
    for seg_feat, total_tok in zip(segment_feature_lengths, total_token_lengths.tolist()):
        seg_tok = reconcile_three_lengths(int(total_tok), qwen_audio_token_length(seg_feat).tolist())
        segment_token_lengths.append(seg_tok)
        mix_token_lengths.append(int(seg_tok[-1].item()))
    segment_token_lengths = torch.stack(segment_token_lengths, dim=0)

    suffixes = [
        torch.tensor([21, 22, 23, 2], dtype=torch.long),
        torch.tensor([24, 25, 26, 2], dtype=torch.long),
    ]
    prefix = torch.tensor([11, 12], dtype=torch.long)
    seq_len = max(prefix.numel() + mix + suffix.numel() for mix, suffix in zip(mix_token_lengths, suffixes))
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    for index, (mix_token_len, suffix) in enumerate(zip(mix_token_lengths, suffixes)):
        audio_tokens = torch.full((mix_token_len,), audio_token_id, dtype=torch.long)
        sequence = torch.cat([prefix, audio_tokens, suffix], dim=0)
        input_ids[index, : sequence.numel()] = sequence
        attention_mask[index, : sequence.numel()] = 1
        target_start = prefix.numel() + mix_token_len
        labels[index, target_start : sequence.numel()] = sequence[target_start:]

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "input_features": input_features,
        "feature_attention_mask": feature_attention_mask,
        "segment_feature_lengths": segment_feature_lengths,
        "segment_token_lengths": segment_token_lengths,
        "audio_feature_lengths": segment_token_lengths[:, -1],
        "labels": labels,
    }
    if include_aux:
        max_mix_len = int(segment_token_lengths[:, -1].max().item())
        overlap_labels = torch.full((batch_size, max_mix_len), -100, dtype=torch.long)
        for index, mix_len in enumerate(segment_token_lengths[:, -1].tolist()):
            mix_len = int(mix_len)
            pattern = torch.tensor([1, 2, 3, 0], dtype=torch.long)
            repeat = (mix_len + pattern.numel() - 1) // pattern.numel()
            overlap_labels[index, :mix_len] = pattern.repeat(repeat)[:mix_len]

        target_feature_lengths = torch.tensor([72, 60], dtype=torch.long)
        max_target_feature_len = int(target_feature_lengths.max().item())
        target_input_features = torch.randn(batch_size, num_mel_bins, max_target_feature_len)
        target_feature_attention_mask = torch.zeros(batch_size, max_target_feature_len, dtype=torch.long)
        for index, total_len in enumerate(target_feature_lengths.tolist()):
            target_feature_attention_mask[index, :total_len] = 1

        target_wave_lengths = torch.tensor([8000, 6400], dtype=torch.long)
        max_target_wave_len = int(target_wave_lengths.max().item())
        target_waveforms = torch.randn(batch_size, max_target_wave_len)
        batch.update(
            {
                "overlap_labels": overlap_labels,
                "overlap_label_resample_count": torch.tensor(0, dtype=torch.long),
                "target_waveforms": target_waveforms,
                "target_wave_lengths": target_wave_lengths,
                "target_input_features": target_input_features,
                "target_feature_attention_mask": target_feature_attention_mask,
            }
        )
    return batch


def batch_to_samples(batch):
    samples = []
    batch_size = batch["input_ids"].shape[0]
    for index in range(batch_size):
        sample = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                if value.ndim == 0:
                    sample[key] = value.clone()
                else:
                    sample[key] = value[index].clone()
            else:
                sample[key] = value
        samples.append(sample)
    return samples


def aux_enabled_ts_config_kwargs(**overrides):
    base = {
        "enable_overlap_head": True,
        "enable_target_consistency": True,
        "enable_router_supervision": True,
        "overlap_loss_weight": 0.1,
        "target_consistency_weight": 0.05,
        "router_loss_weight": 0.02,
    }
    base.update(overrides)
    return base


def write_test_wav(path: Path, seconds: float, sr: int = 16000, freq: float = 220.0):
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    wav = 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(path, wav, sr)
