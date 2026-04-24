from __future__ import annotations

from argparse import Namespace
import os
import json
import sys
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import torch
from transformers import TrainingArguments
import yaml

TSASR_ROOT = Path(__file__).resolve().parents[1]
WESEP_ROOT = TSASR_ROOT.parents[3]
QWEN_ROOT = Path("/scratch/project_465002316/junyi/tse/ts_asr/Qwen3-ASR")
for path in (str(TSASR_ROOT), str(WESEP_ROOT), str(QWEN_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dynatar_qwen.data import TSASRShardDataset
from dynatar_qwen.aux_lmdb import LMDBWriter, decode_aux_payload_bytes, encode_aux_payload, open_readonly_lmdb
from infer_ts_qwen3_asr import (
    extract_generated_text,
    first_supervised_index,
    resolve_checkpoint_path,
    resolve_ts_config,
)
from train_ts_qwen3_asr import (
    CastFloatInputsTrainer,
    copy_required_hf_files_for_qwen_asr,
    patch_outer_forward,
)
from tests.support import (
    ROOT,
    TensorDictDataset,
    aux_enabled_ts_config_kwargs,
    batch_to_samples,
    build_batch,
    build_tiny_model,
    collate_tensor_dicts,
    lazy_model_imports,
    pythonpath_env,
    write_test_wav,
)


def test_shape():
    model = build_tiny_model()
    batch = build_batch(model)
    audio_out = model.thinker.audio_tower(
        batch["input_features"][0],
        feature_lens=batch["feature_attention_mask"][0].sum().unsqueeze(0),
        segment_feature_lengths=batch["segment_feature_lengths"][0].unsqueeze(0),
        segment_token_lengths=batch["segment_token_lengths"][0].unsqueeze(0),
    )
    assert audio_out.last_hidden_state.shape[0] == batch["segment_token_lengths"][0, -1].item()
    assert audio_out.last_hidden_state.shape[1] == model.config.thinker_config.text_config.hidden_size
    debug = model.thinker.audio_tower._last_debug
    assert debug["anchor_shape"] == (1, model.config.thinker_config.audio_config.d_model)
    assert debug["prototype_shape"] == (
        1,
        model.thinker.audio_tower.ts_config.num_prototypes,
        model.config.thinker_config.audio_config.d_model,
    )
    assert audio_out.mix_summary is None
    assert model.thinker.audio_tower.summary_norm is None
    assert model.thinker.audio_tower.mix_summary_attn is None
    assert model.thinker.audio_tower.mix_summary_proj is None


def test_target_consistency_shape():
    model = build_tiny_model(aux_enabled_ts_config_kwargs())
    batch = build_batch(model, include_aux=True)
    audio_out = model.thinker.audio_tower(
        batch["input_features"][0],
        feature_lens=batch["feature_attention_mask"][0].sum().unsqueeze(0),
        segment_feature_lengths=batch["segment_feature_lengths"][0].unsqueeze(0),
        segment_token_lengths=batch["segment_token_lengths"][0].unsqueeze(0),
    )
    assert audio_out.mix_summary is not None
    assert audio_out.mix_summary.shape[-1] == model.thinker.audio_tower.memory_dim


def test_forward():
    model = build_tiny_model()
    batch = build_batch(model)
    outputs = model.thinker(**batch)
    assert torch.isfinite(outputs.loss)
    assert outputs.logits.shape[:2] == batch["input_ids"].shape
    assert outputs.logits.shape[-1] == model.config.thinker_config.text_config.vocab_size


def test_backward():
    model = build_tiny_model()
    batch = build_batch(model)
    outputs = model.thinker(**batch)
    outputs.loss.backward()
    assert model.thinker.audio_tower.mhfa_backend.weights_k.grad is not None
    assert model.thinker.audio_tower.prototype_projector.assign.weight.grad is not None
    assert model.thinker.audio_tower.refinement_blocks[0].router.router.weight.grad is not None


def test_mask_correctness():
    (_, _, RoleConstrainedMaskBuilder, *_) = lazy_model_imports()
    builder = RoleConstrainedMaskBuilder()
    mask = builder(seq_len=12, enroll_len=3, silence_len=2, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.all(mask[..., :5, :5] == 0)
    assert torch.all(mask[..., 5:, 5:] == 0)
    assert torch.all(mask[..., :5, 5:] < -1e20)
    assert torch.all(mask[..., 5:, :5] < -1e20)


def test_smoke():
    torch.manual_seed(7)
    model = build_tiny_model()
    batch = build_batch(model)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        outputs = model.thinker(**batch)
        outputs.loss.backward()
        optimizer.step()
        losses.append(float(outputs.loss.detach().cpu()))
    assert losses[-1] <= losses[0], f"Smoke loss did not decrease: {losses}"


def test_lite_forward_backward():
    model = build_tiny_model(
        {
            "num_role_layers": 5,
            "num_prototypes": 4,
            "mhfa_heads": 2,
            "mhfa_compression_dim": 8,
            "router_hidden_multiplier": 1,
            "refinement_variant": "lite_unshared",
            "shared_refinement": False,
            "memory_dim": 16,
            "expert_rank": 8,
            "num_experts": 3,
            "refinement_layer_strategy": "post_role_all",
        }
    )
    batch = build_batch(model)
    outputs = model.thinker(**batch)
    assert torch.isfinite(outputs.loss)
    outputs.loss.backward()
    assert model.thinker.audio_tower.mix_memory_proj.weight.grad is not None
    assert model.thinker.audio_tower.memory_to_hidden.weight.grad is not None
    assert model.thinker.audio_tower.refinement_blocks[0].router.router.weight.grad is not None


def test_auxiliary_losses_backward():
    model = build_tiny_model(aux_enabled_ts_config_kwargs())
    batch = build_batch(model, include_aux=True)
    outputs = model.thinker(**batch)
    assert torch.isfinite(outputs.loss)
    outputs.loss.backward()
    aux_logs = model.thinker._last_aux_logs
    assert "tsasr/overlap_loss" in aux_logs
    assert "tsasr/target_consistency_loss" in aux_logs
    assert "tsasr/router_loss" in aux_logs
    assert model.thinker.audio_tower.overlap_head.weight.grad is not None
    assert model.thinker.audio_tower.mix_summary_proj.weight.grad is not None
    assert model.thinker.audio_tower.refinement_blocks[0].router.router.weight.grad is not None


def test_target_consistency_disabled_omits_summary_parameters():
    model = build_tiny_model()
    parameter_names = {name for name, _ in model.named_parameters()}
    assert "thinker.audio_tower.summary_norm.weight" not in parameter_names
    assert "thinker.audio_tower.summary_norm.bias" not in parameter_names
    assert "thinker.audio_tower.mix_summary_attn.weight" not in parameter_names
    assert "thinker.audio_tower.mix_summary_attn.bias" not in parameter_names
    assert "thinker.audio_tower.mix_summary_proj.weight" not in parameter_names
    assert "thinker.audio_tower.mix_summary_proj.bias" not in parameter_names


def test_trainer_aux_stage7_smoke():
    model = build_tiny_model(aux_enabled_ts_config_kwargs())
    patch_outer_forward(model)
    samples = batch_to_samples(build_batch(model, include_aux=True))
    dataset = TensorDictDataset(samples)

    with tempfile.TemporaryDirectory() as tmpdir:
        training_args = TrainingArguments(
            output_dir=tmpdir,
            overwrite_output_dir=True,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=1,
            save_steps=1,
            logging_steps=1,
            eval_strategy="no",
            remove_unused_columns=False,
            dataloader_num_workers=0,
            report_to=[],
            use_cpu=True,
            disable_tqdm=True,
            save_safetensors=False,
        )
        trainer = CastFloatInputsTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collate_tensor_dicts,
        )
        trainer.train()

        assert trainer.state.global_step == 1
        checkpoint_dir = Path(tmpdir) / "checkpoint-1"
        assert checkpoint_dir.is_dir()

        log_history = trainer.state.log_history
        assert any("tsasr/overlap_loss" in record for record in log_history)
        assert any("tsasr/router_loss" in record for record in log_history)

def test_asr_shard_dataset():
    script_path = ROOT / "local" / "make_shard_list_asr.py"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        mix_dir = tmp / "mix"
        clean_s1_dir = tmp / "s1"
        clean_s2_dir = tmp / "s2"
        enroll_dir = tmp / "enroll"
        mix_dir.mkdir()
        clean_s1_dir.mkdir()
        clean_s2_dir.mkdir()
        enroll_dir.mkdir()

        mix1 = mix_dir / "100-0001_200-0001.wav"
        mix2 = mix_dir / "100-0002_200-0002.wav"
        s1_mix1 = clean_s1_dir / "100-0001_200-0001.wav"
        s1_mix2 = clean_s1_dir / "100-0002_200-0002.wav"
        s2_mix1 = clean_s2_dir / "100-0001_200-0001.wav"
        s2_mix2 = clean_s2_dir / "100-0002_200-0002.wav"
        write_test_wav(mix1, 0.5, freq=220.0)
        write_test_wav(mix2, 0.5, freq=330.0)
        write_test_wav(s1_mix1, 0.5, freq=221.0)
        write_test_wav(s1_mix2, 0.5, freq=331.0)
        write_test_wav(s2_mix1, 0.5, freq=222.0)
        write_test_wav(s2_mix2, 0.5, freq=332.0)

        enroll_a = enroll_dir / "100_a.wav"
        enroll_b = enroll_dir / "100_b.wav"
        enroll_c = enroll_dir / "200_a.wav"
        enroll_d = enroll_dir / "200_b.wav"
        write_test_wav(enroll_a, 0.2, freq=440.0)
        write_test_wav(enroll_b, 0.2, freq=450.0)
        write_test_wav(enroll_c, 0.2, freq=550.0)
        write_test_wav(enroll_d, 0.2, freq=560.0)

        wav_scp = tmp / "wav.scp"
        wav_scp.write_text(
            "\n".join(
                [
                    f"100-0001_200-0001 {mix1} {s1_mix1} {s2_mix1}",
                    f"100-0002_200-0002 {mix2} {s1_mix2} {s2_mix2}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        utt2spk = tmp / "utt2spk"
        utt2spk.write_text(
            "\n".join(
                [
                    "100-0001_200-0001 100 200",
                    "100-0002_200-0002 100 200",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        spk1_text = tmp / "spk1_text"
        spk1_text.write_text(
            "\n".join(
                [
                    "100-0001_200-0001 hello speaker one",
                    "100-0002_200-0002 hello speaker one again",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        spk2_text = tmp / "spk2_text"
        spk2_text.write_text(
            "\n".join(
                [
                    "100-0001_200-0001 target two first",
                    "100-0002_200-0002 target two second",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        shards_dir = tmp / "shards"
        shard_list = tmp / "shard.list"
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--num_utts_per_shard",
                "2",
                "--num_threads",
                "1",
                "--prefix",
                "asr",
                "--spk1_text",
                str(spk1_text),
                "--spk2_text",
                str(spk2_text),
                str(wav_scp),
                str(utt2spk),
                str(shards_dir),
                str(shard_list),
            ],
            check=True,
            env=pythonpath_env(),
        )

        spk2enroll = tmp / "spk2enroll.json"
        spk2enroll.write_text(
            json.dumps(
                {
                    "100": [["100-0001", str(enroll_a)], ["100-0002", str(enroll_b)]],
                    "200": [["200-0001", str(enroll_c)], ["200-0002", str(enroll_d)]],
                }
            ),
            encoding="utf-8",
        )
        train_dataset = TSASRShardDataset(
            str(shard_list),
            split="train",
            enroll_crop_seconds=1.0,
            train_spk2utt=str(spk2enroll),
        )
        assert len(train_dataset) == 4
        assert len(train_dataset.lengths) == len(train_dataset)
        assert all(length > 0 for length in train_dataset.lengths)
        train_item = train_dataset[0]
        assert train_item["target_role"] in {"spk1", "spk2"}
        assert train_item["target_spk"] in {"100", "200"}
        assert train_item["mix_audio"].ndim == 1
        assert train_item["target_audio"] is not None
        assert train_item["enroll_wav"] != ""
        assert train_item["enroll_audio"].shape[0] == 16000

        aux_labels_lmdb = tmp / "aux_labels.lmdb"
        aux_manifest = tmp / "aux_label_manifest.jsonl"
        aux_records = []
        with LMDBWriter(aux_labels_lmdb, overwrite=True) as lmdb_writer:
            for mix_id in ("100-0001_200-0001", "100-0002_200-0002"):
                for role in ("spk1", "spk2"):
                    lmdb_key = f"{mix_id}__{role}"
                    lmdb_writer.put(
                        lmdb_key,
                        encode_aux_payload(
                            {
                                "router_label": np.asarray([1, 2, 3, 0, 1], dtype=np.int8),
                                "target_active": np.asarray([1, 1, 0, 0, 1], dtype=np.uint8),
                            }
                        ),
                    )
                    aux_records.append(
                        {
                            "storage": "lmdb",
                            "key": lmdb_key,
                            "mix_id": mix_id,
                            "target_role": role,
                            "lmdb_path": aux_labels_lmdb.name,
                            "lmdb_key": lmdb_key,
                        }
                    )
        aux_manifest.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in aux_records) + "\n",
            encoding="utf-8",
        )
        relative_shard_list = tmp / "relative_shard.list"
        relative_shard_list.write_text(
            "\n".join(str(path.relative_to(tmp)) for path in sorted(shards_dir.glob("*.tar"))) + "\n",
            encoding="utf-8",
        )
        train_dataset_with_aux = TSASRShardDataset(
            str(relative_shard_list),
            split="train",
            enroll_crop_seconds=1.0,
            train_spk2utt=str(spk2enroll),
            aux_label_manifest=str(aux_manifest),
        )
        aux_item = train_dataset_with_aux[0]
        assert "overlap_labels" in aux_item
        assert aux_item["overlap_labels"].ndim == 1
        assert aux_item["aux_label_lmdb_key"].endswith(aux_item["target_role"])
        assert Path(aux_item["aux_label_lmdb_path"]).exists()

        spk1_enroll = tmp / "spk1.enroll"
        spk2_enroll = tmp / "spk2.enroll"
        single_wav_scp = tmp / "single.wav.scp"
        single_wav_scp.write_text(
            "\n".join(
                [
                    f"s1/enroll100.wav {enroll_a}",
                    f"s2/enroll200.wav {enroll_c}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        spk1_enroll.write_text(
            "\n".join(
                [
                    "100-0001_200-0001 s1/enroll100.wav",
                    "100-0002_200-0002 s1/enroll100.wav",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        spk2_enroll.write_text(
            "\n".join(
                [
                    "100-0001_200-0001 s2/enroll200.wav",
                    "100-0002_200-0002 s2/enroll200.wav",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        eval_dataset = TSASRShardDataset(
            str(shard_list),
            split="eval",
            enroll_crop_seconds=1.0,
            eval_spk1_enroll=str(spk1_enroll),
            eval_spk2_enroll=str(spk2_enroll),
            eval_spk2utt=str(single_wav_scp),
        )
        assert len(eval_dataset) == 4
        eval_item = eval_dataset[1]
        assert eval_item["target_text"] != ""
        assert Path(eval_item["enroll_wav"]).exists()
        assert eval_item["mix_audio"].dtype == np.float32
        assert eval_item["target_audio"] is not None
        assert eval_item["enroll_audio"].shape[0] == 16000


def test_merge_aux_label_manifests_builds_merged_lmdb():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        split_a = tmp / "split_a"
        split_b = tmp / "split_b"
        merge_dir = tmp / "train-merge"
        split_a.mkdir()
        split_b.mkdir()
        merge_dir.mkdir()

        source_specs = [
            (split_a, "mix-a", "spk1", np.asarray([1, 0, 2], dtype=np.int8)),
            (split_b, "mix-b", "spk2", np.asarray([0, 3, 1], dtype=np.int8)),
        ]
        manifest_paths = []
        for split_dir, mix_id, role, router_label in source_specs:
            lmdb_path = split_dir / "aux_labels.lmdb"
            manifest_path = split_dir / "aux_label_manifest.jsonl"
            lmdb_key = f"{mix_id}__{role}"
            with LMDBWriter(lmdb_path, overwrite=True) as lmdb_writer:
                lmdb_writer.put(
                    lmdb_key,
                    encode_aux_payload(
                        {
                            "router_label": router_label,
                            "target_active": np.asarray([1, 0, 1], dtype=np.uint8),
                        }
                    ),
                )
            manifest_path.write_text(
                json.dumps(
                    {
                        "storage": "lmdb",
                        "key": lmdb_key,
                        "mix_id": mix_id,
                        "target_role": role,
                        "lmdb_path": "aux_labels.lmdb",
                        "lmdb_key": lmdb_key,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_paths.append(str(manifest_path))

        script_path = ROOT / "local" / "merge_aux_label_manifests.py"
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--output_dir",
                str(merge_dir),
                "--num_workers",
                "2",
                *manifest_paths,
            ],
            check=True,
            cwd=ROOT,
            env=pythonpath_env(),
        )

        merged_manifest = merge_dir / "aux_label_manifest.jsonl"
        merged_lmdb = merge_dir / "aux_labels.lmdb"
        merged_lines = [json.loads(line) for line in merged_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(merged_lines) == 2
        assert all(line["storage"] == "lmdb" for line in merged_lines)
        assert all(line["lmdb_path"] == "aux_labels.lmdb" for line in merged_lines)

        env = open_readonly_lmdb(merged_lmdb)
        try:
            with env.begin(write=False) as txn:
                for line in merged_lines:
                    payload = txn.get(line["lmdb_key"].encode("utf-8"))
                    assert payload is not None
                    decoded = decode_aux_payload_bytes(bytes(payload))
                    assert "router_label" in decoded
        finally:
            env.close()


def test_copy_required_hf_files_preserves_checkpoint_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        src = tmp / "src"
        dst = tmp / "dst"
        src.mkdir()
        dst.mkdir()

        (src / "config.json").write_text('{"tsasr":{"enable_overlap_head":true}}', encoding="utf-8")
        (src / "tokenizer_config.json").write_text('{"tokenizer":"source"}', encoding="utf-8")
        (dst / "config.json").write_text('{"tsasr":{"enable_overlap_head":false}}', encoding="utf-8")

        copy_required_hf_files_for_qwen_asr(str(src), str(dst))

        assert (dst / "config.json").read_text(encoding="utf-8") == '{"tsasr":{"enable_overlap_head":false}}'
        assert (dst / "tokenizer_config.json").read_text(encoding="utf-8") == '{"tokenizer":"source"}'


def _build_infer_namespace(**overrides):
    base = {
        "role_layers": 5,
        "num_prototypes": 8,
        "mhfa_heads": 4,
        "mhfa_compression_dim": 128,
        "router_hidden_multiplier": 2,
        "refinement_variant": "full",
        "shared_refinement": 0,
        "memory_dim": 256,
        "expert_rank": 64,
        "num_experts": 3,
        "refinement_layer_strategy": "post_role_all",
        "enable_overlap_head": 0,
        "overlap_num_classes": 4,
        "overlap_loss_weight": 0.10,
        "enable_target_consistency": 0,
        "target_consistency_weight": 0.05,
        "target_consistency_mode": "hybrid",
        "target_consistency_temperature": 0.07,
        "target_consistency_detach_target": 1,
        "enable_router_supervision": 0,
        "router_loss_weight": 0.02,
    }
    base.update(overrides)
    return Namespace(**base)


def test_resolve_ts_config_prefers_checkpoint_then_run_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        run_dir = tmp / "run"
        checkpoint_dir = run_dir / "checkpoint-7"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_path = checkpoint_dir / "model.safetensors"
        checkpoint_path.write_bytes(b"")

        checkpoint_ts = {
            "enable_overlap_head": True,
            "overlap_num_classes": 4,
            "enable_target_consistency": True,
            "enable_router_supervision": True,
            "memory_dim": 192,
        }
        (checkpoint_dir / "config.json").write_text(
            json.dumps({"tsasr": checkpoint_ts}, ensure_ascii=False),
            encoding="utf-8",
        )

        cli_args = _build_infer_namespace(memory_dim=64, enable_overlap_head=0)
        ts_config, strict_load, source = resolve_ts_config(cli_args, checkpoint_path)
        assert strict_load is True
        assert source.endswith("config.json")
        assert ts_config.enable_overlap_head is True
        assert ts_config.enable_target_consistency is True
        assert ts_config.memory_dim == 192

        (checkpoint_dir / "config.json").unlink()
        run_config = {
            "enable_overlap_head": True,
            "enable_target_consistency": True,
            "enable_router_supervision": True,
            "memory_dim": 160,
            "overlap_loss_weight": 0.25,
        }
        (run_dir / "run_config.yaml").write_text(
            yaml.safe_dump(run_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        ts_config, strict_load, source = resolve_ts_config(cli_args, checkpoint_path)
        assert strict_load is False
        assert source.endswith("run_config.yaml")
        assert ts_config.enable_overlap_head is True
        assert ts_config.enable_target_consistency is True
        assert ts_config.memory_dim == 160
        assert abs(ts_config.overlap_loss_weight - 0.25) < 1.0e-8

        (run_dir / "run_config.yaml").unlink()
        ts_config, strict_load, source = resolve_ts_config(cli_args, checkpoint_path)
        assert strict_load is False
        assert source == "cli"
        assert ts_config.enable_overlap_head is False
        assert ts_config.memory_dim == 64


def test_resolve_checkpoint_path_latest():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "checkpoint-2").mkdir()
        (tmp / "checkpoint-2" / "pytorch_model.bin").write_bytes(b"two")
        (tmp / "checkpoint-11").mkdir()
        (tmp / "checkpoint-11" / "model.safetensors").write_bytes(b"eleven")

        resolved = resolve_checkpoint_path(str(tmp))
        assert resolved is not None
        assert resolved.name == "model.safetensors"
        assert resolved.parent.name == "checkpoint-11"


def test_infer_generation_stage8_smoke():
    torch.manual_seed(11)
    model = build_tiny_model(
        {
            "enable_overlap_head": True,
            "enable_target_consistency": True,
            "enable_router_supervision": True,
        }
    ).eval()
    batch = build_batch(model, include_aux=True)

    prefix_len = first_supervised_index(batch["labels"][0])
    assert prefix_len == 2 + int(batch["audio_feature_lengths"][0].item())

    generation_inputs = {
        "input_ids": batch["input_ids"][0:1, :prefix_len],
        "attention_mask": batch["attention_mask"][0:1, :prefix_len],
        "input_features": batch["input_features"][0:1],
        "feature_attention_mask": batch["feature_attention_mask"][0:1],
        "segment_feature_lengths": batch["segment_feature_lengths"][0:1],
        "segment_token_lengths": batch["segment_token_lengths"][0:1],
        "audio_feature_lengths": batch["audio_feature_lengths"][0:1],
        "max_new_tokens": 4,
        "do_sample": False,
    }
    with torch.no_grad():
        generated = model.thinker.generate(**generation_inputs)

    assert generated.shape[0] == 1
    assert generated.shape[1] >= prefix_len
    assert extract_generated_text("<asr_text> HELLO WORLD <|im_end|>") == "HELLO WORLD"


def test_score_stage9_cli_smoke():
    script_path = ROOT / "score_ts_qwen3_asr.py"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        results_jsonl = tmp / "results.jsonl"
        output_dir = tmp / "score_out"
        records = [
            {
                "key": "utt1",
                "reference": "hello world",
                "hypothesis": "<asr_text> hello world <|im_end|>",
            },
            {
                "key": "utt2",
                "reference": "target overlap",
                "hypothesis": "target",
            },
        ]
        results_jsonl.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
            encoding="utf-8",
        )

        subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--results_jsonl",
                str(results_jsonl),
                "--output_dir",
                str(output_dir),
            ],
            check=True,
            env=pythonpath_env(),
        )

        summary = json.loads((output_dir / "wer_summary.json").read_text(encoding="utf-8"))
        scored_records = [
            json.loads(line)
            for line in (output_dir / "scored_results.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summary_txt = (output_dir / "wer_summary.txt").read_text(encoding="utf-8")

        assert summary["num_samples"] == 2
        assert summary["total_ref_words"] == 4
        assert summary["total_word_edits"] == 1
        assert abs(summary["wer"] - 0.25) < 1.0e-8
        assert scored_records[0]["normalized_hypothesis"] == "HELLO WORLD"
        assert "wer\t0.250000" in summary_txt


TEST_CASES = [
    ("shape test", test_shape),
    ("target consistency shape test", test_target_consistency_shape),
    ("forward test", test_forward),
    ("backward test", test_backward),
    ("mask correctness test", test_mask_correctness),
    ("smoke test", test_smoke),
    ("auxiliary losses test", test_auxiliary_losses_backward),
    ("target consistency disabled parameter test", test_target_consistency_disabled_omits_summary_parameters),
    ("trainer aux stage7 smoke test", test_trainer_aux_stage7_smoke),
    ("asr shard dataset test", test_asr_shard_dataset),
    ("copy_required_hf_files test", test_copy_required_hf_files_preserves_checkpoint_config),
    ("resolve_ts_config test", test_resolve_ts_config_prefers_checkpoint_then_run_config),
    ("resolve_checkpoint_path test", test_resolve_checkpoint_path_latest),
    ("infer generation stage8 smoke test", test_infer_generation_stage8_smoke),
    ("score stage9 cli smoke test", test_score_stage9_cli_smoke),
]


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for label, fn in TEST_CASES:
        fn()
        print(f"{label} passed")


if __name__ == "__main__":
    main()
