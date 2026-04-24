from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from transformers import GenerationConfig, TrainingArguments

ROOT = Path("/scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr")
WESEP_ROOT = Path("/scratch/project_465002316/junyi/tse/ts_asr/wesep")
QWEN_ROOT = Path("/scratch/project_465002316/junyi/tse/ts_asr/Qwen3-ASR")
for path in (str(ROOT), str(WESEP_ROOT), str(QWEN_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from qwen_asr import Qwen3ASRModel

from dynatar_qwen.data import DataCollatorForDynaTaRQwen, TSASRShardDataset
from dynatar_qwen.modeling_ts_qwen3_asr import TargetAwareEncoderConfig, upgrade_qwen3_asr_model
from train_ts_qwen3_asr import (
    CastFloatInputsTrainer,
    freeze_text_decoder,
    maybe_subset_dataset,
    patch_outer_forward,
)


DATA_ROOT = ROOT / "data" / "clean"
TRAIN_SHARDS = DATA_ROOT / "train-100" / "shard.list"
DEV_SHARDS = DATA_ROOT / "dev" / "shard.list"
TRAIN_SPK2UTT = DATA_ROOT / "train-100" / "spk2enroll.json"
DEV_SPK1_ENROLL = DATA_ROOT / "dev" / "spk1.enroll"
DEV_SPK2_ENROLL = DATA_ROOT / "dev" / "spk2.enroll"
DEV_SINGLE_WAV = DATA_ROOT / "dev" / "single.wav.scp"
OUTPUT_DIR = ROOT / "exp" / "full_chain_qwen_tsasr_smoke"
MODEL_PATH = os.environ.get("TSASR_MODEL_PATH", "Qwen/Qwen3-ASR-0.6B")


def main():
    torch.manual_seed(42)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    wrapper = Qwen3ASRModel.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        attn_implementation="eager",
        device_map=None,
    )
    model = upgrade_qwen3_asr_model(
        wrapper.model,
        TargetAwareEncoderConfig(
            num_role_layers=5,
            num_prototypes=8,
            mhfa_heads=4,
            mhfa_compression_dim=128,
            router_hidden_multiplier=2,
        ),
    )
    processor = wrapper.processor
    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model.thinker.config, "use_cache"):
        model.thinker.config.use_cache = False
    frozen = freeze_text_decoder(model)
    print(f"[freeze] frozen text decoder parameters: {frozen}")

    train_dataset = maybe_subset_dataset(
        TSASRShardDataset(
            str(TRAIN_SHARDS),
            split="train",
            sampling_rate=16000,
            language="English",
            prompt="",
            train_spk2utt=str(TRAIN_SPK2UTT),
        ),
        2,
    )
    eval_dataset = maybe_subset_dataset(
        TSASRShardDataset(
            str(DEV_SHARDS),
            split="eval",
            sampling_rate=16000,
            language="English",
            prompt="",
            eval_spk1_enroll=str(DEV_SPK1_ENROLL),
            eval_spk2_enroll=str(DEV_SPK2_ENROLL),
            eval_spk2utt=str(DEV_SINGLE_WAV),
        ),
        1,
    )
    collator = DataCollatorForDynaTaRQwen(
        processor=processor,
        sampling_rate=16000,
        silence_seconds=1.0,
        default_prompt="",
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=4e-6,
        warmup_steps=10000,
        lr_scheduler_type="cosine",
        max_steps=2,
        num_train_epochs=1,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=1,
        eval_strategy="steps",
        eval_steps=1,
        save_total_limit=2,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True,
        seed=42,
        data_seed=42,
    )

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
    )

    train_output = trainer.train()
    print(f"[train] global_step={trainer.state.global_step} loss={train_output.training_loss:.6f}")
    eval_metrics = trainer.evaluate()
    print(f"[eval] metrics={eval_metrics}")

    sample = eval_dataset[0]
    batch = collator([sample])
    cast_batch = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.to(model.device)
            if value.is_floating_point():
                value = value.to(dtype=model.dtype)
        cast_batch[key] = value
    prefix_length = int((batch["labels"][0] == -100).sum().item())
    generation_inputs = {
        "input_ids": cast_batch["input_ids"][:, :prefix_length],
        "attention_mask": cast_batch["attention_mask"][:, :prefix_length],
        "input_features": cast_batch["input_features"],
        "feature_attention_mask": cast_batch["feature_attention_mask"],
        "segment_feature_lengths": cast_batch["segment_feature_lengths"],
        "segment_token_lengths": cast_batch["segment_token_lengths"],
        "audio_feature_lengths": cast_batch["audio_feature_lengths"],
        "max_new_tokens": 48,
    }
    model.eval()
    with torch.no_grad():
        generated = model.thinker.generate(**generation_inputs)
    decoded = processor.tokenizer.batch_decode(generated, skip_special_tokens=False)
    print(f"[generate] sample_key={sample['key']}")
    print(f"[generate] target_text={sample['target_text']}")
    print(f"[generate] decoded={decoded[0]}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
