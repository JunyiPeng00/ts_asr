from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import soundfile as sf
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

ROOT = Path("/scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr")
QWEN_ROOT = Path("/scratch/project_465002316/junyi/tse/ts_asr/Qwen3-ASR")
for path in (str(ROOT), str(QWEN_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)
from score_ts_qwen3_asr import score_records

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)


def parse_args():
    parser = argparse.ArgumentParser("Evaluate original Qwen3-ASR on targetwav shards using direct transformers loading")
    parser.add_argument(
        "--shard_list",
        type=str,
        default="/scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr/data/clean/test/shard.list",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr/qwen_models/Qwen3-ASR-0.6B",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr/exp/oracle_original_qwen_test_targetwav_direct",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--language", type=str, default="English")
    return parser.parse_args()


def resolve_tar_path(shard_list: Path, tar_path: str) -> Path:
    candidate = Path(tar_path)
    if candidate.is_file():
        return candidate
    candidate = shard_list.parent / tar_path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Cannot resolve tar path {tar_path} from {shard_list}")


def iter_targetwav_samples(shard_list_path: str, max_samples: int = 0) -> Iterable[Dict[str, object]]:
    shard_list = Path(shard_list_path)
    emitted = 0
    with shard_list.open("r", encoding="utf-8") as handle:
        for line in handle:
            tar_path = resolve_tar_path(shard_list, line.strip())
            with tarfile.open(tar_path, "r") as tar:
                members = {member.name: member for member in tar.getmembers()}
                record_keys = sorted(name[:-10] for name in members if name.endswith(".targetwav"))
                for record_key in record_keys:
                    payload = tar.extractfile(members[f"{record_key}.targetwav"]).read()
                    wav, sr = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
                    if sr != 16000:
                        raise ValueError(f"Unexpected sample rate {sr} in {record_key}.targetwav")
                    if wav.ndim > 1:
                        wav = np.mean(wav, axis=1)
                    yield {
                        "key": record_key,
                        "mix_id": tar.extractfile(members[f"{record_key}.mixid"]).read().decode("utf-8").strip(),
                        "target_role": tar.extractfile(members[f"{record_key}.role"]).read().decode("utf-8").strip(),
                        "audio": np.asarray(wav, dtype=np.float32),
                        "reference": tar.extractfile(members[f"{record_key}.txt"]).read().decode("utf-8").strip(),
                    }
                    emitted += 1
                    if max_samples > 0 and emitted >= max_samples:
                        return


def chunk_list(items: List[Dict[str, object]], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def save_kaldi_text(path: Path, entries: List[Tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for key, text in entries:
            handle.write(f"{key} {text}\n")


def build_prompt(processor, language: str) -> str:
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return prompt + f"language {language}<asr_text>"


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = list(iter_targetwav_samples(args.shard_list, args.max_samples))
    if not samples:
        raise RuntimeError("No targetwav samples were found")

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    model = AutoModel.from_pretrained(args.model_path, dtype=dtype, device_map="cuda:0", attn_implementation="eager")
    processor = AutoProcessor.from_pretrained(args.model_path, fix_mistral_regex=True)
    prompt = build_prompt(processor, args.language)

    results = []
    ref_entries: List[Tuple[str, str]] = []
    hyp_entries: List[Tuple[str, str]] = []

    for batch in chunk_list(samples, args.batch_size):
        audios = [sample["audio"] for sample in batch]
        texts = [prompt] * len(batch)
        inputs = processor(text=texts, audio=audios, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        decoded = processor.batch_decode(
            outputs.sequences[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for sample, hyp in zip(batch, decoded):
            record = {
                "key": sample["key"],
                "mix_id": sample["mix_id"],
                "target_role": sample["target_role"],
                "reference": sample["reference"],
                "hypothesis": hyp,
                "language": args.language,
            }
            results.append(record)
            ref_entries.append((record["key"], record["reference"]))
            hyp_entries.append((record["key"], record["hypothesis"]))
        print(f"[decode] {len(results)}/{len(samples)}", flush=True)

    scored_results, summary = score_records(results)

    results_jsonl = output_dir / "results.jsonl"
    scored_jsonl = output_dir / "scored_results.jsonl"
    ref_text = output_dir / "ref.text"
    hyp_text = output_dir / "hyp.text"
    summary_json = output_dir / "wer_summary.json"
    summary_txt = output_dir / "wer_summary.txt"

    with results_jsonl.open("w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with scored_jsonl.open("w", encoding="utf-8") as handle:
        for record in scored_results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with summary_txt.open("w", encoding="utf-8") as handle:
        handle.write(f"num_samples\t{summary['num_samples']}\n")
        handle.write(f"total_ref_words\t{summary['total_ref_words']}\n")
        handle.write(f"total_word_edits\t{summary['total_word_edits']}\n")
        handle.write(f"wer\t{summary['wer']:.6f}\n")
        handle.write(f"total_ref_chars\t{summary['total_ref_chars']}\n")
        handle.write(f"total_char_edits\t{summary['total_char_edits']}\n")
        handle.write(f"cer\t{summary['cer']:.6f}\n")
    save_kaldi_text(ref_text, ref_entries)
    save_kaldi_text(hyp_text, hyp_entries)

    print(f"[done] output_dir={output_dir}")
    print(f"[done] num_samples={summary['num_samples']}")
    print(f"[done] WER={summary['wer']:.6f} CER={summary['cer']:.6f}")


if __name__ == "__main__":
    main()
