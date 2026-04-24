from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path("/scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr")
WESEP_ROOT = Path("/scratch/project_465002316/junyi/tse/ts_asr/wesep")
QWEN_ROOT = Path("/scratch/project_465002316/junyi/tse/ts_asr/Qwen3-ASR")
for path in (str(ROOT), str(WESEP_ROOT), str(QWEN_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dynatar_qwen.utils import dump_jsonl, load_jsonl


def parse_args():
    parser = argparse.ArgumentParser("Score DynaTaR-Qwen TS-ASR WER/CER")
    parser.add_argument("--results_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"language\s+[^\s<]+<asr_text>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("-", " ")
    text = text.upper()
    text = re.sub(r"[^A-Z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def levenshtein_distance(ref_tokens, hyp_tokens) -> int:
    if len(ref_tokens) < len(hyp_tokens):
        ref_tokens, hyp_tokens = hyp_tokens, ref_tokens
    previous = list(range(len(hyp_tokens) + 1))
    for i, ref_token in enumerate(ref_tokens, start=1):
        current = [i]
        for j, hyp_token in enumerate(hyp_tokens, start=1):
            substitution_cost = 0 if ref_token == hyp_token else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def score_records(records):
    total_word_edits = 0
    total_ref_words = 0
    total_char_edits = 0
    total_ref_chars = 0
    scored_records = []

    for record in records:
        normalized_ref = normalize_text(record["reference"])
        normalized_hyp = normalize_text(record["hypothesis"])

        ref_words = normalized_ref.split()
        hyp_words = normalized_hyp.split()
        word_edits = levenshtein_distance(ref_words, hyp_words)

        ref_chars = list(normalized_ref.replace(" ", ""))
        hyp_chars = list(normalized_hyp.replace(" ", ""))
        char_edits = levenshtein_distance(ref_chars, hyp_chars)

        ref_word_count = max(len(ref_words), 1)
        ref_char_count = max(len(ref_chars), 1)

        scored_record = dict(record)
        scored_record["normalized_reference"] = normalized_ref
        scored_record["normalized_hypothesis"] = normalized_hyp
        scored_record["word_edits"] = word_edits
        scored_record["char_edits"] = char_edits
        scored_record["word_count"] = len(ref_words)
        scored_record["char_count"] = len(ref_chars)
        scored_record["wer"] = word_edits / ref_word_count
        scored_record["cer"] = char_edits / ref_char_count
        scored_records.append(scored_record)

        total_word_edits += word_edits
        total_ref_words += len(ref_words)
        total_char_edits += char_edits
        total_ref_chars += len(ref_chars)

    summary = {
        "num_samples": len(records),
        "total_ref_words": total_ref_words,
        "total_word_edits": total_word_edits,
        "wer": total_word_edits / max(total_ref_words, 1),
        "total_ref_chars": total_ref_chars,
        "total_char_edits": total_char_edits,
        "cer": total_char_edits / max(total_ref_chars, 1),
    }
    return scored_records, summary


def write_summary_text(path: Path, summary: dict):
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"num_samples\t{summary['num_samples']}\n")
        handle.write(f"total_ref_words\t{summary['total_ref_words']}\n")
        handle.write(f"total_word_edits\t{summary['total_word_edits']}\n")
        handle.write(f"wer\t{summary['wer']:.6f}\n")
        handle.write(f"total_ref_chars\t{summary['total_ref_chars']}\n")
        handle.write(f"total_char_edits\t{summary['total_char_edits']}\n")
        handle.write(f"cer\t{summary['cer']:.6f}\n")


def main():
    args = parse_args()
    results_path = Path(args.results_jsonl)
    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(results_path)
    scored_records, summary = score_records(records)

    scored_jsonl = output_dir / "scored_results.jsonl"
    summary_json = output_dir / "wer_summary.json"
    summary_txt = output_dir / "wer_summary.txt"

    dump_jsonl(scored_jsonl, scored_records)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    write_summary_text(summary_txt, summary)

    print(f"[score] scored_jsonl={scored_jsonl}")
    print(f"[score] summary_json={summary_json}")
    print(f"[score] summary_txt={summary_txt}")
    print(f"[score] WER={summary['wer']:.6f} CER={summary['cer']:.6f}")


if __name__ == "__main__":
    main()
