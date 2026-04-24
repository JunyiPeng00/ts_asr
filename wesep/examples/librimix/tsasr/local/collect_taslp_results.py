from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


def parse_args():
    parser = argparse.ArgumentParser("Collect TASLP TS-ASR experiment summaries")
    parser.add_argument(
        "--config_dir",
        type=str,
        default="confs/taslp",
        help="Directory containing TASLP YAML configs",
    )
    parser.add_argument(
        "--exp_root",
        type=str,
        default="exp",
        help="Experiment root used by the TASLP configs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="paper/taslp",
        help="Directory to write aggregated CSV/Markdown summaries",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_checkpoint_step(exp_dir: Path) -> int | None:
    checkpoints = []
    for child in exp_dir.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            try:
                checkpoints.append(int(child.name.split("-", 1)[1]))
            except ValueError:
                continue
    if not checkpoints:
        return None
    return max(checkpoints)


def find_latest_infer_dir(exp_dir: Path) -> Path | None:
    infer_dirs = [child for child in exp_dir.iterdir() if child.is_dir() and child.name.startswith("infer_")]
    if not infer_dirs:
        return None
    infer_dirs.sort(key=lambda item: item.name)
    return infer_dirs[-1]


def format_float(value: Any) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def paper_group_from_name(config_name: str) -> str:
    if "trainmerge" in config_name:
        return "main_ablation"
    if "train100" in config_name:
        return "benchmark_2spk"
    if "train3mix_custom" in config_name:
        return "benchmark_3spk"
    return "other"


def method_tag(row: dict[str, Any]) -> str:
    if row["enable_router_supervision"] == "1":
        return "b3_full"
    if row["enable_target_consistency"] == "1":
        return "b2_overlap_tc"
    if row["enable_overlap_head"] == "1":
        return "b1_overlap"
    return "b0_baseline"


def build_rows(config_dir: Path, exp_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_path in sorted(config_dir.glob("*.yaml")):
        config = load_yaml(config_path)
        exp_dir_value = str(config.get("exp_dir", "")).strip()
        exp_dir = exp_root / exp_dir_value.replace("exp/", "", 1) if exp_dir_value.startswith("exp/") else exp_root / exp_dir_value

        run_config_path = exp_dir / "run_config.yaml"
        run_config = load_yaml(run_config_path) if run_config_path.is_file() else {}

        infer_dir = find_latest_infer_dir(exp_dir) if exp_dir.is_dir() else None
        decode_summary = load_json(infer_dir / "decode_summary.json") if infer_dir and (infer_dir / "decode_summary.json").is_file() else {}
        wer_summary = load_json(infer_dir / "wer_summary.json") if infer_dir and (infer_dir / "wer_summary.json").is_file() else {}

        latest_step = latest_checkpoint_step(exp_dir) if exp_dir.is_dir() else None
        results_jsonl = infer_dir / "results.jsonl" if infer_dir else None

        row = {
            "config_name": config_path.stem,
            "paper_group": paper_group_from_name(config_path.stem),
            "method_tag": "",
            "train_split": str(config.get("train_split", "")),
            "eval_split": str(config.get("eval_split", "")),
            "test_split": str(config.get("test_split", "")),
            "exp_dir": str(exp_dir),
            "config_path": str(config_path),
            "enable_overlap_head": str(int(bool(config.get("enable_overlap_head", 0)))),
            "enable_target_consistency": str(int(bool(config.get("enable_target_consistency", 0)))),
            "enable_router_supervision": str(int(bool(config.get("enable_router_supervision", 0)))),
            "overlap_loss_weight": config.get("overlap_loss_weight", ""),
            "target_consistency_weight": config.get("target_consistency_weight", ""),
            "router_loss_weight": config.get("router_loss_weight", ""),
            "train_aux_label_manifest": str(config.get("train_aux_label_manifest", "")),
            "run_config_exists": str(int(run_config_path.is_file())),
            "latest_checkpoint_step": "" if latest_step is None else str(latest_step),
            "infer_dir": "" if infer_dir is None else str(infer_dir),
            "decode_summary_exists": "" if infer_dir is None else str(int(bool(decode_summary))),
            "results_jsonl_exists": "" if results_jsonl is None else str(int(results_jsonl.is_file())),
            "wer_summary_exists": "" if infer_dir is None else str(int(bool(wer_summary))),
            "num_samples": wer_summary.get("num_samples", decode_summary.get("num_samples", "")),
            "wer": wer_summary.get("wer", ""),
            "cer": wer_summary.get("cer", ""),
            "total_ref_words": wer_summary.get("total_ref_words", ""),
            "total_ref_chars": wer_summary.get("total_ref_chars", ""),
            "best_checkpoint_used": decode_summary.get("checkpoint", ""),
            "high_overlap_wer": "",
            "high_overlap_cer": "",
            "overlap_acc": "",
            "overlap_macro_f1": "",
            "router_label_agreement": "",
            "router_target_ratio": "",
            "router_overlap_ratio": "",
            "router_nontarget_ratio": "",
            "target_cosine": "",
            "notes": "",
        }
        row["method_tag"] = method_tag(row)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(rows: list[dict[str, Any]]) -> str:
    sections = [
        ("Main Ablation", [row for row in rows if row["paper_group"] == "main_ablation"]),
        ("2-Speaker Benchmark", [row for row in rows if row["paper_group"] == "benchmark_2spk"]),
        ("3-Speaker Benchmark", [row for row in rows if row["paper_group"] == "benchmark_3spk"]),
    ]
    lines = [
        "# TASLP Result Snapshot",
        "",
        "Auto-generated from `confs/taslp/*.yaml` and available `exp/*/infer_*/wer_summary.json` files.",
        "",
    ]
    for title, group_rows in sections:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| config | method | split | ckpt | WER | CER | infer_dir |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | --- |")
        if not group_rows:
            lines.append("| - | - | - | - | - | - | - |")
        else:
            for row in group_rows:
                lines.append(
                    "| {config} | {method} | {split} | {ckpt} | {wer} | {cer} | {infer_dir} |".format(
                        config=row["config_name"],
                        method=row["method_tag"],
                        split=row["train_split"],
                        ckpt=row["latest_checkpoint_step"] or "-",
                        wer=format_float(row["wer"]) or "-",
                        cer=format_float(row["cer"]) or "-",
                        infer_dir=row["infer_dir"] or "-",
                    )
                )
        lines.append("")
    lines.extend(
        [
            "## Manual Fields To Fill Later",
            "",
            "- `high_overlap_wer` / `high_overlap_cer`",
            "- `overlap_acc` / `overlap_macro_f1`",
            "- `router_label_agreement`",
            "- `router_target_ratio` / `router_overlap_ratio` / `router_nontarget_ratio`",
            "- `target_cosine`",
            "- `notes`",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    tsasr_root = script_dir.parent

    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = tsasr_root / config_dir
    exp_root = Path(args.exp_root)
    if not exp_root.is_absolute():
        exp_root = tsasr_root / exp_root
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = tsasr_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(config_dir, exp_root)
    fieldnames = list(rows[0].keys()) if rows else []

    csv_path = output_dir / "results_master.csv"
    md_path = output_dir / "results_snapshot.md"
    if rows:
        write_csv(csv_path, rows, fieldnames)
    else:
        csv_path.write_text("", encoding="utf-8")
    md_path.write_text(build_markdown(rows), encoding="utf-8")

    print(f"[taslp] csv={csv_path}")
    print(f"[taslp] markdown={md_path}")
    print(f"[taslp] rows={len(rows)}")


if __name__ == "__main__":
    main()
