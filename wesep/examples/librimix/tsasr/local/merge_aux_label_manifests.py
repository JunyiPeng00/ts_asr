from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TSASR_ROOT = Path(__file__).resolve().parents[1]
if str(TSASR_ROOT) not in sys.path:
    sys.path.insert(0, str(TSASR_ROOT))

from dynatar_qwen.aux_lmdb import (
    DEFAULT_COMMIT_INTERVAL,
    DEFAULT_LMDB_MAP_SIZE,
    LMDBWriter,
    open_readonly_lmdb,
    remove_lmdb_path,
    resolve_sidecar_path,
)


_THREAD_LOCAL = threading.local()


def resolve_lmdb_path(manifest_path: Path, value: str) -> str:
    return resolve_sidecar_path(manifest_path, value)


def _get_thread_env(lmdb_path: str):
    envs = getattr(_THREAD_LOCAL, "envs", None)
    if envs is None:
        envs = {}
        _THREAD_LOCAL.envs = envs
    env = envs.get(lmdb_path)
    if env is None:
        env = open_readonly_lmdb(lmdb_path)
        envs[lmdb_path] = env
    return env


def _load_record_payload(task: dict) -> tuple[dict, bytes]:
    env = _get_thread_env(task["src_lmdb_path"])
    with env.begin(write=False) as txn:
        payload = txn.get(task["src_lmdb_key"].encode("utf-8"))
    if payload is None:
        raise KeyError(f"Missing source LMDB key {task['src_lmdb_key']} in {task['src_lmdb_path']}")
    return task["merged_record"], bytes(payload)


def parse_args():
    parser = argparse.ArgumentParser("Merge TS-ASR overlap label manifests into one LMDB")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--map_size", type=int, default=DEFAULT_LMDB_MAP_SIZE)
    parser.add_argument("--commit_interval", type=int, default=DEFAULT_COMMIT_INTERVAL)
    parser.add_argument("manifests", nargs="+", help="Input aux_label_manifest.jsonl files")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_lmdb = output_dir / "aux_labels.lmdb"
    merged_manifest = output_dir / "aux_label_manifest.jsonl"
    temp_merged_lmdb = output_dir / "aux_labels.lmdb.tmp"
    temp_merged_manifest = output_dir / "aux_label_manifest.jsonl.tmp"
    remove_lmdb_path(temp_merged_lmdb)
    temp_merged_manifest.unlink(missing_ok=True)

    seen_keys = set()
    merge_tasks = []
    for manifest in args.manifests:
        manifest_path = Path(manifest)
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = str(record.get("key", "")).strip()
                if not key:
                    raise KeyError(f"Missing key in {manifest}:{line_number}")
                if key in seen_keys:
                    raise ValueError(f"Duplicate aux-label key while merging: {key}")
                seen_keys.add(key)

                storage = str(record.get("storage", "")).strip().lower()
                if storage != "lmdb":
                    raise ValueError(f"Expected storage=lmdb in {manifest}:{line_number}")
                lmdb_path_value = record.get("lmdb_path")
                src_lmdb_key = str(record.get("lmdb_key", "")).strip()
                if not lmdb_path_value:
                    raise KeyError(f"Missing lmdb_path in {manifest}:{line_number}")
                if not src_lmdb_key:
                    raise KeyError(f"Missing lmdb_key in {manifest}:{line_number}")

                merged_record = dict(record)
                merged_record.pop("label_path", None)
                merged_record.pop("path", None)
                merged_record["storage"] = "lmdb"
                merged_record["lmdb_path"] = merged_lmdb.name
                merged_record["lmdb_key"] = key

                merge_tasks.append(
                    {
                        "src_lmdb_path": resolve_lmdb_path(manifest_path, str(lmdb_path_value)),
                        "src_lmdb_key": src_lmdb_key,
                        "merged_record": merged_record,
                    }
                )

    with LMDBWriter(
        temp_merged_lmdb,
        map_size=args.map_size,
        commit_interval=args.commit_interval,
        overwrite=True,
    ) as lmdb_writer, temp_merged_manifest.open("w", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(int(args.num_workers), 1)) as executor:
            for merged_record, payload_bytes in executor.map(_load_record_payload, merge_tasks):
                lmdb_writer.put(str(merged_record["lmdb_key"]), payload_bytes)
                handle.write(json.dumps(merged_record, ensure_ascii=False) + "\n")

    remove_lmdb_path(merged_lmdb)
    temp_merged_lmdb.replace(merged_lmdb)
    merged_manifest.unlink(missing_ok=True)
    temp_merged_manifest.replace(merged_manifest)

    print(f"[merge-aux-labels] merged {len(merge_tasks)} records")
    print(f"[merge-aux-labels] lmdb={merged_lmdb}")
    print(f"[merge-aux-labels] manifest={merged_manifest}")


if __name__ == "__main__":
    main()
