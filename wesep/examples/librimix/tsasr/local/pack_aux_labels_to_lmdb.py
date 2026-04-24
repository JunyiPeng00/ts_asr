from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

TSASR_ROOT = Path(__file__).resolve().parents[1]
if str(TSASR_ROOT) not in sys.path:
    sys.path.insert(0, str(TSASR_ROOT))

from dynatar_qwen.aux_lmdb import (
    DEFAULT_COMMIT_INTERVAL,
    DEFAULT_LMDB_MAP_SIZE,
    LMDBWriter,
    decode_aux_payload_bytes,
    open_readonly_lmdb,
    read_lmdb_keys,
    remove_lmdb_path,
    resolve_sidecar_path,
)


def resolve_label_path(manifest_path: Path, value: str) -> Path:
    return Path(resolve_sidecar_path(manifest_path, value))


def parse_args():
    parser = argparse.ArgumentParser("Pack legacy aux_labels/*.npz into aux_labels.lmdb")
    parser.add_argument("--manifest", type=str, required=True, help="Legacy aux_label_manifest.jsonl")
    parser.add_argument("--output_dir", type=str, default="", help="Defaults to manifest parent")
    parser.add_argument("--map_size", type=int, default=DEFAULT_LMDB_MAP_SIZE)
    parser.add_argument("--commit_interval", type=int, default=DEFAULT_COMMIT_INTERVAL)
    parser.add_argument("--validate_samples", type=int, default=8)
    parser.add_argument("--delete_source_dir", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else manifest_path.parent.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = output_dir / "aux_label_manifest.jsonl"
    output_lmdb = output_dir / "aux_labels.lmdb"
    temp_output_manifest = output_dir / "aux_label_manifest.jsonl.tmp"
    temp_output_lmdb = output_dir / "aux_labels.lmdb.tmp"
    remove_lmdb_path(temp_output_lmdb)
    temp_output_manifest.unlink(missing_ok=True)

    legacy_records = []
    source_dirs = set()
    seen_keys = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = str(record.get("key", "")).strip()
            if not key:
                mix_id = str(record.get("mix_id", "")).strip()
                target_role = str(record.get("target_role", "")).strip()
                if not mix_id or not target_role:
                    raise KeyError(f"Missing key or (mix_id, target_role) in {manifest_path}:{line_number}")
                key = f"{mix_id}__{target_role}"
            if key in seen_keys:
                raise ValueError(f"Duplicate aux-label key in {manifest_path}: {key}")
            seen_keys.add(key)
            label_value = record.get("label_path") or record.get("path")
            if not label_value:
                raise KeyError(f"Missing label_path in {manifest_path}:{line_number}")
            label_path = resolve_label_path(manifest_path, str(label_value))
            source_dirs.add(label_path.parent)
            legacy_records.append((key, record, label_path))

    with LMDBWriter(
        temp_output_lmdb,
        map_size=args.map_size,
        commit_interval=args.commit_interval,
        overwrite=True,
    ) as lmdb_writer, temp_output_manifest.open("w", encoding="utf-8") as handle:
        for key, record, label_path in legacy_records:
            with label_path.open("rb") as label_handle:
                payload_bytes = label_handle.read()
            lmdb_writer.put(key, payload_bytes)
            updated_record = dict(record)
            updated_record.pop("label_path", None)
            updated_record.pop("path", None)
            updated_record["storage"] = "lmdb"
            updated_record["key"] = key
            updated_record["lmdb_path"] = output_lmdb.name
            updated_record["lmdb_key"] = key
            handle.write(json.dumps(updated_record, ensure_ascii=False) + "\n")

    lmdb_keys = read_lmdb_keys(temp_output_lmdb)
    if len(lmdb_keys) != len(legacy_records):
        raise ValueError(f"LMDB key count mismatch: {len(lmdb_keys)} vs {len(legacy_records)}")
    if set(lmdb_keys) != {key for key, _, _ in legacy_records}:
        raise ValueError("LMDB keys do not match manifest keys after packing")

    sample_count = min(max(int(args.validate_samples), 0), len(legacy_records))
    env = open_readonly_lmdb(temp_output_lmdb)
    try:
        sample_records = random.sample(legacy_records, sample_count) if sample_count else []
        for key, _, _ in sample_records:
            with env.begin(write=False) as txn:
                payload_bytes = txn.get(key.encode("utf-8"))
            if payload_bytes is None:
                raise KeyError(f"Missing LMDB payload for {key}")
            decoded = decode_aux_payload_bytes(bytes(payload_bytes))
            if "router_label" not in decoded:
                raise KeyError(f"Missing router_label in LMDB payload for {key}")
    finally:
        env.close()

    remove_lmdb_path(output_lmdb)
    temp_output_lmdb.replace(output_lmdb)
    output_manifest.unlink(missing_ok=True)
    temp_output_manifest.replace(output_manifest)

    if args.delete_source_dir:
        if len(source_dirs) != 1:
            raise ValueError(f"Expected exactly one source aux_labels directory, got: {sorted(map(str, source_dirs))}")
        source_dir = next(iter(source_dirs))
        if source_dir.exists():
            shutil.rmtree(source_dir)

    print(f"[pack-aux-labels] packed {len(legacy_records)} records")
    print(f"[pack-aux-labels] lmdb={output_lmdb}")
    print(f"[pack-aux-labels] manifest={output_manifest}")
    if args.delete_source_dir:
        print(f"[pack-aux-labels] deleted_source_dir={next(iter(source_dirs))}")


if __name__ == "__main__":
    main()
