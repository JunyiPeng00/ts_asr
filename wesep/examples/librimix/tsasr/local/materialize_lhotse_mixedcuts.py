from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf
from lhotse.serialization import deserialize_item


def parse_args():
    parser = argparse.ArgumentParser("Materialize Lhotse MixedCut manifest to wav files")
    parser.add_argument("--manifest", required=True, help="Path to .jsonl or .jsonl.gz cutset manifest")
    parser.add_argument("--output-dir", required=True, help="Directory for generated wav files")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-cuts", type=int, default=0, help="Only process the first N MixedCuts if > 0")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def iter_mixedcut_lines(manifest_path: Path, max_cuts: int = 0):
    if manifest_path.suffix == ".gz":
        import gzip

        opener = lambda path: gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = lambda path: path.open("r", encoding="utf-8")

    seen = 0
    with opener(manifest_path) as handle:
        for line in handle:
            obj = json.loads(line)
            if obj.get("type") != "MixedCut":
                continue
            yield line
            seen += 1
            if max_cuts > 0 and seen >= max_cuts:
                return


def _materialize_one(task):
    line, output_dir, skip_existing = task
    obj = json.loads(line)
    cut = deserialize_item(obj)
    out_path = Path(output_dir) / f"{cut.id}.wav"

    if skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
        return "exists", cut.id

    audio = cut.load_audio().squeeze(0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, 16000, subtype="PCM_16")
    return "written", cut.id


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    exists = 0
    total = 0

    tasks = (
        (line, str(output_dir), args.skip_existing)
        for line in iter_mixedcut_lines(manifest_path, max_cuts=args.max_cuts)
    )

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for status, cut_id in executor.map(_materialize_one, tasks, chunksize=8):
            total += 1
            if status == "written":
                written += 1
            elif status == "exists":
                exists += 1
            if total % 500 == 0:
                print(f"processed={total} written={written} exists={exists} last={cut_id}", flush=True)

    print(f"done processed={total} written={written} exists={exists}", flush=True)


if __name__ == "__main__":
    main()
