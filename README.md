# TS-ASR

This repository contains the current target-speaker ASR codebase built around
Qwen3-ASR and the WeSep LibriMix recipe.

The active research line is a target-conditioned TS-ASR system aimed at the
TASLP setting, with three main additions on top of the baseline DynaTaR-Qwen
pipeline:

1. overlap-aware acoustic supervision
2. target-faithfulness regularization
3. label-guided dynamic routing

The goal is to move beyond a mixture ASR model with enrollment bias and build a
system that is structurally target-aware and experimentally defensible.

## Repository Layout

- `Qwen3-ASR/`
  - local copy of the Qwen3-ASR backend used by the TS-ASR recipe
- `wesep/examples/librimix/tsasr/`
  - active TS-ASR recipe, configs, data-preparation scripts, tests, and TASLP
    experiment helpers

The main working directory is:

```bash
cd /scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr
```

## Environment

All terminal sessions in this project should use the unified LUMI environment:

```bash
module purge
module load LUMI
module load PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250404
singularity shell $SIF
```

When working inside an existing Slurm allocation, we also pin the session to the
job node explicitly, for example:

```bash
srun -n 1 --overlap --pty --jobid=<JOB_ID> $SHELL
taskset -pc 0-64 $$
```

## Active Recipe

The main recipe README lives at:

- `wesep/examples/librimix/tsasr/README.md`

That document covers:

- LibriMix metadata preparation
- enrollment metadata generation
- target-centric shard generation
- expected data files under `data/clean/<split>/`

The TASLP experiment matrix is documented at:

- `wesep/examples/librimix/tsasr/confs/taslp/README.md`

## Data Contract

The current TS-ASR training path reads:

- `data/clean/<split>/shard.list`
- `data/clean/<split>/shards/*.tar`
- `data/clean/<split>/spk2enroll.json`
- `data/clean/<split>/single.wav.scp`

For overlap and router supervision, the runtime now uses LMDB-backed auxiliary
labels only:

- `data/clean/<split>/aux_label_manifest.jsonl`
- `data/clean/<split>/aux_labels.lmdb`

Key points:

- training no longer reads `aux_labels/*.npz` small files
- the manifest is expected to contain `storage: "lmdb"` records
- `train-merge` builds its own merged `aux_labels.lmdb`

This means transcript edits in loose `spk*_text` files only affect training
after the relevant shards are rebuilt.

## Main Experiment Entry Points

The current paper-facing configs are:

- `wesep/examples/librimix/tsasr/confs/taslp/taslp_trainmerge_b0_baseline.yaml`
- `wesep/examples/librimix/tsasr/confs/taslp/taslp_trainmerge_b1_overlap.yaml`
- `wesep/examples/librimix/tsasr/confs/taslp/taslp_trainmerge_b2_overlap_tc.yaml`
- `wesep/examples/librimix/tsasr/confs/taslp/taslp_trainmerge_b3_full.yaml`

Matched benchmark configs are also provided for:

- `train-100`
- `train-3mix-custom`

Typical usage:

```bash
cd /scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr

bash run.sh \
  --config confs/taslp/taslp_trainmerge_b3_full.yaml \
  --stage 7 \
  --stop_stage 9
```

## Current Status

The repository has already been migrated to the new project path and verified on
GPU with:

- Qwen3-ASR backend connected in-repo
- target-speaker shard loading working on real data
- LMDB-only auxiliary labels working for source splits and `train-merge`
- TS-ASR stage 7/8/9 smoke runs passing on GPU

Recent small-scale validation has also been run for the `train-merge B3`
setting, including a longer pre-experiment smoke with `500` training steps to
confirm that:

- overlap supervision is active
- target-consistency supervision is active
- router supervision is active
- `target_cosine` rises during training, which is the intended direction

## Notes For Contributors

- Keep the work centered on the TS-ASR mainline under
  `wesep/examples/librimix/tsasr/`
- Do not assume loose text-file edits have entered training unless the shard
  path has been rebuilt
- Prefer small, verifiable changes
- Preserve the current recipe split between:
  - recipe/data code in `wesep/examples/librimix/tsasr/`
  - backend model code in `Qwen3-ASR/`

## Tests

From the recipe directory:

```bash
python tests/test_ts_qwen3_asr.py
```

For end-to-end smoke coverage, use the staged `run.sh` flow or the targeted
smoke utilities in `wesep/examples/librimix/tsasr/tests/`.
