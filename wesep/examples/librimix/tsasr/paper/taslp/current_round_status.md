# TS-ASR Current Round Status

Last verified from disk: 2026-04-24

## 1. Directory Layout

The current TS-ASR working tree is organized around four active areas:

- `confs/taslp/`
  The official TASLP experiment configs for the current round.
- `exp/`
  Experiment outputs. The official matrix uses `exp/taslp_dqwen_*`; the
  completed pre-experiment lives under `exp/taslp_trainmerge_b3_lmdb_preexp500`.
- `paper/taslp/`
  Paper-facing progress notes, templates, and auto-collected result summaries.
- `log/YYYY-MM-DD/`
  Slurm submission logs. These are now ignored by git and are treated as runtime
  artifacts rather than source files.

## 2. Official Config Set

The current official TASLP config set under `confs/taslp/` is:

- `taslp_trainmerge_b0_baseline.yaml`
- `taslp_trainmerge_b1_overlap.yaml`
- `taslp_trainmerge_b2_overlap_tc.yaml`
- `taslp_trainmerge_b3_full.yaml`
- `taslp_train100_b0_baseline.yaml`
- `taslp_train100_b3_full.yaml`
- `taslp_train100_full.yaml`
- `taslp_train3mix_custom_b0_baseline.yaml`
- `taslp_train3mix_custom_b3_full.yaml`

The round-level launcher summary is kept in:

- `confs/taslp/this_round_full_matrix.yaml`

## 3. What Is Confirmed Complete

### 3.1 Data and recipe plumbing

The following are already in place in the current repository:

- target-centric shard reading
- LMDB-only auxiliary label loading
- merged `train-merge` aux-label LMDB generation
- TASLP submission scripts and one-shot full-matrix launcher

### 3.2 Completed pre-experiment

The latest fully completed training/inference sanity run is:

- `exp/taslp_trainmerge_b3_lmdb_preexp500`

Verified artifacts:

- checkpoints: `checkpoint-100`, `200`, `300`, `400`, `500`
- decode output:
  `exp/taslp_trainmerge_b3_lmdb_preexp500/infer_test_checkpoint-500_smoke/`

Verified numbers from disk:

- `WER = 0.3031674208144796`
- `CER = 0.2477432296890672`
- `num_samples = 8`
- final `eval_loss = 1.6782660484313965`
- final `target_cosine = 0.9700303077697754`
- final `train_loss = 2.8536173667907714`

This run is useful as a pipeline-stability reference, not as the final paper
result.

## 4. What Exists But Is Not Yet Complete

The official matrix experiment directories already exist:

- `exp/taslp_dqwen_0p6b_trainmerge_b0_baseline`
- `exp/taslp_dqwen_0p6b_trainmerge_b1_overlap`
- `exp/taslp_dqwen_0p6b_trainmerge_b2_overlap_tc`
- `exp/taslp_dqwen_0p6b_trainmerge_b3_full`
- `exp/taslp_dqwen_0p6b_train100_b0_baseline`
- `exp/taslp_dqwen_0p6b_train100_b3_full`
- `exp/taslp_dqwen_0p6b_train100_full`
- `exp/taslp_dqwen_0p6b_train3mix_custom_b0_baseline`
- `exp/taslp_dqwen_0p6b_train3mix_custom_b3_full`

What is confirmed today:

- each official experiment directory already has `run_config.yaml`
- `exp/taslp_dqwen_0p6b_train100_full` has already written `checkpoint-1000`
- the other official matrix directories do not yet show checkpoints on disk
- none of them has produced an official `infer_*/wer_summary.json` yet

So the current official matrix state is:

- config-ready
- launch-ready
- partially submitted
- partially checkpointed
- not yet result-complete

## 5. Current Submission Activity

The current Slurm logs under `log/2026-04-24/` show two submission batches on
disk, including job IDs:

- `17796109` to `17796112`
- `17796430` to `17796434`

The latest visible log modification times I verified were around:

- `2026-04-24 06:21` to `06:22` EEST

This supports the following careful status statement:

- **confirmed**: official TASLP jobs have been submitted and are writing logs
- **confirmed**: the logs are updating recently on disk
- **inference**: the jobs appear active or were active very recently

I am not marking them as fully completed here because the official experiment
directories do not yet contain checkpoints or final decode summaries.

## 6. Auto-Collected Paper Summary

The auto-collected files were refreshed from the current directory state:

- `paper/taslp/results_master.csv`
- `paper/taslp/results_snapshot.md`

Current interpretation:

- the collector now only scans `taslp_*.yaml`
- the non-experiment round summary file is no longer mixed into the CSV
- the official matrix rows show `run_config_exists = 1`
- `taslp_train100_full` already shows `latest_checkpoint_step = 1000`
- the official matrix rows still have empty WER / CER fields

This is the correct current state for the paper summary: the matrix is defined
and launched, with at least one official checkpoint already written, but the
official results are not yet complete.

## 7. Recommended Next Actions

1. Let the current official jobs continue writing checkpoints.
2. Re-run `python local/collect_taslp_results.py` after the first official
   checkpoint and decode outputs appear.
3. Fill `paper/taslp/results_master.csv` from official matrix outputs first.
4. Keep `preexp500` only as a stability reference, not as a final table row.
