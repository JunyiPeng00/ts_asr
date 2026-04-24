# TASLP Paper Assets

This folder stores paper-facing experiment bookkeeping for the current
TS-ASR TASLP line.

## Files

- `results_master_template.csv`
  A manual spreadsheet-style template for the full experiment matrix.
- `ablation_table_template.md`
  A draft table skeleton for the main ablation section.
- `benchmark_table_template.md`
  A draft table skeleton for the matched 2-speaker / 3-speaker benchmarks.
- `current_round_status.md`
  A directory-aware progress note for the current TS-ASR round, including
  official TASLP configs, active log locations, and the latest verified
  completed pre-experiment.

## Auto-collection

Use the helper below to scan `confs/taslp/*.yaml` and available experiment
outputs under `exp/`:

```bash
python local/collect_taslp_results.py
```

This writes:

- `paper/taslp/results_master.csv`
- `paper/taslp/results_snapshot.md`

The auto-collected table fills fields that can be inferred from:

- config files
- `run_config.yaml`
- `infer_*/decode_summary.json`
- `infer_*/wer_summary.json`

The following paper metrics are intentionally left blank and should be filled
after dedicated analysis:

- `high_overlap_wer`
- `high_overlap_cer`
- `overlap_acc`
- `overlap_macro_f1`
- `router_label_agreement`
- `router_target_ratio`
- `router_overlap_ratio`
- `router_nontarget_ratio`
- `target_cosine`

## Suggested workflow

1. Run the TASLP experiments via `submit_run_taslp.sh`.
2. Run `python local/collect_taslp_results.py`.
3. Copy or merge the generated `results_master.csv` into the template if you
   want to preserve manual notes.
4. Fill the analysis-only fields after running overlap/router diagnostics.
