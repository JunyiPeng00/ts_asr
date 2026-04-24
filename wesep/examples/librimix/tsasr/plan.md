# DynaTaR-Qwen TS-ASR Progress And Roadmap

Last updated: 2026-04-19

## 1. Project goal

The current project targets a TASLP-ready target-speaker ASR system built on
top of `Qwen3-ASR`, with the following method line:

1. target-conditioned acoustic encoding with enrollment guidance
2. overlap-aware auxiliary supervision
3. target-faithfulness regularization
4. label-guided dynamic routing specialization

The near-term research goal is no longer just "make TS-ASR run", but:

- make the full training and inference path stable
- build a clean TASLP ablation matrix
- generate paper-ready experiment tracking and result tables

## 2. Current progress

### 2.1 Core TS-ASR path

Completed:

- identified the main TS-ASR path under `examples/librimix/tsasr`
- separated it from pure TSE and cascade baselines
- confirmed the main runtime is the `dynatar_qwen` branch
- validated the training call chain from `run.sh` to `train_ts_qwen3_asr.py`
- validated the inference/scoring call chain from `infer_ts_qwen3_asr.py` to
  `score_ts_qwen3_asr.py`

Current understanding:

- the system is a target-conditioned ASR model, not the main cascade
  `TSE -> ASR` path
- enrollment enters the audio encoder and influences mix hidden states
- the decoder remains a Qwen3-ASR style text generation path

### 2.2 Data pipeline and contract

Completed:

- audited dataset, collator, model input contract, and loss path
- clarified the target-centric sample structure
- confirmed current main sample fields include:
  - `mix_audio`
  - `target_audio`
  - `enroll_audio`
  - `target_text`
  - `target_spk`
  - `target_role`
- fixed shard-list relative path handling
- added sidecar auxiliary label manifest support
- added overlap label loading into the dataset/collator path
- added optional target clean audio batching for target-consistency training

Current status:

- the data contract is now significantly cleaner and more explicit than before
- overlap supervision and target-audio supervision can both enter the runtime
  path

### 2.3 Model and losses

Completed:

- added `overlap head` supervision on final mix hidden states
- added `target consistency` branch using clean target audio
- added `router supervision` on the last refinement router
- integrated the combined training objective:

  `L = L_asr + λ_ov L_ov + λ_tc L_tc + λ_rt L_rt`

- added runtime logging for:
  - `tsasr/overlap_loss`
  - `tsasr/overlap_acc`
  - `tsasr/target_consistency_loss`
  - `tsasr/target_cosine`
  - `tsasr/router_loss`
  - `tsasr/router_target_ratio`
  - `tsasr/router_overlap_ratio`
  - `tsasr/router_nontarget_ratio`

Important note:

- the current `target consistency` implementation uses the internal
  Qwen-side audio tower path on clean target speech
- the stricter "external speaker encoder" version is intentionally not the
  current target

### 2.4 Training, inference, and checkpointing

Completed:

- improved training-time logging through `CastFloatInputsTrainer`
- ensured auxiliary logs are surfaced in `Trainer.log()`
- preserved checkpoint `config.json` instead of overwriting it with base-model
  files
- made inference prefer checkpoint config over manually re-entered CLI config
- made checkpoint loading stricter when checkpoint-side TS config is available
- completed minimal stage-level smoke coverage for:
  - stage 7 training
  - stage 8 generation
  - stage 9 scoring

Current status:

- the main path is now runnable and testable end to end in a minimal form
- full-scale multi-GPU quality experiments still need to be run and analyzed

### 2.5 Auxiliary labels and overlap supervision

Completed:

- finalized a target-centric overlap/routing label format
- implemented:
  - `build_overlap_labels.py`
  - `merge_aux_label_manifests.py`
- integrated stage 4 label generation into `run.sh`

Current status:

- the label format and recipe entrypoint now exist
- the remaining work is large-scale data generation, quality checks, and
  analysis scripts

### 2.6 TASLP experiment setup

Completed:

- created a dedicated TASLP config directory under `confs/taslp`
- defined the main paper ablation matrix:
  - `b0 baseline`
  - `b1 overlap`
  - `b2 overlap + target consistency`
  - `b3 full`
- added matched benchmark configs for:
  - `train-100`
  - `train-3mix-custom`
- added TASLP submission scripts for grouped Slurm launch:
  - `trainmerge_ablation`
  - `benchmark_2spk`
  - `benchmark_3spk`
  - `full_matrix`
- added result tracking and collection files under `paper/taslp`

Current status:

- the experiment matrix is defined and executable
- the remaining step is to actually run the jobs and collect results

## 3. What is already validated

Validated in code and smoke tests:

- dataset and collator contract
- forward and backward for the main TS-ASR path
- auxiliary losses backward path
- stage 7 trainer smoke
- stage 8 generation smoke
- stage 9 scoring smoke
- YAML config parsing for TASLP configs
- Slurm submission script syntax
- TASLP result auto-collection script

Validated by project organization:

- experiment matrix is documented
- result templates exist
- submission entrypoints are grouped for paper runs

Not yet validated at full scale:

- full multi-GPU TASLP ablation runs
- full dev/test decode quality for new configs
- high-overlap subset analysis
- router agreement and overlap-head analysis on real runs

## 4. Current bottlenecks

### 4.1 Research bottlenecks

- no dynamic prototype update yet
- no dedicated router balance / entropy / smoothness regularization yet
- overlap-aware analysis metrics are not yet automatically produced
- no matched mixture-only Qwen3-ASR config inside this TS-ASR recipe
- no fully integrated cascade baseline comparison table in the TASLP pipeline

### 4.2 Engineering bottlenecks

- large-scale experiments still need queue time and cluster stability
- multi-node stability must still be treated cautiously
- not all legacy experiment folders use the new result bookkeeping structure
- automatic paper analysis is still missing deeper diagnostics beyond
  `WER/CER`

## 5. Immediate next plan

### 5.1 Execute the TASLP experiment matrix

Priority order:

1. `train-merge` main ablation:
   - `b0`
   - `b1`
   - `b2`
   - `b3`
2. matched 2-speaker benchmark:
   - `train100 b0`
   - `train100 b3`
3. matched 3-speaker benchmark:
   - `train3mix_custom b0`
   - `train3mix_custom b3`

Execution entrypoint:

- `submit_run_taslp.sh`

Expected outputs:

- checkpoints
- `infer_*` decode folders
- `wer_summary.json`
- auto-collected TASLP result tables

### 5.2 Build paper-facing analysis scripts

Next analysis items:

- high-overlap subset WER/CER
- overlap-ratio bucketed WER
- overlap-head token accuracy and macro F1
- router-label agreement
- router expert usage summaries
- target consistency cosine summaries

This is the most important missing paper layer after the raw experiments.

### 5.3 Lock the paper baseline story

Still needed:

- mixture-only Qwen3-ASR baseline
- cascade TSE -> ASR baseline comparison table
- optional simplified conditioning baseline if needed for rebuttal safety

The goal is to make the paper story robust even if one proposed branch gives
smaller-than-expected gains.

## 6. Medium-term method plan

### 6.1 Dynamic prototype update

Still planned:

- conservative EMA-style update
- confidence-based target frame selection
- anti-drift regularization
- optional layer-wise update interval

This remains a meaningful future method extension, but it is not required for
the current TASLP submission package.

### 6.2 Router regularization

Still planned:

- load-balance loss
- entropy regularization
- temporal smoothness regularization

This should be considered after the current overlap/router supervision results
are observed.

### 6.3 Richer evaluation and logging

Still planned:

- decoded qualitative examples
- TensorBoard-side richer validation snapshots
- experiment summary markdown/json export
- cleaner paper plotting support

## 7. Suggested working order

Recommended order from here:

1. generate or verify stage 4 auxiliary labels for all needed splits
2. launch `train-merge` ablations
3. launch matched 2-speaker and 3-speaker baseline/full runs
4. collect all results with `collect_taslp_results.py`
5. implement overlap/router diagnostic scripts
6. fill paper tables
7. decide whether dynamic prototype update is needed before submission

## 8. Bottom-line project status

The project has moved beyond the "minimum working prototype" phase.

Current status is better described as:

- core method path implemented
- auxiliary supervision package integrated
- training/inference/scoring path validated in smoke form
- TASLP experiment matrix defined
- submission scripts and result templates prepared

The main remaining work is now experimental execution and paper analysis,
rather than basic plumbing.
