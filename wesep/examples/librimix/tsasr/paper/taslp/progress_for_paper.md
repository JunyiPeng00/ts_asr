# TASLP Paper-Oriented Progress Note

Last updated: 2026-04-19

## 1. Recommended paper positioning

The current work is best framed as a **target-speaker-conditioned ASR system**
with three explicitly designed enhancements:

1. **Overlap-aware acoustic supervision**
2. **Target-faithfulness regularization**
3. **Label-guided dynamic routing specialization**

At this stage, the paper should not be positioned as:

- a fully joint separation-and-recognition framework
- a cascade TSE -> ASR system
- a dynamic prototype-update model

The strongest current framing is:

> We build a target-conditioned Qwen3-ASR system whose acoustic encoder is
> guided by enrollment speech, then improve it with explicit overlap-aware
> supervision, target-faithfulness regularization, and weakly supervised router
> specialization.

## 2. What can already be claimed

### 2.1 Confirmed implementation contributions

The following are already implemented in the current codebase:

- a target-aware TS-ASR branch built on top of `Qwen3-ASR`
- target-centric data loading with:
  - mixture waveform
  - enrollment waveform
  - target clean waveform
  - target transcript
- overlap label sidecar generation and loading
- an auxiliary **overlap head** on final mix hidden states
- a **target-consistency** branch using clean target speech
- a **router supervision** loss on the last refinement router
- end-to-end training / inference / scoring scripts
- TASLP experiment configs, job-launch scripts, and result collection scripts

### 2.2 Confirmed engineering status

The following are already validated at smoke-test level:

- stage 7 training path
- stage 8 generation path
- stage 9 scoring path
- config parsing and checkpoint config restoration
- TASLP experiment matrix and Slurm submission entrypoints
- result collection into CSV / Markdown snapshots

### 2.3 What should still be described cautiously

These parts are implemented but should not yet be overclaimed in the paper:

- large-scale performance gains
- robustness under multi-speaker overlap
- stable router specialization on real runs
- improved target faithfulness on benchmark evaluation

Reason:

- the paper-scale experiments have not yet been completed
- current evidence is implementation-complete, not result-complete

## 3. Current paper story

## 3.1 Main technical narrative

The paper can currently be organized around the following causal story:

1. **Baseline target-conditioned TS-ASR**
   Enrollment speech guides the acoustic encoder, but the model is still mainly
   optimized by transcript-level ASR loss.

2. **Overlap-aware supervision**
   The model is explicitly told which mixture regions correspond to
   `target_only`, `target_overlap`, and `non_target_only`, making the
   overlap-awareness claim concrete rather than implicit.

3. **Target-faithfulness regularization**
   Clean target speech is no longer ignored; it becomes a supervision source
   that encourages the conditioned representation to stay closer to the target
   speaker.

4. **Router specialization**
   The dynamic router is no longer trained only through ASR CE. Weak label
   guidance makes the routing path more interpretable and more aligned with the
   overlap structure.

## 3.2 Main message in one sentence

Recommended one-line summary:

> Explicit overlap supervision and target-faithfulness regularization strengthen
> target-conditioned ASR by making the acoustic encoder more aware of who is
> speaking and when the target is masked by interference.

## 4. What the paper can already contain

### 4.1 Method section

The following method content can already be drafted with confidence:

- target-centric sample definition
- unified audio input format `[enroll ; silence ; mix]`
- target-aware encoder design
- overlap label format and supervision path
- target-consistency branch
- router supervision mapping
- total training objective

Recommended equation block:

```text
L = L_asr + λ_ov L_ov + λ_tc L_tc + λ_rt L_rt
```

where:

- `L_asr` is transcript cross-entropy
- `L_ov` is overlap classification loss
- `L_tc` is target-faithfulness / target-consistency loss
- `L_rt` is router supervision loss

### 4.2 Data section

The following are already stable enough to describe:

- target-centric training sample construction
- auxiliary label sidecar design
- 2-speaker and variable-speaker custom mixture support
- distinction between:
  - `train-100`
  - `train-merge`
  - `train-3mix-custom`

### 4.3 Experiment section skeleton

The current experiment section can already be structured as:

1. Main ablation on `train-merge`
2. Matched 2-speaker benchmark on `train-100`
3. Matched multi-speaker benchmark on `train-3mix-custom`
4. Overlap-aware and router-aware analysis

## 5. Recommended experiment matrix for the paper

### 5.1 Main ablation

The central ablation should remain:

- `B0`: baseline target-conditioned TS-ASR
- `B1`: `B0 + overlap head`
- `B2`: `B1 + target consistency`
- `B3`: `B2 + router supervision`

These runs already have matching configs under `confs/taslp`.

### 5.2 Matched benchmarks

The benchmark comparison should report:

- `train-100`: `B0` vs `B3`
- `train-3mix-custom`: `B0` vs `B3`

This gives:

- a clean 2-speaker matched condition
- a clean multi-speaker matched condition

### 5.3 External baselines

The paper should still include, when possible:

- mixture-only Qwen3-ASR baseline
- cascade TSE -> ASR baseline

However, these should be described as **comparison baselines**, not as part of
the main TS-ASR ablation ladder.

## 6. Tables that should appear in the paper

### 6.1 Main ablation table

Already scaffolded in:

- `paper/taslp/ablation_table_template.md`

The final version should minimally include:

- WER
- CER
- high-overlap WER
- overlap accuracy or macro-F1
- router agreement or specialization summary

### 6.2 Matched benchmark table

Already scaffolded in:

- `paper/taslp/benchmark_table_template.md`

The final version should compare:

- baseline vs full model on 2-speaker
- baseline vs full model on 3-speaker

### 6.3 Optional analysis table

Recommended extra table:

- overlap-ratio bucketed WER
- router-label agreement
- target consistency cosine

This would help support the mechanism claim, not just the final WER claim.

## 7. Figures that are worth preparing

Recommended figures:

1. **Model overview figure**
   Show enrollment input, target-aware encoder, overlap head, target
   consistency branch, and router supervision.

2. **Token-level supervision figure**
   Show the four overlap classes on the target-centric mixture timeline.

3. **Ablation bar chart**
   Compare `B0/B1/B2/B3` on WER and high-overlap WER.

4. **Router usage figure**
   Show average expert usage or routing ratios under different model variants.

5. **Overlap bucket figure**
   Plot WER versus overlap ratio bucket.

## 8. Writing constraints and safe wording

### 8.1 Claims that are safe now

Safe wording:

- "we implement"
- "we introduce"
- "we augment the target-conditioned encoder with"
- "we supervise overlap-aware acoustic states using target-centric labels"
- "we regularize the conditioned representation toward target speech"

### 8.2 Claims that should wait for results

Avoid strong wording like:

- "significantly outperforms"
- "substantially improves robustness"
- "consistently specializes"
- "state-of-the-art"

until the real experiment matrix is complete.

### 8.3 Claims that should remain narrow

At the current stage, the paper should avoid implying:

- universal multi-speaker generalization
- full separation capability
- dynamic prototype adaptation

unless those pieces are actually added and verified later.

## 9. Remaining gaps before the paper is result-complete

### 9.1 Must-finish experimental work

- run the full TASLP matrix
- decode and score all completed runs
- collect results into the master CSV / snapshot tables

### 9.2 Must-finish analysis work

- high-overlap subset evaluation
- overlap-ratio bucketed WER
- overlap-head token metrics
- router-label agreement
- target-consistency summary metrics

### 9.3 Strongly recommended comparison work

- mixture-only Qwen3-ASR baseline
- cascade TSE -> ASR comparison

These are important for reviewer-facing credibility.

## 10. Immediate writing plan

Recommended order:

1. write Introduction draft
2. write Related Work draft
3. write Method draft in full detail
4. write Data / Training Setup draft
5. leave Results numbers blank initially
6. run TASLP matrix
7. auto-collect results
8. fill tables and analysis
9. tighten claims based on actual numbers

## 11. Suggested section-level writing outline

### Abstract

Focus on:

- target-speaker ASR challenge under overlap
- explicit overlap-aware and target-faithful supervision
- dynamic routing specialization
- improved recognition under target-speaker interference

### Introduction

Focus on:

- why mixture ASR fails for target-specific transcription
- why enrollment guidance alone is often too weak
- why explicit overlap-aware and target-faithful supervision is needed

### Method

Subsections:

- target-centric TS-ASR formulation
- target-aware Qwen acoustic encoder
- overlap-aware acoustic supervision
- target-faithfulness regularization
- label-guided router specialization
- full loss

### Experiments

Subsections:

- datasets and splits
- implementation details
- baselines
- main ablation
- matched benchmarks
- overlap-aware analysis

### Discussion

Focus on:

- when overlap supervision helps most
- whether router supervision improves interpretability
- whether target consistency helps under heavy overlap

## 12. Bottom-line paper status

The project is already at a stage where the **Method** and **Experiment
Design** sections can be drafted in a paper-oriented way.

What is still missing is not the paper story itself, but the **completed
result layer**:

- full ablation runs
- matched benchmark runs
- analysis metrics
- final tables and figures

So the current writing status is best described as:

> the paper framework is ready, the implementation is ready enough, and the
> remaining work is primarily experimental execution and evidence gathering.
