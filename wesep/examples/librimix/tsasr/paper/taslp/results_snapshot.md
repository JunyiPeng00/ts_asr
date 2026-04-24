# TASLP Result Snapshot

Auto-generated from `confs/taslp/*.yaml` and available `exp/*/infer_*/wer_summary.json` files.

## Main Ablation

| config | method | split | ckpt | WER | CER | infer_dir |
| --- | --- | --- | ---: | ---: | ---: | --- |
| taslp_trainmerge_b0_baseline | b0_baseline | train-merge | - | - | - | - |
| taslp_trainmerge_b1_overlap | b1_overlap | train-merge | - | - | - | - |
| taslp_trainmerge_b2_overlap_tc | b2_overlap_tc | train-merge | - | - | - | - |
| taslp_trainmerge_b3_full | b3_full | train-merge | - | - | - | - |

## 2-Speaker Benchmark

| config | method | split | ckpt | WER | CER | infer_dir |
| --- | --- | --- | ---: | ---: | ---: | --- |
| taslp_train100_b0_baseline | b0_baseline | train-100 | - | - | - | - |
| taslp_train100_b3_full | b3_full | train-100 | - | - | - | - |
| taslp_train100_full | b3_full | train-100 | 1000 | - | - | - |

## 3-Speaker Benchmark

| config | method | split | ckpt | WER | CER | infer_dir |
| --- | --- | --- | ---: | ---: | ---: | --- |
| taslp_train3mix_custom_b0_baseline | b0_baseline | train-3mix-custom | - | - | - | - |
| taslp_train3mix_custom_b3_full | b3_full | train-3mix-custom | - | - | - | - |

## Manual Fields To Fill Later

- `high_overlap_wer` / `high_overlap_cer`
- `overlap_acc` / `overlap_macro_f1`
- `router_label_agreement`
- `router_target_ratio` / `router_overlap_ratio` / `router_nontarget_ratio`
- `target_cosine`
- `notes`
