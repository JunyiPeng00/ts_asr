# Ablation Table Template

| ID | Model Variant | Overlap Head | Target Consistency | Router Supervision | Train Split | WER | CER | High-Overlap WER | Overlap Acc | Router Agreement | Notes |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| B0 | DynaTaR-Qwen baseline | No | No | No | `train-merge` |  |  |  |  |  |  |
| B1 | `+ overlap-aware acoustic supervision` | Yes | No | No | `train-merge` |  |  |  |  |  |  |
| B2 | `+ overlap + target-faithfulness` | Yes | Yes | No | `train-merge` |  |  |  |  |  |  |
| B3 | `+ overlap + target-faithfulness + router supervision` | Yes | Yes | Yes | `train-merge` |  |  |  |  |  |  |

## Caption Draft

Effect of the proposed overlap-aware supervision, target-faithfulness
regularization, and label-guided router specialization on the main
`train-merge` TS-ASR setting.
