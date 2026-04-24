# Overlap / Routing Auxiliary Label Format

## Scope

This label format is designed for:

- `/scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr/data/clean/train-3mix-custom`
- the current target-centric TS-ASR training pipeline
- future routing regularization and overlap-specific auxiliary loss

It is intentionally defined for **variable speaker count mixtures**, not only fixed 3-speaker mixtures.

## What the current data supports

From the current `train-3mix-custom` metadata:

- `wav.scp` contains `key mix_wav src1 src2 ... srcN`
- `utt2spk` contains `key spk1 spk2 ... spkN`
- the current set contains **2 to 6 source speakers per mixture**
- target transcripts already exist as `spk1_text`, `spk2_text`, ..., `spk5_text`

So we already have enough information to derive weak frame-level overlap labels from clean sources.

## Core principle

The auxiliary label should be **target-centric**.

That means one target sample:

- `mix_id = 4d256d78-96af-476f-870d-dc4905e7acca`
- `target_role = spk2`

gets its own frame/token labels, rather than storing one mixture-global label and expanding later.

This matches the current TS-ASR sample definition:

- one mixture
- one target speaker
- one target transcript
- one enrollment

## Recommended storage layout

For each split, create:

- `aux_labels.lmdb`
- `aux_label_manifest.jsonl`

Example:

```text
train-3mix-custom/
  aux_labels.lmdb/
  aux_label_manifest.jsonl
```

## One label record per target sample

Recommended payload format inside LMDB:

- `npz` bytes

Reason:

- variable-length arrays are easy to store
- variable number of speakers per mixture is naturally supported
- runtime can decode directly from LMDB without materializing many small files

## LMDB key naming

Use:

```text
{mix_id}__{target_role}
```

Examples:

- `4d256d78-96af-476f-870d-dc4905e7acca__spk1`
- `4d256d78-96af-476f-870d-dc4905e7acca__spk3`

## Required arrays inside each `.npz`

### 1. `router_label`

Type:

- `int8`

Shape:

- `[T_mix]`

Meaning:

- main 4-class target-centric routing label

Class definition:

- `0`: silence
- `1`: target_only
- `2`: target_overlap
- `3`: non_target_only

Interpretation:

- `silence`: no speaker active
- `target_only`: target active, all interferers inactive
- `target_overlap`: target active, at least one interferer active
- `non_target_only`: target inactive, at least one interferer active

This is the main label used for overlap-specific auxiliary loss.

### 2. `target_active`

Type:

- `uint8`

Shape:

- `[T_mix]`

Values:

- `0` or `1`

Meaning:

- whether the target speaker is active at this frame/token

### 3. `interferer_count`

Type:

- `uint8`

Shape:

- `[T_mix]`

Meaning:

- number of active non-target speakers at this frame/token

Examples:

- `0`: no interferer active
- `1`: one interferer active
- `2`: two interferers active
- `3+`: for larger mixtures

### 4. `active_speaker_count`

Type:

- `uint8`

Shape:

- `[T_mix]`

Meaning:

- total number of active speakers including target

### 5. `source_activity`

Type:

- `uint8`

Shape:

- `[T_mix, N]`

Meaning:

- per-source binary activity
- `N` is the actual number of sources in this mixture

Example:

- for a 4-speaker mixture: shape is `[T_mix, 4]`

This gives maximum flexibility for future supervision or analysis.

## Recommended metadata fields inside each `.npz`

Store these as arrays or scalars:

- `target_index`
- `num_sources`

Where:

- `target_index` is zero-based index in the source list
- `num_sources` is the number of sources in the current mixture

Example:

- if target role is `spk3`, then `target_index = 2`

## Recommended sidecar manifest format

File:

- `aux_label_manifest.jsonl`

One record per target sample.

Recommended fields:

```json
{
  "key": "4d256d78-96af-476f-870d-dc4905e7acca__spk2",
  "mix_id": "4d256d78-96af-476f-870d-dc4905e7acca",
  "target_role": "spk2",
  "target_index": 1,
  "num_sources": 3,
  "mix_wav": "/scratch/project_465002316/junyi_data/wav/4d256d78-96af-476f-870d-dc4905e7acca.wav",
  "source_wavs": [
    "/tmp/librispeech/LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac",
    "/tmp/librispeech/LibriSpeech/train-clean-360/100/121669/100-121669-0000.flac",
    "/tmp/librispeech/LibriSpeech/train-other-500/1006/135212/1006-135212-0000.flac"
  ],
  "speaker_ids": ["103", "100", "1006"],
  "storage": "lmdb",
  "lmdb_path": "aux_labels.lmdb",
  "lmdb_key": "4d256d78-96af-476f-870d-dc4905e7acca__spk2"
}
```

## Alignment convention

The labels should be aligned to the **mixture token timeline used by the router**, not raw waveform samples.

That means:

- labels only cover the `mix` segment
- they do **not** include `enroll` or `sil`
- length is `T_mix`, where `T_mix` matches the model-side mixture token length

This is important because the router lives on mixture hidden states after the audio frontend.

## How to derive the labels

Given one target sample:

1. read all clean source waveforms from `wav.scp`
2. compute per-source activity using short-time energy or VAD
3. align source activity to the same token-rate timeline used for `mix`
4. choose target by `target_role`
5. compute:
   - `target_active`
   - `interferer_count`
   - `active_speaker_count`
   - `router_label`

## Recommended class construction rule

Let:

- `t = target_active`
- `k = interferer_count`

Then:

- if `t == 0` and `k == 0` -> `router_label = 0` (`silence`)
- if `t == 1` and `k == 0` -> `router_label = 1` (`target_only`)
- if `t == 1` and `k >= 1` -> `router_label = 2` (`target_overlap`)
- if `t == 0` and `k >= 1` -> `router_label = 3` (`non_target_only`)

## Why this format is recommended

This format is good for the current project because:

- it matches the target-centric TS-ASR training setup
- it supports 2 to 6 speakers naturally
- it is enough for overlap auxiliary loss
- it is enough for routing analysis
- it preserves richer source activity for future experiments

## Minimal training usage

The first auxiliary loss can use only:

- `router_label`

The stronger version can additionally use:

- `target_active`
- `interferer_count`
- `source_activity`

## Minimal next implementation

The next practical step is:

1. build `aux_labels.lmdb`
2. build `aux_label_manifest.jsonl`
3. load `router_label` from LMDB in the TS-ASR dataset
4. add `L_overlap`

That is enough to support the first overlap-specific auxiliary loss.
