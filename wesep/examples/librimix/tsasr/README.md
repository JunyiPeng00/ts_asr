# TS-ASR Data Preparation

This directory contains the data-preparation and training entry points for the
`DynaTaR-Qwen` target-speaker ASR recipe built on top of LibriMix/WeSep style
metadata and Qwen3-ASR.

The current recipe uses the `max` version of LibriMix audio by default:

- `./Libri2Mix/wav16k/max/`

The prepared metadata and generated shards live under:

- `./data/clean/`

## Environment

Before running Python in this recipe, use:

```bash
module purge
module load LUMI
module load PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250404
```

Useful helpers:

```bash
source ./path.sh
source ./lumi.sh
```

## Data Flow

The current TS-ASR data pipeline is:

1. Build LibriMix metadata from `max` audio.
2. Build enrollment lookup files.
3. Build ASR-aware shards that contain `mix + target wav + target_text + target speaker meta`.

The important design choice is:

- `enroll` handling stays consistent with the original WeSep recipe.
- `target_text` and `target wav` are added for TS-ASR.

That means:

- `train-100/train-360` enrollments are selected dynamically from `spk2enroll.json`
- `dev/test` enrollments come from `spk1.enroll` and `spk2.enroll`
- shards store `target_text` and `targetwav`, but do not store `enroll_wav`

## Scripts

### `local/prepare_data.sh`

Main orchestration script.

It currently implements:

- `stage 1`: create base LibriMix metadata
- `stage 2`: create enrollment-related files
- `stage 3`: create ASR-aware target-centric shards
- `stage 4`: download pretrained speaker encoders
- `stage 6`: prepare MUSAN metadata

Current defaults:

- `mix_data_path=./Libri2Mix/wav16k/max/`
- `num_utts_per_shard=200`
- `train360_num_utts_per_shard=800`
- `shard_dsets="train-100 train-360 dev test"`

### `local/prepare_spk2enroll_librispeech.py`

Builds `spk2enroll.json`.

Output meaning:

- speaker id -> list of candidate enrollment utterances

In this recipe it is called with `--is_librimix True`, so the enrollment pool is
collected from LibriMix clean source utterances rather than external LibriSpeech
FLAC files.

### `local/prepare_librimix_enroll.py`

Builds fixed enrollment mapping files for evaluation splits.

Outputs:

- `spk1.enroll`
- `spk2.enroll`

Behavior:

- `train=True`: write symbolic training-time enrollment hints
- `train=False`: resolve fixed enrollment utterances for each mixture

In the current pipeline we use it for `dev/test` with:

- `mixture2enrollment`
- `spk2enroll.json`

### `local/make_shard_list_asr.py`

Builds the current ASR-aware shards.

Inputs:

- `wav.scp`
- `utt2spk`
- `libri2mix_clean_<split>_spk1_text`
- `libri2mix_clean_<split>_spk2_text`

Each original mixture is expanded into two target-centric records:

- `<mix_id>__spk1`
- `<mix_id>__spk2`

Each tar sample contains:

- `<record_key>.wav`: mixture waveform
- `<record_key>.targetwav`: target clean reference waveform
- `<record_key>.txt`: target transcript
- `<record_key>.spk`: target speaker id
- `<record_key>.role`: `spk1` or `spk2`
- `<record_key>.mixid`: original mixture id

This is intentionally different from the old WeSep premix shard format.

## Stage 1 Outputs

`stage 1` creates the base LibriMix metadata under `data/clean/<split>/`.

Important files:

- `wav.scp`
- `utt2spk`
- `single.wav.scp`
- `single.utt2spk`

Meaning:

- `wav.scp`: `mix_id mix.wav s1.wav s2.wav`
- `utt2spk`: `mix_id spk1_id spk2_id`
- `single.wav.scp`: source clean utterance path lookup
- `single.utt2spk`: speaker id for each `single.wav.scp` item

## Stage 2 Outputs

`stage 2` creates enrollment metadata.

For all splits:

- `spk2enroll.json`

For `dev/test`:

- `spk1.enroll`
- `spk2.enroll`

Enrollment behavior matches the older WeSep shard pipeline:

- train uses speaker pools from `spk2enroll.json`
- eval uses fixed mapping files plus `single.wav.scp`

## Stage 3 Outputs

`stage 3` creates ASR-aware shards under:

- `data/clean/<split>/shards/`
- `data/clean/<split>/shard.list`

The script expects text files named:

- `libri2mix_clean_<split>_spk1_text`
- `libri2mix_clean_<split>_spk2_text`

Current shard sizes:

- `dev/test/train-100`: 200 utterances per shard
- `train-360`: 800 utterances per shard

Because each mixture is expanded into two target-centric records, these settings
give the same shard counts as the earlier premix recipe:

- `dev`: 30
- `test`: 30
- `train-100`: 139
- `train-360`: 127

## Custom Training Splits

The current `run.sh` also supports prebuilt custom training splits, including:

- `train-3mix-custom`
- `train-3mix-360`

These custom splits are treated as training-only datasets:

- they already need to contain `shard.list`, `spk2enroll.json`, and `single.wav.scp`
- `stage 1` and `stage 3` do not rebuild them automatically
- use them with `stage 7` training while keeping `eval_split=dev` or `eval_split=test`

Example:

```bash
cd /scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr

bash run.sh \
  --config confs/dynatar_qwen_0p6b_train3mix_custom.yaml \
  --stage 7 \
  --stop_stage 7
```

## Example Commands

### Build metadata and enrollments

```bash
cd /scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr

bash local/prepare_data.sh \
  --stage 1 \
  --stop_stage 2 \
  --data /scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr/data \
  --noise_type clean
```

### Build ASR-aware shards

```bash
cd /scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr

bash local/prepare_data.sh \
  --stage 3 \
  --stop_stage 3 \
  --data /scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr/data \
  --noise_type clean \
  --num_utts_per_shard 200 \
  --train360_num_utts_per_shard 800 \
  --num_threads 16
```

## Training Inputs

The training entry point is:

- `train_ts_qwen3_asr.py`

It uses one dataset mode:

- `--data_type shard`

Shard mode reads:

- `shard.list`
- `target_text` from shard tar members
- `targetwav` from shard tar members

and resolves enrollments externally:

- train from `--train_spk2utt`
- eval from `--eval_spk1_enroll`, `--eval_spk2_enroll`, `--eval_spk2utt`

## Current Data Semantics

This recipe currently uses a target-centric TS-ASR sample definition.

One mixture produces two training samples:

- target speaker 1 transcript
- target speaker 2 transcript

That matches the TS-ASR objective directly:

- input: `mix + enroll`
- output: transcript of one designated target speaker

## Notes

- This README documents the current `max`-audio, shard-only TS-ASR pipeline.
- The current shard format is TS-ASR specific and is not the same as
  `tools/make_shard_list_premix.py`.
- Enrollment lookup stays aligned with the old WeSep idea even though the shard
  format is new.
