#!/bin/bash

. ./path.sh || exit 1
. ./lumi.sh || exit 1

script_dir=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${script_dir}" || exit 1

# General configuration
stage=-1
stop_stage=-1

# Data preparation related
data=data
noise_type=clean
num_utts_per_shard=200
train360_num_utts_per_shard=800
num_threads=16
text_dir=""
mix_data_path=/flash/project_465002316/junyi/Libri2Mix/wav16k/max

# Model and experiment related
config=confs/dynatar_qwen_0p6b.yaml
qwen_model_size=0.6B
download_model_sizes="0.6B 1.7B"
qwen_models_dir=/scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr/qwen_models
model_path=""
exp_dir=exp/dynatar_qwen_0p6b_shard_train-100
checkpoint=""

# Training related
train_split=train-100
eval_split=dev
test_split=test
eval_enroll_paths_json=""
test_enroll_paths_json=""
gpus=0
language=English
prompt=""
silence_seconds=1.0
enroll_crop_seconds=4.0
role_layers=5
num_prototypes=8
mhfa_heads=4
mhfa_compression_dim=128
router_hidden_multiplier=2
refinement_variant=full
shared_refinement=0
memory_dim=256
expert_rank=64
num_experts=3
refinement_layer_strategy=post_role_all
enable_overlap_head=0
overlap_num_classes=4
overlap_loss_weight=0.10
enable_target_consistency=0
target_consistency_weight=0.05
target_consistency_mode=hybrid
target_consistency_temperature=0.07
target_consistency_detach_target=1
enable_router_supervision=0
router_loss_weight=0.02
train_aux_label_manifest=""
eval_aux_label_manifest=""
batch_size=1
eval_batch_size=1
grad_acc=12
lr=4e-6
warmup_steps=10000
lr_scheduler_type=cosine
max_steps=-1
epochs=1
log_steps=10
eval_steps=1000
save_steps=1000
save_total_limit=5
num_workers=4
prefetch_factor=2
pin_memory=1
weight_decay=0.0
max_grad_norm=1.0
group_by_length=1
use_custom_optimizer=1
new_module_lr_multiplier=50.0
new_module_prefixes=thinker.audio_tower.mhfa_backend,thinker.audio_tower.prototype_projector,thinker.audio_tower.refinement_blocks,thinker.audio_tower.mix_summary_attn,thinker.audio_tower.mix_summary_proj,thinker.audio_tower.overlap_head
seed=42
max_train_samples=0
max_eval_samples=0
gradient_checkpointing=1
save_strategy=steps
eval_strategy=steps
report_to=tensorboard
logging_dir=""
resume=0
resume_from=""
decode_max_new_tokens=256
decode_max_samples=0
decode_batch_size=1
decode_num_workers=0

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

load_yaml_config() {
  local yaml_file="$1"
  local raw_line=""
  local line=""
  local key=""
  local value=""
  while IFS= read -r raw_line || [ -n "${raw_line}" ]; do
    line="${raw_line%%#*}"
    line=$(trim "${line}")
    [ -z "${line}" ] && continue
    case "${line}" in
      *:*)
        key=$(trim "${line%%:*}")
        value=$(trim "${line#*:}")
        if [[ "${value}" =~ ^\".*\"$ ]] || [[ "${value}" =~ ^\'.*\'$ ]]; then
          value="${value:1:${#value}-2}"
        fi
        printf -v "${key}" '%s' "${value}"
        ;;
      *)
        echo "Unsupported YAML line in ${yaml_file}: ${raw_line}" >&2
        exit 1
        ;;
    esac
  done < "${yaml_file}"
}

for arg in "$@"; do
  case "${arg}" in
    --config=*)
      config="${arg#*=}"
      ;;
  esac
done

for ((i = 1; i <= $#; i++)); do
  if [ "${!i}" = "--config" ]; then
    next=$((i + 1))
    if [ "${next}" -le "$#" ]; then
      config="${!next}"
    fi
  fi
done

if [[ "${config}" != /* ]]; then
  config="${script_dir}/${config}"
fi
if [ ! -f "${config}" ]; then
  echo "Missing config file: ${config}" >&2
  exit 1
fi

load_yaml_config "${config}"
. tools/parse_options.sh || exit 1

mkdir -p "${data}" "${qwen_models_dir}"
data=$(realpath "${data}")
qwen_models_dir=$(realpath -m "${qwen_models_dir}")
exp_root=$(realpath -m "${script_dir}/exp")
mkdir -p "${exp_root}"
standard_prepare_train_splits="train-100 train-360"
custom_ready_train_splits="train-3mix-custom train-3mix-360 train-merge"

if [ -z "${text_dir}" ]; then
  text_dir="${data}/${noise_type}"
fi

normalize_exp_path() {
  local path="$1"
  local resolved=""
  if [ -z "${path}" ]; then
    echo ""
    return 0
  fi
  if [[ "${path}" = /* ]]; then
    resolved=$(realpath -m "${path}")
  else
    path="${path#./}"
    if [[ "${path}" == exp/* ]]; then
      resolved=$(realpath -m "${script_dir}/${path}")
    else
      resolved=$(realpath -m "${exp_root}/${path}")
    fi
  fi
  case "${resolved}" in
    "${exp_root}"|${exp_root}/*)
      echo "${resolved}"
      ;;
    *)
      echo "Experiment path must stay under ${exp_root}: ${resolved}" >&2
      exit 1
      ;;
  esac
}

assert_file_exists() {
  local path="$1"
  local desc="$2"
  if [ ! -f "${path}" ]; then
    echo "Missing ${desc}: ${path}" >&2
    exit 1
  fi
}

contains_word() {
  local needle="$1"
  shift
  local item=""
  for item in "$@"; do
    if [ "${item}" = "${needle}" ]; then
      return 0
    fi
  done
  return 1
}

exp_dir=$(normalize_exp_path "${exp_dir}") || exit 1
mkdir -p "${exp_dir}"
if [ -z "${logging_dir}" ]; then
  logging_dir="${exp_dir}"
else
  logging_dir=$(normalize_exp_path "${logging_dir}") || exit 1
  mkdir -p "${logging_dir}"
fi

if [ -z "${model_path}" ]; then
  local_model_path="${qwen_models_dir}/Qwen3-ASR-${qwen_model_size}"
  if [ -d "${local_model_path}" ]; then
    model_path="${local_model_path}"
  elif [ "${stop_stage}" -ge 5 ]; then
    echo "Missing local model directory: ${local_model_path}" >&2
    echo "Run stage 2 first, or set --model_path to a local downloaded model directory." >&2
    exit 1
  else
    model_path="${local_model_path}"
  fi
fi

set_visible_devices() {
  local visible="$1"
  export HIP_VISIBLE_DEVICES="${visible}"
  export ROCR_VISIBLE_DEVICES="${visible}"
  export CUDA_VISIBLE_DEVICES="${visible}"
}

primary_gpu() {
  echo "${gpus}" | awk -F ',' '{print $1}'
}

find_latest_checkpoint_dir() {
  local base_dir="$1"
  local latest=""
  latest=$(find "${base_dir}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)
  if [ -n "${latest}" ]; then
    echo "${latest}"
  fi
}

resolve_decode_checkpoint() {
  if [ -n "${checkpoint}" ]; then
    if [[ "${checkpoint}" = /* ]]; then
      echo "$(realpath -m "${checkpoint}")"
    else
      echo "$(realpath -m "${exp_dir}/${checkpoint}")"
    fi
    return 0
  fi
  find_latest_checkpoint_dir "${exp_dir}"
}

infer_output_dir() {
  local resolved_checkpoint="$1"
  local tag="base_model"
  if [ -n "${resolved_checkpoint}" ]; then
    tag=$(basename "${resolved_checkpoint}")
  fi
  echo "${exp_dir}/infer_${test_split}_${tag}"
}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare LibriMix metadata and enrollment files ..."
  if contains_word "${train_split}" ${custom_ready_train_splits}; then
    echo "Note: ${train_split} is treated as a prebuilt custom training split."
    echo "Stage 1 only refreshes the standard Libri2Mix metadata (${standard_prepare_train_splits})."
    echo "For ${train_split}, use stage 7 directly after its shard/enrollment files are ready."
  fi
  bash ./local/prepare_data.sh \
    --mix_data_path "${mix_data_path}" \
    --data "${data}" \
    --noise_type "${noise_type}" \
    --text_dir "${text_dir}" \
    --num_utts_per_shard "${num_utts_per_shard}" \
    --train360_num_utts_per_shard "${train360_num_utts_per_shard}" \
    --num_threads "${num_threads}" \
    --stage 1 \
    --stop_stage 2
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Download Qwen3-ASR model weights ..."
  bash ./local/download_qwen_models.sh \
    --target_root "${qwen_models_dir}" \
    --models "${download_model_sizes}"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Build ASR-aware shards ..."
  if contains_word "${train_split}" ${custom_ready_train_splits}; then
    echo "Note: ${train_split} is treated as a prebuilt custom training split."
    echo "Stage 3 only rebuilds the standard Libri2Mix shard sets (${standard_prepare_train_splits} dev test)."
    echo "If you want to train on ${train_split}, its shards must already exist under data/${noise_type}/${train_split}."
  fi
  bash ./local/prepare_data.sh \
    --mix_data_path "${mix_data_path}" \
    --data "${data}" \
    --noise_type "${noise_type}" \
    --text_dir "${text_dir}" \
    --num_utts_per_shard "${num_utts_per_shard}" \
    --train360_num_utts_per_shard "${train360_num_utts_per_shard}" \
    --num_threads "${num_threads}" \
    --stage 3 \
    --stop_stage 3
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Build overlap auxiliary labels ..."
  split_dir="${data}/${noise_type}/${train_split}"
  if [ "${train_split}" = "train-merge" ]; then
    merge_manifests=()
    for source_split in train-100 train-360 train-3mix-360 train-3mix-custom; do
      source_manifest="${data}/${noise_type}/${source_split}/aux_label_manifest.jsonl"
      assert_file_exists "${source_manifest}" "source auxiliary label manifest for ${source_split}"
      merge_manifests+=("${source_manifest}")
    done
    python ./local/merge_aux_label_manifests.py \
      --output_dir "${split_dir}" \
      "${merge_manifests[@]}"
  else
    wav_scp="${split_dir}/wav.scp"
    utt2spk="${split_dir}/utt2spk"
    assert_file_exists "${wav_scp}" "wav.scp for overlap labels"
    assert_file_exists "${utt2spk}" "utt2spk for overlap labels"
    build_aux_args=(
      --wav_scp "${wav_scp}"
      --utt2spk "${utt2spk}"
      --output_dir "${split_dir}"
      --sr 16000
    )
    custom_cutset="${split_dir}/librispeechmix_custom_cutset_train-3mix.jsonl.gz"
    if [ "${train_split}" = "train-3mix-custom" ] && [ -f "${custom_cutset}" ]; then
      build_aux_args+=(--cutset_jsonl_gz "${custom_cutset}")
    fi
    python ./local/build_overlap_labels.py "${build_aux_args[@]}"
  fi
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run unit and smoke tests ..."
  set_visible_devices "$(primary_gpu)"
  python -u tests/test_ts_qwen3_asr.py
  python -u smoke_test.py
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Run full-chain smoke on real data ..."
  set_visible_devices "$(primary_gpu)"
  export TSASR_MODEL_PATH="${model_path}"
  python -u tests/full_chain_qwen_tsasr_smoke.py
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Start training ..."
  num_gpus=$(echo "${gpus}" | awk -F ',' '{print NF}')
  set_visible_devices "${gpus}"
  resume_args=()
  if [ -n "${resume_from}" ]; then
    resume_args=(--resume_from "${resume_from}")
  fi

  train_file="${data}/${noise_type}/${train_split}/shard.list"
  eval_file="${data}/${noise_type}/${eval_split}/shard.list"
  train_spk2utt="${data}/${noise_type}/${train_split}/spk2enroll.json"
  train_single_wav_scp="${data}/${noise_type}/${train_split}/single.wav.scp"
  eval_spk1_enroll="${data}/${noise_type}/${eval_split}/spk1.enroll"
  eval_spk2_enroll="${data}/${noise_type}/${eval_split}/spk2.enroll"
  eval_spk2utt="${data}/${noise_type}/${eval_split}/single.wav.scp"
  if [ -z "${train_aux_label_manifest}" ] && { [ "${enable_overlap_head}" -eq 1 ] || [ "${enable_router_supervision}" -eq 1 ]; }; then
    train_aux_label_manifest="${data}/${noise_type}/${train_split}/aux_label_manifest.jsonl"
  fi
  if [ -z "${eval_aux_label_manifest}" ] && [ -f "${data}/${noise_type}/${eval_split}/aux_label_manifest.jsonl" ]; then
    eval_aux_label_manifest="${data}/${noise_type}/${eval_split}/aux_label_manifest.jsonl"
  fi

  assert_file_exists "${train_file}" "training shard list"
  assert_file_exists "${train_spk2utt}" "training enrollment pool"
  assert_file_exists "${train_single_wav_scp}" "training single.wav.scp"
  assert_file_exists "${eval_file}" "evaluation shard list"
  if [ -n "${train_aux_label_manifest}" ]; then
    assert_file_exists "${train_aux_label_manifest}" "training auxiliary label manifest"
  fi
  if [ -n "${eval_aux_label_manifest}" ]; then
    assert_file_exists "${eval_aux_label_manifest}" "evaluation auxiliary label manifest"
  fi
  if [ -n "${eval_enroll_paths_json}" ]; then
    assert_file_exists "${eval_enroll_paths_json}" "evaluation enrollment map json"
  else
    assert_file_exists "${eval_spk1_enroll}" "evaluation spk1 enrollment map"
    assert_file_exists "${eval_spk2_enroll}" "evaluation spk2 enrollment map"
    assert_file_exists "${eval_spk2utt}" "evaluation single.wav.scp"
  fi

  if [ "${num_gpus}" -gt 1 ]; then
    train_launcher="python -u -m torch.distributed.launch --nproc_per_node=$num_gpus"
  else
    train_launcher="python -u"
  fi

  ${train_launcher} train_ts_qwen3_asr.py \
    --model_path "${model_path}" \
    --data_type shard \
    --train_file "${train_file}" \
    --eval_file "${eval_file}" \
    --train_spk2utt "${train_spk2utt}" \
    --train_single_wav_scp "${train_single_wav_scp}" \
    --eval_spk1_enroll "${eval_spk1_enroll}" \
    --eval_spk2_enroll "${eval_spk2_enroll}" \
    --eval_enroll_paths_json "${eval_enroll_paths_json}" \
    --eval_spk2utt "${eval_spk2utt}" \
    --train_aux_label_manifest "${train_aux_label_manifest}" \
    --eval_aux_label_manifest "${eval_aux_label_manifest}" \
    --output_dir "${exp_dir}" \
    --sr 16000 \
    --silence_seconds "${silence_seconds}" \
    --enroll_crop_seconds "${enroll_crop_seconds}" \
    --language "${language}" \
    --prompt "${prompt}" \
    --role_layers "${role_layers}" \
    --num_prototypes "${num_prototypes}" \
    --mhfa_heads "${mhfa_heads}" \
    --mhfa_compression_dim "${mhfa_compression_dim}" \
    --router_hidden_multiplier "${router_hidden_multiplier}" \
    --refinement_variant "${refinement_variant}" \
    --shared_refinement "${shared_refinement}" \
    --memory_dim "${memory_dim}" \
    --expert_rank "${expert_rank}" \
    --num_experts "${num_experts}" \
    --refinement_layer_strategy "${refinement_layer_strategy}" \
    --enable_overlap_head "${enable_overlap_head}" \
    --overlap_num_classes "${overlap_num_classes}" \
    --overlap_loss_weight "${overlap_loss_weight}" \
    --enable_target_consistency "${enable_target_consistency}" \
    --target_consistency_weight "${target_consistency_weight}" \
    --target_consistency_mode "${target_consistency_mode}" \
    --target_consistency_temperature "${target_consistency_temperature}" \
    --target_consistency_detach_target "${target_consistency_detach_target}" \
    --enable_router_supervision "${enable_router_supervision}" \
    --router_loss_weight "${router_loss_weight}" \
    --batch_size "${batch_size}" \
    --eval_batch_size "${eval_batch_size}" \
    --grad_acc "${grad_acc}" \
    --lr "${lr}" \
    --warmup_steps "${warmup_steps}" \
    --lr_scheduler_type "${lr_scheduler_type}" \
    --max_steps "${max_steps}" \
    --epochs "${epochs}" \
    --log_steps "${log_steps}" \
    --eval_steps "${eval_steps}" \
    --save_steps "${save_steps}" \
    --save_total_limit "${save_total_limit}" \
    --num_workers "${num_workers}" \
    --prefetch_factor "${prefetch_factor}" \
    --pin_memory "${pin_memory}" \
    --weight_decay "${weight_decay}" \
    --max_grad_norm "${max_grad_norm}" \
    --group_by_length "${group_by_length}" \
    --use_custom_optimizer "${use_custom_optimizer}" \
    --new_module_lr_multiplier "${new_module_lr_multiplier}" \
    --new_module_prefixes "${new_module_prefixes}" \
    --seed "${seed}" \
    --max_train_samples "${max_train_samples}" \
    --max_eval_samples "${max_eval_samples}" \
    --gradient_checkpointing "${gradient_checkpointing}" \
    --save_strategy "${save_strategy}" \
    --eval_strategy "${eval_strategy}" \
    --report_to "${report_to}" \
    --logging_dir "${logging_dir}" \
    --freeze_text_decoder 1 \
    --resume "${resume}" \
    "${resume_args[@]}"
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Run inference ..."
  set_visible_devices "$(primary_gpu)"

  resolved_checkpoint=$(resolve_decode_checkpoint)
  infer_dir=$(infer_output_dir "${resolved_checkpoint}")
  mkdir -p "${infer_dir}"

  eval_file="${data}/${noise_type}/${test_split}/shard.list"
  eval_spk1_enroll="${data}/${noise_type}/${test_split}/spk1.enroll"
  eval_spk2_enroll="${data}/${noise_type}/${test_split}/spk2.enroll"
  eval_spk2utt="${data}/${noise_type}/${test_split}/single.wav.scp"
  if [ -n "${test_enroll_paths_json}" ]; then
    assert_file_exists "${test_enroll_paths_json}" "test enrollment map json"
  fi

  python infer_ts_qwen3_asr.py \
    --model_path "${model_path}" \
    --checkpoint "${resolved_checkpoint}" \
    --data_type shard \
    --eval_file "${eval_file}" \
    --eval_spk1_enroll "${eval_spk1_enroll}" \
    --eval_spk2_enroll "${eval_spk2_enroll}" \
    --eval_enroll_paths_json "${test_enroll_paths_json}" \
    --eval_spk2utt "${eval_spk2utt}" \
    --output_dir "${infer_dir}" \
    --sr 16000 \
    --silence_seconds "${silence_seconds}" \
    --enroll_crop_seconds "${enroll_crop_seconds}" \
    --language "${language}" \
    --prompt "${prompt}" \
    --role_layers "${role_layers}" \
    --num_prototypes "${num_prototypes}" \
    --mhfa_heads "${mhfa_heads}" \
    --mhfa_compression_dim "${mhfa_compression_dim}" \
    --router_hidden_multiplier "${router_hidden_multiplier}" \
    --refinement_variant "${refinement_variant}" \
    --shared_refinement "${shared_refinement}" \
    --memory_dim "${memory_dim}" \
    --expert_rank "${expert_rank}" \
    --num_experts "${num_experts}" \
    --refinement_layer_strategy "${refinement_layer_strategy}" \
    --enable_overlap_head "${enable_overlap_head}" \
    --overlap_num_classes "${overlap_num_classes}" \
    --overlap_loss_weight "${overlap_loss_weight}" \
    --enable_target_consistency "${enable_target_consistency}" \
    --target_consistency_weight "${target_consistency_weight}" \
    --target_consistency_mode "${target_consistency_mode}" \
    --target_consistency_temperature "${target_consistency_temperature}" \
    --target_consistency_detach_target "${target_consistency_detach_target}" \
    --enable_router_supervision "${enable_router_supervision}" \
    --router_loss_weight "${router_loss_weight}" \
    --batch_size "${decode_batch_size}" \
    --num_workers "${decode_num_workers}" \
    --max_samples "${decode_max_samples}" \
    --max_new_tokens "${decode_max_new_tokens}"
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  echo "Score WER/CER ..."
  resolved_checkpoint=$(resolve_decode_checkpoint)
  infer_dir=$(infer_output_dir "${resolved_checkpoint}")
  results_jsonl="${infer_dir}/results.jsonl"
  if [ ! -f "${results_jsonl}" ]; then
    echo "Missing inference output: ${results_jsonl}" >&2
    exit 1
  fi
  python score_ts_qwen3_asr.py \
    --results_jsonl "${results_jsonl}" \
    --output_dir "${infer_dir}"
fi
