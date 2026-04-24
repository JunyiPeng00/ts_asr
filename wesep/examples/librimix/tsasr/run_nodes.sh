#!/bin/bash

. ./path.sh || exit 1
. ./lumi.sh || exit 1

script_dir=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${script_dir}" || exit 1

# General configuration
stage=7
stop_stage=7

# Distributed launch configuration
nnodes=${SLURM_NNODES:-1}
node_rank=${SLURM_NODEID:-0}
master_addr=""
master_port=29500
launcher_mode=auto

# Data preparation related
data=data
noise_type=clean
text_dir=""

# Model and experiment related
config=confs/dynatar_qwen_0p6b.yaml
qwen_model_size=0.6B
qwen_models_dir=/scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr/qwen_models
model_path=""
exp_dir=exp/dynatar_qwen_0p6b_shard_train-100

# Training related
train_split=train-100
eval_split=dev
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
new_module_prefixes=thinker.audio_tower.mhfa_backend,thinker.audio_tower.prototype_projector,thinker.audio_tower.refinement_blocks
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

set_visible_devices() {
  local visible="$1"
  export HIP_VISIBLE_DEVICES="${visible}"
  export ROCR_VISIBLE_DEVICES="${visible}"
  export CUDA_VISIBLE_DEVICES="${visible}"
}

primary_gpu() {
  echo "${gpus}" | awk -F ',' '{print $1}'
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
  else
    echo "Missing local model directory: ${local_model_path}" >&2
    echo "Run stage 2 in run.sh first, or set --model_path to a local downloaded model directory." >&2
    exit 1
  fi
fi

if [ -z "${master_addr}" ]; then
  if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    master_addr=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
  else
    master_addr="127.0.0.1"
  fi
fi

if [ "${stage}" -ne 7 ] || [ "${stop_stage}" -ne 7 ]; then
  echo "run_nodes.sh only supports stage 7 multi-node training." >&2
  echo "Use run.sh for data prep, smoke test, inference, and scoring stages." >&2
  exit 1
fi

echo "Start multi-node training ..."
echo "nnodes=${nnodes} node_rank=${node_rank} master_addr=${master_addr} master_port=${master_port}"
echo "gpus=${gpus}"
echo "launcher_mode=${launcher_mode}"

num_gpus=$(echo "${gpus}" | awk -F ',' '{print NF}')
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

assert_file_exists "${train_file}" "training shard list"
assert_file_exists "${train_spk2utt}" "training enrollment pool"
assert_file_exists "${train_single_wav_scp}" "training single.wav.scp"
assert_file_exists "${eval_file}" "evaluation shard list"
assert_file_exists "${eval_spk1_enroll}" "evaluation spk1 enrollment map"
assert_file_exists "${eval_spk2_enroll}" "evaluation spk2 enrollment map"
assert_file_exists "${eval_spk2utt}" "evaluation single.wav.scp"

run_training() {
  python -u train_ts_qwen3_asr.py \
    --model_path "${model_path}" \
    --data_type shard \
    --train_file "${train_file}" \
    --eval_file "${eval_file}" \
    --train_spk2utt "${train_spk2utt}" \
    --train_single_wav_scp "${train_single_wav_scp}" \
    --eval_spk1_enroll "${eval_spk1_enroll}" \
    --eval_spk2_enroll "${eval_spk2_enroll}" \
    --eval_spk2utt "${eval_spk2utt}" \
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
    "${resume_args[@]}" \
    "$@"
}

direct_lumi_mode=0
if [ "${launcher_mode}" = "lumi_srun" ]; then
  direct_lumi_mode=1
elif [ "${launcher_mode}" = "auto" ] && [ -n "${SLURM_LOCALID:-}" ] && [ "${SLURM_NTASKS_PER_NODE:-1}" != "1" ]; then
  direct_lumi_mode=1
fi

if [ "${direct_lumi_mode}" -eq 1 ]; then
  export MASTER_ADDR="${master_addr}"
  export MASTER_PORT="${master_port}"
  export WORLD_SIZE="${SLURM_NTASKS}"
  export RANK="${SLURM_PROCID}"
  export LOCAL_RANK="${SLURM_LOCALID}"
  export LOCAL_WORLD_SIZE="${num_gpus}"
  echo "Using direct srun task-per-GPU launch: rank=${RANK} local_rank=${LOCAL_RANK} world_size=${WORLD_SIZE}"
  run_training --local_rank "${SLURM_LOCALID}"
else
  set_visible_devices "${gpus}"
  if [ "${num_gpus}" -gt 1 ] || [ "${nnodes}" -gt 1 ]; then
    train_launcher="python -u -m torch.distributed.launch \
      --nproc_per_node=${num_gpus} \
      --nnodes=${nnodes} \
      --node_rank=${node_rank} \
      --master_addr=${master_addr} \
      --master_port=${master_port} \
      --use_env"
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
    --eval_spk2utt "${eval_spk2utt}" \
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
