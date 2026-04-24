#!/bin/bash
#SBATCH -J TSASR_TASLP_1N
#SBATCH -p standard-g
#SBATCH --account=project_465002316
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --time=47:59:00
#SBATCH --output=log/output_%x_%j.txt
#SBATCH --error=log/error_%x_%j.txt

set -euo pipefail

mkdir -p log

module purge
module load LUMI
module load PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250404

export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

taslp_group="${TASLP_GROUP:-benchmark_3spk}"
run_stage="${RUN_STAGE:-7}"
stop_stage="${STOP_STAGE:-7}"

configs=()
case "${taslp_group}" in
  benchmark_3spk)
    configs=(
      taslp/taslp_train3mix_custom_b0_baseline
      taslp/taslp_train3mix_custom_b3_full
    )
    ;;
  train100_full)
    configs=(
      dynatar_qwen_0p6b_train100_full
    )
    ;;
  *)
    echo "Unsupported TASLP_GROUP for single-node submit_jobs_taslp_single_node.sh: ${taslp_group}" >&2
    echo "Supported groups: benchmark_3spk, train100_full" >&2
    exit 1
    ;;
esac

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "SLURM_ARRAY_TASK_ID is not set. Submit this script through sbatch --array." >&2
  exit 1
fi

config_index=$((SLURM_ARRAY_TASK_ID - 1))
if [ "${config_index}" -lt 0 ] || [ "${config_index}" -ge "${#configs[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is out of range for group ${taslp_group}" >&2
  exit 1
fi

config="${configs[${config_index}]}"

echo "job_id=${SLURM_JOB_ID}"
echo "job_nodelist=${SLURM_JOB_NODELIST}"
echo "nnodes=${SLURM_NNODES}"
echo "taslp_group=${taslp_group}"
echo "config=${config}"
echo "run_stage=${run_stage}"
echo "stop_stage=${stop_stage}"
echo "launcher_mode=single_node_run_sh"
echo "TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG}"
echo "NCCL_DEBUG=${NCCL_DEBUG}"
echo "TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING}"

srun singularity exec "$SIF" bash run.sh \
  --config "confs/${config}.yaml" \
  --stage "${run_stage}" \
  --stop_stage "${stop_stage}"
