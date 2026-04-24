#!/bin/bash
# Canonical distributed TASLP launcher for the current train-merge / train-100
# experiment groups. Submit through submit_run_taslp.sh unless you need to
# manage sbatch arrays manually.

#SBATCH -J TSASR_TASLP
#SBATCH -p standard-g
#SBATCH --account=project_465002316
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
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

taslp_group="${TASLP_GROUP:-trainmerge_ablation}"
case "${taslp_group}" in
  this_round)
    taslp_group="trainmerge_ablation"
    ;;
esac
run_stage="${RUN_STAGE:-7}"
stop_stage="${STOP_STAGE:-7}"
tasks_per_node=8
cpu_bind_map=49,57,17,25,1,9,33,41

configs=()
case "${taslp_group}" in
  trainmerge_ablation)
    configs=(
      taslp/taslp_trainmerge_b0_baseline
      taslp/taslp_trainmerge_b1_overlap
      taslp/taslp_trainmerge_b2_overlap_tc
      taslp/taslp_trainmerge_b3_full
    )
    ;;
  benchmark_2spk)
    configs=(
      taslp/taslp_train100_b0_baseline
      taslp/taslp_train100_b3_full
    )
    ;;
  *)
    echo "Unsupported TASLP_GROUP for multi-node submit_jobs_taslp.sh: ${taslp_group}" >&2
    echo "Supported groups: this_round, trainmerge_ablation, benchmark_2spk" >&2
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
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="${MASTER_PORT:-29500}"

echo "job_id=${SLURM_JOB_ID}"
echo "job_nodelist=${SLURM_JOB_NODELIST}"
echo "nnodes=${SLURM_NNODES}"
echo "tasks_per_node=${tasks_per_node}"
echo "master_addr=${MASTER_ADDR}"
echo "master_port=${MASTER_PORT}"
echo "taslp_group=${taslp_group}"
echo "config=${config}"
echo "run_stage=${run_stage}"
echo "stop_stage=${stop_stage}"
echo "launcher_mode=lumi_srun_task_per_gpu"
echo "cpu_bind_map=${cpu_bind_map}"
echo "TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG}"
echo "NCCL_DEBUG=${NCCL_DEBUG}"
echo "TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING}"

srun \
  --ntasks="$((SLURM_NNODES * tasks_per_node))" \
  --ntasks-per-node="${tasks_per_node}" \
  --cpu-bind="map_cpu:${cpu_bind_map}" \
  --kill-on-bad-exit=1 \
  singularity exec "$SIF" bash run_nodes.sh \
    --config "confs/${config}.yaml" \
    --nnodes "${SLURM_NNODES}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    --launcher_mode lumi_srun \
    --stage "${run_stage}" \
    --stop_stage "${stop_stage}"
