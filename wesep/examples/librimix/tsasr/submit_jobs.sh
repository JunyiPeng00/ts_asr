#!/bin/bash
#SBATCH -J TSASR_multi_node
#SBATCH -p standard-g
#SBATCH --account=project_465002316
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=47:59:00
#SBATCH --output=log/output_%x_%j.txt
#SBATCH --error=log/error_%x_%j.txt
#SBATCH --array=1-1
#SBATCH --exclude=nid005664

set -euo pipefail

mkdir -p log

module purge
module load LUMI
module load PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250404

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

configs=(
    dynatar_qwen_0p6b_trainmerge
    dynatar_qwen_0p6b_train100
)
config=${configs[$SLURM_ARRAY_TASK_ID-1]}
tasks_per_node=8
cpu_bind_map=49,57,17,25,1,9,33,41

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "job_id=${SLURM_JOB_ID}"
echo "job_nodelist=${SLURM_JOB_NODELIST}"
echo "nnodes=${SLURM_NNODES}"
echo "tasks_per_node=${tasks_per_node}"
echo "master_addr=${MASTER_ADDR}"
echo "master_port=${MASTER_PORT}"
echo "config=${config}"
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
    --stage 7 \
    --stop_stage 7
