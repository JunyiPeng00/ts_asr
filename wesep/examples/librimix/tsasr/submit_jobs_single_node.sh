#!/bin/bash
#SBATCH -J TSASR_single_node
#SBATCH -p standard-g
#SBATCH --account=project_465002316
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --time=47:59:00
#SBATCH --output=log/output_%x_%j.txt
#SBATCH --error=log/error_%x_%j.txt
#SBATCH --array=1-2
#SBATCH --exclude=nid005955


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

echo "job_id=${SLURM_JOB_ID}"
echo "job_nodelist=${SLURM_JOB_NODELIST}"
echo "nnodes=${SLURM_NNODES}"
echo "config=${config}"
echo "launcher_mode=single_node_run_sh"
echo "TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG}"
echo "NCCL_DEBUG=${NCCL_DEBUG}"
echo "TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING}"

srun singularity exec "$SIF" bash run.sh \
    --config "confs/${config}.yaml" \
    --stage 7 \
    --stop_stage 7
