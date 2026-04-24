#!/bin/bash
# Compatibility wrapper for the current single-node TASLP benchmark runs.
# Preferred submit path: bash submit_run_taslp.sh benchmark_3spk

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

set -euo pipefail

export TASLP_GROUP="${TASLP_GROUP:-benchmark_3spk}"

case "${TASLP_GROUP}" in
  benchmark_3spk|train100_full)
    ;;
  *)
    echo "submit_jobs_single_node.sh only forwards TASLP single-node groups." >&2
    echo "Supported groups: benchmark_3spk, train100_full" >&2
    echo "Received TASLP_GROUP=${TASLP_GROUP}" >&2
    exit 1
    ;;
esac

echo "submit_jobs_single_node.sh is kept as a compatibility wrapper."
echo "Preferred job script: $(dirname "$0")/submit_jobs_taslp_single_node.sh"
echo "Forwarding TASLP_GROUP=${TASLP_GROUP}"

exec bash "$(dirname "$0")/submit_jobs_taslp_single_node.sh"
