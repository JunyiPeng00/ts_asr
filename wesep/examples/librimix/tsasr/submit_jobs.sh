#!/bin/bash
# Compatibility wrapper for the current distributed TASLP round.
# Preferred submit path: bash submit_run_taslp.sh this_round

#SBATCH -J TSASR_current_round
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

export TASLP_GROUP="${TASLP_GROUP:-this_round}"

echo "submit_jobs.sh is kept as a compatibility wrapper."
echo "Preferred job script: $(dirname "$0")/submit_jobs_taslp.sh"
echo "Forwarding TASLP_GROUP=${TASLP_GROUP}"

exec bash "$(dirname "$0")/submit_jobs_taslp.sh"
