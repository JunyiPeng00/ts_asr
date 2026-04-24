#!/bin/bash

set -euo pipefail

time=$(date +"%Y-%m-%d")
echo "Date: ${time}"

WORK_DIR=/scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr
cd "${WORK_DIR}"

my_folder="${WORK_DIR}/log/${time}"
mkdir -p "${my_folder}"

echo "Work dir: ${WORK_DIR}"
echo "Log dir: ${my_folder}"
echo "Submitting: ${WORK_DIR}/submit_jobs_single_node.sh"

sbatch \
  -J "TSASR_single_node" \
  --time "47:59:00" \
  --array "1-2" \
  -o "${my_folder}/output_%x_%j_%a.txt" \
  -e "${my_folder}/error_%x_%j_%a.txt" \
  "${WORK_DIR}/submit_jobs_single_node.sh"
