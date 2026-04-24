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
echo "Submitting: ${WORK_DIR}/submit_jobs.sh"

sbatch \
  -o "${my_folder}/output_TSASR_multi_node_%j.txt" \
  -e "${my_folder}/error_TSASR_multi_node_%j.txt" \
  --array "2-2" \
  "${WORK_DIR}/submit_jobs.sh"
