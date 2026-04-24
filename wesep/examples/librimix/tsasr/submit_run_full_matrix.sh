#!/bin/bash

set -euo pipefail

WORK_DIR=/scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr
cd "${WORK_DIR}"

echo "submit_run_full_matrix.sh submits the entire current TASLP round."
echo "Delegating to: ${WORK_DIR}/submit_run_taslp.sh full_matrix"
echo "RUN_STAGE=${RUN_STAGE:-7}"
echo "STOP_STAGE=${STOP_STAGE:-7}"
echo "DRY_RUN=${DRY_RUN:-0}"

exec bash "${WORK_DIR}/submit_run_taslp.sh" full_matrix
