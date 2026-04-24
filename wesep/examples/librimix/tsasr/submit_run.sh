#!/bin/bash

set -euo pipefail

WORK_DIR=/scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr
cd "${WORK_DIR}"

group="${1:-this_round}"

echo "submit_run.sh is kept as a compatibility wrapper."
echo "Preferred entrypoint: ${WORK_DIR}/submit_run_taslp.sh"
echo "Forwarding group=${group}"

exec bash "${WORK_DIR}/submit_run_taslp.sh" "${group}"
