#!/bin/bash

set -euo pipefail

time=$(date +"%Y-%m-%d")
echo "Date: ${time}"

WORK_DIR=/scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr
cd "${WORK_DIR}"

log_dir="${WORK_DIR}/log/${time}"
mkdir -p "${log_dir}"

taslp_group="${1:-trainmerge_ablation}"
run_stage="${RUN_STAGE:-7}"
stop_stage="${STOP_STAGE:-7}"

submit_group() {
  local group="$1"
  local script_path=""
  local array_spec=""
  local job_name=""

  case "${group}" in
    trainmerge_ablation)
      script_path="${WORK_DIR}/submit_jobs_taslp.sh"
      array_spec="1-4"
      job_name="TSASR_TASLP_trainmerge"
      ;;
    benchmark_2spk)
      script_path="${WORK_DIR}/submit_jobs_taslp.sh"
      array_spec="1-2"
      job_name="TSASR_TASLP_2spk"
      ;;
    benchmark_3spk)
      script_path="${WORK_DIR}/submit_jobs_taslp_single_node.sh"
      array_spec="1-2"
      job_name="TSASR_TASLP_3spk"
      ;;
    train100_full)
      script_path="${WORK_DIR}/submit_jobs_taslp_single_node.sh"
      array_spec="1-1"
      job_name="TSASR_train100_full"
      ;;
    *)
      echo "Unsupported TASLP group: ${group}" >&2
      echo "Supported groups: trainmerge_ablation, benchmark_2spk, benchmark_3spk, train100_full, full_matrix" >&2
      exit 1
      ;;
  esac

  echo "Submitting group=${group}"
  echo "  script=${script_path}"
  echo "  array=${array_spec}"
  echo "  run_stage=${run_stage}"
  echo "  stop_stage=${stop_stage}"

  sbatch \
    -J "${job_name}" \
    --array "${array_spec}" \
    -o "${log_dir}/output_${job_name}_%j_%a.txt" \
    -e "${log_dir}/error_${job_name}_%j_%a.txt" \
    --export="ALL,TASLP_GROUP=${group},RUN_STAGE=${run_stage},STOP_STAGE=${stop_stage}" \
    "${script_path}"
}

echo "Work dir: ${WORK_DIR}"
echo "Log dir: ${log_dir}"

if [ "${taslp_group}" = "full_matrix" ]; then
  submit_group trainmerge_ablation
  submit_group benchmark_2spk
  submit_group benchmark_3spk
  submit_group train100_full
else
  submit_group "${taslp_group}"
fi
