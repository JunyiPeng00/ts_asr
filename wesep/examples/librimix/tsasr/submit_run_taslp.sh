#!/bin/bash

set -euo pipefail

show_usage() {
  cat <<'EOF'
Usage:
  bash submit_run_taslp.sh <group>

Groups:
  this_round           current recommended round, alias of trainmerge_ablation
  trainmerge_ablation  B0/B1/B2/B3 on train-merge
  benchmark_2spk       B0/B3 on train-100
  benchmark_3spk       B0/B3 on train-3mix-custom
  train100_full        full train-100 config
  full_matrix          submit all TASLP groups
  help                 print this message

Environment:
  RUN_STAGE            default 7
  STOP_STAGE           default 7
  DRY_RUN              set to 1 to print sbatch commands without submitting
EOF
}

resolve_group() {
  local group="$1"
  case "${group}" in
    this_round)
      echo "trainmerge_ablation"
      ;;
    *)
      echo "${group}"
      ;;
  esac
}

submit_cmd() {
  if [ "${DRY_RUN:-0}" = "1" ]; then
    printf 'DRY_RUN:'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

time=$(date +"%Y-%m-%d")
echo "Date: ${time}"

WORK_DIR=/scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr
cd "${WORK_DIR}"

log_dir="${WORK_DIR}/log/${time}"
mkdir -p "${log_dir}"

taslp_group="${1:-this_round}"
run_stage="${RUN_STAGE:-7}"
stop_stage="${STOP_STAGE:-7}"

submit_group() {
  local group="$1"
  local script_path=""
  local array_spec=""
  local job_name=""

  group="$(resolve_group "${group}")"

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
      echo "Supported groups: this_round, trainmerge_ablation, benchmark_2spk, benchmark_3spk, train100_full, full_matrix" >&2
      exit 1
      ;;
  esac

  echo "Submitting group=${group}"
  echo "  script=${script_path}"
  echo "  array=${array_spec}"
  echo "  run_stage=${run_stage}"
  echo "  stop_stage=${stop_stage}"
  echo "  dry_run=${DRY_RUN:-0}"

  submit_cmd sbatch \
    -J "${job_name}" \
    --array "${array_spec}" \
    -o "${log_dir}/output_${job_name}_%j_%a.txt" \
    -e "${log_dir}/error_${job_name}_%j_%a.txt" \
    --export="ALL,TASLP_GROUP=${group},RUN_STAGE=${run_stage},STOP_STAGE=${stop_stage}" \
    "${script_path}"
}

echo "Work dir: ${WORK_DIR}"
echo "Log dir: ${log_dir}"
echo "Requested group: ${taslp_group}"

if [ "${taslp_group}" = "help" ] || [ "${taslp_group}" = "--help" ] || [ "${taslp_group}" = "-h" ]; then
  show_usage
elif [ "${taslp_group}" = "full_matrix" ]; then
  submit_group trainmerge_ablation
  submit_group benchmark_2spk
  submit_group benchmark_3spk
  submit_group train100_full
else
  submit_group "${taslp_group}"
fi
