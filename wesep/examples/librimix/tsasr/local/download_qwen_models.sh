#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
recipe_dir=$(cd -- "${script_dir}/.." && pwd)

target_root="${recipe_dir}/qwen_models"
models="0.6B 1.7B"
token="${HF_TOKEN:-}"
force=0

show_help() {
  cat <<EOF
Usage: $(basename "$0") [options]

Download Qwen3-ASR model weights to the local tsasr recipe directory.

Options:
  --target_root DIR   Destination root directory.
                      Default: ${recipe_dir}/qwen_models
  --models STR        Space-separated model sizes to download.
                      Supported values: 0.6B 1.7B
                      Default: "0.6B 1.7B"
  --token TOKEN       Hugging Face token. Defaults to \$HF_TOKEN if set.
  --force             Re-download even if target directory already exists.
  -h, --help          Show this help message.

Examples:
  bash local/download_qwen_models.sh
  bash local/download_qwen_models.sh --models "0.6B"
  HF_TOKEN=xxxx bash local/download_qwen_models.sh --target_root /path/to/qwen_models
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target_root)
      target_root="$2"
      shift 2
      ;;
    --models)
      models="$2"
      shift 2
      ;;
    --token)
      token="$2"
      shift 2
      ;;
    --force)
      force=1
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_help >&2
      exit 1
      ;;
  esac
done

mkdir -p "${target_root}"
target_root=$(realpath "${target_root}")

if ! command -v python >/dev/null 2>&1; then
  echo "python is required but was not found in PATH." >&2
  exit 1
fi

download_one_model() {
  local model_size="$1"
  local repo_id="Qwen/Qwen3-ASR-${model_size}"
  local local_dir="${target_root}/Qwen3-ASR-${model_size}"

  case "${model_size}" in
    0.6B|1.7B)
      ;;
    *)
      echo "Unsupported model size: ${model_size}" >&2
      exit 1
      ;;
  esac

  if [[ -d "${local_dir}" && "${force}" -ne 1 ]]; then
    if [[ -f "${local_dir}/config.json" ]]; then
      echo "[skip] ${repo_id} already exists at ${local_dir}"
      return 0
    fi
  fi

  echo "[download] ${repo_id} -> ${local_dir}"
  rm -rf "${local_dir}"

  HF_TOKEN_ARG="${token}" \
  QWEN_REPO_ID="${repo_id}" \
  QWEN_LOCAL_DIR="${local_dir}" \
  python - <<'PY'
import os
import sys

repo_id = os.environ["QWEN_REPO_ID"]
local_dir = os.environ["QWEN_LOCAL_DIR"]
token = os.environ.get("HF_TOKEN_ARG") or None

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit(
        "huggingface_hub is required. Install it with `pip install -U huggingface_hub`."
    ) from exc

kwargs = dict(
    repo_id=repo_id,
    local_dir=local_dir,
    resume_download=True,
)

if token:
    kwargs["token"] = token

try:
    kwargs["local_dir_use_symlinks"] = False
    snapshot_download(**kwargs)
except TypeError:
    kwargs.pop("local_dir_use_symlinks", None)
    snapshot_download(**kwargs)

print(f"[done] {repo_id} -> {local_dir}")
PY
}

for model_size in ${models}; do
  download_one_model "${model_size}"
done

cat <<EOF
[summary] Download finished.
[summary] Models root: ${target_root}
EOF
