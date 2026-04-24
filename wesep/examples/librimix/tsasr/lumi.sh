# MIOpen cache: use node-local tmp, not Lustre
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-cache-${SLURM_NODEID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"

if [ "${SLURM_LOCALID:-0}" -eq 0 ]; then
    rm -rf "${MIOPEN_USER_DB_PATH}"
    mkdir -p "${MIOPEN_USER_DB_PATH}"
fi
sleep 2

# Faster startup, usually fine for training
# export MIOPEN_FIND_MODE=2

# PyTorch HIP allocator tuning
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

# 56 CPUs / 8 GPUs -> 7 threads per rank is safer
export OMP_NUM_THREADS=7

export MIOPEN_LOG_LEVEL=3
