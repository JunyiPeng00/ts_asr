export PATH=$PWD:$PATH

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8

export WESEP_ROOT=/scratch/project_465002316/junyi/tse/ts_asr/wesep
export QWEN3_ASR_ROOT=/scratch/project_465002316/junyi/tse/ts_asr/Qwen3-ASR
export QWEN_MODELS_DIR=/scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr/qwen_models
export WESPEAKER_ROOT=/scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tse/v2/wespeaker
export PYTHONPATH=$PWD:$WESEP_ROOT:$QWEN3_ASR_ROOT:$WESPEAKER_ROOT:$PYTHONPATH
