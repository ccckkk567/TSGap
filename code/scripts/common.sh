#!/bin/bash
# Source inside SLURM jobs; keep the interpreter used to create this venv.
ROOT=${ROOT:-/scratch/prjsnnn0000269/Students/shuming/cjk}
CODE=${CODE:-$ROOT/code/TSGap/code}
VENV=${VENV:-$ROOT/envs/alfworld}

module load python/3.11.9
if [[ "${TSGAP_GPU_JOB:-0}" == 1 ]]; then
    module load cuda/12.6.2
fi
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "Existing venv not found: $VENV" >&2
    return 1
fi
source "$VENV/bin/activate"

export ROOT CODE VENV
export ALFWORLD_DATA="$ROOT/data/alfworld"
export PIP_CACHE_DIR="$ROOT/downloads/pip-alfworld"
export HF_HOME="$ROOT/downloads/huggingface"
export TMPDIR="$ROOT/tmp/tsgap-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Required by the existing veRL SPMD/sharding path, including Ray children.
export VLLM_USE_V1=0
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TMPDIR"
