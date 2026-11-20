#!/bin/bash -l
#SBATCH --job-name=tsgap-32b-bc
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/%x-%j.out

set -eo pipefail
mode=fresh
if (( $# == 1 )) && [[ $1 == --resume ]]; then
    mode=resume
elif (( $# != 0 )); then
    echo "Usage: sbatch run_bc.sbatch [--resume]" >&2
    exit 2
fi
ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
REPO="$ROOT/code/TSGap"
CODE="$REPO/code"
VENV="$ROOT/envs/alfworld"
CONFIG="${TSGAP_INTERACTIVE_CONFIG:-$CODE/examples/teaching/interactive_config.yaml}"
CREDENTIALS="$ROOT/dep-prep-credentials.env"
export ROOT CODE VENV TSGAP_GPU_JOB=1

module load python/3.11.9
module load cuda/12.6.2
[[ -x "$VENV/bin/python" ]] || { echo "Missing venv: $VENV" >&2; exit 1; }
source "$VENV/bin/activate"
export ALFWORLD_DATA="$ROOT/data/alfworld"
export PIP_CACHE_DIR="$ROOT/downloads/pip-alfworld"
export HF_HOME="$ROOT/downloads/huggingface"
export TMPDIR="$ROOT/tmp/tsgap-${SLURM_JOB_ID:?}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_USE_V1=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TMPDIR"

if [[ -f "$CREDENTIALS" ]]; then
    source "$CREDENTIALS"
fi
[[ -n "${SWANLAB_API_KEY:-}" ]] || { echo "Missing SWANLAB_API_KEY (private credentials)" >&2; exit 1; }
[[ $CONFIG == /* && -f "$CONFIG" ]] || { echo "Expected an absolute config path: $CONFIG" >&2; exit 1; }
actual_commit=$(git -C "$REPO" rev-parse HEAD)
printf 'TSGap commit: %s\nConfig: %s\n' "$actual_commit" "$CONFIG"
if [[ -n "${TSGAP_EXPECTED_COMMIT:-}" && $actual_commit != "$TSGAP_EXPECTED_COMMIT" ]]; then
    echo "Checkout changed after submission" >&2
    exit 1
fi

cd "$CODE"
"$VENV/bin/python" -c 'import sys; from scripts.teaching_experiment import check_orion_run; check_orion_run(*sys.argv[1:])' "$CONFIG" "$mode"
exec "$VENV/bin/python" -m scripts.teaching_experiment \
    interactive --stage bc "$@" --config "$CONFIG"
