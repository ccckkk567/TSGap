#!/bin/bash -l
#SBATCH --job-name=tsgap-experiment
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/%x-%j.out

# For collect/train/evaluate, submit with --gres=gpu:1
# and --export=ALL,TSGAP_GPU_JOB=1. Other stages use the CPU defaults above.
set -eo pipefail
ROOT=${ROOT:-/scratch/prjsnnn0000269/Students/shuming/cjk}
CODE=${CODE:-$ROOT/code/TSGap/code}
case "${1:-}" in
    interactive|collect|train|evaluate)
        if [[ "${TSGAP_GPU_JOB:-0}" != 1 ]]; then
            echo "Submit this stage with --gres=gpu:1 --export=ALL,TSGAP_GPU_JOB=1" >&2
            exit 1
        fi
        ;;
esac
source "$CODE/scripts/common.sh"
cd "$CODE"
python -m scripts.teaching_experiment "$@"
