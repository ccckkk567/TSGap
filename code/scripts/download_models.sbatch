#!/bin/bash -l
#SBATCH --job-name=tsgap-models
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/%x-%j.out

set -eo pipefail
ROOT=${ROOT:-/scratch/prjsnnn0000269/Students/shuming/cjk}
CODE=${CODE:-$ROOT/code/TSGap/code}
export TSGAP_GPU_JOB=0
source "$CODE/scripts/common.sh"
cd "$CODE"
python - "$@" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

config = OmegaConf.load('examples/teaching/config.yaml')
models = [(f'Qwen/{name}', str(Path(config.paths.models) / name)) for name in sys.argv[1:]]
for model_id, target in models or (
    ('Qwen/Qwen3-14B', config.teacher.model_path),
    ('Qwen/Qwen3-4B', config.student.model_path),
):
    print(f'Downloading {model_id} to {target}', flush=True)
    snapshot_download(model_id, local_dir=target)
    print(f'Completed: {target}', flush=True)
PY
