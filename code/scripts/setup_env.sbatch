#!/bin/bash -l
#SBATCH --job-name=tsgap-env
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/%x-%j.out

set -eo pipefail
ROOT=${ROOT:-/scratch/prjsnnn0000269/Students/shuming/cjk}
CODE=${CODE:-$ROOT/code/TSGap/code}
export TSGAP_GPU_JOB=1
source "$CODE/scripts/common.sh"
export MAX_JOBS=4
cd "$CODE"

hostname
uname -m
module list 2>&1
python --version
python -c 'import sys; print("Python:", sys.executable); print("venv:", sys.prefix)'
nvidia-smi
nvcc --version

# Version records, not environment backups. Do not run alongside model jobs.
python -m pip freeze > "$ROOT/logs/env-before-${SLURM_JOB_ID}.txt"
python -m pip install --upgrade pip
python -m pip install 'setuptools>=74.1.1,<81' wheel packaging ninja
python -m pip install \
    'torch==2.6.0+cu124' 'torchvision==0.21.0+cu124' 'torchaudio==2.6.0+cu124' \
    --index-url https://download.pytorch.org/whl/cu124
python -m pip install --only-binary=vllm -c "$CODE/constraints.txt" -r "$CODE/requirements.txt"
python -m pip install --no-build-isolation -c "$CODE/constraints.txt" 'flash-attn==2.7.4.post1'
python -m pip install -e "$CODE" --no-deps --no-build-isolation
python -m pip check

# One installation check; no model loading, downloads or tracking test run.
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import torch
import vllm
import flash_attn
import alfworld
import textworld
import gymnasium
import swanlab
import verl
from transformers import Qwen3ForCausalLM
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
from verl.utils.tracking import Tracking
from agent_system.teaching.runtime import SequentialRollouts
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from agent_system.environments.env_package.alfworld.envs import AlfworldEnvs

packages = [
    'alfworld', 'textworld', 'gymnasium', 'torch', 'torchvision', 'torchaudio',
    'vllm', 'transformers', 'flash-attn', 'ray', 'tensordict', 'torchdata',
    'accelerate', 'peft', 'datasets', 'swanlab',
]
for name in packages:
    print(f'{name}: {version(name)}')
print('verl source:', Path(verl.__file__).resolve())
print('Torch CUDA build:', torch.version.cuda)
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is not available in this allocated GPU job')
print('GPU:', torch.cuda.get_device_name(0))
x = torch.ones((16, 16), dtype=torch.bfloat16, device='cuda')
print('BF16 CUDA operation:', (x @ x).float().mean().item())
print('ENV_PACKAGES_OK: imports and a small CUDA operation succeeded; model runs are not yet validated.')
PY

python -m pip freeze > "$ROOT/logs/env-after-${SLURM_JOB_ID}.txt"
echo "Environment dependency installation completed. No model collection or SFT was started."
