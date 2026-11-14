# ***RLVMR***: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents


RLVMR provides agents with fine-grained meta-reasoning rewards, encouraging LLM agents to learn how to think rather than simply relying on straightforward outcome feedback.

![The RLVMR framework, comprising cold start and reinforcement learning phases, offers rule-verifiable feedback based on final outcomes and the comparative advantages of meta-reasoning behaviors. The RLVMR framework, comprising cold start and reinforcement learning phases, offers rule-verifiable feedback based on final outcomes and the comparative advantages of meta-reasoning behaviors. ](assets/RLVMR.png "The RLVMR framework, comprising cold start and reinforcement learning phases, offers rule-verifiable feedback based on final outcomes and the comparative advantages of meta-reasoning behaviors. The RLVMR framework, comprising cold start and reinforcement learning phases, offers rule-verifiable feedback based on final outcomes and the comparative advantages of meta-reasoning behaviors. ")

## Environment setup on Orion

The ALFWorld teaching experiment reuses `python/3.11.9` and the existing
`/scratch/prjsnnn0000269/Students/shuming/cjk/envs/alfworld` venv and downloaded
data. Do not recreate the environment or run `alfworld-download` again.
Installation, downloads and model work run in SLURM jobs; the login node is
only used for syncing code and submitting jobs.

On the login node, sync the repository and copy its Slurm submission files to
`$ROOT/scripts`. After pulling updates to the job files, copy them again. The
jobs load `common.sh` and the experiment code from the repository.

```bash
ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
REPO="$ROOT/code/TSGap"
CODE="$REPO/code"
SCRIPTS="$ROOT/scripts"
mkdir -p "$ROOT/code" "$ROOT/logs" "$ROOT/models" "$ROOT/outputs" "$SCRIPTS"
if [[ ! -d "$REPO/.git" ]]; then
    git clone https://github.com/ccckkk567/TSGap.git "$REPO"
else
    git -C "$REPO" pull --ff-only
fi
cp -i "$CODE/scripts/"*.sbatch "$SCRIPTS/"
```

Submit the setup job from this directory when the environment needs installation:

```bash
sbatch "$SCRIPTS/setup_env.sbatch"
squeue -u "$USER"
# Replace JOBID with the submitted job ID:
tail -f "$ROOT/logs/tsgap-env-JOBID.out"
```

The setup job uses `testqueue`, one GPU, 8 CPUs, 64 GB RAM and four hours.
It loads `cuda/12.6.2` for extension compilation, installs the Torch 2.6.0
CUDA 12.4 wheel family, the constrained vLLM/HF dependencies, FlashAttention,
and this checkout's editable `verl`, in that order. `requirements.txt` lists
runtime dependencies; `constraints.txt` fixes the compatibility starting point.
It does not force reinstall packages, change the allocated devices, read
credential files or create a SwanLab test run. Do not install into this venv
while another job is using it.

One final `pip check`, imports and a small BF16 CUDA operation confirm basic
installation. Actual versions are saved in `logs/env-before-JOBID.txt` and
`logs/env-after-JOBID.txt`. Orion job `203043` passed the dependency, import and
BF16 CUDA checks on one H200. Model loading and training remain to be verified.
If installation fails, use the actual job error to adjust the affected package.

After the installation job finishes successfully, download the two models:

```bash
sbatch "$SCRIPTS/download_models.sbatch"
# If Qwen3-4B is already available, download only these Teacher checkpoints.
sbatch "$SCRIPTS/download_models.sbatch" Qwen3-14B Qwen3-32B
```

The `alfworld` venv is used on `testqueue`, including downloads and CPU
experiment stages. The download job uses `snapshot_download` to reuse available
files. By default it downloads Qwen3-14B and Qwen3-4B to the configured paths.
When model names are supplied, it downloads those Qwen checkpoints under
`paths.models` in `examples/teaching/config.yaml`.
Provide Hugging Face / SwanLab credentials through your existing environment
or login mechanism; never add keys or `dep-prep-credentials.env` to Git.

## Start Training

1. **Prepare cold start data**

Prepare a small-batch cold-start set of expert demonstration trajectories. Here, you need to replace YOUR\_API\_KEY with your own API key.

Prepare for ALFWorld:

```python 
python scripts/alfworld_prepare.py
```


Prepare for ScienceWorld:

```bash 
python scripts/sciworld_prepare.py
```


2. **Cold Start**

You can refer to the following startup script to switch to different models or modify configuration parameters.

```bash 
bash examples/sft/cold_start/run_alfworld_qwen2.5-7b.sh

```


```bash 
bash examples/sft/cold_start/run_sciworld_qwen2.5-7b.sh
```


3. **RL Training:**

Before starting the training script, you need to set `actor_rollout_ref.model.path` to the cold start model path.

```bash 
bash examples/rlvmr_trainer/run_alfworld.sh
```


```bash 
bash examples/rlvmr_trainer/run_sciworld.sh
```


## Customize

### ALFWorld teaching experiment

The shared `code/examples/teaching/config.yaml` selects Teacher Qwen3-14B,
Student Qwen3-4B, `data/alfworld`, and one H200 with TP/SP=1. Teacher and Student
workers load sequentially and release their GPU resources on each switch.
`common.sh` selects the existing venv and sets `VLLM_USE_V1=0` before imports;
collection and evaluation also pass this setting into Ray workers.

Actions use thinking mode, a 1024-token budget and sampling
(temperature 0.6, top_p 0.95, top_k 20). Reflection uses the unchanged English
template with native thinking disabled and a 2048-token budget. The Teacher
input budget is 30720 and Student input budget 31744. Overlong prompts fail
instead of silently truncating demonstrations or feedback.

After model download completes, submit each stage below only after its
prerequisites finish successfully. All commands below run on the login node
and submit work to compute nodes. Resource limits can be overridden with
`sbatch`; the run script defaults to four hours.

```bash
cd /scratch/prjsnnn0000269/Students/shuming/cjk/scripts
# testqueue, CPU: fixed train list (six types, five tasks each).
sbatch run_experiment.sbatch sample
# Choose EVAL_TASKS_PER_TYPE before submission, or replace --per-type N with --all.
sbatch run_experiment.sbatch sample --split valid_seen --per-type EVAL_TASKS_PER_TYPE
sbatch run_experiment.sbatch sample --split valid_unseen --per-type EVAL_TASKS_PER_TYPE

# testqueue: single-GPU collection, after all manifests are ready.
sbatch --partition=testqueue --gres=gpu:1 --export=ALL,TSGAP_GPU_JOB=1 run_experiment.sbatch collect

# testqueue, CPU: first select, then export after selection finishes.
sbatch run_experiment.sbatch select
sbatch run_experiment.sbatch export-sft
```

Set `sft.max_length` in the shared config to at least the larger of the two
exported `sft/{high,low}/data_summary.json` maxima before training. This is the
only remaining required model/data-dependent field. Both groups start from
the same Student checkpoint, use full-parameter training for five epochs,
batch size 4, microbatch size 1, gradient checkpointing, and no padding removal.
All paired turns are used, without a validation dataset; complete trajectory
counts are aligned. A changed established setting requires a new run directory.

```bash
# testqueue: complete high before submitting low.
sbatch --partition=testqueue --gres=gpu:1 --export=ALL,TSGAP_GPU_JOB=1 run_experiment.sbatch train --group high
sbatch --partition=testqueue --gres=gpu:1 --export=ALL,TSGAP_GPU_JOB=1 run_experiment.sbatch train --group low

# Then evaluate initial/high/low sequentially on the same fixed lists.
sbatch --partition=testqueue --gres=gpu:1 --export=ALL,TSGAP_GPU_JOB=1 run_experiment.sbatch evaluate --model-path /scratch/prjsnnn0000269/Students/shuming/cjk/models/Qwen3-4B --label initial
# Replace HIGH_CHECKPOINT and LOW_CHECKPOINT with the checkpoint paths recorded
# in the respective sft/{high,low}/training_summary.json files.
sbatch --partition=testqueue --gres=gpu:1 --export=ALL,TSGAP_GPU_JOB=1 run_experiment.sbatch evaluate --model-path HIGH_CHECKPOINT --label high
sbatch --partition=testqueue --gres=gpu:1 --export=ALL,TSGAP_GPU_JOB=1 run_experiment.sbatch evaluate --model-path LOW_CHECKPOINT --label low
# testqueue, CPU: after all evaluations finish.
sbatch run_experiment.sbatch summarize
```

Evaluation performs one sampled episode per task with the same action sampling
for all three models and reports `valid_seen` and `valid_unseen` separately.
Job logs are in `$ROOT/logs/tsgap-experiment-JOBID.out`; complete local records
and SFT checkpoints are in `$ROOT/outputs/teaching/run01` by default.

Logging remains `[console, swanlab]`. Use an existing SwanLab login or supply
`SWANLAB_API_KEY` through the environment. The first real collection metrics
confirm cloud upload; if unavailable, explicitly set `logging.mode: offline`
before starting the run and synchronize later. Local records are retained.
There is no extra reevaluation or model test phase: observe the first real
collection task and model switch, and fix actual errors if they occur.
SFT summaries record actual turns, supervised tokens and optimizer steps;
SFT restarts do not restore optimizer state.

- **Meta-reasoning reward rules**

You can modify or add your custom meta-reasoning reward rules in the `process_trajectory_rlvmr_rewards` function located in `rlvmr/core_rlvmr.py`.

- **New Enviroments**

To add a new environment:

1. Create your environment package in `agent_system/environments/env_package/`, ensuring it follows the gym interface and supports multi-threading.
2. Write the corresponding prompt files in the `agent_system/environments/prompts/` directory.
3. Add an environment manager in `agent_system/environments/env_manager.py` to provide multi-threading support.

## Acknowledgement

We sincerely appreciate the infrastructure provided by [veRL](https://github.com/volcengine/verl "veRL") and [verl-agent](github.com/langfengQ/verl-agent "verl-agent"). We also thank [ALFWorld](https://github.com/alfworld/alfworld "ALFWorld"), [ScienceWorld](https://github.com/allenai/ScienceWorld "ScienceWorld"), and other projects for offering interactive agent environments.

## Citation

If you find RLVMR helpful in your work, we would appreciate it if you cite our work.

```bash 
@article{zhang2025rlvmr,
  title={RLVMR: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents},
  author={Zhang, Zijing and Chen, Ziyang and Li, Mingxiao and Tu, Zhaopeng and Li, Xiaolong},
  journal={arXiv preprint arXiv:2507.22844},
  year={2025}
}
```


