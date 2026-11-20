# ***RLVMR***: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents


RLVMR provides agents with fine-grained meta-reasoning rewards, encouraging LLM agents to learn how to think rather than simply relying on straightforward outcome feedback.

![The RLVMR framework, comprising cold start and reinforcement learning phases, offers rule-verifiable feedback based on final outcomes and the comparative advantages of meta-reasoning behaviors. The RLVMR framework, comprising cold start and reinforcement learning phases, offers rule-verifiable feedback based on final outcomes and the comparative advantages of meta-reasoning behaviors. ](assets/RLVMR.png "The RLVMR framework, comprising cold start and reinforcement learning phases, offers rule-verifiable feedback based on final outcomes and the comparative advantages of meta-reasoning behaviors. The RLVMR framework, comprising cold start and reinforcement learning phases, offers rule-verifiable feedback based on final outcomes and the comparative advantages of meta-reasoning behaviors. ")

## Interactive ALFWorld SFT on Orion

The independent interactive experiment uses Qwen3-32B as the fixed Teacher and
Qwen3-4B as the initial Student on one H200. Its configuration is
`code/examples/teaching/interactive_config.yaml`. The high/low and cross-instance
ICL commands below belong to the earlier teaching experiment.

BC attempts 24 candidate tasks (four per type), retaining at most the first
successful Teacher trajectory in three attempts per task. Two epochs produce
the shared initial checkpoint. Ten rounds each use the same 12 new tasks across
Chosen, Teacher and Guided. Chosen and Guided sample four Student episodes on
each task, while Teacher uses the BC attempt rule. Guided reviews surviving
sessions after actions 10/20/30/40. A diagnostic note is visible for the next
action only. All branches train on successful episodes of at most 50 actions
from the current round. Each example reconstructs the ordinary two-step prompt
and preserves the actual response. A round with zero successes carries its
input model forward. BC requires at least one success.

### Launch the new 32B experiment

The default configuration creates `interactive_run02_teacher32b`. Its manifests,
episodes, exported data, markers and evaluations live under
`$ROOT/outputs/teaching/interactive_run02_teacher32b`. Its SwanLab state and logs
use the new run, and model checkpoints live under
`/scratch/prjsnnn0000269/Students/shuming/storage_ext/cjk/checkpoints/interactive_run02_teacher32b`.
The original 4B Student learns from freshly collected 32B demonstrations.
The 14B run `interactive_run01` experienced a native crash during BC collection.
Its configuration and recovery notes are archived at [d37f87f](https://github.com/ccckkk567/TSGap/tree/d37f87f).

`interactive.static_rollout: true` selects single-GPU vLLM SPMD. The worker loads
HF metadata and the checkpoint's own tokenizer; vLLM owns the model weights.
Each request keeps its sampling seed and resets the prefix cache on success.
Changing the role or checkpoint rebuilds the worker. Teacher GPU utilization
starts at 0.8, Student at 0.5. Actual KV-cache capacity, peak memory and stability
with the 32B Teacher require the first H200 run.

**Inspect the historical queue on the login node.** Confirm the state of BC
213818, the identities of the old pending jobs, and the command and working
directory of jobs using the checkout. Inspect 211432 and 211429 if still active.

```bash
sacct -j 213818 -o JobID,JobName%22,State,ExitCode,Elapsed,Timelimit
squeue -u "$USER" -o "%.18i %.22j %.10T %.40E %R"
scontrol show job 211432
scontrol show job 211429
```

Only if these IDs still identify the obsolete pending chain, cancel its tails
first, then its first segments, then its sync job. Leave unrelated jobs running.

```bash
scancel --state=PENDING 213973 213974 213975
scancel --state=PENDING 213969 213970 213971
scancel --state=PENDING 213968
squeue -u "$USER"
```

**Set up the submission shell.** Run the following in Bash. Repeat this setup
when using a new login session. All `sbatch` commands below run on the login node.

```bash
export ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
REPO="$ROOT/code/TSGap"
export CODE="$REPO/code" TSGAP_GPU_JOB=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CONFIG="$CODE/examples/teaching/interactive_config.yaml"
JOB="$CODE/scripts/run_experiment.sbatch"
RUN=interactive_run02_teacher32b
OUT="$ROOT/outputs/teaching/$RUN"
CKPT="/scratch/prjsnnn0000269/Students/shuming/storage_ext/cjk/checkpoints/$RUN"
export MARKER="$OUT/sft/bc/train_complete.json" BC_MODEL="$CKPT/bc"
PYTHON="$ROOT/envs/alfworld/bin/python"
opts=(--parsable --account=prjsnnn0000269 --partition=testqueue --gres=gpu:1 --export=ALL)
```

**Start BC once, after the checkout is idle.** This checks the local Teacher
tokenizer, config and indexed safetensors shards without loading the model.
Stop and provision the model separately if a check fails. Resolve any existing
run artifacts before starting; retained episodes count toward the attempt budget
only when resuming this same 32B experiment.

```bash
(
  set -euo pipefail
  git -C "$REPO" pull --ff-only
  "$PYTHON" - "$CONFIG" "$ROOT/models/Qwen3-32B" "$OUT" "$CKPT" <<'PY'
import json, sys
from pathlib import Path
from omegaconf import OmegaConf
from safetensors import safe_open
from transformers import AutoTokenizer

config_file, model, out, checkpoints = map(Path, sys.argv[1:])
cfg = OmegaConf.load(config_file)
if (Path(cfg.teacher.model_path) != model or not cfg.interactive.static_rollout
        or Path(cfg.run.output_dir) != out or Path(cfg.run.checkpoint_dir) != checkpoints):
    sys.exit("Unexpected Interactive configuration or run paths")
for directory in (out, checkpoints, Path(cfg.logging.log_dir).parent):
    if directory.exists() and any(directory.iterdir()):
        sys.exit(f"Inspect the existing run directory before starting: {directory}")
for name in ("config.json", "tokenizer_config.json", "model.safetensors.index.json"):
    if not (model / name).is_file():
        sys.exit(f"Missing {model / name}")
metadata = json.loads((model / "config.json").read_text())
if metadata.get("model_type") != "qwen3":
    sys.exit("Unexpected Teacher architecture")
weights = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
if not weights:
    sys.exit("Empty Teacher weight index")
for shard in set(weights.values()):
    with safe_open(str(model / shard), framework="np") as f:
        if not {key for key, name in weights.items() if name == shard}.issubset(f.keys()):
            sys.exit(f"Incomplete Teacher shard: {shard}")
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
messages = [{"role": "user", "content": "Hello"}]
prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=flag) for flag in (True, False)]
if prompts[0] == prompts[1]:
    sys.exit("Check the 32B tokenizer's thinking template")
print(f"Verified local Teacher metadata and {len(set(weights.values()))} indexed shards")
PY
  mkdir -p "$ROOT/logs"
  sbatch "${opts[@]}" --time=08:00:00 --job-name=tsgap-32b-bc1 \
    "$JOB" interactive --stage bc --config "$CONFIG"
)
```

Watch the BC log for `checkpoint=.../Qwen3-32B, static_rollout=True`, vLLM shard
loading, and new `collect/bc/*/attempt_*.json` files. On OOM, inspect the vLLM
profiling and KV capacity at the configured context length. If BC ended before
publishing its checkpoint, resolve the reported failure or timeout and submit
one continuation. Completed successes and failures are reused; an interrupted
episode restarts from its task and slot.

```bash
sbatch "${opts[@]}" --time=08:00:00 --job-name=tsgap-32b-bc-resume \
  "$JOB" interactive --stage bc --resume --config "$CONFIG"
```

**Submit Main after BC publishes its checkpoint.** Define these two functions
in the submission shell. They are shared by both submission options below.

```bash
check_bc() {
  "$ROOT/envs/alfworld/bin/python" - <<'PY'
import json, os
from pathlib import Path
marker = json.loads(Path(os.environ["MARKER"]).read_text())
model = Path(marker["output_checkpoint"])
if ((marker["status"], marker["group"], marker["round_index"]) != ("complete", "bc", 0)
        or model.resolve() != Path(os.environ["BC_MODEL"]).resolve() or not model.is_dir()):
    raise SystemExit("New 32B BC completion record or checkpoint is invalid")
print(f"Verified shared BC checkpoint: {model}")
PY
}

submit_branches() (
  set -euo pipefail
  for group in chosen teacher guided; do
    args=(interactive --stage main --group "$group" --resume --config "$CONFIG")
    first=$(sbatch "${opts[@]}" --time=06:00:00 --job-name="tsgap-32b-$group-1" \
      "$@" "$JOB" "${args[@]}")
    echo "$group first: $first"
    tail_job=$(sbatch "${opts[@]}" --time=06:00:00 --job-name="tsgap-32b-$group-2" \
      --dependency="afterany:${first%%;*}" "$JOB" "${args[@]}")
    echo "$group tail: $tail_job"
  done
)
```

After BC ends successfully, run once:

```bash
check_bc && submit_branches
```

The three first segments run independently. Each tail resumes only its branch.
If submission stops midway, use the printed IDs to submit the missing segments.
To extend one active chain, depend on its current tail, for example:

```bash
read -rp 'Guided tail job ID: ' tail_id
sbatch "${opts[@]}" --time=06:00:00 --job-name=tsgap-32b-guided-resume \
  --dependency="afterany:$tail_id" "$JOB" interactive --stage main --group guided \
  --resume --config "$CONFIG"
```

**Optional advance queueing.** Use this instead of the `check_bc && submit_branches`
submission above. After submitting the first BC, run the following once while
its tail is queued or running. This example budgets one additional eight-hour
BC segment. Slurm then runs a CPU-only marker check. If BC is still incomplete,
the check fails and the Main dependency chain must be rebuilt after more BC time.

```bash
(
  set -euo pipefail
  read -rp 'Current BC chain tail job ID: ' previous
  tail_bc=$(sbatch "${opts[@]}" --time=08:00:00 --job-name=tsgap-32b-bc-resume \
    --dependency="afterany:$previous" "$JOB" interactive --stage bc --resume --config "$CONFIG")
  echo "BC tail: $tail_bc"
  gate=$({
    echo '#!/bin/bash'
    declare -f check_bc
    echo 'check_bc'
  } | sbatch --parsable --account=prjsnnn0000269 --partition=ioqueue \
      --cpus-per-task=1 --mem=2G --time=00:10:00 --export=ALL \
      --job-name=tsgap-32b-check-bc --output="$ROOT/logs/%x-%j.out" \
      --dependency="afterany:${tail_bc%%;*}")
  echo "CPU check: $gate"
  submit_branches --dependency="afterok:${gate%%;*}"
)
```

Chosen owns BC subset/full evaluation. Branches evaluate their own rounds 5
and 10. After all four final evaluations are complete on both splits, submit
the CPU comparison from the same shell:

```bash
TSGAP_GPU_JOB=0 sbatch --parsable --account=prjsnnn0000269 --partition=ioqueue \
  --cpus-per-task=1 --mem=4G --time=00:15:00 --export=ALL --job-name=tsgap-32b-summary \
  "$JOB" interactive-summary --config "$CONFIG"
```

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

For `run02`, Qwen3-32B has been downloaded. Qwen3-8B availability still needs
confirmation at `$ROOT/models/Qwen3-8B`. If needed, use the existing download job:

```bash
sbatch "$SCRIPTS/download_models.sbatch" Qwen3-8B
```

The `alfworld` venv is used on `testqueue`, including downloads and CPU
experiment stages. The download job uses `snapshot_download` to reuse available
files. Pass the model name explicitly to download that Qwen checkpoint under
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

The shared `code/examples/teaching/config.yaml` selects standard Qwen3-32B as
Teacher and Qwen3-8B as Student, `data/alfworld`, and one H200 with TP/SP=1.
Teacher and Student
workers load sequentially and release their GPU resources on each switch.
`common.sh` selects the existing venv and sets `VLLM_USE_V1=0` before imports;
collection and evaluation also pass this setting into Ray workers.

Actions use thinking mode, a 1024-token budget and sampling
(temperature 0.6, top_p 0.95, top_k 20). Reflection uses the unchanged English
template with native thinking disabled and a 2048-token budget. The Teacher
input budget is 30720 and Student input budget 31744. Overlong prompts fail
instead of silently truncating demonstrations or feedback. SFT starts from
Qwen3-8B and uses its tokenizer with `enable_thinking: true`.

The prompt encloses the complete Teacher example in a reference-demonstration
block and presents the current episode separately. With a demonstration,
it explicitly identifies the current history and observations as the source
of the Student's actual state.

`run02` creates fresh manifests, Teacher trajectories, Student rollouts and
SwanLab state using the existing seeds. If `run02` already contains a different
experiment on HPC, select the next unused `run.name` before submission.
The 32B/8B combination still awaits validation through the actual single-H200
collection job. Reflection is attempted after four Student rollouts, and
oversized reflection inputs still raise the existing error with completed
records retained.

Once both model directories are ready and the old collection job has exited,
sync on the login node and submit the existing one-hour `testqueue` job:

```bash
ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
git -C "$ROOT/code/TSGap" pull --ff-only
sbatch --time=01:00:00 "$ROOT/scripts/collect.sbatch"
```

Inside the existing `collect.sbatch`, `CONFIG` points to
`$ROOT/code/TSGap/code/examples/teaching/config.yaml`. The allocated job runs:

```bash
python -u -m scripts.teaching_experiment sample --config "$CONFIG" --split train
python -u -m scripts.teaching_experiment sample --config "$CONFIG" --split valid_seen --all
python -u -m scripts.teaching_experiment sample --config "$CONFIG" --split valid_unseen --all
python -u -m scripts.teaching_experiment collect --config "$CONFIG"
```

After collection completes, the subsequent stages can be submitted separately:

```bash
cd /scratch/prjsnnn0000269/Students/shuming/cjk/scripts
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
sbatch --partition=testqueue --gres=gpu:1 --export=ALL,TSGAP_GPU_JOB=1 run_experiment.sbatch evaluate --model-path /scratch/prjsnnn0000269/Students/shuming/cjk/models/Qwen3-8B --label initial
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
and SFT checkpoints are in `$ROOT/outputs/teaching/run02` by default.

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


