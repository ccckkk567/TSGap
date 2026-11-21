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

#### Jobs and files

| Entry in `$ROOT/scripts` | Where it runs | Responsibility |
| --- | --- | --- |
| `run_bc.sbatch` | One GPU, 8h | Collect 32B demonstrations, train the 4B Student, evaluate full then subset on seen/unseen, publish round 0 metrics. |
| `run_chosen_sft.sbatch` | One GPU, 6h | Train the Student on successful Student episodes. |
| `run_teacher_sft.sbatch` | One GPU, 6h | Train the Student on successful 32B Teacher demonstrations. |
| `run_guided_sft.sbatch` | One GPU, 6h | Train the Student on successful guided episodes. |
| `submit_interactive.sh` | Login node | Check the run and submit the four job types with Slurm dependencies. |

Each GPU entry loads the modules, existing venv and private credentials itself,
changes to the repository's `code` directory, and runs its fixed stage.
A segment is a walltime allocation. Each Main continues as many of its ten rounds
as time allows. Each branch evaluates its Student at rounds 5 and 10.

#### Sync and copy the five entries

On the login node, inspect the queue and the checkout used by active jobs.
Historical IDs include BC 213818, sync 213968, first segments 213969–213971
and tails 213973–213975. Verify current job identities before acting.
Inspect jobs 211432 and 211429 if still active and leave unrelated jobs alone.

```bash
squeue -u "$USER" -o '%.18i %.32j %.10T %.48E %R'
sacct -j 213818,213968,213969,213970,213971,213973,213974,213975 -X -o JobID,JobName%26,State,ExitCode
```

Once jobs using this checkout are finished, run the following on the login
node. Check the printed commit against the published SHA. Copy all five
entries again after updating them. The submitter finds the four GPU files
beside itself, whether launched from the checkout or the copy.

```bash
(
  set -e
  ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
  REPO="$ROOT/code/TSGap"
  git -C "$REPO" pull --ff-only
  git -C "$REPO" log -1 --oneline
  mkdir -p "$ROOT/scripts" "$ROOT/logs"
  files=(run_bc.sbatch run_chosen_sft.sbatch run_teacher_sft.sbatch run_guided_sft.sbatch submit_interactive.sh)
  for file in "${files[@]}"; do
    bash -n "$REPO/code/scripts/$file"
    cp -v "$REPO/code/scripts/$file" "$ROOT/scripts/$file"
  done
)
```

The submitter checks the 32B/4B config, local model headers and tokenizers
before submitting BC. Provision any missing model separately. A fresh run
requires empty run directories. A continuation requires the saved config to
match. To check model metadata before copying or queueing a fresh run, use the
same read-only preflight from the updated checkout:

```bash
(
  set -e
  ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
  cd "$ROOT/code/TSGap/code"
  module load python/3.11.9
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$ROOT/envs/alfworld/bin/python" -c \
    'from scripts.teaching_experiment import check_orion_run; check_orion_run("examples/teaching/interactive_config.yaml", "fresh")'
)
```

#### Choose one submission workflow for this run

**Queue the whole experiment.** Run once from any directory on the login node.

```bash
bash /scratch/prjsnnn0000269/Students/shuming/cjk/scripts/submit_interactive.sh all --bc-segments 1 --main-segments 2
```

This submits seven GPU jobs, one BC and two per Main branch. With two BC
segments it submits eight. Each extra BC segment uses `afterany:previous_BC`
and `--resume`. The three first Main segments use `afterok:BC_tail`. Each
subsequent segment uses `afterok:BC_tail,afterany:previous_same_branch`.
A second BC segment after a completed baseline verifies the artifacts and exits.
BC collection, SFT and full evaluation together can exceed eight hours.

**Submit in two stages.** To choose BC walltime based on progress, use this
workflow instead of advance queueing. First submit BC:

```bash
bash /scratch/prjsnnn0000269/Students/shuming/cjk/scripts/submit_interactive.sh bc-only
```

If BC stops before finishing, inspect its log and job state. Once it has
stopped, resume the same run with its saved config:

```bash
bash /scratch/prjsnnn0000269/Students/shuming/cjk/scripts/submit_interactive.sh bc-only --bc-resume
```

When BC has published its checkpoint and `sft/bc/train_complete.json`, queue
the six Main jobs. Any running BC evaluation can continue alongside them.
Each Main branch's second allocation waits for its own first allocation to end.

```bash
bash /scratch/prjsnnn0000269/Students/shuming/cjk/scripts/submit_interactive.sh mains-only --main-segments 2
```

If deploying an update during BC, keep its checkout at its submitted commit.
Run the updated submitter from a separate Git worktree and set
`TSGAP_MAIN_REPO` to that worktree's absolute path for `mains-only`. The
submitter and the Main jobs use that checkout and the same run artifacts.

Both workflows print job IDs and append each accepted segment immediately to
`$ROOT/logs/interactive_run02_teacher32b-submissions.tsv`. On partial submission
check the journal and queue, then submit only the missing segments with their
intended dependencies. If all BC segments expire, Slurm keeps the old Main
dependencies attached to the old BC tail. Verify the affected IDs and states,
manually cancel only those obsolete **pending** segments, tails first, then
first segments. Resume BC and use `mains-only` when ready. Check OOM or other
runtime errors in the log before spending another allocation.

```bash
squeue -u "$USER" -o '%.18i %.32j %.10T %.48E %R'
# Replace JOBID with the printed job ID.
sacct -j JOBID -X -o JobID,JobName%26,State,ExitCode,Elapsed,Timelimit
```

**Individual submission.** These are alternatives for starting one job. Select
the applicable command only, after checking for existing writers and creating
the logs directory. Run Main entries only after BC is ready. When extending a
queued branch, pass its dependencies to `sbatch` before the script path.

```bash
SCRIPTS=/scratch/prjsnnn0000269/Students/shuming/cjk/scripts
sbatch "$SCRIPTS/run_bc.sbatch"                 # fresh BC
sbatch "$SCRIPTS/run_bc.sbatch" --resume        # continue existing BC
sbatch "$SCRIPTS/run_chosen_sft.sbatch"         # Chosen only
sbatch "$SCRIPTS/run_teacher_sft.sbatch"        # Teacher only
sbatch "$SCRIPTS/run_guided_sft.sbatch"         # Guided only
```

Inspect the BC log for the 32B Teacher, `static_rollout=True`, vLLM weight
loading, GPU memory and continuing episode output. The training marker
`sft/bc/train_complete.json` publishes the shared checkpoint. BC then writes
four reports under `eval/bc/round_000/{full,subset}/{valid_seen,valid_unseen}`
and the subset curve under `rounds/round_000.json`, using the BC tracking
namespace. With matching tasks, checkpoint, settings and seeds, subset reuses
full episodes. A BC resume skips published training and completes missing
evaluation work. Each Main verifies the published BC checkpoint and resumes
from its own saved rounds. Inspect branch-specific round 5/10 artifacts.

**Final CPU comparison.** After the BC evaluation, round-zero report, and all
three branches finish, submit the existing
summary separately. The batch submitter leaves this operation to the user.

```bash
ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
CODE="$ROOT/code/TSGap/code"
TSGAP_GPU_JOB=0 sbatch --account=prjsnnn0000269 --partition=ioqueue \
  --cpus-per-task=1 --mem=4G --time=00:15:00 --export=ALL --job-name=tsgap-32b-summary \
  "$CODE/scripts/run_experiment.sbatch" interactive-summary --config "$CODE/examples/teaching/interactive_config.yaml"
```

The result is `$ROOT/outputs/teaching/interactive_run02_teacher32b/final_comparison.json`.

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

## Run 03 on Orion

Run 03 uses the published Run 02 BC Student and its fixed task manifests. Its own baseline evaluation and three ten-round branches can run concurrently. The rollout process uses one GPU with one offline engine at a time. Qwen3-4B SFT runs in the existing alfworld environment.

### Prepare the checkout and rollout environment

After merging the Run 03 PR, create the new checkout once on the login node. Confirm the repository with `worktree list` first. Keep the new checkout at this revision while its jobs are queued or running.

```bash
(
  set -eu
  ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
  BASE="$ROOT/code/TSGap"
  DEV="$ROOT/code/TSGap-run03"
  git -C "$BASE" worktree list
  git -C "$BASE" fetch origin main
  git -C "$BASE" worktree add --detach "$DEV" origin/main
  git -C "$DEV" log -1 --oneline
)
```

Check the driver and supported CUDA version in an existing H200 job's `nvidia-smi` output. For a driver supporting CUDA 12.9, use the following candidate on the network-enabled IO partition. The wheel resolver selects the corresponding Torch build. The preparation log reports the installed versions.

```bash
ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
mkdir -p "$ROOT/logs"
sbatch --parsable --account=prjsnnn0000269 --partition=ioqueue --cpus-per-task=4 \
  --mem=32G --time=06:00:00 --job-name=tsgap-r03-prep --output="$ROOT/logs/%x-%j.out" <<'PREP'
#!/bin/bash -l
set -eo pipefail
ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
DEV="$ROOT/code/TSGap-run03"
VENV="$ROOT/envs/rollout"
export HF_HOME="$ROOT/downloads/huggingface" UV_CACHE_DIR="$ROOT/downloads/uv"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
module load python/3.11.9
python -m venv "$VENV"
"$VENV/bin/python" -m pip install -U uv
"$VENV/bin/uv" pip install --python "$VENV/bin/python" --torch-backend=cu129 \
  --extra-index-url https://wheels.vllm.ai/0.32.0/cu129 --only-binary=vllm \
  -r "$DEV/code/requirements-rollout.txt"
"$VENV/bin/python" -c 'from importlib.metadata import version; import torch; print("vLLM", version("vllm"), "transformers", version("transformers"), "torch", torch.__version__, "CUDA", torch.version.cuda)'
"$VENV/bin/python" - "$ROOT/models/Qwen3.8-27B" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.8-27B", local_dir=sys.argv[1])
PY
PREP
```

### Initialize the shared BC and submit the eight GPU segments

After preparation succeeds, run the block below once on the login node. Initialization requires Run 02's published `sft/bc/train_complete.json`. It links the source checkpoint into the Run 03 checkpoint root, copies the manifests and small summaries, and publishes a local marker last. Source training counts are inherited. Evaluation scores come from Run 03.

```bash
(
  set -eo pipefail
  ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
  export TSGAP_REPO="$ROOT/code/TSGap-run03"
  CODE="$TSGAP_REPO/code"
  export TSGAP_INTERACTIVE_CONFIG="$CODE/examples/teaching/run03.yaml"
  export ALFWORLD_DATA="$ROOT/data/alfworld" HF_HOME="$ROOT/downloads/huggingface"
  module load python/3.11.9
  cd "$CODE"
  "$ROOT/envs/rollout/bin/python" -m scripts.teaching_experiment init-bc --config "$TSGAP_INTERACTIVE_CONFIG"
  bash "$CODE/scripts/submit_interactive.sh" bc-only --bc-resume --bc-segments 2 --bc-time 06:00:00
  bash "$CODE/scripts/submit_interactive.sh" mains-only --main-segments 2 --main-time 06:00:00
  cat "$ROOT/logs/interactive_run03-submissions.tsv"
)
```

Each branch's first segment is independently eligible. Its second has an `afterany` dependency on its own first segment. The two BC evaluation segments form a separate chain. If submission stops halfway, consult the journal and queue for accepted jobs before submitting missing segments. The first real trajectories record raw completions, environment responses, finish reasons, and request costs. The SFT log records the interpreter and checkout.

After round zero and the three branches finish, generate the comparison in the Run 03 output directory. The online summary uses the same private SwanLab credentials as the experiment jobs.

```bash
(
  set -eo pipefail
  ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
  module load python/3.11.9
  cd "$ROOT/code/TSGap-run03/code"
  source "$ROOT/dep-prep-credentials.env"
  "$ROOT/envs/rollout/bin/python" -m scripts.teaching_experiment interactive-summary \
    --config "$PWD/examples/teaching/run03.yaml"
)
```

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


