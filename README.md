# ***RLVMR***: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents


RLVMR provides agents with fine-grained meta-reasoning rewards, encouraging LLM agents to learn how to think rather than simply relying on straightforward outcome feedback.

![The RLVMR framework, comprising cold start and reinforcement learning phases, offers rule-verifiable feedback based on final outcomes and the comparative advantages of meta-reasoning behaviors. The RLVMR framework, comprising cold start and reinforcement learning phases, offers rule-verifiable feedback based on final outcomes and the comparative advantages of meta-reasoning behaviors. ](assets/RLVMR.png "The RLVMR framework, comprising cold start and reinforcement learning phases, offers rule-verifiable feedback based on final outcomes and the comparative advantages of meta-reasoning behaviors. The RLVMR framework, comprising cold start and reinforcement learning phases, offers rule-verifiable feedback based on final outcomes and the comparative advantages of meta-reasoning behaviors. ")

## Environment Setup&#x20;



```bash 
pip install -r requirements.txt
```


We recommend maintaining a separate conda environment for each environment.

**ALFWorld**

```markdown 
conda create -n rlvmr-alfworld -y
conda activate rlvmr-alfworld
pip install gymnasium==0.29.1
pip install stable-baselines3==2.6.0
pip install alfworld
```


Download PDDL & Game files (by default saved in `~/.cache/alfworld/`):

```markdown 
alfworld-download -f
```


**ScienceWorld**

```bash 
conda create -n rlvmr-sciworld -y
conda activate rlvmr-sciworld
pip install scienceworld
```




Install verl and related dependencies.

```bash 
pip install -e .
```




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

On HPC, run from
`/scratch/prjsnnn0000269/Students/shuming/cjk/code/TSGap/code`.
The shared `examples/teaching/config.yaml` places models, data, outputs and logs
under the configured `cjk` root. `tasks.data_root` must contain
`json_2.1.1/` and `logic/`. Fill checkpoint paths under `paths.models`,
prompt lengths and SFT lengths/batch sizes when the Qwen models are selected.
Pending `???` fields can be filled before their stage starts.

Collection uses one H200 sequentially. Switching roles releases the previous
model workers before loading the next model. Teacher reflection and its next
execution reuse the same loaded Teacher. GPU counts and TP remain configurable.
Run collection, SFT and evaluation on an allocated compute node.
The entry point rejects these commands on `orion-login01`.

Install `swanlab` and run `swanlab login`, or supply `SWANLAB_API_KEY` through
the environment. Collection, SFT, evaluation and summaries use separate SwanLab
runs. Local episode files remain the source for resuming collection and evaluation.

```bash
python -m scripts.teaching_experiment sample --config examples/teaching/config.yaml
python -m scripts.teaching_experiment collect --config examples/teaching/config.yaml
python -m scripts.teaching_experiment select
python -m scripts.teaching_experiment export-sft
bash examples/sft/teaching/run_alfworld.sh high
bash examples/sft/teaching/run_alfworld.sh low
python -m scripts.teaching_experiment sample --split valid_seen --per-type EVAL_TASKS_PER_TYPE
python -m scripts.teaching_experiment sample --split valid_unseen --per-type EVAL_TASKS_PER_TYPE
python -m scripts.teaching_experiment evaluate --model-path MODEL_PATH --label initial
python -m scripts.teaching_experiment summarize
```

Repeat evaluation with labels `high` and `low` using the final checkpoints in
their `training_summary.json` files and the same evaluation manifest.
The default evaluation reads both fixed manifests, runs one greedy episode
per task, and reports `valid_seen` and `valid_unseen` separately.
Choose the evaluation manifest size before sampling; `--all` selects the full
available split instead of a per-type count.
The exported `data_summary.json` files report required token lengths.
SFT uses all paired turns, no validation dataset, and the common configured
epochs. Its final summaries record actual processed turns, supervised tokens
and optimizer steps. SwanLab records Teacher attempts, per-version ICL success,
SFT loss and training volume, independent evaluation SR, and final comparisons.
SFT starts a new logging run on each launch and does not restore optimizer state.

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


