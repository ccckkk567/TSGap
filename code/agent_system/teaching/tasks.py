"""Fixed task manifests, using the existing ALFWorld candidate filtering."""
import os
import random
from pathlib import Path
from .records import read_json, read_jsonl, write_jsonl

TASK_TYPES = (
    "pick_and_place_simple", "look_at_obj_in_light",
    "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
)
SPLITS = {"train": "train", "valid_seen": "eval_in_distribution",
          "valid_unseen": "eval_out_of_distribution"}


def environment_config(config):
    import yaml
    from omegaconf import OmegaConf
    root = Path(os.path.expandvars(str(config.tasks.data_root))).expanduser().resolve()
    with open(config.env.config_path, encoding="utf-8") as stream:
        result = yaml.safe_load(stream)
    for key, value in result["dataset"].items():
        if isinstance(value, str):
            result["dataset"][key] = value.replace("$ALFWORLD_DATA", str(root))
    for key, value in result["logic"].items():
        result["logic"][key] = value.replace("$ALFWORLD_DATA", str(root))
    result["dataset"].update(num_train_games=-1, num_eval_games=-1)
    result["env"].update(domain_randomization=False, task_types=list(range(1, 7)))
    if not OmegaConf.is_missing(config.env, "max_steps"):
        for method in ("dagger", "rl"):
            result[method]["training"]["max_nb_steps_per_episode"] = config.env.max_steps
    return root, result


def _group_game_files(config, split):
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    root, settings = environment_config(config)
    pool = AlfredTWEnv(settings, train_eval=SPLITS[split]).game_files
    groups = {kind: [] for kind in TASK_TYPES}
    for path in sorted(Path(p).resolve().relative_to(root).as_posix() for p in pool):
        kind = read_json((root / path).parent / "traj_data.json")["task_type"]
        groups[kind].append(path)
    return groups


def sample_tasks(config, split, output, per_type=5, all_tasks=False):
    output = Path(output)
    if output.exists():
        return read_jsonl(output)
    groups = _group_game_files(config, split)
    rng = random.Random(config.tasks.sample_seed)
    rows = []
    for kind, paths in groups.items():
        selected = paths if all_tasks else rng.sample(paths, per_type)
        for path in selected:
            rows.append(dict(task_id=f"task_{len(rows):03d}", task_type=kind,
                             split=split, game_file=path))
    if not rows:
        raise ValueError("No ALFWorld tasks found. Check tasks.data_root.")
    write_jsonl(output, rows)
    return rows


def sample_student_targets(config, teacher_tasks, output):
    output = Path(output)
    if output.exists():
        return read_jsonl(output)
    groups = _group_game_files(config, "train")
    n = config.teaching.student_rollouts
    source_files = {row["game_file"] for row in teacher_tasks}
    rng = random.Random(config.tasks.sample_seed + 1)
    rows = []
    for kind in TASK_TYPES:
        source_group = [task for task in teacher_tasks if task["task_type"] == kind]
        available = [path for path in groups[kind] if path not in source_files]
        required = len(source_group) * n
        if len(available) < required:
            raise ValueError(f"Not enough Student targets for {kind}: need {required}, have {len(available)}")
        selected = rng.sample(available, required)
        for i, source in enumerate(source_group):
            for rollout_index in range(n):
                rows.append(dict(
                    source_task_id=source["task_id"], rollout_index=rollout_index,
                    task_id=f"icl_{source['task_id']}_{rollout_index}", task_type=kind,
                    split="train", game_file=selected[i * n + rollout_index],
                ))
    write_jsonl(output, rows)
    return rows
