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


def interactive_seed(config, phase, index, slot=0):
    return config.run.generation_seed + ((phase * 10000 + index) * 8 + slot) * 64


def prepare_interactive_manifests(config, *, allow_create=True):
    directory = Path(config.run.output_dir) / "manifests"
    manifests = {}
    excluded = set()
    pools = {}

    def manifest(name, split, per_type=None, full=None, seed_offset=0):
        target = directory / f"{name}.jsonl"
        if target.exists():
            rows = read_jsonl(target)
        else:
            if not allow_create:
                raise FileNotFoundError(f"Missing manifest from BC: {target}")
            rng = random.Random(config.tasks.sample_seed + seed_offset)
            rows = []
            if full is None and split not in pools:
                pools[split] = _group_game_files(config, split)
            for kind in TASK_TYPES:
                if full is not None:
                    rows.extend(rng.sample([r for r in full if r["task_type"] == kind], per_type))
                    continue
                available = [f for f in pools[split][kind] if f not in excluded]
                selected = available if per_type is None else rng.sample(available, per_type)
                for pos, game_file in enumerate(selected):
                    index = len(rows)
                    row = dict(task_id=f"{name}_{index:03d}", task_type=kind, split=split,
                               game_file=game_file, task_index=index)
                    if split != "train":
                        row["eval_index"] = index
                    if name == "train":
                        row["round_index"] = 1 + pos // config.interactive.tasks_per_type_per_round
                    rows.append(row)
        files = {r["game_file"] for r in rows}
        if not rows or len(files) != len(rows) or files & excluded:
            raise ValueError(f"Empty, duplicated or overlapping tasks in {name}")
        if per_type and any(sum(r["task_type"] == k for r in rows) != per_type for k in TASK_TYPES):
            raise ValueError(f"Incorrect per-type quota in {name}")
        if len({r["task_id"] for r in rows}) != len(rows) or len({r["task_index"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate identity in {name}")
        if name == "train":
            for n in range(1, config.interactive.rounds + 1):
                members = [r["task_type"] for r in rows if r["round_index"] == n]
                if any(members.count(k) != config.interactive.tasks_per_type_per_round for k in TASK_TYPES):
                    raise ValueError(f"Incorrect type quota in round {n}")
        if full is not None and any(r not in full for r in rows):
            raise ValueError(f"{name} does not match its full manifest")
        if allow_create and not target.exists():
            write_jsonl(target, rows)
        manifests[name] = rows
        return files

    excluded.update(manifest("bc", "train", config.bc.tasks_per_type))
    excluded.update(manifest("train", "train", config.interactive.rounds * config.interactive.tasks_per_type_per_round,
                             seed_offset=1))
    for split in ("valid_seen", "valid_unseen"):
        name = f"eval_{split}"
        manifest(name + "_full", split)
        manifest(name + "_subset", split, config.evaluation.subset_per_type, manifests[name + "_full"], 2)
    return manifests
