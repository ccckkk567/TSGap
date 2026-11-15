"""Serial teaching, independent evaluation, and summaries of local records."""
import csv
import json
import re
from contextlib import contextmanager, ExitStack
from pathlib import Path
from statistics import mean
from .records import (
    read_json, write_json, read_jsonl, completed_episode, render_feedback,
)

ANALYSIS_SEED_OFFSET = 1_000_000
LAST_STEPS_ON_OVERFLOW = 30


def reflection_input_length(runtime, prompt, config):
    rendered = runtime.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        **dict(config.teacher.reflection_chat_template_kwargs),
    )
    return len(runtime.tokenizer(rendered, add_special_tokens=False)["input_ids"])


def build_analysis_request(runtime, episode, config):
    """Fit one rollout by omitting only complete early steps when necessary."""
    from agent_system.environments.prompts.teacher_rollout_analysis import TEACHER_ROLLOUT_ANALYSIS_TEMPLATE

    def render(steps):
        first, last = steps[0]["step"], steps[-1]["step"]
        omitted = f"1–{first - 1}" if first > 1 else "none"
        coverage = (
            f"Rollout {episode['rollout_index']}; Student task: {episode['task']}; "
            f"Task type: {episode['task_type']}; Game file: {episode['game_file']}; "
            f"total steps={episode['num_steps']}; visible steps={first}–{last}; "
            f"omitted={omitted}; success={episode['success']}\n"
        )
        if first == 1:
            coverage += f"All steps 1–{last} are provided. No steps were omitted."
        else:
            coverage += (
                f"[Truncation Notice]\nThe original Student execution contains {episode['num_steps']} steps, "
                f"but only steps {first}–{last} are provided because the prompt exceeded the context limit. "
                f"Steps 1–{first - 1} are omitted.\n"
                "The initial observation, task identity, and final outcome are shown separately. "
                "Do not treat the visible suffix as the entire execution or invent observations, reasoning, "
                "actions, or outcomes for omitted steps. Base all detailed claims on the steps actually provided."
            )
        prompt = TEACHER_ROLLOUT_ANALYSIS_TEMPLATE.format(
            max_steps=config.env.max_steps, coverage_notice=coverage,
            student_execution_record=render_feedback([dict(episode, steps=steps)]),
        )
        return prompt, coverage

    prompt, coverage = render(episode["steps"])
    if reflection_input_length(runtime, prompt, config) <= config.teacher.prompt_length:
        return prompt, coverage
    steps = episode["steps"][-LAST_STEPS_ON_OVERFLOW:]
    while True:
        prompt, coverage = render(steps)
        length = reflection_input_length(runtime, prompt, config)
        if length <= config.teacher.prompt_length:
            return prompt, coverage
        if len(steps) == 1:
            raise ValueError(
                f"Analysis for rollout {episode['rollout_index']} exceeds prompt_length even with only "
                f"the last complete step: {length} > {config.teacher.prompt_length}"
            )
        steps = steps[1:]


def validate_analysis(text):
    sections = re.split(r"(?m)^## ([^\r\n]+)\r?$", text)
    expected = ["Execution Overview", "Chronological Execution Trace", "Key Decisions and Evidence",
                "Recurring Behavioral Patterns", "Uncertainties and Limitations"]
    if [heading.strip() for heading in sections[1::2]] != expected or any(
        not body.strip() for body in sections[2::2]
    ):
        raise ValueError("Student analysis is incomplete: expected all five sections with nonempty bodies.")


def analyze_student_episode(runtime, episode, directory, seed, config):
    """Reuse a valid analysis or save one independent Teacher generation."""
    index = episode["rollout_index"]
    path = directory / f"analysis_{index}.txt"
    request_path = directory / f"analysis_{index}_request.txt"
    if path.exists():
        analysis = path.read_text(encoding="utf-8")
        validate_analysis(analysis)
        request = request_path.read_text(encoding="utf-8")
        coverage = request.split("[Execution coverage]\n", 1)[1].split("\n\n[Student execution record]", 1)[0]
        return analysis, coverage
    prompt, coverage = build_analysis_request(runtime, episode, config)
    request_path.write_text(prompt, encoding="utf-8")
    analysis = runtime.reflect(prompt, seed)
    validate_analysis(analysis)
    path.write_text(analysis, encoding="utf-8")
    return analysis, coverage


@contextmanager
def phase_tracking(config, phase, fresh=False, extra=None):
    from omegaconf import OmegaConf
    from verl.utils.tracking import Tracking
    path = Path(config.run.output_dir) / "tracking_state.json"
    states = read_json(path) if path.exists() else {}
    if fresh:
        index = 1
        while f"{phase}-{index:02d}" in states:
            index += 1
        phase = f"{phase}-{index:02d}"
    state = states.get(phase, {"log_step": -1})
    options = dict(workspace=config.logging.workspace, mode=config.logging.mode,
                   logdir=config.logging.log_dir)
    if state.get("run_id"):
        options.update(id=state["run_id"], resume=config.logging.resume)
    public_config = OmegaConf.to_container(config, resolve=True)
    public_config.update(extra or {})
    tracking = Tracking(config.logging.project_name, f"{config.run.name}-{phase}",
                        list(config.logging.backends), config=public_config, swanlab_kwargs=options)
    if tracking.swanlab_run is not None:
        state["run_id"] = tracking.swanlab_run.id
    states[phase] = state
    write_json(path, states)

    def log(metrics, step=None):
        state["log_step"] = state["log_step"] + 1 if step is None else step
        tracking.log(metrics, step=state["log_step"])
        write_json(path, states)

    try:
        yield log
    finally:
        tracking.finish()


def generation_seed(config, task_index, version, slot):
    # Separate streams per episode, with room for all step requests.
    from omegaconf import OmegaConf
    evaluation_budget = 0 if OmegaConf.is_missing(config.evaluation, "rollouts_per_task") else config.evaluation.rollouts_per_task
    slots_per_version = config.teaching.max_teacher_attempts + max(config.teaching.student_rollouts, evaluation_budget) + 1
    stream = (task_index * config.teaching.num_versions + version) * slots_per_version + slot
    return config.run.generation_seed + stream * (config.env.max_steps + 1)


def run_episode(runtime_factory, envs, config, task, path, task_index, version,
                attempt=None, rollout_index=None, demonstration="", guidance=""):
    existing = completed_episode(path)
    if existing is not None:
        return existing, False
    runtime = runtime_factory()
    slot = attempt if attempt is not None else config.teaching.max_teacher_attempts + rollout_index
    metadata = dict(
        role="teacher" if attempt is not None else "student",
        trajectory_version=version, attempt=attempt, rollout_index=rollout_index,
        parent_version=version - 1 if version else None,
        guidance_from_version=version - 1 if attempt is not None and version else None,
        env_seed=config.env.seed,
        generation_settings=dict(runtime.action_sampling,
                                 seed=generation_seed(config, task_index, version, slot)),
    )
    return runtime.episode(envs, task, path, metadata, demonstration, guidance), True


def extract_guidance(text):
    marker = "## Guidance for the Next Trajectory"
    if "## Feedback Analysis" not in text or marker not in text or not text.split(marker, 1)[1].strip():
        raise ValueError("Reflection is incomplete: expected analysis and nonempty next-trajectory guidance.")
    return text.split(marker, 1)[1].strip()


def collect(config):
    import ray
    from omegaconf import OmegaConf
    from agent_system.environments.prompts.alfworld import render_demonstration, TEACHER_OBJECTIVE
    from agent_system.environments.prompts.teacher_reflection import TEACHER_REFLECTION_TEMPLATE
    from .runtime import SequentialRollouts, make_environment
    root = Path(config.run.output_dir)
    manifest = read_jsonl(root / "manifest.jsonl")
    target_path = root / "student_targets.jsonl"
    if not target_path.exists():
        raise FileNotFoundError(f"Missing Student targets: {target_path}. Run sample --split train first.")
    sources = {task["task_id"]: task for task in manifest}
    source_files = {task["game_file"] for task in manifest}
    student_targets, target_files = {}, set()
    expected_keys = {(task["task_id"], i) for task in manifest
                     for i in range(config.teaching.student_rollouts)}
    for row in read_jsonl(target_path):
        key = (row["source_task_id"], row["rollout_index"])
        if key in student_targets:
            raise ValueError(f"Duplicate Student target mapping: {key}")
        if key not in expected_keys:
            raise ValueError(f"Unexpected Student target mapping: {key}")
        if row["split"] != "train" or row["task_type"] != sources[row["source_task_id"]]["task_type"]:
            raise ValueError(f"Student target must be a train instance of the source task type: {key}")
        if row["game_file"] in source_files:
            raise ValueError(f"Student target overlaps a Teacher source: {row['game_file']}")
        if row["game_file"] in target_files:
            raise ValueError(f"Duplicate Student target game: {row['game_file']}")
        student_targets[key] = row
        target_files.add(row["game_file"])
    missing = expected_keys - student_targets.keys()
    if missing:
        raise ValueError(f"Missing Student target mappings: {sorted(missing)}")
    ray_options = {"address": config.hardware.ray_address, "ignore_reinit_error": True,
                   "runtime_env": {"env_vars": {"VLLM_USE_V1": "0"}}}
    if config.hardware.ray_address is None:
        ray_options["num_gpus"] = config.hardware.n_gpus
    ray.init(**ray_options)
    with ExitStack() as stack:
        stack.callback(ray.shutdown)
        required_gpus = max(config.teacher.n_gpus, config.student.n_gpus)
        if required_gpus > config.hardware.n_gpus or ray.available_resources().get("GPU", 0) < required_gpus:
            raise ValueError("The job must fit the larger of the Teacher and Student GPU requests")
        log = stack.enter_context(phase_tracking(config, "collect"))
        envs = make_environment(config)
        stack.callback(envs.envs.close)
        models = SequentialRollouts(config)
        stack.callback(models.close)
        for task_index, task in enumerate(manifest):
            guidance, previous = "", []
            for version in range(config.teaching.num_versions):
                directory = root / task["task_id"] / f"v{version:03d}"
                directory.mkdir(parents=True, exist_ok=True)
                coordinates = {"progress/task_index": task_index, "progress/version": version}
                accepted = completed_episode(directory / "teacher.json")
                if accepted is None:
                    for attempt in range(config.teaching.max_teacher_attempts):
                        episode, new = run_episode(
                            lambda: models.activate("teacher"), envs, config, task, directory / f"teacher_attempt_{attempt}.json",
                            task_index, version, attempt=attempt,
                            guidance=TEACHER_OBJECTIVE + ("\n\n" + guidance if guidance else ""),
                        )
                        if new:
                            log(dict(coordinates, **{
                                "progress/attempt": attempt, "teacher/attempt_number": attempt + 1,
                                "teacher/attempt_success": int(episode["success"]),
                                "teacher/attempt_steps": episode["num_steps"],
                            }))
                        if episode["success"]:
                            accepted = dict(episode, accepted_attempt=attempt)
                            write_json(directory / "teacher.json", accepted)
                            break
                if accepted is None:
                    status_path = directory / "status.json"
                    if not status_path.exists():
                        write_json(status_path, dict(status="teacher_budget_exhausted", version=version))
                        log(dict(coordinates, **{"teacher/budget_exhausted": 1}))
                    break
                demonstration = render_demonstration(accepted)
                (directory / "demo.txt").write_text(demonstration, encoding="utf-8")
                episodes = []
                for rollout_index in range(config.teaching.student_rollouts):
                    target_task = student_targets[(task["task_id"], rollout_index)]
                    episode, new = run_episode(
                        lambda: models.activate("student"), envs, config, target_task, directory / f"student_{rollout_index}.json",
                        task_index, version, rollout_index=rollout_index, demonstration=demonstration,
                    )
                    episodes.append(episode)
                    if new:
                        log(dict(coordinates, **{
                            "progress/rollout_index": rollout_index,
                            "student/rollout_success": int(episode["success"]),
                            "student/rollout_steps": episode["num_steps"],
                        }))
                result = dict(
                    version=version, parent_version=version - 1 if version else None,
                    accepted_attempt=accepted["accepted_attempt"],
                    success_flags=[int(e["success"]) for e in episodes],
                    success_count=sum(e["success"] for e in episodes), num_rollouts=len(episodes),
                    success_rate=mean(e["success"] for e in episodes),
                    steps=[e["num_steps"] for e in episodes],
                )
                result["delta_from_initial"] = result["success_rate"] - (previous[0]["result"]["success_rate"] if previous else result["success_rate"])
                result["delta_from_previous"] = result["success_rate"] - (previous[-1]["result"]["success_rate"] if previous else result["success_rate"])
                result["mean_steps"] = mean(result["steps"])
                successful = [e["num_steps"] for e in episodes if e["success"]]
                if successful:
                    result["success_mean_steps"] = mean(successful)
                if not (directory / "result.json").exists():
                    write_json(directory / "result.json", result)
                    keys = ("success_count", "num_rollouts", "success_rate", "delta_from_initial",
                            "delta_from_previous", "mean_steps", "success_mean_steps")
                    metrics = {f"icl/{k}": result[k] for k in keys if k in result}
                    metrics[f"icl/by_task/{task['task_id']}/success_rate"] = result["success_rate"]
                    log(dict(coordinates, **metrics))
                if version == config.teaching.num_versions - 1:
                    break
                reflection_path = directory / "reflection.txt"
                if reflection_path.exists():
                    reflection = reflection_path.read_text(encoding="utf-8")
                    guidance = extract_guidance(reflection)
                else:
                    teacher_runtime = models.activate("teacher")
                    reflection_slot = config.teaching.max_teacher_attempts + config.teaching.student_rollouts
                    reflection_seed = generation_seed(config, task_index, version, reflection_slot)
                    analysis_blocks = []
                    for episode in episodes:
                        i = episode["rollout_index"]
                        analysis, coverage = analyze_student_episode(
                            teacher_runtime, episode, directory,
                            reflection_seed + ANALYSIS_SEED_OFFSET + i, config,
                        )
                        analysis_blocks.append(f"[Rollout {i}: analysis coverage and task identity]\n{coverage}\n\n{analysis}")
                    target_results = [{key: episode[key] for key in
                                       ("rollout_index", "task", "task_type", "game_file", "success", "num_steps")}
                                      for episode in episodes]
                    prompt = TEACHER_REFLECTION_TEMPLATE.format(
                        task_and_execution_settings=json.dumps(dict(
                            task=accepted["task"], max_steps=config.env.max_steps,
                            history_length=2, student_generation=OmegaConf.to_container(config.student.action, resolve=True),
                        ), ensure_ascii=False),
                        trajectory_version=version, current_demonstration_text=demonstration,
                        rollout_results=json.dumps(dict(result, target_results=target_results), ensure_ascii=False),
                        student_rollout_analyses="\n\n".join(analysis_blocks),
                        previous_round_records=json.dumps(previous, ensure_ascii=False),
                    )
                    (directory / "reflection_request.txt").write_text(prompt, encoding="utf-8")
                    length = reflection_input_length(teacher_runtime, prompt, config)
                    if length > config.teacher.prompt_length:
                        raise ValueError(f"Global Reflection input exceeds prompt_length: {length} > {config.teacher.prompt_length}")
                    reflection = teacher_runtime.reflect(prompt, reflection_seed)
                    guidance = extract_guidance(reflection)
                    reflection_path.write_text(reflection, encoding="utf-8")
                write_json(directory / "guidance.json", {"guidance": guidance, "from_version": version})
                previous.append(dict(version=version, result=result, reflection=reflection))


def baseline_no_icl(config, task_id):
    import ray
    from .runtime import RolloutRuntime, make_environment
    root = Path(config.run.output_dir)
    manifest = read_jsonl(root / "manifest.jsonl")
    task_index = next((i for i, task in enumerate(manifest) if task["task_id"] == task_id), None)
    if task_index is None:
        raise ValueError(f"Source task not found in manifest.jsonl: {task_id}")
    targets = sorted(
        (row for row in read_jsonl(root / "student_targets.jsonl") if row["source_task_id"] == task_id),
        key=lambda row: row["rollout_index"],
    )
    if [row["rollout_index"] for row in targets] != [0, 1, 2, 3]:
        raise ValueError(f"No-ICL baseline requires exactly rollout indices 0, 1, 2, 3 for {task_id}")
    icl_result = read_json(root / task_id / "v000" / "result.json")
    directory = root / "baselines" / "no_icl" / task_id
    ray_options = {"address": config.hardware.ray_address, "ignore_reinit_error": True,
                   "runtime_env": {"env_vars": {"VLLM_USE_V1": "0"}}}
    if config.hardware.ray_address is None:
        ray_options["num_gpus"] = config.hardware.n_gpus
    ray.init(**ray_options)
    with ExitStack() as stack:
        stack.callback(ray.shutdown)
        log = stack.enter_context(phase_tracking(
            config, f"baseline-no-icl-{task_id}",
            extra={"condition": "no_icl", "source_task_id": task_id},
        ))
        envs = make_environment(config)
        stack.callback(envs.envs.close)
        student_runtime = RolloutRuntime(config, "student")
        stack.callback(student_runtime.close)
        episodes = []
        print(f"No-ICL baseline: {task_id}", flush=True)
        for target_task in targets:
            rollout_index = target_task["rollout_index"]
            episode, new = run_episode(
                lambda: student_runtime, envs, config, target_task,
                directory / f"student_{rollout_index}.json",
                task_index=task_index, version=0, rollout_index=rollout_index,
                demonstration="", guidance="",
            )
            episodes.append(episode)
            print(f"Student {rollout_index}: success={episode['success']}, "
                  f"steps={episode['num_steps']}, seed={episode['generation_settings']['seed']}", flush=True)
            if new:
                log({"progress/rollout_index": rollout_index,
                     "baseline_no_icl/rollout_success": int(episode["success"]),
                     "baseline_no_icl/rollout_steps": episode["num_steps"]})
        result = dict(
            condition="no_icl", source_task_id=task_id, compared_icl_version=0,
            success_flags=[int(e["success"]) for e in episodes],
            success_count=sum(e["success"] for e in episodes), num_rollouts=len(episodes),
            success_rate=mean(e["success"] for e in episodes),
            steps=[e["num_steps"] for e in episodes],
            mean_steps=mean(e["num_steps"] for e in episodes),
        )
        successful = [e["num_steps"] for e in episodes if e["success"]]
        if successful:
            result["success_mean_steps"] = mean(successful)
        write_json(directory / "result.json", result)
        log({f"baseline_no_icl/{key}": result[key] for key in ("success_count", "success_rate")})
        print(f"\nNo-ICL SR: {result['success_count']}/{result['num_rollouts']} = {result['success_rate']}", flush=True)
        print(f"ICL v000 SR: {icl_result['success_count']}/{icl_result['num_rollouts']} = {icl_result['success_rate']}", flush=True)
        print(f"Delta SR (ICL - No-ICL): {icl_result['success_rate'] - result['success_rate']}", flush=True)


def evaluate(config, model_path, label, manifests):
    import ray
    from omegaconf import OmegaConf
    from .runtime import RolloutRuntime, make_environment
    root = Path(config.run.output_dir)
    train_games = {r["game_file"] for r in read_jsonl(root / "manifest.jsonl")}
    by_split = {}
    for manifest in manifests:
        for task in read_jsonl(manifest):
            if task["game_file"] in train_games:
                raise ValueError(f"Evaluation task overlaps teaching manifest: {task['game_file']}")
            by_split.setdefault(task["split"], []).append(task)
    if not by_split:
        raise ValueError("Provide at least one fixed evaluation manifest")
    ray_options = {"address": config.hardware.ray_address, "ignore_reinit_error": True,
                   "runtime_env": {"env_vars": {"VLLM_USE_V1": "0"}}}
    if config.hardware.ray_address is None:
        ray_options["num_gpus"] = config.hardware.n_gpus
    ray.init(**ray_options)
    with ExitStack() as stack:
        stack.callback(ray.shutdown)
        log = stack.enter_context(phase_tracking(config, f"eval-{label}", extra={"model_label": label, "model_path": model_path}))
        envs = make_environment(config)
        stack.callback(envs.envs.close)
        runtime = RolloutRuntime(config, "student", model_path=model_path, sampling=config.evaluation.sampling)
        stack.callback(runtime.close)
        for split, tasks in by_split.items():
            if len({t["game_file"] for t in tasks}) != len(tasks) or len({t["task_id"] for t in tasks}) != len(tasks):
                raise ValueError(f"Duplicate task IDs or game files in evaluation split {split}")
            directory = root / "eval" / label / split
            settings = dict(tasks=tasks, sampling=OmegaConf.to_container(config.evaluation.sampling, resolve=True),
                            rollouts_per_task=config.evaluation.rollouts_per_task, env_seed=config.env.seed,
                            max_steps=config.env.max_steps, model_path=model_path)
            settings_path = directory / "settings.json"
            if settings_path.exists() and read_json(settings_path) != settings:
                raise ValueError("Evaluation settings changed. Use a new run directory.")
            write_json(settings_path, settings)
            episodes = []
            for index, task in enumerate(tasks):
                for rollout_index in range(config.evaluation.rollouts_per_task):
                    episode, new = run_episode(
                        lambda: runtime, envs, config, task, directory / task["task_id"] / f"episode_{rollout_index}.json",
                        index, 0, rollout_index=rollout_index,
                    )
                    episodes.append(episode)
                    if new:
                        log({f"eval/{split}/episode_success": int(episode["success"]),
                             f"eval/{split}/episode_steps": episode["num_steps"],
                             f"eval/{split}/running_success_rate": mean(e["success"] for e in episodes),
                             f"eval/{split}/completed_rollouts": len(episodes)})
            summary = dict(status="complete", success_rate=mean(e["success"] for e in episodes),
                           success_count=sum(e["success"] for e in episodes), num_rollouts=len(episodes),
                           mean_steps=mean(e["num_steps"] for e in episodes), settings=settings)
            write_json(directory / "summary.json", summary)
            log({f"eval/{split}/{key}": summary[key] for key in
                 ("success_rate", "success_count", "num_rollouts", "mean_steps")})


def summarize(config):
    root = Path(config.run.output_dir)
    tasks = read_jsonl(root / "manifest.jsonl")
    rows = []
    with phase_tracking(config, "summary", fresh=True) as log:
        for version in range(config.teaching.num_versions):
            metrics, values, by_type = {}, [], {}
            for task in tasks:
                path = root / task["task_id"] / f"v{version:03d}" / "result.json"
                if not path.exists():
                    continue
                result = read_json(path)
                rows.append(dict(task_id=task["task_id"], task_type=task["task_type"],
                                 version=version, success_rate=result["success_rate"],
                                 num_rollouts=result["num_rollouts"]))
                values.append(result["success_rate"])
                by_type.setdefault(task["task_type"], []).append(result["success_rate"])
                for key in ("success_rate", "delta_from_initial"):
                    metrics[f"icl/by_task/{task['task_id']}/{key}"] = result[key]
            if values:
                metrics.update({"icl/by_version/mean_success_rate": mean(values),
                                "icl/by_version/num_tasks": len(values)})
                for kind, rates in by_type.items():
                    metrics[f"icl/by_type/{kind}/success_rate"] = mean(rates)
                    metrics[f"icl/by_type/{kind}/num_tasks"] = len(rates)
                log(metrics, step=version)
        with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["task_id", "task_type", "version", "success_rate", "num_rollouts"])
            writer.writeheader()
            writer.writerows(rows)
        pairs = read_jsonl(root / "pairs.jsonl") if (root / "pairs.jsonl").exists() else []
        final = {"selection/num_pairs": len(pairs), "selection/high_trajectories": len(pairs),
                 "selection/low_trajectories": len(pairs)}
        for group in ("high", "low"):
            for filename in ("data_summary.json", "training_summary.json"):
                path = root / "sft" / group / filename
                if path.exists():
                    for key, value in read_json(path).items():
                        if isinstance(value, (float, int)):
                            final[f"{group}/{filename.split('.')[0]}/{key}"] = value
        splits = {p.parent.name for p in (root / "eval").glob("*/*/summary.json")}
        for split in splits:
            results = {}
            for label in ("initial", "high", "low"):
                path = root / "eval" / label / split / "summary.json"
                if path.exists():
                    result = read_json(path)
                    if result["status"] == "complete":
                        results[label] = result
            settings = [{k: v for k, v in r["settings"].items() if k != "model_path"} for r in results.values()]
            if settings and any(s != settings[0] for s in settings):
                raise ValueError(f"Evaluation settings differ for {split}")
            for label, result in results.items():
                final[f"comparison/{split}/sr_{label}"] = result["success_rate"]
            for key, left, right in (("delta_sft", "high", "low"), ("gain_high", "high", "initial"), ("gain_low", "low", "initial")):
                if left in results and right in results:
                    final[f"comparison/{split}/{key}"] = results[left]["success_rate"] - results[right]["success_rate"]
        log(final, step=config.teaching.num_versions)
        write_json(root / "summary.json", dict(metrics=final, alignment="Paired tasks and trajectory count; turns, tokens and optimizer steps may differ."))
