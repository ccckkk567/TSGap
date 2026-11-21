"""Round-based ALFWorld collection and supervised training on one allocated GPU."""
from contextlib import ExitStack, contextmanager
from pathlib import Path
from time import monotonic
import shutil

from .experiment import evaluate, phase_tracking
from .metrics import collect_metrics, compare_evaluations, evaluation_metrics, successful
from .records import completed_episode, read_json, read_jsonl, write_json
from .runtime import EpisodeSession, SequentialRollouts
from .sft_data import export_episode_sft
from .tasks import interactive_seed, prepare_interactive_manifests


def round_path(group, number):
    return Path(group) if group == "bc" else Path(group) / f"round_{number:03d}"


@contextmanager
def rollout_models(config):
    import os
    import ray
    options = dict(address=config.hardware.ray_address, runtime_env={"env_vars": {"VLLM_USE_V1": "0"}})
    if options["address"] is None:
        options["num_gpus"] = config.hardware.n_gpus
        job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
        options["_temp_dir"] = f"/tmp/tsgap-ray-{job_id}"
    ray.init(**options)
    models = SequentialRollouts(config, static_rollout=bool(config.interactive.get("static_rollout", False)))
    try:
        yield models
    finally:
        try:
            models.close()
        finally:
            ray.shutdown()


def start_session(config, runtime, task, path, group, number, phase, slot, input_checkpoint):
    teacher = group in ("bc", "teacher") and phase in (0, 2)
    index = task["task_index"]
    seed = interactive_seed(config, phase, task.get("eval_index", index), slot)
    meta = dict(group=group, round_index=number, episode_id=f"{group}-r{number}-{task['task_id']}-{slot}",
                source_actor=runtime.role, model_path=runtime.model_path, input_checkpoint=str(input_checkpoint),
                attempt_index=slot if teacher else None, rollout_index=None if teacher else slot,
                env_seed=config.env.seed, generation_settings=dict(runtime.action_sampling, seed=seed))
    return EpisodeSession(config, runtime, task, path, meta)


def collect_batch(config, models, group, number, tasks, directory, input_checkpoint):
    horizon = config.env.max_steps
    teaching = group in ("bc", "teacher")
    prefix = "attempt" if teaching else "rollout"
    budget = config.teaching.max_teacher_attempts if teaching else config.teaching.student_rollouts
    role = "teacher" if teaching else "student"
    model = config.teacher.model_path if teaching else str(input_checkpoint)
    phase = {"bc": 0, "teacher": 2, "chosen": 1, "guided": 1}[group]
    paths = []
    solved = set()
    # Teacher attempts are adaptive; Student slots are all independent.
    waves = [[i] for i in range(budget)] if teaching else [range(budget)]
    for slots in waves:
        with ExitStack() as stack:
            sessions = []
            for task in tasks:
                if task["task_id"] in solved:
                    continue
                for slot in slots:
                    path = directory / task["task_id"] / f"{prefix}_{slot}.json"
                    paths.append(path)
                    ep = completed_episode(path)
                    if ep is not None:
                        if teaching and successful(ep, horizon):
                            solved.add(task["task_id"])
                        continue
                    runtime = models.activate(role, model)
                    session = start_session(config, runtime, task, path, group, number, phase, slot, input_checkpoint)
                    stack.callback(session.close)
                    sessions.append((session, task, slot))
            checkpoints = list(config.teaching.review_steps) if group == "guided" else []
            for stop in checkpoints + [horizon]:
                if not sessions:
                    break
                runtime = models.activate(role, model)
                for session, task, _ in sessions:
                    session.advance(runtime, stop)
                    if session.closed and teaching and successful(session.recorder.record, horizon):
                        solved.add(task["task_id"])
                sessions = [entry for entry in sessions if not entry[0].closed]
                if sessions and stop in checkpoints:
                    reviewer = models.activate("teacher")
                    for session, task, slot in sessions:
                        seed = interactive_seed(config, 3, task["task_index"], slot) + checkpoints.index(stop)
                        session.review(reviewer, seed)
    return paths


def review_examples(episodes):
    examples = {}
    for ep in episodes:
        for review in ep["teacher_reviews"]:
            kind = review["decision"]
            if kind == "feedback":
                kind += "_success" if ep["success"] else "_failure"
            if kind in examples or kind == "error":
                continue
            following = ep["steps"][review["next_step"] - 1]
            examples[kind] = dict(task=ep["task_id"], review=review,
                                  response=following["response"], observation=following["next_observation"],
                                  success=ep["success"])
    return examples


def collect_round(config, group, number, tasks, input_checkpoint, *, tracking_namespace=None):
    import json
    from agent_system.environments.prompts.teacher_review import SYSTEM
    directory = Path(config.run.output_dir) / "collect" / round_path(group, number)
    width = config.interactive.active_tasks
    budget = (config.teaching.max_teacher_attempts if group in ("bc", "teacher")
              else config.teaching.student_rollouts)
    selected, episodes = [], []
    progress = directory / "collection_summary.json"
    old = read_json(progress) if progress.exists() else {}
    prior_switch = old.get("cost/model_switch_seconds", 0)
    prior_wall = old.get("cost/stage_wall_seconds", 0)
    started = monotonic()

    def progress_metrics(records, models):
        errors = [read_json(p) for p in directory.glob("*/*.error.json")]
        counts = collect_metrics(records, tasks, group, budget, config.env.max_steps, errors)
        counts["cost/model_switch_seconds"] = prior_switch + models.switch_seconds
        counts["cost/stage_wall_seconds"] = prior_wall + monotonic() - started
        if counts["collect/completed_tasks"] >= old.get("collect/completed_tasks", 0):
            write_json(progress, counts)
        return counts

    with (
        phase_tracking(config, f"collect-{group}-r{number:03d}", tracking_namespace=tracking_namespace) as log,
        rollout_models(config) as models,
    ):
        for offset in range(0, len(tasks), width):
            try:
                paths = collect_batch(config, models, group, number, tasks[offset:offset + width], directory,
                                      input_checkpoint)
            except Exception:
                records = [completed_episode(p) for p in directory.glob("*/*.json")
                           if not p.name.endswith(".error.json")]
                progress_metrics([ep for ep in records if ep is not None], models)
                raise
            for path in paths:
                ep = read_json(path)
                episodes.append(ep)
                if successful(ep, config.env.max_steps):
                    selected.append(path)
            counts = progress_metrics(episodes, models)
            if counts["collect/completed_tasks"] > old.get("collect/completed_tasks", 0):
                log(counts)
        if group == "guided" and "swanlab" in config.logging.backends:
            import swanlab
            samples = review_examples(episodes)
            if samples:
                log({f"examples/{k}": swanlab.Text(json.dumps(dict(system=SYSTEM, **v), ensure_ascii=False))
                     for k, v in samples.items()})
    return selected


def recover_training_result(directory, work, final, group, number, source):
    summary_file = directory / "training_summary.json"
    if not summary_file.exists():
        return None
    summary = read_json(summary_file)
    if summary.get("status") != "complete":
        return None
    saved = Path(summary["checkpoint"])
    if not final.is_dir():
        if saved.parent.resolve() != work.resolve() or not saved.name.startswith("global_step_") or not saved.is_dir():
            raise RuntimeError(f"Completed trainer output is missing: {saved}")
        final.parent.mkdir(parents=True, exist_ok=True)
        saved.rename(final)
    summary["checkpoint"] = str(final)
    write_json(summary_file, summary)
    marker = dict(status="complete", group=group, round_index=number, updated=True,
                  input_checkpoint=str(source), output_checkpoint=str(final),
                  data_summary=str(directory / "data_summary.json"), training_summary=str(summary_file))
    write_json(directory / "train_complete.json", marker)
    return marker


def train_round(config, group, number, tasks, source, train_fn, *, tracking_namespace=None):
    part = round_path(group, number)
    directory = Path(config.run.output_dir) / "sft" / part
    checkpoints = Path(config.run.checkpoint_dir)
    work, final = checkpoints / "work" / part, checkpoints / part
    marker = recover_training_result(directory, work, final, group, number, source)
    if marker:
        return marker
    if final.exists():
        raise RuntimeError(f"Published model without a complete trainer summary: {final}")
    summary_file = directory / "data_summary.json"
    summary = read_json(summary_file) if summary_file.exists() else {}
    if summary.get("status") != "complete" or (summary["num_turns"] and not (directory / "train.parquet").exists()):
        selected = collect_round(config, group, number, tasks, source, tracking_namespace=tracking_namespace)
        summary = export_episode_sft(config, selected, directory)
    if not summary["num_turns"]:
        if group == "bc":
            raise RuntimeError("No successful BC demonstration. Cannot establish the shared BC checkpoint.")
        marker = dict(status="complete", group=group, round_index=number, updated=False,
                      input_checkpoint=str(source), output_checkpoint=str(source), data_summary=str(summary_file),
                      training_summary=None)
        write_json(directory / "train_complete.json", marker)
        return marker
    if work.exists():
        shutil.rmtree(work)
    (directory / "training_summary.json").unlink(missing_ok=True)
    attempt_file = directory / "training_attempts.json"
    attempts = read_json(attempt_file) if attempt_file.exists() else []
    attempts.append(dict(attempt=len(attempts) + 1, status="started"))
    write_json(attempt_file, attempts)
    stage = config.bc if group == "bc" else config.sft
    started = monotonic()
    try:
        train_fn(config, directory, source, work, group, number, stage.epochs, stage.learning_rate,
                 config.sft.seed, len(attempts))
    finally:
        attempts[-1]["wall_seconds"] = monotonic() - started
        attempts[-1]["status"] = ("summary_written" if (directory / "training_summary.json").exists()
                                  else "interrupted")
        write_json(attempt_file, attempts)
    marker = recover_training_result(directory, work, final, group, number, source)
    if marker is None:
        raise RuntimeError(f"Trainer returned without a complete summary in {directory}")
    return marker


EVAL_SPLITS = ("valid_seen", "valid_unseen")


def evaluation_settings(config, model, split, tasks):
    return dict(tasks=tasks, sampling=dict(config.evaluation.sampling),
                seed_rule="base + ((phase*10000 + eval_index)*8 + slot)*64 + action_index",
                phase={"valid_seen": 4, "valid_unseen": 5}[split], base_seed=config.run.generation_seed,
                env_seed=config.env.seed, history_length=config.env.history_length, max_steps=config.env.max_steps,
                rollouts_per_task=config.evaluation.rollouts_per_task, model_path=str(model),
                demonstration="", teacher_guidance="")


def check_eval_summary(report, settings, path):
    tasks = settings["tasks"]
    budget = settings["rollouts_per_task"]
    kinds = {task["task_type"] for task in tasks}
    if (report["status"] != "complete" or report["settings"] != settings
            or report["count"] != len(tasks) * budget or set(report["by_type"]) != kinds):
        raise ValueError(f"Incomplete evaluation or mismatched settings: {path}")
    for kind, stats in report["by_type"].items():
        if stats["count"] != sum(t["task_type"] == kind for t in tasks) * budget:
            raise ValueError(f"Wrong evaluation count for {kind}: {path}")
    if any(key not in stats for stats in (report, *report["by_type"].values())
           for key in ("success_rate", "mean_steps")):
        raise ValueError(f"Missing evaluation metrics: {path}")


def evaluate_round(config, model, group, number, scope, manifests, excluded, *, tracking_namespace=None):
    root = Path(config.run.output_dir) / "eval" / group / f"round_{number:03d}"
    other = "subset" if scope == "full" else "full"
    totals = {}
    with ExitStack() as stack:
        runtime = log = None
        for manifest in manifests:
            tasks = read_jsonl(manifest)
            split = tasks[0]["split"]
            directory = root / scope / split
            settings = evaluation_settings(config, model, split, tasks)
            phase = settings["phase"]
            if any(t["game_file"] in excluded for t in tasks):
                raise ValueError(f"Evaluation overlaps training: {manifest}")
            saved = directory / "settings.json"
            if saved.exists() and read_json(saved) != settings:
                raise ValueError(f"Evaluation settings changed: {saved}")
            summary = directory / "summary.json"
            report = read_json(summary) if summary.exists() else {}
            if report.get("status") == "complete":
                check_eval_summary(report, settings, summary)
                totals[split] = report
                continue
            if not saved.exists() and any(directory.glob("*/episode_*.json")):
                raise ValueError(f"Cannot verify existing episodes without {saved}")
            write_json(saved, settings)
            if log is None:
                log = stack.enter_context(phase_tracking(config, f"eval-{group}-r{number:03d}-{scope}",
                                                         tracking_namespace=tracking_namespace))
            episodes = []
            reused = fresh_in = fresh_out = 0
            other_dir = root / other / split
            alt = other_dir / "settings.json"
            alt_settings = read_json(alt) if alt.exists() else {}
            compatible = {k:v for k,v in alt_settings.items() if k != "tasks"} == {k:v for k,v in settings.items() if k != "tasks"}
            alt_tasks = {r["task_id"]: r for r in alt_settings.get("tasks", [])}
            for task in tasks:
                for slot in range(config.evaluation.rollouts_per_task):
                    relative = Path(task["task_id"]) / f"episode_{slot}.json"
                    path = directory / relative
                    ep = completed_episode(path)
                    episode_source = path
                    if ep is None and compatible and alt_tasks.get(task["task_id"]) == task:
                        source = other_dir / relative
                        ep = completed_episode(source)
                        if ep is not None:
                            episode_source = source
                    if ep is not None:
                        if (any(ep.get(k) != v for k, v in task.items()) or ep.get("model_path") != str(model)
                                or ep["generation_settings"]["seed"] != interactive_seed(config, phase, task["eval_index"], slot)):
                            raise ValueError(f"Unexpected evaluation episode identity, checkpoint or seed: {episode_source}")
                        if episode_source != path:
                            ep = dict(ep, reused_from=str(episode_source))
                            write_json(path, ep)
                    if ep is None:
                        if runtime is None:
                            require_model(model)
                            models = stack.enter_context(rollout_models(config))
                            runtime = models.activate("student", str(model), config.evaluation.sampling)
                        session = start_session(config, runtime, task, path, group, number, phase, slot, model)
                        try:
                            session.advance(runtime, config.env.max_steps)
                        finally:
                            session.close()
                        ep = read_json(path)
                        log({f"eval/{split}/completed_rollouts": len(episodes) + 1,
                             f"eval/{split}/episode_success": int(ep["success"])})
                    episodes.append(ep)
                    if ep.get("reused_from"):
                        reused += 1
                    else:
                        fresh_in += sum(s["input_tokens"] for s in ep["steps"])
                        fresh_out += sum(s["output_tokens"] for s in ep["steps"])
            result = dict(status="complete", settings=settings, reused_episodes=reused,
                          generated_input_tokens=fresh_in, generated_output_tokens=fresh_out,
                          system_errors=sum(1 for _ in directory.glob("*/*.error.json")),
                          **evaluation_metrics(episodes))
            write_json(summary, result)
            log({f"eval/{split}/{key}": result[key] for key in ("success_rate", "count", "mean_steps", "system_errors")} |
                {f"eval/{split}/{kind}/{k}": v for kind, stats in result["by_type"].items() for k,v in stats.items()})
            totals[split] = result
    return totals


def read_marker(output, group, number):
    path = output / "sft" / round_path(group, number) / "train_complete.json"
    if path.exists():
        marker = read_json(path)
        if marker["status"] != "complete" or (marker["group"], marker["round_index"]) != (group, number):
            raise ValueError(f"Invalid training completion record: {path}")
        return marker


def require_model(path):
    if not Path(path).is_dir():
        raise FileNotFoundError(f"Required checkpoint is missing: {path}")


def cleanup(config, group, records):
    checkpoint_root = Path(config.run.checkpoint_dir)
    output = Path(config.run.output_dir)
    keep = {checkpoint_root / "bc", Path(records[-1]["output_checkpoint"])}
    for marker in records:
        n = marker["round_index"]
        part = round_path(group, n)
        work = checkpoint_root / "work" / part
        if work.is_dir():
            shutil.rmtree(work)
        if group != "bc" and n in config.evaluation.rounds:
            scope = "full" if n == config.interactive.rounds and config.evaluation.full_at_final else "subset"
            summaries = (output / "eval" / part / scope / split / "summary.json"
                         for split in ("valid_seen", "valid_unseen"))
            if any(not path.is_file() or read_json(path).get("status") != "complete" for path in summaries):
                keep.add(Path(marker["output_checkpoint"]))
    for marker in records:
        final = checkpoint_root / round_path(group, marker["round_index"])
        if marker["updated"] and final not in keep and final.is_dir():
            shutil.rmtree(final)


def curve_metrics(group, results):
    return {f"{group}/eval/subset/{split}/{kind}/{key}": value
            for split, report in results.items()
            for kind, stats in {"overall": report, **report["by_type"]}.items()
            for key, value in stats.items() if key in ("count", "success_rate", "mean_steps")}


def round_metrics(output, records):
    metrics = {}
    for group, history in records.items():
        marker = history[-1]
        part = round_path(group, marker["round_index"])
        summary = read_json(marker["data_summary"])
        counts = read_json(output / "collect" / part / "collection_summary.json")
        counts.update({f"data/{k}": v for k, v in summary.items() if isinstance(v, (int, float))})
        training = [read_json(m["training_summary"]) for m in history if m["updated"]]
        for key in ("processed_turns", "optimizer_steps", "supervised_tokens"):
            counts[f"train/{key}"] = training[-1][key] if marker["updated"] else 0
            counts[f"cumulative/{key}"] = sum(s[key] for s in training)
        attempts = output / "sft" / part / "training_attempts.json"
        if attempts.exists():
            runs = read_json(attempts)
            counts["cost/sft_attempts"] = len(runs)
            counts["cost/recorded_retraining_seconds"] = sum(r.get("wall_seconds", 0) for r in runs[:-1])
            if "wall_seconds" in runs[-1]:
                counts["cost/sft_wall_seconds"] = runs[-1]["wall_seconds"]
        metrics.update({f"{group}/{k}": v for k,v in counts.items()})
    return metrics


def record_bc_metrics(config, output, marker, results, *, tracking_namespace=None):
    path = output / "rounds" / "round_000.json"
    if not path.exists():
        metrics = round_metrics(output, {"bc": [marker]})
        metrics.update(curve_metrics("bc", results))
        with phase_tracking(config, "outer-rounds-bc", tracking_namespace=tracking_namespace) as log:
            log(metrics, step=0)
        write_json(path, metrics)


def bc_model(config, marker):
    if not marker:
        raise ValueError("Missing BC training completion record")
    path = marker["output_checkpoint"]
    if Path(path) != Path(config.run.checkpoint_dir) / "bc":
        raise ValueError(f"Unexpected BC checkpoint: {path}")
    require_model(path)
    return path


def require_bc_ready(config):
    """Validate the shared baseline without modifying artifacts or loading weights."""
    output = Path(config.run.output_dir)
    try:
        bc = read_marker(output, "bc", 0)
        model = bc_model(config, bc)
        manifests = prepare_interactive_manifests(config, allow_create=False)
        subset = {}
        for scope in ("full", "subset"):
            for split in EVAL_SPLITS:
                tasks = manifests[f"eval_{split}_{scope}"]
                path = output / "eval/bc/round_000" / scope / split / "summary.json"
                report = read_json(path)
                check_eval_summary(report, evaluation_settings(config, model, split, tasks), path)
                if scope == "subset":
                    subset[split] = report
        metrics = read_json(output / "rounds/round_000.json")
        expected = curve_metrics("bc", subset)
        if ("bc/train/optimizer_steps" not in metrics
                or any(metrics.get(k) != v for k, v in expected.items())):
            raise ValueError("BC round 0 metrics differ from the evaluated baseline")
        return bc, manifests
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise RuntimeError(f"BC training or baseline evaluation incomplete; resume BC first: {error}") from error


def run_interactive(config, stage, train_fn, *, group=None):
    from agent_system.environments.prompts.teacher_review import SYSTEM
    output = Path(config.run.output_dir)
    system_path = output / "prompts" / "teacher_review_system.txt"
    is_bc = stage == "bc"
    if is_bc:
        bc = read_marker(output, "bc", 0)
        manifests = prepare_interactive_manifests(config, allow_create=bc is None)
        system_path.parent.mkdir(parents=True, exist_ok=True)
        if not system_path.exists():
            temporary = system_path.with_suffix(".tmp")
            temporary.write_text(SYSTEM, encoding="utf-8")
            temporary.replace(system_path)
    else:
        if stage != "main" or group not in ("chosen", "teacher", "guided"):
            raise ValueError("Main requires a chosen, teacher or guided group")
        if list(config.interactive.groups) != ["chosen", "teacher", "guided"]:
            raise ValueError("Interactive groups must be chosen, teacher, guided in protocol order")
        bc, manifests = require_bc_ready(config)
    if system_path.read_text(encoding="utf-8") != SYSTEM:
        raise ValueError(f"Teacher review prompt differs from the saved BC input: {system_path}")
    excluded = {r["game_file"] for name in ("bc", "train") for r in manifests[name]}

    def assess(label, number, model, scope):
        paths = [output / "manifests" / f"eval_{split}_{scope}.jsonl" for split in EVAL_SPLITS]
        return evaluate(config, model, label, paths, round_index=number, scope=scope,
                        excluded_game_files=excluded, tracking_namespace=label)

    if is_bc:
        if bc is None:
            bc = train_round(config, "bc", 0, manifests["bc"], config.sft.initial_model_path, train_fn,
                             tracking_namespace="bc")
        model = bc_model(config, bc)
        cleanup(config, "bc", [bc])
        assess("bc", 0, model, "full")
        subset = assess("bc", 0, model, "subset")
        record_bc_metrics(config, output, bc, subset, tracking_namespace="bc")
        require_bc_ready(config)
        return

    total = config.interactive.rounds
    history = []
    for n in range(1, total + 1):
        marker = read_marker(output, group, n)
        if marker:
            if n != len(history) + 1:
                raise ValueError(f"Noncontiguous completed rounds for {group}")
            history.append(marker)
    if history:
        require_model(history[-1]["output_checkpoint"])
        cleanup(config, group, history)
    for n in range(1, total + 1):
        if len(history) < n:
            tasks = [t for t in manifests["train"] if t["round_index"] == n]
            source = history[-1]["output_checkpoint"] if history else bc["output_checkpoint"]
            history.append(train_round(config, group, n, tasks, source, train_fn, tracking_namespace=group))
            cleanup(config, group, history)
        model = history[n - 1]["output_checkpoint"]
        curve = {}
        if n in config.evaluation.rounds:
            if n == total and config.evaluation.full_at_final:
                assess(group, n, model, "full")
            curve = curve_metrics(group, assess(group, n, model, "subset"))
        summary_path = output / "rounds" / group / f"round_{n:03d}.json"
        if not summary_path.exists():
            metrics = round_metrics(output, {group: history[:n]})
            metrics.update(curve)
            with phase_tracking(config, f"outer-rounds-{group}", tracking_namespace=group) as log:
                log(metrics, step=n)
            write_json(summary_path, metrics)

    cleanup(config, group, history)


def summarize_interactive(config):
    output = Path(config.run.output_dir)
    scope = "full" if config.evaluation.full_at_final else "subset"
    comparison = {}
    problems = []
    for group in ("bc", "chosen", "teacher", "guided"):
        number = 0 if group == "bc" else config.interactive.rounds
        marker = read_marker(output, group, number)
        if marker is None:
            problems.append(f"Missing training completion record: {group} round {number}")
        comparison[group] = {}
        for split in ("valid_seen", "valid_unseen"):
            path = output / "eval" / group / f"round_{number:03d}" / scope / split / "summary.json"
            report = read_json(path) if path.is_file() else {}
            if report.get("status") != "complete":
                problems.append(f"Missing or incomplete evaluation: {path}")
            elif marker and report.get("settings", {}).get("model_path") != marker["output_checkpoint"]:
                problems.append(f"Evaluation checkpoint differs from training completion record: {path}")
            else:
                comparison[group][split] = report
    if problems:
        raise ValueError("\n".join(problems))
    result = {split: compare_evaluations({g: r[split] for g, r in comparison.items()})
              for split in ("valid_seen", "valid_unseen")}
    path = output / "final_comparison.json"
    final = dict(scope=scope, splits=result)
    if path.exists() and read_json(path) == final:
        print(f"Interactive comparison already complete: {path}")
        return
    with phase_tracking(config, "final-comparison", tracking_namespace="summary") as log:
        log({f"{scope}/{split}/{kind}/{k}": v for split, report in result.items()
             for kind, stats in {"overall": report["overall"], **report["by_type"]}.items()
             for k,v in (stats["scores"] | stats["differences"]).items()})
    write_json(path, final)
