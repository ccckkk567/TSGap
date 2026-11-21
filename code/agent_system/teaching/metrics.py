"""Metrics reconstructed from recorded environment decisions and model requests."""
from collections import Counter
from statistics import mean


def successful(ep, horizon):
    return ep["status"] == "complete" and ep["success"] is True and ep["num_steps"] <= horizon


def collect_metrics(episodes, tasks, group, budget, horizon, errors):
    counts = Counter(e["task_id"] for e in episodes)
    wins = [e for e in episodes if successful(e, horizon)]
    successes = Counter(e["task_id"] for e in wins)
    teacher = group in ("bc", "teacher")
    finished = [t["task_id"] for t in tasks if counts[t["task_id"]] == budget
                or (teacher and successes[t["task_id"]])]
    turns = [s for e in episodes for s in e["steps"]]
    attempted = [*episodes, *errors]
    reviews = [r for e in attempted for r in e["teacher_reviews"]]
    decisions = Counter(r["decision"] for r in reviews)
    applied = [r for r in reviews if r["applied_step"] is not None]
    result = {"collect/completed_rollouts": len(episodes), "collect/system_errors": len(errors),
              "collect/completed_tasks": len(finished), "collect/planned_tasks": len(tasks),
              "cost/environment_steps": sum(len(e["steps"]) for e in attempted)}
    if episodes:
        result["collect/success_rate"] = len(wins) / len(episodes)
        result["collect/mean_steps"] = mean(e["num_steps"] for e in episodes)
        if teacher:
            result["teacher/attempt_success_rate"] = result["collect/success_rate"]
    if wins:
        result["collect/success_mean_steps"] = mean(e["num_steps"] for e in wins)
    if finished:
        result["collect/task_coverage"] = mean(bool(successes[k]) for k in finished)
    if not teacher:
        for i in range(budget + 1):
            result[f"collect/successes_per_task/k_{i}"] = sum(successes[k] == i for k in finished)
    if turns:
        result["collect/format_invalid_rate"] = mean(not s["format_valid"] for s in turns)
        result["collect/action_unavailable_rate"] = mean(not s["action_available"] for s in turns)
    if group == "guided":
        result["teacher/review_calls"] = len(reviews)
        for key in ("feedback", "no_feedback", "error"):
            result[f"teacher/{key}_count"] = decisions[key]
        result["teacher/applied_count"] = len(applied)
        if len(reviews) > decisions["error"]:
            if any("finish_reason" in r for r in reviews):
                result["teacher/no_feedback_rate"] = decisions["no_feedback"] / (len(reviews) - decisions["error"])
            result["teacher/intervention_rate"] = len(applied) / (len(reviews) - decisions["error"])
        for guided in (True, False):
            subset = [e for e in episodes if any(s["generated_with_feedback"] for s in e["steps"]) == guided]
            if subset:
                result[f"teacher/episode_sr_{'with' if guided else 'without'}_feedback"] = mean(e["success"] for e in subset)
            steps = [s for s in turns if s["generated_with_feedback"] == guided]
            if steps:
                key = "after_feedback" if guided else "ordinary"
                result[f"behavior/{key}_format_valid"] = mean(s["format_valid"] for s in steps)
                result[f"behavior/{key}_available"] = mean(s["action_available"] for s in steps)
        result["behavior/repeated_invalid_actions"] = sum(
            a["action"] == b["action"] and not a["action_available"] and not b["action_available"]
            for e in episodes for a, b in zip(e["steps"], e["steps"][1:]))
    for role in ("teacher", "student"):
        requests = [s for e in attempted if e["source_actor"] == role for s in e["steps"]]
        requests.extend(e["interrupted_step"] for e in errors if e["source_actor"] == role and "interrupted_step" in e)
        if role == "teacher":
            requests.extend(reviews)
        finished = [r for r in requests if r.get("finish_reason") is not None]
        if finished:
            result[f"cost/{role}_truncated_requests"] = sum(r["finish_reason"] == "length" for r in finished)
        for key in ("input_tokens", "output_tokens"):
            result[f"cost/{role}_{key}"] = sum(r.get(key, 0) for r in requests)
    result["cost/review_seconds"] = sum(r.get("generation_seconds", 0) for r in reviews)
    return result


def evaluation_metrics(episodes):
    def stats(group):
        return dict(count=len(group), success_count=sum(e["success"] for e in group),
                    success_rate=mean(e["success"] for e in group),
                    mean_steps=mean(e["num_steps"] for e in group))
    kinds = sorted({e["task_type"] for e in episodes})
    return dict(**stats(episodes), by_type={k: stats([e for e in episodes if e["task_type"] == k]) for k in kinds})


def compare_evaluations(results):
    def compare(stats):
        scores = {label: s["success_rate"] for label, s in stats.items()}
        differences = {f"{a}_minus_{b}": scores[a] - scores[b] for a, b in (
            ("guided", "chosen"), ("guided", "teacher"), ("chosen", "bc"), ("teacher", "bc"), ("guided", "bc"))}
        return dict(scores=scores, differences=differences)
    baseline = results["bc"]
    settings = lambda r: {k: v for k, v in r["settings"].items() if k != "model_path"}
    if any(settings(r) != settings(baseline) for r in results.values()):
        raise ValueError("Evaluation manifests or settings differ")
    return dict(overall=compare(results), by_type={k: compare({g: r["by_type"][k] for g, r in results.items()})
                                                   for k in baseline["by_type"]})
