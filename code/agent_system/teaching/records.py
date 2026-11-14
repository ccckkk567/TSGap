"""Local records and the recorder attached to the existing rollout loop."""
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def completed_episode(path):
    if Path(path).exists():
        record = read_json(path)
        if record.get("status") == "complete":
            return record
    return None


class EpisodeRecorder:
    def __init__(self, metadata):
        self.record = dict(metadata, status="running", success=None, steps=[], num_steps=0)
        self.pending = None

    def start(self, envs, observations, infos):
        self.record.update(
            task=envs.tasks[0], initial_observation=observations["anchor"][0],
            initial_admissible_actions=list(infos[0]["admissible_commands"]),
            actual_game_file=infos[0].get("extra.gamefile"),
        )

    def before_step(self, envs, observations, responses, active):
        if bool(active[0]):
            self.pending = {
                "step": len(self.record["steps"]) + 1,
                "prompt_text": observations["text"][0],
                "observation": observations["anchor"][0],
                "admissible_actions": list(envs.envs.get_admissible_commands[0]),
                "response": responses[0],
            }

    def after_step(self, observations, rewards, dones, infos):
        if self.pending is None:
            return
        info = infos[0]
        self.pending.update(
            action=info["action"], next_observation=observations["anchor"][0],
            reward=float(rewards[0]), done=bool(dones[0]), won=bool(info["won"]),
            format_valid=bool(info["is_action_valid"]),
            action_available=bool(info["action_available"]),
        )
        self.record["steps"].append(self.pending)
        self.record["num_steps"] = len(self.record["steps"])
        self.pending = None

    def finish(self):
        last = self.record["steps"][-1]
        self.record.update(
            status="complete", success=last["won"],
            termination_reason="won" if last["won"] else ("environment_done" if last["done"] else "max_steps"),
        )
        return self.record

    def fail(self, error):
        self.record.update(status="error", success=None, termination_reason="system_error", error=str(error))
        if self.pending is not None:
            self.record["interrupted_step"] = self.pending
        return self.record


def render_feedback(episodes):
    parts = []
    for episode in episodes:
        parts.append(f"Rollout {episode['rollout_index']}: success={episode['success']}, "
                     f"termination={episode['termination_reason']}, steps={episode['num_steps']}")
        for step in episode["steps"]:
            parts.append(
                f"Step {step['step']}\nObservation: {step['observation']}\n"
                f"Admissible actions: {step['admissible_actions']}\n"
                f"Response: {step['response']}\nAction: {step['action']}\n"
                f"Environment returned: {step['next_observation']}\n"
                f"Reward: {step['reward']}; done={step['done']}; won={step['won']}"
            )
    return "\n\n".join(parts)
