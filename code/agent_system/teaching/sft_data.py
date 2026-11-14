"""Within-task pairing and shared-prompt turn export."""
from pathlib import Path
from .records import read_json, read_jsonl, write_json, write_jsonl


def select_pairs(run_dir):
    from omegaconf import OmegaConf
    root = Path(run_dir)
    budget = OmegaConf.load(root / "config.yaml").teaching.student_rollouts
    pairs = []
    for task in read_jsonl(root / "manifest.jsonl"):
        versions = []
        for path in sorted((root / task["task_id"]).glob("v*/result.json")):
            result = read_json(path)
            teacher = read_json(path.parent / "teacher.json")
            if teacher["success"] and result["num_rollouts"] == budget:
                versions.append(result)
        if not versions:
            continue
        high = max(versions, key=lambda r: (r["success_rate"], -r["version"]))
        low = min(versions, key=lambda r: (r["success_rate"], r["version"]))
        if high["success_rate"] == low["success_rate"]:
            continue
        pairs.append(dict(task, high=high, low=low))
    write_jsonl(root / "pairs.jsonl", pairs)
    return pairs


def export_sft(run_dir):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    root = Path(run_dir)
    pairs = read_jsonl(root / "pairs.jsonl")
    if not pairs:
        print("No paired tasks. SFT data is empty.")
        return
    from verl.utils import hf_tokenizer
    from agent_system.environments.prompts.alfworld import build_step_prompt
    config = OmegaConf.load(root / "config.yaml")
    tokenizer = hf_tokenizer(config.sft.initial_model_path)
    for group in ("high", "low"):
        rows, lengths, target_lengths = [], [], []
        for pair in pairs:
            version = pair[group]["version"]
            episode = read_json(root / pair["task_id"] / f"v{version:03d}" / "teacher.json")
            for offset, step in enumerate(episode["steps"]):
                question = build_step_prompt(
                    episode["task"], step["observation"], step["admissible_actions"],
                    episode["steps"][:offset], step["step"],
                )
                answer = step["response"]
                chat = tokenizer.apply_chat_template([dict(role="user", content=question)],
                                                      tokenize=False, add_generation_prompt=True)
                answer_length = len(tokenizer.encode(answer + tokenizer.eos_token, add_special_tokens=False))
                lengths.append(len(tokenizer.encode(chat, add_special_tokens=False)) + answer_length)
                target_lengths.append(answer_length)
                rows.append(dict(data_source="alfworld_teaching", extra_info=dict(
                    question=question, answer=answer, task_id=pair["task_id"],
                    trajectory_version=version, accepted_attempt=episode["accepted_attempt"],
                    step=step["step"], group=group,
                )))
        directory = root / "sft" / group
        directory.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), directory / "train.parquet")
        write_json(directory / "data_summary.json", dict(
            num_trajectories=len(pairs), num_turns=len(rows), max_length=max(lengths),
            total_tokens=sum(lengths), supervised_tokens=sum(target_lengths),
        ))
    maximum = max(read_json(root / "sft" / g / "data_summary.json")["max_length"] for g in ("high", "low"))
    print(f"Exported {len(pairs)} paired trajectories per group. Common sft.max_length must be >= {maximum}.")
