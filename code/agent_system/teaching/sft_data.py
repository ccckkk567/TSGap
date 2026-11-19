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
    config = OmegaConf.load(root / "config.yaml")
    tokenizer = hf_tokenizer(config.sft.initial_model_path)
    for group in ("high", "low"):
        rows, lengths, target_lengths = [], [], []
        for pair in pairs:
            version = pair[group]["version"]
            episode = read_json(root / pair["task_id"] / f"v{version:03d}" / "teacher.json")
            metadata = dict(trajectory_version=version, accepted_attempt=episode["accepted_attempt"], group=group)
            for row, length, target in episode_rows(episode, tokenizer, config.sft.get("chat_template_kwargs", {}), metadata):
                rows.append(row)
                lengths.append(length)
                target_lengths.append(target)
        directory = root / "sft" / group
        directory.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), directory / "train.parquet")
        write_json(directory / "data_summary.json", dict(
            num_trajectories=len(pairs), num_turns=len(rows), max_length=max(lengths),
            total_tokens=sum(lengths), supervised_tokens=sum(target_lengths),
        ))
    maximum = max(read_json(root / "sft" / g / "data_summary.json")["max_length"] for g in ("high", "low"))
    print(f"Exported {len(pairs)} paired trajectories per group. Common sft.max_length must be >= {maximum}.")


def episode_rows(episode, tokenizer, chat_kwargs, metadata):
    from agent_system.environments.prompts.alfworld import build_step_prompt
    for offset, step in enumerate(episode["steps"]):
        question = build_step_prompt(
            episode["task"], step["observation"], step["admissible_actions"],
            episode["steps"][:offset], step["step"],
        )
        answer = step["response"]
        chat = tokenizer.apply_chat_template([dict(role="user", content=question)],
                                              tokenize=False, add_generation_prompt=True,
                                              **chat_kwargs)
        answer_length = len(tokenizer.encode(answer + tokenizer.eos_token, add_special_tokens=False))
        length = len(tokenizer.encode(chat, add_special_tokens=False)) + answer_length
        info = dict(metadata, question=question, answer=answer, task_id=episode["task_id"], step=step["step"])
        if "episode_id" in metadata:
            info.update(review_id=step["review_id"], generated_with_feedback=step["generated_with_feedback"])
        yield dict(data_source="alfworld_teaching", extra_info=info), length, answer_length


def export_episode_sft(config, paths, directory):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from verl.utils import hf_tokenizer
    from .metrics import successful
    tokenizer = hf_tokenizer(config.sft.initial_model_path) if paths else None
    rows, lengths, targets, sources = [], [], [], []
    guided = invalid = unavailable = 0
    for path in paths:
        episode = read_json(path)
        if not successful(episode, config.env.max_steps):
            raise ValueError(f"Unexpected unselected episode: {path}")
        meta = {k: episode[k] for k in ("group", "round_index", "episode_id", "source_actor",
                                         "rollout_index", "attempt_index")}
        meta["episode_path"] = str(path)
        sources.append(dict(meta, task_id=episode["task_id"]))
        for row, length, supervised in episode_rows(episode, tokenizer, config.sft.chat_template_kwargs, meta):
            rows.append(row)
            lengths.append(length)
            targets.append(supervised)
        for step in episode["steps"]:
            guided += step["generated_with_feedback"]
            invalid += not step["format_valid"]
            unavailable += not step["action_available"]
    source_path = directory / "selected_episodes.jsonl"
    write_jsonl(source_path, sources)
    summary = dict(status="complete", selected_episodes=str(source_path), num_trajectories=len(paths),
                   num_tasks=len({s["task_id"] for s in sources}), num_turns=len(rows), max_length=max(lengths, default=0),
                   total_tokens=sum(lengths), supervised_tokens=sum(targets))
    if rows:
        temp = directory / "train.parquet.tmp"
        pq.write_table(pa.Table.from_pylist(rows), temp)
        temp.replace(directory / "train.parquet")
        summary.update(guided_turn_fraction=guided / len(rows), format_invalid_rate=invalid / len(rows),
                       action_unavailable_rate=unavailable / len(rows))
    write_json(directory / "data_summary.json", summary)
    return summary
