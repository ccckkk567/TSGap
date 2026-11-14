"""Command entry point for the fixed ALFWorld teaching experiment."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import socket


def load_config(path):
    from omegaconf import OmegaConf
    config = OmegaConf.load(path)
    root = Path(config.run.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    saved = root / "config.yaml"
    if saved.exists():
        previous = OmegaConf.to_container(OmegaConf.load(saved), resolve=False)
        current = OmegaConf.to_container(config, resolve=False)
        pending = [(previous, current)]
        while pending:
            old, new = pending.pop()
            if isinstance(old, dict) and isinstance(new, dict) and old.keys() == new.keys():
                pending.extend((old[key], new[key]) for key in old)
            elif old != "???" and old != new:
                raise ValueError("A configured experiment setting changed. Use a new run.output_dir.")
    # Model-dependent fields can be filled after sampling/export, before use.
    OmegaConf.save(config, saved)
    return config


def train(config, group):
    from omegaconf import OmegaConf
    from agent_system.teaching.records import read_json
    directory = Path(config.run.output_dir) / "sft" / group
    if config.sft.n_gpus > config.hardware.n_gpus:
        raise ValueError("sft.n_gpus exceeds the configured job GPU allocation")
    summaries = [read_json(Path(config.run.output_dir) / "sft" / g / "data_summary.json") for g in ("high", "low")]
    if not summaries[0]["num_trajectories"]:
        raise ValueError("No paired trajectories available for SFT")
    if config.sft.max_length < max(s["max_length"] for s in summaries):
        raise ValueError("sft.max_length must accommodate both groups; see data_summary.json")
    base = Path(__file__).resolve().parents[1] / "verl/trainer/config/sft_trainer.yaml"
    training = OmegaConf.load(base)
    training.data.update(
        train_files=str(directory / "train.parquet"), val_files=None,
        prompt_key="extra_info", response_key="extra_info",
        prompt_dict_keys=["question"], response_dict_keys=["answer"],
        max_length=config.sft.max_length, truncation="error", drop_last=False,
        train_batch_size=config.sft.train_batch_size,
        micro_batch_size_per_gpu=config.sft.micro_batch_size_per_gpu,
        num_trajectories=summaries[0]["num_trajectories"],
    )
    training.model.partial_pretrain = config.sft.initial_model_path
    training.optim.lr = config.sft.learning_rate
    training.ulysses_sequence_parallel_size = config.sft.sequence_parallel_size
    training.use_remove_padding = True
    training.trainer.update(
        total_epochs=config.sft.epochs, total_training_steps=None, seed=config.sft.seed,
        default_local_dir=str(directory / "checkpoints"), default_hdfs_dir=None,
        logger=list(config.logging.backends), project_name=config.logging.project_name,
        experiment_name=f"{config.run.name}-sft-{group}",
        summary_path=str(directory / "training_summary.json"),
        swanlab_kwargs=dict(workspace=config.logging.workspace, mode=config.logging.mode,
                           logdir=config.logging.log_dir),
        group=group,
    )
    OmegaConf.resolve(training)
    path = directory / "trainer.yaml"
    OmegaConf.save(training, path)
    (directory / "training_summary.json").unlink(missing_ok=True)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc_per_node={config.sft.n_gpus}", "-m", "verl.trainer.fsdp_sft_trainer",
               "--config-path", str(directory.resolve()), "--config-name", "trainer"]
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["sample", "collect", "select", "export-sft", "train", "evaluate", "summarize"])
    parser.add_argument("--config", default="examples/teaching/config.yaml")
    parser.add_argument("--run-dir")
    parser.add_argument("--split", choices=["train", "valid_seen", "valid_unseen"])
    parser.add_argument("--output")
    parser.add_argument("--per-type", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--manifest", action="append")
    parser.add_argument("--model-path")
    parser.add_argument("--label", choices=["initial", "high", "low"])
    parser.add_argument("--group", choices=["high", "low"])
    args = parser.parse_args()
    # Resolve source-relative paths consistently for shell and module entry points.
    os.chdir(Path(__file__).resolve().parents[1])
    config_path = Path(args.run_dir) / "config.yaml" if args.run_dir else Path(args.config)
    config = load_config(config_path)
    if args.command in ("collect", "train", "evaluate"):
        hostname = socket.gethostname().split(".")[0]
        if hostname in config.hardware.login_nodes:
            parser.error(f"GPU work must run on an allocated compute node, not {hostname}")
    if args.command == "sample":
        from agent_system.teaching.tasks import sample_tasks
        split = args.split or config.tasks.split
        output = args.output or str(Path(config.run.output_dir) / ("manifest.jsonl" if split == "train" else f"manifest_{split}.jsonl"))
        if split != "train" and Path(output).resolve() == (Path(config.run.output_dir) / "manifest.jsonl").resolve():
            parser.error("Evaluation sampling must not overwrite the teaching manifest")
        rows = sample_tasks(config, split, output, args.per_type or config.tasks.per_type, args.all)
        print(f"Fixed manifest: {output} ({len(rows)} tasks)")
    elif args.command in ("select", "export-sft"):
        from agent_system.teaching.sft_data import select_pairs, export_sft
        if args.command == "select":
            print(f"Selected {len(select_pairs(config.run.output_dir))} paired tasks")
        else:
            export_sft(config.run.output_dir)
    elif args.command == "train":
        if args.group is None:
            parser.error("train requires --group high or low")
        train(config, args.group)
    else:
        from agent_system.teaching.experiment import collect, evaluate, summarize
        if args.command == "collect":
            collect(config)
        elif args.command == "evaluate":
            if not args.model_path or not args.label:
                parser.error("evaluate requires --model-path and --label")
            evaluate(config, args.model_path, args.label, args.manifest or list(config.evaluation.manifests))
        else:
            summarize(config)


if __name__ == "__main__":
    main()
