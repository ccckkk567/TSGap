"""Command entry point for the fixed ALFWorld teaching experiment."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import socket


def load_config(path, *, read_only=False):
    from omegaconf import OmegaConf
    config = OmegaConf.load(path)
    root = Path(config.run.output_dir)
    saved = root / "config.yaml"
    if read_only and not saved.is_file():
        raise FileNotFoundError(f"Missing configuration from BC: {saved}")
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
    if read_only:
        return config
    root.mkdir(parents=True, exist_ok=True)
    # Model-dependent fields can be filled after sampling/export, before use.
    temp = saved.with_suffix(".yaml.tmp")
    OmegaConf.save(config, temp)
    temp.replace(saved)
    return config


def train(config, group):
    from agent_system.teaching.records import read_json
    directory = Path(config.run.output_dir) / "sft" / group
    if config.sft.n_gpus > config.hardware.n_gpus:
        raise ValueError("sft.n_gpus exceeds the configured job GPU allocation")
    summaries = [read_json(Path(config.run.output_dir) / "sft" / g / "data_summary.json") for g in ("high", "low")]
    if not summaries[0]["num_trajectories"]:
        raise ValueError("No paired trajectories available for SFT")
    if config.sft.max_length < max(s["max_length"] for s in summaries):
        raise ValueError("sft.max_length must accommodate both groups; see data_summary.json")
    (directory / "training_summary.json").unlink(missing_ok=True)
    launch_sft(config, directory, config.sft.initial_model_path, directory / "checkpoints",
               group, None, config.sft.epochs, config.sft.learning_rate, config.sft.seed, 1)


def launch_sft(config, directory, input_checkpoint, checkpoint_work_dir, group, round_index,
               epochs, learning_rate, seed, training_attempt):
    from omegaconf import OmegaConf
    from agent_system.teaching.records import read_json
    summary = read_json(directory / "data_summary.json")
    max_length = config.sft.max_length if config.sft.max_length is not None else summary["max_length"]
    if max_length < summary["max_length"]:
        raise ValueError(f"sft.max_length too short for {directory}")
    experiment_name = f"{config.run.name}-sft-{group}"
    if round_index is not None:
        experiment_name += f"-r{round_index:03d}-attempt{training_attempt}"
    base = Path(__file__).resolve().parents[1] / "verl/trainer/config/sft_trainer.yaml"
    training = OmegaConf.load(base)
    training.data.update(
        train_files=str(directory / "train.parquet"), val_files=None,
        prompt_key="extra_info", response_key="extra_info",
        prompt_dict_keys=["question"], response_dict_keys=["answer"],
        max_length=max_length, truncation="error", drop_last=False,
        train_batch_size=config.sft.train_batch_size,
        micro_batch_size_per_gpu=config.sft.micro_batch_size_per_gpu,
        num_trajectories=summary["num_trajectories"],
        chat_template_kwargs=dict(config.sft.get("chat_template_kwargs", {})),
    )
    training.model.partial_pretrain = str(input_checkpoint)
    training.model.enable_gradient_checkpointing = config.sft.enable_gradient_checkpointing
    training.optim.lr = learning_rate
    training.ulysses_sequence_parallel_size = config.sft.sequence_parallel_size
    training.use_remove_padding = config.sft.use_remove_padding
    training.trainer.update(
        total_epochs=epochs, total_training_steps=None, seed=seed,
        default_local_dir=str(checkpoint_work_dir), default_hdfs_dir=None,
        logger=list(config.logging.backends), project_name=config.logging.project_name,
        experiment_name=experiment_name,
        summary_path=str(directory / "training_summary.json"),
        swanlab_kwargs=dict(workspace=config.logging.workspace, mode=config.logging.mode,
                           logdir=config.logging.log_dir),
        group=group,
    )
    OmegaConf.resolve(training)
    path = directory / "trainer.yaml"
    OmegaConf.save(training, path)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc_per_node={config.sft.n_gpus}", "-m", "verl.trainer.fsdp_sft_trainer",
               "--config-path", str(directory.resolve()), "--config-name", "trainer"]
    subprocess.run(command + [f"hydra.run.dir={directory.resolve()}"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["interactive", "interactive-summary", "sample", "collect", "baseline",
                                           "select", "export-sft", "train", "evaluate", "summarize"])
    parser.add_argument("--stage", choices=["bc", "main"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--run-dir")
    parser.add_argument("--task-id")
    parser.add_argument("--split", choices=["train", "valid_seen", "valid_unseen"])
    parser.add_argument("--output")
    parser.add_argument("--per-type", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--manifest", action="append")
    parser.add_argument("--model-path")
    parser.add_argument("--label", choices=["initial", "high", "low"])
    parser.add_argument("--group", choices=["high", "low", "chosen", "teacher", "guided"])
    args = parser.parse_args()
    # Resolve source-relative paths consistently for shell and module entry points.
    os.chdir(Path(__file__).resolve().parents[1])
    config_path = Path(args.run_dir) / "config.yaml" if args.run_dir else Path(args.config or (
        "examples/teaching/interactive_config.yaml" if args.command in ("interactive", "interactive-summary")
        else "examples/teaching/config.yaml"))
    branch_main = args.command == "interactive" and args.stage == "main"
    if branch_main and args.group not in ("chosen", "teacher", "guided"):
        parser.error("interactive --stage main requires --group chosen, teacher or guided")
    if args.command == "interactive" and args.stage == "bc" and args.group is not None:
        parser.error("--group applies only to the main stage of interactive")
    if args.command == "train" and args.group not in ("high", "low"):
        parser.error("train requires --group high or low")
    if args.command == "interactive" and not args.stage:
        parser.error("interactive requires --stage bc or main")
    if args.command == "interactive" and not args.resume:
        from omegaconf import OmegaConf
        if (Path(OmegaConf.load(config_path).run.output_dir) / "config.yaml").exists():
            parser.error("Use --resume for an existing interactive run")
    config = load_config(config_path, read_only=branch_main or args.command == "interactive-summary")
    if args.command in ("interactive", "collect", "baseline", "train", "evaluate"):
        hostname = socket.gethostname().split(".")[0]
        if hostname in config.hardware.login_nodes:
            parser.error(f"GPU work must run on an allocated compute node, not {hostname}")
    if args.command == "interactive":
        from agent_system.teaching.interactive import run_interactive
        run_interactive(config, args.stage, launch_sft, group=args.group)
    elif args.command == "interactive-summary":
        from agent_system.teaching.interactive import summarize_interactive
        summarize_interactive(config)
    elif args.command == "sample":
        from agent_system.teaching.tasks import sample_tasks, sample_student_targets
        split = args.split or config.tasks.split
        output = args.output or str(Path(config.run.output_dir) / ("manifest.jsonl" if split == "train" else f"manifest_{split}.jsonl"))
        if split != "train" and Path(output).resolve() == (Path(config.run.output_dir) / "manifest.jsonl").resolve():
            parser.error("Evaluation sampling must not overwrite the teaching manifest")
        rows = sample_tasks(config, split, output, args.per_type or config.tasks.per_type, args.all)
        print(f"Fixed manifest: {output} ({len(rows)} tasks)")
        if split == "train" and Path(output).resolve() == (Path(config.run.output_dir) / "manifest.jsonl").resolve():
            target_output = Path(config.run.output_dir) / "student_targets.jsonl"
            targets = sample_student_targets(config, rows, target_output)
            print(f"Fixed Student targets: {target_output} ({len(targets)} tasks)")
    elif args.command in ("select", "export-sft"):
        from agent_system.teaching.sft_data import select_pairs, export_sft
        if args.command == "select":
            print(f"Selected {len(select_pairs(config.run.output_dir))} paired tasks")
        else:
            export_sft(config.run.output_dir)
    elif args.command == "train":
        train(config, args.group)
    else:
        from agent_system.teaching.experiment import collect, baseline_no_icl, evaluate, summarize
        if args.command == "collect":
            collect(config)
        elif args.command == "baseline":
            if not args.task_id:
                parser.error("baseline requires --task-id")
            baseline_no_icl(config, args.task_id)
        elif args.command == "evaluate":
            if not args.model_path or not args.label:
                parser.error("evaluate requires --model-path and --label")
            evaluate(config, args.model_path, args.label, args.manifest or list(config.evaluation.manifests))
        else:
            summarize(config)


if __name__ == "__main__":
    main()
