"""Existing rollout-only Ray workers and the shared vanilla environment loop."""
from pathlib import Path
from itertools import count
from time import monotonic
from .records import EpisodeRecorder, write_json

_runtime_numbers = count()


class SequentialRollouts:
    """Keep one model worker group resident and switch only when needed."""
    def __init__(self, config, *, static_rollout=False):
        self.config = config
        self.static_rollout = bool(static_rollout)
        self.current = None

        self.switch_seconds = 0.0

    def activate(self, role, model_path=None, sampling=None):
        path = str(model_path or self.config[role].model_path)
        if self.current is None or (self.current.role, self.current.model_path) != (role, path):
            start = monotonic()
            self.close()
            self.current = RolloutRuntime(self.config, role, model_path=path, sampling=sampling,
                                          static_rollout=self.static_rollout)
            self.switch_seconds += monotonic() - start
        self.current.action_sampling = dict(sampling or self.config[role].action)
        self.current.sampling = self.current.action_sampling
        return self.current

    def close(self):
        if self.current is not None:
            self.current.close()
            self.current = None


class RolloutRuntime:
    def __init__(self, config, role, model_path=None, sampling=None, *, static_rollout=False):
        import ray
        from omegaconf import OmegaConf
        from verl.utils import hf_tokenizer
        from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
        settings = config[role]
        if settings.n_gpus > config.hardware.n_gpus:
            raise ValueError(f"{role}.n_gpus exceeds the configured job GPU allocation")
        if settings.n_gpus % settings.tensor_parallel_size:
            raise ValueError(f"{role}.n_gpus must be divisible by tensor_parallel_size")
        if ray.available_resources().get("GPU", 0) < settings.n_gpus:
            raise ValueError(f"Insufficient free GPUs for {role}: requested {settings.n_gpus}")
        self.config = config
        self.role = role
        self.model_path = str(model_path or settings.model_path)
        self.action_sampling = OmegaConf.to_container(sampling or settings.action, resolve=True)
        self.action_chat_template_kwargs = dict(settings.get("chat_template_kwargs", {}))
        capacity = self.action_sampling["max_tokens"]
        if role == "teacher":
            capacity = max([capacity] + [settings[name].max_tokens for name in ("review", "reflection") if name in settings])
        backend = config.get("interactive", {}).get("inference_backend", "legacy")
        if backend not in ("legacy", "direct"):
            raise ValueError(f"Unknown inference backend: {backend}")
        self.direct = backend == "direct"
        tokenizer_path = self.model_path
        if not self.direct:
            from verl.utils.fs import copy_to_local
            tokenizer_path = copy_to_local(tokenizer_path)
        self.tokenizer = hf_tokenizer(tokenizer_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.collector = TrajectoryCollector(OmegaConf.create({
            "data": {"max_prompt_length": settings.prompt_length, "return_raw_chat": True},
            "env": {"max_steps": config.env.max_steps, "rollout": {"n": 1}},
        }), self.tokenizer, chat_template_kwargs=self.action_chat_template_kwargs)
        if self.direct:
            from .inference import InferenceWorker
            if settings.n_gpus != 1 or settings.tensor_parallel_size != 1:
                raise ValueError("Direct inference requires one GPU and TP=1")
            self.worker = ray.remote(num_gpus=1)(InferenceWorker).remote(
                self.model_path, settings.prompt_length + capacity, settings.gpu_memory_utilization)
            # Include engine initialization in the model switching time.
            ray.get(self.worker.__ray_ready__.remote())
        else:
            self._init_legacy(settings, capacity, static_rollout)
        self.request_index = 0
        self.sampling = self.action_sampling
        self.episode_seed = 0

    def _init_legacy(self, settings, capacity, static_rollout):
        import ray
        from omegaconf import OmegaConf
        from verl.workers.fsdp_workers import ActorRolloutRefWorker
        from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
        base = Path(__file__).resolve().parents[2] / "verl/trainer/config/generation.yaml"
        worker_config = OmegaConf.load(base)
        worker_config.model.path = self.model_path
        worker_config.rollout.update(
            static_rollout=bool(static_rollout),
            seed=int(self.config.run.generation_seed),
            n=1, prompt_length=settings.prompt_length, response_length=capacity,
            tensor_model_parallel_size=settings.tensor_parallel_size,
            gpu_memory_utilization=settings.gpu_memory_utilization,
            max_model_len=settings.prompt_length + capacity,
            max_num_batched_tokens=settings.prompt_length + capacity,
            enable_chunked_prefill=False,
        )
        self.name_prefix = f"teaching_{self.role}_{next(_runtime_numbers)}"
        self.pool = RayResourcePool([settings.n_gpus], name_prefix=self.name_prefix, max_colocate_count=1)
        worker = RayClassWithInitArgs(ray.remote(ActorRolloutRefWorker), config=worker_config, role="rollout")
        self.group = RayWorkerGroup(self.pool, worker, name_prefix=self.name_prefix)
        print(f"Initializing {self.role} rollout: checkpoint={self.model_path}, static_rollout={static_rollout}", flush=True)
        self.group.init_model()

    def generate_sequences(self, data):
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        data.meta_info["sampling_overrides"] = dict(
            self.sampling, n=1, seed=self.episode_seed + self.request_index,
        )
        self.request_index += 1
        started = monotonic()
        if self.direct:
            return self._generate_direct(data, data.meta_info["sampling_overrides"], started)
        padded, padding = pad_dataproto_to_divisor(data, self.group.world_size)
        output = unpad_dataproto(self.group.generate_sequences(padded), padding)
        width = output.batch["responses"].shape[-1]
        mask = output.batch["attention_mask"][0]
        self.last_generation = dict(generation_seed=self.episode_seed + self.request_index - 1,
                                    input_tokens=int(mask[:-width].sum()),
                                    output_tokens=min(int(mask[-width:].sum()), self.sampling["max_tokens"]),
                                    generation_seconds=monotonic() - started)
        return output

    def _generate_direct(self, data, sampling, started):
        import ray
        import torch
        from verl import DataProto

        prompts = [list(map(int, ids)) for ids in data.non_tensor_batch["raw_prompt_ids"]]
        results = ray.get(self.worker.generate.remote(prompts, sampling))
        width = max(1, max(len(ids) for ids, _, _ in results))
        responses = torch.full((len(results), width), self.tokenizer.pad_token_id, dtype=torch.long)
        response_mask = torch.zeros_like(responses)
        visible = []
        for i, (ids, finish, stop) in enumerate(results):
            responses[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            response_mask[i, :len(ids)] = 1
            # Preserve reasoning delimiters even if the tokenizer marks them special.
            terminal = (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id)
            content = ids[:-1] if ids and ids[-1] in terminal else ids
            raw = self.tokenizer.decode(content, skip_special_tokens=False)
            tail = self.tokenizer.decode(prompts[i][-16:], skip_special_tokens=False)
            prefix = tail[tail.rfind("<think>"):] if tail.rstrip().endswith("<think>") else ""
            visible.append(prefix + raw if not raw.lstrip().startswith("<think>") else raw)
            if i == 0:  # EpisodeSession and review each issue one request at a time.
                self.last_generation = dict(
                    generation_seed=sampling["seed"], input_tokens=len(prompts[i]),
                    output_tokens=len(ids), generation_seconds=monotonic() - started,
                    finish_reason=finish, stop_reason=stop, raw_completion=raw)
        mask = torch.cat((data.batch["attention_mask"], response_mask), dim=1)
        return DataProto.from_dict(tensors={
            "responses": responses, "prompts": data.batch["input_ids"],
            "input_ids": torch.cat((data.batch["input_ids"], responses), dim=1),
            "attention_mask": mask,
        }, meta_info={"visible_responses": visible})

    def initial_batch(self):
        import numpy as np
        import torch
        from verl import DataProto
        raw = np.empty(1, dtype=object)
        raw[0] = [dict(role="user", content="")]
        return DataProto.from_dict(
            tensors={"input_ids": torch.zeros((1, 1), dtype=torch.long)},
            non_tensors={"raw_prompt": raw, "data_source": np.array(["alfworld"], dtype=object)},
        )

    def episode(self, envs, task, path, metadata, demonstration="", guidance=""):
        from .tasks import environment_config
        root, _ = environment_config(self.config)
        recorder = EpisodeRecorder(dict(task, **metadata))
        self.episode_seed = metadata["generation_settings"]["seed"]
        self.request_index = 0
        self.sampling = self.action_sampling
        try:
            self.collector.chat_template_kwargs = self.action_chat_template_kwargs
            self.collector.vanilla_multi_turn_loop(
                self.initial_batch(), self, envs,
                reset_kwargs={
                    "task_specs": [dict(game_file=str(root / task["game_file"]), env_seed=metadata["env_seed"])],
                    "episode_contexts": [dict(demonstration=demonstration, teacher_guidance=guidance)],
                },
                episode_recorder=recorder,
            )
            record = recorder.finish()
        except BaseException as error:
            write_json(Path(path).with_suffix(".error.json"), recorder.fail(error))
            raise
        write_json(path, record)
        return record

    def review(self, messages, seed):
        self.episode_seed, self.request_index = seed, 0
        self.sampling = dict(self.config.teacher.review)
        self.collector.chat_template_kwargs = dict(self.config.teacher.review_chat_template_kwargs)
        try:
            batch = self.collector.preprocess_batch(self.initial_batch(), {"messages": [messages]})
            output = self.generate_sequences(batch)
            if self.direct:
                return self.last_generation["raw_completion"], self.last_generation
            return self.tokenizer.decode(output.batch["responses"][0], skip_special_tokens=True), self.last_generation
        finally:
            self.sampling = self.action_sampling
            self.collector.chat_template_kwargs = self.action_chat_template_kwargs

    def reflect(self, prompt, seed):
        self.episode_seed, self.request_index = seed, 0
        from omegaconf import OmegaConf
        self.sampling = OmegaConf.to_container(self.config.teacher.reflection, resolve=True)
        self.collector.chat_template_kwargs = dict(self.config.teacher.get("reflection_chat_template_kwargs", {}))
        try:
            batch = self.collector.preprocess_batch(self.initial_batch(), {"text": [prompt], "anchor": [prompt]})
            output = self.generate_sequences(batch)
            return self.tokenizer.decode(output.batch["responses"][0], skip_special_tokens=True)
        finally:
            self.sampling = self.action_sampling
            self.collector.chat_template_kwargs = self.action_chat_template_kwargs

    def close(self):
        import ray
        import time
        if self.direct:
            try:
                ray.get(self.worker.close.remote())
            finally:
                ray.kill(self.worker, no_restart=True)
            deadline = monotonic() + 30
            while ray.available_resources().get("GPU", 0) < 1:
                if monotonic() >= deadline:
                    raise RuntimeError("Previous inference actor has not released its GPU")
                time.sleep(0.1)
            return
        from ray.util import remove_placement_group, placement_group_table
        from ray.util.state import get_actor
        workers = list(self.group._workers)
        try:
            workers.append(ray.get_actor(f"{self.name_prefix}_register_center"))
        except ValueError:
            pass
        for worker in workers:
            ray.kill(worker, no_restart=True)
        groups = self.pool.pgs or []
        for group in groups:
            remove_placement_group(group)
        # Ray teardown is asynchronous. Wait before allocating the next model.
        ray_address = ray.get_runtime_context().gcs_address
        deadline = time.monotonic() + 30
        while True:
            actors_done = all(
                (get_actor(worker._actor_id.hex(), address=ray_address) or {}).get("state", "DEAD") == "DEAD"
                for worker in workers
            )
            groups_done = all(placement_group_table(group).get("state") == "REMOVED" for group in groups)
            resources_free = ray.available_resources().get("GPU", 0) >= self.pool.world_size
            if actors_done and groups_done and resources_free:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Previous rollout workers have not released their GPU resources")
            time.sleep(0.1)


def make_environment(config):
    from omegaconf import OmegaConf
    from agent_system.environments.env_package.alfworld.envs import AlfworldEnvs
    from agent_system.environments.env_package.alfworld.projection import alfworld_projection
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager
    from .tasks import environment_config
    _, settings = environment_config(config)
    envs = AlfworldEnvs(config.env.config_path, seed=config.env.seed,
                       config_override=settings)
    manager_config = OmegaConf.create({"env": {"alfworld": {"meta_think": False, "action_only": False}}})
    return AlfWorldEnvironmentManager(envs, alfworld_projection, config.env.env_name, manager_config)


class EpisodeSession:
    """A live environment and its action stream survive changes of model worker."""
    def __init__(self, config, runtime, task, path, metadata):
        from .tasks import environment_config
        self.path = path
        self.seed = metadata["generation_settings"]["seed"]
        self.recorder = EpisodeRecorder({**task, **metadata, "teacher_reviews": []})
        self.feedback = None
        self.closed = False
        self.envs = make_environment(config)
        try:
            root, _ = environment_config(config)
            reset = dict(task_specs=[dict(game_file=str(root / task["game_file"]), env_seed=config.env.seed)],
                         episode_contexts=[dict(demonstration="", teacher_guidance="")])
            self.steps = runtime.collector.episode_steps(runtime.initial_batch(), self.envs, reset,
                                                         self.recorder, record_only=True)
            self.obs = next(self.steps)
        except BaseException as exc:
            self.save_error(exc)
            self.close()
            raise

    def save_error(self, exc):
        path = self.path.with_suffix(".error.json")
        attempt = 1
        while path.exists():
            path = self.path.with_suffix(f".retry_{attempt}.error.json")
            attempt += 1
        write_json(path, self.recorder.fail(exc))

    def close(self):
        if not self.closed:
            if hasattr(self, "steps"):
                self.steps.close()
            self.envs.envs.close()
            self.closed = True

    def advance(self, runtime, until):
        from agent_system.environments.prompts.teacher_review import NOTE
        record = self.recorder.record
        while record["num_steps"] < until and not self.closed:
            index = record["num_steps"]
            runtime.episode_seed, runtime.request_index = self.seed, index
            review = self.feedback
            observation = self.obs
            if review is not None:
                guided = observation["text"][0] + "\n\n" + NOTE.format(teacher_feedback=review["feedback"])
                observation = dict(observation, text=[guided])

            def request_metadata():
                if review is not None:
                    review["applied_step"] = index + 1
                return dict(runtime.last_generation, generated_with_feedback=review is not None,
                            review_id=review["review_id"] if review else None)

            self.recorder.request_metadata = request_metadata
            try:
                self.obs = self.steps.send((runtime, observation))
            except StopIteration:
                write_json(self.path, self.recorder.finish())
                self.close()
            except BaseException as exc:
                self.save_error(exc)
                raise
            finally:
                self.recorder.request_metadata = None
                self.feedback = None

    def review(self, runtime, seed):
        from agent_system.environments.prompts.teacher_review import review_messages
        record = self.recorder.record
        after = record["num_steps"]
        messages = review_messages(self.obs["text"][0], record["steps"])
        event = dict(review_id=f"r{after}", after_step=after, next_step=after + 1, applied_step=None,
                     user_prompt=messages[1]["content"], seed=seed, decision="error", feedback=None, raw_response=None)
        record["teacher_reviews"].append(event)
        try:
            raw, cost = runtime.review(messages, seed)
        except BaseException as exc:
            self.save_error(exc)
            raise
        event.update(cost, raw_response=raw)
        value = raw.strip()
        if runtime.direct and value.startswith("<think>"):
            _, end, body = value.partition("</think>")
            value = body.strip() if end else ""
        if value == "NO_FEEDBACK":
            event["decision"] = "no_feedback"
        elif value:
            event.update(decision="feedback", feedback=value)
            self.feedback = event
