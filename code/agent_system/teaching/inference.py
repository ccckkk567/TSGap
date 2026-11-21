"""Offline inference inside a single Ray GPU actor."""


class InferenceWorker:
    def __init__(self, model_path, model_len, gpu_memory_utilization):
        from vllm import LLM
        self.engine = LLM(model=model_path, tensor_parallel_size=1, dtype="bfloat16",
                          language_model_only=True, max_model_len=model_len,
                          gpu_memory_utilization=gpu_memory_utilization,
                          distributed_executor_backend="uni")

    def generate(self, prompt_token_ids, sampling):
        from vllm import SamplingParams
        results = self.engine.generate([{"prompt_token_ids": ids} for ids in prompt_token_ids],
                                       SamplingParams(**sampling), use_tqdm=False)
        return [(list(result.outputs[0].token_ids), result.outputs[0].finish_reason,
                 result.outputs[0].stop_reason) for result in results]

    def close(self):
        self.engine.llm_engine.engine_core.shutdown()
