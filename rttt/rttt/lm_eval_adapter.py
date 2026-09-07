"""LM Evaluation Harness adapter; prompts and metrics come from the harness."""

import torch
from lm_eval import evaluator
from lm_eval.api.model import LM

from .benchmarks import configure_request_budget, generate


TASK_PRESETS = {
    "h2o": ("copa,mathqa,openbookqa,piqa,rte,winogrande", 5),
    "streamingllm": ("arc_challenge,arc_easy,hellaswag,lambada_openai,openbookqa,piqa,winogrande", 0),
}

# Data-only repositories used by the current upstream harness. Keep the pinned
# task prompts/metrics while avoiding its obsolete script-based dataset paths.
TASK_DATASETS = {
    "piqa": "baber/piqa", "mathqa": "regisss/math_qa",
    "copa": "aps/super_glue", "rte": "nyu-mll/glue",
    "openbookqa": "allenai/openbookqa", "winogrande": "allenai/winogrande",
    "arc_easy": "allenai/ai2_arc", "arc_challenge": "allenai/ai2_arc",
    "hellaswag": "Rowan/hellaswag", "lambada_openai": "EleutherAI/lambada_openai",
}


class SequentialLM(LM):
    """Batch-one scorer using the harness's prompts and token accounting."""

    def __init__(self, model, tokenizer, *, cache_ratio=None, max_length=None):
        super().__init__()
        self.runner = model
        self.tokenizer = tokenizer
        self.base_config = model.attention_config
        self.cache_ratio = cache_ratio
        self.max_length = max_length or model.config.max_position_embeddings
        self.truncated_requests = 0
        self.request_count = 0

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.bos_token_id

    def _start_request(self, ids, max_context):
        self.truncated_requests += len(ids) > max_context
        ids = ids[-max_context:] or [self.eot_token_id]
        configure_request_budget(self.runner, self.base_config, len(ids), self.cache_ratio)
        self.request_count += 1
        return ids

    def _encode_pair(self, context, continuation):
        # Match lm-eval's causal tokenizer handling of context trailing spaces.
        trailing = len(context) - len(context.rstrip())
        if trailing:
            continuation = context[-trailing:] + continuation
            context = context[:-trailing]
        if not context:
            return [self.eot_token_id], self.tokenizer.encode(continuation, add_special_tokens=False)
        whole = self.tokenizer.encode(context + continuation, add_special_tokens=False)
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        return context_ids, whole[len(context_ids):]

    @torch.inference_mode()
    def _score(self, context_ids, continuation_ids):
        if len(continuation_ids) > self.max_length:
            raise ValueError("Continuation exceeds model evaluation context length")
        if not continuation_ids:
            return 0.0, True
        context_ids = self._start_request(context_ids, self.max_length + 1 - len(continuation_ids))
        logits = self.runner.prefill(torch.tensor([context_ids], device=self.runner.device))
        score = 0.0
        greedy = True
        for index, token in enumerate(continuation_ids):
            scores = logits[0].float()
            score += scores.log_softmax(-1)[token].item()
            greedy = greedy and scores.argmax().item() == token
            if index + 1 < len(continuation_ids):
                logits = self.runner.step(torch.tensor([token], device=self.runner.device))
        return score, greedy

    def loglikelihood(self, requests, disable_tqdm=False):
        return [self._score(*self._encode_pair(*request.args)) for request in requests]

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        results = []
        for request in requests:
            ids = self.tokenizer.encode(request.args[0], add_special_tokens=False)
            score = 0.0
            # The first block is BOS-conditioned; later blocks overlap one
            # input token. Every text token contributes one likelihood.
            for start in range(0, len(ids), self.max_length):
                context = [ids[start - 1]] if start else [self.eot_token_id]
                score += self._score(context, ids[start:start + self.max_length])[0]
            results.append(score)
        return results

    def generate_until(self, requests, disable_tqdm=False):
        results = []
        for request in requests:
            prompt, kwargs = request.args
            max_new = kwargs.get("max_gen_toks", 256)
            if max_new < 0 or max_new > self.max_length:
                raise ValueError("max_gen_toks must be between zero and max_length")
            if kwargs.get("do_sample", False) or kwargs.get("temperature", 0) not in (0, 0.0, None):
                raise ValueError("Harness generation uses greedy decoding only")
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            available = max(1, self.max_length - max_new + 1)
            prompt_ids = self._start_request(prompt_ids, available)
            result = generate(self.runner, self.tokenizer, prompt_ids, max_new_tokens=max_new,
                              stop=kwargs.get("until", []))
            results.append(result["text"])
        return results


def run_lm_eval(model, tokenizer, *, tasks="h2o", num_fewshot=None, limit=None, cache_ratio=None, seed=42):
    task_names, shots = TASK_PRESETS.get(tasks, (tasks, 0))
    adapter = SequentialLM(model, tokenizer, cache_ratio=cache_ratio)
    task_configs = [
        {"task": name, "dataset_path": TASK_DATASETS[name], "dataset_kwargs": {"trust_remote_code": False}}
        if name in TASK_DATASETS else name for name in task_names.split(",")
    ]
    result = evaluator.simple_evaluate(
        model=adapter, tasks=task_configs,
        num_fewshot=shots if num_fewshot is None else num_fewshot,
        limit=limit, batch_size=1, random_seed=seed, numpy_random_seed=seed,
        torch_random_seed=seed, fewshot_random_seed=seed, log_samples=False,
        bootstrap_iters=0,
    )
    result["rttt_diagnostics"] = {"requests": adapter.request_count, "truncated_requests": adapter.truncated_requests,
                                  "cache_ratio": cache_ratio, "evaluation_max_length": adapter.max_length,
                                  "dataset_paths": {name: TASK_DATASETS.get(name) for name in task_names.split(",")}}
    return result
