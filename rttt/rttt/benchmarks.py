"""Streaming corpora, next-token perplexity, generation, and HELM replay."""

from dataclasses import replace
from itertools import islice, pairwise
import json
import hashlib
import math
from pathlib import Path
import time
from urllib.request import urlopen

import torch
import torch.nn.functional as F


CORPORA = {
    "tinystories": ("roneneldan/TinyStories", None, "validation"),
    "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1", "test"),
    "wikitext103": ("Salesforce/wikitext", "wikitext-103-raw-v1", "test"),
}


def iter_corpus_texts(corpus="tinystories", *, split=None, text_path=None,
                      max_documents=None, revision="main", cache_dir=None):
    if corpus == "text":
        yield Path(text_path).read_text(encoding="utf-8")
        return
    if corpus == "pg19":
        # Read the official manifest without executing the historical dataset
        # script. Its sorted path order is the HF PG19 script's example order.
        from huggingface_hub import hf_hub_download

        split = split or "test"
        manifest = hf_hub_download("deepmind/pg19", f"data/{split}_files.txt", repo_type="dataset",
                                   revision=revision, cache_dir=cache_dir)
        paths = sorted(Path(manifest).read_text().splitlines())
        for path in islice(paths, max_documents):
            with urlopen("https://storage.googleapis.com/deepmind-gutenberg/" + path, timeout=60) as response:
                yield response.read().decode("utf-8")
        return
    from datasets import load_dataset

    name, config, default_split = CORPORA[corpus]
    dataset = load_dataset(name, config, split=split or default_split, streaming=True,
                           revision=revision, cache_dir=cache_dir)
    for row in islice(dataset, max_documents):
        yield row["text"]


def iter_corpus_tokens(tokenizer, corpus="tinystories", *, separator="\n\n", prepend_bos=False, **kwargs):
    """Keep at most one document in memory; BOS insertion is explicit."""
    if prepend_bos:
        bos = tokenizer.bos_token_id
        if bos is None:
            raise ValueError("The tokenizer has no BOS token")
        yield bos
    for text in iter_corpus_texts(corpus, **kwargs):
        if text:
            yield from tokenizer.encode(text + separator, add_special_tokens=False)


@torch.inference_mode()
def evaluate_perplexity(model, token_ids, *, max_tokens=None, warmup_tokens=0,
                        reset_interval=None, log_every=128, progress=None):
    """Score adjacent pairs once. Warmup updates state but contributes no loss."""
    model.reset()
    device = model.device
    tokens = islice(token_ids, None if max_tokens is None else max_tokens + 1)
    token_hash = hashlib.sha256()
    total_nll = block_nll = 0.0
    count = block_count = predicted = peak_state_bytes = 0
    segments = 1
    trace = []
    for predicted, (current, target) in enumerate(pairwise(tokens), start=1):
        if predicted == 1:
            token_hash.update(int(current).to_bytes(8, "little"))
        token_hash.update(int(target).to_bytes(8, "little"))
        if reset_interval and predicted > 1 and (predicted - 1) % reset_interval == 0:
            model.reset()
            segments += 1

        logits = model.step(torch.tensor([current], device=device))
        loss = F.cross_entropy(logits.float(), torch.tensor([target], device=device)).item()
        if not math.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at prediction {predicted}")
        if predicted > warmup_tokens:
            total_nll += loss
            block_nll += loss
            count += 1
            block_count += 1
        peak_state_bytes = max(peak_state_bytes, model.state_bytes)
        if log_every and predicted % log_every == 0:
            entry = {"predicted_tokens": predicted, "scored_tokens": count,
                     "mean_nll": total_nll / count if count else None,
                     "block_mean_nll": block_nll / block_count if block_count else None,
                     "state_bytes": model.state_bytes}
            trace.append(entry)
            if progress:
                progress(entry)
            block_nll = block_count = 0

    if count == 0:
        raise ValueError("No scored token pairs; provide more text or reduce warmup_tokens")
    mean_nll = total_nll / count
    ppl = math.exp(mean_nll) if mean_nll < 709 else None
    return {
        "tokens": count, "predicted_tokens": predicted, "warmup_tokens": warmup_tokens,
        "nll": total_nll, "mean_nll": mean_nll, "perplexity": ppl,
        "state_bytes": model.state_bytes, "peak_state_bytes": peak_state_bytes,
        "cached_tokens_per_head": model.cached_tokens,
        "reset_interval": reset_interval, "segments": segments, "trace": trace,
        "input_token_sha256": token_hash.hexdigest(),
    }


def configure_request_budget(model, base_config, prompt_length, cache_ratio=None):
    """A ratio is a total retained KV budget, split equally for H2O by default."""
    config = base_config
    if cache_ratio is not None:
        budget = max(1, int(prompt_length * cache_ratio))
        if config.method == "streaming" and budget <= config.sink_tokens:
            raise ValueError("Ratio budget leaves no recent slot; lower sink_tokens or increase cache_ratio")
        heavy = config.heavy_hitter_size
        if heavy is not None:
            heavy = min(heavy, budget - 1)
        config = replace(config, cache_size=budget, heavy_hitter_size=heavy)
    model.set_attention(config)


@torch.inference_mode()
def generate(model, tokenizer, prompt, *, max_new_tokens=64, stop=None,
             temperature=0.0, top_p=1.0):
    """Decode a single prompt, with optional nucleus sampling for HELM replay."""
    model.reset()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else list(prompt)
    if not prompt_ids:
        token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        prompt_ids = [token]
    limit = model.max_sequence_length
    if limit is not None and len(prompt_ids) + max(max_new_tokens - 1, 0) > limit:
        raise ValueError("Prompt plus generation exceeds learned positions; shorten it or use Pythia")
    stop = [stop] if isinstance(stop, str) else list(stop or [])
    generated = []
    logprobs = []
    finish_reason = "length"
    text = ""
    if max_new_tokens:
        logits = model.prefill(torch.tensor([prompt_ids], device=model.device))
    for index in range(max_new_tokens):
        scores = logits[0].float()
        if temperature == 0:
            token = scores.argmax().item()
        else:
            probabilities = (scores / temperature).softmax(-1)
            sorted_probs, indices = probabilities.sort(descending=True)
            remove = sorted_probs.cumsum(-1) - sorted_probs >= top_p
            sorted_probs[remove] = 0
            token = indices[torch.multinomial(sorted_probs, 1)].item()
        generated.append(token)
        logprobs.append(scores.log_softmax(-1)[token].item())
        text = tokenizer.decode(generated, skip_special_tokens=True)
        stops = [text.find(s) for s in stop if s in text]
        if stops or token == tokenizer.eos_token_id:
            if stops:
                text = text[:min(stops)]
            finish_reason = "stop"
            break
        if index + 1 < max_new_tokens:
            logits = model.step(torch.tensor([token], device=model.device))
    return {"text": text, "token_ids": generated, "token_logprobs": logprobs, "finish_reason": finish_reason}


def replay_helm(model, tokenizer, input_path, output_path, *, limit=1000, cache_ratio=None):
    """Replay H2O's request JSONL; use HELM to score the resulting completions."""
    from .performance import synchronize

    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("HELM output must differ from input")
    config = model.attention_config
    count = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, encoding="utf-8") as source, open(output_path, "w", encoding="utf-8") as dest:
        for line in islice((line for line in source if line.strip()), limit):
            request = json.loads(line)["request"]
            if request.get("echo", False) or request.get("frequency_penalty", 0) or request.get("presence_penalty", 0):
                raise ValueError("HELM echo and repetition penalties are unsupported")
            prompt_ids = tokenizer.encode(request["prompt"], add_special_tokens=False)
            choices = []
            n = request.get("n", 1)
            synchronize(model.device)
            started = time.perf_counter()
            for _ in range(n):
                configure_request_budget(model, config, max(1, len(prompt_ids)), cache_ratio)
                result = generate(model, tokenizer, prompt_ids, max_new_tokens=request["max_tokens"],
                                  stop=request.get("stop"), temperature=request.get("temperature", 0),
                                  top_p=request.get("top_p", 1))
                token_strings = tokenizer.convert_ids_to_tokens(result["token_ids"])
                choices.append({"text": result["text"], "finish_reason": result["finish_reason"],
                                "logprobs": {"tokens": token_strings, "token_logprobs": result["token_logprobs"],
                                             "top_logprobs": [{} for _ in token_strings], "text_offset": []}})
            synchronize(model.device)
            row = {"request": request, "result": {"choices": choices,
                   "request_time": {"batch_time": time.perf_counter() - started, "batch_size": n}}}
            dest.write(json.dumps(row, allow_nan=False) + "\n")
            dest.flush()
            count += 1
    return {"requests": count, "output": str(output_path), "metrics": "Run the original HELM scorer on this replay"}
