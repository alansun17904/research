#!/usr/bin/env python3
"""Evaluate Pythia 70M on the first 100,000 WikiText-2 validation tokens.

Install in your environment: python -m pip install torch transformers datasets
Run inside a Slurm compute job, following AGENT.md:
    HF_HOME=/data/cl/u/awsun/.cache/huggingface python evaluate.py --device cuda
Evaluate an editable JSON architecture with its actual modified weights/graph:
    python evaluate.py --graph graphs/example.json --device cuda
Small offline window/counting check (no dependencies or downloads):
    python evaluate.py --self-check

--max-tokens changes the deterministic prefix size; --device cpu also works.
Rows are joined with two newlines, without adding special tokens or padding.
The first token is context only, so 100,000 input tokens give 99,999 targets.
Overlapping windows use the model's context limit and a half-context stride.
NLL is in nats, weighted by scored tokens; perplexity is exp(mean NLL).
Only validation is read; overlap with Pythia's pretraining data is not audited.

References:
https://huggingface.co/EleutherAI/pythia-70m
https://huggingface.co/datasets/Salesforce/wikitext
https://huggingface.co/docs/transformers/en/perplexity
"""

import argparse
import json
import math

MODEL = "EleutherAI/pythia-70m"
MODEL_REVISION = "a39f36b100fe8a5377810d56c3f4789b9c53ac42"
DATASET = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SPLIT = "validation"


def evaluation_windows(token_count, context_length):
    """Yield (start, end, first target offset), scoring each next token once."""
    if token_count < 2 or context_length < 2:
        raise ValueError("Need at least two tokens and a context length of two.")
    previous_end = 1
    for start in range(0, token_count, context_length // 2):
        end = min(start + context_length, token_count)
        yield start, end, previous_end - start
        previous_end = end
        if end == token_count:
            break


def evaluate(model, input_ids):
    import torch

    context_length = model.config.max_position_embeddings
    total_nll = 0.0
    scored_tokens = 0
    model.eval()
    with torch.inference_mode():
        for start, end, offset in evaluation_windows(input_ids.size(1), context_length):
            window = input_ids[:, start:end].to(model.device)
            labels = window.clone()
            labels[:, :offset] = -100
            # offset >= 1 also excludes the label dropped by the causal shift.
            count = end - start - offset
            loss = model(input_ids=window, labels=labels, use_cache=False).loss
            total_nll += loss.item() * count
            scored_tokens += count
    assert scored_tokens == input_ids.size(1) - 1
    mean_nll = total_nll / scored_tokens
    return {
        "input_tokens": input_ids.size(1),
        "scored_tokens": scored_tokens,
        "context_length": context_length,
        "stride": context_length // 2,
        "total_nll_nats": total_nll,
        "mean_nll_nats": mean_nll,
        "perplexity": math.exp(mean_nll),
    }


def self_check():
    for context in (2, 3, 4, 7, 2048):
        for size in (2, 3, 4, 8, 9, 17, 2048, 2049, 4097, 100_000):
            targets = []
            for start, end, offset in evaluation_windows(size, context):
                assert 1 <= offset < end - start <= context
                # Reproduce the causal label shift and ignored-label mask.
                targets.extend(start + i for i in range(1, end - start) if i >= offset)
            assert targets == list(range(1, size))
    print("Self-check passed: every target scored once, including overlap and short tails.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tokens", type=int, default=100_000)
    parser.add_argument("--device", help="PyTorch device; defaults to CUDA if available, otherwise CPU")
    parser.add_argument("--graph", help="JSON architecture; loads the local pinned checkpoint")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.max_tokens < 2:
        parser.error("--max-tokens must be at least 2")

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if args.graph:
        from model import CHECKPOINT, load_model
        model = load_model(args.graph).to(device)
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    rows = load_dataset(
        DATASET, DATASET_CONFIG, split=SPLIT, revision=DATASET_REVISION, streaming=True
    )
    text = "\n\n".join(row["text"] for row in rows)
    input_ids = tokenizer(
        text, add_special_tokens=False, truncation=True,
        max_length=args.max_tokens, return_tensors="pt",
    )["input_ids"]
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION).to(device)
    result = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "dataset": DATASET,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "split": SPLIT,
        "sample": "deterministic prefix; rows joined with two newlines; no special tokens",
        "requested_input_tokens": args.max_tokens,
        "device": str(model.device),
        **evaluate(model, input_ids),
    }
    if args.graph:
        result["graph"] = model.graph
        result["unique_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
