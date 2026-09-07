"""Command-line entry points for controlled attention ablations."""

import argparse
from dataclasses import asdict
from itertools import islice
import json
import os
from pathlib import Path
import sys

import torch
import transformers

from .attention import AttentionConfig
from .benchmarks import evaluate_perplexity, generate, iter_corpus_tokens, replay_helm
from .model import DEFAULT_MODEL, load_model


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "generate", "ppl", "compare", "lm-eval", "helm", "speed"):
        command = commands.add_parser(name)
        command.add_argument("--model", default=DEFAULT_MODEL)
        command.add_argument("--revision", default="main")
        command.add_argument("--device", default="auto")
        command.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
        command.add_argument("--cache-dir", default=".cache/huggingface")
        command.add_argument("--local-files-only", action="store_true", help="Model/tokenizer cache only; does not disable dataset downloads")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--threads", type=int, default=1, help="CPU threads; 1 is efficient for these small sequential runs")
        command.add_argument("--output", type=Path, required=name == "helm")
        command.add_argument("--method", choices=["full", "window", "streaming", "h2o", "linear"], default="full")
        command.add_argument("--cache-size", type=int, default=64, help="Persistent KV tokens per head; current token is additional")
        command.add_argument("--sink-tokens", type=int, default=1)
        command.add_argument("--heavy-hitter-size", type=int)
        command.add_argument("--feature-map", choices=["elu", "relu"], default="elu")
        command.add_argument("--sink-mode", choices=["softmax", "kernel"], default="softmax")
        if name in {"ppl", "compare"}:
            command.add_argument("--corpus", choices=["tinystories", "pg19", "wikitext2", "wikitext103", "text"], default="tinystories")
            command.add_argument("--split")
            command.add_argument("--dataset-revision", default="main")
            command.add_argument("--text-path", type=Path)
            command.add_argument("--max-documents", type=int)
            command.add_argument("--max-tokens", type=int, default=512, help="Number of next-token predictions, including warmup")
            command.add_argument("--warmup-tokens", type=int, default=0)
            command.add_argument("--reset-interval", type=int, help="Explicitly segment streams; omitted means no resets")
            command.add_argument("--log-every", type=int, default=128)
            command.add_argument("--prepend-bos", action="store_true")
            command.add_argument("--separator", default="\n\n", help="Literal text between documents; empty string is allowed")
        if name == "compare":
            command.add_argument("--methods", default="full,window,streaming,h2o,linear,linear-sink",
                                 help="Comma-separated variants; linear means zero sinks, linear-sink uses --sink-tokens")
        if name == "generate":
            command.add_argument("--prompt", default="Once upon a time, there was a little")
            command.add_argument("--max-new-tokens", type=int, default=64)
            command.add_argument("--temperature", type=float, default=0.0)
            command.add_argument("--top-p", type=float, default=1.0)
        if name in {"lm-eval", "helm"}:
            command.add_argument("--cache-ratio", type=float, help="Total cache budget as a fraction of each prompt's tokens")
            command.add_argument("--limit", type=int, default=1000 if name == "helm" else None)
        if name == "lm-eval":
            command.add_argument("--tasks", default="h2o")
            command.add_argument("--num-fewshot", type=int)
        if name == "helm":
            command.add_argument("--input", type=Path, required=True)
        if name == "speed":
            command.add_argument("--context-lengths", default="64,128,256,512")
            command.add_argument("--decode-tokens", type=int, default=32)
            command.add_argument("--repeats", type=int, default=3)
    return parser


def attention_config(args, variant=None):
    method = variant or args.method
    return AttentionConfig(method="linear" if method == "linear-sink" else method,
                           cache_size=args.cache_size, sink_tokens=0 if variant == "linear" else args.sink_tokens,
                           heavy_hitter_size=args.heavy_hitter_size,
                           feature_map=args.feature_map, sink_mode=args.sink_mode)


def metadata(args, model):
    return {
        "arguments": vars(args),
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "architecture": model.config.model_type,
        "device": str(model.device), "dtype": str(model.dtype),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
    }


def save_result(result, output):
    text = json.dumps(result, indent=2, allow_nan=False, default=str)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return text


def main(argv=None):
    args = build_parser().parse_args(argv)
    # Keep all downloads inside the specified cache, including datasets' auxiliary files.
    os.environ.setdefault("HF_HOME", str(Path(args.cache_dir).resolve()))
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(args.cache_dir).resolve() / "datasets"))
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    config = attention_config(args)
    model, tokenizer = load_model(args.model, config, device=args.device, dtype=args.dtype,
                                  revision=args.revision, cache_dir=args.cache_dir,
                                  local_files_only=args.local_files_only)
    result = {"metadata": metadata(args, model), "attention": asdict(config)}
    if args.command == "download":
        result["status"] = "pretrained model and tokenizer loaded successfully"
    elif args.command == "generate":
        result["generation"] = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens,
                                        temperature=args.temperature, top_p=args.top_p)
    elif args.command in {"ppl", "compare"}:
        token_stream = iter_corpus_tokens(
            tokenizer, corpus=args.corpus, split=args.split, text_path=args.text_path,
            max_documents=args.max_documents, revision=args.dataset_revision, cache_dir=args.cache_dir,
            prepend_bos=args.prepend_bos, separator=args.separator,
        )
        variants = args.methods.split(",") if args.command == "compare" else [args.method]
        # One bounded token buffer makes every comparison see exactly the same data.
        # Single-method ppl remains lazy and suitable for multi-million-token streams.
        tokens = list(islice(token_stream, args.max_tokens + 1)) if args.command == "compare" else token_stream
        result["results"] = []
        for variant in variants:
            config = attention_config(args, variant.strip() if args.command == "compare" else None)
            model.set_attention(config)
            print(f"Evaluating {variant}: {asdict(config)}", file=sys.stderr)
            metrics = evaluate_perplexity(
                model, tokens, max_tokens=args.max_tokens, warmup_tokens=args.warmup_tokens,
                reset_interval=args.reset_interval, log_every=args.log_every,
                progress=lambda entry: print(f"  {entry['predicted_tokens']} tokens; mean NLL={entry['mean_nll']}", file=sys.stderr),
            )
            result["results"].append({"variant": variant, "attention": asdict(config), **metrics})
            if args.output:
                # Save completed variants even if a subsequent run is interrupted.
                save_result(result, args.output)
    elif args.command == "lm-eval":
        from .lm_eval_adapter import run_lm_eval
        result["evaluation"] = run_lm_eval(model, tokenizer, tasks=args.tasks, num_fewshot=args.num_fewshot,
                                           limit=args.limit, cache_ratio=args.cache_ratio, seed=args.seed)
    elif args.command == "helm":
        result["replay"] = replay_helm(model, tokenizer, args.input, args.output,
                                      limit=args.limit, cache_ratio=args.cache_ratio)
        print(save_result(result, args.output.with_suffix(".metadata.json")))
        return
    elif args.command == "speed":
        from .performance import benchmark_speed
        result["performance"] = benchmark_speed(model, [int(n) for n in args.context_lengths.split(",")],
                                                  args.decode_tokens, args.repeats, args.seed)
    print(save_result(result, args.output))
