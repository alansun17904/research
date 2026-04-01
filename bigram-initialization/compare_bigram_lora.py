from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import GPT2LMHeadModel

from experiment import (
    C4_FILTER_SMALL_ARROW_PATH,
    build_model_config,
    build_smart_bigram_log_probs,
    encode_corpus,
    estimate_steps_for_one_epoch,
    initialize_model_from_svd,
    load_corpus_text,
    load_tokenizer,
    select_active_vocab,
    set_seed,
    svd_factorize_log_bigrams,
    train_model,
    train_val_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline training against bigram initialization with frozen token tables and LoRA."
    )
    parser.add_argument("--use-c4-filter-small", action="store_true", default=True)
    parser.add_argument("--dataset-arrow-path", type=str, default=None)
    parser.add_argument("--corpus-path", type=str, default=None)
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--config-name", choices=["tiny", "gpt2"], default="tiny")
    parser.add_argument("--tiny-n-layer", type=int, default=4)
    parser.add_argument("--tiny-n-embd", type=int, default=128)
    parser.add_argument("--top-k-vocab", type=int, default=18000)
    parser.add_argument("--max-documents", type=int, default=1000)
    parser.add_argument("--max-corpus-tokens", type=int, default=100000)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=3)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ewma-decay", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    return parser.parse_args()


def resolve_dataset_path(args: argparse.Namespace) -> str | None:
    if args.dataset_arrow_path is not None:
        return args.dataset_arrow_path
    if args.use_c4_filter_small:
        return C4_FILTER_SMALL_ARROW_PATH
    return None


def compute_ewma(values: list[float], decay: float) -> list[float]:
    if not values:
        return []
    running = values[0]
    smoothed = []
    for value in values:
        running = decay * running + (1.0 - decay) * value
        smoothed.append(running)
    return smoothed


def plot_histories(results: dict[str, object], output_path: Path, ewma_decay: float) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    styles = {
        "baseline": {"linestyle": "--", "linewidth": 2.0},
        "baseline_lora": {"linestyle": ":", "linewidth": 2.0},
        "bigram_lora": {"linestyle": "-", "linewidth": 2.3},
    }
    for scheme, metrics in results["results"].items():
        steps = list(range(1, len(metrics["step_losses"]) + 1))
        ewma_losses = compute_ewma(metrics["step_losses"], decay=ewma_decay)
        ax.plot(steps, ewma_losses, label=scheme, **styles.get(scheme, {}))
    ax.set_title(f"EWMA Training Loss vs Step (decay={ewma_decay:.2f})")
    ax.set_xlabel("Training step")
    ax.set_ylabel("EWMA training loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_checkpoint(path: Path, model: GPT2LMHeadModel, extra_state: dict[str, object]) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
            **extra_state,
        },
        path,
    )


def build_bigram_initialized_model(
    model_config,
    active_token_ids: list[int],
    log_bigram_matrix: torch.Tensor,
) -> GPT2LMHeadModel:
    embedding_vectors, unembedding_vectors = svd_factorize_log_bigrams(
        log_bigram_matrix,
        d_model=model_config.n_embd,
    )
    model = GPT2LMHeadModel(model_config)
    initialize_model_from_svd(
        model,
        active_token_ids=active_token_ids,
        embedding_vectors=embedding_vectors,
        unembedding_vectors=unembedding_vectors,
    )
    return model


def main() -> None:
    args = parse_args()
    dataset_arrow_path = resolve_dataset_path(args)
    output_dir = Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    tokenizer = load_tokenizer(args.tokenizer_name)
    text = load_corpus_text(
        corpus_path=args.corpus_path,
        dataset_arrow_path=dataset_arrow_path,
        max_documents=args.max_documents,
    )
    token_ids = encode_corpus(tokenizer, text, args.max_corpus_tokens)
    train_ids, val_ids = train_val_split(token_ids)
    default_epoch_steps = estimate_steps_for_one_epoch(
        train_token_count=train_ids.numel(),
        batch_size=args.batch_size,
        block_size=args.block_size,
    )
    model_config = build_model_config(
        config_name=args.config_name,
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        tiny_n_layer=args.tiny_n_layer,
        tiny_n_embd=args.tiny_n_embd,
    )

    active_token_ids = select_active_vocab(train_ids.tolist(), args.top_k_vocab)
    log_bigram_matrix, _, init_metrics = build_smart_bigram_log_probs(
        train_token_ids=train_ids.tolist(),
        val_token_ids=val_ids.tolist(),
        active_token_ids=active_token_ids,
    )

    baseline_model = GPT2LMHeadModel(model_config)
    baseline_lora_model = copy.deepcopy(baseline_model)
    bigram_lora_model = build_bigram_initialized_model(
        model_config=model_config,
        active_token_ids=active_token_ids,
        log_bigram_matrix=log_bigram_matrix,
    )

    set_seed(args.seed)
    baseline_metrics = train_model(
        model=baseline_model,
        train_data=train_ids,
        val_data=val_ids,
        batch_size=args.batch_size,
        block_size=args.block_size,
        train_steps=args.train_steps,
        eval_batches=args.eval_batches,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
    )
    set_seed(args.seed)
    baseline_lora_metrics = train_model(
        model=baseline_lora_model,
        train_data=train_ids,
        val_data=val_ids,
        batch_size=args.batch_size,
        block_size=args.block_size,
        train_steps=args.train_steps,
        eval_batches=args.eval_batches,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        freeze_token_table_params=True,
        use_lora=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    set_seed(args.seed)
    bigram_lora_metrics = train_model(
        model=bigram_lora_model,
        train_data=train_ids,
        val_data=val_ids,
        batch_size=args.batch_size,
        block_size=args.block_size,
        train_steps=args.train_steps,
        eval_batches=args.eval_batches,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        freeze_token_table_params=True,
        use_lora=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    baseline_ckpt = output_dir / "baseline_model.pt"
    baseline_lora_ckpt = output_dir / "baseline_lora_model.pt"
    bigram_lora_ckpt = output_dir / "bigram_lora_model.pt"
    save_checkpoint(
        baseline_ckpt,
        baseline_model.cpu(),
        {
            "model_config": model_config.to_dict(),
            "training_metrics": baseline_metrics,
        },
    )
    save_checkpoint(
        baseline_lora_ckpt,
        baseline_lora_model.cpu(),
        {
            "model_config": model_config.to_dict(),
            "training_metrics": baseline_lora_metrics,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        },
    )
    save_checkpoint(
        bigram_lora_ckpt,
        bigram_lora_model.cpu(),
        {
            "model_config": model_config.to_dict(),
            "training_metrics": bigram_lora_metrics,
            "init_metrics": init_metrics,
            "active_token_ids": active_token_ids,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        },
    )

    results = {
        "config": {
            "dataset_arrow_path": dataset_arrow_path,
            "corpus_path": args.corpus_path,
            "tokenizer_name": args.tokenizer_name,
            "config_name": args.config_name,
            "tiny_n_layer": args.tiny_n_layer,
            "tiny_n_embd": args.tiny_n_embd,
            "top_k_vocab": args.top_k_vocab,
            "max_documents": args.max_documents,
            "max_corpus_tokens": args.max_corpus_tokens,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "train_steps": args.train_steps,
            "default_epoch_steps": default_epoch_steps,
            "eval_batches": args.eval_batches,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": args.device,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "train_tokens": train_ids.numel(),
            "val_tokens": val_ids.numel(),
            "output_dir": str(output_dir),
        },
        "results": {
            "baseline": baseline_metrics,
            "baseline_lora": baseline_lora_metrics,
            "bigram_lora": {
                **bigram_lora_metrics,
                "init_metrics": {**init_metrics, "active_vocab_size": len(active_token_ids)},
            },
        },
        "artifacts": {
            "baseline_checkpoint": str(baseline_ckpt),
            "baseline_lora_checkpoint": str(baseline_lora_ckpt),
            "bigram_lora_checkpoint": str(bigram_lora_ckpt),
        },
    }

    results_json = output_dir / "comparison_results.json"
    plot_path = output_dir / "comparison_loss.png"
    results_json.write_text(json.dumps(results, indent=2))
    plot_histories(results, plot_path, ewma_decay=args.ewma_decay)
    print(
        json.dumps(
            {
                "results_json": str(results_json),
                "plot_path": str(plot_path),
                **results["artifacts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
