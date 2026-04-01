from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import GPT2LMHeadModel

from experiment import (
    BigramInitGPT2LMHeadModel,
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
        description="Compare baseline, frozen-random baseline, and smart-bigram with full linear adapters."
    )
    parser.add_argument("--use-c4-filter-small", action="store_true", default=True)
    parser.add_argument("--dataset-arrow-path", type=str, default=None)
    parser.add_argument("--corpus-path", type=str, default=None)
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--config-name", choices=["tiny", "gpt2"], default="tiny")
    parser.add_argument("--tiny-n-layer", type=int, default=4)
    parser.add_argument("--tiny-n-embd", type=int, default=128)
    parser.add_argument("--top-k-vocab", type=int, default=1024)
    parser.add_argument("--max-documents", type=int, default=500)
    parser.add_argument("--max-corpus-tokens", type=int, default=200000)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=3)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ewma-decay", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="results/adapter_strategy_comparison")
    parser.add_argument(
        "--freeze-topk-token-tables",
        action="store_true",
        dest="freeze_topk_token_tables",
        help="Freeze the smart-bigram model's token tables during training.",
    )
    parser.add_argument(
        "--train-topk-token-tables",
        action="store_false",
        dest="freeze_topk_token_tables",
        help="Allow the smart-bigram model's token tables to keep training.",
    )
    parser.set_defaults(freeze_topk_token_tables=True)
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
        "baseline_frozen_random": {"linestyle": ":", "linewidth": 2.0},
        "topk_residual_mlp": {"linestyle": "-", "linewidth": 2.2},
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


def build_initialized_topk_model(
    model_config,
    active_token_ids: list[int],
    log_bigram_matrix: torch.Tensor,
    adapter_strategy: str,
):
    embedding_vectors, unembedding_vectors = svd_factorize_log_bigrams(
        log_bigram_matrix,
        d_model=model_config.n_embd,
    )
    model = BigramInitGPT2LMHeadModel(model_config, adapter_strategy=adapter_strategy)
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
    output_dir = Path(args.output_dir)
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
    baseline_frozen_random_model = copy.deepcopy(baseline_model)
    topk_residual_mlp_model = build_initialized_topk_model(
        model_config=model_config,
        active_token_ids=active_token_ids,
        log_bigram_matrix=log_bigram_matrix,
        adapter_strategy="residual_mlp",
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
        freeze_token_table_params=False,
    )
    set_seed(args.seed)
    baseline_frozen_random_metrics = train_model(
        model=baseline_frozen_random_model,
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
    )
    set_seed(args.seed)
    topk_residual_mlp_metrics = train_model(
        model=topk_residual_mlp_model,
        train_data=train_ids,
        val_data=val_ids,
        batch_size=args.batch_size,
        block_size=args.block_size,
        train_steps=args.train_steps,
        eval_batches=args.eval_batches,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        freeze_token_table_params=args.freeze_topk_token_tables,
    )

    baseline_ckpt = output_dir / "baseline_model.pt"
    baseline_frozen_random_ckpt = output_dir / "baseline_frozen_random_model.pt"
    residual_mlp_ckpt = output_dir / "smart_bigram_topk_residual_mlp_model.pt"
    save_checkpoint(
        baseline_ckpt,
        baseline_model.cpu(),
        {
            "model_config": model_config.to_dict(),
            "training_metrics": baseline_metrics,
        },
    )
    save_checkpoint(
        baseline_frozen_random_ckpt,
        baseline_frozen_random_model.cpu(),
        {
            "model_config": model_config.to_dict(),
            "training_metrics": baseline_frozen_random_metrics,
            "freeze_token_tables": True,
        },
    )
    save_checkpoint(
        residual_mlp_ckpt,
        topk_residual_mlp_model.cpu(),
        {
            "model_config": model_config.to_dict(),
            "training_metrics": topk_residual_mlp_metrics,
            "init_metrics": init_metrics,
            "active_token_ids": active_token_ids,
            "freeze_token_tables": args.freeze_topk_token_tables,
            "adapter_strategy": "residual_mlp",
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
            "train_tokens": train_ids.numel(),
            "val_tokens": val_ids.numel(),
            "freeze_topk_token_tables": args.freeze_topk_token_tables,
        },
        "results": {
            "baseline": baseline_metrics,
            "baseline_frozen_random": baseline_frozen_random_metrics,
            "topk_residual_mlp": {
                **topk_residual_mlp_metrics,
                "init_metrics": {**init_metrics, "active_vocab_size": len(active_token_ids)},
            },
        },
        "artifacts": {
            "baseline_checkpoint": str(baseline_ckpt),
            "baseline_frozen_random_checkpoint": str(baseline_frozen_random_ckpt),
            "topk_residual_mlp_checkpoint": str(residual_mlp_ckpt),
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
