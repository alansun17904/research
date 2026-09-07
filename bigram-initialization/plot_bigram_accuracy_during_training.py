from __future__ import annotations

import argparse
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
    compute_bigram_counts,
    encode_corpus,
    evaluate_loss,
    estimate_steps_for_one_epoch,
    initialize_model_from_svd,
    load_corpus_text,
    load_tokenizer,
    sample_batch,
    select_active_vocab,
    set_seed,
    svd_factorize_log_bigrams,
    train_val_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a model end to end and plot the induced token-table "
            "bigram accuracy against corpus-bigram reference lines."
        )
    )
    parser.add_argument("--use-c4-filter-small", action="store_true")
    parser.add_argument("--dataset-arrow-path", type=str, default=None)
    parser.add_argument("--corpus-path", type=str, default=None)
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--config-name", choices=["tiny", "gpt2"], default="tiny")
    parser.add_argument("--init-mode", choices=["svd", "random"], default="svd")
    parser.add_argument("--tiny-n-layer", type=int, default=4)
    parser.add_argument("--tiny-n-embd", type=int, default=128)
    parser.add_argument("--top-k-vocab", type=int, default=512)
    parser.add_argument("--max-documents", type=int, default=500)
    parser.add_argument("--max-corpus-tokens", type=int, default=100000)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=3)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument(
        "--bigram-eval-chunk-size",
        type=int,
        default=256,
        help="Source-row chunk size when scoring the induced token-table bigram matrix.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def resolve_dataset_path(args: argparse.Namespace) -> str | None:
    if args.dataset_arrow_path is not None:
        return args.dataset_arrow_path
    if args.use_c4_filter_small:
        return C4_FILTER_SMALL_ARROW_PATH
    return None


def compute_weighted_top1_accuracy(counts: torch.Tensor, predicted_targets: torch.Tensor) -> float:
    if counts.ndim != 2:
        raise ValueError("Expected a 2D count matrix.")
    total_pairs = counts.sum().item()
    if total_pairs <= 0:
        return float("nan")
    row_indices = torch.arange(counts.size(0), dtype=torch.long)
    predicted_targets = predicted_targets.to(dtype=torch.long, device="cpu")
    correct_pairs = counts[row_indices, predicted_targets].sum().item()
    return correct_pairs / total_pairs


def compute_reference_bigram_accuracy(counts: torch.Tensor, log_bigram_matrix: torch.Tensor) -> float:
    predicted_targets = log_bigram_matrix.argmax(dim=1).cpu()
    return compute_weighted_top1_accuracy(counts, predicted_targets)


def compute_empirical_oracle_accuracy(counts: torch.Tensor) -> float:
    predicted_targets = counts.argmax(dim=1).cpu()
    return compute_weighted_top1_accuracy(counts, predicted_targets)


def compute_model_bigram_accuracy(
    model: GPT2LMHeadModel,
    active_token_ids: list[int],
    counts: torch.Tensor,
    chunk_size: int,
) -> float:
    device = model.transformer.wte.weight.device
    active_token_tensor = torch.tensor(active_token_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        source_embeddings = model.transformer.wte.weight.index_select(0, active_token_tensor)
        target_unembeddings = model.lm_head.weight.index_select(0, active_token_tensor)
        predicted_targets = []
        for start in range(0, source_embeddings.size(0), chunk_size):
            source_chunk = source_embeddings[start : start + chunk_size]
            logits = source_chunk @ target_unembeddings.transpose(0, 1)
            predicted_targets.append(logits.argmax(dim=1).cpu())
    return compute_weighted_top1_accuracy(counts, torch.cat(predicted_targets, dim=0))


def record_bigram_metrics(
    history: dict[str, list[float] | list[int]],
    step: int,
    model: GPT2LMHeadModel,
    active_token_ids: list[int],
    train_counts: torch.Tensor,
    val_counts: torch.Tensor,
    chunk_size: int,
) -> None:
    model.eval()
    history["steps"].append(step)
    history["train_bigram_accuracy"].append(
        compute_model_bigram_accuracy(
            model=model,
            active_token_ids=active_token_ids,
            counts=train_counts,
            chunk_size=chunk_size,
        )
    )
    history["val_bigram_accuracy"].append(
        compute_model_bigram_accuracy(
            model=model,
            active_token_ids=active_token_ids,
            counts=val_counts,
            chunk_size=chunk_size,
        )
    )


def plot_bigram_histories(results: dict[str, object], output_path: Path) -> None:
    history = results["history"]
    ceilings = results["ceilings"]
    steps = history["steps"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    panels = [
        ("train_bigram_accuracy", "train_reference_bigram_accuracy", "Train"),
        ("val_bigram_accuracy", "val_reference_bigram_accuracy", "Validation"),
    ]
    for axis, (history_key, ceiling_key, title) in zip(axes, panels):
        axis.plot(
            steps,
            history[history_key],
            linewidth=2.2,
            label="induced token-table bigram",
        )
        axis.axhline(
            ceilings[ceiling_key],
            color="tab:red",
            linestyle="--",
            linewidth=2.0,
            label="selected corpus bigram",
        )
        oracle_key = ceiling_key.replace("reference_bigram", "empirical_oracle")
        axis.axhline(
            ceilings[oracle_key],
            color="0.35",
            linestyle=":",
            linewidth=2.0,
            label="empirical bigram top-1 ceiling",
        )
        axis.set_title(f"{title} Bigram Accuracy")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Top-1 next-token accuracy")
        axis.grid(alpha=0.3)
        axis.set_ylim(0.0, 1.0)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_arrow_path = resolve_dataset_path(args)
    if args.output_dir is None:
        output_dir = Path("results") / datetime.now().strftime("bigram_accuracy_%Y%m%d_%H%M%S")
    else:
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
    train_counts, _ = compute_bigram_counts(train_ids.tolist(), active_token_ids)
    val_counts, _ = compute_bigram_counts(val_ids.tolist(), active_token_ids)
    log_bigram_matrix, _, reference_bigram_metrics = build_smart_bigram_log_probs(
        train_token_ids=train_ids.tolist(),
        val_token_ids=val_ids.tolist(),
        active_token_ids=active_token_ids,
    )
    model = GPT2LMHeadModel(model_config)
    if args.init_mode == "svd":
        embedding_vectors, unembedding_vectors = svd_factorize_log_bigrams(
            log_bigram_matrix,
            d_model=model_config.n_embd,
        )
        initialize_model_from_svd(
            model,
            active_token_ids=active_token_ids,
            embedding_vectors=embedding_vectors,
            unembedding_vectors=unembedding_vectors,
        )
    model.to(args.device)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    ceilings = {
        "train_reference_bigram_accuracy": compute_reference_bigram_accuracy(train_counts, log_bigram_matrix),
        "val_reference_bigram_accuracy": compute_reference_bigram_accuracy(val_counts, log_bigram_matrix),
        "train_empirical_oracle_accuracy": compute_empirical_oracle_accuracy(train_counts),
        "val_empirical_oracle_accuracy": compute_empirical_oracle_accuracy(val_counts),
    }
    history: dict[str, list[float] | list[int]] = {
        "steps": [],
        "step_losses": [],
        "train_bigram_accuracy": [],
        "val_bigram_accuracy": [],
    }
    record_bigram_metrics(
        history=history,
        step=0,
        model=model,
        active_token_ids=active_token_ids,
        train_counts=train_counts,
        val_counts=val_counts,
        chunk_size=args.bigram_eval_chunk_size,
    )

    initial_train_loss = evaluate_loss(
        model=model,
        data=train_ids,
        batch_size=args.batch_size,
        block_size=args.block_size,
        eval_batches=args.eval_batches,
        device=args.device,
    )
    initial_val_loss = evaluate_loss(
        model=model,
        data=val_ids,
        batch_size=args.batch_size,
        block_size=args.block_size,
        eval_batches=args.eval_batches,
        device=args.device,
    )

    model.train()
    for step in range(1, args.train_steps + 1):
        x, y = sample_batch(train_ids, args.batch_size, args.block_size, args.device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids=x, labels=y)
        outputs.loss.backward()
        optimizer.step()
        history["step_losses"].append(outputs.loss.item())
        if step % args.record_every == 0 or step == args.train_steps:
            record_bigram_metrics(
                history=history,
                step=step,
                model=model,
                active_token_ids=active_token_ids,
                train_counts=train_counts,
                val_counts=val_counts,
                chunk_size=args.bigram_eval_chunk_size,
            )
            model.train()

    final_train_loss = evaluate_loss(
        model=model,
        data=train_ids,
        batch_size=args.batch_size,
        block_size=args.block_size,
        eval_batches=args.eval_batches,
        device=args.device,
    )
    final_val_loss = evaluate_loss(
        model=model,
        data=val_ids,
        batch_size=args.batch_size,
        block_size=args.block_size,
        eval_batches=args.eval_batches,
        device=args.device,
    )

    plot_path = output_dir / "bigram_accuracy_vs_training_step.png"
    results_json = output_dir / "bigram_accuracy_results.json"
    results = {
        "config": {
            "dataset_arrow_path": dataset_arrow_path,
            "corpus_path": args.corpus_path,
            "tokenizer_name": args.tokenizer_name,
            "config_name": args.config_name,
            "init_mode": args.init_mode,
            "tiny_n_layer": args.tiny_n_layer,
            "tiny_n_embd": args.tiny_n_embd,
            "top_k_vocab": args.top_k_vocab,
            "max_documents": args.max_documents,
            "max_corpus_tokens": args.max_corpus_tokens,
            "block_size": args.block_size,
            "batch_size": args.batch_size,
            "train_steps": args.train_steps,
            "record_every": args.record_every,
            "bigram_eval_chunk_size": args.bigram_eval_chunk_size,
            "eval_batches": args.eval_batches,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": args.device,
            "train_tokens": train_ids.numel(),
            "val_tokens": val_ids.numel(),
            "active_vocab_size": len(active_token_ids),
            "default_epoch_steps": default_epoch_steps,
        },
        "reference_bigram_metrics": {
            **reference_bigram_metrics,
            "initial_train_loss": initial_train_loss,
            "initial_val_loss": initial_val_loss,
        },
        "final_metrics": {
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "last_step_loss": history["step_losses"][-1],
            "mean_step_loss": sum(history["step_losses"]) / len(history["step_losses"]),
        },
        "ceilings": ceilings,
        "history": history,
        "artifacts": {
            "plot_path": str(plot_path),
            "results_json": str(results_json),
        },
    }
    results_json.write_text(json.dumps(results, indent=2))
    plot_bigram_histories(results, plot_path)

    print(
        json.dumps(
            {
                "results_json": str(results_json),
                "plot_path": str(plot_path),
                "init_mode": args.init_mode,
                "selected_bigram_model": reference_bigram_metrics["selected_bigram_model"],
                "train_reference_bigram_accuracy": ceilings["train_reference_bigram_accuracy"],
                "val_reference_bigram_accuracy": ceilings["val_reference_bigram_accuracy"],
                "train_empirical_oracle_accuracy": ceilings["train_empirical_oracle_accuracy"],
                "val_empirical_oracle_accuracy": ceilings["val_empirical_oracle_accuracy"],
                "initial_val_bigram_accuracy": history["val_bigram_accuracy"][0],
                "final_val_bigram_accuracy": history["val_bigram_accuracy"][-1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
