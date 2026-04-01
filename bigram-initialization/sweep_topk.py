from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
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
        description="Sweep top_k_vocab values and plot training loss for baseline vs frozen bigram initialization."
    )
    parser.add_argument("--use-c4-filter-small", action="store_true", default=True)
    parser.add_argument("--dataset-arrow-path", type=str, default=None)
    parser.add_argument("--corpus-path", type=str, default=None)
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--config-name", choices=["tiny", "gpt2"], default="tiny")
    parser.add_argument("--top-k-vocabs", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--max-documents", type=int, default=500)
    parser.add_argument("--max-corpus-tokens", type=int, default=200000)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adapter-strategy", choices=["linear", "rotation_scale", "rotation_only"], default="linear")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def resolve_dataset_path(args: argparse.Namespace) -> str | None:
    if args.dataset_arrow_path is not None:
        return args.dataset_arrow_path
    if args.use_c4_filter_small:
        return C4_FILTER_SMALL_ARROW_PATH
    return None


def plot_histories(sweep_results: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for top_k, result in sweep_results["results"].items():
        steps = list(range(1, len(result["random_init"]["step_losses"]) + 1))
        ax.plot(
            steps,
            result["random_init"]["step_losses"],
            linestyle="--",
            linewidth=1.8,
            label=f"baseline top_k={top_k}",
        )
        ax.plot(
            steps,
            result["svd_init"]["step_losses"],
            linestyle="-",
            linewidth=2.0,
            label=f"bigram-frozen top_k={top_k}",
        )
    ax.set_title("Training Loss vs Step on Cached C4 Subset")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


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
    train_steps = estimate_steps_for_one_epoch(
        train_token_count=train_ids.numel(),
        batch_size=args.batch_size,
        block_size=args.block_size,
    )
    model_config = build_model_config(
        config_name=args.config_name,
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
    )

    sweep_results: dict[str, object] = {
        "config": {
            "dataset_arrow_path": dataset_arrow_path,
            "corpus_path": args.corpus_path,
            "tokenizer_name": args.tokenizer_name,
            "config_name": args.config_name,
            "top_k_vocabs": args.top_k_vocabs,
            "max_documents": args.max_documents,
            "max_corpus_tokens": args.max_corpus_tokens,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "train_steps": train_steps,
            "eval_batches": args.eval_batches,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": args.device,
            "train_tokens": train_ids.numel(),
            "val_tokens": val_ids.numel(),
        },
        "results": {},
    }

    for top_k_vocab in args.top_k_vocabs:
        active_token_ids = select_active_vocab(train_ids.tolist(), top_k_vocab)
        log_bigram_matrix, _, init_metrics = build_smart_bigram_log_probs(
            train_token_ids=train_ids.tolist(),
            val_token_ids=val_ids.tolist(),
            active_token_ids=active_token_ids,
        )
        embedding_vectors, unembedding_vectors = svd_factorize_log_bigrams(
            log_bigram_matrix,
            d_model=model_config.n_embd,
        )

        base_model = GPT2LMHeadModel(model_config)
        random_model = copy.deepcopy(base_model)
        svd_model = BigramInitGPT2LMHeadModel(
            model_config,
            adapter_strategy=args.adapter_strategy,
        )
        initialize_model_from_svd(
            svd_model,
            active_token_ids=active_token_ids,
            embedding_vectors=embedding_vectors,
            unembedding_vectors=unembedding_vectors,
        )

        random_metrics = train_model(
            model=random_model,
            train_data=train_ids,
            val_data=val_ids,
            batch_size=args.batch_size,
            block_size=args.block_size,
            train_steps=train_steps,
            eval_batches=args.eval_batches,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            freeze_token_table_params=False,
        )
        svd_metrics = train_model(
            model=svd_model,
            train_data=train_ids,
            val_data=val_ids,
            batch_size=args.batch_size,
            block_size=args.block_size,
            train_steps=train_steps,
            eval_batches=args.eval_batches,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            freeze_token_table_params=True,
        )

        sweep_results["results"][str(top_k_vocab)] = {
            "active_vocab_size": len(active_token_ids),
            "init_metrics": init_metrics,
            "random_init": random_metrics,
            "svd_init": svd_metrics,
        }

    json_path = output_dir / "topk_sweep_results.json"
    plot_path = output_dir / "topk_sweep_loss.png"
    json_path.write_text(json.dumps(sweep_results, indent=2))
    plot_histories(sweep_results, plot_path)
    print(json.dumps({"results_json": str(json_path), "plot_path": str(plot_path), **sweep_results["config"]}, indent=2))


if __name__ == "__main__":
    main()
