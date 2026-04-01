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
    compute_streaming_sparse_log_bigram_matrix,
    encode_corpus,
    estimate_steps_for_one_epoch,
    gaussian_fill_missing_token_rows,
    initialize_backbone_as_identity,
    initialize_model_from_full_vocab_factors,
    initialize_model_from_svd,
    learn_factorized_bigram_tables,
    load_corpus_text,
    load_tokenizer,
    select_active_vocab,
    set_seed,
    sparse_factorize_log_bigrams,
    svd_factorize_log_bigrams,
    train_model,
    train_val_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep initialization schemes and plot training loss on a shared corpus."
    )
    parser.add_argument("--use-c4-filter-small", action="store_true", default=True)
    parser.add_argument("--dataset-arrow-path", type=str, default=None)
    parser.add_argument("--corpus-path", type=str, default=None)
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--config-name", choices=["tiny", "gpt2"], default="tiny")
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=["baseline", "topk", "streaming_full", "streaming_full_gaussian_fill", "factorized_bigram"],
    )
    parser.add_argument("--top-k-vocab", type=int, default=1024)
    parser.add_argument("--max-documents", type=int, default=500)
    parser.add_argument("--max-corpus-tokens", type=int, default=200000)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=3)
    parser.add_argument("--train-steps-override", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--bigram-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bigram-train-steps", type=int, default=200)
    parser.add_argument("--bigram-batch-size", type=int, default=256)
    parser.add_argument("--bigram-backbone-init", choices=["default", "identity"], default="identity")
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


def plot_scheme_histories(results: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    styles = {
        "baseline": {"linestyle": "--", "linewidth": 2.0},
        "topk": {"linestyle": "-", "linewidth": 2.0},
        "streaming_full": {"linestyle": "-.", "linewidth": 2.0},
        "streaming_full_gaussian_fill": {"linestyle": "--", "linewidth": 2.4},
        "streaming_full_identity": {"linestyle": "-", "linewidth": 2.6},
        "factorized_bigram": {"linestyle": ":", "linewidth": 2.4},
        "factorized_bigram_identity": {"linestyle": "-", "linewidth": 2.6},
    }
    for scheme, metrics in results["results"].items():
        steps = list(range(1, len(metrics["step_losses"]) + 1))
        style = styles.get(scheme, {})
        ax.plot(steps, metrics["step_losses"], label=scheme, **style)
    ax.set_title("Training Loss vs Step by Initialization Scheme")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_initialized_model(
    scheme: str,
    model_config,
    tokenizer_vocab_size: int,
    train_ids,
    val_ids,
    args: argparse.Namespace,
) -> tuple[GPT2LMHeadModel, dict[str, object]]:
    set_seed(args.seed)
    base_model = GPT2LMHeadModel(model_config)
    model = copy.deepcopy(base_model)
    init_metrics: dict[str, object] = {}

    if scheme == "baseline":
        return model, init_metrics

    if scheme == "topk":
        active_token_ids = select_active_vocab(train_ids.tolist(), args.top_k_vocab)
        log_bigram_matrix, _, init_metrics = build_smart_bigram_log_probs(
            train_token_ids=train_ids.tolist(),
            val_token_ids=val_ids.tolist(),
            active_token_ids=active_token_ids,
        )
        embedding_vectors, unembedding_vectors = svd_factorize_log_bigrams(
            log_bigram_matrix,
            d_model=model_config.n_embd,
        )
        model = BigramInitGPT2LMHeadModel(
            model_config,
            adapter_strategy=args.adapter_strategy,
        )
        initialize_model_from_svd(
            model,
            active_token_ids=active_token_ids,
            embedding_vectors=embedding_vectors,
            unembedding_vectors=unembedding_vectors,
        )
        init_metrics["active_vocab_size"] = len(active_token_ids)
        return model, init_metrics

    if scheme == "streaming_full":
        sparse_log_bigram_matrix = compute_streaming_sparse_log_bigram_matrix(
            train_ids.tolist(),
            vocab_size=tokenizer_vocab_size,
        )
        embedding_vectors, unembedding_vectors = sparse_factorize_log_bigrams(
            sparse_log_bigram_matrix,
            d_model=model_config.n_embd,
            random_state=args.seed,
        )
        initialize_model_from_full_vocab_factors(
            model,
            embedding_vectors=embedding_vectors,
            unembedding_vectors=unembedding_vectors,
        )
        return model, init_metrics

    if scheme == "streaming_full_identity":
        sparse_log_bigram_matrix = compute_streaming_sparse_log_bigram_matrix(
            train_ids.tolist(),
            vocab_size=tokenizer_vocab_size,
        )
        embedding_vectors, unembedding_vectors = sparse_factorize_log_bigrams(
            sparse_log_bigram_matrix,
            d_model=model_config.n_embd,
            random_state=args.seed,
        )
        initialize_model_from_full_vocab_factors(
            model,
            embedding_vectors=embedding_vectors,
            unembedding_vectors=unembedding_vectors,
        )
        initialize_backbone_as_identity(model)
        return model, init_metrics

    if scheme == "streaming_full_gaussian_fill":
        sparse_log_bigram_matrix = compute_streaming_sparse_log_bigram_matrix(
            train_ids.tolist(),
            vocab_size=tokenizer_vocab_size,
        )
        embedding_vectors, unembedding_vectors = sparse_factorize_log_bigrams(
            sparse_log_bigram_matrix,
            d_model=model_config.n_embd,
            random_state=args.seed,
        )
        embedding_vectors, unembedding_vectors = gaussian_fill_missing_token_rows(
            embedding=embedding_vectors,
            unembedding=unembedding_vectors,
            token_ids=train_ids.tolist(),
            vocab_size=tokenizer_vocab_size,
            seed=args.seed,
        )
        initialize_model_from_full_vocab_factors(
            model,
            embedding_vectors=embedding_vectors,
            unembedding_vectors=unembedding_vectors,
        )
        return model, init_metrics

    if scheme == "factorized_bigram":
        embedding_vectors, unembedding_vectors, init_metrics = learn_factorized_bigram_tables(
            train_data=train_ids,
            vocab_size=tokenizer_vocab_size,
            d_model=model_config.n_embd,
            batch_size=args.bigram_batch_size,
            train_steps=args.bigram_train_steps,
            learning_rate=args.bigram_learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            seed=args.seed,
        )
        initialize_model_from_full_vocab_factors(
            model,
            embedding_vectors=embedding_vectors,
            unembedding_vectors=unembedding_vectors,
        )
        return model, init_metrics

    if scheme == "factorized_bigram_identity":
        embedding_vectors, unembedding_vectors, init_metrics = learn_factorized_bigram_tables(
            train_data=train_ids,
            vocab_size=tokenizer_vocab_size,
            d_model=model_config.n_embd,
            batch_size=args.bigram_batch_size,
            train_steps=args.bigram_train_steps,
            learning_rate=args.bigram_learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            seed=args.seed,
        )
        initialize_model_from_full_vocab_factors(
            model,
            embedding_vectors=embedding_vectors,
            unembedding_vectors=unembedding_vectors,
        )
        initialize_backbone_as_identity(model)
        return model, init_metrics

    raise ValueError(f"Unsupported scheme={scheme!r}")


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
    if args.train_steps_override is not None:
        train_steps = args.train_steps_override
    model_config = build_model_config(
        config_name=args.config_name,
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
    )

    results: dict[str, object] = {
        "config": {
            "dataset_arrow_path": dataset_arrow_path,
            "corpus_path": args.corpus_path,
            "tokenizer_name": args.tokenizer_name,
            "config_name": args.config_name,
            "schemes": args.schemes,
            "top_k_vocab": args.top_k_vocab,
            "max_documents": args.max_documents,
            "max_corpus_tokens": args.max_corpus_tokens,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "train_steps": train_steps,
            "eval_batches": args.eval_batches,
            "train_steps_override": args.train_steps_override,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "bigram_learning_rate": args.bigram_learning_rate,
            "bigram_train_steps": args.bigram_train_steps,
            "bigram_batch_size": args.bigram_batch_size,
            "seed": args.seed,
            "device": args.device,
            "train_tokens": train_ids.numel(),
            "val_tokens": val_ids.numel(),
        },
        "results": {},
    }

    for scheme in args.schemes:
        model, init_metrics = build_initialized_model(
            scheme=scheme,
            model_config=model_config,
            tokenizer_vocab_size=tokenizer.vocab_size,
            train_ids=train_ids,
            val_ids=val_ids,
            args=args,
        )
        freeze_token_tables = scheme != "baseline"
        set_seed(args.seed)
        metrics = train_model(
            model=model,
            train_data=train_ids,
            val_data=val_ids,
            batch_size=args.batch_size,
            block_size=args.block_size,
            train_steps=train_steps,
            eval_batches=args.eval_batches,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            freeze_token_table_params=freeze_token_tables,
        )
        results["results"][scheme] = {
            **metrics,
            "init_metrics": init_metrics,
            "frozen_token_tables": freeze_token_tables,
        }

    json_path = output_dir / "scheme_sweep_results.json"
    plot_path = output_dir / "scheme_sweep_loss.png"
    json_path.write_text(json.dumps(results, indent=2))
    plot_scheme_histories(results, plot_path)
    print(json.dumps({"results_json": str(json_path), "plot_path": str(plot_path), **results["config"]}, indent=2))


if __name__ == "__main__":
    main()
