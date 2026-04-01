from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

from experiment import (
    C4_FILTER_SMALL_ARROW_PATH,
    build_model_config,
    build_smart_bigram_log_probs,
    encode_corpus,
    load_corpus_text,
    load_tokenizer,
    select_active_vocab,
    set_seed,
    svd_factorize_log_bigrams,
    train_val_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export top-k bigram SVD initialization factors and baseline token tables."
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/bigram_init_exports_18000")
    return parser.parse_args()


def resolve_dataset_path(args: argparse.Namespace) -> str | None:
    if args.dataset_arrow_path is not None:
        return args.dataset_arrow_path
    if args.use_c4_filter_small:
        return C4_FILTER_SMALL_ARROW_PATH
    return None


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_arrow_path = resolve_dataset_path(args)

    set_seed(args.seed)
    tokenizer = load_tokenizer(args.tokenizer_name)
    text = load_corpus_text(
        corpus_path=args.corpus_path,
        dataset_arrow_path=dataset_arrow_path,
        max_documents=args.max_documents,
    )
    token_ids = encode_corpus(tokenizer, text, args.max_corpus_tokens)
    train_ids, val_ids = train_val_split(token_ids)
    active_token_ids = select_active_vocab(train_ids.tolist(), args.top_k_vocab)
    model_config = build_model_config(
        config_name=args.config_name,
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        tiny_n_layer=args.tiny_n_layer,
        tiny_n_embd=args.tiny_n_embd,
    )

    log_bigram_matrix, _, init_metrics = build_smart_bigram_log_probs(
        train_token_ids=train_ids.tolist(),
        val_token_ids=val_ids.tolist(),
        active_token_ids=active_token_ids,
    )
    svd_embedding, svd_unembedding = svd_factorize_log_bigrams(
        log_bigram_matrix,
        d_model=model_config.n_embd,
    )
    baseline_model = GPT2LMHeadModel(model_config)

    svd_embedding_path = output_dir / "bigram_svd_embedding.pth"
    svd_unembedding_path = output_dir / "bigram_svd_unembedding.pth"
    baseline_embedding_path = output_dir / "baseline_embedding.pth"
    baseline_unembedding_path = output_dir / "baseline_unembedding.pth"
    active_token_ids_path = output_dir / "active_token_ids.pth"
    metadata_path = output_dir / "export_metadata.json"

    torch.save(svd_embedding.cpu(), svd_embedding_path)
    torch.save(svd_unembedding.cpu(), svd_unembedding_path)
    torch.save(baseline_model.transformer.wte.weight.detach().cpu(), baseline_embedding_path)
    torch.save(baseline_model.lm_head.weight.detach().cpu(), baseline_unembedding_path)
    torch.save(torch.tensor(active_token_ids, dtype=torch.long), active_token_ids_path)

    metadata = {
        "dataset_arrow_path": dataset_arrow_path,
        "corpus_path": args.corpus_path,
        "tokenizer_name": args.tokenizer_name,
        "config_name": args.config_name,
        "tiny_n_layer": args.tiny_n_layer,
        "tiny_n_embd": args.tiny_n_embd,
        "top_k_vocab": args.top_k_vocab,
        "block_size": args.block_size,
        "max_documents": args.max_documents,
        "max_corpus_tokens": args.max_corpus_tokens,
        "seed": args.seed,
        "train_tokens": train_ids.numel(),
        "val_tokens": val_ids.numel(),
        "active_vocab_size": len(active_token_ids),
        "selected_bigram_model": init_metrics["selected_bigram_model"],
        "selected_bigram_model_metadata": init_metrics["selected_bigram_model_metadata"],
        "bigram_model_candidates": init_metrics["bigram_model_candidates"],
        "artifacts": {
            "bigram_svd_embedding": str(svd_embedding_path),
            "bigram_svd_unembedding": str(svd_unembedding_path),
            "baseline_embedding": str(baseline_embedding_path),
            "baseline_unembedding": str(baseline_unembedding_path),
            "active_token_ids": str(active_token_ids_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(
        json.dumps(
            {
                "metadata_json": str(metadata_path),
                **metadata["artifacts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
