from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_edge_attribution import (  # noqa: E402
    PromptPairDataset,
    estimated_activation_buffer_mb,
    extraction_schema,
    get_extraction,
    get_metric,
    rss_megabytes,
    seed_everything,
)
from eap import Graph, attribute  # noqa: E402


TOKEN_MODES = ("all-tok", "last-tok", "avg-tok")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="attn-only-2l")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--ig-steps", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric", type=str, default="kl")
    parser.add_argument("--extraction", type=str, default="last_token")
    parser.add_argument("--chunk-mode", choices=["none", "fixed", "layer"], default="none")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--source-chunk-size", type=int, default=None)
    parser.add_argument("--min-words", type=int, default=64)
    parser.add_argument("--max-words", type=int, default=64)
    parser.add_argument(
        "--dataset-out",
        type=Path,
        default=Path("benchmarks/data/c4_filter_small_1000_pairs.jsonl"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmarks/results/c4_filter_small_1000_token_aggregation"),
    )
    return parser.parse_args()


def _sanitize_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def build_c4_pairs(path: Path, rows: int, min_words: int, max_words: int) -> list[dict[str, str]]:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    dataset = load_dataset("datablations/c4-filter-small", split="train")
    prompts: list[str] = []
    for record in dataset:
        text = _sanitize_text(record["text"])
        if len(text.split()) < min_words:
            continue
        prompts.append(_truncate_words(text, max_words))
        if len(prompts) >= 2 * rows:
            break
    if len(prompts) < 2 * rows:
        raise ValueError(f"Needed {2 * rows} prompts, found {len(prompts)}.")

    data = []
    for index in range(rows):
        data.append(
            {
                "clean": prompts[index],
                "corrupted": prompts[index + rows],
                "label": "",
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in data:
            handle.write(json.dumps(row) + "\n")
    return data


def prepare_inputs(rows: list[dict[str, str]], model, batch_size: int):
    dataset = PromptPairDataset(rows, model)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate)


def benchmark_mode(
    model,
    graph: Graph,
    dataloader,
    metric,
    args,
    token_mode: str,
):
    first_batch = next(iter(dataloader))
    batch_n_pos = first_batch[0][3]
    latencies: list[float] = []
    rss_deltas: list[float] = []
    vectors: list[np.ndarray] = []

    for _repeat in range(args.repeats):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start_rss = rss_megabytes()
        start = time.perf_counter()
        edge_vector = attribute(
            model=model,
            graph=graph,
            dataloader=dataloader,
            metric=metric,
            ig_steps=args.ig_steps,
            quiet=True,
            chunk_mode=args.chunk_mode,
            chunk_size=args.chunk_size,
            source_chunk_size=args.source_chunk_size,
            token_aggregation=token_mode,
        )
        latencies.append(time.perf_counter() - start)
        rss_deltas.append(rss_megabytes() - start_rss)
        vectors.append(edge_vector.copy())

    activation_buffer_mb = estimated_activation_buffer_mb(
        graph=graph,
        model=model,
        n_pos=batch_n_pos,
        chunk_mode=args.chunk_mode,
        chunk_size=args.chunk_size,
        source_chunk_size=args.source_chunk_size,
        token_aggregation=token_mode,
    )
    exact_match = all(np.array_equal(vectors[0], vector) for vector in vectors[1:])
    peak_cuda_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    )
    return {
        "token_aggregation": token_mode,
        "mean_latency_s": statistics.mean(latencies),
        "stdev_latency_s": statistics.pstdev(latencies),
        "latencies_s": latencies,
        "mean_rss_mb_delta": statistics.mean(rss_deltas),
        "activation_buffer_mb_est": activation_buffer_mb,
        "peak_cuda_mb": peak_cuda_mb,
        "exact_match_repeats": exact_match,
    }, vectors[-1]


def correlation_table(vectors: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for left_name, left_vector in vectors.items():
        table[left_name] = {}
        for right_name, right_vector in vectors.items():
            table[left_name][right_name] = float(spearmanr(left_vector, right_vector).statistic)
    return table


def save_correlation_csv(path: Path, table: dict[str, dict[str, float]]) -> None:
    modes = list(table.keys())
    with path.open("w") as handle:
        handle.write("," + ",".join(modes) + "\n")
        for left_mode in modes:
            values = [f"{table[left_mode][right_mode]:.6f}" for right_mode in modes]
            handle.write(left_mode + "," + ",".join(values) + "\n")


def main():
    args = parse_args()
    seed_everything(args.seed)
    rows = build_c4_pairs(
        path=args.dataset_out,
        rows=args.rows,
        min_words=args.min_words,
        max_words=args.max_words,
    )

    model = HookedTransformer.from_pretrained(args.model)
    model.eval()
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True

    dataloader = prepare_inputs(rows, model, args.batch_size)
    metric = extraction_schema(get_extraction(args.extraction), model)(get_metric(args.metric))
    graph = Graph.from_model(model)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    vectors: dict[str, np.ndarray] = {}
    for token_mode in TOKEN_MODES:
        summary, vector = benchmark_mode(model, graph, dataloader, metric, args, token_mode)
        summaries.append(summary)
        vectors[token_mode] = vector
        np.save(args.results_dir / f"{token_mode}_edge_scores.npy", vector)

    correlations = correlation_table(vectors)
    save_correlation_csv(args.results_dir / "spearman_rank_correlation.csv", correlations)

    output = {
        "model": args.model,
        "dataset_path": str(args.dataset_out),
        "rows": args.rows,
        "batch_size": args.batch_size,
        "ig_steps": args.ig_steps,
        "repeats": args.repeats,
        "chunk_mode": args.chunk_mode,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "results_dir": str(args.results_dir),
        "summaries": summaries,
        "spearman_rank_correlation": correlations,
    }
    with (args.results_dir / "summary.json").open("w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
