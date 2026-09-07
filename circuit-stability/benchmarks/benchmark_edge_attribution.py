from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformer_lens import HookedTransformerConfig
from transformer_lens.utils import get_attention_mask

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover
    psutil = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eap import Graph, attribute


LOCAL_CACHE_MODEL_REPOS = {
    "attn-only-2l": "NeelNanda/Attn_Only_2L512W_C4_Code",
}


class PromptPairDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], model):
        self.rows = rows
        self.model = model

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        return row["clean"], row["corrupted"], row.get("label", row["clean"])

    def collate(self, items):
        clean, corrupted, labels = zip(*items)
        all_prompts = list(clean) + list(corrupted)
        tokens = self.model.to_tokens(
            all_prompts, prepend_bos=True, padding_side="left"
        )
        attention_mask = get_attention_mask(self.model.tokenizer, tokens, True)
        input_lengths = attention_mask.sum(1)
        n_pos = attention_mask.size(1)
        batch_size = len(clean)
        return (
            (
                tokens[:batch_size],
                attention_mask[:batch_size],
                input_lengths[:batch_size],
                n_pos,
            ),
            (
                tokens[batch_size:],
                attention_mask[batch_size:],
                input_lengths[batch_size:],
                n_pos,
            ),
            list(labels),
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def kl_metric(model, logits, clean_logits, input_length, labels):
    logit_probs = F.log_softmax(logits, dim=-1)
    clean_probs = F.softmax(clean_logits, dim=-1)
    return F.kl_div(logit_probs, clean_probs, reduction="batchmean", log_target=True)


def extract_last_token(model, logits, clean_logits, input_length, labels):
    return logits[:, -1], clean_logits[:, -1], labels


def extraction_schema(extract_fn, model):
    def decorator(metric_fn):
        def wrapper(logits, clean_logits, input_length, labels, model=model):
            logits_out, clean_out, labels_out = extract_fn(
                model, logits, clean_logits, input_length, labels
            )
            return metric_fn(model, logits_out, clean_out, input_length, labels_out)

        return wrapper

    return decorator


def get_metric(metric_id: str):
    if metric_id != "kl":
        raise ValueError(f"Unsupported metric: {metric_id}")
    return kl_metric


def get_extraction(extraction_id: str):
    if extraction_id != "last_token":
        raise ValueError(f"Unsupported extraction: {extraction_id}")
    return extract_last_token


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", required=True, help="TransformerLens model names."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("benchmarks/data/edge_attribution_pairs.jsonl"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric", type=str, default="kl")
    parser.add_argument("--extraction", type=str, default="last_token")
    parser.add_argument(
        "--chunk-mode",
        choices=["none", "fixed", "layer"],
        default="none",
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--source-chunk-size", type=int, default=None)
    parser.add_argument(
        "--token-aggregation",
        choices=["all-tok", "last-tok", "avg-tok"],
        default="all-tok",
    )
    parser.add_argument("--patch-type", choices=["edge", "node"], default="edge")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rss_megabytes() -> float:
    if psutil is not None:
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    return 0.0


def _cached_snapshot_path(repo_id: str) -> Path:
    root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}"
    ref_path = root / "refs" / "main"
    if not ref_path.exists():
        raise FileNotFoundError(f"No local Hugging Face ref found for {repo_id}.")
    snapshot = root / "snapshots" / ref_path.read_text().strip()
    if not snapshot.exists():
        raise FileNotFoundError(f"No cached snapshot found for {repo_id}.")
    return snapshot


def _load_cached_neel_model(model_name: str, repo_id: str):
    from transformer_lens import HookedTransformer

    snapshot = _cached_snapshot_path(repo_id)
    with (snapshot / "config.json").open() as handle:
        cfg_json = json.load(handle)

    cfg_dict = {
        "model_name": model_name,
        "init_weights": False,
        "device": "cpu",
        "dtype": torch.float32,
        "d_model": cfg_json["d_model"],
        "n_layers": cfg_json["n_layers"],
        "d_mlp": cfg_json["d_mlp"],
        "d_head": cfg_json["d_head"],
        "n_heads": cfg_json["n_heads"],
        "n_ctx": cfg_json["n_ctx"],
        "d_vocab": cfg_json["d_vocab"],
        "tokenizer_name": cfg_json.get("tokenizer_name"),
        "act_fn": cfg_json["act_fn"],
        "attn_only": cfg_json["attn_only"],
        "final_rms": cfg_json.get("final_rms", False),
        "original_architecture": cfg_json.get("architecture", "neel"),
        "normalization_type": (
            cfg_json["normalization"]
            if "normalization" in cfg_json
            else cfg_json["normalization_type"]
        ),
        "positional_embedding_type": (
            "shortformer" if cfg_json.get("shortformer_pos", False) else "standard"
        ),
    }
    tokenizer = None
    tokenizer_name = cfg_json.get("tokenizer_name")
    if tokenizer_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            str(_cached_snapshot_path(tokenizer_name)),
            local_files_only=True,
        )
    model = HookedTransformer(
        HookedTransformerConfig.from_dict(cfg_dict),
        tokenizer=tokenizer,
        default_padding_side="left",
    )
    state_dict = torch.load(snapshot / "model_final.pth", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    return model


def load_model(model_name: str):
    repo_id = LOCAL_CACHE_MODEL_REPOS.get(model_name)
    if repo_id is not None:
        return _load_cached_neel_model(model_name, repo_id)

    from transformer_lens import HookedTransformer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return HookedTransformer.from_pretrained(
        model_name,
        device="cpu",
        local_files_only=True,
    )


def chunked_backward_specs(graph: Graph, chunk_mode: str, chunk_size: int | None):
    specs = list(graph.backward_specs)
    if chunk_mode == "none":
        return [specs]
    if chunk_mode == "layer":
        chunks = []
        offset = 0
        for _layer in range(graph.cfg["n_layers"]):
            chunks.append(specs[offset : offset + 4])
            offset += 4
        chunks.append(specs[offset:])
        return [chunk for chunk in chunks if chunk]
    if chunk_mode == "fixed":
        if chunk_size is None or chunk_size <= 0:
            raise ValueError("chunk_size must be positive when chunk_mode='fixed'.")
        return [
            specs[start : start + chunk_size]
            for start in range(0, len(specs), chunk_size)
        ]
    raise ValueError(f"Unsupported chunk_mode: {chunk_mode}")


def estimated_activation_buffer_mb(
    graph: Graph,
    model,
    n_pos: int,
    chunk_mode: str,
    chunk_size: int | None,
    source_chunk_size: int | None,
    token_aggregation: str,
    patch_type: str = "edge",
) -> float:
    if patch_type == "node":
        peak_forward = min(source_chunk_size or graph.n_forward, graph.n_forward)
    else:
        chunks = chunked_backward_specs(graph, chunk_mode, chunk_size)
        peak_forward = max(
            min(max(spec[1] for spec in chunk), source_chunk_size or graph.n_forward)
            for chunk in chunks
        )
    bytes_per_value = torch.tensor([], dtype=model.cfg.dtype).element_size()
    token_extent = n_pos if token_aggregation == "all-tok" else 1
    total_bytes = token_extent * peak_forward * model.cfg.d_model * bytes_per_value
    return total_bytes / (1024.0 * 1024.0)


def estimated_score_tensor_mb(graph: Graph, model, patch_type: str) -> float:
    bytes_per_value = torch.tensor([], dtype=model.cfg.dtype).element_size()
    score_entries = (
        graph.n_forward * graph.n_backward if patch_type == "edge" else graph.n_forward
    )
    return (score_entries * bytes_per_value) / (1024.0 * 1024.0)


def run_once(model_name: str, rows: list[dict[str, str]], args):
    model = load_model(model_name)
    model.eval()
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True

    dataset = PromptPairDataset(rows, model)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=dataset.collate
    )
    first_batch = next(iter(dataloader))

    metric = extraction_schema(get_extraction(args.extraction), model)(
        get_metric(args.metric)
    )
    graph = Graph.from_model(model)
    batch_n_pos = first_batch[0][3]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_rss = rss_megabytes()
    start = time.perf_counter()
    score_vector = attribute(
        model=model,
        graph=graph,
        dataloader=dataloader,
        metric=metric,
        ig_steps=args.ig_steps,
        quiet=True,
        chunk_mode=args.chunk_mode,
        chunk_size=args.chunk_size,
        source_chunk_size=args.source_chunk_size,
        token_aggregation=args.token_aggregation,
        patch_type=args.patch_type,
    )
    elapsed = time.perf_counter() - start
    peak_cuda_mb = 0.0
    if torch.cuda.is_available():
        peak_cuda_mb = torch.cuda.max_memory_allocated() / (1024**2)

    activation_buffer_mb_est = estimated_activation_buffer_mb(
        graph=graph,
        model=model,
        n_pos=batch_n_pos,
        chunk_mode=args.chunk_mode,
        chunk_size=args.chunk_size,
        source_chunk_size=args.source_chunk_size,
        token_aggregation=args.token_aggregation,
        patch_type=args.patch_type,
    )
    score_tensor_mb_est = estimated_score_tensor_mb(
        graph=graph, model=model, patch_type=args.patch_type
    )
    return {
        "model": model_name,
        "examples": len(rows),
        "batch_size": args.batch_size,
        "ig_steps": args.ig_steps,
        "chunk_mode": args.chunk_mode,
        "chunk_size": args.chunk_size,
        "source_chunk_size": args.source_chunk_size,
        "token_aggregation": args.token_aggregation,
        "patch_type": args.patch_type,
        "latency_s": elapsed,
        "rss_mb_delta": rss_megabytes() - start_rss,
        "activation_buffer_mb_est": activation_buffer_mb_est,
        "score_tensor_mb_est": score_tensor_mb_est,
        "attribution_workspace_mb_est": (
            activation_buffer_mb_est + score_tensor_mb_est
        ),
        "peak_cuda_mb": peak_cuda_mb,
        "score_count": len(score_vector),
        "score_sum": float(score_vector.sum()),
        "score_abs_sum": float(abs(score_vector).sum()),
    }, score_vector


def main():
    args = parse_args()
    seed_everything(args.seed)
    rows = load_rows(args.dataset)

    for model_name in args.models:
        baseline_result = None
        baseline_vector = None
        for repeat in range(args.repeats):
            result, score_vector = run_once(model_name, rows, args)
            result["repeat"] = repeat
            if baseline_vector is None:
                baseline_vector = score_vector
                baseline_result = result
            else:
                result["exact_match_previous"] = bool(
                    (score_vector == baseline_vector).all()
                )
            print(json.dumps(result))

        if baseline_result is not None:
            print(
                json.dumps(
                    {
                        "model": model_name,
                        "summary": "baseline",
                        "latency_s": baseline_result["latency_s"],
                        "rss_mb_delta": baseline_result["rss_mb_delta"],
                        "peak_cuda_mb": baseline_result["peak_cuda_mb"],
                    }
                )
            )


if __name__ == "__main__":
    main()
