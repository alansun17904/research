"""Inspect sinks in the original pretrained attention, before any replacement."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .model import DEFAULT_MODEL, load_model


def summarize_attention(weights, query_start=128, prefix_tokens=4, threshold=0.1):
    """Reduce [heads, queries, keys] probabilities using only later queries."""
    assert 2 * prefix_tokens <= query_start < weights.shape[-2]
    queries = torch.arange(query_start, weights.shape[-2], device=weights.device)
    weights = weights[:, query_start:].float()
    first = weights[..., 0]
    prefix = weights[..., :prefix_tokens].sum(-1)
    control = weights[..., prefix_tokens:2 * prefix_tokens].sum(-1)
    recent_indices = queries[:, None] - torch.arange(prefix_tokens, device=weights.device)
    recent = weights.gather(-1, recent_indices.expand(weights.shape[0], -1, -1)).sum(-1)
    uniform = 1 / (queries + 1)
    profile = weights.mean(-2)
    peak = profile.argmax(-1)
    peak_mass = weights.gather(-1, peak[:, None, None].expand(-1, len(queries), 1)).squeeze(-1)
    return {
        "position_profile": profile.cpu(),
        "first_mass": first.mean(-1).cpu(),
        "prefix_mass": prefix.mean(-1).cpu(),
        "control_mass": control.mean(-1).cpu(),
        "recent_mass": recent.mean(-1).cpu(),
        "first_query_fraction": ((first >= threshold) & (first >= 5 * uniform)).float().mean(-1).cpu(),
        "prefix_query_fraction": ((prefix >= threshold) & (prefix >= 5 * prefix_tokens * uniform)).float().mean(-1).cpu(),
        "peak_mass": peak_mass.mean(-1).cpu(),
        "peak_query_fraction": ((peak_mass >= threshold) & (peak_mass >= 5 * uniform)).float().mean(-1).cpu(),
    }


@torch.inference_mode()
def detect_sinks(model, input_ids, *, query_start=128, prefix_tokens=4, threshold=0.1):
    """Independent, equal-length, unpadded sequences; original HF forward only."""
    model.eval()
    samples = []
    for ids in input_ids:
        output = model.base_model(ids[None].to(model.device), use_cache=False,
                                  output_attentions=True, return_dict=True)
        samples.append([summarize_attention(a[0], query_start, prefix_tokens, threshold)
                        for a in output.attentions])
        del output  # Keep summaries, not the quadratic attention matrices.
    stats = {key: torch.stack([torch.stack([layer[key] for layer in sample]) for sample in samples])
             for key in samples[0][0]}
    heads = []
    for layer in range(stats["prefix_mass"].shape[1]):
        for head in range(stats["prefix_mass"].shape[2]):
            row = {key: values[:, layer, head].mean(0).tolist() for key, values in stats.items()}
            peak_positions = stats["position_profile"][:, layer, head].argmax(-1)
            row.update(layer=layer, head=head,
                       per_sequence_peak_position=peak_positions.tolist(),
                       per_sequence_peak_token_id=input_ids.cpu().gather(1, peak_positions[:, None]).flatten().tolist(),
                       per_sequence_peak_mass=stats["peak_mass"][:, layer, head].tolist(),
                       per_sequence_first_mass=stats["first_mass"][:, layer, head].tolist(),
                       per_sequence_prefix_mass=stats["prefix_mass"][:, layer, head].tolist(),
                       persistent_prefix_sequence_fraction=(
                           stats["prefix_query_fraction"][:, layer, head] >= 0.5).float().mean().item())
            heads.append(row)
    uniform = (1 / torch.arange(query_start + 1, input_ids.shape[1] + 1).double()).mean().item()
    return {
        "source": "Original Hugging Face eager attention probabilities; no replacement or eviction",
        "num_sequences": len(input_ids), "sequence_length": input_ids.shape[1],
        "query_start": query_start, "prefix_tokens": prefix_tokens, "threshold": threshold,
        "uniform_first_mass": uniform, "uniform_prefix_mass": prefix_tokens * uniform,
        "persistent_prefix_rule": "Mass >= threshold and >=5x causal-uniform on >=50% of later queries in >=75% of sequences",
        "persistent_prefix_heads": sum(row["persistent_prefix_sequence_fraction"] >= 0.75 for row in heads),
        "heads": heads,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--texts", type=Path, required=True, help="JSON list of source documents")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=32)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--query-start", type=int, default=128)
    parser.add_argument("--prefix-tokens", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--prepend-bos", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    runner, tokenizer = load_model(args.model, device=args.device, cache_dir=args.cache_dir,
                                   local_files_only=args.local_files_only)
    tokens = [token for text in json.loads(args.texts.read_text())
              for token in tokenizer.encode(text + "\n\n", add_special_tokens=False)]
    tokens = tokens[args.offset:args.offset + args.sequences * args.length]
    assert len(tokens) == args.sequences * args.length, "Provide more source text"
    ids = torch.tensor(tokens).reshape(args.sequences, args.length)
    if args.prepend_bos:
        ids = torch.cat((torch.full_like(ids[:, :1], tokenizer.bos_token_id), ids[:, :-1]), dim=1)
    result = detect_sinks(runner.hf_model, ids, query_start=args.query_start,
                          prefix_tokens=args.prefix_tokens, threshold=args.threshold)
    result["model"] = args.model
    result["model_revision"] = runner.config._commit_hash
    result["arguments"] = vars(args)
    result["source_sha256"] = hashlib.sha256(args.texts.read_bytes()).hexdigest()
    result["input_token_sha256"] = hashlib.sha256(ids.numpy().astype("<i8").tobytes()).hexdigest()
    result["first_tokens"] = [tokenizer.convert_ids_to_tokens(row[:8].tolist()) for row in ids]
    for head in result["heads"]:
        head["per_sequence_peak_tokens"] = tokenizer.convert_ids_to_tokens(head["per_sequence_peak_token_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str, allow_nan=False) + "\n")
    print(f"{args.model}: {result['persistent_prefix_heads']}/{len(result['heads'])} persistent prefix heads")
    print(f"Mean first-token mass: {sum(h['first_mass'] for h in result['heads']) / len(result['heads']):.2%}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
