"""Read-only CUDA diagnostics on the first 2,048 WikiText validation tokens."""

import hashlib
import json
from pathlib import Path
import struct
import sys

import torch
from datasets import load_dataset
from torch.nn import functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import MODEL, MODEL_REVISION, DATASET, DATASET_CONFIG, DATASET_REVISION, SPLIT
from model import CHECKPOINT, load_model, read_graph


def relation(delta, reference):
    delta, reference = delta.flatten().float(), reference.flatten().float()
    delta_rms = delta.square().mean().sqrt()
    reference_rms = reference.square().mean().sqrt()
    assert delta_rms > 0 and reference_rms > 0
    return {
        "rms": delta_rms.item(),
        "rms_over_reference_rms": (delta_rms / reference_rms).item(),
        "cosine_with_reference": F.cosine_similarity(delta, reference, dim=0).item(),
    }


def main():
    assert torch.cuda.is_available(), "Run this diagnostic on the allocated CUDA node."
    baseline_path = ROOT / "graphs/baseline.json"
    baseline = read_graph(baseline_path)
    model = load_model(baseline_path).to("cuda")
    assert all(not parameter.requires_grad for parameter in model.parameters())
    versions = {name: parameter._version for name, parameter in model.named_parameters()}
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    rows = load_dataset(DATASET, DATASET_CONFIG, split=SPLIT,
                        revision=DATASET_REVISION, streaming=True)
    text = "\n\n".join(row["text"] for row in rows)
    ids = tokenizer(text, add_special_tokens=False, truncation=True,
                    max_length=2048, return_tensors="pt")["input_ids"]
    assert ids.shape == (1, 2048)
    token_hash = hashlib.sha256(struct.pack("<2048q", *ids[0].tolist())).hexdigest()
    ids = ids.to(model.device)
    paths = [baseline_path] + [ROOT / f"graphs/screen/repeat-mlp-l{i}.json" for i in range(6)]
    results = []
    with torch.inference_mode():
        unhooked = model(ids).logits
        for path in paths:
            model.graph = read_graph(path)
            assert model.graph["sequence"] == baseline["sequence"]
            captures = [{"inputs": [], "deltas": [], "positive_fractions": []} for _ in range(6)]
            hooks = []
            for index, layer in enumerate(model.gpt_neox.layers):
                def capture_input(module, args, index=index):
                    captures[index]["inputs"].append(args[0].detach())
                    return None

                def capture_delta(module, args, output, index=index):
                    captures[index]["deltas"].append(output.detach())
                    return None

                def capture_pre_gelu(module, args, output, index=index):
                    captures[index]["positive_fractions"].append((output > 0).float().mean().item())
                    return None

                hooks.extend((layer.register_forward_pre_hook(capture_input),
                              layer.mlp.register_forward_hook(capture_delta),
                              layer.mlp.dense_h_to_4h.register_forward_hook(capture_pre_gelu)))
            try:
                hooked = model(ids).logits
                assert torch.isfinite(hooked).all()
                if path == baseline_path:
                    torch.testing.assert_close(hooked, unhooked, atol=0, rtol=0)
                    baseline_hook_error = (hooked - unhooked).abs().max().item()
                    del unhooked
            finally:
                for hook in hooks:
                    hook.remove()
            del hooked
            layers = []
            for index, capture in enumerate(captures):
                expected = model.graph["layers"][index]["mlp_repeat"]
                assert len(capture["inputs"]) == 1
                assert len(capture["deltas"]) == len(capture["positive_fractions"]) == expected
                assert expected == (2 if path.name == f"repeat-mlp-l{index}.json" else 1)
                hidden = capture["inputs"][0]
                deltas = capture["deltas"]
                total = deltas[0]
                for delta in deltas[1:]:
                    total = total + delta
                layers.append({
                    "layer": index, "mlp_applications": expected,
                    "layer_input_rms": hidden.float().square().mean().sqrt().item(),
                    "total_mlp_delta_vs_layer_input": relation(total, hidden),
                    "individual_deltas_vs_layer_input": [relation(delta, hidden) for delta in deltas],
                    "second_delta_vs_first": relation(deltas[1], deltas[0]) if expected == 2 else None,
                    "pre_gelu_positive_fraction_per_application": capture["positive_fractions"],
                })
            assert all(not parameter.requires_grad for parameter in model.parameters())
            assert versions == {name: parameter._version for name, parameter in model.named_parameters()}
            results.append({"graph": str(path.relative_to(ROOT)), "layers": layers})
    print(json.dumps({
        "scope": "Diagnostic only; one full-context forward, not an evaluation or perplexity estimate.",
        "input_tokens": 2048, "available_next_token_targets": 2047, "scored_tokens": 0,
        "sample": "deterministic prefix; rows joined with two newlines; no special tokens",
        "token_sha256": token_hash, "token_hash_encoding": "2048 little-endian signed int64 IDs",
        "model": MODEL, "model_revision": MODEL_REVISION, "dataset": DATASET,
        "dataset_config": DATASET_CONFIG, "dataset_revision": DATASET_REVISION, "split": SPLIT,
        "device": str(model.device), "dtype": str(next(model.parameters()).dtype),
        "cosine_scope": "Flattened across batch, token positions and hidden dimensions.",
        "baseline_hook_max_absolute_logit_error": baseline_hook_error,
        "all_parameters_frozen": True, "parameter_version_counters_unchanged": True,
        "results": results,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
