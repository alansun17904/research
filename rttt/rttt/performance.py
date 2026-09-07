"""Synchronized batch-one prefill/decode timing versus context length."""

import statistics
import time

import torch

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def benchmark_speed(model, context_lengths=(64, 128, 256, 512), decode_tokens=32, repeats=3, seed=42):
    limit = model.max_sequence_length
    if limit and max(context_lengths) + decode_tokens > limit:
        raise ValueError("Requested timing sequence exceeds learned positions; shorten it or use Pythia")
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(model.config.vocab_size, (max(context_lengths) + decode_tokens,), generator=generator).to(model.device)
    # Warm up kernels without contaminating measured sequence state.
    model.reset()
    model.prefill(ids[:8].unsqueeze(0))
    synchronize(model.device)
    results = []
    for length in context_lengths:
        samples = []
        for _ in range(repeats):
            model.reset()
            if model.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(model.device)
            synchronize(model.device)
            started = time.perf_counter()
            model.prefill(ids[:length].unsqueeze(0))
            synchronize(model.device)
            prefill_seconds = time.perf_counter() - started
            state_after_prefill = model.state_bytes
            started = time.perf_counter()
            model.prefill(ids[length:length + decode_tokens].unsqueeze(0))
            synchronize(model.device)
            decode_seconds = time.perf_counter() - started
            samples.append({"prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
                            "prefill_tokens_per_second": length / prefill_seconds,
                            "decode_tokens_per_second": decode_tokens / decode_seconds,
                            "decode_ms_per_token": decode_seconds * 1000 / decode_tokens,
                            "prefill_state_bytes": state_after_prefill, "decode_state_bytes": model.state_bytes,
                            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(model.device)
                                if model.device.type == "cuda" else None})
        results.append({"context_length": length, "decode_tokens": decode_tokens, "samples": samples,
                        "median_prefill_tokens_per_second": statistics.median(s["prefill_tokens_per_second"] for s in samples),
                        "median_decode_ms_per_token": statistics.median(s["decode_ms_per_token"] for s in samples)})
    return {"protocol": "batch-one sequential prefill and teacher-forced synthetic decode; model forward only",
            "repeats": repeats, "results": results}
