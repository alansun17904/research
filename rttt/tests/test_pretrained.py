"""Opt-in validation on cached weights: RTTT_PRETRAINED_TESTS=1 pytest -q."""

import os

import pytest
import torch

from rttt.attention import AttentionConfig
from rttt.model import load_model


@pytest.mark.skipif(os.environ.get("RTTT_PRETRAINED_TESTS") != "1", reason="Opt-in cached pretrained checkpoint test")
def test_cached_tinystories_logits_match_huggingface():
    runner, tokenizer = load_model(device="cpu", cache_dir=".cache/huggingface", local_files_only=True)
    ids = tokenizer.encode("Once upon a time, there was a little girl who found a red ball.", return_tensors="pt")
    with torch.no_grad():
        expected = runner.hf_model(ids).logits
    actual = torch.stack([runner.step(token) for token in ids.unbind(1)], dim=1)
    torch.testing.assert_close(actual, expected, atol=5e-5, rtol=1e-4)
    for method in ("h2o", "streaming"):
        runner.set_attention(AttentionConfig(method=method, cache_size=64))
        torch.testing.assert_close(runner.prefill(ids), expected[:, -1], atol=5e-5, rtol=1e-4)
