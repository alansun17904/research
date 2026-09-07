import pytest
import torch
from transformers import GPTNeoConfig, GPTNeoForCausalLM

from rttt.sinks import detect_sinks, summarize_attention


def test_uniform_causal_attention_has_no_prefix_enrichment():
    weights = torch.ones(2, 512, 512).tril()
    weights /= weights.sum(-1, keepdim=True)
    stats = summarize_attention(weights)
    uniform = (1 / torch.arange(129, 513).float()).mean()
    torch.testing.assert_close(stats["first_mass"], uniform.expand(2))
    for key in ("prefix_mass", "control_mass", "recent_mass"):
        torch.testing.assert_close(stats[key], (4 * uniform).expand(2))
    assert stats["prefix_query_fraction"].sum() == 0


def test_sink_and_recency_heads_are_distinguished_after_warmup():
    weights = torch.zeros(2, 512, 512)
    weights[0, :, 0] = 1
    weights[1] = torch.eye(512)
    stats = summarize_attention(weights)
    assert stats["first_mass"].tolist() == [1, 0]
    assert stats["prefix_query_fraction"].tolist() == [1, 0]
    assert stats["recent_mass"].tolist() == [0, 1]
    assert stats["peak_query_fraction"].tolist() == pytest.approx([1, 1 / 384])


def test_detection_uses_native_attention_and_resets_between_sequences():
    config = GPTNeoConfig(vocab_size=16, hidden_size=8, num_layers=1, num_heads=2,
                          intermediate_size=16, max_position_embeddings=32,
                          attention_types=[[["global"], 1]], attention_dropout=0.9)
    config._attn_implementation = "eager"
    model = GPTNeoForCausalLM(config)
    with torch.no_grad():
        model.transformer.h[0].attn.attention.q_proj.weight.zero_()
    ids = torch.randint(16, (3, 32))
    result = detect_sinks(model, ids, query_start=8, prefix_tokens=2)
    assert not model.training
    assert len(result["heads"]) == 2
    assert result["persistent_prefix_heads"] == 0
    for head in result["heads"]:
        assert head["per_sequence_first_mass"] == pytest.approx([result["uniform_first_mass"]] * 3)
        assert head["prefix_mass"] == pytest.approx(head["control_mass"])
        assert sum(head["position_profile"]) == pytest.approx(1)
