import pytest
import torch
from transformers import GPTNeoConfig, GPTNeoForCausalLM, GPTNeoXConfig, GPTNeoXForCausalLM

from rttt.attention import AttentionConfig
from rttt.model import AttentionModelRunner, rotate


def token_logits(runner, ids):
    return torch.stack([runner.step(token) for token in ids.unbind(1)], dim=1)


def tiny_model(architecture="neo", local=False, parallel=True):
    torch.manual_seed(42)
    if architecture == "neo":
        config = GPTNeoConfig(
            vocab_size=31, hidden_size=16, num_layers=2, num_heads=2,
            intermediate_size=24, max_position_embeddings=32, window_size=4,
            attention_types=[[["global", "local"] if local else ["global"], 1 if local else 2]],
            resid_dropout=0, embed_dropout=0, attention_dropout=0,
        )
        config._attn_implementation = "eager"
        return GPTNeoForCausalLM(config).eval()
    config = GPTNeoXConfig(
        vocab_size=31, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=24, max_position_embeddings=32, rotary_pct=0.5,
        use_parallel_residual=parallel, hidden_dropout=0, attention_dropout=0,
    )
    config._attn_implementation = "eager"
    return GPTNeoXForCausalLM(config).eval()


@pytest.mark.parametrize("architecture,local,parallel", [
    ("neo", False, True), ("neo", True, True), ("neox", False, True), ("neox", False, False),
])
def test_full_logits_match_original_pretrained_architecture(architecture, local, parallel):
    model = tiny_model(architecture, local, parallel)
    ids = torch.tensor([[1, 5, 8, 2, 7, 6, 9, 3], [9, 3, 2, 5, 4, 1, 8, 7]])
    runner = AttentionModelRunner(model)
    with torch.no_grad():
        expected = model(ids).logits
    actual = token_logits(runner, ids)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    runner.reset()
    torch.testing.assert_close(runner.prefill(ids), actual[:, -1])
    assert runner.tokens_seen == ids.shape[1]


@pytest.mark.parametrize("method", ["window", "streaming", "h2o"])
@pytest.mark.parametrize("architecture", ["neo", "neox"])
def test_eviction_policies_match_full_before_budget_fills(architecture, method):
    model = tiny_model(architecture)
    runner = AttentionModelRunner(model, AttentionConfig(method=method, cache_size=16))
    ids = torch.tensor([[1, 5, 8, 2, 7]])
    with torch.no_grad():
        expected = model(ids).logits
    torch.testing.assert_close(token_logits(runner, ids), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("method", ["window", "streaming", "h2o", "linear"])
def test_neox_can_stream_past_training_length_with_bounded_state(method):
    runner = AttentionModelRunner(tiny_model("neox"), AttentionConfig(method=method, cache_size=4))
    sizes = []
    for i in range(70):
        assert torch.isfinite(runner.step(torch.tensor([i % 31]))).all()
        if i > 8:
            sizes.append(runner.state_bytes)
    assert len(set(sizes)) == 1
    assert runner.cached_tokens <= 4


def test_streaming_repositions_cached_raw_keys_after_eviction():
    model = tiny_model("neox")
    runner = AttentionModelRunner(model, AttentionConfig(method="streaming", cache_size=3))
    ids = torch.tensor([[1, 5, 8, 2, 7, 9]])
    runner.prefill(ids)
    # First-layer raw keys depend only on each token's embedding, so they can
    # be compared directly with fresh projections even after position shifts.
    block = model.gpt_neox.layers[0]
    retained = ids[:, [0, 4, 5]]
    with torch.no_grad():
        hidden = model.gpt_neox.embed_in(retained)
        qkv = block.attention.query_key_value(block.input_layernorm(hidden))
        expected = qkv.reshape(1, 3, 2, 24)[..., 8:16].permute(0, 2, 1, 3)
    torch.testing.assert_close(runner.policies[0].keys, expected)
    # At the next step keys use cache positions 0..3, regardless of true age.
    positioned = rotate(expected, torch.arange(3), 4, model.config.rotary_emb_base)
    assert not torch.allclose(positioned[:, :, -1], expected[:, :, -1])


def test_absolute_positions_fail_without_silent_wrap():
    runner = AttentionModelRunner(tiny_model(), AttentionConfig(method="linear"))
    runner.prefill(torch.ones(1, 32, dtype=torch.long))
    with pytest.raises(ValueError, match="position limit"):
        runner.step(torch.tensor([1]))
    assert runner.tokens_seen == 32


def test_local_attention_replacement_requires_global_checkpoint():
    with pytest.raises(ValueError, match="local attention layers"):
        AttentionModelRunner(tiny_model(local=True), AttentionConfig(method="linear"))


def test_switching_policy_resets_all_state():
    runner = AttentionModelRunner(tiny_model())
    runner.step(torch.tensor([1]))
    runner.set_attention(AttentionConfig(method="linear", sink_tokens=0))
    assert runner.tokens_seen == runner.state_bytes == runner.cached_tokens == 0
    assert all(policy.config.score_scale == 1.0 for policy in runner.policies)


def test_speed_benchmark_reports_repetitions_and_bounded_bytes():
    from rttt.performance import benchmark_speed

    runner = AttentionModelRunner(tiny_model(), AttentionConfig(method="h2o", cache_size=3))
    result = benchmark_speed(runner, [4, 8], decode_tokens=2, repeats=2)
    for point in result["results"]:
        assert len(point["samples"]) == 2
        assert point["median_decode_ms_per_token"] > 0
        for sample in point["samples"]:
            assert sample["prefill_state_bytes"] == sample["decode_state_bytes"] == runner.state_bytes
    with pytest.raises(ValueError, match="learned positions"):
        benchmark_speed(runner, [32], decode_tokens=1)
