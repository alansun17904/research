import math

import pytest
import torch

from rttt.attention import Attention, AttentionConfig


def sequence(length=12, batch=2, heads=3, dim=4):
    generator = torch.Generator().manual_seed(123)
    return tuple(torch.randn(length, batch, heads, dim, generator=generator) for _ in range(3))


def reference_softmax(q, keys, values):
    weights = torch.einsum("bhd,tbhd->bht", q, keys).div(math.sqrt(q.shape[-1])).softmax(-1)
    return torch.einsum("bht,tbhd->bhd", weights, values)


def features(x, name):
    x = x / x.shape[-1] ** 0.25
    return torch.nn.functional.elu(x) + 1 if name == "elu" else x.relu()


def reference_linear(q, keys, values, feature_map, sink_tokens=0, sink_mode="kernel"):
    weights = torch.einsum("bhd,tbhd->bht", features(q, feature_map), features(keys, feature_map))
    if sink_mode == "softmax":
        count = min(sink_tokens, keys.shape[0])
        weights[..., :count] = torch.einsum("bhd,tbhd->bht", q, keys[:count]).div(math.sqrt(q.shape[-1])).exp()
    return torch.einsum("bht,tbhd->bhd", weights, values) / weights.sum(-1, keepdim=True).clamp_min(1e-6)


def test_full_matches_dense_causal_attention_and_resets():
    q, k, v = sequence()
    policy = Attention(AttentionConfig(method="full"))
    for index in range(len(q)):
        actual = policy.step(q[index], k[index], v[index])
        torch.testing.assert_close(actual, reference_softmax(q[index], k[:index + 1], v[:index + 1]))
    assert policy.cached_tokens == len(q)
    assert policy.state_bytes > 0
    policy.reset()
    assert policy.cached_tokens == policy.state_bytes == policy.seen_tokens == 0
    torch.testing.assert_close(policy.step(q[0], k[0], v[0]), v[0])


@pytest.mark.parametrize("method,sinks", [("window", 0), ("streaming", 0), ("streaming", 2)])
def test_bounded_softmax_matches_retained_history_plus_current(method, sinks):
    q, k, v = sequence()
    policy = Attention(AttentionConfig(method=method, cache_size=4, sink_tokens=sinks))
    retained = []
    bounded_bytes = []
    for index in range(len(q)):
        attended = retained + [index]
        expected = reference_softmax(q[index], k[attended], v[attended])
        torch.testing.assert_close(policy.step(q[index], k[index], v[index]), expected)
        if len(attended) > 4:
            retained = attended[:sinks] + attended[-(4 - sinks):]
        else:
            retained = attended
        assert policy.cached_tokens == min(index + 1, 4)
        torch.testing.assert_close(policy.keys, k[retained].permute(1, 2, 0, 3))
        if index >= 4:
            bounded_bytes.append(policy.state_bytes)
    assert len(set(bounded_bytes)) == 1


@pytest.mark.parametrize("feature_map", ["elu", "relu"])
@pytest.mark.parametrize("sink_tokens", [0, 1, 3, 20])
@pytest.mark.parametrize("sink_mode", ["kernel", "softmax"])
def test_linear_matches_explicit_normalized_kernel_or_hybrid(feature_map, sink_tokens, sink_mode):
    q, k, v = sequence()
    policy = Attention(AttentionConfig(
        method="linear", sink_tokens=sink_tokens, feature_map=feature_map, sink_mode=sink_mode
    ))
    for index in range(len(q)):
        actual = policy.step(q[index], k[index], v[index])
        expected = reference_linear(q[index], k[:index + 1], v[:index + 1], feature_map, sink_tokens, sink_mode)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert policy.cached_tokens == min(sink_tokens, len(q))


def test_sink_tokens_are_excluded_from_linear_aggregate():
    q, k, v = sequence(length=3)
    policy = Attention(AttentionConfig(method="linear", sink_tokens=2))
    for index in range(2):
        policy.step(q[index], k[index], v[index])
    assert torch.count_nonzero(policy.kv_state).item() == 0
    assert torch.count_nonzero(policy.key_state).item() == 0
    policy.step(q[2], k[2], v[2])
    phi = features(k[2], "elu")
    torch.testing.assert_close(policy.key_state, phi)
    torch.testing.assert_close(policy.kv_state, phi.unsqueeze(-1) * v[2].unsqueeze(-2))


def test_linear_fixed_fp32_state_and_reset():
    q, k, v = sequence(length=30)
    policy = Attention(AttentionConfig(method="linear", sink_tokens=2))
    sizes = []
    for index in range(len(q)):
        output = policy.step(q[index].half(), k[index].half(), v[index].half())
        assert output.dtype == torch.float16
        if index >= 2:
            sizes.append(policy.state_bytes)
    assert len(set(sizes)) == 1
    assert policy.cached_tokens == 2
    assert policy.key_state.dtype == policy.kv_state.dtype == torch.float32
    policy.reset()
    assert policy.state_bytes == policy.cached_tokens == policy.seen_tokens == 0


def test_exact_softmax_sink_is_stable_for_large_logits_and_dominates():
    policy = Attention(AttentionConfig(method="linear", sink_tokens=1))
    positive = torch.full((1, 1, 4), 100.0)
    sink_value = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    policy.step(positive, positive, sink_value)
    actual = policy.step(positive, positive * -1, sink_value * -10)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, sink_value)


@pytest.mark.parametrize("key_sign", [-1, 1])
def test_relu_zero_mass_is_finite_with_and_without_sink(key_sign):
    negative = torch.full((1, 1, 4), -100.0)
    value = torch.full_like(negative, 3.0)
    for sink_tokens in (0, 1):
        policy = Attention(AttentionConfig(method="linear", sink_tokens=sink_tokens, feature_map="relu"))
        actual = policy.step(negative, negative * key_sign, value)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, value if sink_tokens else torch.zeros_like(value))


def test_h2o_cumulative_scores_and_eviction_are_per_head():
    policy = Attention(AttentionConfig(method="h2o", cache_size=2, heavy_hitter_size=1))
    # First head consistently chooses token 0; second chooses token 1.
    keys = [
        torch.tensor([[[10.0, 0.0], [0.0, 10.0]]]),
        torch.tensor([[[0.0, 10.0], [10.0, 0.0]]]),
        torch.zeros(1, 2, 2),
        torch.zeros(1, 2, 2),
    ]
    query = torch.tensor([[[10.0, 0.0], [10.0, 0.0]]])
    expected_scores = torch.zeros(1, 2, 0)
    retained = torch.zeros(1, 2, 0, dtype=torch.long)
    for index, key in enumerate(keys):
        attended = torch.cat((retained, torch.full((1, 2, 1), index, dtype=torch.long)), dim=-1)
        stacked_keys = torch.stack(keys[:index + 1], dim=-2)
        selected_keys = stacked_keys.gather(-2, attended.unsqueeze(-1).expand(1, 2, attended.shape[-1], 2))
        weights = (query.unsqueeze(-2) * selected_keys).sum(-1).div(math.sqrt(2)).softmax(-1)
        expected_scores = torch.cat((expected_scores, torch.zeros(1, 2, 1)), -1) + weights
        policy.step(query, key, torch.full_like(query, float(index)))
        if attended.shape[-1] > 2:
            top = expected_scores[..., :-1].argmax(-1, keepdim=True)
            keep = torch.cat((top, torch.full_like(top, attended.shape[-1] - 1)), -1)
            expected_scores = expected_scores.gather(-1, keep)
            retained = attended.gather(-1, keep)
        else:
            retained = attended
        torch.testing.assert_close(policy.scores, expected_scores)
        torch.testing.assert_close(policy.values, retained.unsqueeze(-1).expand_as(policy.values).float())
    assert retained.tolist() == [[[0, 3], [1, 3]]]
    assert policy.keys.shape[-2] == policy.values.shape[-2] == 2


def test_h2o_zero_heavy_hitters_matches_window():
    h2o = Attention(AttentionConfig(method="h2o", cache_size=3, heavy_hitter_size=0))
    window = Attention(AttentionConfig(method="window", cache_size=3))
    for q, k, v in zip(*sequence()):
        torch.testing.assert_close(h2o.step(q, k, v), window.step(q, k, v))


def test_score_transform_does_not_mutate_raw_streaming_cache():
    policy = Attention(AttentionConfig(method="streaming", cache_size=3, sink_tokens=1))
    q, k, v = sequence(length=5)
    retained = []
    for index in range(len(q)):
        attended = retained + [index]
        actual = policy.step(q[index], k[index], v[index], score_transform=lambda query, keys: (query * 2, keys * 3))
        expected = reference_softmax(q[index] * 2, k[attended] * 3, v[attended])
        torch.testing.assert_close(actual, expected)
        retained = attended if len(attended) <= 3 else attended[:1] + attended[-2:]
        torch.testing.assert_close(policy.keys, k[retained].permute(1, 2, 0, 3))


def test_policies_do_not_retain_autograd_graphs():
    for method in ("full", "window", "streaming", "h2o", "linear"):
        policy = Attention(AttentionConfig(method=method))
        tensor = torch.ones(1, 1, 4, requires_grad=True)
        output = policy.step(tensor, tensor, tensor)
        assert not output.requires_grad
        assert all(not t.requires_grad for t in vars(policy).values() if isinstance(t, torch.Tensor))
