"""Stateful attention: step(q, k, v) maps [batch, heads, dim] to the same shape.

KV methods attend to retained history + the current token, then evict.
Linear attention keeps FP32 sums; explicit prefix sinks stay outside them.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttentionConfig:
    method: str = "full"
    sink_tokens: int = 1
    cache_size: int = 512
    heavy_hitter_size: int | None = None
    feature_map: str = "elu"
    sink_mode: str = "softmax"
    eps: float = 1e-6
    score_scale: float | None = None  # Default: 1/sqrt(d); GPT-Neo uses 1.


def append(cache, value, dim=-2):
    return value.clone() if cache is None else torch.cat((cache, value), dim=dim)


class Attention:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.seen_tokens = 0
        self.keys = self.values = self.scores = self.kv_state = self.key_state = None

    @property
    def cached_tokens(self):
        return 0 if self.keys is None else self.keys.shape[-2]

    @property
    def state_bytes(self):
        tensors = (self.keys, self.values, self.scores, self.kv_state, self.key_state)
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)

    def scale(self, dim):
        return dim ** -0.5 if self.config.score_scale is None else self.config.score_scale

    @torch.no_grad()
    def step(self, q, k, v, *, score_transform=None):
        linear = self.config.method == "linear"
        if not linear or self.seen_tokens < self.config.sink_tokens:
            self.keys = append(self.keys, k.unsqueeze(-2))
            self.values = append(self.values, v.unsqueeze(-2))
        output = self._linear(q, k, v) if linear else self._softmax(q, score_transform)
        self.seen_tokens += 1
        return output.to(v.dtype)

    def _softmax(self, q, score_transform):
        # StreamingLLM rotates raw cached keys at compact positions for scoring.
        query, keys = (q, self.keys) if score_transform is None else score_transform(q, self.keys)
        logits = torch.einsum("bhd,bhtd->bht", query.float(), keys.float()) * self.scale(q.shape[-1])
        weights = logits.softmax(-1)
        output = torch.einsum("bht,bhtd->bhd", weights, self.values.float())
        if self.config.method == "h2o":
            self.scores = append(self.scores, torch.zeros_like(weights[..., :1]), dim=-1) + weights
        self._evict()
        return output

    def _evict(self):
        config, length = self.config, self.cached_tokens
        if config.method == "full" or length <= config.cache_size:
            return
        positions = torch.arange(length, device=self.keys.device)
        if config.method == "h2o":
            heavy = config.cache_size // 2 if config.heavy_hitter_size is None else config.heavy_hitter_size
            recent = config.cache_size - heavy
            # Each head independently ranks older tokens by accumulated attention.
            older = self.scores[..., :length - recent].topk(heavy, dim=-1).indices.sort(-1).values
            newest = positions[length - recent:].expand(*self.keys.shape[:2], recent)
            keep = torch.cat((older, newest), dim=-1)
            self.scores = self.scores.gather(-1, keep)
        else:
            sinks = config.sink_tokens if config.method == "streaming" else 0
            recent = config.cache_size - sinks
            keep = torch.cat((positions[:sinks], positions[length - recent:]))
            keep = keep.expand(*self.keys.shape[:2], -1)
        indices = keep.unsqueeze(-1).expand(*keep.shape, self.keys.shape[-1])
        self.keys = self.keys.gather(-2, indices)
        self.values = self.values.gather(-2, indices)

    def _features(self, x):
        x = x.float() * self.scale(x.shape[-1]) ** 0.5
        if self.config.feature_map == "relu":
            return x.relu()
        # ELU(x)+1, avoiding cancellation at very negative x.
        return torch.where(x > 0, x + 1, x.clamp(max=0).exp())

    def _linear(self, q, k, v):
        if self.kv_state is None:
            self.kv_state = torch.zeros((*q.shape, v.shape[-1]), device=q.device, dtype=torch.float32)
            self.key_state = torch.zeros_like(q, dtype=torch.float32)
        if self.seen_tokens >= self.config.sink_tokens:
            phi_k = self._features(k)
            self.kv_state.add_(phi_k.unsqueeze(-1) * v.float().unsqueeze(-2))
            self.key_state.add_(phi_k)

        phi_q = self._features(q)
        numerator = torch.einsum("bhd,bhde->bhe", phi_q, self.kv_state)
        mass = (phi_q * self.key_state).sum(-1, keepdim=True).clamp_min(0)
        if self.keys is None:
            return numerator / mass.clamp_min(self.config.eps)
        if self.config.sink_mode == "kernel":
            weights = torch.einsum("bhd,bhtd->bht", phi_q, self._features(self.keys))
            numerator = numerator + torch.einsum("bht,bhtd->bhd", weights, self.values.float())
            return numerator / (mass + weights.sum(-1, keepdim=True)).clamp_min(self.config.eps)

        # Treat the linear aggregate as one extra entry with log(mass) as its logit.
        # One stable softmax jointly normalizes sinks and the aggregate.
        logits = torch.einsum("bhd,bhtd->bht", q.float(), self.keys.float()) * self.scale(q.shape[-1])
        weights = torch.cat((logits, mass.log()), dim=-1).softmax(-1)
        kernel_value = numerator / mass.clamp_min(torch.finfo(torch.float32).tiny)
        return (torch.einsum("bht,bhtd->bhd", weights[..., :-1], self.values.float())
                + weights[..., -1:] * kernel_value)
