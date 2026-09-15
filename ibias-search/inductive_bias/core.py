from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


POSITION_ENCODING = "rope_qk_base_10000"


@dataclass(frozen=True)
class ModelConfig:
    n: int = 16
    width: int = 16
    heads: int = 2
    layers: int = 2
    ff_width: int = 32


@dataclass(frozen=True)
class Budget:
    steps: int = 100
    batch_size: int = 128
    learning_rate: float = 0.2
    sampling_seed: int = 0


@dataclass(frozen=True)
class Thresholds:
    max_loss: float = 0.65
    min_accuracy: float = 0.6

    def accepts(self, metrics):
        return all(
            m["loss"] <= self.max_loss and m["accuracy"] >= self.min_accuracy
            for m in metrics.values()
        )


class RotaryEmbedding(nn.Module):
    """Fixed RoPE rotations of adjacent feature pairs within each head."""

    def __init__(self, head_dim, max_length):
        super().__init__()
        frequencies = 10000.0 ** (
            -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )
        angles = torch.arange(max_length, dtype=torch.float32)[:, None] * frequencies
        self.register_buffer("cos", angles.cos())
        self.register_buffer("sin", angles.sin())

    def forward(self, x):
        length = x.shape[-2]
        cos, sin = self.cos[:length].to(x.dtype), self.sin[:length].to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        ).flatten(-2)


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.width // config.heads
        self.norm1 = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.out_proj = nn.Linear(config.width, config.width)
        self.norm2 = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, config.ff_width),
            nn.GELU(),
            nn.Linear(config.ff_width, config.width),
        )
        self.rope = RotaryEmbedding(self.head_dim, config.n - 1)

    def forward(self, x):
        hidden = self.norm1(x)
        batch, length, width = hidden.shape
        q, k, v = (
            tensor.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
            for tensor in self.qkv(hidden).chunk(3, dim=-1)
        )
        attended = F.scaled_dot_product_attention(
            self.rope(q), self.rope(k), v, dropout_p=0.0, is_causal=True
        )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        x = x + self.out_proj(attended)
        return x + self.mlp(self.norm2(x))


class BinaryTransformer(nn.Module):
    """Decoder-only next-token model with a frozen random core and Q/K RoPE.

    Args:
        core: "transformer" or "identity". If "identity", then we are training a bigram model.
    """

    def __init__(self, config, seed, core="transformer"):
        super().__init__()
        self.config = config
        # Construct on CPU without changing the caller's random state. E/U are
        # initialized first so the identity control starts from identical E/U.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.embedding = nn.Embedding(2, config.width)
            self.unembedding = nn.Linear(config.width, 2, bias=False)
            self.blocks = nn.ModuleList(
                DecoderBlock(config)
                for _ in range(config.layers if core == "transformer" else 0)
            )
        self.blocks.requires_grad_(False)

    def forward(self, tokens):
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        return self.unembedding(x)


def encode(sequences, device="cpu"):
    return torch.tensor(
        [[int(bit) for bit in s] for s in sequences], dtype=torch.long, device=device
    )


def adapt(initial, sequences, budget):
    """Every call resets E/U, the frozen core, optimizer and minibatch RNG."""
    model = deepcopy(initial).train()
    data = encode(sequences, model.embedding.weight.device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=budget.learning_rate,
        weight_decay=0.01,
    )
    generator = torch.Generator().manual_seed(budget.sampling_seed)
    for _ in range(budget.steps):
        indices = torch.randint(
            len(data), (budget.batch_size,), generator=generator
        ).to(data.device)
        batch = data[indices]
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 2), batch[:, 1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval()


def score(model, sequences, batch_size=256):
    """Get accuracy and CE loss for each sequence in `sequences`."""
    model.eval()
    metrics = {}
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            data = encode(chunk, model.embedding.weight.device)
            logits = model(data[:, :-1])
            losses = F.cross_entropy(
                logits.transpose(1, 2), data[:, 1:], reduction="none"
            ).mean(1)
            accuracies = (logits.argmax(-1) == data[:, 1:]).float().mean(1)
            metrics.update(
                {
                    s: {"loss": loss, "accuracy": acc}
                    for s, loss, acc in zip(chunk, losses.tolist(), accuracies.tolist())
                }
            )
    return metrics


def summarize(metrics):
    rows = list(metrics.values())
    return {
        "mean_loss": sum(m["loss"] for m in rows) / len(rows),
        "worst_loss": max(m["loss"] for m in rows),
        "mean_accuracy": sum(m["accuracy"] for m in rows) / len(rows),
        "worst_accuracy": min(m["accuracy"] for m in rows),
    }


def simple_baseline(support, test, order):
    """Laplace-smoothed MLEs use the same shifted support tokens as adaptation."""
    counts = torch.ones(2 if order else 1, 2, dtype=torch.float64)
    for sequence in support:
        for previous, target in zip(sequence[:-1], sequence[1:]):
            counts[int(previous) if order else 0, int(target)] += 1
    probabilities = counts / counts.sum(1, keepdim=True)
    metrics = {}
    for sequence in test:
        losses, correct = [], []
        for previous, target in zip(sequence[:-1], sequence[1:]):
            row = probabilities[int(previous) if order else 0]
            losses.append(-math.log(float(row[int(target)])))
            correct.append(int(row.argmax()) == int(target))
        metrics[sequence] = {
            "loss": sum(losses) / len(losses),
            "accuracy": sum(correct) / len(correct),
        }
    return metrics
