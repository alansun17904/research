from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from datasets import Dataset
from scipy import sparse
from sklearn.utils.extmath import randomized_svd
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.pytorch_utils import Conv1D
C4_FILTER_SMALL_ARROW_PATH = (
    "/Users/alansun/.cache/huggingface/datasets/"
    "datablations___c4-filter-small/default/0.0.0/"
    "f975fa88ccfea268f412be33ed62cd3644d9d140/c4-filter-small-train.arrow"
)


@dataclass
class RunConfig:
    corpus_path: str | None
    dataset_arrow_path: str | None
    tokenizer_name: str
    config_name: str
    tiny_n_layer: int
    tiny_n_embd: int
    init_scheme: str
    top_k_vocab: int
    max_corpus_tokens: int
    max_documents: int
    block_size: int
    batch_size: int
    train_steps: int
    eval_batches: int
    learning_rate: float
    weight_decay: float
    bigram_learning_rate: float
    bigram_train_steps: int
    bigram_batch_size: int
    bigram_backbone_init: str
    adapter_strategy: str
    seed: int
    device: str
    freeze_svd_token_tables: bool


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(name: str) -> GPT2TokenizerFast:
    tokenizer = GPT2TokenizerFast.from_pretrained(name, local_files_only=True)
    tokenizer.model_max_length = int(1e9)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_arrow_text_subset(dataset_arrow_path: str, max_documents: int) -> str:
    dataset = Dataset.from_file(dataset_arrow_path)
    documents = []
    for idx in range(min(max_documents, len(dataset))):
        text = dataset[idx]["text"]
        if isinstance(text, str) and text.strip():
            documents.append(text.strip())
    if not documents:
        raise ValueError(f"No usable text found in dataset at {dataset_arrow_path}.")
    return "\n\n".join(documents)


def load_corpus_text(
    corpus_path: str | None,
    dataset_arrow_path: str | None,
    max_documents: int,
) -> str:
    if dataset_arrow_path is not None:
        return load_arrow_text_subset(dataset_arrow_path, max_documents)
    if corpus_path is not None:
        return Path(corpus_path).read_text()
    toy_path = Path(__file__).with_name("toy_corpus.txt")
    return toy_path.read_text()


def encode_corpus(
    tokenizer: GPT2TokenizerFast,
    text: str,
    max_tokens: int,
) -> list[int]:
    token_ids = tokenizer.encode(text)
    if len(token_ids) < 3:
        raise ValueError("Corpus is too small after tokenization.")
    return token_ids[:max_tokens]


def train_val_split(token_ids: list[int], val_fraction: float = 0.1) -> tuple[Tensor, Tensor]:
    split = max(2, int(len(token_ids) * (1.0 - val_fraction)))
    split = min(split, len(token_ids) - 1)
    train_ids = torch.tensor(token_ids[:split], dtype=torch.long)
    val_ids = torch.tensor(token_ids[split - 1 :], dtype=torch.long)
    return train_ids, val_ids


def sample_batch(data: Tensor, batch_size: int, block_size: int, device: str) -> tuple[Tensor, Tensor]:
    max_start = data.size(0) - block_size - 1
    if max_start <= 0:
        raise ValueError("Not enough tokens for the requested block size.")
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([data[s : s + block_size] for s in starts])
    y = torch.stack([data[s + 1 : s + block_size + 1] for s in starts])
    return x.to(device), y.to(device)


def sample_bigram_batch(data: Tensor, batch_size: int, device: str) -> tuple[Tensor, Tensor]:
    max_start = data.size(0) - 2
    if max_start < 0:
        raise ValueError("Not enough tokens for bigram training.")
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = data[starts]
    y = data[starts + 1]
    return x.to(device), y.to(device)


def build_model_config(
    config_name: str,
    vocab_size: int,
    block_size: int,
    tiny_n_layer: int | None = None,
    tiny_n_embd: int | None = None,
) -> GPT2Config:
    if config_name == "gpt2":
        default_gpt2 = GPT2Config()
        default_positions = default_gpt2.n_positions
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=max(block_size, default_positions),
            n_ctx=max(block_size, default_positions),
            tie_word_embeddings=False,
        )
        return config
    if config_name == "tiny":
        return GPT2Config(
            vocab_size=vocab_size,
            n_positions=block_size,
            n_ctx=block_size,
            n_embd=128 if tiny_n_embd is None else tiny_n_embd,
            n_layer=4 if tiny_n_layer is None else tiny_n_layer,
            n_head=4,
            tie_word_embeddings=False,
        )
    raise ValueError(f"Unsupported config_name={config_name!r}")


def select_active_vocab(token_ids: Iterable[int], top_k_vocab: int) -> list[int]:
    counts = Counter(token_ids)
    return [token_id for token_id, _ in counts.most_common(top_k_vocab)]


def compute_bigram_counts(
    token_ids: list[int],
    active_token_ids: list[int],
) -> tuple[Tensor, dict[int, int]]:
    token_to_index = {token_id: idx for idx, token_id in enumerate(active_token_ids)}
    active_vocab_size = len(active_token_ids)
    counts = torch.zeros((active_vocab_size, active_vocab_size), dtype=torch.float64)
    for left, right in zip(token_ids[:-1], token_ids[1:]):
        left_index = token_to_index.get(left)
        right_index = token_to_index.get(right)
        if left_index is None or right_index is None:
            continue
        counts[left_index, right_index] += 1.0
    return counts, token_to_index


def compute_log_bigram_matrix(
    token_ids: list[int],
    active_token_ids: list[int],
    smoothing: float = 1e-3,
) -> tuple[Tensor, dict[int, int]]:
    counts, token_to_index = compute_bigram_counts(token_ids, active_token_ids)
    counts = counts + smoothing
    row_sums = counts.sum(dim=1, keepdim=True)
    log_probs = torch.log(counts / row_sums)
    return log_probs.to(torch.float32), token_to_index


def normalize_probability_rows(probabilities: Tensor, fallback: Tensor) -> Tensor:
    normalized = probabilities.clone()
    row_sums = normalized.sum(dim=1, keepdim=True)
    valid_rows = row_sums.squeeze(-1) > 0
    if valid_rows.any():
        normalized[valid_rows] = normalized[valid_rows] / row_sums[valid_rows]
    if (~valid_rows).any():
        normalized[~valid_rows] = fallback.unsqueeze(0)
    return normalized


def safe_log_probability_matrix(probabilities: Tensor, floor: float = 1e-12) -> Tensor:
    probabilities = probabilities.clamp_min(floor)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(floor)
    return probabilities.log().to(torch.float32)


def compute_unigram_backoff_distribution(counts: Tensor, alpha: float = 1e-3) -> Tensor:
    target_totals = counts.sum(dim=0) + alpha
    return target_totals / target_totals.sum()


def score_bigram_log_probs(
    token_ids: list[int],
    token_to_index: dict[int, int],
    log_probs: Tensor,
) -> dict[str, float]:
    nll_total = 0.0
    covered_pairs = 0
    for left, right in zip(token_ids[:-1], token_ids[1:]):
        left_index = token_to_index.get(left)
        right_index = token_to_index.get(right)
        if left_index is None or right_index is None:
            continue
        nll_total -= float(log_probs[left_index, right_index].item())
        covered_pairs += 1
    mean_nll = nll_total / covered_pairs if covered_pairs > 0 else float("inf")
    perplexity = math.exp(mean_nll) if math.isfinite(mean_nll) else float("inf")
    return {
        "covered_pairs": float(covered_pairs),
        "mean_nll": mean_nll,
        "perplexity": perplexity,
    }


def build_additive_bigram_log_probs(counts: Tensor, alpha: float) -> Tensor:
    probabilities = counts + alpha
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return safe_log_probability_matrix(probabilities)


def estimate_good_turing_adjusted_counts(counts: Tensor, max_count: int = 5) -> dict[int, float]:
    flat_counts = counts.to(torch.int64).reshape(-1)
    frequency_of_frequency = Counter(flat_counts.tolist())
    adjusted_counts: dict[int, float] = {}
    for count in range(1, max_count + 1):
        current_bucket = frequency_of_frequency.get(count, 0)
        next_bucket = frequency_of_frequency.get(count + 1, 0)
        if current_bucket > 0 and next_bucket > 0:
            adjusted_counts[count] = (count + 1) * next_bucket / current_bucket
        else:
            adjusted_counts[count] = float(count)
    return adjusted_counts


def build_good_turing_bigram_log_probs(counts: Tensor, max_count: int = 5) -> tuple[Tensor, dict[str, float]]:
    adjusted_table = estimate_good_turing_adjusted_counts(counts, max_count=max_count)
    unigram_backoff = compute_unigram_backoff_distribution(counts)
    probabilities = torch.zeros_like(counts)
    positive_counts = counts > 0
    for count, adjusted_count in adjusted_table.items():
        probabilities = torch.where(counts == count, torch.full_like(probabilities, adjusted_count), probabilities)
    probabilities = torch.where(positive_counts & (counts > max_count), counts, probabilities)

    row_totals = counts.sum(dim=1, keepdim=True)
    seen_adjusted_mass = probabilities.sum(dim=1, keepdim=True)
    unseen_mask = ~positive_counts
    unseen_backoff = unseen_mask * unigram_backoff.unsqueeze(0)
    unseen_backoff_mass = unseen_backoff.sum(dim=1, keepdim=True)
    leftover_mass = (row_totals - seen_adjusted_mass).clamp_min(0.0)
    probabilities = probabilities + torch.where(
        unseen_backoff_mass > 0,
        unseen_backoff * (leftover_mass / unseen_backoff_mass.clamp_min(1e-12)),
        torch.zeros_like(probabilities),
    )
    probabilities = normalize_probability_rows(probabilities, unigram_backoff)
    return safe_log_probability_matrix(probabilities), {
        f"adjusted_count_{count}": value for count, value in adjusted_table.items()
    }


def estimate_absolute_discount(counts: Tensor, fallback: float = 0.75) -> float:
    flat_counts = counts.to(torch.int64).reshape(-1)
    frequency_of_frequency = Counter(flat_counts.tolist())
    n1 = frequency_of_frequency.get(1, 0)
    n2 = frequency_of_frequency.get(2, 0)
    if n1 == 0:
        return fallback
    if n2 == 0:
        return min(fallback, 0.99)
    return float(max(0.1, min(0.99, n1 / (n1 + 2 * n2))))


def build_katz_backoff_log_probs(counts: Tensor) -> tuple[Tensor, dict[str, float]]:
    discount = estimate_absolute_discount(counts)
    unigram_backoff = compute_unigram_backoff_distribution(counts)
    row_totals = counts.sum(dim=1, keepdim=True)
    seen_mask = counts > 0
    discounted = torch.where(seen_mask, (counts - discount).clamp_min(0.0), torch.zeros_like(counts))
    seen_probs = discounted / row_totals.clamp_min(1e-12)
    leftover_mass = (1.0 - seen_probs.sum(dim=1, keepdim=True)).clamp_min(0.0)
    unseen_backoff = (~seen_mask) * unigram_backoff.unsqueeze(0)
    unseen_backoff_mass = unseen_backoff.sum(dim=1, keepdim=True)
    probabilities = seen_probs + torch.where(
        unseen_backoff_mass > 0,
        unseen_backoff * (leftover_mass / unseen_backoff_mass.clamp_min(1e-12)),
        torch.zeros_like(seen_probs),
    )
    probabilities = normalize_probability_rows(probabilities, unigram_backoff)
    return safe_log_probability_matrix(probabilities), {"discount": discount}


def estimate_modified_kneser_ney_discounts(counts: Tensor) -> dict[str, float]:
    flat_counts = counts.to(torch.int64).reshape(-1)
    frequency_of_frequency = Counter(flat_counts.tolist())
    n1 = frequency_of_frequency.get(1, 0)
    n2 = frequency_of_frequency.get(2, 0)
    n3 = frequency_of_frequency.get(3, 0)
    n4 = frequency_of_frequency.get(4, 0)
    if n1 == 0 or n2 == 0:
        return {"d1": 0.75, "d2": 0.75, "d3p": 0.75}
    y = n1 / (n1 + 2 * n2)
    d1 = 1.0 - 2.0 * y * (n2 / max(n1, 1))
    d2 = 2.0 - 3.0 * y * (n3 / max(n2, 1))
    d3p = 3.0 - 4.0 * y * (n4 / max(n3, 1)) if n3 > 0 else 0.75
    return {
        "d1": float(min(max(d1, 0.1), 0.99)),
        "d2": float(min(max(d2, 0.1), 1.99)),
        "d3p": float(min(max(d3p, 0.1), 2.99)),
    }


def build_modified_kneser_ney_log_probs(counts: Tensor) -> tuple[Tensor, dict[str, float]]:
    discounts = estimate_modified_kneser_ney_discounts(counts)
    row_totals = counts.sum(dim=1, keepdim=True)
    continuation_counts = (counts > 0).sum(dim=0).to(torch.float64)
    continuation_distribution = continuation_counts / continuation_counts.sum().clamp_min(1.0)

    discounted_counts = torch.zeros_like(counts)
    discounted_counts = torch.where(counts == 1, (counts - discounts["d1"]).clamp_min(0.0), discounted_counts)
    discounted_counts = torch.where(counts == 2, (counts - discounts["d2"]).clamp_min(0.0), discounted_counts)
    discounted_counts = torch.where(counts >= 3, (counts - discounts["d3p"]).clamp_min(0.0), discounted_counts)

    n1_row = (counts == 1).sum(dim=1, keepdim=True).to(torch.float64)
    n2_row = (counts == 2).sum(dim=1, keepdim=True).to(torch.float64)
    n3p_row = (counts >= 3).sum(dim=1, keepdim=True).to(torch.float64)
    lambda_row = (
        discounts["d1"] * n1_row
        + discounts["d2"] * n2_row
        + discounts["d3p"] * n3p_row
    ) / row_totals.clamp_min(1e-12)

    probabilities = discounted_counts / row_totals.clamp_min(1e-12)
    probabilities = probabilities + lambda_row * continuation_distribution.unsqueeze(0)
    probabilities = normalize_probability_rows(probabilities, continuation_distribution)
    return safe_log_probability_matrix(probabilities), discounts


def build_smart_bigram_log_probs(
    train_token_ids: list[int],
    val_token_ids: list[int],
    active_token_ids: list[int],
) -> tuple[Tensor, dict[int, int], dict[str, object]]:
    counts, token_to_index = compute_bigram_counts(train_token_ids, active_token_ids)
    candidates: list[tuple[str, Tensor, dict[str, float]]] = []
    for alpha in (1e-3, 1e-2, 1e-1, 1.0):
        candidates.append(
            (
                f"additive_alpha_{alpha:g}",
                build_additive_bigram_log_probs(counts, alpha=alpha),
                {"alpha": alpha},
            )
        )
    good_turing_log_probs, good_turing_meta = build_good_turing_bigram_log_probs(counts)
    katz_log_probs, katz_meta = build_katz_backoff_log_probs(counts)
    kneser_ney_log_probs, kneser_ney_meta = build_modified_kneser_ney_log_probs(counts)
    candidates.extend(
        [
            ("good_turing", good_turing_log_probs, good_turing_meta),
            ("katz_backoff", katz_log_probs, katz_meta),
            ("modified_kneser_ney", kneser_ney_log_probs, kneser_ney_meta),
        ]
    )

    scored_candidates = []
    for name, log_probs, metadata in candidates:
        score = score_bigram_log_probs(val_token_ids, token_to_index, log_probs)
        scored_candidates.append(
            {
                "name": name,
                "log_probs": log_probs,
                "metadata": metadata,
                "score": score,
            }
        )
    best_candidate = min(scored_candidates, key=lambda candidate: candidate["score"]["mean_nll"])
    candidate_metrics = [
        {
            "name": candidate["name"],
            **candidate["metadata"],
            **candidate["score"],
        }
        for candidate in scored_candidates
    ]
    return best_candidate["log_probs"], token_to_index, {
        "selected_bigram_model": best_candidate["name"],
        "selected_bigram_model_metadata": best_candidate["metadata"],
        "bigram_model_candidates": candidate_metrics,
    }


def compute_streaming_sparse_log_bigram_matrix(
    token_ids: list[int],
    vocab_size: int,
) -> sparse.csr_matrix:
    pair_counts = Counter(zip(token_ids[:-1], token_ids[1:]))
    row_totals = Counter(token_ids[:-1])
    rows = []
    cols = []
    values = []
    for (left, right), count in pair_counts.items():
        rows.append(left)
        cols.append(right)
        values.append(math.log(count / row_totals[left]))
    matrix = sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(vocab_size, vocab_size),
        dtype=float,
    )
    return matrix.tocsr()


def svd_factorize_log_bigrams(
    log_bigram_matrix: Tensor,
    d_model: int,
) -> tuple[Tensor, Tensor]:
    u, singular_values, vh = torch.linalg.svd(log_bigram_matrix, full_matrices=False)
    rank = min(d_model, singular_values.numel())
    sqrt_s = singular_values[:rank].clamp_min(0.0).sqrt()
    embedding = u[:, :rank] * sqrt_s.unsqueeze(0)
    unembedding = vh[:rank, :].transpose(0, 1) * sqrt_s.unsqueeze(0)
    if rank < d_model:
        embedding = F.pad(embedding, (0, d_model - rank))
        unembedding = F.pad(unembedding, (0, d_model - rank))
    return embedding, unembedding


def sparse_factorize_log_bigrams(
    log_bigram_matrix: sparse.csr_matrix,
    d_model: int,
    n_iter: int = 5,
    random_state: int = 0,
) -> tuple[Tensor, Tensor]:
    u, singular_values, vh = randomized_svd(
        log_bigram_matrix,
        n_components=d_model,
        n_iter=n_iter,
        random_state=random_state,
    )
    sqrt_s = torch.from_numpy(singular_values).to(torch.float32).clamp_min(0.0).sqrt()
    embedding = torch.from_numpy(u).to(torch.float32) * sqrt_s.unsqueeze(0)
    unembedding = torch.from_numpy(vh.T).to(torch.float32) * sqrt_s.unsqueeze(0)
    return embedding, unembedding


def gaussian_fill_missing_token_rows(
    embedding: Tensor,
    unembedding: Tensor,
    token_ids: list[int],
    vocab_size: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    source_seen = torch.zeros(vocab_size, dtype=torch.bool)
    target_seen = torch.zeros(vocab_size, dtype=torch.bool)
    if len(token_ids) >= 2:
        source_seen[torch.tensor(token_ids[:-1], dtype=torch.long).unique()] = True
        target_seen[torch.tensor(token_ids[1:], dtype=torch.long).unique()] = True

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    filled_embedding = embedding.clone()
    filled_unembedding = unembedding.clone()

    if source_seen.any():
        source_rows = filled_embedding[source_seen]
        source_std = source_rows.std(dim=0, unbiased=False).clamp_min(1e-6)
        missing_sources = (~source_seen).nonzero(as_tuple=False).squeeze(-1)
        if missing_sources.numel() > 0:
            noise = torch.randn(
                (missing_sources.numel(), filled_embedding.size(1)),
                generator=rng,
                dtype=filled_embedding.dtype,
            )
            filled_embedding[missing_sources] = noise * source_std.unsqueeze(0)

    if target_seen.any():
        target_rows = filled_unembedding[target_seen]
        target_std = target_rows.std(dim=0, unbiased=False).clamp_min(1e-6)
        missing_targets = (~target_seen).nonzero(as_tuple=False).squeeze(-1)
        if missing_targets.numel() > 0:
            noise = torch.randn(
                (missing_targets.numel(), filled_unembedding.size(1)),
                generator=rng,
                dtype=filled_unembedding.dtype,
            )
            filled_unembedding[missing_targets] = noise * target_std.unsqueeze(0)

    return filled_embedding, filled_unembedding


class FactorizedBigramModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.v = nn.Embedding(vocab_size, d_model)
        self.u = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden = self.v(input_ids)
        return self.u(hidden)


class RotationScaleAdapter(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.log_scale = nn.Parameter(torch.zeros(d_model))
        self.theta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor) -> Tensor:
        x_scaled = x * self.log_scale.exp()
        return self.apply_rotation(x_scaled) - x

    def apply_rotation(self, x: Tensor) -> Tensor:
        rotated = x
        for index in range(self.d_model):
            left = index
            right = (index + 1) % self.d_model
            if left == right:
                continue
            angle = self.theta[index]
            cos_angle = torch.cos(angle)
            sin_angle = torch.sin(angle)
            left_values = rotated[..., left]
            right_values = rotated[..., right]
            updated_left = cos_angle * left_values - sin_angle * right_values
            updated_right = sin_angle * left_values + cos_angle * right_values
            rotated = rotated.clone()
            rotated[..., left] = updated_left
            rotated[..., right] = updated_right
        return rotated


class RotationOnlyAdapter(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.theta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor) -> Tensor:
        rotated = x
        for index in range(self.d_model):
            left = index
            right = (index + 1) % self.d_model
            if left == right:
                continue
            angle = self.theta[index]
            cos_angle = torch.cos(angle)
            sin_angle = torch.sin(angle)
            left_values = rotated[..., left]
            right_values = rotated[..., right]
            updated_left = cos_angle * left_values - sin_angle * right_values
            updated_right = sin_angle * left_values + cos_angle * right_values
            rotated = rotated.clone()
            rotated[..., left] = updated_left
            rotated[..., right] = updated_right
        return rotated - x


class ResidualMLPAdapter(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model, bias=False)
        self.fc2 = nn.Linear(d_model, d_model, bias=False)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        hidden = x + self.activation(self.fc1(x))
        output = hidden + self.activation(self.fc2(hidden))
        return output - x


class LoRAConv1D(nn.Module):
    def __init__(
        self,
        base_layer: Conv1D,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        in_features, out_features = base_layer.weight.shape
        self.lora_a = nn.Parameter(torch.empty(in_features, rank))
        self.lora_b = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.normal_(self.lora_a, mean=0.0, std=0.02)
        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        base_output = self.base_layer(x)
        lora_output = self.dropout(x) @ self.lora_a @ self.lora_b
        return base_output + lora_output * self.scaling


class BigramInitGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config: GPT2Config, adapter_strategy: str = "linear") -> None:
        super().__init__(config)
        self.adapter_strategy = adapter_strategy
        if adapter_strategy == "linear":
            self.input_projection = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.output_projection = nn.Linear(config.n_embd, config.n_embd, bias=False)
        elif adapter_strategy == "residual_mlp":
            self.input_projection = ResidualMLPAdapter(config.n_embd)
            self.output_projection = ResidualMLPAdapter(config.n_embd)
        elif adapter_strategy == "rotation_scale":
            self.input_projection = RotationScaleAdapter(config.n_embd)
            self.output_projection = RotationScaleAdapter(config.n_embd)
        elif adapter_strategy == "rotation_only":
            self.input_projection = RotationOnlyAdapter(config.n_embd)
            self.output_projection = RotationOnlyAdapter(config.n_embd)
        else:
            raise ValueError(f"Unsupported adapter_strategy={adapter_strategy!r}")
        self.reset_bigram_projections()

    def reset_bigram_projections(self) -> None:
        with torch.no_grad():
            if self.adapter_strategy == "linear":
                self.input_projection.weight.zero_()
                self.output_projection.weight.zero_()
            elif self.adapter_strategy == "residual_mlp":
                self.input_projection.fc1.weight.zero_()
                self.input_projection.fc2.weight.zero_()
                self.output_projection.fc1.weight.zero_()
                self.output_projection.fc2.weight.zero_()
            elif self.adapter_strategy == "rotation_scale":
                self.input_projection.log_scale.zero_()
                self.input_projection.theta.zero_()
                self.output_projection.log_scale.zero_()
                self.output_projection.theta.zero_()
            elif self.adapter_strategy == "rotation_only":
                self.input_projection.theta.zero_()
                self.output_projection.theta.zero_()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values=None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | CausalLMOutputWithCrossAttentions:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.transformer.wte(input_ids)
            input_ids = None
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds + self.input_projection(inputs_embeds)

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        hidden_states = hidden_states + self.output_projection(hidden_states)
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


def set_module_by_name(root_module: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent = root_module
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def apply_lora_to_gpt2(
    model: GPT2LMHeadModel,
    rank: int,
    alpha: float,
    dropout: float,
    target_module_suffixes: tuple[str, ...] = (
        "attn.c_attn",
        "attn.c_proj",
        "mlp.c_fc",
        "mlp.c_proj",
    ),
) -> list[str]:
    replaced_modules: list[str] = []
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, Conv1D):
            continue
        if not any(module_name.endswith(suffix) for suffix in target_module_suffixes):
            continue
        set_module_by_name(
            model,
            module_name,
            LoRAConv1D(
                base_layer=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            ),
        )
        replaced_modules.append(module_name)
    return replaced_modules


def freeze_all_parameters_except_lora(model: GPT2LMHeadModel) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRAConv1D):
            module.lora_a.requires_grad_(True)
            module.lora_b.requires_grad_(True)


def learn_factorized_bigram_tables(
    train_data: Tensor,
    vocab_size: int,
    d_model: int,
    batch_size: int,
    train_steps: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
    seed: int,
) -> tuple[Tensor, Tensor, dict[str, float | list[float]]]:
    set_seed(seed)
    model = FactorizedBigramModel(vocab_size=vocab_size, d_model=d_model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    step_losses = []
    model.train()
    for _ in range(train_steps):
        x, y = sample_bigram_batch(train_data, batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        step_losses.append(loss.item())
    with torch.no_grad():
        embedding = model.v.weight.detach().cpu()
        unembedding = model.u.weight.detach().cpu()
    metrics = {
        "step_losses": step_losses,
        "last_step_loss": step_losses[-1],
        "mean_step_loss": sum(step_losses) / len(step_losses),
    }
    return embedding, unembedding, metrics


def initialize_model_from_svd(
    model: GPT2LMHeadModel,
    active_token_ids: list[int],
    embedding_vectors: Tensor,
    unembedding_vectors: Tensor,
) -> None:
    with torch.no_grad():
        for token_id, embedding_vec, unembedding_vec in zip(
            active_token_ids,
            embedding_vectors,
            unembedding_vectors,
        ):
            model.transformer.wte.weight[token_id].copy_(embedding_vec)
            model.lm_head.weight[token_id].copy_(unembedding_vec)


def initialize_model_from_full_vocab_factors(
    model: GPT2LMHeadModel,
    embedding_vectors: Tensor,
    unembedding_vectors: Tensor,
) -> None:
    with torch.no_grad():
        model.transformer.wte.weight.copy_(embedding_vectors)
        model.lm_head.weight.copy_(unembedding_vectors)


def initialize_backbone_as_identity(model: GPT2LMHeadModel) -> None:
    with torch.no_grad():
        model.transformer.wpe.weight.zero_()
        for block in model.transformer.h:
            block.attn.c_attn.weight.zero_()
            block.attn.c_attn.bias.zero_()
            block.attn.c_proj.weight.zero_()
            block.attn.c_proj.bias.zero_()
            block.mlp.c_fc.weight.zero_()
            block.mlp.c_fc.bias.zero_()
            block.mlp.c_proj.weight.zero_()
            block.mlp.c_proj.bias.zero_()


def evaluate_loss(
    model: GPT2LMHeadModel,
    data: Tensor,
    batch_size: int,
    block_size: int,
    eval_batches: int,
    device: str,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_batches):
            x, y = sample_batch(data, batch_size, block_size, device)
            outputs = model(input_ids=x, labels=y)
            losses.append(outputs.loss.item())
    return sum(losses) / len(losses)


def freeze_token_tables(model: GPT2LMHeadModel) -> None:
    model.transformer.wte.weight.requires_grad_(False)
    model.lm_head.weight.requires_grad_(False)


def train_model(
    model: GPT2LMHeadModel,
    train_data: Tensor,
    val_data: Tensor,
    batch_size: int,
    block_size: int,
    train_steps: int,
    eval_batches: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
    freeze_token_table_params: bool = False,
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
) -> dict[str, float | list[float]]:
    lora_modules: list[str] = []
    if use_lora:
        lora_modules = apply_lora_to_gpt2(
            model,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        freeze_all_parameters_except_lora(model)
    elif freeze_token_table_params:
        freeze_token_tables(model)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    initial_train_loss = evaluate_loss(model, train_data, batch_size, block_size, eval_batches, device)
    initial_val_loss = evaluate_loss(model, val_data, batch_size, block_size, eval_batches, device)
    step_losses = []
    model.train()
    for _ in range(train_steps):
        x, y = sample_batch(train_data, batch_size, block_size, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids=x, labels=y)
        outputs.loss.backward()
        optimizer.step()
        step_losses.append(outputs.loss.item())
    final_train_loss = evaluate_loss(model, train_data, batch_size, block_size, eval_batches, device)
    final_val_loss = evaluate_loss(model, val_data, batch_size, block_size, eval_batches, device)
    return {
        "initial_train_loss": initial_train_loss,
        "initial_val_loss": initial_val_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "step_losses": step_losses,
        "last_step_loss": step_losses[-1],
        "mean_step_loss": sum(step_losses) / len(step_losses),
        "perplexity_initial_val": math.exp(initial_val_loss),
        "perplexity_final_val": math.exp(final_val_loss),
        "frozen_token_tables": freeze_token_table_params or use_lora,
        "used_lora": use_lora,
        "lora_rank": lora_rank if use_lora else 0,
        "lora_alpha": lora_alpha if use_lora else 0.0,
        "lora_dropout": lora_dropout if use_lora else 0.0,
        "lora_modules": lora_modules,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def estimate_steps_for_one_epoch(
    train_token_count: int,
    batch_size: int,
    block_size: int,
) -> int:
    tokens_per_step = batch_size * block_size
    return max(1, train_token_count // tokens_per_step)


def run_experiment(run_config: RunConfig) -> dict[str, object]:
    set_seed(run_config.seed)
    tokenizer = load_tokenizer(run_config.tokenizer_name)
    text = load_corpus_text(
        corpus_path=run_config.corpus_path,
        dataset_arrow_path=run_config.dataset_arrow_path,
        max_documents=run_config.max_documents,
    )
    token_ids = encode_corpus(tokenizer, text, run_config.max_corpus_tokens)
    train_ids, val_ids = train_val_split(token_ids)
    model_config = build_model_config(
        config_name=run_config.config_name,
        vocab_size=tokenizer.vocab_size,
        block_size=run_config.block_size,
        tiny_n_layer=run_config.tiny_n_layer,
        tiny_n_embd=run_config.tiny_n_embd,
    )

    active_token_ids = select_active_vocab(train_ids.tolist(), run_config.top_k_vocab)
    base_model = GPT2LMHeadModel(model_config)
    random_model = copy.deepcopy(base_model)
    svd_model: GPT2LMHeadModel = copy.deepcopy(base_model)
    init_metrics: dict[str, float | list[float]] = {}
    if run_config.init_scheme == "topk":
        log_bigram_matrix, _, smart_bigram_metrics = build_smart_bigram_log_probs(
            train_token_ids=train_ids.tolist(),
            val_token_ids=val_ids.tolist(),
            active_token_ids=active_token_ids,
        )
        embedding_vectors, unembedding_vectors = svd_factorize_log_bigrams(
            log_bigram_matrix,
            d_model=model_config.n_embd,
        )
        svd_model = BigramInitGPT2LMHeadModel(
            model_config,
            adapter_strategy=run_config.adapter_strategy,
        )
        initialize_model_from_svd(
            svd_model,
            active_token_ids=active_token_ids,
            embedding_vectors=embedding_vectors,
            unembedding_vectors=unembedding_vectors,
        )
        init_metrics.update(smart_bigram_metrics)
    elif run_config.init_scheme == "streaming_full":
        sparse_log_bigram_matrix = compute_streaming_sparse_log_bigram_matrix(
            train_ids.tolist(),
            vocab_size=tokenizer.vocab_size,
        )
        embedding_vectors, unembedding_vectors = sparse_factorize_log_bigrams(
            sparse_log_bigram_matrix,
            d_model=model_config.n_embd,
            random_state=run_config.seed,
        )
        initialize_model_from_full_vocab_factors(
            svd_model,
            embedding_vectors=embedding_vectors,
            unembedding_vectors=unembedding_vectors,
        )
    elif run_config.init_scheme == "streaming_full_gaussian_fill":
        sparse_log_bigram_matrix = compute_streaming_sparse_log_bigram_matrix(
            train_ids.tolist(),
            vocab_size=tokenizer.vocab_size,
        )
        embedding_vectors, unembedding_vectors = sparse_factorize_log_bigrams(
            sparse_log_bigram_matrix,
            d_model=model_config.n_embd,
            random_state=run_config.seed,
        )
        embedding_vectors, unembedding_vectors = gaussian_fill_missing_token_rows(
            embedding=embedding_vectors,
            unembedding=unembedding_vectors,
            token_ids=train_ids.tolist(),
            vocab_size=tokenizer.vocab_size,
            seed=run_config.seed,
        )
        initialize_model_from_full_vocab_factors(
            svd_model,
            embedding_vectors=embedding_vectors,
            unembedding_vectors=unembedding_vectors,
        )
    elif run_config.init_scheme == "factorized_bigram":
        embedding_vectors, unembedding_vectors, init_metrics = learn_factorized_bigram_tables(
            train_data=train_ids,
            vocab_size=tokenizer.vocab_size,
            d_model=model_config.n_embd,
            batch_size=run_config.bigram_batch_size,
            train_steps=run_config.bigram_train_steps,
            learning_rate=run_config.bigram_learning_rate,
            weight_decay=run_config.weight_decay,
            device=run_config.device,
            seed=run_config.seed,
        )
        initialize_model_from_full_vocab_factors(
            svd_model,
            embedding_vectors=embedding_vectors,
            unembedding_vectors=unembedding_vectors,
        )
        if run_config.bigram_backbone_init == "identity":
            initialize_backbone_as_identity(svd_model)
    else:
        raise ValueError(f"Unsupported init_scheme={run_config.init_scheme!r}")

    random_metrics = train_model(
        model=random_model,
        train_data=train_ids,
        val_data=val_ids,
        batch_size=run_config.batch_size,
        block_size=run_config.block_size,
        train_steps=run_config.train_steps,
        eval_batches=run_config.eval_batches,
        learning_rate=run_config.learning_rate,
        weight_decay=run_config.weight_decay,
        device=run_config.device,
        freeze_token_table_params=False,
    )
    svd_metrics = train_model(
        model=svd_model,
        train_data=train_ids,
        val_data=val_ids,
        batch_size=run_config.batch_size,
        block_size=run_config.block_size,
        train_steps=run_config.train_steps,
        eval_batches=run_config.eval_batches,
        learning_rate=run_config.learning_rate,
        weight_decay=run_config.weight_decay,
        device=run_config.device,
        freeze_token_table_params=run_config.freeze_svd_token_tables,
    )

    return {
        "config": asdict(run_config),
        "model_config": model_config.to_dict(),
        "num_corpus_tokens": len(token_ids),
        "active_vocab_size": len(active_token_ids),
        "init_metrics": init_metrics,
        "random_init": random_metrics,
        "svd_init": svd_metrics,
    }


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Compare random GPT-2 initialization to SVD-initialized embeddings from a corpus bigram model."
    )
    parser.add_argument("--corpus-path", type=str, default=None)
    parser.add_argument("--dataset-arrow-path", type=str, default=None)
    parser.add_argument("--use-c4-filter-small", action="store_true")
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--config-name", choices=["tiny", "gpt2"], default="tiny")
    parser.add_argument("--tiny-n-layer", type=int, default=4)
    parser.add_argument("--tiny-n-embd", type=int, default=128)
    parser.add_argument(
        "--init-scheme",
        choices=["topk", "streaming_full", "streaming_full_gaussian_fill", "factorized_bigram"],
        default="topk",
    )
    parser.add_argument("--top-k-vocab", type=int, default=256)
    parser.add_argument("--max-corpus-tokens", type=int, default=20000)
    parser.add_argument("--max-documents", type=int, default=1000)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--bigram-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bigram-train-steps", type=int, default=200)
    parser.add_argument("--bigram-batch-size", type=int, default=256)
    parser.add_argument("--bigram-backbone-init", choices=["default", "identity"], default="default")
    parser.add_argument("--adapter-strategy", choices=["linear", "rotation_scale", "rotation_only"], default="linear")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-svd-token-tables", action="store_true")
    args = parser.parse_args()
    dataset_arrow_path = args.dataset_arrow_path
    if args.use_c4_filter_small:
        dataset_arrow_path = C4_FILTER_SMALL_ARROW_PATH
    return RunConfig(
        corpus_path=args.corpus_path,
        dataset_arrow_path=dataset_arrow_path,
        tokenizer_name=args.tokenizer_name,
        config_name=args.config_name,
        tiny_n_layer=args.tiny_n_layer,
        tiny_n_embd=args.tiny_n_embd,
        init_scheme=args.init_scheme,
        top_k_vocab=args.top_k_vocab,
        max_corpus_tokens=args.max_corpus_tokens,
        max_documents=args.max_documents,
        block_size=args.block_size,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        eval_batches=args.eval_batches,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bigram_learning_rate=args.bigram_learning_rate,
        bigram_train_steps=args.bigram_train_steps,
        bigram_batch_size=args.bigram_batch_size,
        bigram_backbone_init=args.bigram_backbone_init,
        adapter_strategy=args.adapter_strategy,
        seed=args.seed,
        device=args.device,
        freeze_svd_token_tables=args.freeze_svd_token_tables,
    )


def main() -> None:
    run_config = parse_args()
    results = run_experiment(run_config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
