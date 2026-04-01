from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def row_norms(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(matrix.float(), dim=1)


def summarize(norms: torch.Tensor) -> dict[str, float]:
    quantiles = torch.quantile(norms, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    return {
        "mean": float(norms.mean().item()),
        "std": float(norms.std(unbiased=False).item()),
        "min": float(norms.min().item()),
        "p05": float(quantiles[0].item()),
        "p25": float(quantiles[1].item()),
        "median": float(quantiles[2].item()),
        "p75": float(quantiles[3].item()),
        "p95": float(quantiles[4].item()),
        "max": float(norms.max().item()),
    }


def main() -> None:
    base = Path("results/bigram_init_exports_18000")
    active_token_ids = torch.load(base / "active_token_ids.pth", map_location="cpu").long()

    bigram_svd_embedding = torch.load(base / "bigram_svd_embedding.pth", map_location="cpu")
    bigram_svd_unembedding = torch.load(base / "bigram_svd_unembedding.pth", map_location="cpu")
    baseline_embedding = torch.load(base / "baseline_embedding.pth", map_location="cpu")[active_token_ids]
    baseline_unembedding = torch.load(base / "baseline_unembedding.pth", map_location="cpu")[active_token_ids]

    matrices = {
        "bigram_svd_embedding_active": bigram_svd_embedding,
        "bigram_svd_unembedding_active": bigram_svd_unembedding,
        "baseline_embedding_active": baseline_embedding,
        "baseline_unembedding_active": baseline_unembedding,
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    stats: dict[str, dict[str, float]] = {}
    styles = {
        "bigram_svd_embedding_active": {"linewidth": 2.2, "linestyle": "-"},
        "bigram_svd_unembedding_active": {"linewidth": 2.2, "linestyle": "--"},
        "baseline_embedding_active": {"linewidth": 2.0, "linestyle": "-."},
        "baseline_unembedding_active": {"linewidth": 2.0, "linestyle": ":"},
    }

    for name, matrix in matrices.items():
        norms = row_norms(matrix)
        stats[name] = summarize(norms)
        axes[0].hist(norms.numpy(), bins=80, density=True, alpha=0.28, label=name)
        sorted_norms, _ = torch.sort(norms)
        cdf = torch.arange(1, sorted_norms.numel() + 1, dtype=torch.float32) / sorted_norms.numel()
        axes[1].plot(sorted_norms.numpy(), cdf.numpy(), label=name, **styles[name])

    axes[0].set_title("Row-Norm Density on Shared Active Vocab")
    axes[0].set_xlabel("Row norm")
    axes[0].set_ylabel("Density")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Row-Norm CDF on Shared Active Vocab")
    axes[1].set_xlabel("Row norm")
    axes[1].set_ylabel("Cumulative fraction of rows")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    plot_path = base / "row_norm_distributions_active_vocab.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    stats_path = base / "row_norm_stats_active_vocab.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps({"plot_path": str(plot_path), "stats_path": str(stats_path)}, indent=2))


if __name__ == "__main__":
    main()
