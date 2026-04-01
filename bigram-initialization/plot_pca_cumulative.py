from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def cumulative_explained_variance(matrix: torch.Tensor) -> torch.Tensor:
    centered = matrix.float() - matrix.float().mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    explained = singular_values.square()
    explained = explained / explained.sum()
    return torch.cumsum(explained, dim=0)


def main() -> None:
    base = Path("results/bigram_init_exports_18000")
    matrices = {
        "bigram_svd_embedding": base / "bigram_svd_embedding.pth",
        "bigram_svd_unembedding": base / "bigram_svd_unembedding.pth",
        "baseline_embedding": base / "baseline_embedding.pth",
        "baseline_unembedding": base / "baseline_unembedding.pth",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    styles = {
        "bigram_svd_embedding": {"linewidth": 2.2, "linestyle": "-"},
        "bigram_svd_unembedding": {"linewidth": 2.2, "linestyle": "--"},
        "baseline_embedding": {"linewidth": 2.0, "linestyle": "-."},
        "baseline_unembedding": {"linewidth": 2.0, "linestyle": ":"},
    }

    for name, path in matrices.items():
        cumulative = cumulative_explained_variance(torch.load(path, map_location="cpu"))
        components = torch.arange(1, cumulative.numel() + 1)
        ax.plot(components, cumulative.numpy(), label=name, **styles[name])

    ax.axhline(0.90, color="gray", linewidth=1.0, linestyle=":")
    ax.axhline(0.95, color="gray", linewidth=1.0, linestyle="--")
    ax.axhline(0.99, color="gray", linewidth=1.0, linestyle="-")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA Cumulative Explained Variance of Exported Token Matrices")
    ax.set_xlim(1, 128)
    ax.set_ylim(0.0, 1.01)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path = base / "pca_cumulative_curves.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
