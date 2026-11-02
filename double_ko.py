"""Joint ATF6/ERN1 reward and the existing five-panel expression comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch

from total_benchmark import evaluate_expression


def nonzero_evaluation_mask(expression: np.ndarray) -> np.ndarray:
    if expression.ndim != 2 or expression.shape[1] != 2000:
        raise ValueError("Evaluation filtering requires the common 2,000-gene panel.")
    if not np.isfinite(expression).all() or (expression < 0).any():
        raise ValueError("Evaluation expression must be finite and nonnegative.")
    return (expression > 0).any(axis=1)


def joint_decrease(current: torch.Tensor, arrival: torch.Tensor) -> torch.Tensor:
    if current.shape[-1] != 2 or arrival.shape[-1] != 2:
        raise ValueError("Joint KO requires exactly two target outputs: ATF6 and ERN1.")
    margin = 8 * torch.finfo(arrival.dtype).eps * current.abs().clamp_min(1)
    return (
        torch.isfinite(arrival).all(dim=-1)
        & torch.isfinite(current).all(dim=-1)
        & (arrival >= 0).all(dim=-1)
        & ((current - arrival) > margin).all(dim=-1)
    )


def joint_reward(
    current: torch.Tensor, arrival: torch.Tensor, start: torch.Tensor,
) -> torch.Tensor:
    if current.shape != arrival.shape or current.shape[-1] != 2 or start.shape != (2,):
        raise ValueError("Reward expects matching (..., 2) targets and a (2,) shared start.")
    if not torch.isfinite(start).all() or (start <= 0).any():
        raise ValueError("Both decoded shared-start targets must be finite and positive.")
    if not torch.isfinite(current).all() or not torch.isfinite(arrival).all():
        raise ValueError("Reward targets must be finite.")
    return ((current - arrival) / start).mean(dim=-1)


def compare_three_conditions(
    prediction: np.ndarray, measured: dict[str, np.ndarray],
    genes: list[str], output_dir: Path,
) -> pd.DataFrame:
    if len(genes) != 2000 or len(set(genes)) != 2000:
        raise ValueError("The five metrics require one common ordered 2,000-gene panel.")
    if len(measured) != 3:
        raise ValueError("Exactly three measured comparison conditions are required.")
    output_dir.mkdir(parents=True, exist_ok=True)
    results, rows, histogram_rows = [], [], []
    bins = np.linspace(0, 1, 51)
    for condition, values in measured.items():
        if prediction.shape[1] != len(genes) or values.shape[1] != len(genes):
            raise ValueError(f"{condition}: expression columns differ from the gene panel.")
        metrics = evaluate_expression(prediction, values)
        predicted_mean, measured_mean = prediction.mean(axis=0), values.mean(axis=0)
        if np.ptp(predicted_mean) == 0 or np.ptp(measured_mean) == 0:
            raise ValueError(f"{condition}: Pearson correlation is undefined for constant means.")
        metrics["gene_Pearson_r"] = float(pearsonr(predicted_mean, measured_mean).statistic)
        if not np.isfinite(metrics["gene_Pearson_r"]):
            raise FloatingPointError(f"{condition}: non-finite Pearson correlation.")
        rows.append({"condition": condition, **{
            key: value for key, value in metrics.items() if key not in {"cosine", "residual"}
        }})
        results.append(metrics)
        pd.DataFrame({
            "gene": genes, "virtual_double_KO_mean": predicted_mean,
            "measured_mean": measured_mean, "residual": metrics["residual"],
        }).to_csv(output_dir / f"{condition}_gene_means.csv", index=False)
        counts, edges = np.histogram(metrics["cosine"], bins)
        for i, count in enumerate(counts):
            histogram_rows.append({
                "condition": condition, "bin_left": edges[i], "bin_right": edges[i + 1],
                "count": int(count), "probability_density": count / counts.sum() / (edges[i + 1] - edges[i]),
            })
    summary = pd.DataFrame(rows)
    names, colors = list(measured), ["#756BB1", "#E45756", "#4C78A8"]
    labels = [name.replace("_pMJ", "\npMJ") for name in names]
    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    for ax, key, title, ylabel in [
        (axes[0, 0], "cosine", "1. Cell-wise Cosine Similarity", "All virtual x measured cell pairs"),
        (axes[0, 1], "residual", "2. Gene-wise Residuals", "Mean virtual double KO - mean measured"),
    ]:
        boxes = ax.boxplot(
            [result[key] for result in results], tick_labels=labels,
            patch_artist=True, showfliers=False, showmeans=True,
            medianprops={"color": "black"},
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"},
        )
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
        ax.set(title=title, ylabel=ylabel)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 1].axhline(0, color="black", linestyle="--")
    x = np.arange(3)
    axes[0, 2].bar(x, summary["gene_Pearson_r"], color=colors)
    axes[0, 2].set(
        title="3. Pearson Correlation", ylabel="Pearson r across 2,000 gene means",
        xticks=x, xticklabels=labels, ylim=(-1, 1.1),
    )
    axes[1, 0].bar(x - 0.18, summary["gene_MAE"], 0.36, label="MAE", color="#4C78A8")
    axes[1, 0].bar(x + 0.18, summary["gene_RMSE"], 0.36, label="RMSE", color="#E45756")
    axes[1, 0].set(
        title="4. Expression Prediction Error", ylabel="Error across 2,000 gene means",
        xticks=x, xticklabels=labels,
    )
    axes[1, 0].legend()
    for name, result, color in zip(names, results, colors):
        axes[1, 1].hist(
            result["cosine"], bins=bins, density=True, histtype="step",
            linewidth=1.7, label=name, color=color,
        )
    axes[1, 1].set(
        title="5. Cosine Similarity Distributions",
        xlabel="Cosine similarity", ylabel="Probability density", xlim=(0, 1),
    )
    axes[1, 1].legend(fontsize=8)
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0, 1, "Same virtual double-KO population for all comparisons.\n"
        "Density bands combined; no pairwise p-value tests.\n"
        "Double-KO cells informed DEG/VAE/support: not a holdout.\n"
        "Single-KO cells are evaluation-only.\n"
        "Fixed-VAE trajectories are not causal KO validation.",
        va="top", fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "five_metric_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    summary.to_csv(output_dir / "five_metric_summary.csv", index=False)
    pd.DataFrame(histogram_rows).to_csv(output_dir / "cosine_histogram_50bins.csv", index=False)
    np.savez_compressed(
        output_dir / "five_metric_distributions.npz", conditions=np.array(names),
        genes=np.array(genes),
        **{f"{key}_{i}": result[key] for i, result in enumerate(results) for key in ["cosine", "residual"]},
    )
    pd.DataFrame({
        "metric_number": [1, 6, 7, 8, 12],
        "figure_panel_number": [1, 2, 3, 4, 5],
        "definition": [
            "All predicted x measured cell-pair cosine over the common 2,000-gene panel",
            "Per-gene mean(predicted) - mean(measured)",
            "Pearson correlation of predicted and measured gene mean vectors",
            "MAE and RMSE of the per-gene mean residual vector",
            "Probability-density histogram of metric 1; 50 shared bins over [0, 1]",
        ],
    }).to_csv(output_dir / "metric_definitions.csv", index=False)
    return summary
