"""Label-informed gene selection for PCA, with an untouched cell-level test set.

Run on the log-normalized AnnData produced by Peturb-seq.ipynb. Discovery
labels select genes, validation labels select the panel, and test labels are
used only after selection. This is supervised selection, not unsupervised
evidence of a perturbation effect or independent biological replication.
"""

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import spearmanr, ttest_ind
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    pairwise_distances,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


CONTROL = "3x_neg_ctrl_pMJ144-1"
PERTURBED = "ATF6_only_pMJ145"
SEED = 20261029


def _fit_pca(x, fit_rows, genes, scale):
    scaler = StandardScaler(with_std=scale).fit(x[np.ix_(fit_rows, genes)])
    z = scaler.transform(x[:, genes])
    if scale:
        z = np.clip(z, -10, 10)
    model = PCA(n_components=2, svd_solver="full")
    model.fit(z[fit_rows])
    return model.transform(z), scaler, model


def _metrics(coords, labels, fit_rows, evaluation_rows):
    classifier = LogisticRegression(random_state=SEED).fit(
        coords[fit_rows], labels[fit_rows]
    )
    probabilities = classifier.predict_proba(coords[evaluation_rows])[:, 1]
    return {
        "silhouette_2pc": float(
            silhouette_score(coords[evaluation_rows], labels[evaluation_rows])
        ),
        "balanced_accuracy_2pc": float(
            balanced_accuracy_score(
                labels[evaluation_rows], classifier.predict(coords[evaluation_rows])
            )
        ),
        "auroc_2pc": float(roc_auc_score(labels[evaluation_rows], probabilities)),
    }


def _rank_genes(adata, x, labels, train):
    discovery = x[train]
    groups = [discovery[labels[train] == label] for label in (0, 1)]
    means = [group.mean(axis=0) for group in groups]
    variances = [group.var(axis=0, ddof=1) for group in groups]
    detection = [(group > 0).mean(axis=0) for group in groups]
    difference = means[1] - means[0]
    pooled_sd = np.sqrt((variances[0] + variances[1]) / 2)
    effect = np.divide(
        difference, pooled_sd, out=np.zeros_like(difference), where=pooled_sd > 0
    )
    welch = ttest_ind(groups[1], groups[0], axis=0, equal_var=False)
    nuisance = np.asarray(
        adata.var_names.str.startswith(("MT-", "RPS", "RPL", "HBA", "HBB"))
    )
    eligible = (
        (np.maximum(*detection) >= 0.10)
        & (pooled_sd > 0.05)
        & ~nuisance
        & np.isfinite(welch.statistic)
    )
    indices = np.flatnonzero(eligible)
    if len(indices) < 10:
        raise ValueError("Fewer than 10 genes pass the discovery-only filters.")

    def ordered(score, subset=indices):
        return subset[np.argsort(-score[subset], kind="stable")]

    rankings = {
        "effect": ordered(np.abs(effect)),
        "welch": ordered(np.abs(welch.statistic)),
        "ATF6_up": ordered(effect, indices[difference[indices] > 0]),
        "ATF6_down": ordered(-effect, indices[difference[indices] < 0]),
    }
    up, down = rankings["ATF6_up"], rankings["ATF6_down"]
    balanced = []
    for i in range(max(len(up), len(down))):
        if i < len(up):
            balanced.append(up[i])
        if i < len(down):
            balanced.append(down[i])
    rankings["balanced_effect"] = np.asarray(balanced, dtype=int)

    scaler = StandardScaler().fit(discovery[:, indices])
    classifier = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=0.1,
        max_iter=3000,
        random_state=SEED,
    ).fit(scaler.transform(discovery[:, indices]), labels[train])
    weights = np.zeros(x.shape[1])
    weights[indices] = np.abs(classifier.coef_[0])
    rankings["sparse_logistic"] = ordered(
        weights, indices[weights[indices] > 0]
    )

    hvg = adata[train].copy()
    sc.pp.highly_variable_genes(hvg, flavor="seurat", n_top_genes=2000)
    hvg_score = np.asarray(hvg.var["dispersions_norm"], dtype=float)
    hvg_indices = np.flatnonzero(
        np.asarray(hvg.var["highly_variable"]) & np.isfinite(hvg_score)
    )
    rankings["HVG_baseline"] = ordered(hvg_score, hvg_indices)
    table = pd.DataFrame(
        {
            "gene": adata.var_names,
            "mean_log1p_control": means[0],
            "mean_log1p_ATF6": means[1],
            "difference_log1p_ATF6_minus_control": difference,
            "standardized_effect": effect,
            "welch_t_discovery": welch.statistic,
            "pvalue_discovery": welch.pvalue,
            "detection_control": detection[0],
            "detection_ATF6": detection[1],
            "eligible": eligible,
            "sparse_logistic_abs_weight": weights,
        }
    ).set_index("gene")
    return rankings, table


def optimize_perturbation_pca(adata, output_dir="perturbation_pca_results"):
    """Return a selected panel and PCA AnnData; persist figures and audit tables.

    Candidate selection maximizes validation silhouette in the actual PC1/PC2
    plane, not classification in the full expression matrix. No test score
    participates in selecting genes, scaling, panel size, or PCA orientation.
    """
    with threadpool_limits(limits=2):
        return _optimize(adata, Path(output_dir))


def _optimize(adata, output_dir):
    if "counts" not in adata.layers or "log1p" not in adata.uns:
        raise ValueError("Run the original counts/normalize_total/log1p cell first.")
    if not adata.var_names.is_unique or not adata.obs_names.is_unique:
        raise ValueError("Unique gene names and cell names are required.")
    if adata.obs["perturbation"].isna().any():
        raise ValueError("Missing perturbation labels are not supported.")
    observed = set(adata.obs["perturbation"].astype(str))
    if observed != {CONTROL, PERTURBED}:
        raise ValueError(f"Expected the notebook's two groups; found {observed}.")
    raw_counts = adata.layers["counts"]
    values = raw_counts.data if sparse.issparse(raw_counts) else np.asarray(raw_counts)
    if not np.allclose(values, np.round(values), atol=1e-5, rtol=0):
        raise ValueError("The counts layer does not contain untransformed counts.")
    x = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Expression contains NaN or infinite values.")
    labels = (adata.obs["perturbation"].astype(str) == PERTURBED).to_numpy(int)
    train_validation, test = train_test_split(
        np.arange(adata.n_obs), test_size=0.20, stratify=labels, random_state=SEED
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        stratify=labels[train_validation],
        random_state=SEED + 1,
    )
    split = np.full(adata.n_obs, "discovery", dtype=object)
    split[validation] = "validation"
    split[test] = "test"
    assert not (set(train) & set(validation) or set(train_validation) & set(test))
    print(pd.crosstab(pd.Series(split, name="split"), labels).to_string())
    rankings, gene_table = _rank_genes(adata, x, labels, train)
    sizes = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500]
    records = []
    panels = {}
    for ranking, order in rankings.items():
        candidate_sizes = sorted(set(min(size, len(order)) for size in sizes))
        if ranking == "HVG_baseline":
            candidate_sizes = [len(order)]
        for size in candidate_sizes:
            if size < 2:
                continue
            genes = order[:size]
            for scale in (False, True):
                coords, _, model = _fit_pca(x, train, genes, scale)
                record = {
                    "ranking": ranking,
                    "n_genes": size,
                    "scale": scale,
                    **_metrics(coords, labels, train, validation),
                    "variance_explained_2pc": float(
                        model.explained_variance_ratio_.sum()
                    ),
                }
                key = len(records)
                records.append(record)
                panels[key] = genes.copy()
        print(f"Finished {ranking}: {len(records)} candidate panels", flush=True)
    pool = []
    for pair in zip(rankings["sparse_logistic"], rankings["effect"]):
        for gene in pair:
            if gene not in pool:
                pool.append(int(gene))
        if len(pool) >= 8:
            break
    pool = pool[:8]
    for size in range(2, len(pool) + 1):
        for subset in combinations(pool, size):
            genes = np.asarray(subset)
            for scale in (False, True):
                coords, _, model = _fit_pca(x, train, genes, scale)
                key = len(records)
                records.append(
                    {
                        "ranking": "top8_subsets",
                        "n_genes": size,
                        "scale": scale,
                        **_metrics(coords, labels, train, validation),
                        "variance_explained_2pc": float(
                            model.explained_variance_ratio_.sum()
                        ),
                    }
                )
                panels[key] = genes.copy()
    print(f"Finished discovery-ranked top-8 subsets: {len(records)} candidates", flush=True)
    search = pd.DataFrame(records)
    search["genes"] = [
        ";".join(adata.var_names[panels[key]]) for key in search.index
    ]
    supervised = search[search["ranking"] != "HVG_baseline"]
    best_index = supervised.sort_values(
        ["silhouette_2pc", "balanced_accuracy_2pc", "n_genes"],
        ascending=[False, False, True],
        kind="stable",
    ).index[0]
    best = search.loc[best_index]
    baseline_index = search[search["ranking"] == "HVG_baseline"].sort_values(
        "silhouette_2pc", ascending=False
    ).index[0]
    downstream_candidates = supervised[
        [
            "ATF6" not in set(adata.var_names[panels[key]])
            for key in supervised.index
        ]
    ]
    downstream_index = downstream_candidates.sort_values(
        ["silhouette_2pc", "balanced_accuracy_2pc", "n_genes"],
        ascending=[False, False, True],
        kind="stable",
    ).index[0]
    selected_indices = panels[best_index]
    selected_genes = adata.var_names[selected_indices].tolist()
    # Freeze genes and configuration before using the untouched test cells.
    coords, _, pca = _fit_pca(
        x, train_validation, selected_indices, bool(best["scale"])
    )
    baseline_coords, _, _ = _fit_pca(
        x,
        train_validation,
        panels[baseline_index],
        bool(search.loc[baseline_index, "scale"]),
    )
    downstream_genes = adata.var_names[panels[downstream_index]].tolist()
    downstream_coords, _, _ = _fit_pca(
        x,
        train_validation,
        panels[downstream_index],
        bool(search.loc[downstream_index, "scale"]),
    )
    evaluation = pd.DataFrame(
        [
            {
                "panel": name,
                "n_genes": count,
                **_metrics(embedding, labels, train_validation, test),
            }
            for name, embedding, count in [
                ("selected", coords, len(selected_genes)),
                ("HVG_baseline", baseline_coords, len(panels[baseline_index])),
                ("without_ATF6_target", downstream_coords, len(downstream_genes)),
            ]
        ]
    ).set_index("panel")
    distances = pairwise_distances(coords[test])
    np.fill_diagonal(distances, 0)
    rng = np.random.default_rng(SEED)
    null = np.array(
        [
            silhouette_score(
                distances, rng.permutation(labels[test]), metric="precomputed"
            )
            for _ in range(199)
        ]
    )
    permutation_p = float(
        (1 + (null >= evaluation.loc["selected", "silhouette_2pc"]).sum())
        / (len(null) + 1)
    )
    qc_columns = ["log1p_total_counts", "n_genes_by_counts", "pct_counts_mt"]
    qc = adata.obs[qc_columns].to_numpy(dtype=float)
    qc_model = LogisticRegression(max_iter=2000, random_state=SEED)
    qc_scaler = StandardScaler().fit(qc[train_validation])
    qc_model.fit(qc_scaler.transform(qc[train_validation]), labels[train_validation])
    qc_auroc = float(
        roc_auc_score(
            labels[test], qc_model.predict_proba(qc_scaler.transform(qc[test]))[:, 1]
        )
    )
    qc_correlations = pd.DataFrame(
        {
            column: [
                spearmanr(coords[test, pc], adata.obs.iloc[test][column]).statistic
                for pc in (0, 1)
            ]
            for column in qc_columns
        },
        index=["PC1", "PC2"],
    )
    # Sensitivity analysis only: never use this score to change the frozen panel.
    covariates = np.column_stack([np.ones(adata.n_obs), qc_scaler.transform(qc)])
    expression = x[:, selected_indices]
    coefficients = np.linalg.lstsq(
        covariates[train_validation], expression[train_validation], rcond=None
    )[0]
    residuals = expression - covariates @ coefficients
    residual_coords, _, _ = _fit_pca(
        residuals, train_validation, np.arange(len(selected_genes)), bool(best["scale"])
    )
    qc_adjusted = _metrics(residual_coords, labels, train_validation, test)

    # The all-cell embedding is for visualization, not the held-out evaluation.
    full_coords, _, full_model = _fit_pca(
        x, np.arange(adata.n_obs), selected_indices, bool(best["scale"])
    )
    adata_pca = adata[:, selected_genes].copy()
    adata_pca.obs["pca_selection_split"] = pd.Categorical(split)
    adata_pca.obsm["X_pca"] = full_coords.astype(np.float32)
    adata_pca.obsm["X_pca_heldout"] = coords.astype(np.float32)
    adata_pca.varm["PCs"] = full_model.components_.T.astype(np.float32)
    adata_pca.uns["pca"] = {
        "params": {"zero_center": True, "use_highly_variable": False},
        "variance": full_model.explained_variance_,
        "variance_ratio": full_model.explained_variance_ratio_,
    }
    adata.var["pca_perturbation_selected"] = adata.var_names.isin(selected_genes)
    selected_table = gene_table.loc[selected_genes].copy()
    selected_table.insert(0, "selection_rank", np.arange(1, len(selected_genes) + 1))
    selected_table["PC1_loading"] = full_model.components_[0]
    selected_table["PC2_loading"] = full_model.components_[1]
    selected_table["direction"] = np.where(
        selected_table["standardized_effect"] > 0, "ATF6_up", "ATF6_down"
    )
    summary = {
        "seed": SEED,
        "groups": {"control": CONTROL, "perturbed": PERTURBED},
        "cells": int(adata.n_obs),
        "split_sizes": {
            "discovery": len(train), "validation": len(validation), "test": len(test)
        },
        "candidate_panels": len(search),
        "subset_search_pool": adata.var_names[pool].tolist(),
        "selection_metric": "validation silhouette in PC1/PC2",
        "best_ranking": str(best["ranking"]),
        "scale": bool(best["scale"]),
        "clip_standardized_at": 10 if best["scale"] else None,
        "n_genes": len(selected_genes),
        "selected_genes": selected_genes,
        "without_ATF6_target_genes": downstream_genes,
        "without_ATF6_target_validation_silhouette": float(
            search.loc[downstream_index, "silhouette_2pc"]
        ),
        "validation_silhouette_2pc": float(best["silhouette_2pc"]),
        "test": evaluation.to_dict(orient="index"),
        "test_label_permutation_pvalue": permutation_p,
        "test_permutations": len(null),
        "qc_only_test_auroc": qc_auroc,
        "qc_residualized_test": qc_adjusted,
        "test_pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "interpretation": (
            "Supervised gene selection followed by unsupervised PCA. Cell-level "
            "holdout from one experiment, not independent biological replication. "
            "QC adjustment is a sensitivity analysis and can remove biological signal. "
            "All-cell PCA is a refit for visualization; use heldout coordinates "
            "for the reported test metrics. Best among searched panels, not a "
            "guaranteed globally optimal gene set."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_table.to_csv(output_dir / "selected_genes.csv")
    (output_dir / "selected_genes.txt").write_text(
        "\n".join(selected_genes) + "\n", encoding="utf-8"
    )
    (output_dir / "without_ATF6_target_genes.txt").write_text(
        "\n".join(downstream_genes) + "\n", encoding="utf-8"
    )
    search.sort_values("silhouette_2pc", ascending=False).to_csv(
        output_dir / "candidate_scores.csv", index=False
    )
    gene_table.to_csv(output_dir / "discovery_gene_statistics.csv")
    evaluation.to_csv(output_dir / "heldout_metrics.csv")
    qc_correlations.to_csv(output_dir / "heldout_qc_correlations.csv")
    coordinates = adata.obs[["perturbation"]].copy()
    coordinates["split"] = split
    coordinates[["PC1_heldout_model", "PC2_heldout_model"]] = coords
    coordinates[["PC1_all_cell_refit", "PC2_all_cell_refit"]] = full_coords
    coordinates.to_csv(output_dir / "pca_coordinates.csv")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    adata_pca.uns["perturbation_gene_selection"] = json.dumps(summary)
    adata_pca.write_h5ad(output_dir / "selected_gene_pca.h5ad", compression="gzip")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    views = [
        (baseline_coords, test, "HVG baseline: untouched test"),
        (coords, test, "Selected genes: untouched test"),
        (full_coords, np.arange(adata.n_obs), "Selected genes: all-cell refit"),
    ]
    for ax, (embedding, rows, title) in zip(axes, views):
        for label, name, color in [(0, "Control", "#4477AA"), (1, "ATF6", "#EE6677")]:
            cells = rows[labels[rows] == label]
            ax.scatter(
                embedding[cells, 0], embedding[cells, 1],
                s=7, alpha=0.45, c=color, label=name, rasterized=True,
            )
        ax.set(title=title, xlabel="PC1", ylabel="PC2")
        ax.legend(frameon=False, markerscale=2)
    fig.suptitle(f"Label-informed PCA gene panel: {len(selected_genes)} genes")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "pca_comparison.pdf", bbox_inches="tight")
    plt.show()
    print("\nBest validation configuration:\n", best.to_string())
    print("\nUntouched test metrics:\n", evaluation.to_string())
    print("\nQC-only test AUROC:", qc_auroc)
    print("QC-residualized test metrics:", qc_adjusted)
    print("Held-out label permutation p-value:", permutation_p)
    print("Selected genes:", ", ".join(selected_genes))
    return {
        "selected_genes": selected_genes,
        "selected_gene_table": selected_table,
        "candidate_scores": search.sort_values("silhouette_2pc", ascending=False),
        "heldout_metrics": evaluation,
        "summary": summary,
        "adata_pca": adata_pca,
    }
