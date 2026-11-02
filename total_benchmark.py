"""Evaluate ATF6 expression predictions against the saved Validation 2 panel."""

from dataclasses import dataclass
import ast
import hashlib
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scanpy as sc
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


DATA_URL = (
    "https://zenodo.org/records/7041849/files/"
    "AdamsonWeissman2016_GSM2406677_10X005.h5ad?download=1"
)
CONDITIONS = ["3x_neg_ctrl_pMJ144-1", "ATF6_only_pMJ145"]
EXPRESSION_SPACE = "log1p_CP10K_full_gene_library"
N_GENES = 2000


@dataclass
class BenchmarkData:
    genes: list[str]
    control: ad.AnnData
    measured: np.ndarray
    reference: np.ndarray
    reference_cosine: np.ndarray
    real_barcodes: list[str]
    density_band: np.ndarray
    gene_hash: str
    measured_adata: ad.AnnData | None = None
    split: dict | None = None


def dense_expression(matrix) -> np.ndarray:
    values = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    return np.asarray(values, dtype=np.float32)


def check_expression(values: np.ndarray, name: str) -> None:
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != N_GENES:
        raise ValueError(f"{name}: expected nonempty cells x {N_GENES}, got {values.shape}")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name}: expression must be finite and nonnegative")
    if (np.linalg.norm(values, axis=1) == 0).any():
        raise ValueError(f"{name}: zero-expression cells have undefined biological cosine")


def prepare_data(data_path: Path, reference_dir: Path) -> BenchmarkData:
    """Freeze the archived panel; never select genes from the evaluation labels."""
    with np.load(reference_dir / "virtual_vs_real_KO.npz", allow_pickle=False) as archive:
        genes = archive["genes"].astype(str).tolist()
        reference = archive["virtual_expression"].copy()
        measured_saved = archive["real_expression"].copy()
        cosine_saved = archive["cosine_similarity"].copy()
        real_barcodes = archive["real_cell_barcodes"].astype(str).tolist()
        bands = archive["density_band"].copy()
    panel = pd.read_csv(reference_dir / "comparison_genes_2000.csv")
    if len(genes) != N_GENES or len(set(genes)) != N_GENES or "ATF6" not in genes:
        raise ValueError("Archived panel must contain exactly 2,000 unique genes including ATF6")
    if panel["gene"].astype(str).tolist() != genes:
        raise ValueError("Comparison CSV and archived expression gene order differ")
    if len(real_barcodes) != len(set(real_barcodes)):
        raise ValueError("Duplicate measured KO barcodes")
    check_expression(reference, "DQN reference")
    check_expression(measured_saved, "Archived measured KO")
    if len(real_barcodes) != len(measured_saved):
        raise ValueError("Measured KO barcode and expression counts differ")
    if bands.shape != (len(reference),) or set(bands.tolist()) != {0, 1, 2}:
        raise ValueError("Expected three archived density bands aligned to virtual cells")
    cosine_expected = cosine_similarity(reference, measured_saved)
    if cosine_saved.shape != cosine_expected.shape:
        raise ValueError("Archived cosine dimensions differ from the all-pairs definition")
    np.testing.assert_allclose(cosine_saved, cosine_expected, rtol=1e-5, atol=1e-6)

    if not data_path.exists():
        partial = data_path.with_suffix(".h5ad.part")
        with requests.get(DATA_URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
        partial.replace(data_path)
    backed = ad.read_h5ad(data_path, backed="r")
    try:
        if "perturbation" not in backed.obs:
            raise ValueError("Missing obs['perturbation']")
        if not backed.obs_names.is_unique or not backed.var_names.is_unique:
            raise ValueError("Input gene symbols and cell barcodes must be unique")
        selected = backed[backed.obs["perturbation"].isin(CONDITIONS)].to_memory()
    finally:
        backed.file.close()
    counts = selected.X.data if sparse.issparse(selected.X) else np.asarray(selected.X)
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Input counts must be finite and nonnegative")
    if not np.allclose(counts, np.rint(counts), rtol=0, atol=1e-6):
        raise ValueError("Input X must contain raw integer counts, not normalized expression")
    if (np.asarray(selected.X.sum(axis=1)).ravel() <= 0).any():
        raise ValueError("Zero-library cells must be resolved before normalization")
    missing = sorted(set(genes) - set(selected.var_names))
    if missing:
        raise ValueError(f"Archived genes missing from source: {missing}")
    selected.layers["counts"] = selected.X.copy()
    selected.uns.pop("log1p", None)
    sc.pp.normalize_total(selected, target_sum=1e4)
    sc.pp.log1p(selected)
    selected = selected[:, genes].copy()
    if not set(real_barcodes).issubset(selected.obs_names):
        raise ValueError("Archived KO barcodes missing from source")
    measured_adata = selected[real_barcodes]
    if not measured_adata.obs["perturbation"].eq(CONDITIONS[1]).all():
        raise ValueError("Archived evaluation barcodes include non-ATF6 cells")
    measured = dense_expression(measured_adata.X)
    np.testing.assert_allclose(measured, measured_saved, rtol=1e-5, atol=2e-5)
    control = selected[selected.obs["perturbation"].eq(CONDITIONS[0])].copy()
    if control.n_obs < 2:
        raise ValueError("At least two control cells are required")
    check_expression(dense_expression(control.X), "Control")
    gene_hash = hashlib.sha256("\n".join(genes).encode("utf-8")).hexdigest()
    return BenchmarkData(
        genes, control, measured, reference, cosine_saved,
        real_barcodes, bands, gene_hash, measured_adata.copy(),
    )


def prepare_supervised_split(
    data: BenchmarkData, input_dir: Path, seed: int = 42
) -> BenchmarkData:
    """Use the archived KO population and disjoint 70/10/20 train/valid/test cells."""
    if data.measured_adata is None:
        raise ValueError("Source measured AnnData is required for supervised splitting")
    population = ad.concat([data.control, data.measured_adata], merge="same")
    if not population.obs_names.is_unique:
        raise ValueError("Duplicate cells in supervised population")
    rng = np.random.default_rng(seed)
    split_labels = pd.Series(index=population.obs_names, dtype=str)
    for condition in CONDITIONS:
        ids = population.obs_names[population.obs["perturbation"].eq(condition)].to_numpy()
        ids = rng.permutation(ids)
        n_test, n_valid = int(np.ceil(len(ids) * 0.2)), int(np.ceil(len(ids) * 0.1))
        if len(ids) - n_test - n_valid < 2:
            raise ValueError("Too few cells to split this condition")
        split_labels.loc[ids[:n_test]] = "test"
        split_labels.loc[ids[n_test:n_test + n_valid]] = "valid"
        split_labels.loc[ids[n_test + n_valid:]] = "train"
    population.obs["benchmark_split"] = split_labels
    population.obs["cell_type"] = "K562"
    population.obs["condition"] = np.where(
        population.obs["perturbation"].eq(CONDITIONS[0]), "ctrl", "ATF6",
    )
    population.obs["dose"] = "1.0"
    train = population[population.obs["benchmark_split"].eq("train")].copy()
    valid = population[population.obs["benchmark_split"].eq("valid")].copy()
    query = population[
        population.obs["benchmark_split"].eq("test")
        & population.obs["perturbation"].eq(CONDITIONS[0])
    ].copy()
    ko_test = population[
        population.obs["benchmark_split"].eq("test")
        & population.obs["perturbation"].eq(CONDITIONS[1])
    ].copy()
    groups = [set(x.obs_names) for x in [train, valid, query, ko_test]]
    for i, left in enumerate(groups):
        if any(left & right for right in groups[i + 1:]):
            raise ValueError("Supervised split contains overlapping cell IDs")
    input_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in [("train", train), ("valid", valid), ("query_control", query)]:
        subset.X = dense_expression(subset.X)
        subset.write_h5ad(input_dir / f"{name}.h5ad")
    split = {
        "seed": seed, "mode": "seen_ATF6_heldout_cells",
        "gene_panel_sha256": data.gene_hash, "genes": data.genes,
        "training_barcodes": train.obs_names.tolist(),
        "control_training_barcodes": train.obs_names[train.obs["perturbation"].eq(CONDITIONS[0])].tolist(),
        "validation_barcodes": valid.obs_names.tolist(),
        "query_barcodes": query.obs_names.tolist(),
        "test_KO_barcodes": ko_test.obs_names.tolist(),
        "population": "1715 control and 1202 archived Validation 2 KO cells; other KO cells excluded",
        "fractions": {"train": 0.7, "valid": 0.1, "test": 0.2},
        "counts": pd.crosstab(population.obs["perturbation"], population.obs["benchmark_split"]).to_dict(),
        "panel_limit": "Panel selected using full KO/control labels; not independent gene selection",
    }
    (input_dir / "split.json").write_text(json.dumps(split, indent=2))
    population.obs[["perturbation", "benchmark_split"]].to_csv(input_dir / "cell_split.csv")
    ko_indices = pd.Index(data.real_barcodes).get_indexer(ko_test.obs_names)
    if (ko_indices < 0).any():
        raise ValueError("Test KO cells must be from the archived measured population")
    return BenchmarkData(
        data.genes, query, dense_expression(ko_test.X), data.reference,
        data.reference_cosine[:, ko_indices], ko_test.obs_names.tolist(),
        data.density_band, data.gene_hash, ko_test, split,
    )


def reuse_supervised_split(data: BenchmarkData, input_dir: Path) -> BenchmarkData:
    """Reattach a new DQN export to the original split without overwriting training inputs."""
    split = json.loads((input_dir / "split.json").read_text())
    if split["genes"] != data.genes or split["gene_panel_sha256"] != data.gene_hash:
        raise ValueError("Latest DQN panel differs from the previously trained tools; retraining is required")
    query = ad.read_h5ad(input_dir / "query_control.h5ad")
    if query.var_names.tolist() != data.genes or query.obs_names.tolist() != split["query_barcodes"]:
        raise ValueError("Saved query genes or barcode order differs from the original split")
    if not query.obs["perturbation"].eq(CONDITIONS[0]).all():
        raise ValueError("Saved query includes non-control cells")
    control_indices = data.control.obs_names.get_indexer(query.obs_names)
    if (control_indices < 0).any():
        raise ValueError("Saved query control cells are absent from the current input")
    np.testing.assert_allclose(
        dense_expression(query.X), dense_expression(data.control[query.obs_names].X),
        rtol=1e-5, atol=2e-5,
    )
    ko_indices = pd.Index(data.real_barcodes).get_indexer(split["test_KO_barcodes"])
    if (ko_indices < 0).any():
        raise ValueError("Original held-out KO cells are absent from the latest DQN export")
    return BenchmarkData(
        data.genes, query, data.measured[ko_indices], data.reference,
        data.reference_cosine[:, ko_indices], split["test_KO_barcodes"],
        data.density_band, data.gene_hash, split=split,
    )


def validate_dqn_export(reference_dir: Path, notebook_path: Path) -> dict:
    """Anchor the comparison to the current notebook algorithm and completed export."""
    notebook = json.loads(notebook_path.read_text())
    code_sources = [
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    versions = []
    for source in code_sources:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "LG_REWARD_VERSION"
                for target in node.targets
            ):
                if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
                    raise ValueError("LG_REWARD_VERSION must be an explicit string")
                versions.append(node.value.value)
    if len(versions) != 1:
        raise ValueError("Expected exactly one LG_REWARD_VERSION in the DQN notebook")
    metadata_path = reference_dir.parent / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("reward_version") != versions[0]:
        raise ValueError("Saved DQN results use a different algorithm; rerun DQN before benchmarking")
    export_path = reference_dir / "virtual_vs_real_KO.npz"
    if export_path.stat().st_mtime < metadata_path.stat().st_mtime:
        raise ValueError("DQN expression export predates the current algorithm run metadata")
    with export_path.open("rb") as stream:
        export_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "reward_version": versions[0],
        "reference_export": str(export_path.resolve()),
        "export_sha256": export_hash,
        "export_mtime_ns": export_path.stat().st_mtime_ns,
        "notebook_code_sha256": hashlib.sha256("\n".join(code_sources).encode()).hexdigest(),
        "run_metadata": metadata,
        "sampling_metadata": json.loads((reference_dir / "sampling_metadata.json").read_text()),
    }


def candidate_catalog(training_allowed: bool = False) -> pd.DataFrame:
    catalog = pd.DataFrame([
        {
            "method": "GEARS", "status": "blocked_training_data",
            "reason": "Requires other perturbation training data; control alone cannot learn ATF6 response.",
            "source": "https://github.com/snap-stanford/GEARS",
        },
        {
            "method": "CPA", "status": "blocked_training_data",
            "reason": "No learned ATF6 perturbation embedding; unseen combinations are not an unseen single KO.",
            "source": "https://github.com/theislab/cpa",
        },
        {
            "method": "scGen", "status": "blocked_training_data",
            "reason": "Requires an observed perturbation latent shift; ATF6 KO training is forbidden.",
            "source": "https://github.com/theislab/scgen",
        },
        {
            "method": "CellOracle", "status": "blocked_grn_and_prediction",
            "reason": "Control-only TF KO is possible, but a fitted GRN/prior and calibrated expression prediction are not supplied.",
            "source": "https://github.com/morris-lab/CellOracle",
        },
        {
            "method": "scTenifoldKnk", "status": "incompatible_expression_endpoint",
            "reason": "WT-only virtual KO is possible, but differential-regulation/direction scores are not a cell x gene expression matrix.",
            "source": "https://github.com/cailab-tamu/scTenifoldKnk",
        },
    ])
    if training_allowed:
        for method in ["scGen", "CPA"]:
            catalog.loc[catalog["method"].eq(method), ["status", "reason"]] = [
                "ready_for_supervised_training", "ATF6 KO training permitted; shared held-out cells required.",
            ]
        catalog.loc[catalog["method"].eq("GEARS"), ["status", "reason"]] = [
            "not_run_two_condition_scope", "GEARS is designed/tested for multiple perturbations; not trained on this two-condition benchmark.",
        ]
    return catalog


def load_external_prediction(
    name: str, prediction_path: Path, manifest_path: Path, data: BenchmarkData
) -> tuple[np.ndarray, dict]:
    """Validate declarations and expression; declarations are not a provenance audit."""
    manifest = json.loads(manifest_path.read_text())
    required = {
        "method", "target", "expression_space", "gene_panel_sha256",
        "training_conditions", "used_evaluation_cells", "used_atf6_ko_expression",
        "training_genes", "model_version", "training_provenance",
    }
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"{name}: manifest lacks {sorted(missing)}")
    if manifest["method"] != name or manifest["target"] != "ATF6":
        raise ValueError(f"{name}: method or target mismatch")
    if manifest["expression_space"] != EXPRESSION_SPACE:
        raise ValueError(f"{name}: incompatible normalization; GRN scores/deltas are not expression")
    if manifest["gene_panel_sha256"] != data.gene_hash:
        raise ValueError(f"{name}: wrong gene panel")
    if manifest["training_genes"] != data.genes:
        raise ValueError(f"{name}: training input must use the same ordered 2,000 genes")
    expected_conditions = CONDITIONS if data.split is not None and name != "CellOracle" else [CONDITIONS[0]]
    if manifest["training_conditions"] != expected_conditions:
        raise ValueError(f"{name}: training conditions differ from the benchmark contract")
    expected_ko_usage = len(expected_conditions) == 2
    if manifest["used_evaluation_cells"] is not False or manifest["used_atf6_ko_expression"] is not expected_ko_usage:
        raise ValueError(f"{name}: training/evaluation usage differs from the benchmark contract")
    if data.split is not None:
        for key in ["training_barcodes", "validation_barcodes", "query_barcodes"]:
            if key not in manifest or not isinstance(manifest[key], list):
                raise ValueError(f"{name}: missing {key}")
        if name == "CellOracle":
            expected_training = data.split["control_training_barcodes"]
            expected_validation = []
        else:
            expected_training = data.split["training_barcodes"]
            expected_validation = data.split["validation_barcodes"]
        if manifest["training_barcodes"] != expected_training or manifest["validation_barcodes"] != expected_validation:
            raise ValueError(f"{name}: model fitted/validated on the wrong cells")
        if manifest["query_barcodes"] != data.split["query_barcodes"]:
            raise ValueError(f"{name}: prediction must use common held-out control query cells")
    if not isinstance(manifest["training_provenance"], str) or not manifest["training_provenance"].strip():
        raise ValueError(f"{name}: nonempty training/pretraining provenance is required")
    if not isinstance(manifest["model_version"], str) or not manifest["model_version"].strip():
        raise ValueError(f"{name}: nonempty model version is required")
    prediction = ad.read_h5ad(prediction_path)
    if not prediction.var_names.is_unique or not prediction.obs_names.is_unique:
        raise ValueError(f"{name}: duplicate gene or cell identifiers")
    if prediction.n_vars != N_GENES or set(prediction.var_names) != set(data.genes):
        raise ValueError(f"{name}: require all 2,000 genes; no intersections or zero filling")
    if set(prediction.obs_names).intersection(data.real_barcodes):
        raise ValueError(f"{name}: prediction IDs overlap measured KO IDs")
    values = dense_expression(prediction[:, data.genes].X)
    check_expression(values, name)
    if data.split is not None and len(values) != len(data.control):
        raise ValueError(f"{name}: prediction count differs from common control query")
    if data.split is not None:
        if "source_control_barcode" not in prediction.obs:
            raise ValueError(f"{name}: missing source control barcode mapping")
        if prediction.obs["source_control_barcode"].tolist() != data.split["query_barcodes"]:
            raise ValueError(f"{name}: prediction row mapping differs from common query order")
    return values, manifest


def evaluate_expression(prediction: np.ndarray, measured: np.ndarray) -> dict:
    check_expression(prediction, "Prediction")
    check_expression(measured, "Measured KO")
    cosine = np.clip(
        cosine_similarity(prediction.astype(np.float64), measured.astype(np.float64)), 0, 1,
    ).ravel()
    residual = prediction.mean(axis=0) - measured.mean(axis=0)
    return {
        "cosine": cosine,
        "residual": residual,
        "n_predicted_cells": len(prediction),
        "n_measured_cells": len(measured),
        "n_genes": N_GENES,
        "n_cosine_pairs": len(cosine),
        "cosine_mean": float(cosine.mean()),
        "cosine_median": float(np.median(cosine)),
        "gene_MAE": float(np.abs(residual).mean()),
        "gene_RMSE": float(np.sqrt(np.mean(residual ** 2))),
        "gene_mean_residual": float(residual.mean()),
    }


def run_benchmark(
    data: BenchmarkData, output_dir: Path, external_results: list[dict], seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, plt.Figure]:
    control_values = dense_expression(data.control.X)
    sample_size = len(data.control) if data.split is not None else len(data.reference)
    if len(control_values) < sample_size:
        raise ValueError("Not enough control cells to match the archived virtual-cell count")
    control_indices = np.random.default_rng(seed).choice(len(control_values), sample_size, replace=False)
    no_change = control_values[control_indices].copy()
    atf6_zero = no_change.copy()
    atf6_zero[:, data.genes.index("ATF6")] = 0
    predictions = {
        "Control / no change": (no_change, "sanity_baseline"),
        "ATF6 zero only": (atf6_zero, "sanity_baseline"),
    }
    catalog = candidate_catalog(training_allowed=data.split is not None)
    manifests = {}
    for entry in external_results:
        name = entry["method"]
        if name in predictions or name == "DQN Combined [reference]":
            raise ValueError(f"Duplicate/reserved method name: {name}")
        if name not in catalog["method"].values:
            raise ValueError(f"Unknown candidate: {name}")
        if name == "scTenifoldKnk":
            raise ValueError("scTenifoldKnk regulation scores cannot be evaluated as expression")
        values, manifest = load_external_prediction(
            name, Path(entry["prediction_path"]), Path(entry["manifest_path"]), data,
        )
        role = (
            "heldout_control_only_GRN_predictor" if name == "CellOracle" and data.split is not None
            else "heldout_seen_ATF6_predictor" if data.split is not None
            else "external_prediction_declared_control_only"
        )
        predictions[name] = (values, role)
        manifests[name] = manifest
        catalog.loc[catalog["method"].eq(name), ["status", "reason"]] = [
            "evaluated_supplied_prediction", "Actual prediction evaluated; split and barcode mapping validated.",
        ]
    predictions["DQN Combined [reference]"] = (data.reference, "reference_only_KO_informed")
    evaluated = {}
    rows = []
    for name, (values, role) in predictions.items():
        metrics = evaluate_expression(values, data.measured)
        evaluated[name] = metrics
        rows.append({
            "method": name, "role": role,
            **{key: value for key, value in metrics.items() if key not in {"cosine", "residual"}},
        })
    np.testing.assert_allclose(
        evaluated["DQN Combined [reference]"]["cosine"],
        data.reference_cosine.ravel(), rtol=1e-5, atol=1e-6,
    )
    summary = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(max(12, 2.5 * len(evaluated)), 5.5))
    colors = ["#4C78A8", "#72B7B2"] + ["#F2CF5B"] * len(manifests) + ["#756BB1"]
    for axis, key, title, ylabel in [
        (axes[0], "cosine", "All-pairs cell-wise cosine", "Cosine similarity"),
        (axes[1], "residual", "Gene-wise mean expression residual", "Mean predicted KO - mean measured KO"),
    ]:
        boxes = axis.boxplot(
            [metrics[key] for metrics in evaluated.values()],
            tick_labels=list(evaluated), patch_artist=True, showfliers=False,
            showmeans=True, meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"},
            medianprops={"color": "black"},
        )
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        boxes["boxes"][-1].set_hatch("//")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", labelrotation=18)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0, 1)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    fig.suptitle("Validation 2: Three Density Bands Combined\nDiamonds = means; hatched DQN = KO-informed reference, not an independent predictor")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "benchmark_summary.csv", index=False)
    catalog.to_csv(output_dir / "candidate_status.csv", index=False)
    pd.DataFrame({
        "gene": data.genes,
        "measured_KO_mean": data.measured.mean(axis=0),
        **{name: values.mean(axis=0) for name, (values, _) in predictions.items()},
    }).to_csv(output_dir / "gene_mean_expression.csv", index=False)
    pd.DataFrame({
        "gene": data.genes,
        **{name: metrics["residual"] for name, metrics in evaluated.items()},
    }).to_csv(output_dir / "gene_mean_residuals.csv", index=False)
    pd.DataFrame({"gene": data.genes}).to_csv(output_dir / "fixed_genes_2000.csv", index=False)
    pd.DataFrame({"barcode": data.real_barcodes}).to_csv(output_dir / "measured_KO_barcodes.csv", index=False)
    pd.DataFrame({"barcode": data.control.obs_names[control_indices]}).to_csv(
        output_dir / "baseline_control_barcodes.csv", index=False,
    )
    np.savez_compressed(
        output_dir / "boxplot_distributions.npz",
        methods=np.asarray(list(evaluated)),
        **{f"cosine_{i}": metrics["cosine"] for i, metrics in enumerate(evaluated.values())},
        **{f"residual_{i}": metrics["residual"] for i, metrics in enumerate(evaluated.values())},
    )
    provenance = {
        "data_url": DATA_URL, "conditions": CONDITIONS, "seed": seed,
        "expression_space": EXPRESSION_SPACE, "gene_panel_sha256": data.gene_hash,
        "n_genes": N_GENES, "measured_KO_cells": len(data.measured),
        "density_band_counts": np.bincount(data.density_band).tolist(),
        "combined_weighting": "all archived virtual cells, not equal band means",
        "cosine_definition": "all predicted x measured cell pairs; pairs are not independent",
        "gene_panel_leakage": "archived KO/control DEG-informed fixed evaluation panel",
        "DQN_role": "KO-informed reference only, excluded from independent predictor ranking",
        "external_training_provenance": "self-declared, not independently audited",
        "external_manifests": manifests,
        "shared_split": data.split,
    }
    (output_dir / "benchmark_provenance.json").write_text(json.dumps(provenance, indent=2))
    fig.savefig(output_dir / "validation2_tool_comparison.png", dpi=180, bbox_inches="tight")
    return summary, catalog, fig


def run_five_metric_benchmark(
    data: BenchmarkData, output_dir: Path, external_results: list[dict],
    dqn_provenance: dict, seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, plt.Figure]:
    """Reproduce DQN dashboard metrics 1, 6, 7, 8 and 12 on common measured cells."""
    summary, catalog, previous_figure = run_benchmark(data, output_dir, external_results, seed)
    plt.close(previous_figure)
    means = pd.read_csv(output_dir / "gene_mean_expression.csv", float_precision="round_trip")
    measured_mean = means["measured_KO_mean"].to_numpy()
    correlations = []
    for method in summary["method"]:
        predicted_mean = means[method].to_numpy()
        if np.ptp(predicted_mean) == 0 or np.ptp(measured_mean) == 0:
            raise ValueError(f"{method}: Pearson correlation is undefined for constant gene means")
        correlations.append(float(pearsonr(predicted_mean, measured_mean).statistic))
    summary["gene_Pearson_r"] = correlations
    summary.to_csv(output_dir / "benchmark_summary.csv", index=False)
    summary.to_csv(output_dir / "five_metric_summary.csv", index=False)
    names = summary["method"].tolist()
    labels = [
        "Control\n(no change)" if name == "Control / no change"
        else "ATF6\nzero only" if name == "ATF6 zero only"
        else "DQN Combined\n[KO-informed reference]" if name == "DQN Combined [reference]"
        else name
        for name in names
    ]
    palette = {
        "Control / no change": "#9E9E9E", "ATF6 zero only": "#72B7B2",
        "scGen": "#4C78A8", "CPA": "#F2A541", "CellOracle": "#54A24B",
        "DQN Combined [reference]": "#756BB1",
    }
    colors = [palette[name] for name in names]
    with np.load(output_dir / "boxplot_distributions.npz", allow_pickle=False) as archive:
        if archive["methods"].tolist() != names:
            raise ValueError("Saved boxplot method order differs from metrics")
        cosines = [archive[f"cosine_{i}"].copy() for i in range(len(names))]
        residuals = [archive[f"residual_{i}"].copy() for i in range(len(names))]
    fig, axes = plt.subplots(2, 3, figsize=(23, 11))
    for axis, values, title, ylabel in [
        (axes[0, 0], cosines, "1. Cell-wise Cosine Similarity", "All predicted x measured cell pairs"),
        (axes[0, 1], residuals, "2. Gene-wise Residuals", "Mean predicted KO - mean measured KO"),
    ]:
        boxes = axis.boxplot(
            values, tick_labels=labels, patch_artist=True, showfliers=False,
            showmeans=True, medianprops={"color": "black"},
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"},
        )
        for i, (patch, color) in enumerate(zip(boxes["boxes"], colors)):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            if summary.iloc[i]["role"].startswith("reference_only"):
                patch.set_hatch("//")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", labelsize=9)
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylim(-0.12, 1)
    axes[0, 0].set_yticks(np.linspace(0, 1, 6))
    residual_lower, residual_upper = axes[0, 1].get_ylim()
    axes[0, 1].set_ylim(
        residual_lower - 0.18 * (residual_upper - residual_lower), residual_upper,
    )
    for axis, values, precision in [
        (axes[0, 0], cosines, 4), (axes[0, 1], residuals, 6),
    ]:
        for i, distribution in enumerate(values):
            axis.text(
                i + 1, 0.025, f"Median\n{np.median(distribution):.{precision}f}",
                transform=axis.get_xaxis_transform(), ha="center", va="bottom",
                color="red", fontsize=9,
            )
    axes[0, 1].axhline(0, color="black", linestyle="--", linewidth=1)
    x = np.arange(len(names))
    bars = axes[0, 2].bar(x, summary["gene_Pearson_r"], color=colors, alpha=0.8)
    for i, bar in enumerate(bars):
        if summary.iloc[i]["role"].startswith("reference_only"):
            bar.set_hatch("//")
        axes[0, 2].annotate(
            f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color="red",
        )
    axes[0, 2].set(
        title="3. Pearson Correlation", ylabel="Pearson r across 2,000 gene means",
        ylim=(-1, 1.12), xticks=x, xticklabels=labels,
    )
    axes[0, 2].tick_params(axis="x", labelsize=9)
    axes[0, 2].axhline(0, color="black", linewidth=0.8)
    axes[0, 2].grid(axis="y", alpha=0.2)
    width = 0.36
    for shift, metric, color, label in [
        (-width / 2, "gene_MAE", "#4C78A8", "MAE"),
        (width / 2, "gene_RMSE", "#E45756", "RMSE"),
    ]:
        bars = axes[1, 0].bar(x + shift, summary[metric], width, color=color, label=label, alpha=0.8)
        for i, bar in enumerate(bars):
            if summary.iloc[i]["role"].startswith("reference_only"):
                bar.set_hatch("//")
            axes[1, 0].annotate(
                f"{bar.get_height():.4f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points", ha="center",
                fontsize=8, color="red",
            )
    axes[1, 0].set(
        title="4. Expression Prediction Error", ylabel="Error across 2,000 gene means",
        xticks=x, xticklabels=labels,
    )
    axes[1, 0].tick_params(axis="x", labelsize=9)
    axes[1, 0].legend()
    axes[1, 0].margins(y=0.15)
    axes[1, 0].grid(axis="y", alpha=0.2)
    bins = np.linspace(0, 1, 51)
    histogram_rows = []
    for name, values, color in zip(names, cosines, colors):
        density, edges, _ = axes[1, 1].hist(
            values, bins=bins, density=True, histtype="step", linewidth=1.7,
            color=color, label=name,
        )
        counts, _ = np.histogram(values, bins)
        for j in range(len(counts)):
            histogram_rows.append({
                "method": name, "bin_left": edges[j], "bin_right": edges[j + 1],
                "count": int(counts[j]), "probability_density": density[j],
            })
    axes[1, 1].set(
        title="5. Cosine Similarity Distributions", xlabel="Cosine similarity",
        ylabel="Probability density", xlim=(0, 1),
    )
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(axis="y", alpha=0.2)
    axes[1, 2].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "five_metric_comparison.png", dpi=180, bbox_inches="tight")
    pd.DataFrame(histogram_rows).to_csv(output_dir / "cosine_histogram_50bins.csv", index=False)
    pd.DataFrame({
        "metric_number": [1, 6, 7, 8, 12],
        "figure_panel_number": [1, 2, 3, 4, 5],
        "definition": [
            "All predicted x measured cell-pair cosine over the fixed gene panel",
            "Per-gene mean(predicted) - mean(measured KO)",
            "Pearson correlation of predicted and measured gene mean vectors",
            "MAE and RMSE of the per-gene mean residual vector",
            "Probability-density histogram of metric 1; 50 shared bins over [0, 1]",
        ],
    }).to_csv(output_dir / "metric_definitions.csv", index=False)
    provenance_path = output_dir / "benchmark_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["latest_DQN"] = dqn_provenance
    provenance["metric_numbers"] = [1, 6, 7, 8, 12]
    provenance["figure_panel_numbers"] = [1, 2, 3, 4, 5]
    provenance_path.write_text(json.dumps(provenance, indent=2))
    return summary, catalog, fig
