"""Run official perturbation models on an exported, shared ATF6 cell split."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import random
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


CONTROL = "3x_neg_ctrl_pMJ144-1"
KO = "ATF6_only_pMJ145"
SPACE = "log1p_CP10K_full_gene_library"


def dense(matrix):
    return np.asarray(matrix.toarray() if sparse.issparse(matrix) else matrix, dtype=np.float32)


def read_inputs(input_dir):
    split = json.loads((input_dir / "split.json").read_text())
    train = ad.read_h5ad(input_dir / "train.h5ad")
    valid = ad.read_h5ad(input_dir / "valid.h5ad")
    query = ad.read_h5ad(input_dir / "query_control.h5ad")
    for name, data, ids in [
        ("train", train, split["training_barcodes"]),
        ("valid", valid, split["validation_barcodes"]),
        ("query", query, split["query_barcodes"]),
    ]:
        if data.obs_names.tolist() != ids or data.var_names.tolist() != split["genes"]:
            raise ValueError(f"{name}: split identities or gene order differ")
        if not np.isfinite(dense(data.X)).all():
            raise ValueError(f"{name}: non-finite input")
    if set(train.obs_names) & set(query.obs_names) or set(valid.obs_names) & set(query.obs_names):
        raise ValueError("Query cells overlap fitted/validation cells")
    return split, train, valid, query


def scgen_predict(train, valid, query, output_dir, epochs, seed):
    import scgen
    import scvi
    import torch
    from scvi.dataloaders import DataSplitter

    scvi.settings.seed = seed
    torch.set_num_threads(4)
    pool = ad.concat([train, valid], merge="same")
    pool.X = dense(pool.X)
    scgen.SCGEN.setup_anndata(pool, batch_key="condition", labels_key="cell_type")
    model = scgen.SCGEN(pool, n_hidden=256, n_latent=32, n_layers=2)
    class FixedSplitter(DataSplitter):
        def setup(self, stage=None):
            super().setup(stage)
            self.train_idx = np.arange(train.n_obs)
            self.val_idx = np.arange(train.n_obs, pool.n_obs)
            self.test_idx = np.array([], dtype=int)

    model._data_splitter_cls = FixedSplitter
    model.train(
        max_epochs=epochs, use_gpu=False, batch_size=128,
        train_size=train.n_obs / pool.n_obs, validation_size=valid.n_obs / pool.n_obs,
        early_stopping=False,
        enable_progress_bar=False, check_val_every_n_epoch=1,
    )
    if set(pool.obs_names[model.train_indices]) != set(train.obs_names):
        raise ValueError("scGen training split differs from exported train cells")
    if set(pool.obs_names[model.validation_indices]) != set(valid.obs_names):
        raise ValueError("scGen validation split differs from exported validation cells")
    model.save(str(output_dir / "model"), overwrite=True, save_anndata=True)
    # Predict's latent arithmetic is restricted to training cells, not validation or test KO.
    model.to_device("cpu")
    prediction, delta = model.predict(
        ctrl_key="ctrl", stim_key="ATF6", adata_to_predict=query.copy(),
        restrict_arithmetic_to={"benchmark_split": ["train"]},
    )
    np.save(output_dir / "latent_ATF6_shift.npy", delta)
    return dense(prediction.X), {
        "model_version": importlib.metadata.version("scgen"),
        "epochs_completed": int(len(model.history["elbo_train"])),
        "architecture": {"n_hidden": 256, "n_latent": 32, "n_layers": 2},
        "source_commit": "d79e1f04233c30f9a4eb5b8d57718127909807d7",
        "training_provenance": "Official scgen.SCGEN 2.1.1; scvi DataSplitter adapter enforces common train/valid indices; ATF6-control latent arithmetic uses training cells only.",
    }


def cpa_predict(train, valid, query, output_dir, epochs, seed):
    import cpa
    import scvi
    import torch

    scvi.settings.seed = seed
    torch.set_num_threads(4)
    pool = ad.concat([train, valid], merge="same")
    pool.X = dense(pool.X)
    cpa.CPA.setup_anndata(
        pool, perturbation_key="condition", control_group="ctrl",
        dosage_key="dose", is_count_data=False,
        categorical_covariate_keys=["cell_type"], max_comb_len=1,
    )
    model = cpa.CPA(
        pool, split_key="benchmark_split", train_split="train",
        valid_split="valid", test_split="unused",
        n_latent=32, n_hidden_encoder=256, n_layers_encoder=2,
        n_hidden_decoder=256, n_layers_decoder=2, recon_loss="gauss",
        seed=seed,
    )
    model.train(
        max_epochs=epochs, use_gpu=False, batch_size=128,
        early_stopping=False, check_val_every_n_epoch=1,
        enable_progress_bar=False, save_path=False,
        early_stopping_patience=epochs + 1,
    )
    if set(pool.obs_names[model.train_indices]) != set(train.obs_names):
        raise ValueError("CPA training split differs from exported train cells")
    if set(pool.obs_names[model.valid_indices]) != set(valid.obs_names):
        raise ValueError("CPA validation split differs from exported validation cells")
    model.save(str(output_dir / "model"), overwrite=True, save_anndata=True)
    counterfactual = query.copy()
    counterfactual.obs["condition"] = "ATF6"
    cpa.CPA.setup_anndata(
        counterfactual, perturbation_key="condition", control_group="ctrl",
        dosage_key="dose", is_count_data=False,
        categorical_covariate_keys=["cell_type"], max_comb_len=1,
    )
    model.predict(counterfactual, n_samples=1, return_mean=True)
    return dense(counterfactual.obsm["CPA_pred"]), {
        "model_version": importlib.metadata.version("cpa-tools"),
        "epochs_completed": int(model.runner.trainer.current_epoch),
        "architecture": {"n_latent": 32, "n_hidden": 256, "n_layers": 2, "recon_loss": "gauss"},
        "training_provenance": "Official cpa.CPA; common train/valid split; counterfactual X contains held-out control expression, target condition ATF6.",
    }


def celloracle_predict(train, query, output_dir, seed):
    import celloracle as co
    from sklearn.neighbors import NearestNeighbors

    control_train = train[train.obs["perturbation"].eq(CONTROL)].copy()
    control_train.X = np.expm1(dense(control_train.X))
    control_train.uns.pop("log1p", None)
    control_train.obs["grn_cluster"] = pd.Categorical(["control"] * control_train.n_obs)
    control_train.obsm["X_display"] = np.zeros((control_train.n_obs, 2), dtype=np.float32)
    class FixedPanelOracle(co.Oracle):
        def score_cv_vs_mean(self, N=3000, **kwargs):
            values = dense(self.adata.layers["raw_count"])
            minimum_cells = kwargs.get("min_expr_cells", 2)
            max_mean = kwargs.get("max_expr_avg", 20)
            min_mean = kwargs.get("min_expr_avg", 0)
            eligible = (
                ((values > 0).sum(axis=0) > minimum_cells)
                & (values.mean(axis=0) < max_mean)
                & (values.mean(axis=0) > min_mean)
            )
            available = int(eligible.sum())
            if available < 2:
                raise ValueError("Too few control-expressed genes for CellOracle CV QC")
            requested = min(N, available - 1)
            print(f"CellOracle CV QC rank: {N} -> {requested}; eligible genes: {available}; fixed 2000-gene panel retained")
            return super().score_cv_vs_mean(N=requested, **kwargs)

    oracle = FixedPanelOracle()
    oracle.import_anndata_as_raw_count(
        control_train, cluster_column_name="grn_cluster", embedding_name="X_display",
        transform="natural_log",
    )
    prior_version = "hg38_gimmemotifsv5_fpr2"
    prior = co.data.load_human_promoter_base_GRN(version=prior_version)
    prior = prior[prior["gene_short_name"].isin(train.var_names)].copy()
    tf_columns = [column for column in prior.columns if column in train.var_names]
    prior = prior[["peak_id", "gene_short_name"] + tf_columns]
    if "ATF6" not in tf_columns or not (prior["ATF6"] > 0).any():
        raise ValueError("Public promoter prior has no ATF6 targets in the fixed gene panel")
    oracle.import_TF_data(TF_info_matrix=prior)
    oracle.perform_PCA(n_components=30)
    oracle.knn_imputation(k=30, n_pca_dims=30, n_jobs=4)
    oracle.fit_GRN_for_simulation(GRN_unit="whole", alpha=1, verbose_level=0)
    if "ATF6" not in oracle.active_regulatory_genes:
        raise ValueError("Fitted control GRN has no active outgoing ATF6 connections")
    oracle.coef_matrix.to_csv(output_dir / "fitted_GRN_coefficients.csv")
    train_normalized = dense(oracle.adata.layers["normalized_count"])
    import joblib
    joblib.dump(
        {"coef_matrix": oracle.coef_matrix, "pca": oracle.pca,
         "train_normalized": train_normalized, "prior_version": prior_version},
        output_dir / "fitted_model.joblib",
    )
    query_normalized = dense(query.X)
    neighbors = NearestNeighbors(n_neighbors=30, n_jobs=4).fit(
        oracle.pca.transform(train_normalized),
    )
    indices = neighbors.kneighbors(oracle.pca.transform(query_normalized), return_distance=False)
    query_imputed = train_normalized[indices].mean(axis=1)
    query_adata = query.copy()
    query_adata.layers["raw_count"] = np.expm1(query_normalized)
    query_adata.layers["normalized_count"] = query_normalized
    query_adata.layers["imputed_count"] = query_imputed
    query_adata.obs["grn_cluster"] = pd.Categorical(["control"] * query.n_obs)
    query_adata.obsm["X_display"] = np.zeros((query.n_obs, 2), dtype=np.float32)
    oracle.adata = query_adata
    oracle.simulate_shift(perturb_condition={"ATF6": 0.0}, GRN_unit="whole", n_propagation=3)
    prediction = dense(oracle.adata.layers["simulated_count"])
    oracle.adata.write_h5ad(output_dir / "query_simulation.h5ad")
    return prediction, {
        "model_version": importlib.metadata.version("celloracle"),
        "prior_version": prior_version, "alpha": 1, "n_propagation": 3, "k_imputation": 30,
        "prior_genes_covered": int(prior["gene_short_name"].nunique()),
        "ATF6_nonzero_GRN_edges": int((oracle.coef_matrix.loc["ATF6"].abs() > 0).sum()),
        "ATF6_genes_changed": int((np.abs(prediction - query_imputed).max(axis=0) > 1e-8).sum()),
        "training_provenance": "Official CellOracle; public hg38 promoter prior; GRN/PCA fitted on training control only. Query imputed from 30 training-control neighbors; simulated_count is log1p CP10K because imported raw_count is full-library CP10K. No KO expression used.",
        "compatibility_adapter": "Cap CV quality-control rank at eligible control-expressed genes to prevent upstream IndexError; does not filter or change the 2000-gene panel.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["scGen", "CPA", "CellOracle"], required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    split, train, valid, query = read_inputs(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    if args.method == "scGen":
        prediction, details = scgen_predict(train, valid, query, args.output_dir, args.epochs, args.seed)
    elif args.method == "CPA":
        prediction, details = cpa_predict(train, valid, query, args.output_dir, args.epochs, args.seed)
    else:
        prediction, details = celloracle_predict(train, query, args.output_dir, args.seed)
    if prediction.shape != query.shape or not np.isfinite(prediction).all():
        raise ValueError(f"{args.method}: invalid prediction shape or non-finite output")
    negative_count = int((prediction < 0).sum())
    print(f"{args.method}: clipping {negative_count}/{prediction.size} negative log-expression values at zero")
    prediction = np.maximum(prediction, 0)
    if (np.linalg.norm(prediction, axis=1) == 0).any():
        raise ValueError(f"{args.method}: all-zero prediction cells")
    obs = pd.DataFrame(index=[f"{args.method}_virtual_{i:04d}" for i in range(query.n_obs)])
    obs["source_control_barcode"] = query.obs_names.to_numpy()
    result = ad.AnnData(prediction, obs=obs, var=query.var.copy())
    result.uns["expression_space"] = SPACE
    result.write_h5ad(args.output_dir / "prediction.h5ad")
    control_only = args.method == "CellOracle"
    manifest = {
        "method": args.method, "target": "ATF6", "expression_space": SPACE,
        "gene_panel_sha256": split["gene_panel_sha256"], "training_genes": split["genes"],
        "training_conditions": [CONTROL] if control_only else [CONTROL, KO],
        "used_evaluation_cells": False, "used_atf6_ko_expression": not control_only,
        "training_barcodes": split["control_training_barcodes"] if control_only else split["training_barcodes"],
        "validation_barcodes": [] if control_only else split["validation_barcodes"],
        "query_barcodes": split["query_barcodes"],
        "seed": args.seed, "negative_values_clipped": negative_count,
        "runtime_seconds": time.monotonic() - start, **details,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: value for key, value in manifest.items() if not isinstance(value, list)}, indent=2))


if __name__ == "__main__":
    main()
