import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

from total_benchmark import (
    BenchmarkData, CONDITIONS, EXPRESSION_SPACE,
    evaluate_expression, load_external_prediction,
    reuse_supervised_split, run_five_metric_benchmark, validate_dqn_export,
)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.genes = ["ATF6"] + [f"gene_{i}" for i in range(1999)]
        self.values = np.tile(np.arange(1, 2001, dtype=np.float32), (2, 1))
        self.query_ids = ["query_0", "query_1"]
        self.control = ad.AnnData(
            self.values, obs=pd.DataFrame(index=self.query_ids),
            var=pd.DataFrame(index=self.genes),
        )
        self.split = {
            "training_barcodes": ["train_control", "train_KO"],
            "control_training_barcodes": ["train_control"],
            "validation_barcodes": ["valid_control", "valid_KO"],
            "query_barcodes": self.query_ids,
        }
        gene_hash = hashlib.sha256("\n".join(self.genes).encode()).hexdigest()
        self.data = BenchmarkData(
            self.genes, self.control, self.values, self.values,
            np.ones((2, 2)), ["test_KO_0", "test_KO_1"],
            np.array([0, 1]), gene_hash, split=self.split,
        )
        self.manifest = {
            "method": "scGen", "target": "ATF6",
            "expression_space": EXPRESSION_SPACE,
            "gene_panel_sha256": gene_hash, "training_genes": self.genes,
            "training_conditions": CONDITIONS,
            "used_evaluation_cells": False, "used_atf6_ko_expression": True,
            "model_version": "synthetic-test", "training_provenance": "Synthetic test only",
            **{key: value for key, value in self.split.items() if key != "control_training_barcodes"},
        }
        self.prediction = self.control.copy()
        self.prediction.obs_names = ["pred_0", "pred_1"]
        self.prediction.obs["source_control_barcode"] = self.query_ids
        self.folder = tempfile.TemporaryDirectory(prefix="atf6-benchmark-test-")
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)

    def load(self, prediction=None, changes=None):
        if prediction is None:
            prediction = self.prediction
        prediction.write_h5ad(self.path / "prediction.h5ad")
        (self.path / "manifest.json").write_text(json.dumps({**self.manifest, **(changes or {})}))
        return load_external_prediction(
            "scGen", self.path / "prediction.h5ad", self.path / "manifest.json", self.data,
        )[0]

    def test_reorders_full_panel_without_losing_genes(self):
        values = self.load(self.prediction[:, ::-1].copy())
        np.testing.assert_array_equal(values, self.values)

    def test_rejects_leakage_wrong_split_and_units(self):
        cases = [
            {"used_evaluation_cells": True},
            {"used_atf6_ko_expression": False},
            {"training_barcodes": ["test_KO_0"]},
            {"validation_barcodes": ["test_KO_1"]},
            {"query_barcodes": self.query_ids[::-1]},
            {"training_conditions": [CONDITIONS[0]]},
            {"expression_space": "GRN_delta"},
            {"training_genes": self.genes[::-1]},
            {"gene_panel_sha256": "wrong"},
            {"training_provenance": ""},
        ]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.load(changes=changes)

    def test_rejects_missing_genes_and_wrong_query_mapping(self):
        with self.assertRaises(ValueError):
            self.load(self.prediction[:, :-1].copy())
        wrong = self.prediction.copy()
        wrong.obs["source_control_barcode"] = self.query_ids[::-1]
        with self.assertRaises(ValueError):
            self.load(wrong)
        wrong.obs_names = ["test_KO_0", "pred_1"]
        with self.assertRaises(ValueError):
            self.load(wrong)

    def test_exact_all_pairs_and_gene_mean_residual(self):
        values = np.ones((2, 2000), dtype=np.float32)
        metrics = evaluate_expression(values, values)
        self.assertEqual(metrics["n_cosine_pairs"], 4)
        np.testing.assert_allclose(metrics["cosine"], 1, atol=1e-12)
        np.testing.assert_array_equal(metrics["residual"], np.zeros(2000))
        changed = values.copy()
        changed[:, 0] += 1
        metrics = evaluate_expression(changed, values)
        self.assertEqual(metrics["residual"][0], 1)
        self.assertAlmostEqual(metrics["gene_MAE"], 1 / 2000)

    def test_rejects_invalid_expression(self):
        for values in [
            np.ones((2, 1999)), np.zeros((2, 2000)),
            np.full((2, 2000), np.nan), -np.ones((2, 2000)),
        ]:
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                evaluate_expression(values, self.values)

    def test_reuses_original_split_without_rewriting_input(self):
        split = {
            **self.split, "genes": self.genes,
            "gene_panel_sha256": self.data.gene_hash, "test_KO_barcodes": self.data.real_barcodes[::-1],
        }
        self.control.obs["perturbation"] = CONDITIONS[0]
        self.control.write_h5ad(self.path / "query_control.h5ad")
        split_path = self.path / "split.json"
        split_path.write_text(json.dumps(split))
        before = split_path.read_bytes()
        result = reuse_supervised_split(self.data, self.path)
        self.assertEqual(before, split_path.read_bytes())
        self.assertEqual(result.real_barcodes, self.data.real_barcodes[::-1])
        np.testing.assert_array_equal(result.measured, self.values[::-1])
        split["genes"] = self.genes[::-1]
        split_path.write_text(json.dumps(split))
        with self.assertRaises(ValueError):
            reuse_supervised_split(self.data, self.path)

    def test_dqn_algorithm_version_mismatch_rejected(self):
        reference = self.path / "gaussian_ko_comparison"
        reference.mkdir()
        notebook = self.path / "DQN.ipynb"
        notebook.write_text(json.dumps({
            "cells": [{"cell_type": "code", "source": ["LG_REWARD_VERSION = 'new_version'"]}],
        }))
        (self.path / "run_metadata.json").write_text(json.dumps({"reward_version": "old_version"}))
        with self.assertRaises(ValueError):
            validate_dqn_export(reference, notebook)
        (self.path / "run_metadata.json").write_text(json.dumps({"reward_version": "new_version"}))
        np.savez(reference / "virtual_vs_real_KO.npz", value=np.ones(2))
        (reference / "sampling_metadata.json").write_text(json.dumps({"counts": [1, 1, 1]}))
        provenance = validate_dqn_export(reference, notebook)
        self.assertEqual(provenance["reward_version"], "new_version")
        self.assertEqual(len(provenance["export_sha256"]), 64)

    def test_five_metrics_use_combined_cells_and_shared_histogram_bins(self):
        measured = self.values.copy()
        virtual = np.vstack([self.values[0], self.values[0] * 2, self.values[0] * 4])
        virtual[2, 0] += 100
        data = BenchmarkData(
            self.genes, self.control, measured, virtual,
            cosine_similarity(virtual, measured), self.data.real_barcodes,
            np.array([0, 0, 1]), self.data.gene_hash, split={},
        )
        summary, _, figure = run_five_metric_benchmark(
            data, self.path / "evaluation", [],
            {"reward_version": "synthetic_combined_test"},
        )
        self.addCleanup(plt.close, figure)
        self.assertEqual(len(figure.axes), 6)
        self.assertIsNone(figure._suptitle)
        self.assertEqual(len(figure.axes[5].texts), 0)
        for index, number in enumerate([1, 2, 3, 4, 5]):
            self.assertTrue(figure.axes[index].get_title().startswith(f"{number}. "))
        for axis_index in [0, 1, 2, 3]:
            self.assertTrue(figure.axes[axis_index].texts)
            self.assertTrue(all(text.get_color() == "red" for text in figure.axes[axis_index].texts))
        for axis_index in [0, 1]:
            self.assertEqual(len(figure.axes[axis_index].texts), len(summary))
        for annotation, median in zip(figure.axes[0].texts, summary["cosine_median"]):
            self.assertEqual(annotation.get_text(), f"Median\n{median:.4f}")
        row = summary.loc[summary["method"].eq("DQN Combined [reference]")].iloc[0]
        residual = virtual.mean(axis=0) - measured.mean(axis=0)
        self.assertAlmostEqual(row["gene_MAE"], np.abs(residual).mean())
        self.assertAlmostEqual(row["gene_RMSE"], np.sqrt(np.mean(residual ** 2)))
        self.assertAlmostEqual(
            row["gene_Pearson_r"], pearsonr(virtual.mean(axis=0), measured.mean(axis=0)).statistic,
            places=6,
        )
        self.assertEqual(row["n_cosine_pairs"], 6)
        hist = pd.read_csv(self.path / "evaluation/cosine_histogram_50bins.csv")
        for method, group in hist.groupby("method"):
            self.assertEqual(len(group), 50)
            self.assertEqual(group["count"].sum(), summary.loc[summary["method"].eq(method), "n_cosine_pairs"].iloc[0])
            self.assertAlmostEqual(
                (group["probability_density"] * (group["bin_right"] - group["bin_left"])).sum(), 1,
            )
        self.assertTrue((self.path / "evaluation/five_metric_comparison.png").is_file())


if __name__ == "__main__":
    unittest.main()
