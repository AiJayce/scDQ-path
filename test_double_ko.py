import ast
import json
import tempfile
from pathlib import Path
import unittest

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
import torch

from double_ko import compare_three_conditions, joint_decrease, joint_reward, nonzero_evaluation_mask
from total_benchmark import evaluate_expression


class DoubleKOTests(unittest.TestCase):
    def evaluation_selection_state(self, folder, retained_count=1040):
        notebook = json.loads(Path(__file__).with_name("Double-KO.ipynb").read_text())
        code = next(
            "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code" and "evaluation_source =" in "".join(cell["source"])
        )
        selection_code = code[
            code.index("evaluation_adata = evaluation_adata[evaluation_nonzero].copy()"):
            code.index("DOUBLE_EVAL_DIR = LG_DIR / 'three_condition_five_metrics'", code.index("measured_by_condition, measured_ids"))
        ]
        genes = ["ATF6", "ERN1"] + [f"gene_{i}" for i in range(1998)]
        double_ids = [f"double_{i}" for i in range(1446)]
        single_ids = ["atf6_0", "atf6_1", "ire1_0", "ire1_1"]
        double_condition = "ATF6_IRE1_pMJ152"
        conditions = [double_condition, "ATF6_only_pMJ145", "IRE1_only_pMJ148"]
        values = sparse.csr_matrix(
            np.repeat(np.arange(1, 1451, dtype=np.float32)[:, None], 2000, axis=1),
        )
        population = ad.AnnData(
            values, obs=pd.DataFrame({
                "perturbation": [double_condition] * 1446 + [conditions[1]] * 2 + [conditions[2]] * 2,
            }, index=double_ids + single_ids),
            var=pd.DataFrame(index=genes),
        )
        retained_ids = double_ids[:retained_count][::-1]
        retained = population[retained_ids].copy()
        state = {
            "adata": retained, "evaluation_adata": population,
            "evaluation_nonzero": np.ones(1450, dtype=bool),
            "gs_real_ids": np.array(retained_ids), "gs_genes": genes,
            "gs_real_expression": retained.X.toarray(), "ATF6_CONDITION": double_condition,
            "COMPARISON_CONDITIONS": conditions, "DOUBLE_EVAL_DIR": Path(folder),
            "nonzero_evaluation_mask": nonzero_evaluation_mask,
            "np": np, "pd": pd, "sparse": sparse, "display": lambda value: None,
        }
        return selection_code, state

    def test_final_cell_respects_1040_retained_barcodes_and_order(self):
        with tempfile.TemporaryDirectory() as folder:
            code, state = self.evaluation_selection_state(folder)
            exec(compile(code, "<double-ko-evaluation>", "exec"), state)
            self.assertEqual(len(state["measured_ids"]["ATF6_IRE1_pMJ152"]), 1040)
            self.assertEqual(state["measured_ids"]["ATF6_IRE1_pMJ152"], state["gs_real_ids"].tolist())
            self.assertEqual(len(state["excluded_selection_cells"]), 406)
            self.assertEqual(state["measured_ids"]["ATF6_only_pMJ145"], ["atf6_0", "atf6_1"])
            self.assertEqual(state["measured_ids"]["IRE1_only_pMJ148"], ["ire1_0", "ire1_1"])
            self.assertEqual(len(pd.read_csv(Path(folder) / "excluded_cell_selection.csv")), 406)

    def test_final_cell_preserves_unfiltered_population(self):
        with tempfile.TemporaryDirectory() as folder:
            code, state = self.evaluation_selection_state(folder, retained_count=1446)
            exec(compile(code, "<double-ko-evaluation>", "exec"), state)
            self.assertEqual(len(state["excluded_selection_cells"]), 0)
            self.assertEqual(len(state["measured_ids"]["ATF6_IRE1_pMJ152"]), 1446)

    def test_final_cell_rejects_stale_sampling_after_selection_change(self):
        with tempfile.TemporaryDirectory() as folder:
            code, state = self.evaluation_selection_state(folder)
            state["gs_real_ids"] = np.array([f"double_{i}" for i in range(1446)])
            with self.assertRaisesRegex(ValueError, "Rerun section 14"):
                exec(compile(code, "<double-ko-evaluation>", "exec"), state)

    def test_initial_dqn_chunking_preserves_entropy_and_support(self):
        notebook = json.loads(Path(__file__).with_name("Double-KO.ipynb").read_text())
        code = next(
            "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code" and "def decoded_entropy_t" in "".join(cell["source"])
        )
        definitions = [
            node for node in ast.parse(code).body
            if isinstance(node, ast.FunctionDef) and node.name in {"decoded_entropy_t", "support_distance_t"}
        ]
        generator = torch.Generator().manual_seed(42)
        decoder = torch.nn.Linear(2, 3)
        with torch.no_grad():
            decoder.weight.fill_(0.1)
            decoder.bias.copy_(torch.tensor([1.0, 2.0, 3.0]))
        values = torch.randn(4101, 2, generator=generator)
        observed = torch.randn(8, 2, generator=generator)
        state = {"torch": torch, "rl_decoder": decoder, "LATENT_DIM": 2, "Z_tensor": observed, "REGION_SIZE": 5}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "<double-ko-entropy>", "exec"), state)
        with torch.no_grad():
            linear = torch.expm1(decoder(values).double().clamp_min(0))
            p = linear / linear.sum(1, keepdim=True)
            entropy = -torch.xlogy(p, p).sum(1).float()
            support = torch.cdist(values, observed).topk(5, largest=False).values[:, -1]
        torch.testing.assert_close(state["decoded_entropy_t"](values), entropy)
        torch.testing.assert_close(state["support_distance_t"](values), support)

    def test_zero_panel_filter_is_evaluation_only(self):
        values = np.zeros((3, 2000), np.float32)
        values[1, 0], values[2, 1] = 1.0, 2.0
        original = values.copy()
        self.assertEqual(nonzero_evaluation_mask(values).tolist(), [False, True, True])
        np.testing.assert_array_equal(values, original)
        with self.assertRaises(ValueError):
            nonzero_evaluation_mask(-np.ones((2, 2000)))

    def test_notebook_joint_candidate_features_and_rollout(self):
        notebook = json.loads(Path(__file__).with_name("Double-KO.ipynb").read_text())
        code = next(
            "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code" and "def lg_simulate" in "".join(cell["source"])
        )
        definitions = [
            node for node in ast.parse(code).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in {"lg_measure", "lg_candidates", "lg_features", "lg_reward", "LgTracker", "lg_simulate"}
        ]
        decoder = torch.nn.Linear(2, 3)
        with torch.no_grad():
            decoder.weight.copy_(torch.tensor([[1.0, 1.0], [1.0, -1.0], [0.0, 0.0]]))
            decoder.bias.copy_(torch.tensor([2.0, 4.0, 1.0]))
        state = {
            "torch": torch, "np": np, "joint_decrease": joint_decrease, "joint_reward": joint_reward,
            "lg_decoder": decoder, "lg_target_indices": [0, 1], "lg_dim": 2,
            "lg_cells_t": torch.zeros(6, 2), "lg_start_t": torch.zeros(2),
            "lg_start_z": np.zeros(2, np.float32), "lg_start_pair": torch.tensor([2.0, 4.0]),
            "lg_start_h": 1.0, "lg_start_support": 0.0, "lg_f_scale": torch.tensor([2.0, 4.0]),
            "lg_directions": torch.tensor([[0.01, 0.0], [0.0, 0.01], [-0.01, 0.0], [0.0, -0.01]]),
            "lg_substep_index": torch.tensor([1.0]), "LG_STRIDES": (1,), "lg_support_radius": 1.0,
            "lg_feature_mean": torch.zeros(2), "lg_feature_scale": torch.ones(2),
            "lg_h_mean": 1.0, "lg_h_scale": 1.0, "lg_diversity_scale": 1.0,
            "LG_MAX_MOVES": 5, "LG_MOVE_PATIENCE": 10, "LG_MOVE_MIN": 0.04,
            "LG_DEVICE": torch.device("cpu"), "LG_GAMMA": 0.999,
        }
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "<double-ko-notebook>", "exec"), state)
        start = state["lg_start_t"][None]
        targets = state["lg_start_pair"][None]
        candidates = state["lg_candidates"](start, targets)
        self.assertEqual(candidates["valid"].tolist(), [[False, False, True, False]])
        self.assertEqual(state["lg_features"](start, targets, torch.ones(1), torch.zeros(1)).shape, (1, 9))
        model = torch.nn.Linear(9, 4)
        result = state["lg_simulate"](model, [0.0, 0.3], 42)
        for rollout in result:
            self.assertEqual(len(rollout["actions"]), 5)
            self.assertTrue((np.diff(rollout["target"]) < 0).all())
            self.assertTrue((np.diff(rollout["ire1"]) < 0).all())
            pair = torch.tensor(np.column_stack([rollout["target"], rollout["ire1"]]))
            np.testing.assert_allclose(
                rollout["rewards"], joint_reward(pair[:-1], pair[1:], state["lg_f_scale"]).numpy(),
                rtol=1e-6,
            )

    def test_both_targets_must_decrease(self):
        current = torch.tensor([2.0, 4.0])
        arrival = torch.tensor([[1.0, 3.0], [1.0, 5.0], [3.0, 1.0], [1.0, 4.0], [-1.0, 3.0]])
        self.assertEqual(joint_decrease(current, arrival).tolist(), [True, False, False, False, False])
        self.assertFalse(bool(joint_decrease(current, current - 1e-7)))

    def test_equal_weight_normalized_reward(self):
        current = torch.tensor([[2.0, 4.0], [1.0, 3.0]])
        arrival = torch.tensor([[1.0, 3.0], [0.5, 2.0]])
        start = torch.tensor([2.0, 4.0])
        torch.testing.assert_close(joint_reward(current, arrival, start), torch.tensor([0.375, 0.25]))
        with self.assertRaises(ValueError):
            joint_reward(current, arrival, torch.tensor([0.0, 4.0]))

    def test_three_comparisons_reuse_exact_metrics(self):
        genes = ["ATF6", "ERN1"] + [f"gene_{i}" for i in range(1998)]
        predicted = np.tile(np.linspace(0.1, 2.0, 2000, dtype=np.float32), (2, 1))
        measured = {
            "ATF6_IRE1_pMJ152": predicted * 0.9,
            "ATF6_only_pMJ145": np.repeat(predicted[:1] * 0.8, 3, axis=0),
            "IRE1_only_pMJ148": np.repeat(predicted[:1] * 0.7, 4, axis=0),
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            summary = compare_three_conditions(predicted, measured, genes, path)
            self.assertEqual(summary["n_cosine_pairs"].tolist(), [4, 6, 8])
            for i, values in enumerate(measured.values()):
                expected = evaluate_expression(predicted, values)
                for key in ["cosine_mean", "gene_MAE", "gene_RMSE", "gene_mean_residual"]:
                    self.assertAlmostEqual(summary.iloc[i][key], expected[key])
                self.assertAlmostEqual(
                    summary.iloc[i]["gene_Pearson_r"],
                    pearsonr(predicted.mean(0), values.mean(0)).statistic,
                )
            with np.load(path / "five_metric_distributions.npz") as saved:
                np.testing.assert_array_equal(saved["genes"], genes)
                np.testing.assert_array_equal(saved["cosine_1"], evaluate_expression(predicted, measured["ATF6_only_pMJ145"])["cosine"])
            histogram = pd.read_csv(path / "cosine_histogram_50bins.csv")
            self.assertEqual(histogram.groupby("condition").size().tolist(), [50, 50, 50])
            self.assertEqual(histogram.groupby("condition")["count"].sum().sort_index().tolist(), [4, 6, 8])
            self.assertTrue((path / "five_metric_comparison.png").is_file())
            self.assertEqual(pd.read_csv(path / "metric_definitions.csv")["metric_number"].tolist(), [1, 6, 7, 8, 12])


if __name__ == "__main__":
    unittest.main()
