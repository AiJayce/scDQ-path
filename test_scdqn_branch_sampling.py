import ast
import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scdqn_gaussian import band_counts, sample_density_bands, expression_cosines


NOTEBOOK = Path(__file__).with_name("scDQN.ipynb")


def notebook_functions(names, namespace):
    notebook = json.loads(NOTEBOOK.read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"] if cell["id"] == "treg-long-range-dqn")
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError("Notebook functions not found.")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


class BranchRewardTests(unittest.TestCase):
    def setUp(self):
        self.namespace = {
            "torch": torch, "LG_STRIDES": (1,), "lg_substep_index": torch.tensor([1.]),
            "lg_directions": torch.tensor([[0.01, 0.], [0., 0.01], [-0.01, 0.], [0., -0.01]]),
            "lg_support_radius": 1., "LG_FOXP3_WEIGHT": 1., "lg_f_scale": 2.,
        }

        def measure(z):
            return torch.ones(z.shape[:-1]), 2 - z[..., 0], torch.full(z.shape[:-1], 0.5)

        self.namespace["lg_measure"] = measure
        notebook_functions(["lg_candidates", "lg_reward"], self.namespace)

    def test_only_supported_strict_decrease_is_valid(self):
        candidates = self.namespace["lg_candidates"](torch.zeros(1, 2), torch.tensor([2.]))
        self.assertEqual(candidates["valid"].tolist(), [[True, False, False, False]])
        np.testing.assert_allclose(candidates["z"].norm(dim=-1), 0.01)
        self.namespace["lg_support_radius"] = 0.1
        self.assertFalse(self.namespace["lg_candidates"](torch.zeros(1, 2), torch.tensor([2.]))["valid"].any())

    def test_roundoff_is_not_accepted_as_a_decrease(self):
        self.namespace["lg_directions"] *= 1e-5
        self.assertFalse(self.namespace["lg_candidates"](torch.zeros(1, 2), torch.tensor([2.]))["valid"].any())

    def test_current_recorded_value_controls_mask(self):
        self.assertFalse(self.namespace["lg_candidates"](torch.zeros(1, 2), torch.tensor([1.]))["valid"].any())

    def test_entropy_independent_foxp3_reward(self):
        reward = self.namespace["lg_reward"]
        a = reward(torch.tensor([2.]), torch.tensor([3.]), torch.tensor([1.5]), torch.tensor([4.]))
        b = reward(torch.tensor([2.]), torch.tensor([100.]), torch.tensor([1.5]), torch.tensor([-100.]))
        torch.testing.assert_close(a, torch.tensor([0.25]))
        torch.testing.assert_close(a, b)

    def test_training_horizon_and_exhaustion_are_terminal(self):
        notebook = json.loads(NOTEBOOK.read_text())
        source = next("".join(c["source"]) for c in notebook["cells"] if c["id"] == "treg-long-range-dqn")
        train = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "lg_train")
        terminal = next(n for n in ast.walk(train) if isinstance(n, ast.Assign) and
                        any(isinstance(t, ast.Name) and t.id == "terminal" for t in n.targets))
        code = compile(ast.Expression(terminal.value), str(NOTEBOOK), "eval")
        observed = eval(code, {"next_valid": torch.tensor([[True], [False], [True]]),
                               "t": torch.tensor([499, 1, 500]), "LG_MAX_MOVES": 500})
        self.assertEqual(observed.tolist(), [False, True, True])


class GaussianTests(unittest.TestCase):
    def inputs(self):
        return dict(
            centers=np.array([[0., 0.], [1., 0.], [2., 0.]]),
            observed_sd=np.ones(2), pca_mean=np.zeros(2), components=np.eye(2),
            density=np.tile([.9, 1., .7, .5, .5], (3, 1)),
            x_edges=np.arange(-1.5, 4.), y_edges=np.arange(-1.5, 2.),
            total=100, sigma_sd=0.001, seed=5,
        )

    def test_exact_integer_counts(self):
        np.testing.assert_array_equal(band_counts(100), [68, 27, 5])
        self.assertEqual(band_counts(1202).sum(), 1202)

    def test_actual_sampling_thresholds_projection_and_reproducibility(self):
        inputs = self.inputs()
        z, xy, density, labels, counts, proposed = sample_density_bands(**inputs)
        self.assertEqual(z.shape, (100, 2))
        np.testing.assert_array_equal(counts, [68, 27, 5])
        np.testing.assert_allclose(xy, z)
        self.assertTrue(np.all((density[labels == 0] >= .8) & (density[labels == 0] <= 1)))
        self.assertTrue(np.all((density[labels == 1] >= .6) & (density[labels == 1] < .8)))
        self.assertTrue(np.all((density[labels == 2] >= .4) & (density[labels == 2] < .6)))
        np.testing.assert_array_equal(z, sample_density_bands(**inputs)[0])
        self.assertGreaterEqual(proposed, 100)

    def test_unfilled_bands_fail_without_changing_thresholds(self):
        with self.assertRaisesRegex(RuntimeError, "not filled"):
            sample_density_bands(**self.inputs(), max_candidates=1)

    def test_cosines_are_gene_mean_patterns_and_cell_pairs(self):
        expression = np.array([[1., 0.], [3., 0.], [0., 2.], [1., 1.]])
        means, patterns, cells = expression_cosines(expression, np.array([0, 0, 1, 2]))
        np.testing.assert_array_equal(means, [[2., 0.], [0., 2.], [1., 1.]])
        self.assertEqual(patterns.shape, (3, 3))
        self.assertEqual(cells.shape, (4, 4))
        self.assertEqual(patterns[0, 1], 0)
        self.assertAlmostEqual(patterns[0, 2], 1 / np.sqrt(2))
        np.testing.assert_allclose(np.diag(cells), 1)

    def test_undefined_cosine_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "zero"):
            expression_cosines(np.array([[0., 0.], [1., 0.], [0., 1.]]), np.arange(3))

    def test_notebook_sampling_exports_and_renders_exact_shapes(self):
        import matplotlib.pyplot as plt
        import pandas as pd

        inputs = self.inputs()
        decoder = torch.nn.Sequential(torch.nn.Linear(2, 2000), torch.nn.Softplus())
        notebook = json.loads(NOTEBOOK.read_text())
        code = next("".join(c["source"]) for c in notebook["cells"] if c["id"] == "gaussian-foxp3-density-comparison")
        with tempfile.TemporaryDirectory() as directory:
            namespace = {
                "np": np, "torch": torch, "pd": pd, "plt": plt, "json": json,
                "display": lambda value: None,
                "lg_sims": [{"path": np.vstack([np.zeros(2), inputs["centers"]])}],
                "lg_density": inputs["density"], "lg_x_edges": inputs["x_edges"], "lg_y_edges": inputs["y_edges"],
                "lg_z_sd": inputs["observed_sd"], "lg_pca_mean": inputs["pca_mean"],
                "lg_components": inputs["components"], "lg_decoder": decoder,
                "LG_DEVICE": torch.device("cpu"), "LG_DIR": Path(directory),
                "LG_FOXP3_INDEX": 0, "lg_symbols": np.array(["FOXP3"] + [f"gene{i}" for i in range(1999)]),
                "lg_observed_xy": inputs["centers"], "lg_types": np.array(["CD4 Naive T", "Regulatory T", "Regulatory T"]),
                "lg_completed_reward_version": "foxp3_decrease_only_small_steps_v1",
            }
            try:
                with redirect_stdout(io.StringIO()):
                    exec(compile(code, str(NOTEBOOK), "exec"), namespace)
                output = Path(directory) / "gaussian_density_100"
                self.assertEqual(pd.read_csv(output / "virtual_cells.csv").shape[0], 100)
                self.assertEqual(pd.read_csv(output / "decoded_expression.csv", index_col=0).shape, (100, 2000))
                self.assertEqual(pd.read_csv(output / "band_pattern_cosine.csv", index_col=0).shape, (3, 3))
                self.assertEqual(pd.read_csv(output / "cell_cosine.csv", index_col=0).shape, (100, 100))
                for band, count in enumerate([68, 27, 5], start=1):
                    self.assertEqual(pd.read_csv(output / f"band{band}_cell_cosine.csv", index_col=0).shape, (count, count))
                self.assertGreater((output / "density_band_cosine_heatmaps.png").stat().st_size, 1000)
                self.assertGreater((output / "gaussian_cells_pca.png").stat().st_size, 1000)
            finally:
                plt.close("all")


if __name__ == "__main__":
    unittest.main()
