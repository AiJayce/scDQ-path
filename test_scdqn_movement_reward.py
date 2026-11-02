"""Exercise the actual notebook environment functions without training the VAE."""

import ast
import json
from pathlib import Path
import unittest

import torch


class MovementRewardTests(unittest.TestCase):
    def setUp(self):
        notebook = json.loads(Path(__file__).with_name("scDQN.ipynb").read_text())
        source = "".join(next(
            cell["source"] for cell in notebook["cells"] if cell.get("id") == "medoid-environment"
        ))
        names = {
            "candidate_moves", "action_mask", "short_latent_move_reward",
            "transition_reward_components", "distance_stop_state", "distance_stop_status",
            "environment_step",
        }
        nodes = [
            node for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        self.assertEqual(len(nodes), len(names))
        axes = torch.eye(16)
        goal = torch.zeros(16)
        goal[0] = 1
        self.context = {
            "torch": torch, "DEVICE": torch.device("cpu"), "LATENT_DIM": 16,
            "latent_axes": axes, "z_goal_t": goal, "RL_STEP_LENGTH": 0.004,
            "RL_MAX_STEPS": 500, "RL_MOVE_FRACTIONS": (1.0, 0.5),
            "N_DIRECTIONS": 34, "N_ACTIONS": 68, "rl_support_radius": 10,
            "RL_SHORT_MOVE_WEIGHT": 1.0, "RL_PROGRESS_WEIGHT": 1.0,
            "RL_GOAL_BONUS": 1.0, "RL_STEP_COST": 0.03,
            "RL_DISTANCE_PATIENCE": 20, "RL_DISTANCE_MIN_DELTA": 0.0004,
            "support_distance_t": lambda z: torch.zeros(z.numel() // 16),
            "decoded_entropy_t": lambda z: torch.full((len(z),), 3.0),
        }
        module = ast.Module(body=nodes, type_ignores=[])
        exec(compile(module, "scDQN.ipynb:medoid-environment", "exec"), self.context)

    def function(self, name):
        return self.context[name]

    def test_two_lengths_and_nonzero_actions(self):
        moves, valid = self.function("candidate_moves")(torch.zeros(1, 16), torch.tensor([0]))
        self.assertEqual(moves.shape, (1, 68, 16))
        torch.testing.assert_close(moves[:, :34].norm(dim=2), torch.full((1, 34), 0.004))
        torch.testing.assert_close(moves[:, 34:].norm(dim=2), torch.full((1, 34), 0.002))
        self.assertTrue(valid.all())

    def test_short_move_score_and_equal_weight_tradeoff(self):
        reward = self.function("short_latent_move_reward")
        torch.testing.assert_close(
            reward(torch.tensor([0.001, 0.002, 0.004])), torch.tensor([0.75, 0.5, 0]),
        )
        components = self.function("transition_reward_components")
        for length in [0.004, 0.002]:
            progress, short, goal = components(
                torch.tensor([1.0]), torch.tensor([1 - length]), torch.tensor([length]),
                torch.tensor([False]),
            )
            self.assertAlmostEqual(float(progress + short), 1, places=4)
            self.assertEqual(float(goal), 0)
        for invalid in [torch.tensor([-1.0]), torch.tensor([float("nan")]), torch.tensor([0.005])]:
            with self.assertRaises(ValueError):
                reward(invalid)

    def test_entropy_does_not_change_transition_reward(self):
        z, t = torch.zeros(1, 16), torch.tensor([0])
        _, moves, _ = self.function("action_mask")(z, t)
        rewards = []
        for entropy in [0.1, 100.0]:
            self.context["decoded_entropy_t"] = lambda z, value=entropy: torch.full((len(z),), value)
            history = torch.ones(21, 1)
            result = self.function("environment_step")(
                z, torch.tensor([entropy]), t, torch.tensor([34]), moves, history,
            )
            rewards.append(result[4])
            self.assertAlmostEqual(float(result[1]), entropy, places=4)
            self.assertFalse(bool(result[5]))
        torch.testing.assert_close(rewards[0], rewards[1])
        self.assertAlmostEqual(float(rewards[0]), 0.97, places=4)

    def test_goal_snap_actual_displacement_and_bonus(self):
        z = self.context["z_goal_t"][None].clone()
        z[0, 0] -= 0.001
        t = torch.tensor([100])
        mask, moves, _ = self.function("action_mask")(z, t)
        self.assertEqual(mask.sum().item(), 1)
        self.assertTrue(mask[0, 0])
        result = self.function("environment_step")(
            z, torch.tensor([3.0]), t, torch.tensor([0]), moves, torch.full((21, 1), 0.001),
        )
        torch.testing.assert_close(result[0][0], self.context["z_goal_t"])
        self.assertTrue(bool(result[5]))
        self.assertAlmostEqual(float(result[4]), 1.97, places=4)

    def test_deadline_masks_short_moves_and_preserves_full_goal(self):
        z = torch.zeros(1, 16)
        z[0, 0] = 0.0001
        mask, _, _ = self.function("action_mask")(z, torch.tensor([250]))
        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[0, 34])

    def test_same_twenty_step_distance_convergence_rule(self):
        history = torch.ones(21, 1)
        stop = self.function("distance_stop_state")
        for step in range(1, 20):
            self.assertFalse(bool(stop(history, torch.tensor([1.0]), torch.tensor([step]))))
        self.assertTrue(bool(stop(history, torch.tensor([1.0]), torch.tensor([20]))))
        history = torch.ones(21, 1)
        for step in range(1, 21):
            self.assertFalse(bool(stop(history, torch.tensor([1 - step * 0.001]), torch.tensor([step]))))


if __name__ == "__main__":
    unittest.main()
