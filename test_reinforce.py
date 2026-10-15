"""Tests for the REINFORCE policy: python -m unittest test_reinforce -v"""
import os
import random
import tempfile
import unittest

import torch

from poker import CallAgent, PokerGame
from reinforce import INPUT_SIZE, ReinforceAgent, ReinforcePolicy, train


class ReinforcePolicyTests(unittest.TestCase):
    def test_probabilities_are_normalized_over_legal_actions(self):
        policy = ReinforcePolicy(seed=1)
        probabilities = policy.probabilities([0.0] * INPUT_SIZE, [1, 3])
        self.assertEqual(len(probabilities), 2)
        self.assertAlmostEqual(sum(probabilities), 1.0, places=6)
        self.assertTrue(all(0.0 <= probability <= 1.0 for probability in probabilities))

    def test_update_changes_parameters_and_updates_baseline(self):
        policy = ReinforcePolicy(seed=2)
        before = [parameter.detach().clone() for parameter in policy.model.parameters()]
        episode = [([0.1] * INPUT_SIZE, 1, (0, 1, 2)),
                   ([0.2] * INPUT_SIZE, 2, (0, 1, 2))]
        policy.update(episode, 4.0)
        self.assertEqual(policy.updates, 1)
        self.assertGreater(policy.baseline, 0.0)
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, policy.model.parameters())))

    def test_checkpoint_round_trip(self):
        policy = ReinforcePolicy(seed=3)
        policy.update([([0.3] * INPUT_SIZE, 2, (0, 1, 2))], 1.5)
        features = [0.25] * INPUT_SIZE
        expected = policy.probabilities(features, [0, 2, 4])
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "policy.pt")
            policy.save(path)
            loaded = ReinforcePolicy.load(path)
        self.assertEqual(loaded.updates, 1)
        self.assertAlmostEqual(loaded.baseline, policy.baseline)
        for actual, wanted in zip(loaded.probabilities(features, [0, 2, 4]), expected):
            self.assertAlmostEqual(actual, wanted, places=6)

    def test_training_updates_policy_and_agent_plays_legally(self):
        policy = ReinforcePolicy(seed=4)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "policy.pt")
            played = train(policy, 8, opponents=("call",), max_players=2, samples=4,
                           seed=5, checkpoint=path, checkpoint_every=4, log_every=100)
            self.assertEqual(played, 8)
            self.assertEqual(policy.updates, 8)
            self.assertTrue(os.path.exists(path))

        agent = ReinforceAgent("R", policy, samples=4, rng=random.Random(6))
        game = PokerGame([agent, CallAgent("C")], 200, seed=7, verbose=False)
        result = game.play_hand()
        self.assertEqual(sum(result["net"].values()), 0)
        self.assertIn(result["history"][0]["action"], {"fold", "call", "raise", "check"})


if __name__ == "__main__":
    unittest.main()
