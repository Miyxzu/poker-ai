"""Tests for the tabular Monte Carlo agent: python -m unittest test_qlearn -v"""
import os
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

from arena import run_arena
from poker import CallAgent, PokerGame, RandomAgent, parse_cards
from qlearn import ACTIONS, CALL_A, FOLD_A, QAgent, QTable, bucket, estimate_equity, train
from sim import card_id


def _ids(text):
    return tuple(card_id(c) for c in parse_cards(text))


class EquityTests(unittest.TestCase):
    def test_aces_beat_seven_deuce(self):
        rng = random.Random(0)
        self.assertGreater(estimate_equity(_ids("As Ah"), (), 1, 300, rng), 0.75)
        self.assertLess(estimate_equity(_ids("7c 2d"), (), 1, 300, rng), 0.45)

    def test_nuts_on_river(self):
        self.assertEqual(estimate_equity(_ids("As Ks"), _ids("Qs Js Ts 2d 3c"), 2, 50, random.Random(1)), 1.0)

    def test_bucket(self):
        self.assertEqual([bucket(v, (1, 2)) for v in (0, 1, 1.5, 2, 9)], [0, 1, 1, 2, 2])


class LearningTests(unittest.TestCase):
    def test_update_is_sample_mean_of_returns(self):
        table = QTable()
        agent = QAgent("Q", table, learn=True)
        agent._seat, agent._big_blind = 1, 10
        for net in (30, -10):
            agent.episode = [((0, 0, 0, 0, 0), CALL_A)]
            agent.on_hand_end({"net": {1: net}})
        q, n = table.values[(0, 0, 0, 0, 0)]
        self.assertEqual(n[CALL_A], 2)
        self.assertAlmostEqual(q[CALL_A], 1.0)  # mean of +3 and -1 bb
        self.assertEqual(agent.episode, [])

    def test_save_load_round_trip(self):
        table = QTable()
        table.row((1, 2, 0, 3, 1))[0][FOLD_A] = -0.5
        table.hands = 7
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "q.json")
            table.save(path)
            loaded = QTable.load(path)
        self.assertEqual((loaded.hands, loaded.values), (7, table.values))
        self.assertEqual(len(ACTIONS), len(loaded.values[(1, 2, 0, 3, 1)][0]))

    def test_training_fills_table_and_plays_legally(self):
        table = QTable()
        with redirect_stdout(StringIO()):
            train(table, 60, max_players=4, samples=10, seed=3, log_every=30)
        self.assertEqual(table.hands, 60)
        self.assertTrue(table.values)
        agents = [QAgent("Q", table, rng=random.Random(1)), RandomAgent("R", random.Random(2)), CallAgent("C")]
        results = run_arena(agents, deals=2, seed=1)
        self.assertEqual(set(results), {"Q", "R", "C"})
        self.assertAlmostEqual(sum(v[0] for v in results.values()), 0.0)

    def test_chips_conserved_with_learners(self):
        table = QTable()
        agents = [QAgent(f"Q{i}", table, 0.5, True, samples=10, rng=random.Random(i)) for i in range(3)]
        game = PokerGame(agents, 500, seed=5, verbose=False)
        for _ in range(20):
            if sum(p.stack > 0 for p in game.players) < 2:
                break
            game.play_hand()
            self.assertEqual(sum(p.stack for p in game.players), 1500)


if __name__ == "__main__":
    unittest.main()
