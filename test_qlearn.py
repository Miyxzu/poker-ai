"""Tests for the tabular Monte Carlo agent: python -m unittest test_qlearn -v"""
import os
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

from arena import run_arena
from poker import CALL, CHECK, RAISE, CallAgent, Observation, PokerGame, RandomAgent, parse_cards
from qlearn import ACTIONS, CALL_A, FOLD_A, QAgent, QTable, bucket, estimate_equity, train
from sim import card_id


def _ids(text):
    return tuple(card_id(c) for c in parse_cards(text))


def _observation(stack=100, opponent_stack=100, pot=100, dealer=0, max_raise=None):
    players = [
        {"seat": seat, "name": f"P{seat}", "stack": stack if seat == 0 else opponent_stack,
         "bet": 0, "total_bet": 0, "folded": False, "all_in": False}
        for seat in range(3)
    ]
    return Observation(0, "P0", parse_cards("As Ah"), [], "preflop", pot, stack, 0,
                       10, max_raise if max_raise is not None else stack, [CHECK, RAISE], dealer,
                       5, 10, players, [])


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

    def test_old_table_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "old.json")
            with open(path, "w") as table_file:
                table_file.write('{"actions": ["fold", "call", "raise_half", "raise_pot", "all_in"], "hands": 0, "states": {}}')
            with self.assertRaisesRegex(ValueError, "old Q-table format"):
                QTable.load(path)

    def test_key_separates_position_and_effective_depth(self):
        agent = QAgent("Q", QTable(), samples=5, rng=random.Random(1))
        base_obs = _observation()
        base, _ = agent.encode(base_obs)
        other_position, _ = agent.encode(_observation(dealer=2))
        deeper, _ = agent.encode(_observation(stack=200, opponent_stack=200))
        raised_obs = _observation()
        raised_obs.history = [{"seat": 1, "street": "preflop", "action": RAISE, "amount": 10}]
        raised, _ = agent.encode(raised_obs)
        dead_button_obs = _observation()
        dead_button_obs.players[0]["folded"] = True
        dead_button_obs.seat = 1
        dead_button, _ = agent.encode(dead_button_obs)
        self.assertNotEqual(base, other_position)
        self.assertNotEqual(base, deeper)
        self.assertNotEqual(base, raised)
        self.assertEqual(dead_button[5], 0)

    def test_greedy_learning_selects_untried_legal_action(self):
        table = QTable()
        key = (0, 0, 0, 0, 0, 0, 0)
        row = table.row(key)
        row[1][CALL_A] = 1
        agent = QAgent("Q", table, learn=True, rng=random.Random(2))
        obs = _observation()
        legal = agent.legal_actions(obs)
        self.assertNotEqual(agent._greedy(key, legal, obs, 0.5), CALL_A)

    def test_duplicate_raise_sizes_are_collapsed(self):
        obs = _observation(stack=20, opponent_stack=100, pot=100, max_raise=20)
        raises = [action for action in QAgent.legal_actions(obs)
                  if QAgent._to_action(obs, action).kind == RAISE]
        self.assertEqual(raises, [ACTIONS.index("all_in")])
        self.assertEqual(len({(QAgent._to_action(obs, action).kind,
                               QAgent._to_action(obs, action).amount) for action in raises}), 1)

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
