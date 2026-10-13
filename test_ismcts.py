"""Tests for the simulator and the ISMCTS agent: python -m unittest test_ismcts -v"""
import random
import unittest
from contextlib import redirect_stdout
from io import StringIO

from poker import (CALL, FOLD, RAISE, Card, CallAgent, Observation, PokerGame, RandomAgent,
                   evaluate_hand, parse_cards)
from arena import report, run_arena
from ismcts import ISMCTSAgent
from sim import SimState, card_id, score_cards

FULL_DECK = [Card(r, s) for s in "shdc" for r in range(2, 15)]
CRAMPED_DECK = [Card(r, s) for s in "sh" for r in (2, 3, 4, 5, 6, 7, 14)]  # many pairs, flushes, straights


def _ids(text):
    return [card_id(c) for c in parse_cards(text)]


def _cmp(a, b):
    return (a > b) - (a < b)


class ScoreTests(unittest.TestCase):
    def test_categories(self):
        hands = [("2c 5d 7h 9s Jc", 0), ("2c 2d 7h 9s Jc", 1), ("2c 2d 7h 7s Jc", 2),
                 ("2c 2d 2h 9s Jc", 3), ("5c 6d 7h 8s 9c", 4), ("2c 5c 7c 9c Jc", 5),
                 ("2c 2d 2h 9s 9c", 6), ("2c 2d 2h 2s Jc", 7), ("5c 6c 7c 8c 9c", 8),
                 ("As Ks Qs Js Ts 2d 3c", 8), ("2c 2d 2h 9s 9c 9d Ac", 6), ("Ac 2c 3c 4c 5c Kd Kh", 8)]
        for text, category in hands:
            self.assertEqual(score_cards(_ids(text)) >> 20, category, text)

    def test_matches_reference_ordering(self):
        rng = random.Random(7)
        for deck in (FULL_DECK, CRAMPED_DECK):
            for size in (5, 6, 7):
                for _ in range(400):
                    a, b = rng.sample(deck, size), rng.sample(deck, size)
                    expected = _cmp(evaluate_hand(a), evaluate_hand(b))
                    got = _cmp(score_cards([card_id(c) for c in a]), score_cards([card_id(c) for c in b]))
                    self.assertEqual(got, expected, (a, b))


class _Tap(RandomAgent):
    """Random player that logs its actions and checks SimState against what the engine reports."""

    def __init__(self, name, test, log, rng):
        super().__init__(name, rng)
        self.test = test
        self.log = log

    def act(self, obs):
        state = SimState.from_observation(obs)
        self.test.assertEqual(state.to_act, obs.seat)
        if RAISE in obs.legal_actions:
            low = min(state.current_bet + state.min_raise, state.bet[obs.seat] + state.stack[obs.seat])
            self.test.assertEqual(low, obs.min_raise_to)
        for kind, amount in state.actions():
            self.test.assertIn(kind, obs.legal_actions)
            if kind == RAISE:
                self.test.assertTrue(obs.min_raise_to <= amount <= obs.max_raise_to)
        action = super().act(obs)
        self.log.setdefault("first", obs)
        self.log.setdefault("actions", []).append((obs.seat, action))
        return action


class SimStateTests(unittest.TestCase):
    def test_replays_engine_hands_exactly(self):
        rng = random.Random(11)
        for hand in range(200):
            n = rng.randint(2, 6)
            stacks = [rng.choice([30, 80, 250, 1000]) for _ in range(n)]
            log = {}
            agents = [_Tap(f"P{i}", self, log, random.Random(rng.random())) for i in range(n)]
            game = PokerGame(agents, stacks, verbose=False, seed=hand)
            result = game.play_hand()

            state = SimState.from_observation(log["first"])
            for p in game.players:
                if p.hole:
                    state.hole[p.seat] = tuple(card_id(c) for c in p.hole)
            state.board = tuple(card_id(c) for c in game.community)
            for seat, action in log["actions"]:
                self.assertEqual(state.to_act, seat)
                state.apply((action.kind, action.amount))
            self.assertEqual(state.to_act, -1)
            self.assertEqual(state.payoffs(), [result["net"][i] for i in range(n)])

    def test_passive_payoffs_are_zero_sum(self):
        rng = random.Random(5)
        for hand in range(60):
            n = rng.randint(2, 5)
            log = {}
            agents = [_Tap(f"P{i}", self, log, random.Random(rng.random())) for i in range(n)]
            PokerGame(agents, [rng.choice([40, 300]) for _ in range(n)], verbose=False, seed=hand).play_hand()
            state = SimState.from_observation(log["first"])
            unseen = [c for c in range(52) if c not in state.hole[log["first"].seat]]
            drawn = rng.sample(unseen, 2 * n + 5)
            for seat in range(n):
                if seat != log["first"].seat and not state.folded[seat]:
                    state.hole[seat] = (drawn.pop(), drawn.pop())
            state.board = tuple(drawn[:5])
            self.assertEqual(sum(state.passive_payoffs()), 0)


class ArenaReportTests(unittest.TestCase):
    def test_aggregates_and_reports_agent_statistics(self):
        results = run_arena([CallAgent("A"), CallAgent("B")], deals=2, seed=4)

        self.assertEqual(set(results), {"A", "B"})
        for name in results:
            stats = results.agent_stats[name]
            self.assertEqual(stats["hands"], 4)
            self.assertEqual(stats["vpip"], 2)
            self.assertEqual(stats["pfr"], 0)
            self.assertEqual(stats["showdowns"], 4)
            self.assertEqual(sum(count for _, count in results.position_stats[name].values()), 4)

        output = StringIO()
        with redirect_stdout(output):
            report(results)
        text = output.getvalue()
        self.assertIn("Performance", text)
        self.assertIn("Action frequency by street", text)
        self.assertIn("Results by position", text)
        self.assertIn("BTN/SB", text)


class ISMCTSAgentTests(unittest.TestCase):
    def test_actions_are_legal_and_chips_conserved(self):
        test = self

        class Checked(ISMCTSAgent):
            def act(self, obs):
                action = super().act(obs)
                test.assertIn(action.kind, obs.legal_actions)
                if action.kind == RAISE:
                    test.assertTrue(obs.min_raise_to <= action.amount <= obs.max_raise_to)
                return action

        agents = [Checked("S", iterations=40, rng=random.Random(1)),
                  RandomAgent("R", random.Random(2)), CallAgent("C")]
        game = PokerGame(agents, 300, verbose=False, seed=5)
        game.play_match(max_hands=20)
        self.assertEqual(sum(p.stack for p in game.players), 900)

    def test_does_not_fold_the_nuts(self):
        players = [
            {"seat": 0, "name": "S", "stack": 800, "bet": 0, "total_bet": 200, "folded": False, "all_in": False},
            {"seat": 1, "name": "O", "stack": 600, "bet": 200, "total_bet": 400, "folded": False, "all_in": False},
        ]
        obs = Observation(
            seat=0, name="S", hole_cards=parse_cards("As Ks"), community_cards=parse_cards("Qs Js Ts 2d 3c"),
            street="river", pot=600, stack=800, to_call=200, min_raise_to=400, max_raise_to=800,
            legal_actions=[FOLD, CALL, RAISE], dealer=1, small_blind=5, big_blind=10, players=players,
            history=[{"seat": 1, "street": "river", "action": RAISE, "amount": 200}])
        agent = ISMCTSAgent("S", iterations=200, rng=random.Random(3))
        self.assertNotEqual(agent.act(obs).kind, FOLD)
        self.assertEqual(sum(visits for visits, _ in agent.last_search.values()), 200)


if __name__ == "__main__":
    unittest.main()
