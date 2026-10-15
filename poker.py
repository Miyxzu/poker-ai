"""Console Texas Hold'em engine with a pluggable agent layer.

Play against bots:  python poker.py
Bot-only match:     python poker.py --sim --hands 200
Run the tests:      python -m unittest poker -v
"""
import argparse
import itertools
import random
import unittest
from collections import Counter
from dataclasses import dataclass
from typing import List

SUITS = "shdc"
RANK_CHARS = "23456789TJQKA"
HAND_NAMES = ["High card", "Pair", "Two pair", "Three of a kind", "Straight",
              "Flush", "Full house", "Four of a kind", "Straight flush"]
STREETS = [("preflop", 0), ("flop", 3), ("turn", 1), ("river", 1)]

FOLD, CHECK, CALL, RAISE = "fold", "check", "call", "raise"


# ---------------------------------------------------------------- cards

@dataclass(frozen=True)
class Card:
    rank: int  # 2..14, ace high
    suit: str  # one of SUITS

    @classmethod
    def from_str(cls, text):
        if len(text) != 2 or text[0].upper() not in RANK_CHARS or text[1].lower() not in SUITS:
            raise ValueError(f"Bad card: {text!r}")
        return cls(RANK_CHARS.index(text[0].upper()) + 2, text[1].lower())

    def __str__(self):
        return RANK_CHARS[self.rank - 2] + self.suit


def parse_cards(text):
    return [Card.from_str(t) for t in text.split()]


def fmt_cards(cards):
    return " ".join(str(c) for c in cards)


class Deck:
    """Shuffled 52-card deck; pass `stacked` (cards dealt from the front) for deterministic games."""

    def __init__(self, rng=None, stacked=None):
        if stacked is not None:
            self.cards = list(stacked)
        else:
            self.cards = [Card(r, s) for s in SUITS for r in range(2, 15)]
            (rng or random).shuffle(self.cards)

    def deal(self, num):
        return [self.cards.pop(0) for _ in range(num)]


# ---------------------------------------------------------------- hand evaluation

def _score5(cards):
    ranks = sorted((c.rank for c in cards), reverse=True)
    groups = sorted(Counter(ranks).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    ordered = tuple(r for r, _ in groups)
    shape = [n for _, n in groups]
    flush = len({c.suit for c in cards}) == 1
    straight_high = 0
    if len(groups) == 5:
        if ranks[0] - ranks[4] == 4:
            straight_high = ranks[0]
        elif ranks == [14, 5, 4, 3, 2]:
            straight_high = 5

    if flush and straight_high:
        return (8, (straight_high,))
    if shape[0] == 4:
        return (7, ordered)
    if shape == [3, 2]:
        return (6, ordered)
    if flush:
        return (5, tuple(ranks))
    if straight_high:
        return (4, (straight_high,))
    if shape[0] == 3:
        return (3, ordered)
    if shape == [2, 2, 1]:
        return (2, ordered)
    if shape[0] == 2:
        return (1, ordered)
    return (0, tuple(ranks))


def evaluate_hand(cards):
    """Best 5-card score from 5-7 cards; higher tuples beat lower ones."""
    if not 5 <= len(cards) <= 7:
        raise ValueError("Need 5 to 7 cards")
    return max(_score5(combo) for combo in itertools.combinations(cards, 5))


def hand_name(score):
    return HAND_NAMES[score[0]]


def split_pots(totals, folded, scores, dealer):
    """Chips won per seat; `scores` need only be comparable and is read for showdown seats."""
    n = len(totals)
    winnings = [0] * n
    prev = 0
    carry = 0
    for level in sorted({t for t in totals if t}):
        contributors = [s for s in range(n) if totals[s] >= level]
        pot = carry + (level - prev) * len(contributors)
        prev = level
        eligible = [s for s in contributors if not folded[s]]
        if not eligible:
            carry = pot
            continue
        carry = 0
        if len(eligible) == 1:
            winners = eligible
        else:
            best = max(scores[s] for s in eligible)
            winners = [s for s in eligible if scores[s] == best]
        winners.sort(key=lambda s: (s - dealer - 1) % n)  # odd chips go left of the dealer
        share, odd = divmod(pot, len(winners))
        for k, s in enumerate(winners):
            winnings[s] += share + (1 if k < odd else 0)
    return winnings


# ---------------------------------------------------------------- agent interface

@dataclass(frozen=True)
class Action:
    kind: str
    amount: int = 0  # RAISE only: total bet for the street ("raise to")


@dataclass
class Observation:
    """Everything an agent may see when it must act; opponents' hole cards are never included."""
    seat: int
    name: str
    hole_cards: List[Card]
    community_cards: List[Card]
    street: str
    pot: int
    stack: int
    to_call: int
    min_raise_to: int  # 0 when raising is illegal
    max_raise_to: int  # 0 when raising is illegal
    legal_actions: List[str]
    dealer: int
    small_blind: int
    big_blind: int
    players: List[dict]  # public per-seat info: seat, name, stack, bet, total_bet, folded, all_in
    history: List[dict]  # this hand's actions: seat, street, action, amount (chips added)


class Agent:
    is_human = False

    def __init__(self, name):
        self.name = name

    def act(self, obs):
        raise NotImplementedError

    def on_hand_end(self, result):
        """Called after each hand with the result dict from PokerGame.play_hand()."""


def _passive(obs):
    return Action(CHECK) if CHECK in obs.legal_actions else Action(CALL)


class CallAgent(Agent):
    def act(self, obs):
        return _passive(obs)


class RandomAgent(Agent):
    def __init__(self, name, rng=None):
        super().__init__(name)
        self.rng = rng or random.Random()

    def act(self, obs):
        legal = obs.legal_actions
        choices = [_passive(obs).kind]
        weights = [0.6]
        if FOLD in legal and CALL in legal:
            choices.append(FOLD)
            weights.append(0.15)
        if RAISE in legal:
            choices.append(RAISE)
            weights.append(0.25)
        kind = self.rng.choices(choices, weights)[0]
        if kind != RAISE:
            return Action(kind)
        if self.rng.random() < 0.1:
            return Action(RAISE, obs.max_raise_to)
        top = min(obs.max_raise_to, obs.min_raise_to + 4 * obs.big_blind)
        return Action(RAISE, self.rng.randint(obs.min_raise_to, top))


class ScriptedAgent(Agent):
    """Plays a fixed list of actions, then checks/calls."""

    def __init__(self, name, actions):
        super().__init__(name)
        self.actions = list(actions)

    def act(self, obs):
        return self.actions.pop(0) if self.actions else _passive(obs)


class HumanAgent(Agent):
    is_human = True

    def __init__(self, name="You", input_fn=input, output_fn=print):
        super().__init__(name)
        self.input_fn = input_fn
        self.output_fn = output_fn

    def act(self, obs):
        out = self.output_fn
        out(f"\n[{obs.street.upper()}] Board: {fmt_cards(obs.community_cards) or '-'}   Pot: {obs.pot}")
        out(f"Your hand: {fmt_cards(obs.hole_cards)}   Stack: {obs.stack}   To call: {obs.to_call}")
        menu = ["q=quit", "f=fold", "c=check" if CHECK in obs.legal_actions else f"c=call {obs.to_call}"]
        if RAISE in obs.legal_actions:
            menu.append(f"r <to>=raise ({obs.min_raise_to}-{obs.max_raise_to})")
            menu.append("a=all-in")
        while True:
            parts = self.input_fn(f"{', '.join(menu)} > ").strip().lower().split()
            if not parts:
                continue
            cmd = parts[0]
            if cmd in ("f", "fold"):
                return Action(FOLD)
            if cmd in ("q", "quit"):
                raise KeyboardInterrupt
            if cmd in ("c", "call", "check", "k"):
                return _passive(obs)
            if RAISE in obs.legal_actions:
                if cmd in ("a", "allin", "all-in"):
                    return Action(RAISE, obs.max_raise_to)
                if cmd in ("r", "raise", "bet"):
                    if len(parts) > 1 and parts[1].isdigit() and obs.min_raise_to <= int(parts[1]) <= obs.max_raise_to:
                        return Action(RAISE, int(parts[1]))
                    out(f"Raise amount must be {obs.min_raise_to}-{obs.max_raise_to}.")
                    continue
            out("Invalid choice.")


# ---------------------------------------------------------------- game engine

class Player:
    def __init__(self, seat, agent, stack):
        self.seat = seat
        self.agent = agent
        self.name = agent.name
        self.stack = stack
        self.reset()

    def reset(self):
        self.hole = []
        self.folded = False
        self.all_in = False
        self.active = False  # holds chips at the start of the hand
        self.bet = 0  # chips put in on the current street
        self.total_bet = 0  # chips put in this hand

    def __str__(self):
        return f"{self.name} ({self.stack})"


class PokerGame:
    def __init__(self, agents, starting_stack=1000, small_blind=5, big_blind=10,
                 seed=None, verbose=True, deck_factory=None):
        if len(agents) < 2:
            raise ValueError("Need at least two players")
        stacks = list(starting_stack) if isinstance(starting_stack, (list, tuple)) else [starting_stack] * len(agents)
        self.players = [Player(i, a, s) for i, (a, s) in enumerate(zip(agents, stacks))]
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.verbose = verbose
        self.rng = random.Random(seed)
        self.deck_factory = deck_factory or (lambda: Deck(self.rng))
        self.dealer = len(self.players) - 1  # first hand moves the button to seat 0
        self.hand_number = 0
        self.community = []
        self.current_bet = 0
        self.min_raise = big_blind
        self.street = "preflop"
        self.history = []

    @property
    def pot(self):
        return sum(p.total_bet for p in self.players)

    def _say(self, msg):
        if self.verbose:
            print(msg)

    def _next_seat(self, seat):
        n = len(self.players)
        for k in range(1, n + 1):
            if self.players[(seat + k) % n].active:
                return (seat + k) % n
        raise RuntimeError("No active players")

    # -- chips and actions

    def _post(self, p, amount):
        amount = max(0, min(amount, p.stack))
        p.stack -= amount
        p.bet += amount
        p.total_bet += amount
        if p.stack == 0:
            p.all_in = True
        self.current_bet = max(self.current_bet, p.bet)
        return amount

    def _legal(self, p):
        to_call = self.current_bet - p.bet
        legal = [FOLD, CHECK if to_call == 0 else CALL]
        if p.stack > to_call:
            legal.append(RAISE)
        return legal

    def _raise_bounds(self, p):
        cap = p.bet + p.stack
        return min(self.current_bet + self.min_raise, cap), cap

    def _observation(self, p):
        legal = self._legal(p)
        lo, hi = self._raise_bounds(p) if RAISE in legal else (0, 0)
        return Observation(
            seat=p.seat, name=p.name, hole_cards=list(p.hole), community_cards=list(self.community),
            street=self.street, pot=self.pot, stack=p.stack,
            to_call=min(self.current_bet - p.bet, p.stack), min_raise_to=lo, max_raise_to=hi,
            legal_actions=legal, dealer=self.dealer, small_blind=self.small_blind, big_blind=self.big_blind,
            players=[{"seat": q.seat, "name": q.name, "stack": q.stack, "bet": q.bet,
                      "total_bet": q.total_bet, "folded": q.folded, "all_in": q.all_in}
                     for q in self.players],
            history=[dict(h) for h in self.history])

    def _sanitise(self, p, action):
        """Illegal actions become check/fold; raise amounts are clamped to the legal range."""
        legal = self._legal(p)
        if not isinstance(action, Action) or action.kind not in legal:
            self._say(f"  ({p.name} made an illegal move; defaulting)")
            return Action(CHECK) if CHECK in legal else Action(FOLD)
        if action.kind == RAISE:
            try:
                amount = int(action.amount)
            except (TypeError, ValueError):
                return Action(CHECK) if CHECK in legal else Action(FOLD)
            lo, hi = self._raise_bounds(p)
            return Action(RAISE, max(lo, min(hi, amount)))
        return action

    def _apply(self, p, action):
        amount = 0
        if action.kind == FOLD:
            p.folded = True
            text = "folds"
        elif action.kind == CHECK:
            text = "checks"
        elif action.kind == CALL:
            amount = self._post(p, self.current_bet - p.bet)
            text = f"calls {amount}"
        else:
            previous = self.current_bet
            amount = self._post(p, action.amount - p.bet)
            if p.bet - previous >= self.min_raise:
                self.min_raise = p.bet - previous
            text = f"raises to {p.bet}"
        if p.all_in and action.kind != FOLD:
            text += " (all-in)"
        self.history.append({"seat": p.seat, "street": self.street, "action": action.kind, "amount": amount})
        self._say(f"  {p.name} {text}")
        return action.kind == RAISE

    def _betting_round(self, start):
        acted = set()
        seat = start
        n = len(self.players)
        while True:
            live = [p for p in self.players if not p.folded]
            able = [p for p in live if not p.all_in]
            if len(live) < 2 or not able:
                return
            if len(able) == 1 and able[0].bet >= self.current_bet:
                return
            pending = {p.seat for p in able if p.seat not in acted or p.bet != self.current_bet}
            if not pending:
                return
            while seat % n not in pending:
                seat += 1
            p = self.players[seat % n]
            seat += 1
            action = self._sanitise(p, p.agent.act(self._observation(p)))
            raised = self._apply(p, action)
            acted = {p.seat} if raised else acted | {p.seat}

    # -- hand flow

    def _start_hand(self):
        self.hand_number += 1
        for p in self.players:
            p.reset()
            p.active = p.stack > 0
            p.folded = not p.active
        if sum(p.active for p in self.players) < 2:
            raise ValueError("Need at least two players with chips")
        self.deck = self.deck_factory()
        self.community = []
        self.history = []
        self.current_bet = 0
        self.min_raise = self.big_blind
        self.street = "preflop"

        self.dealer = self._next_seat(self.dealer)
        if sum(p.active for p in self.players) == 2:  # heads-up: dealer is the small blind
            sb = self.dealer
            bb = self._next_seat(sb)
            first = sb
        else:
            sb = self._next_seat(self.dealer)
            bb = self._next_seat(sb)
            first = self._next_seat(bb)
        self._say(f"\n===== Hand {self.hand_number} | dealer: {self.players[self.dealer].name} =====")
        self._say("Stacks: " + ", ".join(str(p) for p in self.players))
        for p in self.players:
            if p.active:
                p.hole = self.deck.deal(2)
        self._say(f"{self.players[sb].name} posts small blind {self._post(self.players[sb], self.small_blind)}")
        self._say(f"{self.players[bb].name} posts big blind {self._post(self.players[bb], self.big_blind)}")
        return first

    def play_hand(self):
        """Plays one full hand and returns a result dict (seat-keyed `winnings` and `net`)."""
        first = self._start_hand()
        for street, new_cards in STREETS:
            self.street = street
            if street != "preflop":
                for p in self.players:
                    p.bet = 0
                self.current_bet = 0
                self.min_raise = self.big_blind
                first = self._next_seat(self.dealer)
            if new_cards:
                self.community += self.deck.deal(new_cards)
                self._say(f"--- {street.upper()}: {fmt_cards(self.community)} ---")
            self._betting_round(first)
        result = self._settle()
        for p in self.players:
            p.agent.on_hand_end(result)
        return result

    def _settle(self):
        live = [p for p in self.players if not p.folded]
        scores = {}
        showdown = {}
        if len(live) > 1:
            self._say("--- SHOWDOWN ---")
            for p in live:
                scores[p.seat] = evaluate_hand(p.hole + self.community)
                showdown[p.seat] = {"hole": list(p.hole), "hand": hand_name(scores[p.seat])}
                self._say(f"  {p.name}: {fmt_cards(p.hole)} -> {showdown[p.seat]['hand']}")

        pot_total = self.pot
        winnings = dict(enumerate(split_pots(
            [p.total_bet for p in self.players], [p.folded for p in self.players], scores, self.dealer)))

        for p in self.players:
            p.stack += winnings[p.seat]
            if winnings[p.seat]:
                self._say(f"{p.name} wins {winnings[p.seat]}")
        return {
            "hand_number": self.hand_number,
            "pot": pot_total,
            "community": list(self.community),
            "showdown": showdown,
            "winnings": winnings,
            "net": {p.seat: winnings[p.seat] - p.total_bet for p in self.players},
            "history": [dict(h) for h in self.history],
        }

    def play_match(self, max_hands=None):
        """Plays hands until one player has all the chips, `max_hands` is reached, or a human busts."""
        results = []
        while sum(p.stack > 0 for p in self.players) > 1 and (max_hands is None or len(results) < max_hands):
            results.append(self.play_hand())
            busted = [p for p in self.players if p.stack == 0 and p.agent.is_human]
            if busted:
                self._say(f"{busted[0].name} is out of chips.")
                break
        self._say("\nFinal stacks: " + ", ".join(str(p) for p in self.players))
        return results


# ---------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description="Console Texas Hold'em")
    parser.add_argument("--bots", type=int, default=2, help="number of bot opponents")
    parser.add_argument("--hands", type=int, default=None, help="max hands to play")
    parser.add_argument("--stack", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sim", action="store_true", help="bots only, no console output per hand")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    bots = [RandomAgent(f"Bot {i + 1}", random.Random(rng.random())) for i in range(args.bots)]
    agents = bots if args.sim else [HumanAgent("You")] + bots
    if args.sim and len(agents) < 2:
        parser.error("--sim needs at least 2 bots")
    game = PokerGame(agents, args.stack, seed=args.seed, verbose=not args.sim)
    results = game.play_match(args.hands if not args.sim else args.hands or 100)
    if args.sim:
        print(f"{len(results)} hands. Final stacks: " + ", ".join(str(p) for p in game.players))


# ---------------------------------------------------------------- tests

def _make_game(agents, stacks=1000, cards=None, **kwargs):
    factory = (lambda: Deck(stacked=parse_cards(cards))) if cards else None
    return PokerGame(agents, stacks, verbose=False, deck_factory=factory, **kwargs)


def _score(text):
    return evaluate_hand(parse_cards(text))


class HandEvaluationTests(unittest.TestCase):
    def test_categories(self):
        hands = [("2c 5d 7h 9s Jc", 0), ("2c 2d 7h 9s Jc", 1), ("2c 2d 7h 7s Jc", 2),
                 ("2c 2d 2h 9s Jc", 3), ("5c 6d 7h 8s 9c", 4), ("2c 5c 7c 9c Jc", 5),
                 ("2c 2d 2h 9s 9c", 6), ("2c 2d 2h 2s Jc", 7), ("5c 6c 7c 8c 9c", 8)]
        for text, category in hands:
            self.assertEqual(_score(text)[0], category, text)

    def test_ordering(self):
        self.assertGreater(_score("Ac Ad Kh 3s 2c"), _score("Ac Ad Qh 3s 2c"))  # kicker
        self.assertGreater(_score("2c 3d 4h 5s 6c"), _score("Ac 2d 3h 4s 5c"))  # wheel is lowest straight
        self.assertGreater(_score("2c 2d 2h 9s 9c"), _score("Ac Kc Qc Jc 9c"))  # full house > flush
        self.assertGreater(_score("3c 3d 3h 2s 2c"), _score("2c 2d 2h As Ac"))  # trips rank first
        self.assertEqual(_score("Ac Kd Qh Js 9c"), _score("Ad Kh Qs Jc 9d"))  # suits never break ties

    def test_wheel_straight_flush(self):
        self.assertEqual(_score("Ac 2c 3c 4c 5c"), (8, (5,)))

    def test_best_five_of_seven(self):
        self.assertEqual(_score("As Ks Qs Js Ts 2d 3c")[0], 8)
        self.assertEqual(_score("2c 2d 2h 9s 9c 9d Ac"), (6, (9, 2)))

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            evaluate_hand(parse_cards("As Ks Qs Js"))
        with self.assertRaises(ValueError):
            Card.from_str("1x")


class DeckTests(unittest.TestCase):
    def test_full_unique_deck(self):
        deck = Deck(random.Random(0))
        cards = deck.deal(52)
        self.assertEqual(len(set(cards)), 52)
        self.assertEqual(deck.cards, [])


class GameFlowTests(unittest.TestCase):
    def test_fold_preflop_heads_up(self):
        game = _make_game([ScriptedAgent("A", [Action(FOLD)]), CallAgent("B")])
        result = game.play_hand()  # seat 0 is dealer/small blind and acts first
        self.assertEqual(result["net"], {0: -5, 1: 5})
        self.assertEqual(result["showdown"], {})

    def test_check_down_showdown(self):
        game = _make_game([CallAgent("A"), CallAgent("B")],
                          cards="As Ah Kd Kc 2c 7d 9h Jc 3s")
        result = game.play_hand()
        self.assertEqual(result["net"], {0: 10, 1: -10})
        self.assertEqual(result["showdown"][0]["hand"], "Pair")
        self.assertEqual(len(game.community), 5)

    def test_split_pot(self):
        game = _make_game([CallAgent("A"), CallAgent("B")],
                          cards="2c 3d 4h 5d As Ks Qs Js Ts")
        result = game.play_hand()
        self.assertEqual(result["net"], {0: 0, 1: 0})

    def test_min_raise_is_enforced(self):
        game = _make_game([ScriptedAgent("A", [Action(RAISE, 11)]), ScriptedAgent("B", [Action(FOLD)])])
        result = game.play_hand()
        self.assertEqual(result["history"][0], {"seat": 0, "street": "preflop", "action": RAISE, "amount": 15})
        self.assertEqual(result["net"], {0: 10, 1: -10})

    def test_raise_is_capped_at_stack(self):
        game = _make_game([ScriptedAgent("A", [Action(RAISE, 10 ** 9)]), CallAgent("B")],
                          cards="As Ah Kd Kc 2c 7d 9h Jc 3s")
        result = game.play_hand()
        self.assertEqual(result["net"], {0: 1000, 1: -1000})

    def test_illegal_action_falls_back(self):
        game = _make_game([ScriptedAgent("A", [Action(CHECK)]), CallAgent("B")])  # cannot check facing the big blind
        self.assertEqual(game.play_hand()["net"], {0: -5, 1: 5})

    def test_side_pots(self):
        agents = [ScriptedAgent("A", [Action(RAISE, 50)]),
                  ScriptedAgent("B", [Action(RAISE, 100)]),
                  CallAgent("C")]
        game = _make_game(agents, [50, 100, 200],
                          cards="As Ah Kd Kc 7h 2d 3c 4d 9s Jh Qc")
        result = game.play_hand()
        self.assertEqual(result["net"], {0: 100, 1: 0, 2: -100})
        self.assertEqual([p.stack for p in game.players], [150, 100, 100])
        self.assertEqual(len(game.community), 5)

    def test_dealer_rotates(self):
        game = _make_game([CallAgent(c) for c in "ABC"])
        game.play_hand()
        self.assertEqual(game.dealer, 0)
        game.play_hand()
        self.assertEqual(game.dealer, 1)

    def test_busted_player_is_skipped(self):
        game = _make_game([CallAgent(c) for c in "ABC"], [0, 100, 100])
        game.play_hand()
        self.assertEqual(game.players[0].hole, [])
        self.assertEqual(sum(p.stack for p in game.players), 200)

    def test_play_match_stops_with_one_player(self):
        agents = [ScriptedAgent("A", [Action(RAISE, 10 ** 9)]), CallAgent("B")]
        game = _make_game(agents, 100, seed=3)
        game.play_match()
        self.assertEqual(sorted(p.stack for p in game.players)[0], 0)

    def test_chips_conserved_in_random_play(self):
        agents = [RandomAgent(f"R{i}", random.Random(i)) for i in range(6)]
        game = _make_game(agents, 500, seed=1)
        for _ in range(150):
            if sum(p.stack > 0 for p in game.players) < 2:
                break
            game.play_hand()
            self.assertEqual(sum(p.stack for p in game.players), 3000)
            self.assertTrue(all(p.stack >= 0 for p in game.players))


class AgentInterfaceTests(unittest.TestCase):
    def test_observation_contents(self):
        seen = []

        class Recorder(Agent):
            def act(self, obs):
                seen.append(obs)
                return Action(FOLD)

        _make_game([Recorder("A"), CallAgent("B")]).play_hand()
        obs = seen[0]
        self.assertEqual((obs.seat, obs.street, obs.pot, obs.stack, obs.to_call), (0, "preflop", 15, 995, 5))
        self.assertEqual(obs.legal_actions, [FOLD, CALL, RAISE])
        self.assertEqual((obs.min_raise_to, obs.max_raise_to), (20, 1000))
        self.assertEqual(len(obs.hole_cards), 2)
        self.assertEqual(obs.community_cards, [])
        self.assertTrue(all("hole_cards" not in q for q in obs.players))

    def test_on_hand_end_called(self):
        results = []

        class Listener(CallAgent):
            def on_hand_end(self, result):
                results.append(result)

        _make_game([Listener("A"), CallAgent("B")]).play_hand()
        self.assertEqual(len(results), 1)
        self.assertEqual(sum(results[0]["net"].values()), 0)

    def test_human_agent_parsing(self):
        answers = iter(["", "bogus", "r 5", "r 50"])
        human = HumanAgent("H", input_fn=lambda _: next(answers), output_fn=lambda *_: None)
        obs = Observation(0, "H", parse_cards("As Kd"), [], "preflop", 15, 995, 5, 20, 1000,
                          [FOLD, CALL, RAISE], 0, 5, 10, [], [])
        self.assertEqual(human.act(obs), Action(RAISE, 50))
        answers = iter(["c"])
        human.input_fn = lambda _: next(answers)
        self.assertEqual(human.act(obs), Action(CALL))


if __name__ == "__main__":
    main()