"""Tabular Monte Carlo control agent.

State = (street, hand-strength bucket, opponents, pot-odds bucket, commitment bucket); strength is Monte Carlo
equity against random hands, relative to a fair share. Actions are fold, check/call, raise 1/2 pot, raise pot,
all-in. After each hand every (state, action) the agent visited is moved towards the hand's net result in big
blinds (every-visit Monte Carlo, epsilon-greedy). All learners in a training run share one table (self-play).

Train:     python qlearn.py --hands 50000 --out qtable.json   (Ctrl-C or --minutes stops and saves; --resume continues)
Evaluate:  python arena.py --lineup qlearn,random,call --qtable qtable.json
"""
import argparse
import json
import os
import random
import time

from poker import CALL, CHECK, FOLD, RAISE, Action, Agent, CallAgent, PokerGame, RandomAgent
from sim import card_id, score_cards

ACTIONS = ("fold", "call", "raise_half", "raise_pot", "all_in")
FOLD_A, CALL_A, HALF_A, POT_A, ALLIN_A = range(len(ACTIONS))
RAISE_FRACTION = {HALF_A: 0.5, POT_A: 1.0}

STREET_INDEX = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
RATIO_EDGES = (0.5, 0.75, 0.9, 1.05, 1.2, 1.4, 1.6, 1.8, 2.2, 3.0)  # equity / fair share
ODDS_EDGES = (0.15, 0.25, 0.33, 0.45)  # to_call / (pot + to_call)
COMMIT_EDGES = (0.1, 0.35)  # share of the hand's chips already invested
DEPTH_EDGES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
PREFLOP_SAMPLES = 400
QTABLE_VERSION = 2

_preflop_equity = {}


def bucket(value, edges):
    return sum(value >= e for e in edges)


def estimate_equity(hole, board, opponents, samples, rng):
    """Win share (ties split) against `opponents` random hands; cards are ints 0..51."""
    seen = set(hole) | set(board)
    unseen = [c for c in range(52) if c not in seen]
    hands = 2 * opponents
    draws = hands + 5 - len(board)
    share = 0.0
    for _ in range(samples):
        drawn = rng.sample(unseen, draws)
        full = tuple(board) + tuple(drawn[hands:])
        mine = score_cards(tuple(hole) + full)
        theirs = [score_cards((drawn[2 * i], drawn[2 * i + 1]) + full) for i in range(opponents)]
        top = max(theirs)
        if mine > top:
            share += 1
        elif mine == top:
            share += 1 / (1 + theirs.count(top))
    return share / samples


class QTable:
    """state key -> [q values, visit counts], one entry per abstract action."""

    def __init__(self):
        self.values = {}
        self.hands = 0

    def row(self, key):
        row = self.values.get(key)
        if row is None:
            row = self.values[key] = [[0.0] * len(ACTIONS), [0] * len(ACTIONS)]
        return row

    def save(self, path):
        data = {"version": QTABLE_VERSION, "actions": ACTIONS, "hands": self.hands,
                "states": {",".join(map(str, k)): v for k, v in self.values.items()}}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        if data.get("version") != QTABLE_VERSION:
            raise ValueError(f"{path} uses an old Q-table format; retrain it with python qlearn.py")
        if tuple(data["actions"]) != ACTIONS:
            raise ValueError(f"{path} was trained with a different action set")
        table = cls()
        table.hands = data["hands"]
        table.values = {tuple(int(x) for x in k.split(",")): v for k, v in data["states"].items()}
        return table


class QAgent(Agent):
    def __init__(self, name, table, epsilon=0.0, learn=False, alpha_floor=0.002, samples=60, rng=None):
        super().__init__(name)
        self.table = table
        self.epsilon = epsilon
        self.learn = learn
        self.alpha_floor = alpha_floor  # keeps adapting as the self-play opponents change
        self.samples = samples
        self.rng = rng or random.Random()
        self.episode = []
        self._seat = None
        self._big_blind = 1

    def encode(self, obs):
        """Returns (state key, equity vs one random hand per live opponent)."""
        opponents = sum(not p["folded"] for p in obs.players) - 1
        hole = tuple(card_id(c) for c in obs.hole_cards)
        board = tuple(card_id(c) for c in obs.community_cards)
        if board:
            equity = estimate_equity(hole, board, opponents, self.samples, self.rng)
        else:
            a, b = sorted(c // 4 for c in hole)
            cache_key = (a, b, hole[0] % 4 == hole[1] % 4, opponents)
            if cache_key not in _preflop_equity:
                _preflop_equity[cache_key] = estimate_equity(hole, board, opponents, PREFLOP_SAMPLES, self.rng)
            equity = _preflop_equity[cache_key]
        ratio = equity * (opponents + 1)
        required = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
        invested = obs.players[obs.seat]["total_bet"]
        live_seats = [i for i, player in enumerate(obs.players) if not player["folded"]]
        position_order = [((obs.dealer + offset) % len(obs.players)) for offset in range(len(obs.players))
                          if not obs.players[(obs.dealer + offset) % len(obs.players)]["folded"]]
        position = position_order.index(obs.seat)
        position = min(position, 5)
        other_stacks = [obs.players[i]["stack"] for i in live_seats if i != obs.seat]
        effective_stack = min(obs.stack, max(other_stacks)) if other_stacks else obs.stack
        depth = effective_stack / max(obs.pot, obs.big_blind)
        street_raises = min(2, sum(h["street"] == obs.street and h["action"] == RAISE
                       for h in obs.history))
        preflop_raiser = next((h["seat"] for h in reversed(obs.history)
                       if h["street"] == "preflop" and h["action"] == RAISE), None)
        key = (STREET_INDEX[obs.street], bucket(ratio, RATIO_EDGES), min(opponents, 3) - 1,
               bucket(required, ODDS_EDGES) + 1 if obs.to_call else 0,
               bucket(invested / (invested + obs.stack) if invested + obs.stack else 0.0, COMMIT_EDGES),
                             position, bucket(depth, DEPTH_EDGES), street_raises,
                             int(preflop_raiser == obs.seat))
        return key, equity

    @staticmethod
    def legal_actions(obs):
        legal = [CALL_A]
        if obs.to_call > 0:
            legal.append(FOLD_A)
        if RAISE in obs.legal_actions:
            seen = set()
            for choice in (ALLIN_A, HALF_A, POT_A):
                action = QAgent._to_action(obs, choice)
                action_key = (action.kind, action.amount)
                if action_key not in seen:
                    legal.append(choice)
                    seen.add(action_key)
        return legal

    def act(self, obs):
        self._seat = obs.seat
        self._big_blind = obs.big_blind
        key, equity = self.encode(obs)
        legal = self.legal_actions(obs)
        if self.learn and self.rng.random() < self.epsilon:
            choice = self.rng.choice(legal)
        else:
            choice = self._greedy(key, legal, obs, equity)
        if self.learn:
            self.episode.append((key, choice))
        return self._to_action(obs, choice)

    def _greedy(self, key, legal, obs, equity):
        row = self.table.values.get(key)
        untried = [a for a in legal if row is None or row[1][a] == 0]
        if self.learn and untried:
            return self.rng.choice(untried)
        tried = [a for a in legal if row and row[1][a] > 0]
        if tried:
            return max(tried, key=lambda a: row[0][a])
        # Unseen state: call when equity covers the price of the call.
        if obs.to_call and equity < obs.to_call / (obs.pot + obs.to_call):
            return FOLD_A
        return CALL_A

    @staticmethod
    def _to_action(obs, choice):
        if choice == FOLD_A:
            return Action(FOLD)
        if choice == CALL_A:
            return Action(CHECK if CHECK in obs.legal_actions else CALL)
        if choice == ALLIN_A:
            return Action(RAISE, obs.max_raise_to)
        current = obs.players[obs.seat]["bet"] + obs.to_call
        target = current + int(RAISE_FRACTION[choice] * (obs.pot + obs.to_call))
        return Action(RAISE, max(obs.min_raise_to, min(obs.max_raise_to, target)))

    def on_hand_end(self, result):
        if not (self.learn and self.episode):
            return
        reward = result["net"][self._seat] / self._big_blind
        for key, action in self.episode:
            q, n = self.table.row(key)
            n[action] += 1
            q[action] += max(1 / n[action], self.alpha_floor) * (reward - q[action])
        self.episode = []


# ---------------------------------------------------------------- training

def _opponent(kind, name, rng, ismcts_iterations):
    if kind == "random":
        return RandomAgent(name, random.Random(rng.random()))
    if kind == "call":
        return CallAgent(name)
    from ismcts import ISMCTSAgent
    return ISMCTSAgent(name, ismcts_iterations, rng=random.Random(rng.random()))


def train(table, hands, opponents=("random", "call"), max_players=4, self_play=0.5, epsilon=(0.3, 0.05),
          samples=60, seed=0, stack=1000, big_blind=10, deadline=None, checkpoint=None,
          checkpoint_every=5000, log_every=1000, ismcts_iterations=30):
    """Plays `hands` self-play/baseline hands; learners share `table`. Returns hands played this call."""
    rng = random.Random(seed)
    window, window_hands, start = 0.0, 0, time.time()
    played = 0
    try:
        for played in range(1, hands + 1):
            eps = epsilon[0] + (epsilon[1] - epsilon[0]) * played / hands
            seats = rng.randint(2, max_players)
            learner_flags = [rng.random() < self_play for _ in range(seats)]
            if not any(learner_flags):
                learner_flags[rng.randrange(seats)] = True
            agents = []
            for i, is_learner in enumerate(learner_flags):
                if is_learner:
                    agents.append(QAgent(f"Q{i}", table, eps, True, samples=samples, rng=random.Random(rng.random())))
                else:
                    agents.append(_opponent(rng.choice(opponents), f"Opp{i}", rng, ismcts_iterations))
            game = PokerGame(agents, stack, big_blind // 2, big_blind, seed=rng.getrandbits(32), verbose=False)
            net = game.play_hand()["net"]
            table.hands += 1
            if not all(learner_flags):  # only hands against baselines say anything about strength
                learners = [s for s, f in enumerate(learner_flags) if f]
                window += sum(net[s] for s in learners) / len(learners) / big_blind
                window_hands += 1
            if played % log_every == 0:
                rate = played / (time.time() - start)
                mean = window / window_hands if window_hands else 0.0
                print(f"hands {table.hands:>8}  states {len(table.values):>6}  eps {eps:.3f}  "
                      f"learner vs baselines {mean:+.2f} bb/hand  {rate:.0f} hands/s", flush=True)
                window, window_hands = 0.0, 0
            if checkpoint and played % checkpoint_every == 0:
                table.save(checkpoint)
            if deadline and time.time() >= deadline:
                break
    except KeyboardInterrupt:
        print("Interrupted.")
    if checkpoint:
        table.save(checkpoint)
        print(f"Saved {len(table.values)} states / {table.hands} hands to {checkpoint}")
    return played


def main():
    parser = argparse.ArgumentParser(description="Train the tabular Monte Carlo Q agent")
    parser.add_argument("--hands", type=int, default=50000)
    parser.add_argument("--minutes", type=float, help="stop (and save) after this long")
    parser.add_argument("--out", default="qtable.json")
    parser.add_argument("--resume", action="store_true", help="continue from --out if it exists")
    parser.add_argument("--opponents", default="random,call", help="baseline kinds: random, call, ismcts")
    parser.add_argument("--max-players", type=int, default=4, choices=range(2, 7))
    parser.add_argument("--self-play", type=float, default=0.5, help="chance each seat is a learner")
    parser.add_argument("--eps-start", type=float, default=0.3)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--samples", type=int, default=60, help="equity samples per postflop decision")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args()

    opponents = tuple(k.strip() for k in args.opponents.split(",") if k.strip())
    if not opponents or any(k not in ("random", "call", "ismcts") for k in opponents):
        parser.error("--opponents must list random, call and/or ismcts")
    table = QTable.load(args.out) if args.resume and os.path.exists(args.out) else QTable()
    deadline = time.time() + args.minutes * 60 if args.minutes else None
    train(table, args.hands, opponents, args.max_players, args.self_play, (args.eps_start, args.eps_end),
          args.samples, args.seed, deadline=deadline, checkpoint=args.out,
          checkpoint_every=args.checkpoint_every, log_every=args.log_every)


if __name__ == "__main__":
    main()
