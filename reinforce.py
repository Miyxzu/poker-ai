"""REINFORCE policy-gradient agent for the poker engine.

Train:    python reinforce.py --hands 100000 --out reinforce.pt
Evaluate: python arena.py --lineup reinforce,random,call --policy reinforce.pt
"""
import argparse
import importlib
import math
import os
import random
import time

from poker import Agent, CallAgent, PokerGame, RandomAgent
from qlearn import ACTIONS, QAgent, QTable

try:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    functional = importlib.import_module("torch.nn.functional")
except ImportError:
    torch = None
    nn = None
    functional = None

INPUT_SIZE = 18
HIDDEN_SIZE = 64
POLICY_VERSION = 1


def _require_torch():
    if torch is None:
        raise ImportError("REINFORCE requires PyTorch; install it with `python -m pip install -r requirements.txt`")


if nn is not None:
    class _PolicyNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
                nn.Tanh(),
                nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                nn.Tanh(),
                nn.Linear(HIDDEN_SIZE, len(ACTIONS)),
            )

        def forward(self, features):
            return self.layers(features)
else:
    class _PolicyNetwork:
        def __init__(self):
            _require_torch()


class ReinforcePolicy:
    def __init__(self, learning_rate=0.001, entropy_coefficient=0.01, baseline_decay=0.95, seed=None):
        _require_torch()
        if seed is not None:
            torch.manual_seed(seed)
        self.model = _PolicyNetwork()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.entropy_coefficient = entropy_coefficient
        self.baseline_decay = baseline_decay
        self.baseline = 0.0
        self.updates = 0

    def probabilities(self, features, legal_actions):
        with torch.no_grad():
            inputs = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            logits = self.model(inputs)[0]
            indices = torch.tensor(legal_actions, dtype=torch.long)
            return torch.softmax(logits[indices], dim=0).tolist()

    def update(self, episode, reward):
        if not episode:
            return
        advantage = reward - self.baseline
        losses = []
        for features, action, legal_actions in episode:
            inputs = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            logits = self.model(inputs)[0]
            indices = torch.tensor(legal_actions, dtype=torch.long)
            log_probabilities = functional.log_softmax(logits[indices], dim=0)
            probabilities = log_probabilities.exp()
            action_index = legal_actions.index(action)
            entropy = -(probabilities * log_probabilities).sum()
            losses.append(-advantage * log_probabilities[action_index] - self.entropy_coefficient * entropy)
        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward
        self.updates += 1

    def save(self, path):
        data = {
            "version": POLICY_VERSION,
            "actions": ACTIONS,
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "entropy_coefficient": self.entropy_coefficient,
            "baseline_decay": self.baseline_decay,
            "baseline": self.baseline,
            "updates": self.updates,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        temporary = path + ".tmp"
        torch.save(data, temporary)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path, learning_rate=0.001):
        _require_torch()
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("version") != POLICY_VERSION or tuple(data.get("actions", ())) != ACTIONS:
            raise ValueError(f"{path} uses an incompatible REINFORCE policy format")
        if data.get("input_size") != INPUT_SIZE or data.get("hidden_size") != HIDDEN_SIZE:
            raise ValueError(f"{path} was trained with a different network architecture")
        policy = cls(learning_rate, data["entropy_coefficient"], data["baseline_decay"])
        policy.model.load_state_dict(data["model"])
        policy.optimizer.load_state_dict(data["optimizer"])
        policy.baseline = data["baseline"]
        policy.updates = data["updates"]
        return policy


class ReinforceAgent(Agent):
    def __init__(self, name, policy, samples=60, rng=None, learn=False):
        super().__init__(name)
        self.policy = policy
        self.samples = samples
        self.rng = rng or random.Random()
        self.learn = learn
        self.encoder = QAgent(f"{name} encoder", QTable(), samples=samples, rng=self.rng)
        self.episode = []
        self._seat = None
        self._big_blind = 1

    def features(self, obs):
        key, equity = self.encoder.encode(obs)
        opponents = sum(not player["folded"] for player in obs.players) - 1
        big_blind = max(obs.big_blind, 1)
        other_stacks = [player["stack"] for player in obs.players
                        if not player["folded"] and player["seat"] != obs.seat]
        effective_stack = min(obs.stack, max(other_stacks)) if other_stacks else obs.stack
        pot_odds = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
        invested = obs.players[obs.seat]["total_bet"]
        commitment = invested / (invested + obs.stack) if invested + obs.stack else 0.0
        values = [
            equity,
            equity * (opponents + 1) / 5,
            pot_odds,
            math.log1p(obs.to_call / big_blind) / 8,
            math.log1p(obs.pot / big_blind) / 10,
            math.log1p(effective_stack / max(obs.pot, big_blind)) / 6,
            commitment,
            min(opponents, 4) / 4,
        ]
        street = key[0]
        position = key[5]
        values.extend(float(street == index) for index in range(4))
        values.extend(float(position == index) for index in range(6))
        return values

    def act(self, obs):
        self._seat = obs.seat
        self._big_blind = obs.big_blind
        features = self.features(obs)
        legal = QAgent.legal_actions(obs)
        probabilities = self.policy.probabilities(features, legal)
        choice = self.rng.choices(legal, weights=probabilities, k=1)[0]
        if self.learn:
            self.episode.append((features, choice, legal))
        return QAgent._to_action(obs, choice)

    def on_hand_end(self, result):
        if self.learn and self.episode:
            reward = result["net"][self._seat] / max(self._big_blind, 1)
            self.policy.update(self.episode, reward)
            self.episode = []


def _opponent(kind, name, rng, ismcts_iterations):
    if kind == "random":
        return RandomAgent(name, random.Random(rng.random()))
    if kind == "call":
        return CallAgent(name)
    from ismcts import ISMCTSAgent
    return ISMCTSAgent(name, ismcts_iterations, rng=random.Random(rng.random()))


def train(policy, hands, opponents=("random", "call"), max_players=4, samples=60, seed=0, stack=1000,
          big_blind=10, deadline=None, checkpoint=None, checkpoint_every=5000, log_every=1000,
          ismcts_iterations=30):
    """Trains one policy-controlled seat per hand against sampled fixed-policy opponents."""
    rng = random.Random(seed)
    window_reward = 0.0
    window_hands = 0
    started = time.time()
    played = 0
    try:
        for played in range(1, hands + 1):
            seats = rng.randint(2, max_players)
            learner_seat = rng.randrange(seats)
            agents = []
            for seat in range(seats):
                if seat == learner_seat:
                    agents.append(ReinforceAgent("Reinforce", policy, samples=samples,
                                                 rng=random.Random(rng.random()), learn=True))
                else:
                    kind = rng.choice(opponents)
                    agents.append(_opponent(kind, f"Opp{seat}", rng, ismcts_iterations))
            game = PokerGame(agents, stack, big_blind // 2, big_blind,
                             seed=rng.getrandbits(32), verbose=False)
            reward = game.play_hand()["net"][learner_seat] / big_blind
            window_reward += reward
            window_hands += 1
            if played % log_every == 0:
                mean_reward = window_reward / window_hands if window_hands else 0.0
                rate = played / max(time.time() - started, 1e-9)
                print(f"hands {played:>8}  updates {policy.updates:>8}  baseline {policy.baseline:+.2f} bb  "
                      f"reward {mean_reward:+.2f} bb/hand  {rate:.0f} hands/s", flush=True)
                window_reward = 0.0
                window_hands = 0
            if checkpoint and played % checkpoint_every == 0:
                policy.save(checkpoint)
            if deadline and time.time() >= deadline:
                break
    except KeyboardInterrupt:
        print("Interrupted.")
    if checkpoint:
        policy.save(checkpoint)
        print(f"Saved policy after {policy.updates} updates to {checkpoint}")
    return played


def main():
    parser = argparse.ArgumentParser(description="Train a REINFORCE poker policy")
    parser.add_argument("--hands", type=int, default=100000)
    parser.add_argument("--minutes", type=float, help="stop and save after this long")
    parser.add_argument("--out", default="reinforce.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--opponents", default="random,call", help="baseline kinds: random, call, ismcts")
    parser.add_argument("--max-players", type=int, default=4, choices=range(2, 7))
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args()
    opponents = tuple(kind.strip() for kind in args.opponents.split(",") if kind.strip())
    if not opponents or any(kind not in ("random", "call", "ismcts") for kind in opponents):
        parser.error("--opponents must list random, call and/or ismcts")
    if args.resume and os.path.exists(args.out):
        policy = ReinforcePolicy.load(args.out, args.learning_rate)
    else:
        policy = ReinforcePolicy(args.learning_rate, args.entropy, seed=args.seed)
    deadline = time.time() + args.minutes * 60 if args.minutes else None
    train(policy, args.hands, opponents, args.max_players, args.samples, args.seed,
          deadline=deadline, checkpoint=args.out, checkpoint_every=args.checkpoint_every,
          log_every=args.log_every)


if __name__ == "__main__":
    main()
