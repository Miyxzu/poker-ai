"""Single-observer ISMCTS agent.

Each iteration samples ("determinizes") the opponents' hole cards and the unseen board, then walks one
shared tree keyed by public action history. Legal actions depend only on public chips, so every action
is always available and plain UCB1 is exact here. Rollouts past the tree are check/call-down playouts.
"""
import math
import random

from poker import Action, Agent
from sim import SimState, card_id


class _Node:
    __slots__ = ("n", "w", "children")

    def __init__(self):
        self.n = 0  # times this node was reached
        self.w = 0.0  # summed payoff for the seat that chose the action leading here
        self.children = {}


class ISMCTSAgent(Agent):
    def __init__(self, name, iterations=500, exploration=0.7, bet_fractions=(0.5, 1.0), rng=None):
        super().__init__(name)
        self.iterations = iterations
        self.exploration = exploration
        self.bet_fractions = tuple(bet_fractions)
        self.rng = rng or random.Random()
        self.last_search = {}  # action -> (visits, mean chips) from the latest decision

    def act(self, obs):
        kind, amount = self._search(obs, SimState.from_observation(obs))
        return Action(kind, amount)

    def _search(self, obs, root):
        board = tuple(card_id(c) for c in obs.community_cards)
        known = {card_id(c) for c in obs.hole_cards} | set(board)
        unseen = [c for c in range(52) if c not in known]
        opponents = [s for s in range(root.n) if s != obs.seat and not root.folded[s]]
        hands = 2 * len(opponents)
        draws = hands + 5 - len(board)

        tree = _Node()
        self._lo, self._hi = math.inf, -math.inf
        for _ in range(self.iterations):
            drawn = self.rng.sample(unseen, draws)
            state = root.copy()
            for i, seat in enumerate(opponents):
                state.hole[seat] = (drawn[2 * i], drawn[2 * i + 1])
            state.board = board + tuple(drawn[hands:])
            self._iterate(tree, state)

        self.last_search = {a: (c.n, c.w / c.n) for a, c in tree.children.items()}
        return max(tree.children.items(), key=lambda kv: (kv[1].n, kv[1].w / kv[1].n))[0]

    def _iterate(self, root, state):
        node = root
        path = []
        while state.to_act >= 0:
            actor = state.to_act
            actions = state.actions(self.bet_fractions)
            untried = [a for a in actions if a not in node.children]
            if untried:
                action = self.rng.choice(untried)
                child = node.children[action] = _Node()
            else:
                action = self._select(node, actions)
                child = node.children[action]
            state.apply(action)
            path.append((child, actor))
            if untried:
                break
            node = child

        payoffs = state.passive_payoffs()
        root.n += 1
        for child, actor in path:
            child.n += 1
            child.w += payoffs[actor]
        self._lo = min(self._lo, min(payoffs))
        self._hi = max(self._hi, max(payoffs))

    def _select(self, node, actions):
        span = self._hi - self._lo
        log_visits = math.log(node.n)
        best, best_score = None, -math.inf
        for a in actions:
            child = node.children[a]
            value = (child.w / child.n - self._lo) / span if span > 0 else 0.5
            score = value + self.exploration * math.sqrt(log_visits / child.n)
            if score > best_score:
                best, best_score = a, score
        return best
