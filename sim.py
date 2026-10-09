"""Copyable hand simulator and fast 5-7 card scorer for search agents.

Cards are ints 0..51 (rank = c // 4 + 2, suit = SUITS[c % 4]). `SimState` follows the
PokerGame betting rules but has no agents, I/O or hidden-information handling.
"""
from poker import CALL, CHECK, FOLD, RAISE, STREETS, SUITS, split_pots

STREET_NAMES = [name for name, _ in STREETS]


def card_id(card):
    return (card.rank - 2) * 4 + SUITS.index(card.suit)


# ---------------------------------------------------------------- scoring

def _pack(ranks):
    value = 0
    for r in ranks:
        value = (value << 4) | r
    return value << (4 * (5 - len(ranks)))


def _straight_high(mask):
    if mask & 0x4000:  # ace also plays low
        mask |= 2
    runs = mask & (mask >> 1) & (mask >> 2) & (mask >> 3) & (mask >> 4)
    return runs.bit_length() + 3 if runs else 0


def _top_ranks(mask, count):
    ranks = []
    for r in range(14, 1, -1):
        if mask >> r & 1:
            ranks.append(r)
            if len(ranks) == count:
                break
    return ranks


def score_cards(cards):
    """Integer score of the best 5-card hand from 5-7 card ids; order matches evaluate_hand."""
    counts = [0] * 15
    suit_mask = [0, 0, 0, 0]
    suit_size = [0, 0, 0, 0]
    for c in cards:
        r = (c >> 2) + 2
        s = c & 3
        counts[r] += 1
        suit_mask[s] |= 1 << r
        suit_size[s] += 1

    flush_mask = 0
    for s in range(4):
        if suit_size[s] >= 5:
            flush_mask = suit_mask[s]
    if flush_mask:
        high = _straight_high(flush_mask)
        if high:
            return (8 << 20) | _pack([high])

    quad = 0
    trips, pairs, singles = [], [], []
    for r in range(14, 1, -1):
        k = counts[r]
        if k == 4:
            quad = r
        elif k == 3:
            trips.append(r)
        elif k == 2:
            pairs.append(r)
        elif k == 1:
            singles.append(r)

    if quad:
        kicker = max(trips[:1] + pairs[:1] + singles[:1])
        return (7 << 20) | _pack([quad, kicker])
    if trips and (len(trips) > 1 or pairs):
        pair = max(trips[1:2] + pairs[:1])
        return (6 << 20) | _pack([trips[0], pair])
    if flush_mask:
        return (5 << 20) | _pack(_top_ranks(flush_mask, 5))
    high = _straight_high(suit_mask[0] | suit_mask[1] | suit_mask[2] | suit_mask[3])
    if high:
        return (4 << 20) | _pack([high])
    if trips:
        return (3 << 20) | _pack([trips[0]] + sorted(pairs + singles, reverse=True)[:2])
    if len(pairs) > 1:
        kicker = max(pairs[2:] + singles)
        return (2 << 20) | _pack([pairs[0], pairs[1], kicker])
    if pairs:
        return (1 << 20) | _pack([pairs[0]] + singles[:3])
    return _pack(singles[:5])


# ---------------------------------------------------------------- state

def _replay_street(obs):
    """Recover (current_bet, min_raise, acted bitmask) for the street in `obs` from its history."""
    actions = [h for h in obs.history if h["street"] == obs.street]
    bets = [p["bet"] for p in obs.players]
    for h in actions:
        bets[h["seat"]] -= h["amount"]
    current = max(bets)
    min_raise = obs.big_blind
    acted = 0
    for h in actions:
        seat = h["seat"]
        previous = current
        bets[seat] += h["amount"]
        if h["action"] == RAISE:
            current = max(current, bets[seat])
            if bets[seat] - previous >= min_raise:
                min_raise = bets[seat] - previous
            acted = 1 << seat
        else:
            acted |= 1 << seat
    return current, min_raise, acted


class SimState:
    """One hand in progress. `to_act` is the acting seat, or -1 once the hand is over."""

    __slots__ = ("n", "dealer", "big_blind", "stack", "bet", "total", "folded", "all_in",
                 "hole", "board", "street", "current_bet", "min_raise", "acted", "to_act")

    @classmethod
    def from_observation(cls, obs):
        """State at `obs.seat`'s decision; only that seat's hole cards are known (others are None)."""
        s = cls.__new__(cls)
        s.n = len(obs.players)
        s.dealer = obs.dealer
        s.big_blind = obs.big_blind
        s.stack = [p["stack"] for p in obs.players]
        s.bet = [p["bet"] for p in obs.players]
        s.total = [p["total_bet"] for p in obs.players]
        s.folded = [p["folded"] for p in obs.players]
        s.all_in = [p["all_in"] for p in obs.players]
        s.hole = [None] * s.n
        s.hole[obs.seat] = tuple(card_id(c) for c in obs.hole_cards)
        s.board = tuple(card_id(c) for c in obs.community_cards)
        s.street = STREET_NAMES.index(obs.street)
        s.current_bet, s.min_raise, s.acted = _replay_street(obs)
        s.to_act = obs.seat
        return s

    def copy(self):
        c = SimState.__new__(SimState)
        c.n = self.n
        c.dealer = self.dealer
        c.big_blind = self.big_blind
        c.stack = self.stack[:]
        c.bet = self.bet[:]
        c.total = self.total[:]
        c.folded = self.folded[:]
        c.all_in = self.all_in[:]
        c.hole = self.hole[:]
        c.board = self.board
        c.street = self.street
        c.current_bet = self.current_bet
        c.min_raise = self.min_raise
        c.acted = self.acted
        c.to_act = self.to_act
        return c

    @property
    def pot(self):
        return sum(self.total)

    def actions(self, fractions=(0.5, 1.0)):
        """Abstract actions as (kind, raise_to) tuples: fold only when facing a bet, raises sized by pot fraction plus all-in."""
        p = self.to_act
        to_call = self.current_bet - self.bet[p]
        acts = [(FOLD, 0), (CALL, 0)] if to_call > 0 else [(CHECK, 0)]
        if self.stack[p] > to_call:
            cap = self.bet[p] + self.stack[p]
            low = min(self.current_bet + self.min_raise, cap)
            pot_after_call = self.pot + to_call
            targets = []
            for f in fractions:
                target = max(low, min(cap, self.current_bet + int(f * pot_after_call)))
                if target not in targets:
                    targets.append(target)
            if cap not in targets:
                targets.append(cap)
            acts.extend((RAISE, t) for t in targets)
        return acts

    def _post(self, p, amount):
        amount = max(0, min(amount, self.stack[p]))
        self.stack[p] -= amount
        self.bet[p] += amount
        self.total[p] += amount
        if self.stack[p] == 0:
            self.all_in[p] = True
        if self.bet[p] > self.current_bet:
            self.current_bet = self.bet[p]

    def apply(self, action):
        kind, amount = action
        p = self.to_act
        raised = False
        if kind == FOLD:
            self.folded[p] = True
        elif kind == CALL:
            self._post(p, self.current_bet - self.bet[p])
        elif kind == RAISE:
            previous = self.current_bet
            self._post(p, amount - self.bet[p])
            if self.bet[p] - previous >= self.min_raise:
                self.min_raise = self.bet[p] - previous
            raised = True
        self.acted = (1 << p) if raised else self.acted | (1 << p)
        self._advance(p + 1)

    def _advance(self, pointer):
        """Mirror of PokerGame._betting_round: pick the next seat to act, rolling streets forward."""
        n = self.n
        while True:
            live = [s for s in range(n) if not self.folded[s]]
            able = [s for s in live if not self.all_in[s]]
            done = len(live) < 2 or not able
            if not done and len(able) == 1 and self.bet[able[0]] >= self.current_bet:
                done = True
            if not done:
                pending = [s for s in able if not self.acted >> s & 1 or self.bet[s] != self.current_bet]
                if pending:
                    for k in range(n):
                        s = (pointer + k) % n
                        if s in pending:
                            self.to_act = s
                            return
            if len(live) < 2 or self.street == len(STREETS) - 1:
                self.to_act = -1
                return
            self.street += 1
            self.bet = [0] * n
            self.current_bet = 0
            self.min_raise = self.big_blind
            self.acted = 0
            pointer = self.dealer + 1

    def _settle(self, totals):
        live = [s for s in range(self.n) if not self.folded[s]]
        scores = {}
        if len(live) > 1:
            for s in live:
                scores[s] = score_cards(self.hole[s] + self.board)
        won = split_pots(totals, self.folded, scores, self.dealer)
        return [won[s] - totals[s] for s in range(self.n)]

    def payoffs(self):
        """Net chips per seat; valid once the hand is over and all hole cards and the board are set."""
        return self._settle(self.total)

    def passive_payoffs(self):
        """Net chips per seat if everyone still in simply checks and calls down to showdown."""
        if self.to_act < 0:
            return self.payoffs()
        totals = self.total[:]
        for s in range(self.n):
            if not self.folded[s]:
                totals[s] += min(self.current_bet - self.bet[s], self.stack[s])
        return self._settle(totals)
