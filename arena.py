"""Seat-rotated tournaments on shared deals, so luck of the cards cancels out.

Evaluate:        python arena.py --deals 30 --iterations 200 --players 4
Choose agents:   python arena.py --lineup ismcts,ismcts,random
Play vs agents:  python arena.py --play --lineup ismcts,random   (you plus the listed opponents)

Agent kinds: ismcts, random (RandomAgent), call (check/call-down).
"""
import argparse
import itertools
import math
import random
import statistics
from collections import Counter

from poker import CallAgent, Deck, HumanAgent, PokerGame, RandomAgent
from ismcts import ISMCTSAgent

AGENT_KINDS = {
    "ismcts": ("ISMCTS", lambda name, args, rng: ISMCTSAgent(name, args.iterations, args.exploration, rng=rng)),
    "random": ("Random", lambda name, args, rng: RandomAgent(name, rng)),
    "call": ("Call", lambda name, args, rng: CallAgent(name)),
}


def build_lineup(kinds, args, rng):
    """Creates one agent per kind; repeated kinds are numbered so every name is unique."""
    totals = Counter(kinds)
    seen = Counter()
    agents = []
    for kind in kinds:
        label, factory = AGENT_KINDS[kind]
        seen[kind] += 1
        name = f"{label} {seen[kind]}" if totals[kind] > 1 else label
        agents.append(factory(name, args, random.Random(rng.random())))
    return agents


def run_arena(agents, deals=50, stack=1000, small_blind=5, big_blind=10, seed=0):
    """Plays each deal once per seat order; returns {name: per-deal average result in big blinds per hand}."""
    rng = random.Random(seed)
    agents = list(agents)
    if len(agents) <= 4:
        orders = list(itertools.permutations(agents))
    else:  # n! orders get too slow, so only rotate
        orders = [agents[k:] + agents[:k] for k in range(len(agents))]
    results = {a.name: [] for a in agents}
    for _ in range(deals):
        cards = Deck(rng).cards
        totals = dict.fromkeys(results, 0.0)
        for order in orders:
            game = PokerGame(list(order), stack, small_blind, big_blind, verbose=False,
                             deck_factory=lambda: Deck(stacked=cards))
            net = game.play_hand()["net"]
            for seat, agent in enumerate(order):
                totals[agent.name] += net[seat] / big_blind
        for name, total in totals.items():
            results[name].append(total / len(orders))
    return results


def report(results):
    print(f"{'agent':<12}{'bb/100 hands':>14}   95% CI")
    for name, per_deal in results.items():
        mean = statistics.fmean(per_deal)
        half = 1.96 * statistics.stdev(per_deal) / math.sqrt(len(per_deal)) if len(per_deal) > 1 else 0.0
        print(f"{name:<12}{mean * 100:>+14.1f}   +/- {half * 100:.1f}")


def main():
    parser = argparse.ArgumentParser(description="ISMCTS vs baseline agents")
    parser.add_argument("--deals", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200, help="ISMCTS iterations per decision")
    parser.add_argument("--exploration", type=float, default=0.7)
    parser.add_argument("--stack", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lineup", help="comma-separated agent kinds, e.g. ismcts,ismcts,random")
    parser.add_argument("--players", type=int, default=3, choices=range(2, 7),
                        help="total seats when --lineup is not given")
    parser.add_argument("--play", action="store_true", help="play against the agents (you take an extra seat)")
    args = parser.parse_args()

    if args.lineup:
        kinds = [k.strip().lower() for k in args.lineup.split(",") if k.strip()]
        unknown = [k for k in kinds if k not in AGENT_KINDS]
        if unknown:
            parser.error(f"unknown agent {unknown[0]!r}; choose from {', '.join(AGENT_KINDS)}")
    elif args.play:
        kinds = ["ismcts"] * (args.players - 1)
    else:
        kinds = ["ismcts"] + [("random", "call")[i % 2] for i in range(args.players - 1)]
    if not 2 <= len(kinds) + args.play <= 6:
        parser.error("need 2 to 6 players in total")

    agents = build_lineup(kinds, args, random.Random(args.seed))
    if args.play:
        PokerGame([HumanAgent("You")] + agents, args.stack, seed=args.seed).play_match()
        return
    report(run_arena(agents, args.deals, args.stack, seed=args.seed))


if __name__ == "__main__":
    main()
