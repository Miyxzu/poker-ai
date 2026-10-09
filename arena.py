"""Seat-rotated tournaments on shared deals, so luck of the cards cancels out.

Evaluate:        python arena.py --deals 30 --iterations 200
Play vs ISMCTS:  python arena.py --play
"""
import argparse
import itertools
import math
import random
import statistics

from poker import CallAgent, Deck, HumanAgent, PokerGame, RandomAgent
from ismcts import ISMCTSAgent


def run_arena(agents, deals=50, stack=1000, small_blind=5, big_blind=10, seed=0):
    """Plays every seat order of each deal once; returns {name: per-deal average result in big blinds per hand}."""
    rng = random.Random(seed)
    orders = list(itertools.permutations(agents))
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
    parser.add_argument("--play", action="store_true", help="play against two ISMCTS bots")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    def ismcts(name):
        return ISMCTSAgent(name, args.iterations, args.exploration, rng=random.Random(rng.random()))

    if args.play:
        agents = [HumanAgent("You"), ismcts("ISMCTS 1"), ismcts("ISMCTS 2")]
        PokerGame(agents, args.stack, seed=args.seed).play_match()
        return
    agents = [ismcts("ISMCTS"), RandomAgent("Random", random.Random(rng.random())), CallAgent("Call")]
    report(run_arena(agents, args.deals, args.stack, seed=args.seed))


if __name__ == "__main__":
    main()
