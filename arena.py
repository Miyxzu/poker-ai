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

from poker import CALL, CHECK, FOLD, RAISE, CallAgent, Deck, HumanAgent, PokerGame, RandomAgent
from ismcts import ISMCTSAgent

AGENT_KINDS = {
    "ismcts": ("ISMCTS", lambda name, args, rng: ISMCTSAgent(name, args.iterations, args.exploration, rng=rng)),
    "random": ("Random", lambda name, args, rng: RandomAgent(name, rng)),
    "call": ("Call", lambda name, args, rng: CallAgent(name)),
}

STREETS = ("preflop", "flop", "turn", "river")


class ArenaResults(dict):
    """Per-deal bb results with aggregate hand, action, and position statistics."""

    def __init__(self, per_deal, agent_stats, position_stats, street_actions):
        super().__init__(per_deal)
        self.agent_stats = agent_stats
        self.position_stats = position_stats
        self.street_actions = street_actions


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
    """Plays shared deals across seat orders and returns per-deal results plus performance stats."""
    rng = random.Random(seed)
    agents = list(agents)
    if len(agents) <= 4:
        orders = list(itertools.permutations(agents))
    else:  # n! orders get too slow, so only rotate
        orders = [agents[k:] + agents[:k] for k in range(len(agents))]
    results = {a.name: [] for a in agents}
    agent_stats = {name: Counter() for name in results}
    position_stats = {name: {} for name in results}
    street_actions = {name: {street: Counter() for street in STREETS} for name in results}
    for _ in range(deals):
        cards = Deck(rng).cards
        totals = dict.fromkeys(results, 0.0)
        for order in orders:
            game = PokerGame(list(order), stack, small_blind, big_blind, verbose=False,
                             deck_factory=lambda: Deck(stacked=cards))
            hand_result = game.play_hand()
            net = hand_result["net"]
            winnings = hand_result["winnings"]
            showdown = hand_result["showdown"]

            if len(order) == 2:
                positions = {game.dealer: "BTN/SB", (game.dealer + 1) % 2: "BB"}
            else:
                positions = {
                    game.dealer: "BTN",
                    (game.dealer + 1) % len(order): "SB",
                    (game.dealer + 2) % len(order): "BB",
                }
                for offset in range(3, len(order)):
                    positions[(game.dealer + offset) % len(order)] = "Other"

            vpip = set()
            pfr = set()
            for action in hand_result["history"]:
                agent = order[action["seat"]]
                if action["street"] == "preflop":
                    if action["action"] in (CALL, RAISE):
                        vpip.add(agent.name)
                    if action["action"] == RAISE:
                        pfr.add(agent.name)
                street_actions[agent.name][action["street"]][action["action"]] += 1

            for seat, agent in enumerate(order):
                totals[agent.name] += net[seat] / big_blind
                stats = agent_stats[agent.name]
                stats["hands"] += 1
                stats["pot_wins"] += winnings[seat] > 0
                stats["vpip"] += agent.name in vpip
                stats["pfr"] += agent.name in pfr
                stats["showdowns"] += seat in showdown
                stats["showdown_wins"] += seat in showdown and winnings[seat] > 0

                position = positions[seat]
                position_result = position_stats[agent.name].setdefault(position, [0.0, 0])
                position_result[0] += net[seat] / big_blind
                position_result[1] += 1
        for name, total in totals.items():
            results[name].append(total / len(orders))
    return ArenaResults(results, agent_stats, position_stats, street_actions)


def report(results):
    agent_stats = getattr(results, "agent_stats", None)
    if agent_stats is None:
        print(f"{'agent':<12}{'bb/100 hands':>14}   95% CI")
        for name, per_deal in results.items():
            mean = statistics.fmean(per_deal)
            half = 1.96 * statistics.stdev(per_deal) / math.sqrt(len(per_deal)) if len(per_deal) > 1 else 0.0
            print(f"{name:<12}{mean * 100:>+14.1f}   +/- {half * 100:.1f}")
        return

    print("Performance")
    print(f"{'agent':<12}{'bb/100':>9}{'95% CI':>12}{'median':>10}{'lose deal%':>12}"
          f"{'pot win%':>10}{'VPIP%':>8}{'PFR%':>7}{'SD%':>7}{'SD win%':>10}")
    for name, per_deal in results.items():
        mean = statistics.fmean(per_deal)
        half = 1.96 * statistics.stdev(per_deal) / math.sqrt(len(per_deal)) if len(per_deal) > 1 else 0.0
        stats = agent_stats[name]
        hands = stats["hands"]
        percent = lambda count, total: 100 * count / total if total else 0.0
        print(f"{name:<12}{mean * 100:>+9.1f}{half * 100:>+12.1f}{statistics.median(per_deal) * 100:>+10.1f}"
              f"{percent(sum(value < 0 for value in per_deal), len(per_deal)):>12.1f}"
              f"{percent(stats['pot_wins'], hands):>10.1f}{percent(stats['vpip'], hands):>8.1f}"
              f"{percent(stats['pfr'], hands):>7.1f}{percent(stats['showdowns'], hands):>7.1f}"
              f"{percent(stats['showdown_wins'], stats['showdowns']):>10.1f}")

    print("\nAction frequency by street (% of decisions on that street)")
    print(f"{'agent':<12}{'street':<10}{'fold':>8}{'check':>8}{'call':>8}{'raise':>8}")
    for name in results:
        for street in STREETS:
            actions = results.street_actions[name][street]
            decisions = sum(actions.values())
            percent = lambda action: 100 * actions[action] / decisions if decisions else 0.0
            print(f"{name:<12}{street:<10}{percent(FOLD):>8.1f}{percent(CHECK):>8.1f}"
                  f"{percent(CALL):>8.1f}{percent(RAISE):>8.1f}")

    print("\nResults by position (bb/100 hands)")
    print(f"{'agent':<12}{'position':<10}{'bb/100':>9}")
    for name in results:
        for position, (total_bb, hands) in results.position_stats[name].items():
            print(f"{name:<12}{position:<10}{total_bb / hands * 100:>+9.1f}")


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
