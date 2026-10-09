# Repo for Poker Proj

Play against two bots with python poker.py.
Run a bot-only match with python poker.py --sim --hands 200.
Run the tests with python -m unittest poker test_ismcts -v.

ISMCTS agent (ismcts.py, built on the simulator in sim.py):
Compare it with the random and call-down bots with python arena.py --deals 400.
Play against ISMCTS bots with python arena.py --play.
Both take --players N (2 to 6 seats, default 3).
Pick the table with --lineup, e.g. python arena.py --lineup ismcts,ismcts,random (kinds: ismcts, random, call).
With --play the lineup is your opponents, e.g. python arena.py --play --lineup ismcts,random.
