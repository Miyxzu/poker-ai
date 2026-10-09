# Repo for Poker Proj

Play against two bots with python poker.py.
Run a bot-only match with python poker.py --sim --hands 200.
Run the tests with python -m unittest poker test_ismcts -v.

ISMCTS agent (ismcts.py, built on the simulator in sim.py):
Compare it with the random and call-down bots with python arena.py --deals 400.
Play against two ISMCTS bots with python arena.py --play.
