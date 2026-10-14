# Repo for Poker Proj

``poker.py`` contains the engine for the No-limit Hold'em Poker used for this case study

## Arena.py

Contains 4 different agents that are avaliable to use

- Call: Plays passively by deciding to check or call, never folding or raising. Will always go to showdown.
- Random: Picks an action based on weighted probablity and availablity; 60% Check/Call, 15% Fold (Only when facing a bet), 25% Raise (Only when raising is possible)
- ISMCTS: Uses Information Set Monte Carlo Tree Search to determine actions; Trained using the ``sim.py`` file to improve its strategy
- Tabular Monte Carlo Q-Learning (Qlearn): Uses a combination of a Tabular Q-Learning agent and Monte Carlo learning to play (ML solution); Uses Q-table to decide action via Markov Decision Process

### Arguments

Main arguments to use, see ``-h`` for the rest

``--lineup``: Pick the agents to use for the table (See agents)

``--players``: Number of agents to use, 2 to 6 seats (Default 4 w/ISMCTS, Qlearn, Random and Call; 3 w/--play)

``--play``: Play against the agents (Default: ISMCTS and Qlearn), can be used in tandem with --lineup

``--iterations``: Adjust ISMCTS iterations per decision

## Extra Info

### Tabular Monte Carlo Q-Learning

Train Q-Learning agent using ``python qlearn.py --hands 200000 --out qtable.json``  (--minutes N to cap, --resume to continue, --opponents random,call,ismcts)

### Unit Testing

Run unit tests for ISMCTS agent: ``python -m unittest poker test_ismcts -v``

Run unit tests for Qlearn agent: ``python -m unittest poker test_qlearn -v``
