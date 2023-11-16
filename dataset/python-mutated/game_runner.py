"""Utils for computing gradient information: run games and record payoffs.
"""
import itertools
from absl import logging
import numpy as np

def construct_game_queries(base_profile, num_checkpts):
    if False:
        print('Hello World!')
    "Constructs a list of checkpoint selection tuples to query value function.\n\n  Each query tuple (key, query) where key = (pi, pj) and query is\n  (p1's selected checkpt, ..., p7's selected checkpt) fixes the players in the\n  game of diplomacy to be played. It may be necessary to play several games with\n  the same players to form an accurate estimate of the value or payoff for each\n  player as checkpts contain stochastic policies.\n\n  Args:\n    base_profile: list of selected checkpts for each player, i.e.,\n      a sample from the player strategy profile ([x_i ~ p(x_i)])\n    num_checkpts: list of ints, number of strats (or ckpts) per player\n  Returns:\n    Set of query tuples containing a selected checkpoint index for each player.\n  "
    new_queries = set([])
    num_players = len(base_profile)
    for (pi, pj) in itertools.combinations(range(num_players), 2):
        new_profile = list(base_profile)
        for ai in range(num_checkpts[pi]):
            new_profile[pi] = ai
            for aj in range(num_checkpts[pj]):
                new_profile[pj] = aj
                query = tuple(new_profile)
                pair = (pi, pj)
                new_queries.update([(pair, query)])
    return new_queries

def construct_game_queries_for_exp(base_profile, num_checkpts):
    if False:
        for i in range(10):
            print('nop')
    "Constructs a list of checkpoint selection tuples to query value function.\n\n  Each query tuple (key, query) where key = (pi,) and query is\n  (p1's selected checkpt, ..., p7's selected checkpt) fixes the players in the\n  game of diplomacy to be played. It may be necessary to play several games with\n  the same players to form an accurate estimate of the value or payoff for each\n  player as checkpts contain stochastic policies.\n\n  Args:\n    base_profile: list of selected checkpts for each player, i.e.,\n      a sample from the player strategy profile ([x_i ~ p(x_i)])\n    num_checkpts: list of ints, number of strats (or ckpts) per player\n  Returns:\n    Set of query tuples containing a selected checkpoint index for each player.\n  "
    new_queries = set([])
    num_players = len(base_profile)
    for pi in range(num_players):
        new_profile = list(base_profile)
        for ai in range(num_checkpts[pi]):
            new_profile[pi] = ai
            query = tuple(new_profile)
            new_queries.update([(pi, query)])
    return new_queries

def run_games_and_record_payoffs(game_queries, evaluate_game, ckpt_to_policy):
    if False:
        for i in range(10):
            print('nop')
    'Simulate games according to game queries and return results.\n\n  Args:\n    game_queries: set of tuples containing indices specifying each players strat\n      key_query = (agent_tuple, profile_tuple) format\n    evaluate_game: callable function that takes a list of policies as argument\n    ckpt_to_policy: list of maps from strat (or checkpoint) to a policy, one\n      map for each player\n  Returns:\n    dictionary: key=key_query, value=np.array of payoffs (1 for each player)\n  '
    game_results = {}
    for key_query in game_queries:
        (_, query) = key_query
        policies = [ckpt_to_policy[pi][ckpt_i] for (pi, ckpt_i) in enumerate(query)]
        payoffs = evaluate_game(policies)
        game_results.update({key_query: payoffs})
    return game_results

def form_payoff_matrices(game_results, num_checkpts):
    if False:
        while True:
            i = 10
    'Packages dictionary of game results into a payoff tensor.\n\n  Args:\n    game_results: dictionary of payoffs for each game evaluated, keys are\n      (pair, profile) where pair is a tuple of the two agents played against\n      each other and profile indicates pure joint action played by all agents\n    num_checkpts: list of ints, number of strats (or ckpts) per player\n  Returns:\n    payoff_matrices: dict of np.arrays (2 x num_checkpts x num_checkpts) with\n      payoffs for two players. keys are pairs above with lowest index agent\n      first\n  '
    num_players = len(num_checkpts)
    payoff_matrices = {}
    for (pi, pj) in itertools.combinations(range(num_players), 2):
        key = (pi, pj)
        payoff_matrices[key] = np.zeros((2, num_checkpts[pi], num_checkpts[pj]))
    for (key_profile, payoffs) in game_results.items():
        (key, profile) = key_profile
        (i, j) = key
        ai = profile[i]
        aj = profile[j]
        payoff_matrices[key][0, ai, aj] = payoffs[i]
        payoff_matrices[key][1, ai, aj] = payoffs[j]
    return payoff_matrices