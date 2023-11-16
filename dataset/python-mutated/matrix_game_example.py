"""Python spiel example."""
import random
from absl import app
import numpy as np
import pyspiel
from open_spiel.python.utils import file_utils

def _manually_create_game():
    if False:
        i = 10
        return i + 15
    'Creates the game manually from the spiel building blocks.'
    game_type = pyspiel.GameType('matching_pennies', 'Matching Pennies', pyspiel.GameType.Dynamics.SIMULTANEOUS, pyspiel.GameType.ChanceMode.DETERMINISTIC, pyspiel.GameType.Information.ONE_SHOT, pyspiel.GameType.Utility.ZERO_SUM, pyspiel.GameType.RewardModel.TERMINAL, 2, 2, True, True, False, False, dict())
    game = pyspiel.MatrixGame(game_type, {}, ['Heads', 'Tails'], ['Heads', 'Tails'], [[-1, 1], [1, -1]], [[1, -1], [-1, 1]])
    return game

def _easy_create_game():
    if False:
        print('Hello World!')
    'Uses the helper function to create the same game as above.'
    return pyspiel.create_matrix_game('matching_pennies', 'Matching Pennies', ['Heads', 'Tails'], ['Heads', 'Tails'], [[-1, 1], [1, -1]], [[1, -1], [-1, 1]])

def _even_easier_create_game():
    if False:
        for i in range(10):
            print('nop')
    'Leave out the names too, if you prefer.'
    return pyspiel.create_matrix_game([[-1, 1], [1, -1]], [[1, -1], [-1, 1]])

def _import_data_create_game():
    if False:
        print('Hello World!')
    'Creates a game via imported payoff data.'
    payoff_file = file_utils.find_file('open_spiel/data/paper_data/response_graph_ucb/soccer.txt', 2)
    payoffs = np.loadtxt(payoff_file) * 2 - 1
    return pyspiel.create_matrix_game(payoffs, payoffs.T)

def main(_):
    if False:
        while True:
            i = 10
    games_list = pyspiel.registered_games()
    print('Registered games:')
    print(games_list)
    blotto_matrix_game = pyspiel.load_matrix_game('blotto')
    print('Number of rows in 2-player Blotto with default settings is {}'.format(blotto_matrix_game.num_rows()))
    print('Creating matrix game...')
    game = pyspiel.load_matrix_game('matrix_mp')
    game = _manually_create_game()
    game = _import_data_create_game()
    game = _easy_create_game()
    game = _even_easier_create_game()
    print('Values for joint action ({},{}) is {},{}'.format(game.row_action_name(0), game.col_action_name(0), game.player_utility(0, 0, 0), game.player_utility(1, 0, 0)))
    state = game.new_initial_state()
    print('State:')
    print(str(state))
    assert state.is_simultaneous_node()
    chosen_actions = [random.choice(state.legal_actions(pid)) for pid in range(game.num_players())]
    print('Chosen actions: ', [state.action_to_string(pid, action) for (pid, action) in enumerate(chosen_actions)])
    state.apply_actions(chosen_actions)
    assert state.is_terminal()
    returns = state.returns()
    for pid in range(game.num_players()):
        print('Utility for player {} is {}'.format(pid, returns[pid]))
if __name__ == '__main__':
    app.run(main)