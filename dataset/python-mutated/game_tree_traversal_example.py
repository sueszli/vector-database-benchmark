"""Example to traverse an entire game tree."""
from absl import app
from absl import flags
from open_spiel.python import games
import pyspiel
_GAME_STRING = flags.DEFINE_string('game_string', 'tic_tac_toe', 'Name of the game')

class GameStats:
    num_states: int = 0
    num_chance_nodes: int = 0
    num_decision_nodes: int = 0
    num_simultaneous_nodes: int = 0
    num_terminals: int = 0
    info_state_dict: dict[str, list[int]] = {}

    def __str__(self):
        if False:
            print('Hello World!')
        return f'Number of states {self.num_states} \n' + f'Number of chance nodes {self.num_chance_nodes} \n' + f'Number of decision nodes {self.num_decision_nodes} \n' + f'Number of simultaneous nodes {self.num_simultaneous_nodes} \n' + f'Number of terminals {self.num_terminals} \n'

def traverse_game_tree(game: pyspiel.Game, state: pyspiel.State, game_stats: GameStats):
    if False:
        print('Hello World!')
    'Traverses the game tree, collecting information about the game.'
    if state.is_terminal():
        game_stats.num_terminals += 1
    elif state.is_chance_node():
        game_stats.num_chance_nodes += 1
        for outcome in state.legal_actions():
            child = state.child(outcome)
            traverse_game_tree(game, child, game_stats)
    elif state.is_simultaneous_node():
        game_stats.num_simultaneous_nodes += 1
        for joint_action in state.legal_actions():
            child = state.child(joint_action)
            traverse_game_tree(game, child, game_stats)
    else:
        game_stats.num_decision_nodes += 1
        legal_actions = state.legal_actions()
        if game.get_type().provides_information_state_string:
            game_stats.info_state_dict[state.information_state_string()] = legal_actions
        for action in state.legal_actions():
            child = state.child(action)
            traverse_game_tree(game, child, game_stats)

def main(_):
    if False:
        print('Hello World!')
    game = pyspiel.load_game(_GAME_STRING.value)
    game_stats = GameStats()
    state = game.new_initial_state()
    traverse_game_tree(game, state, game_stats)
    print(game_stats)
if __name__ == '__main__':
    app.run(main)