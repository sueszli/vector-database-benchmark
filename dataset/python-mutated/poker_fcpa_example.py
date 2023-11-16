"""Python spiel example."""
from absl import app
from absl import flags
import numpy as np
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel
FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 12761381, 'The seed to use for the RNG.')
flags.DEFINE_string('player0', 'random', 'Type of the agent for player 0.')
flags.DEFINE_string('player1', 'random', 'Type of the agent for player 1.')

def LoadAgent(agent_type, game, player_id, rng):
    if False:
        i = 10
        return i + 15
    'Return a bot based on the agent type.'
    if agent_type == 'random':
        return uniform_random.UniformRandomBot(player_id, rng)
    elif agent_type == 'human':
        return human.HumanBot()
    elif agent_type == 'check_call':
        policy = pyspiel.PreferredActionPolicy([1, 0])
        return pyspiel.make_policy_bot(game, player_id, FLAGS.seed, policy)
    elif agent_type == 'fold':
        policy = pyspiel.PreferredActionPolicy([0, 1])
        return pyspiel.make_policy_bot(game, player_id, FLAGS.seed, policy)
    else:
        raise RuntimeError('Unrecognized agent type: {}'.format(agent_type))

def main(_):
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(FLAGS.seed)
    games_list = pyspiel.registered_names()
    assert 'universal_poker' in games_list
    fcpa_game_string = pyspiel.hunl_game_string('fcpa')
    print('Creating game: {}'.format(fcpa_game_string))
    game = pyspiel.load_game(fcpa_game_string)
    agents = [LoadAgent(FLAGS.player0, game, 0, rng), LoadAgent(FLAGS.player1, game, 1, rng)]
    state = game.new_initial_state()
    print('INITIAL STATE')
    print(str(state))
    while not state.is_terminal():
        current_player = state.current_player()
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            num_actions = len(outcomes)
            print('Chance node with ' + str(num_actions) + ' outcomes')
            (action_list, prob_list) = zip(*outcomes)
            action = rng.choice(action_list, p=prob_list)
            print('Sampled outcome: ', state.action_to_string(state.current_player(), action))
            state.apply_action(action)
        else:
            legal_actions = state.legal_actions()
            for action in legal_actions:
                print('Legal action: {} ({})'.format(state.action_to_string(current_player, action), action))
            action = agents[current_player].step(state)
            action_string = state.action_to_string(current_player, action)
            print('Player ', current_player, ', chose action: ', action_string)
            state.apply_action(action)
        print('')
        print('NEXT STATE:')
        print(str(state))
    returns = state.returns()
    for pid in range(game.num_players()):
        print('Utility for player {} is {}'.format(pid, returns[pid]))
if __name__ == '__main__':
    app.run(main)