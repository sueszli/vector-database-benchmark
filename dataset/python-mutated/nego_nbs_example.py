"""An example that computes the Nash bargaining score from negotiations.

This uses the bargaining game that was introduced in:

[1] Lewis et al., Deal or no deal? End-to-end learning of negotiation
    dialogues, 2017. https://arxiv.org/abs/1706.05125
[2] David DeVault, Johnathan Mell, and Jonathan Gratch.
    2015. Toward Natural Turn-taking in a Virtual Human Negotiation Agent

It computes the empirical Nash bargaining score (NBS) from three sources:
  - Human play
  - IS-MCTS in self-play
  - A theoretical maximum NBS if the players had full information and can see
    each other's utilities and then maximize their NBS.

These are all run on a data set extracted from the Lewis et al. '17 data set:
https://github.com/facebookresearch/end-to-end-negotiator/blob/master/src/data/negotiate/data.txt

This example is inspired by the paper (Iwasa and Fujita, "Prediction of Nash
Bargaining Solution in Negotiation Dialogue", 2018).
"""
from absl import app
from absl import flags
import numpy as np
from open_spiel.python import games
import pyspiel
FLAGS = flags.FLAGS
flags.DEFINE_string('data_file', None, "Lewis et al. '17 data set file")
flags.DEFINE_string('instances_file', '/tmp/instances.txt', 'Filename for the temp instances database file.')

class Instance(object):
    """An instance of a bargaining problem."""

    def __init__(self, pool, p1values, p2values):
        if False:
            for i in range(10):
                print('nop')
        self.pool = np.array(pool)
        self.p1values = np.array(p1values)
        self.p2values = np.array(p2values)
        assert 5 <= sum(pool) <= 7
        assert np.dot(pool, p1values) == 10
        assert np.dot(pool, p2values) == 10

    def __str__(self):
        if False:
            return 10
        return ','.join([str(x) for x in self.pool]) + ' ' + ','.join([str(x) for x in self.p1values]) + ' ' + ','.join([str(x) for x in self.p2values])

class Negotiation(object):
    """An instance of a bargaining game."""

    def __init__(self, instance, outcome, rewards):
        if False:
            return 10
        self.instance = instance
        self.outcome = outcome
        self.rewards = rewards

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return str(self.instance) + ' ' + str(self.outcome) + ' ' + str(self.rewards)

def dialogue_matches_prev_line(line1, line2):
    if False:
        i = 10
        return i + 15
    "Checks if the dialogue matches the previous line's."
    parts1 = line1.split(' ')
    parts2 = line2.split(' ')
    for i in range(6, min(len(parts1), len(parts2))):
        if parts1[i] == 'YOU:' or parts1[i] == 'THEM:':
            if parts1[i] == 'YOU:' and parts2[i] != 'THEM:':
                return False
            if parts1[i] == 'THEM:' and parts2[i] != 'YOU:':
                return False
        elif parts1[i] != parts2[i]:
            return False
        if parts1[i] == '<selection>':
            break
    return True

def parse_dataset(filename):
    if False:
        for i in range(10):
            print('nop')
    "Parse the Lewis et al. '17 data file."
    contents = pyspiel.read_contents_from_file(filename, 'r')
    lines = contents.split('\n')
    cur_nego = None
    negotiations = []
    instances = []
    for line_no in range(len(lines)):
        line = lines[line_no]
        if line:
            parts = line.split(' ')
            pool = [int(parts[0]), int(parts[2]), int(parts[4])]
            my_values = [int(parts[1]), int(parts[3]), int(parts[5])]
            pool2 = [int(parts[-6]), int(parts[-4]), int(parts[-2])]
            other_values = [int(parts[-5]), int(parts[-3]), int(parts[-1])]
            assert pool == pool2
            rewards = [0, 0]
            add_nego = False
            outcome_str = parts[-7]
            if parts[6] == 'YOU:':
                player_id = 0
                instance = Instance(pool, my_values, other_values)
            elif parts[6] == 'THEM:':
                player_id = 1
                instance = Instance(pool, other_values, my_values)
            else:
                assert False, parts[6]
            outcome = False
            my_reward = 0
            instances.append(instance)
            if 'disconnect' in line:
                continue
            if outcome_str == 'disagree' or parts[-9] + ' ' + parts[-8] == 'reward=no agreement' or parts[-8] == 'reward=disconnect':
                add_nego = False
            elif outcome_str == 'agree':
                outcome = True
                reward_parts = parts[-8].split('=')
                assert len(reward_parts) == 2, f'reward parts str: {parts[-8]}'
                assert reward_parts[0] == 'reward'
                my_reward = int(reward_parts[1])
            else:
                assert False, f'Bad outcome: {outcome_str}'
            if cur_nego is None:
                rewards[player_id] = my_reward
                if player_id == 0:
                    cur_nego = Negotiation(instance, outcome, rewards)
                else:
                    cur_nego = Negotiation(instance, outcome, rewards)
            elif dialogue_matches_prev_line(line, lines[line_no - 1]):
                assert list(cur_nego.instance.pool) == pool
                if player_id == 1:
                    assert list(cur_nego.instance.p2values) == my_values
                    assert list(cur_nego.instance.p1values) == other_values
                elif player_id == 0:
                    assert list(cur_nego.instance.p1values) == my_values
                    assert list(cur_nego.instance.p2values) == other_values
                cur_nego.rewards[player_id] = my_reward
                add_nego = True
            else:
                rewards[player_id] = my_reward
                if player_id == 0:
                    cur_nego = Negotiation(instance, outcome, rewards)
                else:
                    cur_nego = Negotiation(instance, outcome, rewards)
                add_nego = False
            if add_nego or outcome_str == 'disagree':
                negotiations.append(cur_nego)
                print(str(cur_nego))
                print(len(negotiations))
                cur_nego = None
                if outcome_str != 'disagree':
                    instances.pop()
    return (instances, negotiations)

def write_instances_file(negotiations, filename):
    if False:
        i = 10
        return i + 15
    contents = ''
    for nego in negotiations:
        contents += str(nego.instance) + '\n'
    pyspiel.write_contents_to_file(filename, 'w', contents)

def compute_nbs_from_simulations(game, num_games, bots):
    if False:
        print('Hello World!')
    'Compute empirical NBS from simulations.'
    avg_returns = np.zeros(game.num_players())
    for _ in range(num_games):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                (action_list, prob_list) = zip(*outcomes)
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)
            else:
                player = state.current_player()
                action = bots[player].step(state)
                state.apply_action(action)
        returns = np.asarray(state.returns())
        avg_returns += returns
    avg_returns /= num_games
    return np.prod(avg_returns)

class MaxBot(object):
    """Finds the single (deterministic) trade offer that maximizes the NBS."""

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def step(self, state):
        if False:
            print('Hello World!')
        "Returns the NBS-maximizing action.\n\n    If i'm player 0, then search over all possible moves, assume player 2\n    takes the agree action, and choose the action that maximizes the NBS\n    Player 1 just always agrees.\n\n    Args:\n      state: the OpenSpiel state to act from.\n    "
        player = state.current_player()
        if player == 1:
            return state.agree_action()
        max_nbs = -1
        max_action = -1
        for action in state.legal_actions():
            state_clone = state.clone()
            state_clone.apply_action(action)
            state_clone.apply_action(state.agree_action())
            returns = state_clone.returns()
            nbs = np.prod(returns)
            if nbs > max_nbs:
                max_nbs = nbs
                max_action = action
        assert max_action >= 0
        return max_action

def main(_):
    if False:
        return 10
    assert FLAGS.data_file is not None
    (_, negotiations) = parse_dataset(FLAGS.data_file)
    print(f'Writing instances database: {FLAGS.instances_file}')
    write_instances_file(negotiations, FLAGS.instances_file)
    human_rewards = np.zeros(2, dtype=np.float64)
    avg_human_nbs = 0
    for neg in negotiations:
        human_rewards += neg.rewards
    human_rewards /= len(negotiations)
    avg_human_nbs += np.prod(human_rewards)
    print(f'Average human rewards: {human_rewards}')
    print(f'Average human NBS: {avg_human_nbs}')
    game = pyspiel.load_game('bargaining', {'instances_file': FLAGS.instances_file})
    bots = [MaxBot(), MaxBot()]
    avg_max_nbs = compute_nbs_from_simulations(game, 6796, bots)
    print(f'Average max NBS: {avg_max_nbs}')
    bots = [pyspiel.make_uniform_random_bot(0, np.random.randint(0, 1000000)), pyspiel.make_uniform_random_bot(1, np.random.randint(0, 1000000))]
    avg_uniform_nbs = compute_nbs_from_simulations(game, 6796, bots)
    print(f'Average uniform NBS: {avg_uniform_nbs}')
    evaluator = pyspiel.RandomRolloutEvaluator(1, np.random.randint(0, 1000000))
    bots = [pyspiel.ISMCTSBot(np.random.randint(0, 1000000), evaluator, 10.0, 1000, -1, pyspiel.ISMCTSFinalPolicyType.MAX_VISIT_COUNT, False, False), pyspiel.ISMCTSBot(np.random.randint(0, 1000000), evaluator, 10.0, 1000, -1, pyspiel.ISMCTSFinalPolicyType.MAX_VISIT_COUNT, False, False)]
    avg_ismcts_nbs = compute_nbs_from_simulations(game, 6796, bots)
    print(f'Average IS-MCTS NBS: {avg_ismcts_nbs}')
if __name__ == '__main__':
    app.run(main)