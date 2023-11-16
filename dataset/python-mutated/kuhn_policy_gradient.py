"""Policy gradient agents trained and evaluated on Kuhn Poker."""
from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import policy_gradient
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes', int(1000000.0), 'Number of train episodes.')
flags.DEFINE_integer('eval_every', int(10000.0), 'Eval agents every x episodes.')
flags.DEFINE_enum('loss_str', 'rpg', ['a2c', 'rpg', 'qpg', 'rm'], 'PG loss to use.')

class PolicyGradientPolicies(policy.Policy):
    """Joint policy to be evaluated."""

    def __init__(self, env, nfsp_policies):
        if False:
            print('Hello World!')
        game = env.game
        player_ids = [0, 1]
        super(PolicyGradientPolicies, self).__init__(game, player_ids)
        self._policies = nfsp_policies
        self._obs = {'info_state': [None, None], 'legal_actions': [None, None]}

    def action_probabilities(self, state, player_id=None):
        if False:
            return 10
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        self._obs['current_player'] = cur_player
        self._obs['info_state'][cur_player] = state.information_state_tensor(cur_player)
        self._obs['legal_actions'][cur_player] = legal_actions
        info_state = rl_environment.TimeStep(observations=self._obs, rewards=None, discounts=None, step_type=None)
        p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
        prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict

def main(_):
    if False:
        for i in range(10):
            print('nop')
    game = 'kuhn_poker'
    num_players = 2
    env_configs = {'players': num_players}
    env = rl_environment.Environment(game, **env_configs)
    info_state_size = env.observation_spec()['info_state'][0]
    num_actions = env.action_spec()['num_actions']
    with tf.Session() as sess:
        agents = [policy_gradient.PolicyGradient(sess, idx, info_state_size, num_actions, loss_str=FLAGS.loss_str, hidden_layers_sizes=(128,)) for idx in range(num_players)]
        expl_policies_avg = PolicyGradientPolicies(env, agents)
        sess.run(tf.global_variables_initializer())
        for ep in range(FLAGS.num_episodes):
            if (ep + 1) % FLAGS.eval_every == 0:
                losses = [agent.loss for agent in agents]
                expl = exploitability.exploitability(env.game, expl_policies_avg)
                msg = '-' * 80 + '\n'
                msg += '{}: {}\n{}\n'.format(ep + 1, expl, losses)
                logging.info('%s', msg)
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations['current_player']
                agent_output = agents[player_id].step(time_step)
                action_list = [agent_output.action]
                time_step = env.step(action_list)
            for agent in agents:
                agent.step(time_step)
if __name__ == '__main__':
    app.run(main)