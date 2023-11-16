"""Example for policy_aggregator_example.

Example.
"""
from absl import app
from absl import flags
import numpy as np
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import policy_aggregator
FLAGS = flags.FLAGS
flags.DEFINE_string('game_name', 'kuhn_poker', 'Game name')

class TestPolicy(policy.Policy):

    def __init__(self, action_int):
        if False:
            print('Hello World!')
        self._action_int = action_int

    def action_probabilities(self, state, player_id=None):
        if False:
            return 10
        return {self._action_int: 1.0}

def main(unused_argv):
    if False:
        while True:
            i = 10
    env = rl_environment.Environment(FLAGS.game_name)
    policies = [[policy.TabularPolicy(env.game).copy_with_noise(alpha=float(i), beta=1.0) for i in range(2)] for _ in range(2)]
    probabilities = [list(np.ones(len(policies[i])) / len(policies[i])) for i in range(2)]
    pol_ag = policy_aggregator.PolicyAggregator(env.game)
    aggr_policies = pol_ag.aggregate([0, 1], policies, probabilities)
    exploitabilities = exploitability.nash_conv(env.game, aggr_policies)
    print('Exploitability : {}'.format(exploitabilities))
    print(policies[0][0].action_probability_array)
    print(policies[0][1].action_probability_array)
    print(aggr_policies.policy)
    print('\nCopy Example')
    mother_policy = policy.TabularPolicy(env.game).copy_with_noise(1, 10)
    policies = [[mother_policy.__copy__() for _ in range(2)] for _ in range(2)]
    probabilities = [list(np.ones(len(policies)) / len(policies)) for _ in range(2)]
    pol_ag = policy_aggregator.PolicyAggregator(env.game)
    aggr_policy = pol_ag.aggregate([0], policies, probabilities)
    for (state, value) in aggr_policy.policy[0].items():
        polici = mother_policy.policy_for_key(state)
        value_normal = {action: probability for (action, probability) in enumerate(polici) if probability > 0}
        for key in value.keys():
            print('State : {}. Key : {}. Aggregated : {}. Real : {}. Passed : {}'.format(state, key, value[key], value_normal[key], np.abs(value[key] - value_normal[key]) < 1e-08))
if __name__ == '__main__':
    app.run(main)