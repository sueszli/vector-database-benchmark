"""Boltzmann Q learning agent.

This algorithm is a variation of Q learning that uses action selection
based on boltzmann probability interpretation of Q-values.

For more details, see equation (2) page 2 in
   https://arxiv.org/pdf/1109.1528.pdf
"""
import numpy as np
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import tabular_qlearner

class BoltzmannQLearner(tabular_qlearner.QLearner):
    """Tabular Boltzmann Q-Learning agent.

  See open_spiel/python/examples/tic_tac_toe_qlearner.py for an usage example.

  The tic_tac_toe example uses the standard Qlearner. Using the
  BoltzmannQlearner is
  identical and only differs in the initialization of the agents.
  """

    def __init__(self, player_id, num_actions, step_size=0.1, discount_factor=1.0, temperature_schedule=rl_tools.ConstantSchedule(0.5), centralized=False):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(player_id, num_actions, step_size=step_size, discount_factor=discount_factor, epsilon_schedule=temperature_schedule, centralized=centralized)

    def _softmax(self, info_state, legal_actions, temperature):
        if False:
            while True:
                i = 10
        'Action selection based on boltzmann probability interpretation of Q-values.\n\n    For more details, see equation (2) page 2 in\n    https://arxiv.org/pdf/1109.1528.pdf\n\n    Args:\n        info_state: hashable representation of the information state.\n        legal_actions: list of actions at `info_state`.\n        temperature: temperature used for softmax.\n\n    Returns:\n        A valid soft-max selected action and valid action probabilities.\n    '
        probs = np.zeros(self._num_actions)
        if temperature > 0.0:
            probs += [np.exp(1 / temperature * self._q_values[info_state][i]) for i in range(self._num_actions)]
            probs /= np.sum(probs)
        else:
            greedy_q = max([self._q_values[info_state][a] for a in legal_actions])
            greedy_actions = [a for a in legal_actions if self._q_values[info_state][a] == greedy_q]
            probs[greedy_actions] += 1 / len(greedy_actions)
        action = np.random.choice(range(self._num_actions), p=probs)
        return (action, probs)

    def _get_action_probs(self, info_state, legal_actions, epsilon):
        if False:
            print('Hello World!')
        'Returns a selected action and the probabilities of legal actions.'
        return self._softmax(info_state, legal_actions, temperature=epsilon)