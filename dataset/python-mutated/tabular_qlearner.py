"""Tabular Q-learning agent."""
import collections
import numpy as np
from open_spiel.python import rl_agent
from open_spiel.python import rl_tools

def valuedict():
    if False:
        for i in range(10):
            print('nop')
    return collections.defaultdict(float)

class QLearner(rl_agent.AbstractAgent):
    """Tabular Q-Learning agent.

  See open_spiel/python/examples/tic_tac_toe_qlearner.py for an usage example.
  """

    def __init__(self, player_id, num_actions, step_size=0.1, epsilon_schedule=rl_tools.ConstantSchedule(0.2), discount_factor=1.0, centralized=False):
        if False:
            while True:
                i = 10
        'Initialize the Q-Learning agent.'
        self._player_id = player_id
        self._num_actions = num_actions
        self._step_size = step_size
        self._epsilon_schedule = epsilon_schedule
        self._epsilon = epsilon_schedule.value
        self._discount_factor = discount_factor
        self._centralized = centralized
        self._q_values = collections.defaultdict(valuedict)
        self._prev_info_state = None
        self._last_loss_value = None

    def _epsilon_greedy(self, info_state, legal_actions, epsilon):
        if False:
            return 10
        'Returns a valid epsilon-greedy action and valid action probs.\n\n    If the agent has not been to `info_state`, a valid random action is chosen.\n\n    Args:\n      info_state: hashable representation of the information state.\n      legal_actions: list of actions at `info_state`.\n      epsilon: float, prob of taking an exploratory action.\n\n    Returns:\n      A valid epsilon-greedy action and valid action probabilities.\n    '
        probs = np.zeros(self._num_actions)
        greedy_q = max([self._q_values[info_state][a] for a in legal_actions])
        greedy_actions = [a for a in legal_actions if self._q_values[info_state][a] == greedy_q]
        probs[legal_actions] = epsilon / len(legal_actions)
        probs[greedy_actions] += (1 - epsilon) / len(greedy_actions)
        action = np.random.choice(range(self._num_actions), p=probs)
        return (action, probs)

    def _get_action_probs(self, info_state, legal_actions, epsilon):
        if False:
            i = 10
            return i + 15
        'Returns a selected action and the probabilities of legal actions.\n\n    To be overwritten by subclasses that implement other action selection\n    methods.\n\n    Args:\n      info_state: hashable representation of the information state.\n      legal_actions: list of actions at `info_state`.\n      epsilon: float: current value of the epsilon schedule or 0 in case\n        evaluation. QLearner uses it as the exploration parameter in\n        epsilon-greedy, but subclasses are free to interpret in different ways\n        (e.g. as temperature in softmax).\n    '
        return self._epsilon_greedy(info_state, legal_actions, epsilon)

    def step(self, time_step, is_evaluation=False):
        if False:
            for i in range(10):
                print('nop')
        'Returns the action to be taken and updates the Q-values if needed.\n\n    Args:\n      time_step: an instance of rl_environment.TimeStep.\n      is_evaluation: bool, whether this is a training or evaluation call.\n\n    Returns:\n      A `rl_agent.StepOutput` containing the action probs and chosen action.\n    '
        if self._centralized:
            info_state = str(time_step.observations['info_state'])
        else:
            info_state = str(time_step.observations['info_state'][self._player_id])
        legal_actions = time_step.observations['legal_actions'][self._player_id]
        (action, probs) = (None, None)
        if not time_step.last():
            epsilon = 0.0 if is_evaluation else self._epsilon
            (action, probs) = self._get_action_probs(info_state, legal_actions, epsilon)
        if self._prev_info_state and (not is_evaluation):
            target = time_step.rewards[self._player_id]
            if not time_step.last():
                target += self._discount_factor * max([self._q_values[info_state][a] for a in legal_actions])
            prev_q_value = self._q_values[self._prev_info_state][self._prev_action]
            self._last_loss_value = target - prev_q_value
            self._q_values[self._prev_info_state][self._prev_action] += self._step_size * self._last_loss_value
            self._epsilon = self._epsilon_schedule.step()
            if time_step.last():
                self._prev_info_state = None
                return
        if not is_evaluation:
            self._prev_info_state = info_state
            self._prev_action = action
        return rl_agent.StepOutput(action=action, probs=probs)

    @property
    def loss(self):
        if False:
            while True:
                i = 10
        return self._last_loss_value