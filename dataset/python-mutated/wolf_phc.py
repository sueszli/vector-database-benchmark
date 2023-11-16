"""WoLF policy-hill climbing agent.

Based on: https://www.sciencedirect.com/science/article/pii/S0004370202001212
"""
import collections
import numpy as np
from open_spiel.python import rl_agent
from open_spiel.python import rl_tools
from open_spiel.python.algorithms.projected_replicator_dynamics import _simplex_projection

def valuedict():
    if False:
        for i in range(10):
            print('nop')
    return collections.defaultdict(float)

class WoLFSchedule(rl_tools.ValueSchedule):
    """Schedule rules described in the WoLF paper.

  at step t the step size is (t0 / (t + t1))
  """

    def __init__(self, t0, t1):
        if False:
            i = 10
            return i + 15
        super(WoLFSchedule, self).__init__()
        self._t0 = t0
        self._t1 = t1
        self._step_taken = 0

    def step(self):
        if False:
            while True:
                i = 10
        value = self._t0 / (self._step_taken + self._t1)
        self._step_taken += 1
        return value

    @property
    def value(self):
        if False:
            while True:
                i = 10
        return self._t0 / (self._step_taken + self._t1)

class WoLFPHC(rl_agent.AbstractAgent):
    """WoLF policy-hill climbing agent agent.


    Based on win or learn fast principle.
    Based on:
    https://www.sciencedirect.com/science/article/pii/S0004370202001212
  """

    def __init__(self, player_id, num_actions, step_size=WoLFSchedule(10000, 1000000), epsilon_schedule=rl_tools.ConstantSchedule(0.2), delta_w=WoLFSchedule(1, 20000), delta_l=WoLFSchedule(2, 20000), discount_factor=1.0):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the WoLF-PHC agent.'
        self._player_id = player_id
        self._num_actions = num_actions
        self._step_size = step_size
        self._epsilon_schedule = epsilon_schedule
        self._epsilon = epsilon_schedule.value
        self._discount_factor = discount_factor
        self._delta_w = delta_w
        self._delta_l = delta_l
        self._cur_policy = collections.defaultdict(valuedict)
        self._avg_policy = collections.defaultdict(valuedict)
        self._q_values = collections.defaultdict(valuedict)
        self._state_counters = valuedict()
        self._prev_info_state = None
        self._last_loss_value = None
        self._cur_delta_value = self._delta_l.value

    def _hill_climbing(self, info_state, legal_actions):
        if False:
            i = 10
            return i + 15
        'Does the hill-climbing update.\n\n    Args:\n      info_state: hashable representation of the information state.\n      legal_actions: list of actions at `info_state`.\n    '
        greedy_q = max([self._q_values[info_state][action] for action in legal_actions])
        greedy_actions = [action for action in legal_actions if self._q_values[info_state][action] == greedy_q]
        if len(greedy_actions) == len(legal_actions):
            return
        deltas = {action: min(self._cur_policy[info_state][action], self._cur_delta_value / (len(legal_actions) - len(greedy_actions))) for action in legal_actions}
        delta_greedy = sum([deltas[action] for action in legal_actions if action not in greedy_actions]) / len(greedy_actions)
        deltas = {action: -deltas[action] if action not in greedy_actions else delta_greedy for action in legal_actions}
        new_policy = np.array([self._cur_policy[info_state][action] + deltas[action] for action in legal_actions])
        new_policy = _simplex_projection(new_policy)
        for i in range(len(legal_actions)):
            self._cur_policy[info_state][legal_actions[i]] = new_policy[i]

    def _get_action_probs(self, info_state, legal_actions, epsilon):
        if False:
            print('Hello World!')
        'Returns a selected action and the probabilities of legal actions.\n\n    To be overwritten by subclasses that implement other action selection\n    methods.\n    Args:\n      info_state: hashable representation of the information state.\n      legal_actions: list of actions at `info_state`.\n      epsilon: float: current value of the epsilon schedule or 0 in case\n        evaluation. QLearner uses it as the exploration parameter in\n        epsilon-greedy, but subclasses are free to interpret in different ways\n        (e.g. as temperature in softmax).\n    '
        if info_state not in self._cur_policy:
            for action in legal_actions:
                self._cur_policy[info_state][action] = 1.0 / len(legal_actions)
                self._avg_policy[info_state][action] = 1.0 / len(legal_actions)
        probs = np.zeros(self._num_actions)
        for action in legal_actions:
            probs[action] = (1 - epsilon) * self._cur_policy[info_state][action] + epsilon * 1.0 / len(legal_actions)
        action = np.random.choice(range(self._num_actions), p=probs)
        return (action, probs)

    def step(self, time_step, is_evaluation=False):
        if False:
            while True:
                i = 10
        'Returns the action to be taken and updates the Q-values if needed.\n\n    Args:\n      time_step: an instance of rl_environment.TimeStep.\n      is_evaluation: bool, whether this is a training or evaluation call.\n\n    Returns:\n      A `rl_agent.StepOutput` containing the action probs and chosen action.\n    '
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
            self._q_values[self._prev_info_state][self._prev_action] += self._step_size.value * self._last_loss_value
            self._state_counters[info_state] += 1
            for action_ in legal_actions:
                self._avg_policy[info_state][action_] = self._avg_policy[info_state][action_] + 1 / self._state_counters[info_state] * (self._cur_policy[info_state][action_] - self._avg_policy[info_state][action_])
            assert self._delta_l.value > self._delta_w.value
            cur_policy_value = sum([self._cur_policy[info_state][action] * self._q_values[info_state][action] for action in legal_actions])
            avg_policy_value = sum([self._avg_policy[info_state][action] * self._q_values[info_state][action] for action in legal_actions])
            if cur_policy_value > avg_policy_value:
                self._cur_delta_value = self._delta_w.value
            else:
                self._cur_delta_value = self._delta_l.value
            if not time_step.last():
                self._hill_climbing(info_state, legal_actions)
                self._epsilon = self._epsilon_schedule.step()
                self._delta_l.step()
                self._delta_w.step()
                self._step_size.step()
            else:
                self._prev_info_state = None
                return
        if not is_evaluation:
            self._prev_info_state = info_state
            self._prev_action = action
        return rl_agent.StepOutput(action=action, probs=probs)

    @property
    def loss(self):
        if False:
            for i in range(10):
                print('nop')
        return self._last_loss_value