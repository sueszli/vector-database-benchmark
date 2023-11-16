"""Tabular Multiagent Q-learning agent.

Currently implementations include:
Nash-Q: https://www.jmlr.org/papers/volume4/hu03a/hu03a.pdf
Correlated-Q: https://www.aaai.org/Papers/ICML/2003/ICML03-034.pdf, where both
CE-Q and CCE-Q are supported.
Asymmetric-Q: https://ieeexplore.ieee.org/document/1241094
"""
import abc
import collections
import itertools
import nashpy as nash
import numpy as np
from open_spiel.python import rl_agent
from open_spiel.python import rl_tools
from open_spiel.python.algorithms.jpsro import _mgcce
from open_spiel.python.algorithms.stackelberg_lp import solve_stackelberg
import pyspiel

def valuedict():
    if False:
        for i in range(10):
            print('nop')
    return collections.defaultdict(float)

class JointActionSolver:

    @abc.abstractmethod
    def __call__(self, payoffs_array):
        if False:
            print('Hello World!')
        'Find a joint action mixture and values for the current one-step game.\n\n    Args:\n      payoffs_array: a `numpy.ndarray` of utilities of a game.\n\n    Returns:\n      res_mixtures: a list of mixed strategies for each agent\n      res_values: a list of expected utilities for each agent\n    '

class TwoPlayerNashSolver(JointActionSolver):
    """A joint action solver solving for Nash for two-player games.

  Uses python.algorithms.matrix_nash.lemke_howson_solve
  """

    def __call__(self, payoffs_array):
        if False:
            print('Hello World!')
        assert len(payoffs_array) == 2
        (row_payoffs, col_payoffs) = (payoffs_array[0], payoffs_array[1])
        (a0, a1) = payoffs_array.shape[1:]
        nashpy_game = nash.Game(row_payoffs, col_payoffs)
        best_value = float('-inf')
        (res_mixtures, res_values) = (None, None)
        for (row_mixture, col_mixture) in nashpy_game.support_enumeration():
            if np.sum(np.isnan(row_mixture)) or np.sum(np.isnan(col_mixture)):
                continue
            (row_mixture_, col_mixture_) = (row_mixture.reshape((-1, 1)), col_mixture.reshape((-1, 1)))
            (row_value, col_value) = (row_mixture_.T.dot(row_payoffs).dot(col_mixture_).item(), row_mixture_.T.dot(col_payoffs).dot(col_mixture_).item())
            if row_value + col_value > best_value:
                best_value = row_value + col_value
                res_mixtures = [row_mixture, col_mixture]
                res_values = [row_value, col_value]
        if not res_mixtures:
            res_mixtures = [np.ones(a0) / a0, np.ones(a1) / a1]
            (row_mixture_, col_mixture_) = (res_mixtures[0].reshape((-1, 1)), res_mixtures[1].reshape((-1, 1)))
            res_values = [row_mixture_.T.dot(row_payoffs).dot(col_mixture_).item(), row_mixture_.T.dot(col_payoffs).dot(col_mixture_).item()]
        return (res_mixtures, res_values)

class CorrelatedEqSolver(JointActionSolver):
    """A joint action solver solving for correlated equilibrium.

  Uses python.algorithms.jspro._mgce and _mgcce for solving (coarse) correlated
  equilibrium.
  """

    def __init__(self, is_cce=False):
        if False:
            print('Hello World!')
        self._is_cce = is_cce

    def __call__(self, payoffs_array):
        if False:
            print('Hello World!')
        num_players = len(payoffs_array)
        assert num_players > 0
        num_strategies_per_player = payoffs_array.shape[1:]
        (mixture, _) = _mgcce(payoffs_array, [np.ones([ns], dtype=np.int32) for ns in num_strategies_per_player], ignore_repeats=True) if self._is_cce else _mgcce(payoffs_array, [np.ones([ns], dtype=np.int32) for ns in num_strategies_per_player], ignore_repeats=True)
        (mixtures, values) = ([], [])
        for n in range(num_players):
            values.append(np.sum(payoffs_array[n] * mixture))
            mixtures.append(np.sum(mixture, axis=tuple([n_ for n_ in range(num_players) if n_ != n])))
        return (mixtures, values)

class StackelbergEqSolver(JointActionSolver):
    """A joint action solver solving for Stackelverg equilibrium.

  Uses python.algorithms.stackelberg_lp.py.
  """

    def __init__(self, is_first_leader=True):
        if False:
            i = 10
            return i + 15
        self._is_first_leader = is_first_leader

    def __call__(self, payoffs_array):
        if False:
            while True:
                i = 10
        assert len(payoffs_array) == 2
        game = pyspiel.create_matrix_game(payoffs_array[0], payoffs_array[1])
        try:
            (player0_strategy, player1_strategy, player0_value, player1_value) = solve_stackelberg(game, self._is_first_leader)
            return ([player0_strategy, player1_strategy], [player0_value, player1_value])
        except:
            (num_player0_strategies, num_player1_strategies) = payoffs_array[0].shape
            (player0_strategy, player1_strategy) = (np.ones(num_player0_strategies) / num_player0_strategies, np.ones(num_player1_strategies) / num_player1_strategies)
            (player0_value, player1_value) = (player0_strategy.reshape(1, -1).dot(payoffs_array[0]).dot(player1_strategy.reshape(-1, 1)), player0_strategy.reshape(1, -1).dot(payoffs_array[1]).dot(player1_strategy.reshape(-1, 1)))
            return ([player0_strategy, player1_strategy], [player0_value, player1_value])

class MultiagentQLearner(rl_agent.AbstractAgent):
    """A multiagent joint action learner."""

    def __init__(self, player_id, num_players, num_actions, joint_action_solver, step_size=0.1, epsilon_schedule=rl_tools.ConstantSchedule(0.2), discount_factor=1.0):
        if False:
            print('Hello World!')
        'Initialize the Multiagent joint-action Q-Learning agent.\n\n    The joint_action_solver solves for one-step matrix game defined by Q-tables.\n\n    Args:\n      player_id: the player id this agent will play as,\n      num_players: the number of players in the game,\n      num_actions: the number of distinct actions in the game,\n      joint_action_solver: the joint action solver class to use to solve the\n        one-step matrix games\n      step_size: learning rate for Q-learning,\n      epsilon_schedule: exploration parameter,\n      discount_factor: the discount factor as in Q-learning.\n    '
        self._player_id = player_id
        self._num_players = num_players
        self._num_actions = num_actions
        self._joint_action_solver = joint_action_solver
        self._step_size = step_size
        self._epsilon_schedule = epsilon_schedule
        self._epsilon = epsilon_schedule.value
        self._discount_factor = discount_factor
        self._q_values = [collections.defaultdict(valuedict) for _ in range(num_players)]
        self._prev_info_state = None

    def _get_payoffs_array(self, info_state):
        if False:
            print('Hello World!')
        payoffs_array = np.zeros((self._num_players,) + tuple(self._num_actions))
        for joint_action in itertools.product(*[range(dim) for dim in self._num_actions]):
            for n in range(self._num_players):
                payoffs_array[(n,) + joint_action] = self._q_values[n][info_state][joint_action]
        return payoffs_array

    def _epsilon_greedy(self, info_state, legal_actions, epsilon):
        if False:
            while True:
                i = 10
        'Returns a valid epsilon-greedy action and valid action probs.\n\n    If the agent has not been to `info_state`, a valid random action is chosen.\n    Args:\n      info_state: hashable representation of the information state.\n      legal_actions: list of actions at `info_state`.\n      epsilon: float, prob of taking an exploratory action.\n\n    Returns:\n      A valid epsilon-greedy action and valid action probabilities.\n    '
        probs = np.zeros(self._num_actions[self._player_id])
        (state_probs, _) = self._joint_action_solver(self._get_payoffs_array(info_state))
        probs[legal_actions[self._player_id]] = epsilon / len(legal_actions[self._player_id])
        probs += (1 - epsilon) * state_probs[self._player_id]
        action = np.random.choice(range(self._num_actions[self._player_id]), p=probs)
        return (action, probs)

    def step(self, time_step, actions=None, is_evaluation=False):
        if False:
            for i in range(10):
                print('nop')
        'Returns the action to be taken and updates the Q-values if needed.\n\n    Args:\n        time_step: an instance of rl_environment.TimeStep,\n        actions: list of actions taken by all agents from the previous step,\n        is_evaluation: bool, whether this is a training or evaluation call,\n\n    Returns:\n        A `rl_agent.StepOutput` containing the action probs and chosen action.\n    '
        info_state = str(time_step.observations['info_state'])
        legal_actions = time_step.observations['legal_actions']
        (action, probs) = (None, None)
        if not time_step.last():
            epsilon = 0.0 if is_evaluation else self._epsilon
            (action, probs) = self._epsilon_greedy(info_state, legal_actions, epsilon=epsilon)
        actions = tuple(actions)
        if self._prev_info_state and (not is_evaluation):
            (_, next_state_values) = self._joint_action_solver(self._get_payoffs_array(info_state))
            for n in range(self._num_players):
                target = time_step.rewards[n]
                if not time_step.last():
                    target += self._discount_factor * next_state_values[n]
                prev_q_value = self._q_values[n][self._prev_info_state][actions]
                self._q_values[n][self._prev_info_state][actions] += self._step_size * (target - prev_q_value)
            self._epsilon = self._epsilon_schedule.step()
            if time_step.last():
                self._prev_info_state = None
                return
        if not is_evaluation:
            self._prev_info_state = info_state
        return rl_agent.StepOutput(action=action, probs=probs)