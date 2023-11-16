"""Catch reinforcement learning environment."""
import collections
import numpy as np
from open_spiel.python import rl_environment
NOOP = 0
LEFT = 1
RIGHT = 2
_Point = collections.namedtuple('Point', ['x', 'y'])

class Environment(object):
    """A catch reinforcement learning environment.

  The implementation considers illegal actions: trying to move the paddle in the
  wall direction when next to a wall will incur in an invalid action and an
  error will be purposely raised.
  """

    def __init__(self, discount=1.0, width=5, height=10, seed=None):
        if False:
            return 10
        self._rng = np.random.RandomState(seed)
        self._width = width
        self._height = height
        self._should_reset = True
        self._num_actions = 3
        self._discounts = [discount] * self.num_players

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        'Resets the environment.'
        self._should_reset = False
        self._ball_pos = _Point(x=self._rng.randint(0, self._width - 1), y=0)
        self._paddle_pos = _Point(x=self._rng.randint(0, self._width - 1), y=self._height - 1)
        legal_actions = [NOOP]
        if self._paddle_pos.x > 0:
            legal_actions.append(LEFT)
        if self._paddle_pos.x < self._width - 1:
            legal_actions.append(RIGHT)
        observations = {'info_state': [self._get_observation()], 'legal_actions': [legal_actions], 'current_player': 0}
        return rl_environment.TimeStep(observations=observations, rewards=None, discounts=None, step_type=rl_environment.StepType.FIRST)

    def step(self, actions):
        if False:
            return 10
        'Updates the environment according to `actions` and returns a `TimeStep`.\n\n    Args:\n      actions: A singleton list with an integer, or an integer, representing the\n        action the agent took.\n\n    Returns:\n      A `rl_environment.TimeStep` namedtuple containing:\n        observation: singleton list of dicts containing player observations,\n            each corresponding to `observation_spec()`.\n        reward: singleton list containing the reward at this timestep, or None\n            if step_type is `rl_environment.StepType.FIRST`.\n        discount: singleton list containing the discount in the range [0, 1], or\n            None if step_type is `rl_environment.StepType.FIRST`.\n        step_type: A `rl_environment.StepType` value.\n    '
        if self._should_reset:
            return self.reset()
        if isinstance(actions, list):
            action = actions[0]
        elif isinstance(actions, int):
            action = actions
        else:
            raise ValueError('Action not supported.', actions)
        (x, y) = (self._paddle_pos.x, self._paddle_pos.y)
        if action == LEFT:
            x -= 1
        elif action == RIGHT:
            x += 1
        elif action != NOOP:
            raise ValueError('unrecognized action ', action)
        assert 0 <= x < self._width, 'Illegal action detected ({}), new state: ({},{})'.format(action, x, y)
        self._paddle_pos = _Point(x, y)
        (x, y) = (self._ball_pos.x, self._ball_pos.y)
        if y == self._height - 1:
            done = True
            reward = 1.0 if x == self._paddle_pos.x else -1.0
        else:
            done = False
            y += 1
            reward = 0.0
            self._ball_pos = _Point(x, y)
        step_type = rl_environment.StepType.LAST if done else rl_environment.StepType.MID
        self._should_reset = step_type == rl_environment.StepType.LAST
        legal_actions = [NOOP]
        if self._paddle_pos.x > 0:
            legal_actions.append(LEFT)
        if self._paddle_pos.x < self._width - 1:
            legal_actions.append(RIGHT)
        observations = {'info_state': [self._get_observation()], 'legal_actions': [legal_actions], 'current_player': 0}
        return rl_environment.TimeStep(observations=observations, rewards=[reward], discounts=self._discounts, step_type=step_type)

    def _get_observation(self):
        if False:
            while True:
                i = 10
        board = np.zeros((self._height, self._width), dtype=np.float32)
        board[self._ball_pos.y, self._ball_pos.x] = 1.0
        board[self._paddle_pos.y, self._paddle_pos.x] = 1.0
        return board.flatten()

    def observation_spec(self):
        if False:
            print('Hello World!')
        'Defines the observation provided by the environment.\n\n    Each dict member will contain its expected structure and shape.\n\n    Returns:\n      A specification dict describing the observation fields and shapes.\n    '
        return dict(info_state=tuple([self._height * self._width]), legal_actions=(self._num_actions,), current_player=())

    def action_spec(self):
        if False:
            for i in range(10):
                print('nop')
        'Defines action specifications.\n\n    Specifications include action boundaries and their data type.\n\n    Returns:\n      A specification dict containing action properties.\n    '
        return dict(num_actions=self._num_actions, min=0, max=2, dtype=int)

    @property
    def num_players(self):
        if False:
            print('Hello World!')
        return 1

    @property
    def is_turn_based(self):
        if False:
            while True:
                i = 10
        return False