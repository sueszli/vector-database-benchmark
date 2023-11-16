"""A cliff walking single agent reinforcement learning environment."""
import numpy as np
from open_spiel.python import rl_environment
(RIGHT, UP, LEFT, DOWN) = range(4)

class Environment(object):
    """A cliff walking reinforcement learning environment.

  This is a deterministic environment that can be used to test RL algorithms.
  Note there are *no illegal moves* in this environment--if the agent is on the
  edge of the cliff and takes an action which would yield an invalid position,
  the action is ignored (as if there were walls surrounding the cliff).

  Cliff example for height=3 and width=5:

                |   |   |   |   |   |
                |   |   |   |   |   |
                | S | x | x | x | G |

  where `S` is always the starting position, `G` is always the goal and `x`
  represents the zone of high negative reward to be avoided. For this instance,
  the optimum policy is depicted as follows:

                |   |   |   |   |   |
                |-->|-->|-->|-->|\\|/|
                |/|\\| x | x | x | G |

  yielding a reward of -6 (minus 1 per time step).

  See pages 132 of Rich Sutton's book for details:
  http://www.incompleteideas.net/book/bookdraft2018mar21.pdf
  """

    def __init__(self, height=4, width=8, discount=1.0, max_t=100):
        if False:
            return 10
        if height < 2 or width < 3:
            raise ValueError('height must be >= 2 and width >= 3.')
        self._height = height
        self._width = width
        self._legal_actions = [RIGHT, UP, LEFT, DOWN]
        self._should_reset = True
        self._max_t = max_t
        self._discounts = [discount] * self.num_players

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        'Resets the environment.'
        self._should_reset = False
        self._time_counter = 0
        self._state = np.array([self._height - 1, 0])
        observations = {'info_state': [self._state.copy()], 'legal_actions': [self._legal_actions], 'current_player': 0}
        return rl_environment.TimeStep(observations=observations, rewards=None, discounts=None, step_type=rl_environment.StepType.FIRST)

    def step(self, actions):
        if False:
            for i in range(10):
                print('nop')
        'Updates the environment according to `actions` and returns a `TimeStep`.\n\n    Args:\n      actions: A singleton list with an integer, or an integer, representing the\n        action the agent took.\n\n    Returns:\n      A `rl_environment.TimeStep` namedtuple containing:\n        observation: singleton list of dicts containing player observations,\n            each corresponding to `observation_spec()`.\n        reward: singleton list containing the reward at this timestep, or None\n            if step_type is `rl_environment.StepType.FIRST`.\n        discount: singleton list containing the discount in the range [0, 1], or\n            None if step_type is `rl_environment.StepType.FIRST`.\n        step_type: A `rl_environment.StepType` value.\n    '
        if self._should_reset:
            return self.reset()
        self._time_counter += 1
        if isinstance(actions, list):
            action = actions[0]
        elif isinstance(actions, int):
            action = actions
        else:
            raise ValueError('Action not supported.', actions)
        dx = 0
        dy = 0
        if action == LEFT:
            dx -= 1
        elif action == RIGHT:
            dx += 1
        if action == UP:
            dy -= 1
        elif action == DOWN:
            dy += 1
        self._state += np.array([dy, dx])
        self._state = self._state.clip(0, [self._height - 1, self._width - 1])
        done = self._is_pit(self._state) or self._is_goal(self._state)
        done = done or self._time_counter >= self._max_t
        step_type = rl_environment.StepType.LAST if done else rl_environment.StepType.MID
        self._should_reset = step_type == rl_environment.StepType.LAST
        observations = {'info_state': [self._state.copy()], 'legal_actions': [self._legal_actions], 'current_player': 0}
        return rl_environment.TimeStep(observations=observations, rewards=[self._get_reward(self._state)], discounts=self._discounts, step_type=step_type)

    def _is_goal(self, pos):
        if False:
            print('Hello World!')
        'Check if position is bottom right corner of grid.'
        return pos[0] == self._height - 1 and pos[1] == self._width - 1

    def _is_pit(self, pos):
        if False:
            print('Hello World!')
        'Check if position is in bottom row between start and goal.'
        return pos[1] > 0 and pos[1] < self._width - 1 and (pos[0] == self._height - 1)

    def _get_reward(self, pos):
        if False:
            for i in range(10):
                print('nop')
        if self._is_pit(pos):
            return -100.0
        else:
            return -1.0

    def observation_spec(self):
        if False:
            i = 10
            return i + 15
        'Defines the observation provided by the environment.\n\n    Each dict member will contain its expected structure and shape.\n\n    Returns:\n      A specification dict describing the observation fields and shapes.\n    '
        return dict(info_state=tuple([2]), legal_actions=(len(self._legal_actions),), current_player=())

    def action_spec(self):
        if False:
            i = 10
            return i + 15
        'Defines action specifications.\n\n    Specifications include action boundaries and their data type.\n\n    Returns:\n      A specification dict containing action properties.\n    '
        return dict(num_actions=len(self._legal_actions), min=min(self._legal_actions), max=max(self._legal_actions), dtype=int)

    @property
    def num_players(self):
        if False:
            return 10
        return 1

    @property
    def is_turn_based(self):
        if False:
            for i in range(10):
                print('nop')
        return False