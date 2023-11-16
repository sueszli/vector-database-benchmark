import gymnasium as gym
import numpy as np
import random
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.view_requirement import ViewRequirement
ROCK = 0
PAPER = 1
SCISSORS = 2

class AlwaysSameHeuristic(Policy):
    """Pick a random move and stick with it for the entire episode."""

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.exploration = self._create_exploration()
        self.view_requirements.update({'state_in_0': ViewRequirement('state_out_0', shift=-1, space=gym.spaces.Box(ROCK, SCISSORS, shape=(1,), dtype=np.int32))})

    def get_initial_state(self):
        if False:
            i = 10
            return i + 15
        return [random.choice([ROCK, PAPER, SCISSORS])]

    def is_recurrent(self) -> bool:
        if False:
            while True:
                i = 10
        return True

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        if False:
            i = 10
            return i + 15
        return ([state_batches[0][0] for x in obs_batch], state_batches, {})

class BeatLastHeuristic(Policy):
    """Play the move that would beat the last move of the opponent."""

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self.exploration = self._create_exploration()

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        if False:
            return 10

        def successor(x):
            if False:
                return 10
            if isinstance(self.observation_space, gym.spaces.Discrete):
                if x == ROCK:
                    return PAPER
                elif x == PAPER:
                    return SCISSORS
                elif x == SCISSORS:
                    return ROCK
                else:
                    return random.choice([ROCK, PAPER, SCISSORS])
            elif x[ROCK] == 1:
                return PAPER
            elif x[PAPER] == 1:
                return SCISSORS
            elif x[SCISSORS] == 1:
                return ROCK
            elif x[-1] == 1:
                return random.choice([ROCK, PAPER, SCISSORS])
        return ([successor(x) for x in obs_batch], [], {})

    def learn_on_batch(self, samples):
        if False:
            return 10
        pass

    def get_weights(self):
        if False:
            print('Hello World!')
        pass

    def set_weights(self, weights):
        if False:
            print('Hello World!')
        pass