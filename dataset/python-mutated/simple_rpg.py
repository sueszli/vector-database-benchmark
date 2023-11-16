import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
from ray.rllib.utils.spaces.repeated import Repeated
MAX_PLAYERS = 4
MAX_ITEMS = 7
MAX_EFFECTS = 2

class SimpleRPG(gym.Env):
    """Example of a custom env with a complex, structured observation.

    The observation is a list of players, each of which is a Dict of
    attributes, and may further hold a list of items (categorical space).

    Note that the env doesn't train, it's just a dummy example to show how to
    use spaces.Repeated in a custom model (see CustomRPGModel below).
    """

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        self.cur_pos = 0
        self.action_space = Discrete(4)
        self.item_space = Discrete(5)
        self.effect_space = Box(9000, 9999, shape=(4,))
        self.player_space = Dict({'location': Box(-100, 100, shape=(2,)), 'status': Box(-1, 1, shape=(10,)), 'items': Repeated(self.item_space, max_len=MAX_ITEMS), 'effects': Repeated(self.effect_space, max_len=MAX_EFFECTS)})
        self.observation_space = Repeated(self.player_space, max_len=MAX_PLAYERS)

    def reset(self, *, seed=None, options=None):
        if False:
            while True:
                i = 10
        return (self.observation_space.sample(), {})

    def step(self, action):
        if False:
            return 10
        return (self.observation_space.sample(), 1, True, False, {})