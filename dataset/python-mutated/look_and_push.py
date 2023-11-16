import gymnasium as gym
import numpy as np

class LookAndPush(gym.Env):
    """Memory-requiring Env: Best sequence of actions depends on prev. states.

    Optimal behavior:
        0) a=0 -> observe next state (s'), which is the "hidden" state.
            If a=1 here, the hidden state is not observed.
        1) a=1 to always jump to s=2 (not matter what the prev. state was).
        2) a=1 to move to s=3.
        3) a=1 to move to s=4.
        4) a=0 OR 1 depending on s' observed after 0): +10 reward and done.
            otherwise: -10 reward and done.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(5)
        self._state = None
        self._case = None

    def reset(self, *, seed=None, options=None):
        if False:
            while True:
                i = 10
        self._state = 2
        self._case = np.random.choice(2)
        return (self._state, {})

    def step(self, action):
        if False:
            while True:
                i = 10
        assert self.action_space.contains(action)
        if self._state == 4:
            if action and self._case:
                return (self._state, 10.0, True, {})
            else:
                return (self._state, -10, True, {})
        elif action:
            if self._state == 0:
                self._state = 2
            else:
                self._state += 1
        elif self._state == 2:
            self._state = self._case
        return (self._state, -1, False, False, {})

class OneHot(gym.Wrapper):

    def __init__(self, env):
        if False:
            print('Hello World!')
        super(OneHot, self).__init__(env)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n,))

    def reset(self, *, seed=None, options=None):
        if False:
            print('Hello World!')
        (obs, info) = self.env.reset(seed=seed, options=options)
        return (self._encode_obs(obs), info)

    def step(self, action):
        if False:
            while True:
                i = 10
        (obs, reward, terminated, truncated, info) = self.env.step(action)
        return (self._encode_obs(obs), reward, terminated, truncated, info)

    def _encode_obs(self, obs):
        if False:
            return 10
        new_obs = np.ones(self.env.observation_space.n)
        new_obs[obs] = 1.0
        return new_obs