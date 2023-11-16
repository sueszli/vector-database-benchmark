"""Wrapper around gym env.

Allows for using batches of possibly identitically seeded environments.
"""
import gym
import numpy as np
import random
from six.moves import xrange
import env_spec

def get_env(env_str):
    if False:
        print('Hello World!')
    return gym.make(env_str)

class GymWrapper(object):

    def __init__(self, env_str, distinct=1, count=1, seeds=None):
        if False:
            for i in range(10):
                print('nop')
        self.distinct = distinct
        self.count = count
        self.total = self.distinct * self.count
        self.seeds = seeds or [random.randint(0, 1000000000000.0) for _ in xrange(self.distinct)]
        self.envs = []
        for seed in self.seeds:
            for _ in xrange(self.count):
                env = get_env(env_str)
                env.seed(seed)
                if hasattr(env, 'last'):
                    env.last = 100
                self.envs.append(env)
        self.dones = [True] * self.total
        self.num_episodes_played = 0
        one_env = self.get_one()
        self.use_action_list = hasattr(one_env.action_space, 'spaces')
        self.env_spec = env_spec.EnvSpec(self.get_one())

    def get_seeds(self):
        if False:
            i = 10
            return i + 15
        return self.seeds

    def reset(self):
        if False:
            while True:
                i = 10
        self.dones = [False] * self.total
        self.num_episodes_played += len(self.envs)
        self.seeds = [random.randint(0, 1000000000000.0) for _ in xrange(self.distinct)]
        counter = 0
        for seed in self.seeds:
            for _ in xrange(self.count):
                self.envs[counter].seed(seed)
                counter += 1
        return [self.env_spec.convert_obs_to_list(env.reset()) for env in self.envs]

    def reset_if(self, predicate=None):
        if False:
            print('Hello World!')
        if predicate is None:
            predicate = self.dones
        if self.count != 1:
            assert np.all(predicate)
            return self.reset()
        self.num_episodes_played += sum(predicate)
        output = [self.env_spec.convert_obs_to_list(env.reset()) if pred else None for (env, pred) in zip(self.envs, predicate)]
        for (i, pred) in enumerate(predicate):
            if pred:
                self.dones[i] = False
        return output

    def all_done(self):
        if False:
            for i in range(10):
                print('nop')
        return all(self.dones)

    def step(self, actions):
        if False:
            i = 10
            return i + 15

        def env_step(env, action):
            if False:
                while True:
                    i = 10
            action = self.env_spec.convert_action_to_gym(action)
            (obs, reward, done, tt) = env.step(action)
            obs = self.env_spec.convert_obs_to_list(obs)
            return (obs, reward, done, tt)
        actions = zip(*actions)
        outputs = [env_step(env, action) if not done else (self.env_spec.initial_obs(None), 0, True, None) for (action, env, done) in zip(actions, self.envs, self.dones)]
        for (i, (_, _, done, _)) in enumerate(outputs):
            self.dones[i] = self.dones[i] or done
        (obs, reward, done, tt) = zip(*outputs)
        obs = [list(oo) for oo in zip(*obs)]
        return [obs, reward, done, tt]

    def get_one(self):
        if False:
            print('Hello World!')
        return random.choice(self.envs)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.envs)