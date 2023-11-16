import unittest
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility
from ray.rllib.utils.gym import try_import_gymnasium_and_gym
from ray.tune.registry import register_env
(gym, old_gym) = try_import_gymnasium_and_gym()

class GymnasiumOldAPI(gym.Env):

    def __init__(self, config=None):
        if False:
            for i in range(10):
                print('nop')
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (1,))
        self.action_space = gym.spaces.Discrete(2)

    def reset(self):
        if False:
            while True:
                i = 10
        return self.observation_space.sample()

    def step(self, action):
        if False:
            while True:
                i = 10
        done = True
        return (self.observation_space.sample(), 1.0, done, {})

    def seed(self, seed=None):
        if False:
            i = 10
            return i + 15
        pass

    def render(self, mode='human'):
        if False:
            i = 10
            return i + 15
        pass

class GymnasiumNewAPIButOldSpaces(gym.Env):
    render_mode = 'human'

    def __init__(self, config=None):
        if False:
            print('Hello World!')
        self.observation_space = old_gym.spaces.Box(-1.0, 1.0, (1,))
        self.action_space = old_gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        if False:
            return 10
        return (self.observation_space.sample(), {})

    def step(self, action):
        if False:
            while True:
                i = 10
        terminated = truncated = True
        return (self.observation_space.sample(), 1.0, terminated, truncated, {})

    def render(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class GymnasiumNewAPIButThrowsErrorOnReset(gym.Env):
    render_mode = 'human'

    def __init__(self, config=None):
        if False:
            print('Hello World!')
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (1,))
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        if False:
            for i in range(10):
                print('nop')
        assert False, 'kaboom!'
        return (self.observation_space.sample(), {})

    def step(self, action):
        if False:
            return 10
        terminated = truncated = True
        return (self.observation_space.sample(), 1.0, terminated, truncated, {})

    def render(self):
        if False:
            while True:
                i = 10
        pass

class OldGymEnv(old_gym.Env):

    def __init__(self, config=None):
        if False:
            print('Hello World!')
        self.observation_space = old_gym.spaces.Box(-1.0, 1.0, (1,))
        self.action_space = old_gym.spaces.Discrete(2)

    def reset(self):
        if False:
            return 10
        return self.observation_space.sample()

    def step(self, action):
        if False:
            i = 10
            return i + 15
        done = True
        return (self.observation_space.sample(), 1.0, done, {})

    def seed(self, seed=None):
        if False:
            print('Hello World!')
        pass

    def render(self, mode='human'):
        if False:
            print('Hello World!')
        pass

class MultiAgentGymnasiumOldAPI(MultiAgentEnv):

    def __init__(self, config=None):
        if False:
            return 10
        super().__init__()
        self.observation_space = gym.spaces.Dict({'agent0': gym.spaces.Box(-1.0, 1.0, (1,))})
        self.action_space = gym.spaces.Dict({'agent0': gym.spaces.Discrete(2)})
        self._agent_ids = {'agent0'}

    def reset(self):
        if False:
            print('Hello World!')
        return {'agent0': self.observation_space.sample()}

    def step(self, action):
        if False:
            print('Hello World!')
        done = True
        return ({'agent0': self.observation_space.sample()}, {'agent0': 1.0}, {'agent0': done, '__all__': done}, {})

    def seed(self, seed=None):
        if False:
            return 10
        pass

    def render(self, mode='human'):
        if False:
            while True:
                i = 10
        pass

class TestGymEnvAPIs(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            return 10
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            return 10
        ray.shutdown()

    def test_gymnasium_old_api(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests a gymnasium Env that uses the old API.'

        def test_():
            if False:
                while True:
                    i = 10
            PPOConfig().environment(env=GymnasiumOldAPI, auto_wrap_old_gym_envs=False).rollouts(num_rollout_workers=0).build()
        self.assertRaisesRegex(ValueError, '.*In particular, the `reset\\(\\)` method seems to be faulty..*', lambda : test_())

    def test_gymnasium_old_api_using_auto_wrap(self):
        if False:
            return 10
        'Tests a gymnasium Env that uses the old API, but is auto-wrapped by RLlib.'
        algo = PPOConfig().environment(env=GymnasiumOldAPI, auto_wrap_old_gym_envs=True).rollouts(num_rollout_workers=0).build()
        algo.train()
        algo.stop()

    def test_gymnasium_new_api_but_old_spaces(self):
        if False:
            while True:
                i = 10
        'Tests a gymnasium Env that uses the new API, but has old spaces.'

        def test_():
            if False:
                return 10
            PPOConfig().environment(GymnasiumNewAPIButOldSpaces, auto_wrap_old_gym_envs=True).rollouts(num_rollout_workers=0).build()
        self.assertRaisesRegex(ValueError, 'Observation space must be a gymnasium.Space!', lambda : test_())

    def test_gymnasium_new_api_but_throws_error_on_reset(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests a gymnasium Env that uses the new API, but errors on reset() call.'

        def test_():
            if False:
                print('Hello World!')
            PPOConfig().environment(GymnasiumNewAPIButThrowsErrorOnReset, auto_wrap_old_gym_envs=True).rollouts(num_rollout_workers=0).build()
        self.assertRaisesRegex(AssertionError, 'kaboom!', lambda : test_())

    def test_gymnasium_old_api_but_manually_wrapped(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests a gymnasium Env that uses the old API, but is correctly wrapped.'
        from gymnasium.wrappers import EnvCompatibility
        register_env('test', lambda env_ctx: EnvCompatibility(GymnasiumOldAPI(env_ctx)))
        algo = PPOConfig().environment('test', auto_wrap_old_gym_envs=False).rollouts(num_rollout_workers=0).build()
        algo.train()
        algo.stop()

    def test_old_gym_env(self):
        if False:
            print('Hello World!')
        'Tests a old gym.Env (should fail, even with auto-wrapping enabled).'

        def test_():
            if False:
                i = 10
                return i + 15
            PPOConfig().environment(env=OldGymEnv, auto_wrap_old_gym_envs=True).rollouts(num_rollout_workers=0).build()
        self.assertRaisesRegex(ValueError, 'does not abide to the new gymnasium-style API', lambda : test_())

    def test_multi_agent_gymnasium_old_api(self):
        if False:
            print('Hello World!')
        'Tests a MultiAgentEnv (gymnasium.Env subclass) that uses the old API.'

        def test_():
            if False:
                for i in range(10):
                    print('nop')
            PPOConfig().environment(MultiAgentGymnasiumOldAPI, auto_wrap_old_gym_envs=False).rollouts(num_rollout_workers=0).build()
        self.assertRaisesRegex(ValueError, '.*In particular, the `reset\\(\\)` method seems to be faulty..*', lambda : test_())

    def test_multi_agent_gymnasium_old_api_auto_wrapped(self):
        if False:
            i = 10
            return i + 15
        'Tests a MultiAgentEnv (gymnasium.Env subclass) that uses the old API.'
        algo = PPOConfig().environment(MultiAgentGymnasiumOldAPI, auto_wrap_old_gym_envs=True, disable_env_checking=True).rollouts(num_rollout_workers=0).build()
        algo.train()
        algo.stop()

    def test_multi_agent_gymnasium_old_api_manually_wrapped(self):
        if False:
            return 10
        'Tests a MultiAgentEnv (gymnasium.Env subclass) that uses the old API.'
        register_env('test', lambda env_ctx: MultiAgentEnvCompatibility(MultiAgentGymnasiumOldAPI(env_ctx)))
        algo = PPOConfig().environment('test', auto_wrap_old_gym_envs=False, disable_env_checking=True).rollouts(num_rollout_workers=0).build()
        algo.train()
        algo.stop()
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))