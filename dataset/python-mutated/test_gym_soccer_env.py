import numpy as np
import pytest
from dizoo.gym_soccer.envs.gym_soccer_env import GymSoccerEnv
from easydict import EasyDict

@pytest.mark.envtest
class TestGymSoccerEnv:

    def test_naive(self):
        if False:
            for i in range(10):
                print('nop')
        env = GymSoccerEnv(EasyDict({'env_id': 'Soccer-v0', 'act_scale': True}))
        env.seed(25, dynamic_seed=False)
        assert env._seed == 25
        obs = env.reset()
        assert obs.shape == (59,)
        for i in range(1000):
            random_action = env.get_random_action()
            timestep = env.step(random_action)
            assert isinstance(timestep.obs, np.ndarray)
            assert isinstance(timestep.done, bool)
            assert timestep.obs.shape == (59,)
            assert timestep.reward.shape == (1,)
            assert timestep.info['action_args_mask'].shape == (3, 5)
            if timestep.done:
                print('reset env')
                env.reset()
                assert env._eval_episode_return == 0
        print(env.info())
        env.close()