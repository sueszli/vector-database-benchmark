import pytest
import numpy as np
from stable_baselines import A2C, ACER, ACKTR, DDPG, DQN, PPO1, PPO2, SAC, TRPO, TD3
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.identity_env import IdentityEnv, IdentityEnvBox, IdentityEnvMultiBinary, IdentityEnvMultiDiscrete
from stable_baselines.common.evaluation import evaluate_policy

def check_shape(make_env, model_class, shape_1, shape_2):
    if False:
        for i in range(10):
            print('nop')
    model = model_class(policy='MlpPolicy', env=DummyVecEnv([make_env]))
    env0 = make_env()
    env1 = DummyVecEnv([make_env])
    for (env, expected_shape) in [(env0, shape_1), (env1, shape_2)]:

        def callback(locals_, _globals):
            if False:
                i = 10
                return i + 15
            assert np.array(locals_['action']).shape == expected_shape
        evaluate_policy(model, env, n_eval_episodes=5, callback=callback)

@pytest.mark.slow
@pytest.mark.parametrize('model_class', [A2C, ACER, ACKTR, DQN, PPO1, PPO2, TRPO])
def test_identity(model_class):
    if False:
        return 10
    '\n    test the Disrete environment vectorisation detection\n\n    :param model_class: (BaseRLModel) the RL model\n    '
    check_shape(lambda : IdentityEnv(dim=10), model_class, (), (1,))

@pytest.mark.slow
@pytest.mark.parametrize('model_class', [A2C, DDPG, PPO1, PPO2, SAC, TRPO, TD3])
def test_identity_box(model_class):
    if False:
        return 10
    '\n    test the Box environment vectorisation detection\n\n    :param model_class: (BaseRLModel) the RL model\n    '
    check_shape(lambda : IdentityEnvBox(eps=0.5), model_class, (1,), (1, 1))

@pytest.mark.slow
@pytest.mark.parametrize('model_class', [A2C, PPO1, PPO2, TRPO])
def test_identity_multi_binary(model_class):
    if False:
        for i in range(10):
            print('nop')
    '\n    test the MultiBinary environment vectorisation detection\n\n    :param model_class: (BaseRLModel) the RL model\n    '
    check_shape(lambda : IdentityEnvMultiBinary(dim=10), model_class, (10,), (1, 10))

@pytest.mark.slow
@pytest.mark.parametrize('model_class', [A2C, PPO1, PPO2, TRPO])
def test_identity_multi_discrete(model_class):
    if False:
        return 10
    '\n    test the MultiDiscrete environment vectorisation detection\n\n    :param model_class: (BaseRLModel) the RL model\n    '
    check_shape(lambda : IdentityEnvMultiDiscrete(dim=10), model_class, (2,), (1, 2))