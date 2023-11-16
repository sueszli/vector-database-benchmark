import pytest
from stable_baselines import A2C, ACER, ACKTR, PPO2
from stable_baselines.common.identity_env import IdentityEnv, IdentityEnvBox
from stable_baselines.common.vec_env import DummyVecEnv
MODEL_LIST = [A2C, ACER, ACKTR, PPO2]

@pytest.mark.parametrize('model_class', MODEL_LIST)
def test_model_multiple_learn_no_reset(model_class):
    if False:
        while True:
            i = 10
    "Check that when we call learn multiple times, we don't unnecessarily\n    reset the environment.\n    "
    if model_class is ACER:

        def make_env():
            if False:
                i = 10
                return i + 15
            return IdentityEnv(ep_length=10000000000.0, dim=2)
    else:

        def make_env():
            if False:
                while True:
                    i = 10
            return IdentityEnvBox(ep_length=10000000000.0)
    env = make_env()
    venv = DummyVecEnv([lambda : env])
    model = model_class(policy='MlpPolicy', env=venv)
    _check_reset_count(model, env)
    env = make_env()
    venv = DummyVecEnv([lambda : env])
    assert env.num_resets == 0
    model.set_env(venv)
    _check_reset_count(model, env)

def _check_reset_count(model, env: IdentityEnv):
    if False:
        return 10
    assert env.num_resets == 0
    _prev_runner = None
    for _ in range(2):
        model.learn(total_timesteps=300)
        assert env.num_resets == 1
        if _prev_runner is not None:
            assert _prev_runner is model.runner, "Runner shouldn't change"
        _prev_runner = model.runner