import subprocess
import os
import gym
import pytest
import numpy as np
from stable_baselines import A2C, ACKTR, SAC, DDPG, PPO1, PPO2, TRPO, TD3
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.identity_env import IdentityEnvBox
from stable_baselines.ddpg import AdaptiveParamNoiseSpec, NormalActionNoise
from stable_baselines.common.evaluation import evaluate_policy
from tests.test_common import _assert_eq
N_EVAL_EPISODES = 20
NUM_TIMESTEPS = 300
MODEL_LIST = [A2C, ACKTR, DDPG, PPO1, PPO2, SAC, TD3, TRPO]

@pytest.mark.slow
@pytest.mark.parametrize('model_class', MODEL_LIST)
def test_model_manipulation(request, model_class):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if the algorithm can be loaded and saved without any issues, the environment switching\n    works and that the action prediction works\n\n    :param model_class: (BaseRLModel) A model\n    '
    model_fname = None
    try:
        env = DummyVecEnv([lambda : IdentityEnvBox(eps=0.5)])
        model = model_class(policy='MlpPolicy', env=env, seed=0)
        model.learn(total_timesteps=NUM_TIMESTEPS)
        env.reset()
        observations = np.concatenate([env.step([env.action_space.sample()])[0] for _ in range(10)], axis=0)
        (selected_actions, _) = model.predict(observations, deterministic=True)
        model_fname = './test_model_{}.zip'.format(request.node.name)
        model.save(model_fname)
        del model, env
        model = model_class.load(model_fname)
        (new_selected_actions, _) = model.predict(observations, deterministic=True)
        assert np.allclose(selected_actions, new_selected_actions, 0.0001)
        env = DummyVecEnv([lambda : IdentityEnvBox(eps=0.5)])
        model.set_env(env)
        obs = env.reset()
        with pytest.warns(None) as record:
            act_prob = model.action_probability(obs)
        if model_class in [DDPG, SAC, TD3]:
            assert len(record) == 1, 'No warning was raised for {}'.format(model_class)
            assert act_prob is None, 'Error: action_probability should be None for {}'.format(model_class)
        else:
            assert act_prob[0].shape == (1, 1) and act_prob[1].shape == (1, 1), 'Error: action_probability not returning correct shape'
        env = model.get_env()
        obs = env.reset()
        observations = np.array([obs for _ in range(10)])
        observations = np.squeeze(observations)
        observations = observations.reshape((-1, 1))
        actions = np.array([env.action_space.sample() for _ in range(10)])
        if model_class in [DDPG, SAC, TD3]:
            with pytest.raises(ValueError):
                model.action_probability(observations, actions=actions)
        else:
            actions_probas = model.action_probability(observations, actions=actions)
            assert actions_probas.shape == (len(actions), 1), actions_probas.shape
            assert np.all(actions_probas >= 0), actions_probas
            actions_logprobas = model.action_probability(observations, actions=actions, logp=True)
            assert np.allclose(actions_probas, np.exp(actions_logprobas)), (actions_probas, actions_logprobas)
        model.learn(total_timesteps=100)
        evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES)
        del model, env
    finally:
        if model_fname is not None and os.path.exists(model_fname):
            os.remove(model_fname)

def test_ddpg():
    if False:
        return 10
    args = ['--env-id', 'Pendulum-v0', '--num-timesteps', 300, '--noise-type', 'ou_0.01']
    args = list(map(str, args))
    return_code = subprocess.call(['python', '-m', 'stable_baselines.ddpg.main'] + args)
    _assert_eq(return_code, 0)

def test_ddpg_eval_env():
    if False:
        i = 10
        return i + 15
    '\n    Additional test to check that everything is working when passing\n    an eval env.\n    '
    eval_env = gym.make('Pendulum-v0')
    model = DDPG('MlpPolicy', 'Pendulum-v0', nb_rollout_steps=5, nb_train_steps=2, nb_eval_steps=10, eval_env=eval_env, verbose=0)
    model.learn(NUM_TIMESTEPS)

def test_ddpg_normalization():
    if False:
        return 10
    '\n    Test that observations and returns normalizations are properly saved and loaded.\n    '
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05, desired_action_stddev=0.05)
    model = DDPG('MlpPolicy', 'Pendulum-v0', memory_limit=50000, normalize_observations=True, normalize_returns=True, nb_rollout_steps=128, nb_train_steps=1, batch_size=64, param_noise=param_noise)
    model.learn(NUM_TIMESTEPS)
    obs_rms_params = model.sess.run(model.obs_rms_params)
    ret_rms_params = model.sess.run(model.ret_rms_params)
    model.save('./test_ddpg.zip')
    loaded_model = DDPG.load('./test_ddpg.zip')
    obs_rms_params_2 = loaded_model.sess.run(loaded_model.obs_rms_params)
    ret_rms_params_2 = loaded_model.sess.run(loaded_model.ret_rms_params)
    for (param, param_loaded) in zip(obs_rms_params + ret_rms_params, obs_rms_params_2 + ret_rms_params_2):
        assert np.allclose(param, param_loaded)
    del model, loaded_model
    if os.path.exists('./test_ddpg.zip'):
        os.remove('./test_ddpg.zip')

def test_ddpg_popart():
    if False:
        i = 10
        return i + 15
    '\n    Test DDPG with pop-art normalization\n    '
    n_actions = 1
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = DDPG('MlpPolicy', 'Pendulum-v0', memory_limit=50000, normalize_observations=True, normalize_returns=True, nb_rollout_steps=128, nb_train_steps=1, batch_size=64, action_noise=action_noise, enable_popart=True)
    model.learn(NUM_TIMESTEPS)