import os
from io import BytesIO
import pytest
import numpy as np
from stable_baselines import A2C, ACER, ACKTR, DQN, PPO1, PPO2, TRPO
from stable_baselines.common.identity_env import IdentityEnv
from stable_baselines.common.vec_env import DummyVecEnv
MODEL_LIST = [A2C, ACER, ACKTR, DQN, PPO1, PPO2, TRPO]

@pytest.mark.parametrize('model_class', MODEL_LIST)
def test_load_parameters(request, model_class):
    if False:
        while True:
            i = 10
    '\n    Test if ``load_parameters`` loads given parameters correctly (the model actually changes)\n    and that the backwards compatability with a list of params works\n\n    :param model_class: (BaseRLModel) A RL model\n    '
    env = DummyVecEnv([lambda : IdentityEnv(10)])
    model = model_class(policy='MlpPolicy', env=env)
    env = model.get_env()
    obs = env.reset()
    observations = np.array([obs for _ in range(10)])
    observations = np.squeeze(observations)
    actions = np.array([env.action_space.sample() for _ in range(10)])
    original_actions_probas = model.action_probability(observations, actions=actions)
    params = model.get_parameters()
    random_params = dict(((param_name, np.random.random(size=param.shape)) for (param_name, param) in params.items()))
    model.load_parameters(random_params)
    new_actions_probas = model.action_probability(observations, actions=actions)
    assert not np.any(np.isclose(original_actions_probas, new_actions_probas)), 'Action probabilities did not change after changing model parameters.'
    new_params = model.get_parameters()
    comparisons = [np.all(np.isclose(new_params[key], random_params[key])) for key in random_params.keys()]
    assert all(comparisons), 'Parameters of model are not the same as provided ones.'
    tf_param_list = model.get_parameter_list()
    random_param_list = [-np.random.random(size=tf_param.shape) for tf_param in tf_param_list]
    model.load_parameters(random_param_list)
    new_actions_probas_list = model.action_probability(observations, actions=actions)
    assert not np.any(np.isclose(new_actions_probas, new_actions_probas_list)), 'Action probabilities did not change after changing model parameters (list).'
    original_actions_probas = model.action_probability(observations, actions=actions)
    model_fname = './test_model_{}.zip'.format(request.node.name)
    try:
        model.save(model_fname)
        b_io = BytesIO()
        model.save(b_io)
        model_bytes = b_io.getvalue()
        b_io.close()
        random_params = dict(((param_name, np.random.random(size=param.shape)) for (param_name, param) in params.items()))
        model.load_parameters(random_params)
        model.load_parameters(model_fname)
        new_actions_probas = model.action_probability(observations, actions=actions)
        assert np.all(np.isclose(original_actions_probas, new_actions_probas)), 'Action probabilities changed after load_parameters from a file.'
        model.load_parameters(random_params)
        b_io = BytesIO(model_bytes)
        model.load_parameters(b_io)
        b_io.close()
        new_actions_probas = model.action_probability(observations, actions=actions)
        assert np.all(np.isclose(original_actions_probas, new_actions_probas)), 'Action probabilities changed afterload_parameters from a file-like.'
    finally:
        if os.path.exists(model_fname):
            os.remove(model_fname)
    original_actions_probas = model.action_probability(observations, actions=actions)
    truncated_random_params = dict(((param_name, np.random.random(size=param.shape)) for (param_name, param) in params.items()))
    _ = truncated_random_params.pop(list(truncated_random_params.keys())[0])
    with pytest.raises(RuntimeError):
        model.load_parameters(truncated_random_params, exact_match=True)
    new_actions_probas = model.action_probability(observations, actions=actions)
    assert np.all(np.isclose(original_actions_probas, new_actions_probas)), 'Action probabilities changed after load_parameters raised RunTimeError (exact_match=True).'
    model.load_parameters(truncated_random_params, exact_match=False)
    new_actions_probas = model.action_probability(observations, actions=actions)
    assert not np.any(np.isclose(original_actions_probas, new_actions_probas)), 'Action probabilities did not change after changing model parameters (exact_match=False).'
    del model, env