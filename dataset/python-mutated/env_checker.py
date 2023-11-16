import warnings
from typing import Any, Dict, Union
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.preprocessing import check_for_nested_spaces, is_image_space_channels_first
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan

def _is_numpy_array_space(space: spaces.Space) -> bool:
    if False:
        print('Hello World!')
    '\n    Returns False if provided space is not representable as a single numpy array\n    (e.g. Dict and Tuple spaces return False)\n    '
    return not isinstance(space, (spaces.Dict, spaces.Tuple))

def _check_image_input(observation_space: spaces.Box, key: str='') -> None:
    if False:
        while True:
            i = 10
    '\n    Check that the input will be compatible with Stable-Baselines\n    when the observation is apparently an image.\n\n    :param observation_space: Observation space\n    :key: When the observation space comes from a Dict space, we pass the\n        corresponding key to have more precise warning messages. Defaults to "".\n    '
    if observation_space.dtype != np.uint8:
        warnings.warn(f'It seems that your observation {key} is an image but its `dtype` is ({observation_space.dtype}) whereas it has to be `np.uint8`. If your observation is not an image, we recommend you to flatten the observation to have only a 1D vector')
    if np.any(observation_space.low != 0) or np.any(observation_space.high != 255):
        warnings.warn(f'It seems that your observation space {key} is an image but the upper and lower bounds are not in [0, 255]. Because the CNN policy normalize automatically the observation you may encounter issue if the values are not in that range.')
    non_channel_idx = 0
    if is_image_space_channels_first(observation_space):
        non_channel_idx = -1
    if observation_space.shape[non_channel_idx] < 36 or observation_space.shape[1] < 36:
        warnings.warn('The minimal resolution for an image is 36x36 for the default `CnnPolicy`. You might need to use a custom features extractor cf. https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html')

def _check_unsupported_spaces(env: gym.Env, observation_space: spaces.Space, action_space: spaces.Space) -> None:
    if False:
        while True:
            i = 10
    'Emit warnings when the observation space or action space used is not supported by Stable-Baselines.'
    if isinstance(observation_space, spaces.Dict):
        nested_dict = False
        for (key, space) in observation_space.spaces.items():
            if isinstance(space, spaces.Dict):
                nested_dict = True
            if isinstance(space, spaces.Discrete) and space.start != 0:
                warnings.warn(f"Discrete observation space (key '{key}') with a non-zero start is not supported by Stable-Baselines3. You can use a wrapper or update your observation space.")
        if nested_dict:
            warnings.warn('Nested observation spaces are not supported by Stable Baselines3 (Dict spaces inside Dict space). You should flatten it to have only one level of keys.For example, `dict(space1=dict(space2=Box(), space3=Box()), spaces4=Discrete())` is not supported but `dict(space2=Box(), spaces3=Box(), spaces4=Discrete())` is.')
    if isinstance(observation_space, spaces.Tuple):
        warnings.warn('The observation space is a Tuple, this is currently not supported by Stable Baselines3. However, you can convert it to a Dict observation space (cf. https://gymnasium.farama.org/api/spaces/composite/#dict). which is supported by SB3.')
    if isinstance(observation_space, spaces.Discrete) and observation_space.start != 0:
        warnings.warn('Discrete observation space with a non-zero start is not supported by Stable-Baselines3. You can use a wrapper or update your observation space.')
    if isinstance(observation_space, spaces.Sequence):
        warnings.warn('Sequence observation space is not supported by Stable-Baselines3. You can pad your observation to have a fixed size instead.\nNote: The checks for returned values are skipped.')
    if isinstance(action_space, spaces.Discrete) and action_space.start != 0:
        warnings.warn('Discrete action space with a non-zero start is not supported by Stable-Baselines3. You can use a wrapper or update your action space.')
    if not _is_numpy_array_space(action_space):
        warnings.warn("The action space is not based off a numpy array. Typically this means it's either a Dict or Tuple space. This type of action space is currently not supported by Stable Baselines 3. You should try to flatten the action using a wrapper.")

def _check_nan(env: gym.Env) -> None:
    if False:
        while True:
            i = 10
    'Check for Inf and NaN using the VecWrapper.'
    vec_env = VecCheckNan(DummyVecEnv([lambda : env]))
    vec_env.reset()
    for _ in range(10):
        action = np.array([env.action_space.sample()])
        (_, _, _, _) = vec_env.step(action)

def _is_goal_env(env: gym.Env) -> bool:
    if False:
        while True:
            i = 10
    '\n    Check if the env uses the convention for goal-conditioned envs (previously, the gym.GoalEnv interface)\n    '
    return hasattr(env.unwrapped, 'compute_reward')

def _check_goal_env_obs(obs: dict, observation_space: spaces.Dict, method_name: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Check that an environment implementing the `compute_rewards()` method\n    (previously known as GoalEnv in gym) contains at least three elements,\n    namely `observation`, `achieved_goal`, and `desired_goal`.\n    '
    assert len(observation_space.spaces) >= 3, f'A goal conditioned env must contain at least 3 observation keys: `observation`, `achieved_goal`, and `desired_goal`. The current observation contains {len(observation_space.spaces)} keys: {list(observation_space.spaces.keys())}'
    for key in ['achieved_goal', 'desired_goal']:
        if key not in observation_space.spaces:
            raise AssertionError(f"The observation returned by the `{method_name}()` method of a goal-conditioned env requires the '{key}' key to be part of the observation dictionary. Current keys are {list(observation_space.spaces.keys())}")

def _check_goal_env_compute_reward(obs: Dict[str, Union[np.ndarray, int]], env: gym.Env, reward: float, info: Dict[str, Any]) -> None:
    if False:
        print('Hello World!')
    '\n    Check that reward is computed with `compute_reward`\n    and that the implementation is vectorized.\n    '
    (achieved_goal, desired_goal) = (obs['achieved_goal'], obs['desired_goal'])
    assert reward == env.compute_reward(achieved_goal, desired_goal, info), 'The reward was not computed with `compute_reward()`'
    (achieved_goal, desired_goal) = (np.array(achieved_goal), np.array(desired_goal))
    batch_achieved_goals = np.array([achieved_goal, achieved_goal])
    batch_desired_goals = np.array([desired_goal, desired_goal])
    if isinstance(achieved_goal, int) or len(achieved_goal.shape) == 0:
        batch_achieved_goals = batch_achieved_goals.reshape(2, 1)
        batch_desired_goals = batch_desired_goals.reshape(2, 1)
    batch_infos = np.array([info, info])
    rewards = env.compute_reward(batch_achieved_goals, batch_desired_goals, batch_infos)
    assert rewards.shape == (2,), f'Unexpected shape for vectorized computation of reward: {rewards.shape} != (2,)'
    assert rewards[0] == reward, f'Vectorized computation of reward differs from single computation: {rewards[0]} != {reward}'

def _check_obs(obs: Union[tuple, dict, np.ndarray, int], observation_space: spaces.Space, method_name: str) -> None:
    if False:
        while True:
            i = 10
    '\n    Check that the observation returned by the environment\n    correspond to the declared one.\n    '
    if not isinstance(observation_space, spaces.Tuple):
        assert not isinstance(obs, tuple), f'The observation returned by the `{method_name}()` method should be a single value, not a tuple'
    if isinstance(observation_space, spaces.Discrete):
        assert np.issubdtype(type(obs), np.integer), f'The observation returned by `{method_name}()` method must be an int'
    elif _is_numpy_array_space(observation_space):
        assert isinstance(obs, np.ndarray), f'The observation returned by `{method_name}()` method must be a numpy array'
    if isinstance(obs, np.ndarray):
        assert observation_space.shape == obs.shape, f'The observation returned by the `{method_name}()` method does not match the shape of the given observation space {observation_space}. Expected: {observation_space.shape}, actual shape: {obs.shape}'
        assert np.can_cast(obs.dtype, observation_space.dtype), f'The observation returned by the `{method_name}()` method does not match the data type (cannot cast) of the given observation space {observation_space}. Expected: {observation_space.dtype}, actual dtype: {obs.dtype}'
        if isinstance(observation_space, spaces.Box):
            (lower_bounds, upper_bounds) = (observation_space.low, observation_space.high)
            invalid_indices = np.where(np.logical_or(obs < lower_bounds, obs > upper_bounds))
            if (obs > upper_bounds).any() or (obs < lower_bounds).any():
                message = f'The observation returned by the `{method_name}()` method does not match the bounds of the given observation space {observation_space}. \n'
                message += f'{len(invalid_indices[0])} invalid indices: \n'
                for index in zip(*invalid_indices):
                    index_str = ','.join(map(str, index))
                    message += f'Expected: {lower_bounds[index]} <= obs[{index_str}] <= {upper_bounds[index]}, actual value: {obs[index]} \n'
                raise AssertionError(message)
    assert observation_space.contains(obs), f'The observation returned by the `{method_name}()` method does not match the given observation space {observation_space}'

def _check_box_obs(observation_space: spaces.Box, key: str='') -> None:
    if False:
        print('Hello World!')
    '\n    Check that the observation space is correctly formatted\n    when dealing with a ``Box()`` space. In particular, it checks:\n    - that the dimensions are big enough when it is an image, and that the type matches\n    - that the observation has an expected shape (warn the user if not)\n    '
    if len(observation_space.shape) == 3:
        _check_image_input(observation_space, key)
    if len(observation_space.shape) not in [1, 3]:
        warnings.warn(f'Your observation {key} has an unconventional shape (neither an image, nor a 1D vector). We recommend you to flatten the observation to have only a 1D vector or use a custom policy to properly process the data.')

def _check_returned_values(env: gym.Env, observation_space: spaces.Space, action_space: spaces.Space) -> None:
    if False:
        print('Hello World!')
    '\n    Check the returned values by the env when calling `.reset()` or `.step()` methods.\n    '
    reset_returns = env.reset()
    assert isinstance(reset_returns, tuple), '`reset()` must return a tuple (obs, info)'
    assert len(reset_returns) == 2, f'`reset()` must return a tuple of size 2 (obs, info), not {len(reset_returns)}'
    (obs, info) = reset_returns
    assert isinstance(info, dict), f'The second element of the tuple return by `reset()` must be a dictionary not {info}'
    if _is_goal_env(env):
        assert isinstance(observation_space, spaces.Dict)
        _check_goal_env_obs(obs, observation_space, 'reset')
    elif isinstance(observation_space, spaces.Dict):
        assert isinstance(obs, dict), 'The observation returned by `reset()` must be a dictionary'
        if not obs.keys() == observation_space.spaces.keys():
            raise AssertionError(f'The observation keys returned by `reset()` must match the observation space keys: {obs.keys()} != {observation_space.spaces.keys()}')
        for key in observation_space.spaces.keys():
            try:
                _check_obs(obs[key], observation_space.spaces[key], 'reset')
            except AssertionError as e:
                raise AssertionError(f'Error while checking key={key}: ' + str(e)) from e
    else:
        _check_obs(obs, observation_space, 'reset')
    action = action_space.sample()
    data = env.step(action)
    assert len(data) == 5, f'The `step()` method must return five values: obs, reward, terminated, truncated, info. Actual: {len(data)} values returned.'
    (obs, reward, terminated, truncated, info) = data
    if isinstance(observation_space, spaces.Dict):
        assert isinstance(obs, dict), 'The observation returned by `step()` must be a dictionary'
        if _is_goal_env(env):
            assert isinstance(observation_space, spaces.Dict)
            _check_goal_env_obs(obs, observation_space, 'step')
            _check_goal_env_compute_reward(obs, env, float(reward), info)
        if not obs.keys() == observation_space.spaces.keys():
            raise AssertionError(f'The observation keys returned by `step()` must match the observation space keys: {obs.keys()} != {observation_space.spaces.keys()}')
        for key in observation_space.spaces.keys():
            try:
                _check_obs(obs[key], observation_space.spaces[key], 'step')
            except AssertionError as e:
                raise AssertionError(f'Error while checking key={key}: ' + str(e)) from e
    else:
        _check_obs(obs, observation_space, 'step')
    assert isinstance(reward, (float, int)), 'The reward returned by `step()` must be a float'
    assert isinstance(terminated, bool), 'The `terminated` signal must be a boolean'
    assert isinstance(truncated, bool), 'The `truncated` signal must be a boolean'
    assert isinstance(info, dict), 'The `info` returned by `step()` must be a python dictionary'
    if _is_goal_env(env):
        assert hasattr(env, 'compute_reward')
        assert reward == env.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)

def _check_spaces(env: gym.Env) -> None:
    if False:
        return 10
    '\n    Check that the observation and action spaces are defined and inherit from spaces.Space. For\n    envs that follow the goal-conditioned standard (previously, the gym.GoalEnv interface) we check\n    the observation space is gymnasium.spaces.Dict\n    '
    gym_spaces = 'cf. https://gymnasium.farama.org/api/spaces/'
    assert hasattr(env, 'observation_space'), f'You must specify an observation space ({gym_spaces})'
    assert hasattr(env, 'action_space'), f'You must specify an action space ({gym_spaces})'
    assert isinstance(env.observation_space, spaces.Space), f'The observation space must inherit from gymnasium.spaces ({gym_spaces})'
    assert isinstance(env.action_space, spaces.Space), f'The action space must inherit from gymnasium.spaces ({gym_spaces})'
    if _is_goal_env(env):
        print("We detected your env to be a GoalEnv because `env.compute_reward()` was defined.\nIf it's not the case, please rename `env.compute_reward()` to something else to avoid False positives.")
        assert isinstance(env.observation_space, spaces.Dict), 'Goal conditioned envs (previously gym.GoalEnv) require the observation space to be gymnasium.spaces.Dict.\nNote: if your env is not a GoalEnv, please rename `env.compute_reward()` to something else to avoid False positive.'

def _check_render(env: gym.Env, warn: bool=False) -> None:
    if False:
        while True:
            i = 10
    '\n    Check the instantiated render mode (if any) by calling the `render()`/`close()`\n    method of the environment.\n\n    :param env: The environment to check\n    :param warn: Whether to output additional warnings\n    :param headless: Whether to disable render modes\n        that require a graphical interface. False by default.\n    '
    render_modes = env.metadata.get('render_modes')
    if render_modes is None:
        if warn:
            warnings.warn("No render modes was declared in the environment (env.metadata['render_modes'] is None or not defined), you may have trouble when calling `.render()`")
    if env.render_mode:
        env.render()
    env.close()

def check_env(env: gym.Env, warn: bool=True, skip_render_check: bool=True) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Check that an environment follows Gym API.\n    This is particularly useful when using a custom environment.\n    Please take a look at https://gymnasium.farama.org/api/env/\n    for more information about the API.\n\n    It also optionally check that the environment is compatible with Stable-Baselines.\n\n    :param env: The Gym environment that will be checked\n    :param warn: Whether to output additional warnings\n        mainly related to the interaction with Stable Baselines\n    :param skip_render_check: Whether to skip the checks for the render method.\n        True by default (useful for the CI)\n    '
    assert isinstance(env, gym.Env), 'Your environment must inherit from the gymnasium.Env class cf. https://gymnasium.farama.org/api/env/'
    _check_spaces(env)
    observation_space = env.observation_space
    action_space = env.action_space
    try:
        env.reset(seed=0)
    except TypeError as e:
        raise TypeError('The reset() method must accept a `seed` parameter') from e
    if warn:
        _check_unsupported_spaces(env, observation_space, action_space)
        obs_spaces = observation_space.spaces if isinstance(observation_space, spaces.Dict) else {'': observation_space}
        for (key, space) in obs_spaces.items():
            if isinstance(space, spaces.Box):
                _check_box_obs(space, key)
        if isinstance(action_space, spaces.Box) and (np.any(np.abs(action_space.low) != np.abs(action_space.high)) or np.any(action_space.low != -1) or np.any(action_space.high != 1)):
            warnings.warn('We recommend you to use a symmetric and normalized Box action space (range=[-1, 1]) cf. https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html')
        if isinstance(action_space, spaces.Box):
            assert np.all(np.isfinite(np.array([action_space.low, action_space.high]))), 'Continuous action space must have a finite lower and upper bound'
        if isinstance(action_space, spaces.Box) and action_space.dtype != np.dtype(np.float32):
            warnings.warn(f'Your action space has dtype {action_space.dtype}, we recommend using np.float32 to avoid cast errors.')
    if isinstance(observation_space, spaces.Sequence):
        return
    _check_returned_values(env, observation_space, action_space)
    if not skip_render_check:
        _check_render(env, warn)
    try:
        check_for_nested_spaces(env.observation_space)
        _check_nan(env)
    except NotImplementedError:
        pass