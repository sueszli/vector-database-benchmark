import warnings
from inspect import signature
from typing import Union
import gymnasium
try:
    import gym
    gym_installed = True
except ImportError:
    gym_installed = False

def _patch_env(env: Union['gym.Env', gymnasium.Env]) -> gymnasium.Env:
    if False:
        return 10
    '\n    Adapted from https://github.com/thu-ml/tianshou.\n\n    Takes an environment and patches it to return Gymnasium env.\n    This function takes the environment object and returns a patched\n    env, using shimmy wrapper to convert it to Gymnasium,\n    if necessary.\n\n    :param env: A gym/gymnasium env\n    :return: Patched env (gymnasium env)\n    '
    if isinstance(env, gymnasium.Env):
        return env
    if not gym_installed or not isinstance(env, gym.Env):
        raise ValueError(f'The environment is of type {type(env)}, not a Gymnasium environment. In this case, we expect OpenAI Gym to be installed and the environment to be an OpenAI Gym environment.')
    try:
        import shimmy
    except ImportError as e:
        raise ImportError("Missing shimmy installation. You provided an OpenAI Gym environment. Stable-Baselines3 (SB3) has transitioned to using Gymnasium internally. In order to use OpenAI Gym environments with SB3, you need to install shimmy (`pip install 'shimmy>=0.2.1'`).") from e
    warnings.warn('You provided an OpenAI Gym environment. We strongly recommend transitioning to Gymnasium environments. Stable-Baselines3 is automatically wrapping your environments in a compatibility layer, which could potentially cause issues.')
    if 'seed' in signature(env.unwrapped.reset).parameters:
        return shimmy.GymV26CompatibilityV0(env=env)
    return shimmy.GymV21CompatibilityV0(env=env)

def _convert_space(space: Union['gym.Space', gymnasium.Space]) -> gymnasium.Space:
    if False:
        print('Hello World!')
    '\n    Takes a space and patches it to return Gymnasium Space.\n    This function takes the space object and returns a patched\n    space, using shimmy wrapper to convert it to Gymnasium,\n    if necessary.\n\n    :param env: A gym/gymnasium Space\n    :return: Patched space (gymnasium Space)\n    '
    if isinstance(space, gymnasium.Space):
        return space
    if not gym_installed or not isinstance(space, gym.Space):
        raise ValueError(f'The space is of type {type(space)}, not a Gymnasium space. In this case, we expect OpenAI Gym to be installed and the space to be an OpenAI Gym space.')
    try:
        import shimmy
    except ImportError as e:
        raise ImportError("Missing shimmy installation. You provided an OpenAI Gym space. Stable-Baselines3 (SB3) has transitioned to using Gymnasium internally. In order to use OpenAI Gym space with SB3, you need to install shimmy (`pip install 'shimmy>=0.2.1'`).") from e
    warnings.warn('You loaded a model that was trained using OpenAI Gym. We strongly recommend transitioning to Gymnasium by saving that model again.')
    return shimmy.openai_gym_compatibility._convert_space(space)