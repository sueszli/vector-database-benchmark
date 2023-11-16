from typing import TYPE_CHECKING, Optional
from numpy import ndarray
if TYPE_CHECKING:
    from ding.envs import BaseEnv, BaseEnvManager

def render_env(env, render_mode: Optional[str]='rgb_array') -> 'ndarray':
    if False:
        for i in range(10):
            print('nop')
    "\n    Overview:\n        Render the environment's current frame.\n    Arguments:\n        - env (:obj:`gym.Env`): DI-engine env instance.\n        - render_mode (:obj:`str`): Render mode.\n    Returns:\n        - frame (:obj:`numpy.ndarray`): [H * W * C]\n    "
    if hasattr(env, 'sim'):
        return env.sim.render(camera_name='track', height=128, width=128)[::-1]
    else:
        return env.render(mode=render_mode)

def render(env: 'BaseEnv', render_mode: Optional[str]='rgb_array') -> 'ndarray':
    if False:
        while True:
            i = 10
    "\n    Overview:\n        Render the environment's current frame.\n    Arguments:\n        - env (:obj:`BaseEnv`): DI-engine env instance.\n        - render_mode (:obj:`str`): Render mode.\n    Returns:\n        - frame (:obj:`numpy.ndarray`): [H * W * C]\n    "
    gym_env = env._env
    return render_env(gym_env, render_mode=render_mode)

def get_env_fps(env) -> 'int':
    if False:
        print('Hello World!')
    "\n    Overview:\n        Get the environment's fps.\n    Arguments:\n        - env (:obj:`gym.Env`): DI-engine env instance.\n    Returns:\n        - fps (:obj:`int`).\n    "
    if hasattr(env, 'model'):
        fps = 1 / env.model.opt.timestep
    elif hasattr(env, 'env') and 'video.frames_per_second' in env.env.metadata.keys():
        fps = env.env.metadata['video.frames_per_second']
    else:
        fps = 30
    return fps

def fps(env_manager: 'BaseEnvManager') -> 'int':
    if False:
        print('Hello World!')
    "\n    Overview:\n        Render the environment's fps.\n    Arguments:\n        - env (:obj:`BaseEnvManager`): DI-engine env manager instance.\n    Returns:\n        - fps (:obj:`int`).\n    "
    try:
        gym_env = env_manager.env_ref._env
        return get_env_fps(gym_env)
    except:
        return 30