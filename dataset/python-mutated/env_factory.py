from gym import envs
from . import maze2d, puddleworld

def register_env(env_id):
    if False:
        i = 10
        return i + 15
    if env_id == 'Maze2D-v0':
        envs.register(id=env_id, entry_point='env:maze2d.Maze2D', kwargs={}, max_episode_steps=200, reward_threshold=-110.0)
    elif env_id == 'PuddleWorld-v0':
        envs.register(id=env_id, entry_point='env:puddleworld.PuddleWorld', kwargs={}, max_episode_steps=200, reward_threshold=-100.0)
    else:
        raise ValueError('Cannot find environment "{0}"\n'.format(env_id))
    return True