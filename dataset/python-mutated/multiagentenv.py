from collections import namedtuple
import numpy as np

def convert(dictionary):
    if False:
        return 10
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)

class MultiAgentEnv(object):

    def __init__(self, batch_size=None, **kwargs):
        if False:
            return 10
        args = kwargs['env_args']
        if isinstance(args, dict):
            args = convert(args)
        self.args = args
        if getattr(args, 'seed', None) is not None:
            self.seed = args.seed
            self.rs = np.random.RandomState(self.seed)

    def step(self, actions):
        if False:
            for i in range(10):
                print('nop')
        ' Returns reward, terminated, info '
        raise NotImplementedError

    def get_obs(self):
        if False:
            while True:
                i = 10
        ' Returns all agent observations in a list '
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        if False:
            print('Hello World!')
        ' Returns observation for agent_id '
        raise NotImplementedError

    def get_obs_size(self):
        if False:
            print('Hello World!')
        ' Returns the shape of the observation '
        raise NotImplementedError

    def get_state(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def get_state_size(self):
        if False:
            return 10
        ' Returns the shape of the state'
        raise NotImplementedError

    def get_avail_actions(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        if False:
            i = 10
            return i + 15
        ' Returns the available actions for agent_id '
        raise NotImplementedError

    def get_total_actions(self):
        if False:
            while True:
                i = 10
        ' Returns the total number of actions an agent could ever take '
        raise NotImplementedError

    def get_stats(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def get_agg_stats(self, stats):
        if False:
            return 10
        return {}

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        ' Returns initial observations and states'
        raise NotImplementedError

    def render(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def close(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def seed(self, seed):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def get_env_info(self):
        if False:
            print('Hello World!')
        env_info = {'state_shape': self.get_state_size(), 'obs_shape': self.get_obs_size(), 'n_actions': self.get_total_actions(), 'n_agents': self.n_agents, 'episode_limit': self.episode_limit}
        return env_info