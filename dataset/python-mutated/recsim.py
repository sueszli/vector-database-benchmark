"""Tools and utils to create RLlib-ready recommender system envs using RecSim.

For examples on how to generate a RecSim env class (usable in RLlib):
See ray.rllib.examples.env.recommender_system_envs_with_recsim.py

For more information on google's RecSim itself:
https://github.com/google-research/recsim
"""
from collections import OrderedDict
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, MultiDiscrete
from gymnasium.wrappers import EnvCompatibility
import numpy as np
from recsim.document import AbstractDocumentSampler
from recsim.simulator import environment, recsim_gym
from recsim.user import AbstractUserModel, AbstractResponse
from typing import Callable, List, Optional, Type
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.gym import convert_old_gym_space_to_gymnasium_space
from ray.rllib.utils.spaces.space_utils import convert_element_to_space_type

class RecSimObservationSpaceWrapper(gym.ObservationWrapper):
    """Fix RecSim environment's observation space

    In RecSim's observation spaces, the "doc" field is a dictionary keyed by
    document IDs. Those IDs are changing every step, thus generating a
    different observation space in each time. This causes issues for RLlib
    because it expects the observation space to remain the same across steps.

    This environment wrapper fixes that by reindexing the documents by their
    positions in the list.
    """

    def __init__(self, env: gym.Env):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(env)
        obs_space = convert_old_gym_space_to_gymnasium_space(self.env.observation_space)
        doc_space = Dict(OrderedDict([(str(k), doc) for (k, (_, doc)) in enumerate(obs_space['doc'].spaces.items())]))
        self.observation_space = Dict(OrderedDict([('user', obs_space['user']), ('doc', doc_space), ('response', obs_space['response'])]))
        self._sampled_obs = self.observation_space.sample()
        self.action_space = convert_old_gym_space_to_gymnasium_space(self.env.action_space)

    def observation(self, obs):
        if False:
            return 10
        new_obs = OrderedDict()
        new_obs['user'] = obs['user']
        new_obs['doc'] = {str(k): v for (k, (_, v)) in enumerate(obs['doc'].items())}
        new_obs['response'] = obs['response']
        new_obs = convert_element_to_space_type(new_obs, self._sampled_obs)
        return new_obs

class RecSimObservationBanditWrapper(gym.ObservationWrapper):
    """Fix RecSim environment's observation format

    RecSim's observations are keyed by document IDs, and nested under
    "doc" key.
    Our Bandits agent expects the observations to be flat 2D array
    and under "item" key.

    This environment wrapper converts obs into the right format.
    """

    def __init__(self, env: gym.Env):
        if False:
            while True:
                i = 10
        super().__init__(env)
        obs_space = convert_old_gym_space_to_gymnasium_space(self.env.observation_space)
        num_items = len(obs_space['doc'])
        embedding_dim = next(iter(obs_space['doc'].values())).shape[-1]
        self.observation_space = Dict(OrderedDict([('item', gym.spaces.Box(low=-1.0, high=1.0, shape=(num_items, embedding_dim)))]))
        self._sampled_obs = self.observation_space.sample()
        self.action_space = convert_old_gym_space_to_gymnasium_space(self.env.action_space)

    def observation(self, obs):
        if False:
            print('Hello World!')
        new_obs = OrderedDict()
        new_obs['item'] = np.vstack(list(obs['doc'].values()))
        new_obs = convert_element_to_space_type(new_obs, self._sampled_obs)
        return new_obs

class RecSimResetWrapper(gym.Wrapper):
    """Fix RecSim environment's reset() and close() function

    RecSim's reset() function returns an observation without the "response"
    field, breaking RLlib's check. This wrapper fixes that by assigning a
    random "response".

    RecSim's close() function raises NotImplementedError. We change the
    behavior to doing nothing.
    """

    def __init__(self, env: gym.Env):
        if False:
            while True:
                i = 10
        super().__init__(env)
        self._sampled_obs = self.env.observation_space.sample()

    def reset(self, *, seed=None, options=None):
        if False:
            i = 10
            return i + 15
        (obs, info) = super().reset()
        obs['response'] = self.env.observation_space['response'].sample()
        obs = convert_element_to_space_type(obs, self._sampled_obs)
        return (obs, info)

    def close(self):
        if False:
            i = 10
            return i + 15
        pass

class MultiDiscreteToDiscreteActionWrapper(gym.ActionWrapper):
    """Convert the action space from MultiDiscrete to Discrete

    At this moment, RLlib's DQN algorithms only work on Discrete action space.
    This wrapper allows us to apply DQN algorithms to the RecSim environment.
    """

    def __init__(self, env: gym.Env):
        if False:
            return 10
        super().__init__(env)
        if not isinstance(env.action_space, MultiDiscrete):
            raise UnsupportedSpaceException(f'Action space {env.action_space} is not supported by {self.__class__.__name__}')
        self.action_space_dimensions = env.action_space.nvec
        self.action_space = Discrete(np.prod(self.action_space_dimensions))

    def action(self, action: int) -> List[int]:
        if False:
            i = 10
            return i + 15
        'Convert a Discrete action to a MultiDiscrete action'
        multi_action = [None] * len(self.action_space_dimensions)
        for (idx, n) in enumerate(self.action_space_dimensions):
            (action, dim_action) = divmod(action, n)
            multi_action[idx] = dim_action
        return multi_action

def recsim_gym_wrapper(recsim_gym_env: gym.Env, convert_to_discrete_action_space: bool=False, wrap_for_bandits: bool=False) -> gym.Env:
    if False:
        i = 10
        return i + 15
    'Makes sure a RecSim gym.Env can ba handled by RLlib.\n\n    In RecSim\'s observation spaces, the "doc" field is a dictionary keyed by\n    document IDs. Those IDs are changing every step, thus generating a\n    different observation space in each time. This causes issues for RLlib\n    because it expects the observation space to remain the same across steps.\n\n    Also, RecSim\'s reset() function returns an observation without the\n    "response" field, breaking RLlib\'s check. This wrapper fixes that by\n    assigning a random "response".\n\n    Args:\n        recsim_gym_env: The RecSim gym.Env instance. Usually resulting from a\n            raw RecSim env having been passed through RecSim\'s utility function:\n            `recsim.simulator.recsim_gym.RecSimGymEnv()`.\n        convert_to_discrete_action_space: Optional bool indicating, whether\n            the action space of the created env class should be Discrete\n            (rather than MultiDiscrete, even if slate size > 1). This is useful\n            for algorithms that don\'t support MultiDiscrete action spaces,\n            such as RLlib\'s DQN. If None, `convert_to_discrete_action_space`\n            may also be provided via the EnvContext (config) when creating an\n            actual env instance.\n        wrap_for_bandits: Bool indicating, whether this RecSim env should be\n            wrapped for use with our Bandits agent.\n\n    Returns:\n        An RLlib-ready gym.Env instance.\n    '
    env = RecSimResetWrapper(recsim_gym_env)
    env = RecSimObservationSpaceWrapper(env)
    if convert_to_discrete_action_space:
        env = MultiDiscreteToDiscreteActionWrapper(env)
    if wrap_for_bandits:
        env = RecSimObservationBanditWrapper(env)
    return env

def make_recsim_env(recsim_user_model_creator: Callable[[EnvContext], AbstractUserModel], recsim_document_sampler_creator: Callable[[EnvContext], AbstractDocumentSampler], reward_aggregator: Callable[[List[AbstractResponse]], float]) -> Type[gym.Env]:
    if False:
        return 10
    'Creates a RLlib-ready gym.Env class given RecSim user and doc models.\n\n    See https://github.com/google-research/recsim for more information on how to\n    build the required components from scratch in python using RecSim.\n\n    Args:\n        recsim_user_model_creator: A callable taking an EnvContext and returning\n            a RecSim AbstractUserModel instance to use.\n        recsim_document_sampler_creator: A callable taking an EnvContext and\n            returning a RecSim AbstractDocumentSampler\n            to use. This will include a AbstractDocument as well.\n        reward_aggregator: Callable taking a list of RecSim\n            AbstractResponse instances and returning a float (aggregated\n            reward).\n\n    Returns:\n        An RLlib-ready gym.Env class to use inside an Algorithm.\n    '

    class _RecSimEnv(gym.Wrapper):

        def __init__(self, config: Optional[EnvContext]=None):
            if False:
                for i in range(10):
                    print('nop')
            default_config = {'num_candidates': 10, 'slate_size': 2, 'resample_documents': True, 'seed': 0, 'convert_to_discrete_action_space': False, 'wrap_for_bandits': False}
            if config is None or isinstance(config, dict):
                config = EnvContext(config or default_config, worker_index=0)
            config.set_defaults(default_config)
            recsim_user_model = recsim_user_model_creator(config)
            recsim_document_sampler = recsim_document_sampler_creator(config)
            raw_recsim_env = environment.SingleUserEnvironment(recsim_user_model, recsim_document_sampler, config['num_candidates'], config['slate_size'], resample_documents=config['resample_documents'])
            gym_env = recsim_gym.RecSimGymEnv(raw_recsim_env, reward_aggregator)
            gym_env = EnvCompatibility(gym_env)
            env = recsim_gym_wrapper(gym_env, config['convert_to_discrete_action_space'], config['wrap_for_bandits'])
            super().__init__(env=env)
    return _RecSimEnv