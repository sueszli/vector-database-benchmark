import functools
import gymnasium as gym
import numpy as np
from typing import Optional, Union
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.exploration.random import Random
from ray.rllib.utils.framework import get_variable, try_import_tf, try_import_torch, TensorType
from ray.rllib.utils.tf_utils import zero_logps_from_actions
(tf1, tf, tfv) = try_import_tf()
(torch, _) = try_import_torch()

@PublicAPI
class StochasticSampling(Exploration):
    """An exploration that simply samples from a distribution.

    The sampling can be made deterministic by passing explore=False into
    the call to `get_exploration_action`.
    Also allows for scheduled parameters for the distributions, such as
    lowering stddev, temperature, etc.. over time.
    """

    def __init__(self, action_space: gym.spaces.Space, *, framework: str, model: ModelV2, random_timesteps: int=0, **kwargs):
        if False:
            print('Hello World!')
        'Initializes a StochasticSampling Exploration object.\n\n        Args:\n            action_space: The gym action space used by the environment.\n            framework: One of None, "tf", "torch".\n            model: The ModelV2 used by the owning Policy.\n            random_timesteps: The number of timesteps for which to act\n                completely randomly. Only after this number of timesteps,\n                actual samples will be drawn to get exploration actions.\n        '
        assert framework is not None
        super().__init__(action_space, model=model, framework=framework, **kwargs)
        self.random_timesteps = random_timesteps
        self.random_exploration = Random(action_space, model=self.model, framework=self.framework, **kwargs)
        self.last_timestep = get_variable(np.array(0, np.int64), framework=self.framework, tf_name='timestep', dtype=np.int64)

    @override(Exploration)
    def get_exploration_action(self, *, action_distribution: ActionDistribution, timestep: Optional[Union[int, TensorType]]=None, explore: bool=True):
        if False:
            i = 10
            return i + 15
        if self.framework == 'torch':
            return self._get_torch_exploration_action(action_distribution, timestep, explore)
        else:
            return self._get_tf_exploration_action_op(action_distribution, timestep, explore)

    def _get_tf_exploration_action_op(self, action_dist, timestep, explore):
        if False:
            i = 10
            return i + 15
        ts = self.last_timestep + 1
        stochastic_actions = tf.cond(pred=tf.convert_to_tensor(ts < self.random_timesteps), true_fn=lambda : self.random_exploration.get_tf_exploration_action_op(action_dist, explore=True)[0], false_fn=lambda : action_dist.sample())
        deterministic_actions = action_dist.deterministic_sample()
        action = tf.cond(tf.constant(explore) if isinstance(explore, bool) else explore, true_fn=lambda : stochastic_actions, false_fn=lambda : deterministic_actions)
        logp = tf.cond(tf.math.logical_and(explore, tf.convert_to_tensor(ts >= self.random_timesteps)), true_fn=lambda : action_dist.sampled_action_logp(), false_fn=functools.partial(zero_logps_from_actions, deterministic_actions))
        if self.framework == 'tf2':
            self.last_timestep.assign_add(1)
            return (action, logp)
        else:
            assign_op = tf1.assign_add(self.last_timestep, 1) if timestep is None else tf1.assign(self.last_timestep, timestep)
            with tf1.control_dependencies([assign_op]):
                return (action, logp)

    def _get_torch_exploration_action(self, action_dist: ActionDistribution, timestep: Union[TensorType, int], explore: Union[TensorType, bool]):
        if False:
            for i in range(10):
                print('nop')
        self.last_timestep = timestep if timestep is not None else self.last_timestep + 1
        if explore:
            if self.last_timestep < self.random_timesteps:
                (action, logp) = self.random_exploration.get_torch_exploration_action(action_dist, explore=True)
            else:
                action = action_dist.sample()
                logp = action_dist.sampled_action_logp()
        else:
            action = action_dist.deterministic_sample()
            logp = torch.zeros_like(action_dist.sampled_action_logp())
        return (action, logp)