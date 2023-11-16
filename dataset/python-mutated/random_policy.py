from gymnasium.spaces import Box
import numpy as np
import random
import tree
from typing import List, Optional, Union
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelWeights, TensorStructType, TensorType

class RandomPolicy(Policy):
    """Hand-coded policy that returns random actions."""

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        if self.config.get('ignore_action_bounds', False) and isinstance(self.action_space, Box):
            self.action_space_for_sampling = Box(-float('inf'), float('inf'), shape=self.action_space.shape, dtype=self.action_space.dtype)
        else:
            self.action_space_for_sampling = self.action_space

    @override(Policy)
    def init_view_requirements(self):
        if False:
            return 10
        super().init_view_requirements()
        vr = self.view_requirements[SampleBatch.INFOS]
        vr.used_for_training = False
        vr.used_for_compute_actions = False

    @override(Policy)
    def compute_actions(self, obs_batch: Union[List[TensorStructType], TensorStructType], state_batches: Optional[List[TensorType]]=None, prev_action_batch: Union[List[TensorStructType], TensorStructType]=None, prev_reward_batch: Union[List[TensorStructType], TensorStructType]=None, **kwargs):
        if False:
            i = 10
            return i + 15
        obs_batch_size = len(tree.flatten(obs_batch)[0])
        return ([self.action_space_for_sampling.sample() for _ in range(obs_batch_size)], [], {})

    @override(Policy)
    def learn_on_batch(self, samples):
        if False:
            for i in range(10):
                print('nop')
        'No learning.'
        return {}

    @override(Policy)
    def compute_log_likelihoods(self, actions, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, **kwargs):
        if False:
            print('Hello World!')
        return np.array([random.random()] * len(obs_batch))

    @override(Policy)
    def get_weights(self) -> ModelWeights:
        if False:
            return 10
        'No weights to save.'
        return {}

    @override(Policy)
    def set_weights(self, weights: ModelWeights) -> None:
        if False:
            while True:
                i = 10
        'No weights to set.'
        pass

    @override(Policy)
    def _get_dummy_batch_from_view_requirements(self, batch_size: int=1):
        if False:
            print('Hello World!')
        return SampleBatch({SampleBatch.OBS: tree.map_structure(lambda s: s[None], self.observation_space.sample())})