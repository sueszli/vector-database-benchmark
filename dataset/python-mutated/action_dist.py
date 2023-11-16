import numpy as np
import gymnasium as gym
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.typing import TensorType, List, Union, ModelConfigDict

@DeveloperAPI
class ActionDistribution:
    """The policy action distribution of an agent.

    Attributes:
        inputs: input vector to compute samples from.
        model (ModelV2): reference to model producing the inputs.
    """

    @DeveloperAPI
    def __init__(self, inputs: List[TensorType], model: ModelV2):
        if False:
            while True:
                i = 10
        'Initializes an ActionDist object.\n\n        Args:\n            inputs: input vector to compute samples from.\n            model (ModelV2): reference to model producing the inputs. This\n                is mainly useful if you want to use model variables to compute\n                action outputs (i.e., for auto-regressive action distributions,\n                see examples/autoregressive_action_dist.py).\n        '
        self.inputs = inputs
        self.model = model

    @DeveloperAPI
    def sample(self) -> TensorType:
        if False:
            print('Hello World!')
        'Draw a sample from the action distribution.'
        raise NotImplementedError

    @DeveloperAPI
    def deterministic_sample(self) -> TensorType:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the deterministic "sampling" output from the distribution.\n        This is usually the max likelihood output, i.e. mean for Normal, argmax\n        for Categorical, etc..\n        '
        raise NotImplementedError

    @DeveloperAPI
    def sampled_action_logp(self) -> TensorType:
        if False:
            for i in range(10):
                print('nop')
        'Returns the log probability of the last sampled action.'
        raise NotImplementedError

    @DeveloperAPI
    def logp(self, x: TensorType) -> TensorType:
        if False:
            print('Hello World!')
        'The log-likelihood of the action distribution.'
        raise NotImplementedError

    @DeveloperAPI
    def kl(self, other: 'ActionDistribution') -> TensorType:
        if False:
            print('Hello World!')
        'The KL-divergence between two action distributions.'
        raise NotImplementedError

    @DeveloperAPI
    def entropy(self) -> TensorType:
        if False:
            i = 10
            return i + 15
        'The entropy of the action distribution.'
        raise NotImplementedError

    def multi_kl(self, other: 'ActionDistribution') -> TensorType:
        if False:
            while True:
                i = 10
        'The KL-divergence between two action distributions.\n\n        This differs from kl() in that it can return an array for\n        MultiDiscrete. TODO(ekl) consider removing this.\n        '
        return self.kl(other)

    def multi_entropy(self) -> TensorType:
        if False:
            return 10
        'The entropy of the action distribution.\n\n        This differs from entropy() in that it can return an array for\n        MultiDiscrete. TODO(ekl) consider removing this.\n        '
        return self.entropy()

    @staticmethod
    @DeveloperAPI
    def required_model_output_shape(action_space: gym.Space, model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        if False:
            return 10
        "Returns the required shape of an input parameter tensor for a\n        particular action space and an optional dict of distribution-specific\n        options.\n\n        Args:\n            action_space (gym.Space): The action space this distribution will\n                be used for, whose shape attributes will be used to determine\n                the required shape of the input parameter tensor.\n            model_config: Model's config dict (as defined in catalog.py)\n\n        Returns:\n            model_output_shape (int or np.ndarray of ints): size of the\n                required input vector (minus leading batch dimension).\n        "
        raise NotImplementedError