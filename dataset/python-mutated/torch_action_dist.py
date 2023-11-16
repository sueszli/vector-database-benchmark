import functools
import gymnasium as gym
from math import log
import numpy as np
import tree
from typing import Optional
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override, DeveloperAPI, ExperimentalAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import SMALL_NUMBER, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import TensorType, List, Union, Tuple, ModelConfigDict
(torch, nn) = try_import_torch()

@DeveloperAPI
class TorchDistributionWrapper(ActionDistribution):
    """Wrapper class for torch.distributions."""

    @override(ActionDistribution)
    def __init__(self, inputs: List[TensorType], model: TorchModelV2):
        if False:
            while True:
                i = 10
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(inputs)
            if isinstance(model, TorchModelV2):
                inputs = inputs.to(next(model.parameters()).device)
        super().__init__(inputs, model)
        self.last_sample = None

    @override(ActionDistribution)
    def logp(self, actions: TensorType) -> TensorType:
        if False:
            i = 10
            return i + 15
        return self.dist.log_prob(actions)

    @override(ActionDistribution)
    def entropy(self) -> TensorType:
        if False:
            print('Hello World!')
        return self.dist.entropy()

    @override(ActionDistribution)
    def kl(self, other: ActionDistribution) -> TensorType:
        if False:
            while True:
                i = 10
        return torch.distributions.kl.kl_divergence(self.dist, other.dist)

    @override(ActionDistribution)
    def sample(self) -> TensorType:
        if False:
            for i in range(10):
                print('nop')
        self.last_sample = self.dist.sample()
        return self.last_sample

    @override(ActionDistribution)
    def sampled_action_logp(self) -> TensorType:
        if False:
            print('Hello World!')
        assert self.last_sample is not None
        return self.logp(self.last_sample)

@DeveloperAPI
class TorchCategorical(TorchDistributionWrapper):
    """Wrapper class for PyTorch Categorical distribution."""

    @override(ActionDistribution)
    def __init__(self, inputs: List[TensorType], model: TorchModelV2=None, temperature: float=1.0):
        if False:
            print('Hello World!')
        if temperature != 1.0:
            assert temperature > 0.0, 'Categorical `temperature` must be > 0.0!'
            inputs /= temperature
        super().__init__(inputs, model)
        self.dist = torch.distributions.categorical.Categorical(logits=self.inputs)

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        if False:
            for i in range(10):
                print('nop')
        self.last_sample = self.dist.probs.argmax(dim=1)
        return self.last_sample

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space: gym.Space, model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        if False:
            return 10
        return action_space.n

@DeveloperAPI
def get_torch_categorical_class_with_temperature(t: float):
    if False:
        while True:
            i = 10
    'TorchCategorical distribution class that has customized default temperature.'

    class TorchCategoricalWithTemperature(TorchCategorical):

        def __init__(self, inputs, model=None, temperature=t):
            if False:
                while True:
                    i = 10
            super().__init__(inputs, model, temperature)
    return TorchCategoricalWithTemperature

@DeveloperAPI
class TorchMultiCategorical(TorchDistributionWrapper):
    """MultiCategorical distribution for MultiDiscrete action spaces."""

    @override(TorchDistributionWrapper)
    def __init__(self, inputs: List[TensorType], model: TorchModelV2, input_lens: Union[List[int], np.ndarray, Tuple[int, ...]], action_space=None):
        if False:
            return 10
        super().__init__(inputs, model)
        inputs_split = self.inputs.split(tuple(input_lens), dim=1)
        self.cats = [torch.distributions.categorical.Categorical(logits=input_) for input_ in inputs_split]
        self.action_space = action_space

    @override(TorchDistributionWrapper)
    def sample(self) -> TensorType:
        if False:
            print('Hello World!')
        arr = [cat.sample() for cat in self.cats]
        sample_ = torch.stack(arr, dim=1)
        if isinstance(self.action_space, gym.spaces.Box):
            sample_ = torch.reshape(sample_, [-1] + list(self.action_space.shape))
        self.last_sample = sample_
        return sample_

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        if False:
            for i in range(10):
                print('nop')
        arr = [torch.argmax(cat.probs, -1) for cat in self.cats]
        sample_ = torch.stack(arr, dim=1)
        if isinstance(self.action_space, gym.spaces.Box):
            sample_ = torch.reshape(sample_, [-1] + list(self.action_space.shape))
        self.last_sample = sample_
        return sample_

    @override(TorchDistributionWrapper)
    def logp(self, actions: TensorType) -> TensorType:
        if False:
            print('Hello World!')
        if isinstance(actions, torch.Tensor):
            if isinstance(self.action_space, gym.spaces.Box):
                actions = torch.reshape(actions, [-1, int(np.prod(self.action_space.shape))])
            actions = torch.unbind(actions, dim=1)
        logps = torch.stack([cat.log_prob(act) for (cat, act) in zip(self.cats, actions)])
        return torch.sum(logps, dim=0)

    @override(ActionDistribution)
    def multi_entropy(self) -> TensorType:
        if False:
            print('Hello World!')
        return torch.stack([cat.entropy() for cat in self.cats], dim=1)

    @override(TorchDistributionWrapper)
    def entropy(self) -> TensorType:
        if False:
            print('Hello World!')
        return torch.sum(self.multi_entropy(), dim=1)

    @override(ActionDistribution)
    def multi_kl(self, other: ActionDistribution) -> TensorType:
        if False:
            print('Hello World!')
        return torch.stack([torch.distributions.kl.kl_divergence(cat, oth_cat) for (cat, oth_cat) in zip(self.cats, other.cats)], dim=1)

    @override(TorchDistributionWrapper)
    def kl(self, other: ActionDistribution) -> TensorType:
        if False:
            print('Hello World!')
        return torch.sum(self.multi_kl(other), dim=1)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space: gym.Space, model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        if False:
            while True:
                i = 10
        if isinstance(action_space, gym.spaces.Box):
            assert action_space.dtype.name.startswith('int')
            low_ = np.min(action_space.low)
            high_ = np.max(action_space.high)
            assert np.all(action_space.low == low_)
            assert np.all(action_space.high == high_)
            return np.prod(action_space.shape, dtype=np.int32) * (high_ - low_ + 1)
        else:
            return np.sum(action_space.nvec)

@ExperimentalAPI
class TorchSlateMultiCategorical(TorchCategorical):
    """MultiCategorical distribution for MultiDiscrete action spaces.

    The action space must be uniform, meaning all nvec items have the same size, e.g.
    MultiDiscrete([10, 10, 10]), where 10 is the number of candidates to pick from
    and 3 is the slate size (pick 3 out of 10). When picking candidates, no candidate
    must be picked more than once.
    """

    def __init__(self, inputs: List[TensorType], model: TorchModelV2=None, temperature: float=1.0, action_space: Optional[gym.spaces.MultiDiscrete]=None, all_slates=None):
        if False:
            print('Hello World!')
        assert temperature > 0.0, 'Categorical `temperature` must be > 0.0!'
        super().__init__(inputs / temperature, model)
        self.action_space = action_space
        assert isinstance(self.action_space, gym.spaces.MultiDiscrete) and all((n == self.action_space.nvec[0] for n in self.action_space.nvec))
        self.all_slates = all_slates

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        if False:
            i = 10
            return i + 15
        sample = super().deterministic_sample()
        return torch.take_along_dim(self.all_slates, sample.long(), dim=-1)

    @override(ActionDistribution)
    def logp(self, x: TensorType) -> TensorType:
        if False:
            return 10
        return torch.ones_like(self.inputs[:, 0])

@DeveloperAPI
class TorchDiagGaussian(TorchDistributionWrapper):
    """Wrapper class for PyTorch Normal distribution."""

    @override(ActionDistribution)
    def __init__(self, inputs: List[TensorType], model: TorchModelV2, *, action_space: Optional[gym.spaces.Space]=None):
        if False:
            print('Hello World!')
        super().__init__(inputs, model)
        (mean, log_std) = torch.chunk(self.inputs, 2, dim=1)
        self.log_std = log_std
        self.dist = torch.distributions.normal.Normal(mean, torch.exp(log_std))
        self.zero_action_dim = action_space and action_space.shape == ()

    @override(TorchDistributionWrapper)
    def sample(self) -> TensorType:
        if False:
            print('Hello World!')
        sample = super().sample()
        if self.zero_action_dim:
            return torch.squeeze(sample, dim=-1)
        return sample

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        if False:
            print('Hello World!')
        self.last_sample = self.dist.mean
        return self.last_sample

    @override(TorchDistributionWrapper)
    def logp(self, actions: TensorType) -> TensorType:
        if False:
            i = 10
            return i + 15
        return super().logp(actions).sum(-1)

    @override(TorchDistributionWrapper)
    def entropy(self) -> TensorType:
        if False:
            i = 10
            return i + 15
        return super().entropy().sum(-1)

    @override(TorchDistributionWrapper)
    def kl(self, other: ActionDistribution) -> TensorType:
        if False:
            return 10
        return super().kl(other).sum(-1)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space: gym.Space, model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        if False:
            while True:
                i = 10
        return np.prod(action_space.shape, dtype=np.int32) * 2

@DeveloperAPI
class TorchSquashedGaussian(TorchDistributionWrapper):
    """A tanh-squashed Gaussian distribution defined by: mean, std, low, high.

    The distribution will never return low or high exactly, but
    `low`+SMALL_NUMBER or `high`-SMALL_NUMBER respectively.
    """

    def __init__(self, inputs: List[TensorType], model: TorchModelV2, low: float=-1.0, high: float=1.0):
        if False:
            return 10
        'Parameterizes the distribution via `inputs`.\n\n        Args:\n            low: The lowest possible sampling value\n                (excluding this value).\n            high: The highest possible sampling value\n                (excluding this value).\n        '
        super().__init__(inputs, model)
        (mean, log_std) = torch.chunk(self.inputs, 2, dim=-1)
        log_std = torch.clamp(log_std, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT)
        std = torch.exp(log_std)
        self.dist = torch.distributions.normal.Normal(mean, std)
        assert np.all(np.less(low, high))
        self.low = low
        self.high = high
        self.mean = mean
        self.std = std

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        if False:
            print('Hello World!')
        self.last_sample = self._squash(self.dist.mean)
        return self.last_sample

    @override(TorchDistributionWrapper)
    def sample(self) -> TensorType:
        if False:
            return 10
        normal_sample = self.dist.rsample()
        self.last_sample = self._squash(normal_sample)
        return self.last_sample

    @override(ActionDistribution)
    def logp(self, x: TensorType) -> TensorType:
        if False:
            i = 10
            return i + 15
        unsquashed_values = self._unsquash(x)
        log_prob_gaussian = self.dist.log_prob(unsquashed_values)
        log_prob_gaussian = torch.clamp(log_prob_gaussian, -100, 100)
        log_prob_gaussian = torch.sum(log_prob_gaussian, dim=-1)
        unsquashed_values_tanhd = torch.tanh(unsquashed_values)
        log_prob = log_prob_gaussian - torch.sum(torch.log(1 - unsquashed_values_tanhd ** 2 + SMALL_NUMBER), dim=-1)
        return log_prob

    def sample_logp(self):
        if False:
            return 10
        z = self.dist.rsample()
        actions = self._squash(z)
        return (actions, torch.sum(self.dist.log_prob(z) - torch.log(1 - actions * actions + SMALL_NUMBER), dim=-1))

    @override(TorchDistributionWrapper)
    def entropy(self) -> TensorType:
        if False:
            for i in range(10):
                print('nop')
        raise ValueError('Entropy not defined for SquashedGaussian!')

    @override(TorchDistributionWrapper)
    def kl(self, other: ActionDistribution) -> TensorType:
        if False:
            return 10
        raise ValueError('KL not defined for SquashedGaussian!')

    def _squash(self, raw_values: TensorType) -> TensorType:
        if False:
            print('Hello World!')
        squashed = (torch.tanh(raw_values) + 1.0) / 2.0 * (self.high - self.low) + self.low
        return torch.clamp(squashed, self.low, self.high)

    def _unsquash(self, values: TensorType) -> TensorType:
        if False:
            for i in range(10):
                print('nop')
        normed_values = (values - self.low) / (self.high - self.low) * 2.0 - 1.0
        save_normed_values = torch.clamp(normed_values, -1.0 + SMALL_NUMBER, 1.0 - SMALL_NUMBER)
        unsquashed = torch.atanh(save_normed_values)
        return unsquashed

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space: gym.Space, model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        if False:
            i = 10
            return i + 15
        return np.prod(action_space.shape, dtype=np.int32) * 2

@DeveloperAPI
class TorchBeta(TorchDistributionWrapper):
    """
    A Beta distribution is defined on the interval [0, 1] and parameterized by
    shape parameters alpha and beta (also called concentration parameters).

    PDF(x; alpha, beta) = x**(alpha - 1) (1 - x)**(beta - 1) / Z
        with Z = Gamma(alpha) Gamma(beta) / Gamma(alpha + beta)
        and Gamma(n) = (n - 1)!
    """

    def __init__(self, inputs: List[TensorType], model: TorchModelV2, low: float=0.0, high: float=1.0):
        if False:
            return 10
        super().__init__(inputs, model)
        self.inputs = torch.clamp(self.inputs, log(SMALL_NUMBER), -log(SMALL_NUMBER))
        self.inputs = torch.log(torch.exp(self.inputs) + 1.0) + 1.0
        self.low = low
        self.high = high
        (alpha, beta) = torch.chunk(self.inputs, 2, dim=-1)
        self.dist = torch.distributions.Beta(concentration1=alpha, concentration0=beta)

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        if False:
            i = 10
            return i + 15
        self.last_sample = self._squash(self.dist.mean)
        return self.last_sample

    @override(TorchDistributionWrapper)
    def sample(self) -> TensorType:
        if False:
            return 10
        normal_sample = self.dist.rsample()
        self.last_sample = self._squash(normal_sample)
        return self.last_sample

    @override(ActionDistribution)
    def logp(self, x: TensorType) -> TensorType:
        if False:
            for i in range(10):
                print('nop')
        unsquashed_values = self._unsquash(x)
        return torch.sum(self.dist.log_prob(unsquashed_values), dim=-1)

    def _squash(self, raw_values: TensorType) -> TensorType:
        if False:
            print('Hello World!')
        return raw_values * (self.high - self.low) + self.low

    def _unsquash(self, values: TensorType) -> TensorType:
        if False:
            print('Hello World!')
        return (values - self.low) / (self.high - self.low)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space: gym.Space, model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        if False:
            for i in range(10):
                print('nop')
        return np.prod(action_space.shape, dtype=np.int32) * 2

@DeveloperAPI
class TorchDeterministic(TorchDistributionWrapper):
    """Action distribution that returns the input values directly.

    This is similar to DiagGaussian with standard deviation zero (thus only
    requiring the "mean" values as NN output).
    """

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        if False:
            for i in range(10):
                print('nop')
        return self.inputs

    @override(TorchDistributionWrapper)
    def sampled_action_logp(self) -> TensorType:
        if False:
            for i in range(10):
                print('nop')
        return torch.zeros((self.inputs.size()[0],), dtype=torch.float32)

    @override(TorchDistributionWrapper)
    def sample(self) -> TensorType:
        if False:
            i = 10
            return i + 15
        return self.deterministic_sample()

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space: gym.Space, model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        if False:
            for i in range(10):
                print('nop')
        return np.prod(action_space.shape, dtype=np.int32)

@DeveloperAPI
class TorchMultiActionDistribution(TorchDistributionWrapper):
    """Action distribution that operates on multiple, possibly nested actions."""

    def __init__(self, inputs, model, *, child_distributions, input_lens, action_space):
        if False:
            i = 10
            return i + 15
        'Initializes a TorchMultiActionDistribution object.\n\n        Args:\n            inputs (torch.Tensor): A single tensor of shape [BATCH, size].\n            model (TorchModelV2): The TorchModelV2 object used to produce\n                inputs for this distribution.\n            child_distributions (any[torch.Tensor]): Any struct\n                that contains the child distribution classes to use to\n                instantiate the child distributions from `inputs`. This could\n                be an already flattened list or a struct according to\n                `action_space`.\n            input_lens (any[int]): A flat list or a nested struct of input\n                split lengths used to split `inputs`.\n            action_space (Union[gym.spaces.Dict,gym.spaces.Tuple]): The complex\n                and possibly nested action space.\n        '
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(inputs)
            if isinstance(model, TorchModelV2):
                inputs = inputs.to(next(model.parameters()).device)
        super().__init__(inputs, model)
        self.action_space_struct = get_base_struct_from_space(action_space)
        self.input_lens = tree.flatten(input_lens)
        flat_child_distributions = tree.flatten(child_distributions)
        split_inputs = torch.split(inputs, self.input_lens, dim=1)
        self.flat_child_distributions = tree.map_structure(lambda dist, input_: dist(input_, model), flat_child_distributions, list(split_inputs))

    @override(ActionDistribution)
    def logp(self, x):
        if False:
            while True:
                i = 10
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        if isinstance(x, torch.Tensor):
            split_indices = []
            for dist in self.flat_child_distributions:
                if isinstance(dist, TorchCategorical):
                    split_indices.append(1)
                elif isinstance(dist, TorchMultiCategorical) and dist.action_space is not None:
                    split_indices.append(int(np.prod(dist.action_space.shape)))
                else:
                    sample = dist.sample()
                    if len(sample.shape) == 1:
                        split_indices.append(1)
                    else:
                        split_indices.append(sample.size()[1])
            split_x = list(torch.split(x, split_indices, dim=1))
        else:
            split_x = tree.flatten(x)

        def map_(val, dist):
            if False:
                i = 10
                return i + 15
            if isinstance(dist, TorchCategorical):
                val = (torch.squeeze(val, dim=-1) if len(val.shape) > 1 else val).int()
            return dist.logp(val)
        flat_logps = tree.map_structure(map_, split_x, self.flat_child_distributions)
        return functools.reduce(lambda a, b: a + b, flat_logps)

    @override(ActionDistribution)
    def kl(self, other):
        if False:
            for i in range(10):
                print('nop')
        kl_list = [d.kl(o) for (d, o) in zip(self.flat_child_distributions, other.flat_child_distributions)]
        return functools.reduce(lambda a, b: a + b, kl_list)

    @override(ActionDistribution)
    def entropy(self):
        if False:
            while True:
                i = 10
        entropy_list = [d.entropy() for d in self.flat_child_distributions]
        return functools.reduce(lambda a, b: a + b, entropy_list)

    @override(ActionDistribution)
    def sample(self):
        if False:
            while True:
                i = 10
        child_distributions = tree.unflatten_as(self.action_space_struct, self.flat_child_distributions)
        return tree.map_structure(lambda s: s.sample(), child_distributions)

    @override(ActionDistribution)
    def deterministic_sample(self):
        if False:
            return 10
        child_distributions = tree.unflatten_as(self.action_space_struct, self.flat_child_distributions)
        return tree.map_structure(lambda s: s.deterministic_sample(), child_distributions)

    @override(TorchDistributionWrapper)
    def sampled_action_logp(self):
        if False:
            while True:
                i = 10
        p = self.flat_child_distributions[0].sampled_action_logp()
        for c in self.flat_child_distributions[1:]:
            p += c.sampled_action_logp()
        return p

    @override(ActionDistribution)
    def required_model_output_shape(self, action_space, model_config):
        if False:
            while True:
                i = 10
        return np.sum(self.input_lens, dtype=np.int32)

@DeveloperAPI
class TorchDirichlet(TorchDistributionWrapper):
    """Dirichlet distribution for continuous actions that are between
    [0,1] and sum to 1.

    e.g. actions that represent resource allocation."""

    def __init__(self, inputs, model):
        if False:
            return 10
        'Input is a tensor of logits. The exponential of logits is used to\n        parametrize the Dirichlet distribution as all parameters need to be\n        positive. An arbitrary small epsilon is added to the concentration\n        parameters to be zero due to numerical error.\n\n        See issue #4440 for more details.\n        '
        self.epsilon = torch.tensor(1e-07).to(inputs.device)
        concentration = torch.exp(inputs) + self.epsilon
        self.dist = torch.distributions.dirichlet.Dirichlet(concentration=concentration, validate_args=True)
        super().__init__(concentration, model)

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        if False:
            return 10
        self.last_sample = nn.functional.softmax(self.dist.concentration, dim=-1)
        return self.last_sample

    @override(ActionDistribution)
    def logp(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = torch.max(x, self.epsilon)
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return self.dist.log_prob(x)

    @override(ActionDistribution)
    def entropy(self):
        if False:
            return 10
        return self.dist.entropy()

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        if False:
            while True:
                i = 10
        return np.prod(action_space.shape, dtype=np.int32)