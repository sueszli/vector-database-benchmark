"""The main difference between this and the old ActionDistribution is that this one
has more explicit input args. So that the input format does not have to be guessed from
the code. This matches the design pattern of torch distribution which developers may
already be familiar with.
"""
import gymnasium as gym
import numpy as np
from typing import Optional, List, Mapping, Iterable, Dict
import tree
import abc
from ray.rllib.models.distributions import Distribution
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType, Union, Tuple
(torch, nn) = try_import_torch()

@DeveloperAPI
class TorchDistribution(Distribution, abc.ABC):
    """Wrapper class for torch.distributions."""

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__()
        self._dist = self._get_torch_distribution(*args, **kwargs)

    @abc.abstractmethod
    def _get_torch_distribution(self, *args, **kwargs) -> 'torch.distributions.Distribution':
        if False:
            return 10
        'Returns the torch.distributions.Distribution object to use.'

    @override(Distribution)
    def logp(self, value: TensorType, **kwargs) -> TensorType:
        if False:
            for i in range(10):
                print('nop')
        return self._dist.log_prob(value, **kwargs)

    @override(Distribution)
    def entropy(self) -> TensorType:
        if False:
            i = 10
            return i + 15
        return self._dist.entropy()

    @override(Distribution)
    def kl(self, other: 'Distribution') -> TensorType:
        if False:
            return 10
        return torch.distributions.kl.kl_divergence(self._dist, other._dist)

    @override(Distribution)
    def sample(self, *, sample_shape=torch.Size()) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        if False:
            i = 10
            return i + 15
        sample = self._dist.sample(sample_shape)
        return sample

    @override(Distribution)
    def rsample(self, *, sample_shape=torch.Size()) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        if False:
            i = 10
            return i + 15
        rsample = self._dist.rsample(sample_shape)
        return rsample

@DeveloperAPI
class TorchCategorical(TorchDistribution):
    """Wrapper class for PyTorch Categorical distribution.

    Creates a categorical distribution parameterized by either :attr:`probs` or
    :attr:`logits` (but not both).

    Samples are integers from :math:`\\{0, \\ldots, K-1\\}` where `K` is
    ``probs.size(-1)``.

    If `probs` is 1-dimensional with length-`K`, each element is the relative
    probability of sampling the class at that index.

    If `probs` is N-dimensional, the first N-1 dimensions are treated as a batch of
    relative probability vectors.

    .. testcode::
        :skipif: True

        m = TorchCategorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        m.sample(sample_shape=(2,))  # equal probability of 0, 1, 2, 3

    .. testoutput::

        tensor([3, 4])

    Args:
        logits: Event log probabilities (unnormalized)
        probs: The probablities of each event.
        temperature: In case of using logits, this parameter can be used to determine
            the sharpness of the distribution. i.e.
            ``probs = softmax(logits / temperature)``. The temperature must be strictly
            positive. A low value (e.g. 1e-10) will result in argmax sampling while a
            larger value will result in uniform sampling.
    """

    @override(TorchDistribution)
    def __init__(self, logits: torch.Tensor=None, probs: torch.Tensor=None) -> None:
        if False:
            i = 10
            return i + 15
        assert (probs is None) != (logits is None), 'Exactly one out of `probs` and `logits` must be set!'
        self.probs = probs
        self.logits = logits
        self.one_hot = torch.distributions.one_hot_categorical.OneHotCategorical(logits=logits, probs=probs)
        super().__init__(logits=logits, probs=probs)

    @override(TorchDistribution)
    def _get_torch_distribution(self, logits: torch.Tensor=None, probs: torch.Tensor=None) -> 'torch.distributions.Distribution':
        if False:
            i = 10
            return i + 15
        return torch.distributions.categorical.Categorical(logits=logits, probs=probs)

    @staticmethod
    @override(Distribution)
    def required_input_dim(space: gym.Space, **kwargs) -> int:
        if False:
            return 10
        assert isinstance(space, gym.spaces.Discrete)
        return int(space.n)

    @override(Distribution)
    def rsample(self, sample_shape=()):
        if False:
            print('Hello World!')
        one_hot_sample = self.one_hot.sample(sample_shape)
        return (one_hot_sample - self.probs).detach() + self.probs

    @classmethod
    @override(Distribution)
    def from_logits(cls, logits: TensorType, **kwargs) -> 'TorchCategorical':
        if False:
            print('Hello World!')
        return TorchCategorical(logits=logits, **kwargs)

    def to_deterministic(self) -> 'TorchDeterministic':
        if False:
            while True:
                i = 10
        if self.probs is not None:
            probs_or_logits = self.probs
        else:
            probs_or_logits = self.logits
        return TorchDeterministic(loc=torch.argmax(probs_or_logits, dim=-1))

@DeveloperAPI
class TorchDiagGaussian(TorchDistribution):
    """Wrapper class for PyTorch Normal distribution.

    Creates a normal distribution parameterized by :attr:`loc` and :attr:`scale`. In
    case of multi-dimensional distribution, the variance is assumed to be diagonal.

    .. testcode::
        :skipif: True

        loc, scale = torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])
        m = TorchDiagGaussian(loc=loc, scale=scale)
        m.sample(sample_shape=(2,))  # 2d normal dist with loc=0 and scale=1

    .. testoutput::

        tensor([[ 0.1046, -0.6120], [ 0.234, 0.556]])

    .. testcode::
        :skipif: True

        # scale is None
        m = TorchDiagGaussian(loc=torch.tensor([0.0, 1.0]))
        m.sample(sample_shape=(2,))  # normally distributed with loc=0 and scale=1

    .. testoutput::

        tensor([0.1046, 0.6120])


    Args:
        loc: mean of the distribution (often referred to as mu). If scale is None, the
            second half of the `loc` will be used as the log of scale.
        scale: standard deviation of the distribution (often referred to as sigma).
            Has to be positive.
    """

    @override(TorchDistribution)
    def __init__(self, loc: Union[float, torch.Tensor], scale: Optional[Union[float, torch.Tensor]]):
        if False:
            for i in range(10):
                print('nop')
        self.loc = loc
        super().__init__(loc=loc, scale=scale)

    def _get_torch_distribution(self, loc, scale) -> 'torch.distributions.Distribution':
        if False:
            while True:
                i = 10
        return torch.distributions.normal.Normal(loc, scale)

    @override(TorchDistribution)
    def logp(self, value: TensorType) -> TensorType:
        if False:
            print('Hello World!')
        return super().logp(value).sum(-1)

    @override(TorchDistribution)
    def entropy(self) -> TensorType:
        if False:
            print('Hello World!')
        return super().entropy().sum(-1)

    @override(TorchDistribution)
    def kl(self, other: 'TorchDistribution') -> TensorType:
        if False:
            while True:
                i = 10
        return super().kl(other).sum(-1)

    @staticmethod
    @override(Distribution)
    def required_input_dim(space: gym.Space, **kwargs) -> int:
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(space, gym.spaces.Box)
        return int(np.prod(space.shape, dtype=np.int32) * 2)

    @classmethod
    @override(Distribution)
    def from_logits(cls, logits: TensorType, **kwargs) -> 'TorchDiagGaussian':
        if False:
            i = 10
            return i + 15
        (loc, log_std) = logits.chunk(2, dim=-1)
        scale = log_std.exp()
        return TorchDiagGaussian(loc=loc, scale=scale)

    def to_deterministic(self) -> 'TorchDeterministic':
        if False:
            while True:
                i = 10
        return TorchDeterministic(loc=self.loc)

@DeveloperAPI
class TorchDeterministic(Distribution):
    """The distribution that returns the input values directly.

    This is similar to DiagGaussian with standard deviation zero (thus only
    requiring the "mean" values as NN output).

    Note: entropy is always zero, ang logp and kl are not implemented.

    .. testcode::
        :skipif: True

        m = TorchDeterministic(loc=torch.tensor([0.0, 0.0]))
        m.sample(sample_shape=(2,))

    .. testoutput::

        tensor([[ 0.0, 0.0], [ 0.0, 0.0]])

    Args:
        loc: the determinsitic value to return
    """

    @override(Distribution)
    def __init__(self, loc: torch.Tensor) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.loc = loc

    @override(Distribution)
    def sample(self, *, sample_shape: Tuple[int, ...]=torch.Size(), **kwargs) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        if False:
            print('Hello World!')
        device = self.loc.device
        dtype = self.loc.dtype
        shape = sample_shape + self.loc.shape
        return torch.ones(shape, device=device, dtype=dtype) * self.loc

    def rsample(self, *, sample_shape: Tuple[int, ...]=None, **kwargs) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @override(Distribution)
    def logp(self, value: TensorType, **kwargs) -> TensorType:
        if False:
            return 10
        raise ValueError(f'Cannot return logp for {self.__class__.__name__}.')

    @override(Distribution)
    def entropy(self, **kwargs) -> TensorType:
        if False:
            return 10
        raise torch.zeros_like(self.loc)

    @override(Distribution)
    def kl(self, other: 'Distribution', **kwargs) -> TensorType:
        if False:
            i = 10
            return i + 15
        raise ValueError(f'Cannot return kl for {self.__class__.__name__}.')

    @staticmethod
    @override(Distribution)
    def required_input_dim(space: gym.Space, **kwargs) -> int:
        if False:
            print('Hello World!')
        assert isinstance(space, gym.spaces.Box)
        return int(np.prod(space.shape, dtype=np.int32))

    @classmethod
    @override(Distribution)
    def from_logits(cls, logits: TensorType, **kwargs) -> 'TorchDeterministic':
        if False:
            i = 10
            return i + 15
        return TorchDeterministic(loc=logits)

    def to_deterministic(self) -> 'TorchDeterministic':
        if False:
            i = 10
            return i + 15
        return self

@DeveloperAPI
class TorchMultiCategorical(Distribution):
    """MultiCategorical distribution for MultiDiscrete action spaces."""

    @override(Distribution)
    def __init__(self, categoricals: List[TorchCategorical]):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._cats = categoricals

    @override(Distribution)
    def sample(self) -> TensorType:
        if False:
            while True:
                i = 10
        arr = [cat.sample() for cat in self._cats]
        sample_ = torch.stack(arr, dim=-1)
        return sample_

    @override(Distribution)
    def rsample(self, sample_shape=()):
        if False:
            for i in range(10):
                print('nop')
        arr = [cat.rsample() for cat in self._cats]
        sample_ = torch.stack(arr, dim=-1)
        return sample_

    @override(Distribution)
    def logp(self, value: torch.Tensor) -> TensorType:
        if False:
            i = 10
            return i + 15
        value = torch.unbind(value, dim=-1)
        logps = torch.stack([cat.logp(act) for (cat, act) in zip(self._cats, value)])
        return torch.sum(logps, dim=0)

    @override(Distribution)
    def entropy(self) -> TensorType:
        if False:
            return 10
        return torch.sum(torch.stack([cat.entropy() for cat in self._cats], dim=-1), dim=-1)

    @override(Distribution)
    def kl(self, other: Distribution) -> TensorType:
        if False:
            for i in range(10):
                print('nop')
        kls = torch.stack([cat.kl(oth_cat) for (cat, oth_cat) in zip(self._cats, other._cats)], dim=-1)
        return torch.sum(kls, dim=-1)

    @staticmethod
    @override(Distribution)
    def required_input_dim(space: gym.Space, **kwargs) -> int:
        if False:
            print('Hello World!')
        assert isinstance(space, gym.spaces.MultiDiscrete)
        return int(np.sum(space.nvec))

    @classmethod
    @override(Distribution)
    def from_logits(cls, logits: torch.Tensor, input_lens: List[int], temperatures: List[float]=None, **kwargs) -> 'TorchMultiCategorical':
        if False:
            for i in range(10):
                print('nop')
        'Creates this Distribution from logits (and additional arguments).\n\n        If you wish to create this distribution from logits only, please refer to\n        `Distribution.get_partial_dist_cls()`.\n\n        Args:\n            logits: The tensor containing logits to be separated by logit_lens.\n                child_distribution_cls_struct: A struct of Distribution classes that can\n                be instantiated from the given logits.\n            input_lens: A list of integers that indicate the length of the logits\n                vectors to be passed into each child distribution.\n            temperatures: A list of floats representing the temperature to use for\n                each Categorical distribution. If not provided, 1.0 is used for all.\n            **kwargs: Forward compatibility kwargs.\n        '
        if not temperatures:
            temperatures = [1.0] * len(input_lens)
        assert sum(input_lens) == logits.shape[-1], 'input_lens must sum to logits.shape[-1]'
        assert len(input_lens) == len(temperatures), 'input_lens and temperatures must be same length'
        categoricals = [TorchCategorical(logits=logits) for logits in torch.split(logits, input_lens, dim=-1)]
        return TorchMultiCategorical(categoricals=categoricals)

    def to_deterministic(self) -> 'TorchMultiDistribution':
        if False:
            print('Hello World!')
        return TorchMultiDistribution([cat.to_deterministic() for cat in self._cats])

@DeveloperAPI
class TorchMultiDistribution(Distribution):
    """Action distribution that operates on multiple, possibly nested actions."""

    def __init__(self, child_distribution_struct: Union[Tuple, List, Dict]):
        if False:
            print('Hello World!')
        'Initializes a TorchMultiActionDistribution object.\n\n        Args:\n            child_distribution_struct: Any struct\n                that contains the child distribution classes to use to\n                instantiate the child distributions from `logits`.\n        '
        super().__init__()
        self._original_struct = child_distribution_struct
        self._flat_child_distributions = tree.flatten(child_distribution_struct)

    @override(Distribution)
    def rsample(self, *, sample_shape: Tuple[int, ...]=None, **kwargs) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        if False:
            print('Hello World!')
        rsamples = []
        for dist in self._flat_child_distributions:
            rsample = dist.rsample(sample_shape=sample_shape, **kwargs)
            rsamples.append(rsample)
        rsamples = tree.unflatten_as(self._original_struct, rsamples)
        return rsamples

    @override(Distribution)
    def logp(self, value: TensorType) -> TensorType:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(value, torch.Tensor):
            split_indices = []
            for dist in self._flat_child_distributions:
                if isinstance(dist, TorchCategorical):
                    split_indices.append(1)
                elif isinstance(dist, TorchMultiCategorical):
                    split_indices.append(len(dist._cats))
                else:
                    sample = dist.sample()
                    if len(sample.shape) == 1:
                        split_indices.append(1)
                    else:
                        split_indices.append(sample.size()[1])
            split_value = list(torch.split(value, split_indices, dim=1))
        else:
            split_value = tree.flatten(value)

        def map_(val, dist):
            if False:
                return 10
            if isinstance(dist, TorchCategorical) and val.shape[-1] == 1 and (len(val.shape) > 1):
                val = torch.squeeze(val, dim=-1)
            return dist.logp(val)
        flat_logps = tree.map_structure(map_, split_value, self._flat_child_distributions)
        return sum(flat_logps)

    @override(Distribution)
    def kl(self, other: Distribution) -> TensorType:
        if False:
            print('Hello World!')
        kl_list = [d.kl(o) for (d, o) in zip(self._flat_child_distributions, other._flat_child_distributions)]
        return sum(kl_list)

    @override(Distribution)
    def entropy(self):
        if False:
            while True:
                i = 10
        entropy_list = [d.entropy() for d in self._flat_child_distributions]
        return sum(entropy_list)

    @override(Distribution)
    def sample(self):
        if False:
            print('Hello World!')
        child_distributions_struct = tree.unflatten_as(self._original_struct, self._flat_child_distributions)
        return tree.map_structure(lambda s: s.sample(), child_distributions_struct)

    @staticmethod
    @override(Distribution)
    def required_input_dim(space: gym.Space, input_lens: List[int], **kwargs) -> int:
        if False:
            i = 10
            return i + 15
        return sum(input_lens)

    @classmethod
    @override(Distribution)
    def from_logits(cls, logits: torch.Tensor, child_distribution_cls_struct: Union[Mapping, Iterable], input_lens: Union[Dict, List[int]], space: gym.Space, **kwargs) -> 'TorchMultiDistribution':
        if False:
            return 10
        'Creates this Distribution from logits (and additional arguments).\n\n        If you wish to create this distribution from logits only, please refer to\n        `Distribution.get_partial_dist_cls()`.\n\n        Args:\n            logits: The tensor containing logits to be separated by `input_lens`.\n                child_distribution_cls_struct: A struct of Distribution classes that can\n                be instantiated from the given logits.\n            child_distribution_cls_struct: A struct of Distribution classes that can\n                be instantiated from the given logits.\n            input_lens: A list or dict of integers that indicate the length of each\n                logit. If this is given as a dict, the structure should match the\n                structure of child_distribution_cls_struct.\n            space: The possibly nested output space.\n            **kwargs: Forward compatibility kwargs.\n\n        Returns:\n            A TorchMultiActionDistribution object.\n        '
        logit_lens = tree.flatten(input_lens)
        child_distribution_cls_list = tree.flatten(child_distribution_cls_struct)
        split_logits = torch.split(logits, logit_lens, dim=1)
        child_distribution_list = tree.map_structure(lambda dist, input_: dist.from_logits(input_), child_distribution_cls_list, list(split_logits))
        child_distribution_struct = tree.unflatten_as(child_distribution_cls_struct, child_distribution_list)
        return TorchMultiDistribution(child_distribution_struct=child_distribution_struct)

    def to_deterministic(self) -> 'TorchMultiDistribution':
        if False:
            i = 10
            return i + 15
        flat_deterministic_dists = [dist.to_deterministic() for dist in self._flat_child_distributions]
        deterministic_dists = tree.unflatten_as(self._original_struct, flat_deterministic_dists)
        return TorchMultiDistribution(deterministic_dists)