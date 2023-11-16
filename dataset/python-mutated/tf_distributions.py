"""The main difference between this and the old ActionDistribution is that this one
has more explicit input args. So that the input format does not have to be guessed from
the code. This matches the design pattern of torch distribution which developers may
already be familiar with.
"""
import gymnasium as gym
import tree
import numpy as np
from typing import Optional, List, Mapping, Iterable, Dict
import abc
from ray.rllib.models.distributions import Distribution
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
from ray.rllib.utils.typing import TensorType, Union, Tuple
(_, tf, _) = try_import_tf()
tfp = try_import_tfp()

@DeveloperAPI
class TfDistribution(Distribution, abc.ABC):
    """Wrapper class for tfp.distributions."""

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._dist = self._get_tf_distribution(*args, **kwargs)

    @abc.abstractmethod
    def _get_tf_distribution(self, *args, **kwargs) -> 'tfp.distributions.Distribution':
        if False:
            print('Hello World!')
        'Returns the tfp.distributions.Distribution object to use.'

    @override(Distribution)
    def logp(self, value: TensorType, **kwargs) -> TensorType:
        if False:
            print('Hello World!')
        return self._dist.log_prob(value, **kwargs)

    @override(Distribution)
    def entropy(self) -> TensorType:
        if False:
            return 10
        return self._dist.entropy()

    @override(Distribution)
    def kl(self, other: 'Distribution') -> TensorType:
        if False:
            while True:
                i = 10
        return self._dist.kl_divergence(other._dist)

    @override(Distribution)
    def sample(self, *, sample_shape=()) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        if False:
            print('Hello World!')
        sample = self._dist.sample(sample_shape)
        return sample

    @override(Distribution)
    def rsample(self, *, sample_shape=()) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

@DeveloperAPI
class TfCategorical(TfDistribution):
    """Wrapper class for Categorical distribution.

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

        m = TfCategorical([ 0.25, 0.25, 0.25, 0.25 ])
        m.sample(sample_shape=(2,))  # equal probability of 0, 1, 2, 3

    .. testoutput::

        tf.Tensor([2 3], shape=(2,), dtype=int32)

    Args:
        probs: The probablities of each event.
        logits: Event log probabilities (unnormalized)
        temperature: In case of using logits, this parameter can be used to determine
            the sharpness of the distribution. i.e.
            ``probs = softmax(logits / temperature)``. The temperature must be strictly
            positive. A low value (e.g. 1e-10) will result in argmax sampling while a
            larger value will result in uniform sampling.
    """

    @override(TfDistribution)
    def __init__(self, probs: 'tf.Tensor'=None, logits: 'tf.Tensor'=None) -> None:
        if False:
            print('Hello World!')
        assert (probs is None) != (logits is None), 'Exactly one out of `probs` and `logits` must be set!'
        self.probs = probs
        self.logits = logits
        self.one_hot = tfp.distributions.OneHotCategorical(logits=logits, probs=probs)
        super().__init__(logits=logits, probs=probs)

    @override(Distribution)
    def logp(self, value: TensorType, **kwargs) -> TensorType:
        if False:
            return 10
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits if self.logits is not None else tf.log(self.probs), labels=tf.cast(value, tf.int32))

    @override(TfDistribution)
    def _get_tf_distribution(self, probs: 'tf.Tensor'=None, logits: 'tf.Tensor'=None) -> 'tfp.distributions.Distribution':
        if False:
            while True:
                i = 10
        return tfp.distributions.Categorical(probs=probs, logits=logits)

    @staticmethod
    @override(Distribution)
    def required_input_dim(space: gym.Space, **kwargs) -> int:
        if False:
            while True:
                i = 10
        assert isinstance(space, gym.spaces.Discrete)
        return int(space.n)

    @override(Distribution)
    def rsample(self, sample_shape=()):
        if False:
            while True:
                i = 10
        one_hot_sample = self.one_hot.sample(sample_shape)
        return tf.stop_gradients(one_hot_sample - self.probs) + self.probs

    @classmethod
    @override(Distribution)
    def from_logits(cls, logits: TensorType, **kwargs) -> 'TfCategorical':
        if False:
            for i in range(10):
                print('nop')
        return TfCategorical(logits=logits, **kwargs)

    def to_deterministic(self) -> 'TfDeterministic':
        if False:
            i = 10
            return i + 15
        if self.probs is not None:
            probs_or_logits = self.probs
        else:
            probs_or_logits = self.logits
        return TfDeterministic(loc=tf.math.argmax(probs_or_logits, axis=-1))

@DeveloperAPI
class TfDiagGaussian(TfDistribution):
    """Wrapper class for Normal distribution.

    Creates a normal distribution parameterized by :attr:`loc` and :attr:`scale`. In
    case of multi-dimensional distribution, the variance is assumed to be diagonal.

    .. testcode::
        :skipif: True

        m = TfDiagGaussian(loc=[0.0, 0.0], scale=[1.0, 1.0])
        m.sample(sample_shape=(2,))  # 2d normal dist with loc=0 and scale=1

    .. testoutput::

        tensor([[ 0.1046, -0.6120], [ 0.234, 0.556]])

    .. testcode::
        :skipif: True

        # scale is None
        m = TfDiagGaussian(loc=[0.0, 1.0])
        m.sample(sample_shape=(2,))  # normally distributed with loc=0 and scale=1

    .. testoutput::

        tensor([0.1046, 0.6120])


    Args:
        loc: mean of the distribution (often referred to as mu). If scale is None, the
            second half of the `loc` will be used as the log of scale.
        scale: standard deviation of the distribution (often referred to as sigma).
            Has to be positive.
    """

    @override(TfDistribution)
    def __init__(self, loc: Union[float, TensorType], scale: Optional[Union[float, TensorType]]=None):
        if False:
            while True:
                i = 10
        self.loc = loc
        super().__init__(loc=loc, scale=scale)

    @override(TfDistribution)
    def _get_tf_distribution(self, loc, scale) -> 'tfp.distributions.Distribution':
        if False:
            while True:
                i = 10
        return tfp.distributions.Normal(loc=loc, scale=scale)

    @override(TfDistribution)
    def logp(self, value: TensorType) -> TensorType:
        if False:
            print('Hello World!')
        return tf.math.reduce_sum(super().logp(value), axis=-1)

    @override(TfDistribution)
    def entropy(self) -> TensorType:
        if False:
            while True:
                i = 10
        return tf.math.reduce_sum(super().entropy(), axis=-1)

    @override(TfDistribution)
    def kl(self, other: 'TfDistribution') -> TensorType:
        if False:
            for i in range(10):
                print('nop')
        return tf.math.reduce_sum(super().kl(other), axis=-1)

    @staticmethod
    @override(Distribution)
    def required_input_dim(space: gym.Space, **kwargs) -> int:
        if False:
            while True:
                i = 10
        assert isinstance(space, gym.spaces.Box)
        return int(np.prod(space.shape, dtype=np.int32) * 2)

    @override(Distribution)
    def rsample(self, sample_shape=()):
        if False:
            return 10
        eps = tf.random.normal(sample_shape)
        return self._dist.loc + eps * self._dist.scale

    @classmethod
    @override(Distribution)
    def from_logits(cls, logits: TensorType, **kwargs) -> 'TfDiagGaussian':
        if False:
            while True:
                i = 10
        (loc, log_std) = tf.split(logits, num_or_size_splits=2, axis=-1)
        scale = tf.math.exp(log_std)
        return TfDiagGaussian(loc=loc, scale=scale)

    def to_deterministic(self) -> 'TfDeterministic':
        if False:
            while True:
                i = 10
        return TfDeterministic(loc=self.loc)

@DeveloperAPI
class TfDeterministic(Distribution):
    """The distribution that returns the input values directly.

    This is similar to DiagGaussian with standard deviation zero (thus only
    requiring the "mean" values as NN output).

    Note: entropy is always zero, ang logp and kl are not implemented.

    .. testcode::
        :skipif: True

        m = TfDeterministic(loc=tf.constant([0.0, 0.0]))
        m.sample(sample_shape=(2,))

    .. testoutput::

        Tensor([[ 0.0, 0.0], [ 0.0, 0.0]])

    Args:
        loc: the determinsitic value to return
    """

    @override(Distribution)
    def __init__(self, loc: 'tf.Tensor') -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.loc = loc

    @override(Distribution)
    def sample(self, *, sample_shape: Tuple[int, ...]=(), **kwargs) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        if False:
            print('Hello World!')
        shape = sample_shape + self.loc.shape
        return tf.ones(shape, dtype=self.loc.dtype) * self.loc

    @override(Distribution)
    def rsample(self, *, sample_shape: Tuple[int, ...]=None, **kwargs) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @override(Distribution)
    def logp(self, value: TensorType, **kwargs) -> TensorType:
        if False:
            while True:
                i = 10
        raise ValueError(f'Cannot return logp for {self.__class__.__name__}.')

    @override(Distribution)
    def entropy(self, **kwargs) -> TensorType:
        if False:
            return 10
        raise tf.zeros_like(self.loc)

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
            return 10
        assert isinstance(space, gym.spaces.Box)
        return int(np.prod(space.shape, dtype=np.int32))

    @classmethod
    @override(Distribution)
    def from_logits(cls, logits: TensorType, **kwargs) -> 'TfDeterministic':
        if False:
            print('Hello World!')
        return TfDeterministic(loc=logits)

    def to_deterministic(self) -> 'TfDeterministic':
        if False:
            for i in range(10):
                print('nop')
        return self

@DeveloperAPI
class TfMultiCategorical(Distribution):
    """MultiCategorical distribution for MultiDiscrete action spaces."""

    @override(Distribution)
    def __init__(self, categoricals: List[TfCategorical]):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._cats = categoricals

    @override(Distribution)
    def sample(self) -> TensorType:
        if False:
            while True:
                i = 10
        arr = [cat.sample() for cat in self._cats]
        sample_ = tf.stack(arr, axis=-1)
        return sample_

    @override(Distribution)
    def rsample(self, sample_shape=()):
        if False:
            for i in range(10):
                print('nop')
        arr = [cat.rsample() for cat in self._cats]
        sample_ = tf.stack(arr, axis=-1)
        return sample_

    @override(Distribution)
    def logp(self, value: tf.Tensor) -> TensorType:
        if False:
            i = 10
            return i + 15
        actions = tf.unstack(tf.cast(value, tf.int32), axis=-1)
        logps = tf.stack([cat.logp(act) for (cat, act) in zip(self._cats, actions)])
        return tf.reduce_sum(logps, axis=0)

    @override(Distribution)
    def entropy(self) -> TensorType:
        if False:
            return 10
        return tf.reduce_sum(tf.stack([cat.entropy() for cat in self._cats], axis=-1), axis=-1)

    @override(Distribution)
    def kl(self, other: Distribution) -> TensorType:
        if False:
            return 10
        kls = tf.stack([cat.kl(oth_cat) for (cat, oth_cat) in zip(self._cats, other._cats)], axis=-1)
        return tf.reduce_sum(kls, axis=-1)

    @staticmethod
    @override(Distribution)
    def required_input_dim(space: gym.Space, **kwargs) -> int:
        if False:
            while True:
                i = 10
        assert isinstance(space, gym.spaces.MultiDiscrete)
        return int(np.sum(space.nvec))

    @classmethod
    @override(Distribution)
    def from_logits(cls, logits: tf.Tensor, input_lens: List[int], **kwargs) -> 'TfMultiCategorical':
        if False:
            for i in range(10):
                print('nop')
        'Creates this Distribution from logits (and additional arguments).\n\n        If you wish to create this distribution from logits only, please refer to\n        `Distribution.get_partial_dist_cls()`.\n\n        Args:\n            logits: The tensor containing logits to be separated by logit_lens.\n                child_distribution_cls_struct: A struct of Distribution classes that can\n                be instantiated from the given logits.\n            input_lens: A list of integers that indicate the length of the logits\n                vectors to be passed into each child distribution.\n            **kwargs: Forward compatibility kwargs.\n        '
        categoricals = [TfCategorical(logits=logits) for logits in tf.split(logits, input_lens, axis=-1)]
        return TfMultiCategorical(categoricals=categoricals)

    def to_deterministic(self) -> 'TfMultiDistribution':
        if False:
            i = 10
            return i + 15
        return TfMultiDistribution([cat.to_deterministic() for cat in self._cats])

@DeveloperAPI
class TfMultiDistribution(Distribution):
    """Action distribution that operates on multiple, possibly nested actions."""

    def __init__(self, child_distribution_struct: Union[Tuple, List, Dict]):
        if False:
            for i in range(10):
                print('nop')
        'Initializes a TfMultiDistribution object.\n\n        Args:\n            child_distribution_struct: Any struct\n                that contains the child distribution classes to use to\n                instantiate the child distributions from `logits`.\n        '
        super().__init__()
        self._original_struct = child_distribution_struct
        self._flat_child_distributions = tree.flatten(child_distribution_struct)

    @override(Distribution)
    def rsample(self, *, sample_shape: Tuple[int, ...]=None, **kwargs) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        if False:
            i = 10
            return i + 15
        rsamples = []
        for dist in self._flat_child_distributions:
            rsample = dist.rsample(sample_shape=sample_shape, **kwargs)
            rsamples.append(rsample)
        rsamples = tree.unflatten_as(self._original_struct, rsamples)
        return rsamples

    @override(Distribution)
    def logp(self, value):
        if False:
            print('Hello World!')
        if isinstance(value, (tf.Tensor, np.ndarray)):
            split_indices = []
            for dist in self._flat_child_distributions:
                if isinstance(dist, TfCategorical):
                    split_indices.append(1)
                elif isinstance(dist, TfMultiCategorical):
                    split_indices.append(len(dist._cats))
                else:
                    sample = dist.sample()
                    if len(sample.shape) == 1:
                        split_indices.append(1)
                    else:
                        split_indices.append(tf.shape(sample)[1])
            split_value = tf.split(value, split_indices, axis=1)
        else:
            split_value = tree.flatten(value)

        def map_(val, dist):
            if False:
                i = 10
                return i + 15
            if isinstance(dist, TfCategorical) and len(val.shape) > 1 and (val.shape[-1] == 1):
                val = tf.squeeze(val, axis=-1)
            return dist.logp(val)
        flat_logps = tree.map_structure(map_, split_value, self._flat_child_distributions)
        return sum(flat_logps)

    @override(Distribution)
    def kl(self, other):
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
            while True:
                i = 10
        child_distributions_struct = tree.unflatten_as(self._original_struct, self._flat_child_distributions)
        return tree.map_structure(lambda s: s.sample(), child_distributions_struct)

    @staticmethod
    @override(Distribution)
    def required_input_dim(space: gym.Space, input_lens: List[int], **kwargs) -> int:
        if False:
            while True:
                i = 10
        return sum(input_lens)

    @classmethod
    @override(Distribution)
    def from_logits(cls, logits: tf.Tensor, child_distribution_cls_struct: Union[Mapping, Iterable], input_lens: Union[Dict, List[int]], space: gym.Space, **kwargs) -> 'TfMultiDistribution':
        if False:
            while True:
                i = 10
        'Creates this Distribution from logits (and additional arguments).\n\n        If you wish to create this distribution from logits only, please refer to\n        `Distribution.get_partial_dist_cls()`.\n\n        Args:\n            logits: The tensor containing logits to be separated by `input_lens`.\n                child_distribution_cls_struct: A struct of Distribution classes that can\n                be instantiated from the given logits.\n            child_distribution_cls_struct: A struct of Distribution classes that can\n                be instantiated from the given logits.\n            input_lens: A list or dict of integers that indicate the length of each\n                logit. If this is given as a dict, the structure should match the\n                structure of child_distribution_cls_struct.\n            space: The possibly nested output space.\n            **kwargs: Forward compatibility kwargs.\n\n        Returns:\n            A TfMultiDistribution object.\n        '
        logit_lens = tree.flatten(input_lens)
        child_distribution_cls_list = tree.flatten(child_distribution_cls_struct)
        split_logits = tf.split(logits, logit_lens, axis=1)
        child_distribution_list = tree.map_structure(lambda dist, input_: dist.from_logits(input_), child_distribution_cls_list, list(split_logits))
        child_distribution_struct = tree.unflatten_as(child_distribution_cls_struct, child_distribution_list)
        return TfMultiDistribution(child_distribution_struct=child_distribution_struct)

    def to_deterministic(self) -> 'TfMultiDistribution':
        if False:
            for i in range(10):
                print('nop')
        flat_deterministic_dists = [dist.to_deterministic for dist in self._flat_child_distributions]
        deterministic_dists = tree.unflatten_as(self._original_struct, flat_deterministic_dists)
        return TfMultiDistribution(deterministic_dists)