import warnings
from collections import OrderedDict
import torch
from torch.distributions.kl import kl_divergence, register_kl
import pyro.distributions.torch
from . import constraints
from .distribution import Distribution
from .score_parts import ScoreParts
from .util import broadcast_shape, scale_and_mask

class TorchDistributionMixin(Distribution):
    """
    Mixin to provide Pyro compatibility for PyTorch distributions.

    You should instead use `TorchDistribution` for new distribution classes.

    This is mainly useful for wrapping existing PyTorch distributions for
    use in Pyro.  Derived classes must first inherit from
    :class:`torch.distributions.distribution.Distribution` and then inherit
    from :class:`TorchDistributionMixin`.
    """

    def __call__(self, sample_shape: torch.Size=torch.Size()) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        Samples a random value.\n\n        This is reparameterized whenever possible, calling\n        :meth:`~torch.distributions.distribution.Distribution.rsample` for\n        reparameterized distributions and\n        :meth:`~torch.distributions.distribution.Distribution.sample` for\n        non-reparameterized distributions.\n\n        :param sample_shape: the size of the iid batch to be drawn from the\n            distribution.\n        :type sample_shape: torch.Size\n        :return: A random value or batch of random values (if parameters are\n            batched). The shape of the result should be `self.shape()`.\n        :rtype: torch.Tensor\n        '
        return self.rsample(sample_shape) if self.has_rsample else self.sample(sample_shape)

    @property
    def event_dim(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        :return: Number of dimensions of individual events.\n        :rtype: int\n        '
        return len(self.event_shape)

    def shape(self, sample_shape=torch.Size()):
        if False:
            i = 10
            return i + 15
        '\n        The tensor shape of samples from this distribution.\n\n        Samples are of shape::\n\n          d.shape(sample_shape) == sample_shape + d.batch_shape + d.event_shape\n\n        :param sample_shape: the size of the iid batch to be drawn from the\n            distribution.\n        :type sample_shape: torch.Size\n        :return: Tensor shape of samples.\n        :rtype: torch.Size\n        '
        return sample_shape + self.batch_shape + self.event_shape

    @classmethod
    def infer_shapes(cls, **arg_shapes):
        if False:
            i = 10
            return i + 15
        '\n        Infers ``batch_shape`` and ``event_shape`` given shapes of args to\n        :meth:`__init__`.\n\n        .. note:: This assumes distribution shape depends only on the shapes\n            of tensor inputs, not in the data contained in those inputs.\n\n        :param \\*\\*arg_shapes: Keywords mapping name of input arg to\n            :class:`torch.Size` or tuple representing the sizes of each\n            tensor input.\n        :returns: A pair ``(batch_shape, event_shape)`` of the shapes of a\n            distribution that would be created with input args of the given\n            shapes.\n        :rtype: tuple\n        '
        if cls.support.event_dim > 0:
            raise NotImplementedError
        batch_shapes = []
        for (name, shape) in arg_shapes.items():
            event_dim = cls.arg_constraints.get(name, constraints.real).event_dim
            batch_shapes.append(shape[:len(shape) - event_dim])
        batch_shape = torch.Size(broadcast_shape(*batch_shapes))
        event_shape = torch.Size()
        return (batch_shape, event_shape)

    def expand(self, batch_shape, _instance=None):
        if False:
            while True:
                i = 10
        '\n        Returns a new :class:`ExpandedDistribution` instance with batch\n        dimensions expanded to `batch_shape`.\n\n        :param tuple batch_shape: batch shape to expand to.\n        :param _instance: unused argument for compatibility with\n            :meth:`torch.distributions.Distribution.expand`\n        :return: an instance of `ExpandedDistribution`.\n        :rtype: :class:`ExpandedDistribution`\n        '
        return ExpandedDistribution(self, batch_shape)

    def expand_by(self, sample_shape):
        if False:
            print('Hello World!')
        '\n        Expands a distribution by adding ``sample_shape`` to the left side of\n        its :attr:`~torch.distributions.distribution.Distribution.batch_shape`.\n\n        To expand internal dims of ``self.batch_shape`` from 1 to something\n        larger, use :meth:`expand` instead.\n\n        :param torch.Size sample_shape: The size of the iid batch to be drawn\n            from the distribution.\n        :return: An expanded version of this distribution.\n        :rtype: :class:`ExpandedDistribution`\n        '
        try:
            expanded_dist = self.expand(torch.Size(sample_shape) + self.batch_shape)
        except NotImplementedError:
            expanded_dist = TorchDistributionMixin.expand(self, torch.Size(sample_shape) + self.batch_shape)
        return expanded_dist

    def reshape(self, sample_shape=None, extra_event_dims=None):
        if False:
            for i in range(10):
                print('nop')
        raise Exception('\n            .reshape(sample_shape=s, extra_event_dims=n) was renamed and split into\n            .expand_by(sample_shape=s).to_event(reinterpreted_batch_ndims=n).')

    def to_event(self, reinterpreted_batch_ndims=None):
        if False:
            i = 10
            return i + 15
        '\n        Reinterprets the ``n`` rightmost dimensions of this distributions\n        :attr:`~torch.distributions.distribution.Distribution.batch_shape`\n        as event dims, adding them to the left side of\n        :attr:`~torch.distributions.distribution.Distribution.event_shape`.\n\n        Example:\n\n            .. doctest::\n               :hide:\n\n               >>> d0 = dist.Normal(torch.zeros(2, 3, 4, 5), torch.ones(2, 3, 4, 5))\n               >>> [d0.batch_shape, d0.event_shape]\n               [torch.Size([2, 3, 4, 5]), torch.Size([])]\n               >>> d1 = d0.to_event(2)\n\n            >>> [d1.batch_shape, d1.event_shape]\n            [torch.Size([2, 3]), torch.Size([4, 5])]\n            >>> d2 = d1.to_event(1)\n            >>> [d2.batch_shape, d2.event_shape]\n            [torch.Size([2]), torch.Size([3, 4, 5])]\n            >>> d3 = d1.to_event(2)\n            >>> [d3.batch_shape, d3.event_shape]\n            [torch.Size([]), torch.Size([2, 3, 4, 5])]\n\n        :param int reinterpreted_batch_ndims: The number of batch dimensions to\n            reinterpret as event dimensions. May be negative to remove\n            dimensions from an :class:`pyro.distributions.torch.Independent` .\n            If None, convert all dimensions to event dimensions.\n        :return: A reshaped version of this distribution.\n        :rtype: :class:`pyro.distributions.torch.Independent`\n        '
        if reinterpreted_batch_ndims is None:
            reinterpreted_batch_ndims = len(self.batch_shape)
        base_dist = self
        while isinstance(base_dist, torch.distributions.Independent):
            reinterpreted_batch_ndims += base_dist.reinterpreted_batch_ndims
            base_dist = base_dist.base_dist
        if reinterpreted_batch_ndims == 0:
            return base_dist
        if reinterpreted_batch_ndims < 0:
            raise ValueError('Cannot remove event dimensions from {}'.format(type(self)))
        return pyro.distributions.torch.Independent(base_dist, reinterpreted_batch_ndims)

    def independent(self, reinterpreted_batch_ndims=None):
        if False:
            i = 10
            return i + 15
        warnings.warn('independent is deprecated; use to_event instead', DeprecationWarning)
        return self.to_event(reinterpreted_batch_ndims=reinterpreted_batch_ndims)

    def mask(self, mask):
        if False:
            i = 10
            return i + 15
        '\n        Masks a distribution by a boolean or boolean-valued tensor that is\n        broadcastable to the distributions\n        :attr:`~torch.distributions.distribution.Distribution.batch_shape` .\n\n        :param mask: A boolean or boolean valued tensor.\n        :type mask: bool or torch.Tensor\n        :return: A masked copy of this distribution.\n        :rtype: :class:`MaskedDistribution`\n        '
        return MaskedDistribution(self, mask)

class TorchDistribution(torch.distributions.Distribution, TorchDistributionMixin):
    """
    Base class for PyTorch-compatible distributions with Pyro support.

    This should be the base class for almost all new Pyro distributions.

    .. note::

        Parameters and data should be of type :class:`~torch.Tensor`
        and all methods return type :class:`~torch.Tensor` unless
        otherwise noted.

    **Tensor Shapes**:

    TorchDistributions provide a method ``.shape()`` for the tensor shape of samples::

      x = d.sample(sample_shape)
      assert x.shape == d.shape(sample_shape)

    Pyro follows the same distribution shape semantics as PyTorch. It distinguishes
    between three different roles for tensor shapes of samples:

    - *sample shape* corresponds to the shape of the iid samples drawn from the distribution.
      This is taken as an argument by the distribution's `sample` method.
    - *batch shape* corresponds to non-identical (independent) parameterizations of
      the distribution, inferred from the distribution's parameter shapes. This is
      fixed for a distribution instance.
    - *event shape* corresponds to the event dimensions of the distribution, which
      is fixed for a distribution class. These are collapsed when we try to score
      a sample from the distribution via `d.log_prob(x)`.

    These shapes are related by the equation::

      assert d.shape(sample_shape) == sample_shape + d.batch_shape + d.event_shape

    Distributions provide a vectorized
    :meth:`~torch.distributions.distribution.Distribution.log_prob` method that
    evaluates the log probability density of each event in a batch
    independently, returning a tensor of shape
    ``sample_shape + d.batch_shape``::

      x = d.sample(sample_shape)
      assert x.shape == d.shape(sample_shape)
      log_p = d.log_prob(x)
      assert log_p.shape == sample_shape + d.batch_shape

    **Implementing New Distributions**:

    Derived classes must implement the methods
    :meth:`~torch.distributions.distribution.Distribution.sample`
    (or :meth:`~torch.distributions.distribution.Distribution.rsample` if
    ``.has_rsample == True``) and
    :meth:`~torch.distributions.distribution.Distribution.log_prob`, and must
    implement the properties
    :attr:`~torch.distributions.distribution.Distribution.batch_shape`,
    and :attr:`~torch.distributions.distribution.Distribution.event_shape`.
    Discrete classes may also implement the
    :meth:`~torch.distributions.distribution.Distribution.enumerate_support`
    method to improve gradient estimates and set
    ``.has_enumerate_support = True``.
    """
    expand = TorchDistributionMixin.expand

class MaskedDistribution(TorchDistribution):
    """
    Masks a distribution by a boolean tensor that is broadcastable to the
    distribution's :attr:`~torch.distributions.distribution.Distribution.batch_shape`.

    In the special case ``mask is False``, computation of :meth:`log_prob` ,
    :meth:`score_parts` , and ``kl_divergence()`` is skipped, and constant zero
    values are returned instead.

    :param mask: A boolean or boolean-valued tensor.
    :type mask: torch.Tensor or bool
    """
    arg_constraints = {}

    def __init__(self, base_dist, mask):
        if False:
            i = 10
            return i + 15
        if isinstance(mask, bool):
            self._mask = mask
        else:
            batch_shape = broadcast_shape(mask.shape, base_dist.batch_shape)
            if mask.shape != batch_shape:
                mask = mask.expand(batch_shape)
            if base_dist.batch_shape != batch_shape:
                base_dist = base_dist.expand(batch_shape)
            self._mask = mask.bool()
        self.base_dist = base_dist
        super().__init__(base_dist.batch_shape, base_dist.event_shape)

    def expand(self, batch_shape, _instance=None):
        if False:
            return 10
        new = self._get_checked_instance(MaskedDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        new.base_dist = self.base_dist.expand(batch_shape)
        new._mask = self._mask
        if isinstance(new._mask, torch.Tensor):
            new._mask = new._mask.expand(batch_shape)
        super(MaskedDistribution, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def has_rsample(self):
        if False:
            while True:
                i = 10
        return self.base_dist.has_rsample

    @property
    def has_enumerate_support(self):
        if False:
            return 10
        return self.base_dist.has_enumerate_support

    @constraints.dependent_property
    def support(self):
        if False:
            return 10
        return self.base_dist.support

    def sample(self, sample_shape=torch.Size()):
        if False:
            i = 10
            return i + 15
        return self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        if False:
            while True:
                i = 10
        return self.base_dist.rsample(sample_shape)

    def log_prob(self, value):
        if False:
            for i in range(10):
                print('nop')
        if self._mask is False:
            shape = broadcast_shape(self.base_dist.batch_shape, value.shape[:value.dim() - self.event_dim])
            return torch.zeros((), device=value.device).expand(shape)
        if self._mask is True:
            return self.base_dist.log_prob(value)
        return scale_and_mask(self.base_dist.log_prob(value), mask=self._mask)

    def score_parts(self, value):
        if False:
            print('Hello World!')
        if isinstance(self._mask, bool):
            return super().score_parts(value)
        return self.base_dist.score_parts(value).scale_and_mask(mask=self._mask)

    def enumerate_support(self, expand=True):
        if False:
            i = 10
            return i + 15
        return self.base_dist.enumerate_support(expand=expand)

    @property
    def mean(self):
        if False:
            print('Hello World!')
        return self.base_dist.mean

    @property
    def variance(self):
        if False:
            for i in range(10):
                print('nop')
        return self.base_dist.variance

    def conjugate_update(self, other):
        if False:
            while True:
                i = 10
        '\n        EXPERIMENTAL.\n        '
        (updated, log_normalizer) = self.base_dist.conjugate_update(other)
        updated = updated.mask(self._mask)
        log_normalizer = torch.where(self._mask, log_normalizer, torch.zeros_like(log_normalizer))
        return (updated, log_normalizer)

class ExpandedDistribution(TorchDistribution):
    arg_constraints = {}

    def __init__(self, base_dist, batch_shape=torch.Size()):
        if False:
            for i in range(10):
                print('nop')
        self.base_dist = base_dist
        super().__init__(base_dist.batch_shape, base_dist.event_shape)
        self.expand(batch_shape)

    def expand(self, batch_shape, _instance=None):
        if False:
            return 10
        (new_shape, _, _) = self._broadcast_shape(self.batch_shape, batch_shape)
        (new_shape, expanded_sizes, interstitial_sizes) = self._broadcast_shape(self.base_dist.batch_shape, new_shape)
        self._batch_shape = new_shape
        self._expanded_sizes = expanded_sizes
        self._interstitial_sizes = interstitial_sizes
        return self

    @staticmethod
    def _broadcast_shape(existing_shape, new_shape):
        if False:
            for i in range(10):
                print('nop')
        if len(new_shape) < len(existing_shape):
            raise ValueError('Cannot broadcast distribution of shape {} to shape {}'.format(existing_shape, new_shape))
        reversed_shape = list(reversed(existing_shape))
        (expanded_sizes, interstitial_sizes) = ([], [])
        for (i, size) in enumerate(reversed(new_shape)):
            if i >= len(reversed_shape):
                reversed_shape.append(size)
                expanded_sizes.append((-i - 1, size))
            elif reversed_shape[i] == 1:
                if size != 1:
                    reversed_shape[i] = size
                    interstitial_sizes.append((-i - 1, size))
            elif reversed_shape[i] != size:
                raise ValueError('Cannot broadcast distribution of shape {} to shape {}'.format(existing_shape, new_shape))
        return (tuple(reversed(reversed_shape)), OrderedDict(expanded_sizes), OrderedDict(interstitial_sizes))

    @property
    def has_rsample(self):
        if False:
            while True:
                i = 10
        return self.base_dist.has_rsample

    @property
    def has_enumerate_support(self):
        if False:
            while True:
                i = 10
        return self.base_dist.has_enumerate_support

    @constraints.dependent_property
    def support(self):
        if False:
            for i in range(10):
                print('nop')
        return self.base_dist.support

    def _sample(self, sample_fn, sample_shape):
        if False:
            print('Hello World!')
        interstitial_dims = tuple(self._interstitial_sizes.keys())
        interstitial_dims = tuple((i - self.event_dim for i in interstitial_dims))
        interstitial_sizes = tuple(self._interstitial_sizes.values())
        expanded_sizes = tuple(self._expanded_sizes.values())
        batch_shape = expanded_sizes + interstitial_sizes
        samples = sample_fn(sample_shape + batch_shape)
        interstitial_idx = len(sample_shape) + len(expanded_sizes)
        interstitial_sample_dims = tuple(range(interstitial_idx, interstitial_idx + len(interstitial_sizes)))
        for (dim1, dim2) in zip(interstitial_dims, interstitial_sample_dims):
            samples = samples.transpose(dim1, dim2)
        return samples.reshape(sample_shape + self.batch_shape + self.event_shape)

    def sample(self, sample_shape=torch.Size()):
        if False:
            print('Hello World!')
        return self._sample(self.base_dist.sample, sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        if False:
            return 10
        return self._sample(self.base_dist.rsample, sample_shape)

    def log_prob(self, value):
        if False:
            while True:
                i = 10
        shape = broadcast_shape(self.batch_shape, value.shape[:value.dim() - self.event_dim])
        log_prob = self.base_dist.log_prob(value)
        return log_prob.expand(shape)

    def score_parts(self, value):
        if False:
            while True:
                i = 10
        shape = broadcast_shape(self.batch_shape, value.shape[:value.dim() - self.event_dim])
        (log_prob, score_function, entropy_term) = self.base_dist.score_parts(value)
        if self.batch_shape != self.base_dist.batch_shape:
            log_prob = log_prob.expand(shape)
            if isinstance(score_function, torch.Tensor):
                score_function = score_function.expand(shape)
            if isinstance(entropy_term, torch.Tensor):
                entropy_term = entropy_term.expand(shape)
        return ScoreParts(log_prob, score_function, entropy_term)

    def enumerate_support(self, expand=True):
        if False:
            print('Hello World!')
        samples = self.base_dist.enumerate_support(expand=False)
        enum_shape = samples.shape[:1]
        samples = samples.reshape(enum_shape + (1,) * len(self.batch_shape))
        if expand:
            samples = samples.expand(enum_shape + self.batch_shape)
        return samples

    @property
    def mean(self):
        if False:
            while True:
                i = 10
        return self.base_dist.mean.expand(self.batch_shape + self.event_shape)

    @property
    def variance(self):
        if False:
            i = 10
            return i + 15
        return self.base_dist.variance.expand(self.batch_shape + self.event_shape)

    def conjugate_update(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        EXPERIMENTAL.\n        '
        (updated, log_normalizer) = self.base_dist.conjugate_update(other)
        updated = updated.expand(self.batch_shape)
        log_normalizer = log_normalizer.expand(self.batch_shape)
        return (updated, log_normalizer)

@register_kl(MaskedDistribution, MaskedDistribution)
def _kl_masked_masked(p, q):
    if False:
        return 10
    if p._mask is False or q._mask is False:
        mask = False
    elif p._mask is True:
        mask = q._mask
    elif q._mask is True:
        mask = p._mask
    elif p._mask is q._mask:
        mask = p._mask
    else:
        mask = p._mask & q._mask
    if mask is False:
        return 0.0
    if mask is True:
        return kl_divergence(p.base_dist, q.base_dist)
    kl = kl_divergence(p.base_dist, q.base_dist)
    return scale_and_mask(kl, mask=mask)