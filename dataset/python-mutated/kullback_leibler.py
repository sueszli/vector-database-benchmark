"""Registration and usage mechanisms for KL-divergences."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
_DIVERGENCES = {}
__all__ = ['RegisterKL', 'kl_divergence']

def _registered_kl(type_a, type_b):
    if False:
        i = 10
        return i + 15
    'Get the KL function registered for classes a and b.'
    hierarchy_a = tf_inspect.getmro(type_a)
    hierarchy_b = tf_inspect.getmro(type_b)
    dist_to_children = None
    kl_fn = None
    for (mro_to_a, parent_a) in enumerate(hierarchy_a):
        for (mro_to_b, parent_b) in enumerate(hierarchy_b):
            candidate_dist = mro_to_a + mro_to_b
            candidate_kl_fn = _DIVERGENCES.get((parent_a, parent_b), None)
            if not kl_fn or (candidate_kl_fn and candidate_dist < dist_to_children):
                dist_to_children = candidate_dist
                kl_fn = candidate_kl_fn
    return kl_fn

@deprecation.deprecated('2019-01-01', 'The TensorFlow Distributions library has moved to TensorFlow Probability (https://github.com/tensorflow/probability). You should update all references to use `tfp.distributions` instead of `tf.distributions`.', warn_once=True)
@tf_export(v1=['distributions.kl_divergence'])
def kl_divergence(distribution_a, distribution_b, allow_nan_stats=True, name=None):
    if False:
        while True:
            i = 10
    'Get the KL-divergence KL(distribution_a || distribution_b).\n\n  If there is no KL method registered specifically for `type(distribution_a)`\n  and `type(distribution_b)`, then the class hierarchies of these types are\n  searched.\n\n  If one KL method is registered between any pairs of classes in these two\n  parent hierarchies, it is used.\n\n  If more than one such registered method exists, the method whose registered\n  classes have the shortest sum MRO paths to the input types is used.\n\n  If more than one such shortest path exists, the first method\n  identified in the search is used (favoring a shorter MRO distance to\n  `type(distribution_a)`).\n\n  Args:\n    distribution_a: The first distribution.\n    distribution_b: The second distribution.\n    allow_nan_stats: Python `bool`, default `True`. When `True`,\n      statistics (e.g., mean, mode, variance) use the value "`NaN`" to\n      indicate the result is undefined. When `False`, an exception is raised\n      if one or more of the statistic\'s batch members are undefined.\n    name: Python `str` name prefixed to Ops created by this class.\n\n  Returns:\n    A Tensor with the batchwise KL-divergence between `distribution_a`\n    and `distribution_b`.\n\n  Raises:\n    NotImplementedError: If no KL method is defined for distribution types\n      of `distribution_a` and `distribution_b`.\n  '
    kl_fn = _registered_kl(type(distribution_a), type(distribution_b))
    if kl_fn is None:
        raise NotImplementedError('No KL(distribution_a || distribution_b) registered for distribution_a type %s and distribution_b type %s' % (type(distribution_a).__name__, type(distribution_b).__name__))
    with ops.name_scope('KullbackLeibler'):
        kl_t = kl_fn(distribution_a, distribution_b, name=name)
        if allow_nan_stats:
            return kl_t
        kl_t = array_ops.identity(kl_t, name='kl')
        with ops.control_dependencies([control_flow_assert.Assert(math_ops.logical_not(math_ops.reduce_any(math_ops.is_nan(kl_t))), ['KL calculation between %s and %s returned NaN values (and was called with allow_nan_stats=False). Values:' % (distribution_a.name, distribution_b.name), kl_t])]):
            return array_ops.identity(kl_t, name='checked_kl')

@deprecation.deprecated('2019-01-01', 'The TensorFlow Distributions library has moved to TensorFlow Probability (https://github.com/tensorflow/probability). You should update all references to use `tfp.distributions` instead of `tf.distributions`.', warn_once=True)
def cross_entropy(ref, other, allow_nan_stats=True, name=None):
    if False:
        i = 10
        return i + 15
    'Computes the (Shannon) cross entropy.\n\n  Denote two distributions by `P` (`ref`) and `Q` (`other`). Assuming `P, Q`\n  are absolutely continuous with respect to one another and permit densities\n  `p(x) dr(x)` and `q(x) dr(x)`, (Shanon) cross entropy is defined as:\n\n  ```none\n  H[P, Q] = E_p[-log q(X)] = -int_F p(x) log q(x) dr(x)\n  ```\n\n  where `F` denotes the support of the random variable `X ~ P`.\n\n  Args:\n    ref: `tfd.Distribution` instance.\n    other: `tfd.Distribution` instance.\n    allow_nan_stats: Python `bool`, default `True`. When `True`,\n      statistics (e.g., mean, mode, variance) use the value "`NaN`" to\n      indicate the result is undefined. When `False`, an exception is raised\n      if one or more of the statistic\'s batch members are undefined.\n    name: Python `str` prepended to names of ops created by this function.\n\n  Returns:\n    cross_entropy: `ref.dtype` `Tensor` with shape `[B1, ..., Bn]`\n      representing `n` different calculations of (Shanon) cross entropy.\n  '
    with ops.name_scope(name, 'cross_entropy'):
        return ref.entropy() + kl_divergence(ref, other, allow_nan_stats=allow_nan_stats)

@tf_export(v1=['distributions.RegisterKL'])
class RegisterKL:
    """Decorator to register a KL divergence implementation function.

  Usage:

  @distributions.RegisterKL(distributions.Normal, distributions.Normal)
  def _kl_normal_mvn(norm_a, norm_b):
    # Return KL(norm_a || norm_b)
  """

    @deprecation.deprecated('2019-01-01', 'The TensorFlow Distributions library has moved to TensorFlow Probability (https://github.com/tensorflow/probability). You should update all references to use `tfp.distributions` instead of `tf.distributions`.', warn_once=True)
    def __init__(self, dist_cls_a, dist_cls_b):
        if False:
            print('Hello World!')
        'Initialize the KL registrar.\n\n    Args:\n      dist_cls_a: the class of the first argument of the KL divergence.\n      dist_cls_b: the class of the second argument of the KL divergence.\n    '
        self._key = (dist_cls_a, dist_cls_b)

    def __call__(self, kl_fn):
        if False:
            i = 10
            return i + 15
        'Perform the KL registration.\n\n    Args:\n      kl_fn: The function to use for the KL divergence.\n\n    Returns:\n      kl_fn\n\n    Raises:\n      TypeError: if kl_fn is not a callable.\n      ValueError: if a KL divergence function has already been registered for\n        the given argument classes.\n    '
        if not callable(kl_fn):
            raise TypeError('kl_fn must be callable, received: %s' % kl_fn)
        if self._key in _DIVERGENCES:
            raise ValueError('KL(%s || %s) has already been registered to: %s' % (self._key[0].__name__, self._key[1].__name__, _DIVERGENCES[self._key]))
        _DIVERGENCES[self._key] = kl_fn
        return kl_fn