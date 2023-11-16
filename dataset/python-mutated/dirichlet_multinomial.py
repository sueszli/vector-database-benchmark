"""The DirichletMultinomial distribution class."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
__all__ = ['DirichletMultinomial']
_dirichlet_multinomial_sample_note = 'For each batch of counts,\n`value = [n_0, ..., n_{K-1}]`, `P[value]` is the probability that after\nsampling `self.total_count` draws from this Dirichlet-Multinomial distribution,\nthe number of draws falling in class `j` is `n_j`. Since this definition is\n[exchangeable](https://en.wikipedia.org/wiki/Exchangeable_random_variables);\ndifferent sequences have the same counts so the probability includes a\ncombinatorial coefficient.\n\nNote: `value` must be a non-negative tensor with dtype `self.dtype`, have no\nfractional components, and such that\n`tf.reduce_sum(value, -1) = self.total_count`. Its shape must be broadcastable\nwith `self.concentration` and `self.total_count`.'

@tf_export(v1=['distributions.DirichletMultinomial'])
class DirichletMultinomial(distribution.Distribution):
    """Dirichlet-Multinomial compound distribution.

  The Dirichlet-Multinomial distribution is parameterized by a (batch of)
  length-`K` `concentration` vectors (`K > 1`) and a `total_count` number of
  trials, i.e., the number of trials per draw from the DirichletMultinomial. It
  is defined over a (batch of) length-`K` vector `counts` such that
  `tf.reduce_sum(counts, -1) = total_count`. The Dirichlet-Multinomial is
  identically the Beta-Binomial distribution when `K = 2`.

  #### Mathematical Details

  The Dirichlet-Multinomial is a distribution over `K`-class counts, i.e., a
  length-`K` vector of non-negative integer `counts = n = [n_0, ..., n_{K-1}]`.

  The probability mass function (pmf) is,

  ```none
  pmf(n; alpha, N) = Beta(alpha + n) / (prod_j n_j!) / Z
  Z = Beta(alpha) / N!
  ```

  where:

  * `concentration = alpha = [alpha_0, ..., alpha_{K-1}]`, `alpha_j > 0`,
  * `total_count = N`, `N` a positive integer,
  * `N!` is `N` factorial, and,
  * `Beta(x) = prod_j Gamma(x_j) / Gamma(sum_j x_j)` is the
    [multivariate beta function](
    https://en.wikipedia.org/wiki/Beta_function#Multivariate_beta_function),
    and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  Dirichlet-Multinomial is a [compound distribution](
  https://en.wikipedia.org/wiki/Compound_probability_distribution), i.e., its
  samples are generated as follows.

    1. Choose class probabilities:
       `probs = [p_0,...,p_{K-1}] ~ Dir(concentration)`
    2. Draw integers:
       `counts = [n_0,...,n_{K-1}] ~ Multinomial(total_count, probs)`

  The last `concentration` dimension parametrizes a single Dirichlet-Multinomial
  distribution. When calling distribution functions (e.g., `dist.prob(counts)`),
  `concentration`, `total_count` and `counts` are broadcast to the same shape.
  The last dimension of `counts` corresponds single Dirichlet-Multinomial
  distributions.

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.

  #### Pitfalls

  The number of classes, `K`, must not exceed:
  - the largest integer representable by `self.dtype`, i.e.,
    `2**(mantissa_bits+1)` (IEE754),
  - the maximum `Tensor` index, i.e., `2**31-1`.

  In other words,

  ```python
  K <= min(2**31-1, {
    tf.float16: 2**11,
    tf.float32: 2**24,
    tf.float64: 2**53 }[param.dtype])
  ```

  Note: This condition is validated only when `self.validate_args = True`.

  #### Examples

  ```python
  alpha = [1., 2., 3.]
  n = 2.
  dist = DirichletMultinomial(n, alpha)
  ```

  Creates a 3-class distribution, with the 3rd class is most likely to be
  drawn.
  The distribution functions can be evaluated on counts.

  ```python
  # counts same shape as alpha.
  counts = [0., 0., 2.]
  dist.prob(counts)  # Shape []

  # alpha will be broadcast to [[1., 2., 3.], [1., 2., 3.]] to match counts.
  counts = [[1., 1., 0.], [1., 0., 1.]]
  dist.prob(counts)  # Shape [2]

  # alpha will be broadcast to shape [5, 7, 3] to match counts.
  counts = [[...]]  # Shape [5, 7, 3]
  dist.prob(counts)  # Shape [5, 7]
  ```

  Creates a 2-batch of 3-class distributions.

  ```python
  alpha = [[1., 2., 3.], [4., 5., 6.]]  # Shape [2, 3]
  n = [3., 3.]
  dist = DirichletMultinomial(n, alpha)

  # counts will be broadcast to [[2., 1., 0.], [2., 1., 0.]] to match alpha.
  counts = [2., 1., 0.]
  dist.prob(counts)  # Shape [2]
  ```

  """

    @deprecation.deprecated('2019-01-01', 'The TensorFlow Distributions library has moved to TensorFlow Probability (https://github.com/tensorflow/probability). You should update all references to use `tfp.distributions` instead of `tf.distributions`.', warn_once=True)
    def __init__(self, total_count, concentration, validate_args=False, allow_nan_stats=True, name='DirichletMultinomial'):
        if False:
            return 10
        'Initialize a batch of DirichletMultinomial distributions.\n\n    Args:\n      total_count:  Non-negative floating point tensor, whose dtype is the same\n        as `concentration`. The shape is broadcastable to `[N1,..., Nm]` with\n        `m >= 0`. Defines this as a batch of `N1 x ... x Nm` different\n        Dirichlet multinomial distributions. Its components should be equal to\n        integer values.\n      concentration: Positive floating point tensor, whose dtype is the\n        same as `n` with shape broadcastable to `[N1,..., Nm, K]` `m >= 0`.\n        Defines this as a batch of `N1 x ... x Nm` different `K` class Dirichlet\n        multinomial distributions.\n      validate_args: Python `bool`, default `False`. When `True` distribution\n        parameters are checked for validity despite possibly degrading runtime\n        performance. When `False` invalid inputs may silently render incorrect\n        outputs.\n      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics\n        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the\n        result is undefined. When `False`, an exception is raised if one or\n        more of the statistic\'s batch members are undefined.\n      name: Python `str` name prefixed to Ops created by this class.\n    '
        parameters = dict(locals())
        with ops.name_scope(name, values=[total_count, concentration]) as name:
            self._total_count = ops.convert_to_tensor(total_count, name='total_count')
            if validate_args:
                self._total_count = distribution_util.embed_check_nonnegative_integer_form(self._total_count)
            self._concentration = self._maybe_assert_valid_concentration(ops.convert_to_tensor(concentration, name='concentration'), validate_args)
            self._total_concentration = math_ops.reduce_sum(self._concentration, -1)
        super(DirichletMultinomial, self).__init__(dtype=self._concentration.dtype, validate_args=validate_args, allow_nan_stats=allow_nan_stats, reparameterization_type=distribution.NOT_REPARAMETERIZED, parameters=parameters, graph_parents=[self._total_count, self._concentration], name=name)

    @property
    def total_count(self):
        if False:
            for i in range(10):
                print('nop')
        'Number of trials used to construct a sample.'
        return self._total_count

    @property
    def concentration(self):
        if False:
            return 10
        'Concentration parameter; expected prior counts for that coordinate.'
        return self._concentration

    @property
    def total_concentration(self):
        if False:
            print('Hello World!')
        'Sum of last dim of concentration parameter.'
        return self._total_concentration

    def _batch_shape_tensor(self):
        if False:
            print('Hello World!')
        return array_ops.shape(self.total_concentration)

    def _batch_shape(self):
        if False:
            return 10
        return self.total_concentration.get_shape()

    def _event_shape_tensor(self):
        if False:
            i = 10
            return i + 15
        return array_ops.shape(self.concentration)[-1:]

    def _event_shape(self):
        if False:
            while True:
                i = 10
        return self.concentration.get_shape().with_rank_at_least(1)[-1:]

    def _sample_n(self, n, seed=None):
        if False:
            i = 10
            return i + 15
        n_draws = math_ops.cast(self.total_count, dtype=dtypes.int32)
        k = self.event_shape_tensor()[0]
        unnormalized_logits = array_ops.reshape(math_ops.log(random_ops.random_gamma(shape=[n], alpha=self.concentration, dtype=self.dtype, seed=seed)), shape=[-1, k])
        draws = random_ops.multinomial(logits=unnormalized_logits, num_samples=n_draws, seed=distribution_util.gen_new_seed(seed, salt='dirichlet_multinomial'))
        x = math_ops.reduce_sum(array_ops.one_hot(draws, depth=k), -2)
        final_shape = array_ops.concat([[n], self.batch_shape_tensor(), [k]], 0)
        x = array_ops.reshape(x, final_shape)
        return math_ops.cast(x, self.dtype)

    @distribution_util.AppendDocstring(_dirichlet_multinomial_sample_note)
    def _log_prob(self, counts):
        if False:
            print('Hello World!')
        counts = self._maybe_assert_valid_sample(counts)
        ordered_prob = special_math_ops.lbeta(self.concentration + counts) - special_math_ops.lbeta(self.concentration)
        return ordered_prob + distribution_util.log_combinations(self.total_count, counts)

    @distribution_util.AppendDocstring(_dirichlet_multinomial_sample_note)
    def _prob(self, counts):
        if False:
            while True:
                i = 10
        return math_ops.exp(self._log_prob(counts))

    def _mean(self):
        if False:
            while True:
                i = 10
        return self.total_count * (self.concentration / self.total_concentration[..., array_ops.newaxis])

    @distribution_util.AppendDocstring('The covariance for each batch member is defined as the following:\n\n      ```none\n      Var(X_j) = n * alpha_j / alpha_0 * (1 - alpha_j / alpha_0) *\n      (n + alpha_0) / (1 + alpha_0)\n      ```\n\n      where `concentration = alpha` and\n      `total_concentration = alpha_0 = sum_j alpha_j`.\n\n      The covariance between elements in a batch is defined as:\n\n      ```none\n      Cov(X_i, X_j) = -n * alpha_i * alpha_j / alpha_0 ** 2 *\n      (n + alpha_0) / (1 + alpha_0)\n      ```\n      ')
    def _covariance(self):
        if False:
            i = 10
            return i + 15
        x = self._variance_scale_term() * self._mean()
        return array_ops.matrix_set_diag(-math_ops.matmul(x[..., array_ops.newaxis], x[..., array_ops.newaxis, :]), self._variance())

    def _variance(self):
        if False:
            print('Hello World!')
        scale = self._variance_scale_term()
        x = scale * self._mean()
        return x * (self.total_count * scale - x)

    def _variance_scale_term(self):
        if False:
            print('Hello World!')
        'Helper to `_covariance` and `_variance` which computes a shared scale.'
        c0 = self.total_concentration[..., array_ops.newaxis]
        return math_ops.sqrt((1.0 + c0 / self.total_count) / (1.0 + c0))

    def _maybe_assert_valid_concentration(self, concentration, validate_args):
        if False:
            for i in range(10):
                print('nop')
        'Checks the validity of the concentration parameter.'
        if not validate_args:
            return concentration
        concentration = distribution_util.embed_check_categorical_event_shape(concentration)
        return control_flow_ops.with_dependencies([check_ops.assert_positive(concentration, message='Concentration parameter must be positive.')], concentration)

    def _maybe_assert_valid_sample(self, counts):
        if False:
            return 10
        'Check counts for proper shape, values, then return tensor version.'
        if not self.validate_args:
            return counts
        counts = distribution_util.embed_check_nonnegative_integer_form(counts)
        return control_flow_ops.with_dependencies([check_ops.assert_equal(self.total_count, math_ops.reduce_sum(counts, -1), message='counts last-dimension must sum to `self.total_count`')], counts)