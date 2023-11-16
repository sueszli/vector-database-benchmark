"""Implementation of Loss operations for use in neural networks."""
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.losses import util
from tensorflow.python.util import dispatch
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export

@tf_export(v1=['losses.Reduction'])
class Reduction:
    """Types of loss reduction.

  Contains the following values:

  * `NONE`: Un-reduced weighted losses with the same shape as input.
  * `SUM`: Scalar sum of weighted losses.
  * `MEAN`: Scalar `SUM` divided by sum of weights. DEPRECATED.
  * `SUM_OVER_BATCH_SIZE`: Scalar `SUM` divided by number of elements in losses.
  * `SUM_OVER_NONZERO_WEIGHTS`: Scalar `SUM` divided by number of non-zero
     weights. DEPRECATED.
  * `SUM_BY_NONZERO_WEIGHTS`: Same as `SUM_OVER_NONZERO_WEIGHTS`. DEPRECATED.
  """
    NONE = 'none'
    SUM = 'weighted_sum'
    SUM_OVER_BATCH_SIZE = 'weighted_sum_over_batch_size'
    MEAN = 'weighted_mean'
    SUM_BY_NONZERO_WEIGHTS = 'weighted_sum_by_nonzero_weights'
    SUM_OVER_NONZERO_WEIGHTS = SUM_BY_NONZERO_WEIGHTS

    @classmethod
    def all(cls):
        if False:
            print('Hello World!')
        return (cls.NONE, cls.SUM, cls.MEAN, cls.SUM_OVER_BATCH_SIZE, cls.SUM_OVER_NONZERO_WEIGHTS, cls.SUM_BY_NONZERO_WEIGHTS)

    @classmethod
    def validate(cls, key):
        if False:
            for i in range(10):
                print('nop')
        if key not in cls.all():
            raise ValueError(f'Invalid Reduction Key {key}. Key should be one of {cls.all()}.')

def _safe_mean(losses, num_present):
    if False:
        while True:
            i = 10
    'Computes a safe mean of the losses.\n\n  Args:\n    losses: `Tensor` whose elements contain individual loss measurements.\n    num_present: The number of measurable elements in `losses`.\n\n  Returns:\n    A scalar representing the mean of `losses`. If `num_present` is zero,\n      then zero is returned.\n  '
    total_loss = math_ops.reduce_sum(losses)
    return math_ops.div_no_nan(total_loss, num_present, name='value')

def _num_present(losses, weights, per_batch=False):
    if False:
        for i in range(10):
            print('nop')
    'Computes the number of elements in the loss function induced by `weights`.\n\n  A given weights tensor induces different numbers of usable elements in the\n  `losses` tensor. The `weights` tensor is broadcast across `losses` for all\n  possible dimensions. For example, if `losses` is a tensor of dimension\n  `[4, 5, 6, 3]` and `weights` is a tensor of shape `[4, 5]`, then `weights` is,\n  in effect, tiled to match the shape of `losses`. Following this effective\n  tile, the total number of present elements is the number of non-zero weights.\n\n  Args:\n    losses: `Tensor` of shape `[batch_size, d1, ... dN]`.\n    weights: `Tensor` of shape `[]`, `[batch_size]` or\n      `[batch_size, d1, ... dK]`, where K < N.\n    per_batch: Whether to return the number of elements per batch or as a sum\n      total.\n\n  Returns:\n    The number of present (non-zero) elements in the losses tensor. If\n      `per_batch` is `True`, the value is returned as a tensor of size\n      `[batch_size]`. Otherwise, a single scalar tensor is returned.\n  '
    if isinstance(weights, float) and weights != 0.0 or (context.executing_eagerly() and weights._rank() == 0 and (not math_ops.equal(weights, 0.0))):
        return _num_elements(losses)
    with ops.name_scope(None, 'num_present', (losses, weights)) as scope:
        weights = math_ops.cast(weights, dtype=dtypes.float32)
        present = array_ops.where(math_ops.equal(weights, 0.0), array_ops.zeros_like(weights), array_ops.ones_like(weights))
        present = weights_broadcast_ops.broadcast_weights(present, losses)
        if per_batch:
            return math_ops.reduce_sum(present, axis=math_ops.range(1, array_ops.rank(present)), keepdims=True, name=scope)
        return math_ops.reduce_sum(present, name=scope)

def _num_elements(losses):
    if False:
        i = 10
        return i + 15
    'Computes the number of elements in `losses` tensor.'
    with ops.name_scope(None, 'num_elements', values=[losses]) as scope:
        return math_ops.cast(array_ops.size(losses, name=scope), dtype=losses.dtype)

@tf_export(v1=['losses.compute_weighted_loss'])
@dispatch.add_dispatch_support
def compute_weighted_loss(losses, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    if False:
        for i in range(10):
            print('nop')
    'Computes the weighted loss.\n\n  Args:\n    losses: `Tensor` of shape `[batch_size, d1, ... dN]`.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `losses`, and must be broadcastable to `losses` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `losses` dimension).\n    scope: the scope for the operations performed in computing the loss.\n    loss_collection: the loss will be added to these collections.\n    reduction: Type of reduction to apply to loss.\n\n  Returns:\n    Weighted loss `Tensor` of the same type as `losses`. If `reduction` is\n    `NONE`, this has the same shape as `losses`; otherwise, it is scalar.\n\n  Raises:\n    ValueError: If `weights` is `None` or the shape is not compatible with\n      `losses`, or if the number of dimensions (rank) of either `losses` or\n      `weights` is missing.\n\n  Note:\n    When calculating the gradient of a weighted loss contributions from\n    both `losses` and `weights` are considered. If your `weights` depend\n    on some model parameters but you do not want this to affect the loss\n    gradient, you need to apply `tf.stop_gradient` to `weights` before\n    passing them to `compute_weighted_loss`.\n\n  @compatibility(eager)\n  The `loss_collection` argument is ignored when executing eagerly. Consider\n  holding on to the return value or collecting losses via a `tf.keras.Model`.\n  @end_compatibility\n  '
    Reduction.validate(reduction)
    with ops.name_scope(scope, 'weighted_loss', (losses, weights)):
        ops.get_default_graph()._last_loss_reduction = reduction

        def compute_loss(losses, weights, loss_collection, reduction):
            if False:
                for i in range(10):
                    print('nop')
            losses = ops.convert_to_tensor(losses)
            input_dtype = losses.dtype
            losses = math_ops.cast(losses, dtype=dtypes.float32)
            weights = math_ops.cast(weights, dtype=dtypes.float32)
            weighted_losses = math_ops.multiply(losses, weights)
            if reduction == Reduction.NONE:
                loss = weighted_losses
            else:
                loss = math_ops.reduce_sum(weighted_losses)
                if reduction == Reduction.MEAN:
                    loss = _safe_mean(loss, math_ops.reduce_sum(array_ops.ones_like(losses) * weights))
                elif reduction == Reduction.SUM_BY_NONZERO_WEIGHTS or reduction == Reduction.SUM_OVER_NONZERO_WEIGHTS:
                    loss = _safe_mean(loss, _num_present(losses, weights))
                elif reduction == Reduction.SUM_OVER_BATCH_SIZE:
                    loss = _safe_mean(loss, _num_elements(losses))
            loss = math_ops.cast(loss, input_dtype)
            util.add_loss(loss, loss_collection)
            return loss
        if control_flow_ops.get_enclosing_xla_context() is not None:
            return compute_loss(losses, weights, loss_collection, reduction)
        else:
            with ops.control_dependencies((weights_broadcast_ops.assert_broadcastable(weights, losses),)):
                return compute_loss(losses, weights, loss_collection, reduction)

@tf_export(v1=['losses.absolute_difference'])
@dispatch.add_dispatch_support
def absolute_difference(labels, predictions, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    if False:
        print('Hello World!')
    "Adds an Absolute Difference loss to the training procedure.\n\n  `weights` acts as a coefficient for the loss. If a scalar is provided, then\n  the loss is simply scaled by the given value. If `weights` is a `Tensor` of\n  shape `[batch_size]`, then the total loss for each sample of the batch is\n  rescaled by the corresponding element in the `weights` vector. If the shape of\n  `weights` matches the shape of `predictions`, then the loss of each\n  measurable element of `predictions` is scaled by the corresponding value of\n  `weights`.\n\n  Args:\n    labels: The ground truth output tensor, same dimensions as 'predictions'.\n    predictions: The predicted outputs.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `losses` dimension).\n    scope: The scope for the operations performed in computing the loss.\n    loss_collection: collection to which this loss will be added.\n    reduction: Type of reduction to apply to loss.\n\n  Returns:\n    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same\n    shape as `labels`; otherwise, it is scalar.\n\n  Raises:\n    ValueError: If the shape of `predictions` doesn't match that of\n      `labels` or if the shape of `weights` is invalid or if `labels`\n      or `predictions` is None.\n\n  @compatibility(eager)\n  The `loss_collection` argument is ignored when executing eagerly. Consider\n  holding on to the return value or collecting losses via a `tf.keras.Model`.\n  @end_compatibility\n  "
    if labels is None:
        raise ValueError('Argument `labels` must not be None.')
    if predictions is None:
        raise ValueError('Argument `predictions` must not be None.')
    with ops.name_scope(scope, 'absolute_difference', (predictions, labels, weights)) as scope:
        predictions = math_ops.cast(predictions, dtype=dtypes.float32)
        labels = math_ops.cast(labels, dtype=dtypes.float32)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        losses = math_ops.abs(math_ops.subtract(predictions, labels))
        return compute_weighted_loss(losses, weights, scope, loss_collection, reduction=reduction)

@tf_export(v1=['losses.cosine_distance'])
@dispatch.add_dispatch_support
@deprecated_args(None, 'dim is deprecated, use axis instead', 'dim')
def cosine_distance(labels, predictions, axis=None, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS, dim=None):
    if False:
        return 10
    "Adds a cosine-distance loss to the training procedure.\n\n  Note that the function assumes that `predictions` and `labels` are already\n  unit-normalized.\n\n  Args:\n    labels: `Tensor` whose shape matches 'predictions'\n    predictions: An arbitrary matrix.\n    axis: The dimension along which the cosine distance is computed.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `losses` dimension).\n    scope: The scope for the operations performed in computing the loss.\n    loss_collection: collection to which this loss will be added.\n    reduction: Type of reduction to apply to loss.\n    dim: The old (deprecated) name for `axis`.\n\n  Returns:\n    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same\n    shape as `labels`; otherwise, it is scalar.\n\n  Raises:\n    ValueError: If `predictions` shape doesn't match `labels` shape, or\n      `axis`, `labels`, `predictions` or `weights` is `None`.\n\n  @compatibility(eager)\n  The `loss_collection` argument is ignored when executing eagerly. Consider\n  holding on to the return value or collecting losses via a `tf.keras.Model`.\n  @end_compatibility\n  "
    axis = deprecated_argument_lookup('axis', axis, 'dim', dim)
    if axis is None:
        raise ValueError('You must specify argument `axis`.')
    if labels is None:
        raise ValueError('Argument `labels` must not be None.')
    if predictions is None:
        raise ValueError('Argument `predictions` must not be None.')
    with ops.name_scope(scope, 'cosine_distance_loss', (predictions, labels, weights)) as scope:
        predictions = math_ops.cast(predictions, dtype=dtypes.float32)
        labels = math_ops.cast(labels, dtype=dtypes.float32)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        radial_diffs = math_ops.multiply(predictions, labels)
        losses = 1 - math_ops.reduce_sum(radial_diffs, axis=(axis,), keepdims=True)
        return compute_weighted_loss(losses, weights, scope, loss_collection, reduction=reduction)

@tf_export(v1=['losses.hinge_loss'])
@dispatch.add_dispatch_support
def hinge_loss(labels, logits, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    if False:
        i = 10
        return i + 15
    "Adds a hinge loss to the training procedure.\n\n  Args:\n    labels: The ground truth output tensor. Its shape should match the shape of\n      logits. The values of the tensor are expected to be 0.0 or 1.0. Internally\n      the {0,1} labels are converted to {-1,1} when calculating the hinge loss.\n    logits: The logits, a float tensor. Note that logits are assumed to be\n      unbounded and 0-centered. A value > 0 (resp. < 0) is considered a positive\n      (resp. negative) binary prediction.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `losses` dimension).\n    scope: The scope for the operations performed in computing the loss.\n    loss_collection: collection to which the loss will be added.\n    reduction: Type of reduction to apply to loss.\n\n  Returns:\n    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same\n    shape as `labels`; otherwise, it is scalar.\n\n  Raises:\n    ValueError: If the shapes of `logits` and `labels` don't match or\n      if `labels` or `logits` is None.\n\n  @compatibility(eager)\n  The `loss_collection` argument is ignored when executing eagerly. Consider\n  holding on to the return value or collecting losses via a `tf.keras.Model`.\n  @end_compatibility\n  "
    if labels is None:
        raise ValueError('Argument `labels` must not be None.')
    if logits is None:
        raise ValueError('Argument `logits` must not be None.')
    with ops.name_scope(scope, 'hinge_loss', (logits, labels, weights)) as scope:
        logits = math_ops.cast(logits, dtype=dtypes.float32)
        labels = math_ops.cast(labels, dtype=dtypes.float32)
        logits.get_shape().assert_is_compatible_with(labels.get_shape())
        all_ones = array_ops.ones_like(labels)
        labels = math_ops.subtract(2 * labels, all_ones)
        losses = nn_ops.relu(math_ops.subtract(all_ones, math_ops.multiply(labels, logits)))
        return compute_weighted_loss(losses, weights, scope, loss_collection, reduction=reduction)

@tf_export(v1=['losses.huber_loss'])
@dispatch.add_dispatch_support
def huber_loss(labels, predictions, weights=1.0, delta=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    if False:
        return 10
    "Adds a [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss) term to the training procedure.\n\n  For each value x in `error=labels-predictions`, the following is calculated:\n\n  ```\n    0.5 * x^2                  if |x| <= d\n    0.5 * d^2 + d * (|x| - d)  if |x| > d\n  ```\n\n  where d is `delta`.\n\n  `weights` acts as a coefficient for the loss. If a scalar is provided, then\n  the loss is simply scaled by the given value. If `weights` is a tensor of size\n  `[batch_size]`, then the total loss for each sample of the batch is rescaled\n  by the corresponding element in the `weights` vector. If the shape of\n  `weights` matches the shape of `predictions`, then the loss of each\n  measurable element of `predictions` is scaled by the corresponding value of\n  `weights`.\n\n  Args:\n    labels: The ground truth output tensor, same dimensions as 'predictions'.\n    predictions: The predicted outputs.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `losses` dimension).\n    delta: `float`, the point where the huber loss function changes from a\n      quadratic to linear.\n    scope: The scope for the operations performed in computing the loss.\n    loss_collection: collection to which the loss will be added.\n    reduction: Type of reduction to apply to loss.\n\n  Returns:\n    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same\n    shape as `labels`; otherwise, it is scalar.\n\n  Raises:\n    ValueError: If the shape of `predictions` doesn't match that of `labels` or\n      if the shape of `weights` is invalid.  Also if `labels` or\n     `predictions` is None.\n\n  @compatibility(eager)\n  The `loss_collection` argument is ignored when executing eagerly. Consider\n  holding on to the return value or collecting losses via a `tf.keras.Model`.\n  @end_compatibility\n  "
    if labels is None:
        raise ValueError('Argument `labels` must not be None.')
    if predictions is None:
        raise ValueError('Argument `predictions` must not be None.')
    with ops.name_scope(scope, 'huber_loss', (predictions, labels, weights)) as scope:
        predictions = math_ops.cast(predictions, dtype=dtypes.float32)
        labels = math_ops.cast(labels, dtype=dtypes.float32)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        error = math_ops.subtract(predictions, labels)
        abs_error = math_ops.abs(error)
        quadratic = math_ops.minimum(abs_error, delta)
        linear = math_ops.subtract(abs_error, quadratic)
        losses = math_ops.add(math_ops.multiply(ops.convert_to_tensor(0.5, dtype=quadratic.dtype), math_ops.multiply(quadratic, quadratic)), math_ops.multiply(delta, linear))
        return compute_weighted_loss(losses, weights, scope, loss_collection, reduction=reduction)

@tf_export(v1=['losses.log_loss'])
@dispatch.add_dispatch_support
def log_loss(labels, predictions, weights=1.0, epsilon=1e-07, scope=None, loss_collection=ops.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    if False:
        while True:
            i = 10
    "Adds a Log Loss term to the training procedure.\n\n  `weights` acts as a coefficient for the loss. If a scalar is provided, then\n  the loss is simply scaled by the given value. If `weights` is a tensor of size\n  `[batch_size]`, then the total loss for each sample of the batch is rescaled\n  by the corresponding element in the `weights` vector. If the shape of\n  `weights` matches the shape of `predictions`, then the loss of each\n  measurable element of `predictions` is scaled by the corresponding value of\n  `weights`.\n\n  Args:\n    labels: The ground truth output tensor, same dimensions as 'predictions'.\n    predictions: The predicted outputs.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `losses` dimension).\n    epsilon: A small increment to add to avoid taking a log of zero.\n    scope: The scope for the operations performed in computing the loss.\n    loss_collection: collection to which the loss will be added.\n    reduction: Type of reduction to apply to loss.\n\n  Returns:\n    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same\n    shape as `labels`; otherwise, it is scalar.\n\n  Raises:\n    ValueError: If the shape of `predictions` doesn't match that of `labels` or\n      if the shape of `weights` is invalid.  Also if `labels` or `predictions`\n      is None.\n\n  @compatibility(eager)\n  The `loss_collection` argument is ignored when executing eagerly. Consider\n  holding on to the return value or collecting losses via a `tf.keras.Model`.\n  @end_compatibility\n  "
    if labels is None:
        raise ValueError('Argument `labels` must not be None.')
    if predictions is None:
        raise ValueError('Argument `predictions` must not be None.')
    with ops.name_scope(scope, 'log_loss', (predictions, labels, weights)) as scope:
        predictions = math_ops.cast(predictions, dtype=dtypes.float32)
        labels = math_ops.cast(labels, dtype=dtypes.float32)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        losses = -math_ops.multiply(labels, math_ops.log(predictions + epsilon)) - math_ops.multiply(1 - labels, math_ops.log(1 - predictions + epsilon))
        return compute_weighted_loss(losses, weights, scope, loss_collection, reduction=reduction)

@tf_export(v1=['losses.mean_pairwise_squared_error'])
@dispatch.add_dispatch_support
def mean_pairwise_squared_error(labels, predictions, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES):
    if False:
        while True:
            i = 10
    "Adds a pairwise-errors-squared loss to the training procedure.\n\n  Unlike `mean_squared_error`, which is a measure of the differences between\n  corresponding elements of `predictions` and `labels`,\n  `mean_pairwise_squared_error` is a measure of the differences between pairs of\n  corresponding elements of `predictions` and `labels`.\n\n  For example, if `labels`=[a, b, c] and `predictions`=[x, y, z], there are\n  three pairs of differences are summed to compute the loss:\n    loss = [ ((a-b) - (x-y)).^2 + ((a-c) - (x-z)).^2 + ((b-c) - (y-z)).^2 ] / 3\n\n  Note that since the inputs are of shape `[batch_size, d0, ... dN]`, the\n  corresponding pairs are computed within each batch sample but not across\n  samples within a batch. For example, if `predictions` represents a batch of\n  16 grayscale images of dimension [batch_size, 100, 200], then the set of pairs\n  is drawn from each image, but not across images.\n\n  `weights` acts as a coefficient for the loss. If a scalar is provided, then\n  the loss is simply scaled by the given value. If `weights` is a tensor of size\n  `[batch_size]`, then the total loss for each sample of the batch is rescaled\n  by the corresponding element in the `weights` vector.\n\n  Args:\n    labels: The ground truth output tensor, whose shape must match the shape of\n      `predictions`.\n    predictions: The predicted outputs, a tensor of size\n      `[batch_size, d0, .. dN]` where N+1 is the total number of dimensions in\n      `predictions`.\n    weights: Coefficients for the loss a scalar, a tensor of shape\n      `[batch_size]` or a tensor whose shape matches `predictions`.\n    scope: The scope for the operations performed in computing the loss.\n    loss_collection: collection to which the loss will be added.\n\n  Returns:\n    A scalar `Tensor` that returns the weighted loss.\n\n  Raises:\n    ValueError: If the shape of `predictions` doesn't match that of `labels` or\n      if the shape of `weights` is invalid.  Also if `labels` or `predictions`\n      is None.\n\n  @compatibility(eager)\n  The `loss_collection` argument is ignored when executing eagerly. Consider\n  holding on to the return value or collecting losses via a `tf.keras.Model`.\n  @end_compatibility\n  "
    if labels is None:
        raise ValueError('Argument `labels` must not be None.')
    if predictions is None:
        raise ValueError('Argument `predictions` must not be None.')
    with ops.name_scope(scope, 'mean_pairwise_squared_error', (predictions, labels, weights)) as scope:
        weights = math_ops.cast(weights, dtype=dtypes.float32)
        labels = math_ops.cast(labels, dtype=dtypes.float32)

        def compute_loss(labels, predictions, weights, loss_collection):
            if False:
                i = 10
                return i + 15
            predictions = math_ops.cast(predictions, dtype=dtypes.float32)
            predictions.get_shape().assert_is_compatible_with(labels.get_shape())
            diffs = math_ops.subtract(predictions, labels)
            axis = math_ops.range(1, array_ops.rank(diffs))
            sum_squares_diff_per_batch = math_ops.reduce_sum(math_ops.square(diffs), axis=axis, keepdims=True)
            num_present_per_batch = _num_present(diffs, weights, per_batch=True)
            term1 = 2.0 * math_ops.div_no_nan(sum_squares_diff_per_batch, math_ops.maximum(num_present_per_batch - 1, 0), name='value')
            sum_diff = math_ops.reduce_sum(diffs, axis=axis, keepdims=True)
            term2 = 2.0 * math_ops.div_no_nan(math_ops.square(sum_diff), math_ops.maximum(math_ops.multiply(num_present_per_batch, num_present_per_batch - 1), 0), name='value')
            weighted_losses = math_ops.multiply(term1 - term2, weights)
            loss = math_ops.reduce_sum(weighted_losses)
            mean_loss = array_ops.where(math_ops.reduce_sum(num_present_per_batch) > 0, loss, array_ops.zeros_like(loss), name='value')
            util.add_loss(mean_loss, loss_collection)
            return mean_loss
        if control_flow_ops.get_enclosing_xla_context() is not None:
            return compute_loss(labels, predictions, weights, loss_collection)
        else:
            with ops.control_dependencies((weights_broadcast_ops.assert_broadcastable(weights, labels),)):
                return compute_loss(labels, predictions, weights, loss_collection)

@tf_export(v1=['losses.mean_squared_error'])
@dispatch.add_dispatch_support
def mean_squared_error(labels, predictions, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    if False:
        while True:
            i = 10
    "Adds a Sum-of-Squares loss to the training procedure.\n\n  `weights` acts as a coefficient for the loss. If a scalar is provided, then\n  the loss is simply scaled by the given value. If `weights` is a tensor of size\n  `[batch_size]`, then the total loss for each sample of the batch is rescaled\n  by the corresponding element in the `weights` vector. If the shape of\n  `weights` matches the shape of `predictions`, then the loss of each\n  measurable element of `predictions` is scaled by the corresponding value of\n  `weights`.\n\n  Args:\n    labels: The ground truth output tensor, same dimensions as 'predictions'.\n    predictions: The predicted outputs.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `losses` dimension).\n    scope: The scope for the operations performed in computing the loss.\n    loss_collection: collection to which the loss will be added.\n    reduction: Type of reduction to apply to loss.\n\n  Returns:\n    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same\n    shape as `labels`; otherwise, it is scalar.\n\n  Raises:\n    ValueError: If the shape of `predictions` doesn't match that of `labels` or\n      if the shape of `weights` is invalid.  Also if `labels` or `predictions`\n      is None.\n\n  @compatibility(TF2)\n\n  `tf.compat.v1.losses.mean_squared_error` is mostly compatible with eager\n  execution and `tf.function`. But, the `loss_collection` argument is\n  ignored when executing eagerly and no loss will be written to the loss\n  collections. You will need to either hold on to the return value manually\n  or rely on `tf.keras.Model` loss tracking.\n\n\n  To switch to native TF2 style, instantiate the\n   `tf.keras.losses.MeanSquaredError` class and call the object instead.\n\n\n  #### Structural Mapping to Native TF2\n\n  Before:\n\n  ```python\n  loss = tf.compat.v1.losses.mean_squared_error(\n    labels=labels,\n    predictions=predictions,\n    weights=weights,\n    reduction=reduction)\n  ```\n\n  After:\n\n  ```python\n  loss_fn = tf.keras.losses.MeanSquaredError(\n    reduction=reduction)\n  loss = loss_fn(\n    y_true=labels,\n    y_pred=predictions,\n    sample_weight=weights)\n  ```\n\n  #### How to Map Arguments\n\n  | TF1 Arg Name          | TF2 Arg Name     | Note                       |\n  | :-------------------- | :--------------- | :------------------------- |\n  | `labels`              | `y_true`         | In `__call__()` method     |\n  | `predictions`         | `y_pred`         | In `__call__()` method     |\n  | `weights`             | `sample_weight`  | In `__call__()` method.    |\n  : : : The shape requirements for `sample_weight` is different from      :\n  : : : `weights`. Please check the [argument definition][api_docs] for   :\n  : : : details.                                                          :\n  | `scope`               | Not supported    | -                          |\n  | `loss_collection`     | Not supported    | Losses should be tracked   |\n  : : : explicitly or with Keras APIs, for example, [add_loss][add_loss], :\n  : : : instead of via collections                                        :\n  | `reduction`           | `reduction`      | In constructor. Value of   |\n  : : : `tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE`,              :\n  : : : `tf.compat.v1.losses.Reduction.SUM`,                              :\n  : : : `tf.compat.v1.losses.Reduction.NONE` in                           :\n  : : : `tf.compat.v1.losses.softmax_cross_entropy` correspond to         :\n  : : : `tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE`,                  :\n  : : : `tf.keras.losses.Reduction.SUM`,                                  :\n  : : : `tf.keras.losses.Reduction.NONE`, respectively. If you            :\n  : : : used other value for `reduction`, including the default value     :\n  : : :  `tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS`, there is :\n  : : : no directly corresponding value. Please modify the loss           :\n  : : : implementation manually.                                          :\n\n  [add_loss]:https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_loss\n  [api_docs]:https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredError#__call__\n\n\n  #### Before & After Usage Example\n\n  Before:\n\n  >>> y_true = [1, 2, 3]\n  >>> y_pred = [1, 3, 5]\n  >>> weights = [0, 1, 0.25]\n  >>> # samples with zero-weight are excluded from calculation when `reduction`\n  >>> # argument is set to default value `Reduction.SUM_BY_NONZERO_WEIGHTS`\n  >>> tf.compat.v1.losses.mean_squared_error(\n  ...    labels=y_true,\n  ...    predictions=y_pred,\n  ...    weights=weights).numpy()\n  1.0\n\n  >>> tf.compat.v1.losses.mean_squared_error(\n  ...    labels=y_true,\n  ...    predictions=y_pred,\n  ...    weights=weights,\n  ...    reduction=tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE).numpy()\n  0.66667\n\n  After:\n\n  >>> y_true = [[1.0], [2.0], [3.0]]\n  >>> y_pred = [[1.0], [3.0], [5.0]]\n  >>> weights = [1, 1, 0.25]\n  >>> mse = tf.keras.losses.MeanSquaredError(\n  ...    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)\n  >>> mse(y_true=y_true, y_pred=y_pred, sample_weight=weights).numpy()\n  0.66667\n\n  @end_compatibility\n  "
    if labels is None:
        raise ValueError('Argument `labels` must not be None.')
    if predictions is None:
        raise ValueError('Argument `predictions` must not be None.')
    with ops.name_scope(scope, 'mean_squared_error', (predictions, labels, weights)) as scope:
        predictions = math_ops.cast(predictions, dtype=dtypes.float32)
        labels = math_ops.cast(labels, dtype=dtypes.float32)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        losses = math_ops.squared_difference(predictions, labels)
        return compute_weighted_loss(losses, weights, scope, loss_collection, reduction=reduction)

@tf_export(v1=['losses.sigmoid_cross_entropy'])
@dispatch.add_dispatch_support
def sigmoid_cross_entropy(multi_class_labels, logits, weights=1.0, label_smoothing=0, scope=None, loss_collection=ops.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    if False:
        return 10
    "Creates a cross-entropy loss using tf.nn.sigmoid_cross_entropy_with_logits.\n\n  `weights` acts as a coefficient for the loss. If a scalar is provided,\n  then the loss is simply scaled by the given value. If `weights` is a\n  tensor of shape `[batch_size]`, then the loss weights apply to each\n  corresponding sample.\n\n  If `label_smoothing` is nonzero, smooth the labels towards 1/2:\n\n      new_multiclass_labels = multiclass_labels * (1 - label_smoothing)\n                              + 0.5 * label_smoothing\n\n  Args:\n    multi_class_labels: `[batch_size, num_classes]` target integer labels in\n      `{0, 1}`.\n    logits: Float `[batch_size, num_classes]` logits outputs of the network.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n    `multi_class_labels`, and must be broadcastable to `multi_class_labels`\n    (i.e., all dimensions must be either `1`, or the same as the\n    corresponding `losses` dimension).\n    label_smoothing: If greater than `0` then smooth the labels.\n    scope: The scope for the operations performed in computing the loss.\n    loss_collection: collection to which the loss will be added.\n    reduction: Type of reduction to apply to loss.\n\n  Returns:\n    Weighted loss `Tensor` of the same type as `logits`. If `reduction` is\n    `NONE`, this has the same shape as `logits`; otherwise, it is scalar.\n\n  Raises:\n    ValueError: If the shape of `logits` doesn't match that of\n      `multi_class_labels` or if the shape of `weights` is invalid, or if\n      `weights` is None.  Also if `multi_class_labels` or `logits` is None.\n\n  @compatibility(eager)\n  The `loss_collection` argument is ignored when executing eagerly. Consider\n  holding on to the return value or collecting losses via a `tf.keras.Model`.\n  @end_compatibility\n  "
    if multi_class_labels is None:
        raise ValueError('Argument `multi_class_labels` must not be None.')
    if logits is None:
        raise ValueError('Argument `logits` must not be None.')
    with ops.name_scope(scope, 'sigmoid_cross_entropy_loss', (logits, multi_class_labels, weights)) as scope:
        logits = ops.convert_to_tensor(logits)
        multi_class_labels = math_ops.cast(multi_class_labels, logits.dtype)
        logits.get_shape().assert_is_compatible_with(multi_class_labels.get_shape())
        if label_smoothing > 0:
            multi_class_labels = multi_class_labels * (1 - label_smoothing) + 0.5 * label_smoothing
        losses = nn.sigmoid_cross_entropy_with_logits(labels=multi_class_labels, logits=logits, name='xentropy')
        return compute_weighted_loss(losses, weights, scope, loss_collection, reduction=reduction)

@tf_export(v1=['losses.softmax_cross_entropy'])
@dispatch.add_dispatch_support
def softmax_cross_entropy(onehot_labels, logits, weights=1.0, label_smoothing=0, scope=None, loss_collection=ops.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    if False:
        while True:
            i = 10
    "Creates a cross-entropy loss using tf.nn.softmax_cross_entropy_with_logits_v2.\n\n  `weights` acts as a coefficient for the loss. If a scalar is provided,\n  then the loss is simply scaled by the given value. If `weights` is a\n  tensor of shape `[batch_size]`, then the loss weights apply to each\n  corresponding sample.\n\n  If `label_smoothing` is nonzero, smooth the labels towards 1/num_classes:\n      new_onehot_labels = onehot_labels * (1 - label_smoothing)\n                          + label_smoothing / num_classes\n\n  Note that `onehot_labels` and `logits` must have the same shape,\n  e.g. `[batch_size, num_classes]`. The shape of `weights` must be\n  broadcastable to loss, whose shape is decided by the shape of `logits`.\n  In case the shape of `logits` is `[batch_size, num_classes]`, loss is\n  a `Tensor` of shape `[batch_size]`.\n\n  Args:\n    onehot_labels: One-hot-encoded labels.\n    logits: Logits outputs of the network.\n    weights: Optional `Tensor` that is broadcastable to loss.\n    label_smoothing: If greater than 0 then smooth the labels.\n    scope: the scope for the operations performed in computing the loss.\n    loss_collection: collection to which the loss will be added.\n    reduction: Type of reduction to apply to loss.\n\n  Returns:\n    Weighted loss `Tensor` of the same type as `logits`. If `reduction` is\n    `NONE`, this has shape `[batch_size]`; otherwise, it is scalar.\n\n  Raises:\n    ValueError: If the shape of `logits` doesn't match that of `onehot_labels`\n      or if the shape of `weights` is invalid or if `weights` is None.  Also if\n      `onehot_labels` or `logits` is None.\n\n  @compatibility(TF2)\n\n  `tf.compat.v1.losses.softmax_cross_entropy` is mostly compatible with eager\n  execution and `tf.function`. But, the `loss_collection` argument is\n  ignored when executing eagerly and no loss will be written to the loss\n  collections. You will need to either hold on to the return value manually\n  or rely on `tf.keras.Model` loss tracking.\n\n\n  To switch to native TF2 style, instantiate the\n   `tf.keras.losses.CategoricalCrossentropy` class with `from_logits` set\n  as `True` and call the object instead.\n\n\n  #### Structural Mapping to Native TF2\n\n  Before:\n\n  ```python\n  loss = tf.compat.v1.losses.softmax_cross_entropy(\n    onehot_labels=onehot_labels,\n    logits=logits,\n    weights=weights,\n    label_smoothing=smoothing)\n  ```\n\n  After:\n\n  ```python\n  loss_fn = tf.keras.losses.CategoricalCrossentropy(\n    from_logits=True,\n    label_smoothing=smoothing)\n  loss = loss_fn(\n    y_true=onehot_labels,\n    y_pred=logits,\n    sample_weight=weights)\n  ```\n\n  #### How to Map Arguments\n\n  | TF1 Arg Name          | TF2 Arg Name     | Note                       |\n  | :-------------------- | :--------------- | :------------------------- |\n  |  -                    | `from_logits`    | Set `from_logits` as True  |\n  :                       :                  : to have identical behavior :\n  | `onehot_labels`       | `y_true`         | In `__call__()` method     |\n  | `logits`              | `y_pred`         | In `__call__()` method     |\n  | `weights`             | `sample_weight`  | In `__call__()` method     |\n  | `label_smoothing`     | `label_smoothing`| In constructor             |\n  | `scope`               | Not supported    | -                          |\n  | `loss_collection`     | Not supported    | Losses should be tracked   |\n  :                       :                  : explicitly or with Keras   :\n  :                       :                  : APIs, for example,         :\n  :                       :                  : [add_loss][add_loss],      :\n  :                       :                  : instead of via collections :\n  | `reduction`           | `reduction`      | In constructor. Value of   |\n  : : : `tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE`,              :\n  : : : `tf.compat.v1.losses.Reduction.SUM`,                              :\n  : : : `tf.compat.v1.losses.Reduction.NONE` in                           :\n  : : : `tf.compat.v1.losses.softmax_cross_entropy` correspond to         :\n  : : : `tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE`,                  :\n  : : : `tf.keras.losses.Reduction.SUM`,                                  :\n  : : : `tf.keras.losses.Reduction.NONE`, respectively. If you            :\n  : : : used other value for `reduction`, including the default value     :\n  : : :  `tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS`, there is :\n  : : : no directly corresponding value. Please modify the loss           :\n  : : : implementation manually.                                          :\n\n  [add_loss]:https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_loss\n\n\n  #### Before & After Usage Example\n\n  Before:\n\n  >>> y_true = [[0, 1, 0], [0, 0, 1]]\n  >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]\n  >>> weights = [0.3, 0.7]\n  >>> smoothing = 0.2\n  >>> tf.compat.v1.losses.softmax_cross_entropy(y_true, y_pred, weights=weights,\n  ...   label_smoothing=smoothing).numpy()\n  0.57618\n\n  After:\n\n  >>> cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True,\n  ...   label_smoothing=smoothing)\n  >>> cce(y_true, y_pred, sample_weight=weights).numpy()\n  0.57618\n\n  @end_compatibility\n  "
    if onehot_labels is None:
        raise ValueError('Argument `onehot_labels` must not be None.')
    if logits is None:
        raise ValueError('Argument `logits` must not be None.')
    with ops.name_scope(scope, 'softmax_cross_entropy_loss', (logits, onehot_labels, weights)) as scope:
        logits = ops.convert_to_tensor(logits)
        onehot_labels = math_ops.cast(onehot_labels, logits.dtype)
        logits.get_shape().assert_is_compatible_with(onehot_labels.get_shape())
        if label_smoothing > 0:
            num_classes = math_ops.cast(array_ops.shape(onehot_labels)[-1], logits.dtype)
            smooth_positives = 1.0 - label_smoothing
            smooth_negatives = label_smoothing / num_classes
            onehot_labels = onehot_labels * smooth_positives + smooth_negatives
        onehot_labels = array_ops.stop_gradient(onehot_labels, name='labels_stop_gradient')
        losses = nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels, logits=logits, name='xentropy')
        return compute_weighted_loss(losses, weights, scope, loss_collection, reduction=reduction)

def _remove_squeezable_dimensions(labels, predictions, weights=None, expected_rank_diff=0):
    if False:
        for i in range(10):
            print('nop')
    "Internal version of _remove_squeezable_dimensions which handles weights.\n\n  Squeezes `predictions` and `labels` if their ranks differ from expected by\n  exactly 1.\n  Squeezes `weights` if its rank is 1 more than the new rank of `predictions`\n\n  This will use static shape if available. Otherwise, it will add graph\n  operations, which could result in a performance hit.\n\n  Args:\n    labels: Label values, a `Tensor` whose dimensions match `predictions`.\n    predictions: Predicted values, a `Tensor` of arbitrary dimensions.\n    weights: Optional weight `Tensor`. It will be squeezed if it's not scalar,\n      and its rank is 1 more than the new rank of `labels`.\n    expected_rank_diff: Expected result of `rank(predictions) - rank(labels)`.\n\n  Returns:\n    Tuple of `predictions`, `labels` and `weights`, possibly with the last\n    dimension squeezed.\n  "
    (labels, predictions) = confusion_matrix.remove_squeezable_dimensions(labels, predictions, expected_rank_diff=expected_rank_diff)
    if weights is not None:
        weights = ops.convert_to_tensor(weights)
        labels_rank = labels.get_shape().ndims
        weights_shape = weights.get_shape()
        weights_rank = weights_shape.ndims
        if labels_rank is not None and weights_rank is not None:
            rank_diff = weights_rank - labels_rank
            if rank_diff == 1:
                weights = array_ops.squeeze(weights, [-1])
            return (labels, predictions, weights)
        rank_diff = array_ops.rank(weights) - array_ops.rank(labels)
        if weights_rank is None or (weights_rank > 0 and weights_shape.dims[-1].is_compatible_with(1)):
            weights = cond.cond(math_ops.equal(1, rank_diff), lambda : array_ops.squeeze(weights, [-1]), lambda : weights)
    return (labels, predictions, weights)

@tf_export(v1=['losses.sparse_softmax_cross_entropy'])
@dispatch.add_dispatch_support
def sparse_softmax_cross_entropy(labels, logits, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    if False:
        print('Hello World!')
    'Cross-entropy loss using `tf.nn.sparse_softmax_cross_entropy_with_logits`.\n\n  `weights` acts as a coefficient for the loss. If a scalar is provided,\n  then the loss is simply scaled by the given value. If `weights` is a\n  tensor of shape `[batch_size]`, then the loss weights apply to each\n  corresponding sample.\n\n  Args:\n    labels: `Tensor` of shape `[d_0, d_1, ..., d_{r-1}]` (where `r` is rank of\n      `labels` and result) and dtype `int32` or `int64`. Each entry in `labels`\n      must be an index in `[0, num_classes)`. Other values will raise an\n      exception when this op is run on CPU, and return `NaN` for corresponding\n      loss and gradient rows on GPU.\n    logits: Unscaled log probabilities of shape\n      `[d_0, d_1, ..., d_{r-1}, num_classes]` and dtype `float16`, `float32` or\n      `float64`.\n    weights: Coefficients for the loss. This must be scalar or broadcastable to\n      `labels` (i.e. same rank and each dimension is either 1 or the same).\n    scope: the scope for the operations performed in computing the loss.\n    loss_collection: collection to which the loss will be added.\n    reduction: Type of reduction to apply to loss.\n\n  Returns:\n    Weighted loss `Tensor` of the same type as `logits`. If `reduction` is\n    `NONE`, this has the same shape as `labels`; otherwise, it is scalar.\n\n  Raises:\n    ValueError: If the shapes of `logits`, `labels`, and `weights` are\n      incompatible, or if any of them are None.\n\n  @compatibility(eager)\n  The `loss_collection` argument is ignored when executing eagerly. Consider\n  holding on to the return value or collecting losses via a `tf.keras.Model`.\n  @end_compatibility\n  '
    if labels is None:
        raise ValueError('Argument `labels` must not be None.')
    if logits is None:
        raise ValueError('Argument `logits` must not be None.')
    with ops.name_scope(scope, 'sparse_softmax_cross_entropy_loss', (logits, labels, weights)) as scope:
        (labels, logits, weights) = _remove_squeezable_dimensions(labels, logits, weights, expected_rank_diff=1)
        losses = nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
        return compute_weighted_loss(losses, weights, scope, loss_collection, reduction=reduction)