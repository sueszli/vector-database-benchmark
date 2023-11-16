"""Utilities related to loss functions."""
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor

class ReductionV2(object):
    """Types of loss reduction.

  Contains the following values:

  * `AUTO`: Indicates that the reduction option will be determined by the usage
     context. For almost all cases this defaults to `SUM_OVER_BATCH_SIZE`. When
     used with `tf.distribute.Strategy`, outside of built-in training loops such
     as `tf.keras` `compile` and `fit`, we expect reduction value to be
     `SUM` or `NONE`. Using `AUTO` in that case will raise an error.
  * `NONE`: No **additional** reduction is applied to the output of the wrapped
     loss function. When non-scalar losses are returned to Keras functions like
     `fit`/`evaluate`, the unreduced vector loss is passed to the optimizer
     but the reported loss will be a scalar value.

     Caution: **Verify the shape of the outputs when using** `Reduction.NONE`.
     The builtin loss functions wrapped by the loss classes reduce
     one dimension (`axis=-1`, or `axis` if specified by loss function).
     `Reduction.NONE` just means that no **additional** reduction is applied by
     the class wrapper. For categorical losses with an example input shape of
     `[batch, W, H, n_classes]` the `n_classes` dimension is reduced. For
     pointwise losses your must include a dummy axis so that `[batch, W, H, 1]`
     is reduced to `[batch, W, H]`. Without the dummy axis `[batch, W, H]`
     will be incorrectly reduced to `[batch, W]`.

  * `SUM`: Scalar sum of weighted losses.
  * `SUM_OVER_BATCH_SIZE`: Scalar `SUM` divided by number of elements in losses.
     This reduction type is not supported when used with
     `tf.distribute.Strategy` outside of built-in training loops like `tf.keras`
     `compile`/`fit`.

     You can implement 'SUM_OVER_BATCH_SIZE' using global batch size like:
     ```
     with strategy.scope():
       loss_obj = tf.keras.losses.CategoricalCrossentropy(
           reduction=tf.keras.losses.Reduction.NONE)
       ....
       loss = tf.reduce_sum(loss_obj(labels, predictions)) *
           (1. / global_batch_size)
     ```

  Please see the [custom training guide](
  https://www.tensorflow.org/tutorials/distribute/custom_training) for more
  details on this.
  """
    AUTO = 'auto'
    NONE = 'none'
    SUM = 'sum'
    SUM_OVER_BATCH_SIZE = 'sum_over_batch_size'

    @classmethod
    def all(cls):
        if False:
            print('Hello World!')
        return (cls.AUTO, cls.NONE, cls.SUM, cls.SUM_OVER_BATCH_SIZE)

    @classmethod
    def validate(cls, key):
        if False:
            i = 10
            return i + 15
        if key not in cls.all():
            raise ValueError('Invalid Reduction Key %s.' % key)

def remove_squeezable_dimensions(labels, predictions, expected_rank_diff=0, name=None):
    if False:
        while True:
            i = 10
    "Squeeze last dim if ranks differ from expected by exactly 1.\n\n  In the common case where we expect shapes to match, `expected_rank_diff`\n  defaults to 0, and we squeeze the last dimension of the larger rank if they\n  differ by 1.\n\n  But, for example, if `labels` contains class IDs and `predictions` contains 1\n  probability per class, we expect `predictions` to have 1 more dimension than\n  `labels`, so `expected_rank_diff` would be 1. In this case, we'd squeeze\n  `labels` if `rank(predictions) - rank(labels) == 0`, and\n  `predictions` if `rank(predictions) - rank(labels) == 2`.\n\n  This will use static shape if available. Otherwise, it will add graph\n  operations, which could result in a performance hit.\n\n  Args:\n    labels: Label values, a `Tensor` whose dimensions match `predictions`.\n    predictions: Predicted values, a `Tensor` of arbitrary dimensions.\n    expected_rank_diff: Expected result of `rank(predictions) - rank(labels)`.\n    name: Name of the op.\n\n  Returns:\n    Tuple of `labels` and `predictions`, possibly with last dim squeezed.\n  "
    with backend.name_scope(name or 'remove_squeezable_dimensions'):
        if not isinstance(predictions, ragged_tensor.RaggedTensor):
            predictions = tensor_conversion.convert_to_tensor_v2_with_dispatch(predictions)
        if not isinstance(labels, ragged_tensor.RaggedTensor):
            labels = tensor_conversion.convert_to_tensor_v2_with_dispatch(labels)
        predictions_shape = predictions.shape
        predictions_rank = predictions_shape.ndims
        labels_shape = labels.shape
        labels_rank = labels_shape.ndims
        if labels_rank is not None and predictions_rank is not None:
            rank_diff = predictions_rank - labels_rank
            if rank_diff == expected_rank_diff + 1 and predictions_shape.dims[-1].is_compatible_with(1):
                predictions = array_ops.squeeze(predictions, [-1])
            elif rank_diff == expected_rank_diff - 1 and labels_shape.dims[-1].is_compatible_with(1):
                labels = array_ops.squeeze(labels, [-1])
            return (labels, predictions)
        rank_diff = array_ops.rank(predictions) - array_ops.rank(labels)
        if predictions_rank is None or predictions_shape.dims[-1].is_compatible_with(1):
            predictions = cond.cond(math_ops.equal(expected_rank_diff + 1, rank_diff), lambda : array_ops.squeeze(predictions, [-1]), lambda : predictions)
        if labels_rank is None or labels_shape.dims[-1].is_compatible_with(1):
            labels = cond.cond(math_ops.equal(expected_rank_diff - 1, rank_diff), lambda : array_ops.squeeze(labels, [-1]), lambda : labels)
        return (labels, predictions)

def squeeze_or_expand_dimensions(y_pred, y_true=None, sample_weight=None):
    if False:
        i = 10
        return i + 15
    'Squeeze or expand last dimension if needed.\n\n  1. Squeezes last dim of `y_pred` or `y_true` if their rank differs by 1\n  (using `remove_squeezable_dimensions`).\n  2. Squeezes or expands last dim of `sample_weight` if its rank differs by 1\n  from the new rank of `y_pred`.\n  If `sample_weight` is scalar, it is kept scalar.\n\n  This will use static shape if available. Otherwise, it will add graph\n  operations, which could result in a performance hit.\n\n  Args:\n    y_pred: Predicted values, a `Tensor` of arbitrary dimensions.\n    y_true: Optional label `Tensor` whose dimensions match `y_pred`.\n    sample_weight: Optional weight scalar or `Tensor` whose dimensions match\n      `y_pred`.\n\n  Returns:\n    Tuple of `y_pred`, `y_true` and `sample_weight`. Each of them possibly has\n    the last dimension squeezed,\n    `sample_weight` could be extended by one dimension.\n    If `sample_weight` is None, (y_pred, y_true) is returned.\n  '
    y_pred_shape = y_pred.shape
    y_pred_rank = y_pred_shape.ndims
    if y_true is not None:
        y_true_shape = y_true.shape
        y_true_rank = y_true_shape.ndims
        if y_true_rank is not None and y_pred_rank is not None:
            if y_pred_rank - y_true_rank != 1 or y_pred_shape[-1] == 1:
                (y_true, y_pred) = remove_squeezable_dimensions(y_true, y_pred)
        else:
            rank_diff = array_ops.rank(y_pred) - array_ops.rank(y_true)
            squeeze_dims = lambda : remove_squeezable_dimensions(y_true, y_pred)
            is_last_dim_1 = math_ops.equal(1, array_ops.shape(y_pred)[-1])
            maybe_squeeze_dims = lambda : cond.cond(is_last_dim_1, squeeze_dims, lambda : (y_true, y_pred))
            (y_true, y_pred) = cond.cond(math_ops.equal(1, rank_diff), maybe_squeeze_dims, squeeze_dims)
    if sample_weight is None:
        return (y_pred, y_true)
    weights_shape = sample_weight.shape
    weights_rank = weights_shape.ndims
    if weights_rank == 0:
        return (y_pred, y_true, sample_weight)
    if y_pred_rank is not None and weights_rank is not None:
        if weights_rank - y_pred_rank == 1:
            sample_weight = array_ops.squeeze(sample_weight, [-1])
        elif y_pred_rank - weights_rank == 1:
            sample_weight = array_ops.expand_dims(sample_weight, [-1])
        return (y_pred, y_true, sample_weight)
    weights_rank_tensor = array_ops.rank(sample_weight)
    rank_diff = weights_rank_tensor - array_ops.rank(y_pred)
    maybe_squeeze_weights = lambda : array_ops.squeeze(sample_weight, [-1])

    def _maybe_expand_weights():
        if False:
            i = 10
            return i + 15
        expand_weights = lambda : array_ops.expand_dims(sample_weight, [-1])
        return cond.cond(math_ops.equal(rank_diff, -1), expand_weights, lambda : sample_weight)

    def _maybe_adjust_weights():
        if False:
            i = 10
            return i + 15
        return cond.cond(math_ops.equal(rank_diff, 1), maybe_squeeze_weights, _maybe_expand_weights)
    sample_weight = cond.cond(math_ops.equal(weights_rank_tensor, 0), lambda : sample_weight, _maybe_adjust_weights)
    return (y_pred, y_true, sample_weight)

def _safe_mean(losses, num_present):
    if False:
        i = 10
        return i + 15
    'Computes a safe mean of the losses.\n\n  Args:\n    losses: `Tensor` whose elements contain individual loss measurements.\n    num_present: The number of measurable elements in `losses`.\n\n  Returns:\n    A scalar representing the mean of `losses`. If `num_present` is zero,\n      then zero is returned.\n  '
    total_loss = math_ops.reduce_sum(losses)
    return math_ops.div_no_nan(total_loss, num_present, name='value')

def _num_elements(losses):
    if False:
        return 10
    'Computes the number of elements in `losses` tensor.'
    with backend.name_scope('num_elements') as scope:
        return math_ops.cast(array_ops.size(losses, name=scope), dtype=losses.dtype)

def reduce_weighted_loss(weighted_losses, reduction=ReductionV2.SUM_OVER_BATCH_SIZE):
    if False:
        i = 10
        return i + 15
    'Reduces the individual weighted loss measurements.'
    if reduction == ReductionV2.NONE:
        loss = weighted_losses
    else:
        loss = math_ops.reduce_sum(weighted_losses)
        if reduction == ReductionV2.SUM_OVER_BATCH_SIZE:
            loss = _safe_mean(loss, _num_elements(weighted_losses))
    return loss

def compute_weighted_loss(losses, sample_weight=None, reduction=ReductionV2.SUM_OVER_BATCH_SIZE, name=None):
    if False:
        i = 10
        return i + 15
    'Computes the weighted loss.\n\n  Args:\n    losses: `Tensor` of shape `[batch_size, d1, ... dN]`.\n    sample_weight: Optional `Tensor` whose rank is either 0, or the same rank as\n      `losses`, or be broadcastable to `losses`.\n    reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to loss.\n      Default value is `SUM_OVER_BATCH_SIZE`.\n    name: Optional name for the op.\n\n  Raises:\n    ValueError: If the shape of `sample_weight` is not compatible with `losses`.\n\n  Returns:\n    Weighted loss `Tensor` of the same type as `losses`. If `reduction` is\n    `NONE`, this has the same shape as `losses`; otherwise, it is scalar.\n  '
    ReductionV2.validate(reduction)
    if reduction == ReductionV2.AUTO:
        reduction = ReductionV2.SUM_OVER_BATCH_SIZE
    if sample_weight is None:
        sample_weight = 1.0
    with backend.name_scope(name or 'weighted_loss'):
        ops.get_default_graph()._last_loss_reduction = reduction
        if not isinstance(losses, (keras_tensor.KerasTensor, ragged_tensor.RaggedTensor)):
            losses = tensor_conversion.convert_to_tensor_v2_with_dispatch(losses)
        input_dtype = losses.dtype
        if not isinstance(sample_weight, keras_tensor.KerasTensor):
            sample_weight = tensor_conversion.convert_to_tensor_v2_with_dispatch(sample_weight)
        losses = math_ops.cast(losses, 'float32')
        sample_weight = math_ops.cast(sample_weight, 'float32')
        (losses, _, sample_weight) = squeeze_or_expand_dimensions(losses, None, sample_weight)
        weighted_losses = math_ops.multiply(losses, sample_weight)
        loss = reduce_weighted_loss(weighted_losses, reduction)
        loss = math_ops.cast(loss, input_dtype)
        return loss

def scale_loss_for_distribution(loss_value):
    if False:
        i = 10
        return i + 15
    'Scales and returns the given loss value by the number of replicas.'
    num_replicas = distribute_lib.get_strategy().num_replicas_in_sync
    if num_replicas > 1:
        loss_value *= 1.0 / num_replicas
    return loss_value

def cast_losses_to_common_dtype(losses):
    if False:
        for i in range(10):
            print('nop')
    'Cast a list of losses to a common dtype.\n\n  If any loss is floating-point, they will all be casted to the most-precise\n  floating-point loss. Otherwise the losses are not casted. We also skip casting\n  losses if there are any complex losses.\n\n  Args:\n    losses: A list of losses.\n\n  Returns:\n    `losses`, but they have been casted to a common dtype.\n  '
    highest_float = None
    for loss in losses:
        if loss.dtype.is_floating:
            if highest_float is None or loss.dtype.size > highest_float.size:
                highest_float = loss.dtype
            elif {loss.dtype, highest_float} == {'bfloat16', 'float16'}:
                highest_float = 'float32'
        if loss.dtype.is_complex:
            return losses
    if highest_float:
        losses = [math_ops.cast(loss, highest_float) for loss in losses]
    return losses