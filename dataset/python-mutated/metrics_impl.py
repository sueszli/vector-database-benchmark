"""Implementation of tf.metrics module."""
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sets
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export

def metric_variable(shape, dtype, validate_shape=True, name=None):
    if False:
        return 10
    'Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES)` collections.\n\n  If running in a `DistributionStrategy` context, the variable will be\n  "sync on read". This means:\n\n  *   The returned object will be a container with separate variables\n      per replica of the model.\n\n  *   When writing to the variable, e.g. using `assign_add` in a metric\n      update, the update will be applied to the variable local to the\n      replica.\n\n  *   To get a metric\'s result value, we need to sum the variable values\n      across the replicas before computing the final answer. Furthermore,\n      the final answer should be computed once instead of in every\n      replica. Both of these are accomplished by running the computation\n      of the final result value inside\n      `distribute_lib.get_replica_context().merge_call(fn)`.\n      Inside the `merge_call()`, ops are only added to the graph once\n      and access to a sync on read variable in a computation returns\n      the sum across all replicas.\n\n  Args:\n    shape: Shape of the created variable.\n    dtype: Type of the created variable.\n    validate_shape: (Optional) Whether shape validation is enabled for\n      the created variable.\n    name: (Optional) String name of the created variable.\n\n  Returns:\n    A (non-trainable) variable initialized to zero, or if inside a\n    `DistributionStrategy` scope a sync on read variable container.\n  '
    return variable_v1.VariableV1(lambda : array_ops.zeros(shape, dtype), trainable=False, collections=[ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.METRIC_VARIABLES], validate_shape=validate_shape, synchronization=variables.VariableSynchronization.ON_READ, aggregation=variables.VariableAggregation.SUM, name=name)

def _remove_squeezable_dimensions(predictions, labels, weights):
    if False:
        return 10
    'Squeeze or expand last dim if needed.\n\n  Squeezes last dim of `predictions` or `labels` if their rank differs by 1\n  (using confusion_matrix.remove_squeezable_dimensions).\n  Squeezes or expands last dim of `weights` if its rank differs by 1 from the\n  new rank of `predictions`.\n\n  If `weights` is scalar, it is kept scalar.\n\n  This will use static shape if available. Otherwise, it will add graph\n  operations, which could result in a performance hit.\n\n  Args:\n    predictions: Predicted values, a `Tensor` of arbitrary dimensions.\n    labels: Optional label `Tensor` whose dimensions match `predictions`.\n    weights: Optional weight scalar or `Tensor` whose dimensions match\n      `predictions`.\n\n  Returns:\n    Tuple of `predictions`, `labels` and `weights`. Each of them possibly has\n    the last dimension squeezed, `weights` could be extended by one dimension.\n  '
    predictions = ops.convert_to_tensor(predictions)
    if labels is not None:
        (labels, predictions) = confusion_matrix.remove_squeezable_dimensions(labels, predictions)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    if weights is None:
        return (predictions, labels, None)
    weights = ops.convert_to_tensor(weights)
    weights_shape = weights.get_shape()
    weights_rank = weights_shape.ndims
    if weights_rank == 0:
        return (predictions, labels, weights)
    predictions_shape = predictions.get_shape()
    predictions_rank = predictions_shape.ndims
    if predictions_rank is not None and weights_rank is not None:
        if weights_rank - predictions_rank == 1:
            weights = array_ops.squeeze(weights, [-1])
        elif predictions_rank - weights_rank == 1:
            weights = array_ops.expand_dims(weights, [-1])
    else:
        weights_rank_tensor = array_ops.rank(weights)
        rank_diff = weights_rank_tensor - array_ops.rank(predictions)

        def _maybe_expand_weights():
            if False:
                for i in range(10):
                    print('nop')
            return cond.cond(math_ops.equal(rank_diff, -1), lambda : array_ops.expand_dims(weights, [-1]), lambda : weights)
        if weights_rank is not None and (not weights_shape.dims[-1].is_compatible_with(1)):
            maybe_squeeze_weights = lambda : weights
        else:
            maybe_squeeze_weights = lambda : array_ops.squeeze(weights, [-1])

        def _maybe_adjust_weights():
            if False:
                return 10
            return cond.cond(math_ops.equal(rank_diff, 1), maybe_squeeze_weights, _maybe_expand_weights)
        weights = cond.cond(math_ops.equal(weights_rank_tensor, 0), lambda : weights, _maybe_adjust_weights)
    return (predictions, labels, weights)

def _maybe_expand_labels(labels, predictions):
    if False:
        i = 10
        return i + 15
    'If necessary, expand `labels` along last dimension to match `predictions`.\n\n  Args:\n    labels: `Tensor` or `SparseTensor` with shape\n      [D1, ... DN, num_labels] or [D1, ... DN]. The latter implies\n      num_labels=1, in which case the result is an expanded `labels` with shape\n      [D1, ... DN, 1].\n    predictions: `Tensor` with shape [D1, ... DN, num_classes].\n\n  Returns:\n    `labels` with the same rank as `predictions`.\n\n  Raises:\n    ValueError: if `labels` has invalid shape.\n  '
    with ops.name_scope(None, 'expand_labels', (labels, predictions)) as scope:
        labels = sparse_tensor.convert_to_tensor_or_sparse_tensor(labels)
        if isinstance(labels, sparse_tensor.SparseTensor):
            return cond.cond(math_ops.equal(array_ops.rank(predictions), array_ops.size(labels.dense_shape) + 1), lambda : sparse_ops.sparse_reshape(labels, shape=array_ops.concat((labels.dense_shape, (1,)), 0), name=scope), lambda : labels)
        labels_rank = labels.get_shape().ndims
        if labels_rank is not None:
            predictions_rank = predictions.get_shape().ndims
            if predictions_rank is not None:
                if predictions_rank == labels_rank:
                    return labels
                if predictions_rank == labels_rank + 1:
                    return array_ops.expand_dims(labels, -1, name=scope)
                raise ValueError(f'Unexpected labels shape {labels.get_shape()} for predictions shape {predictions.get_shape()}. Predictions rank should be the same rank as labels rank or labels rank plus one .')
        return cond.cond(math_ops.equal(array_ops.rank(predictions), array_ops.rank(labels) + 1), lambda : array_ops.expand_dims(labels, -1, name=scope), lambda : labels)

def _safe_scalar_div(numerator, denominator, name):
    if False:
        i = 10
        return i + 15
    'Divides two values, returning 0 if the denominator is 0.\n\n  Args:\n    numerator: A scalar `float64` `Tensor`.\n    denominator: A scalar `float64` `Tensor`.\n    name: Name for the returned op.\n\n  Returns:\n    0 if `denominator` == 0, else `numerator` / `denominator`\n  '
    numerator.get_shape().with_rank_at_most(1)
    denominator.get_shape().with_rank_at_most(1)
    return math_ops.div_no_nan(numerator, denominator, name=name)

def _streaming_confusion_matrix(labels, predictions, num_classes, weights=None):
    if False:
        print('Hello World!')
    'Calculate a streaming confusion matrix.\n\n  Calculates a confusion matrix. For estimation over a stream of data,\n  the function creates an  `update_op` operation.\n\n  Args:\n    labels: A `Tensor` of ground truth labels with shape [batch size] and of\n      type `int32` or `int64`. The tensor will be flattened if its rank > 1.\n    predictions: A `Tensor` of prediction results for semantic labels, whose\n      shape is [batch size] and type `int32` or `int64`. The tensor will be\n      flattened if its rank > 1.\n    num_classes: The possible number of labels the prediction task can\n      have. This value must be provided, since a confusion matrix of\n      dimension = [num_classes, num_classes] will be allocated.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n\n  Returns:\n    total_cm: A `Tensor` representing the confusion matrix.\n    update_op: An operation that increments the confusion matrix.\n  '
    total_cm = metric_variable([num_classes, num_classes], dtypes.float64, name='total_confusion_matrix')
    predictions = math_ops.cast(predictions, dtypes.int64)
    labels = math_ops.cast(labels, dtypes.int64)
    num_classes = math_ops.cast(num_classes, dtypes.int64)
    if predictions.get_shape().ndims > 1:
        predictions = array_ops.reshape(predictions, [-1])
    if labels.get_shape().ndims > 1:
        labels = array_ops.reshape(labels, [-1])
    if weights is not None and weights.get_shape().ndims > 1:
        weights = array_ops.reshape(weights, [-1])
    current_cm = confusion_matrix.confusion_matrix(labels, predictions, num_classes, weights=weights, dtype=dtypes.float64)
    update_op = state_ops.assign_add(total_cm, current_cm)
    return (total_cm, update_op)

def _aggregate_across_replicas(metrics_collections, metric_value_fn, *args):
    if False:
        i = 10
        return i + 15
    'Aggregate metric value across replicas.'

    def fn(distribution, *a):
        if False:
            while True:
                i = 10
        'Call `metric_value_fn` in the correct control flow context.'
        if hasattr(distribution.extended, '_outer_control_flow_context'):
            if distribution.extended._outer_control_flow_context is None:
                with ops.control_dependencies(None):
                    metric_value = metric_value_fn(distribution, *a)
            else:
                distribution.extended._outer_control_flow_context.Enter()
                metric_value = metric_value_fn(distribution, *a)
                distribution.extended._outer_control_flow_context.Exit()
        else:
            metric_value = metric_value_fn(distribution, *a)
        if metrics_collections:
            ops.add_to_collections(metrics_collections, metric_value)
        return metric_value
    return distribute_lib.get_replica_context().merge_call(fn, args=args)

@tf_export(v1=['metrics.mean'])
def mean(values, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        i = 10
        return i + 15
    "Computes the (weighted) mean of the given values.\n\n  The `mean` function creates two local variables, `total` and `count`\n  that are used to compute the average of `values`. This average is ultimately\n  returned as `mean` which is an idempotent operation that simply divides\n  `total` by `count`.\n\n  For estimation of the metric over a stream of data, the function creates an\n  `update_op` operation that updates these variables and returns the `mean`.\n  `update_op` increments `total` with the reduced sum of the product of `values`\n  and `weights`, and it increments `count` with the reduced sum of `weights`.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    values: A `Tensor` of arbitrary dimensions.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `values`, and must be broadcastable to `values` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `values` dimension).\n    metrics_collections: An optional list of collections that `mean`\n      should be added to.\n    updates_collections: An optional list of collections that `update_op`\n      should be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    mean: A `Tensor` representing the current mean, the value of `total` divided\n      by `count`.\n    update_op: An operation that increments the `total` and `count` variables\n      appropriately and whose value matches `mean_value`.\n\n  Raises:\n    ValueError: If `weights` is not `None` and its shape doesn't match `values`,\n      or if either `metrics_collections` or `updates_collections` are not a list\n      or tuple.\n    RuntimeError: If eager execution is enabled.\n\n  @compatibility(TF2)\n  `tf.compat.v1.metrics.mean` is not compatible with eager\n  execution or `tf.function`.\n  Please use `tf.keras.metrics.Mean` instead for TF2 migration. After\n  instantiating a `tf.keras.metrics.Mean` object, you can first call the\n  `update_state()` method to record the new values, and then call the\n  `result()` method to get the mean eagerly. You can also attach it to a\n  Keras model with the `add_metric` method.  Please refer to the [migration\n  guide](https://www.tensorflow.org/guide/migrate#new-style_metrics_and_losses)\n  for more details.\n\n  #### Structural Mapping to TF2\n\n  Before:\n\n  ```python\n  mean, update_op = tf.compat.v1.metrics.mean(\n    values=values,\n    weights=weights,\n    metrics_collections=metrics_collections,\n    update_collections=update_collections,\n    name=name)\n  ```\n\n  After:\n\n  ```python\n   m = tf.keras.metrics.Mean(\n     name=name)\n\n   m.update_state(\n     values=values,\n     sample_weight=weights)\n\n   mean = m.result()\n  ```\n\n  #### How to Map Arguments\n\n  | TF1 Arg Name          | TF2 Arg Name    | Note                       |\n  | :-------------------- | :-------------- | :------------------------- |\n  | `values`              | `values`        | In `update_state()` method |\n  | `weights`             | `sample_weight` | In `update_state()` method |\n  | `metrics_collections` | Not supported   | Metrics should be tracked  |\n  :                       :                 : explicitly or with Keras   :\n  :                       :                 : APIs, for example,         :\n  :                       :                 : [add_metric][add_metric],  :\n  :                       :                 : instead of via collections :\n  | `updates_collections` | Not supported   | -                          |\n  | `name`                | `name`          | In constructor             |\n\n  [add_metric]:https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_metric\n\n\n  #### Before & After Usage Example\n\n  Before:\n\n  >>> g = tf.Graph()\n  >>> with g.as_default():\n  ...   values = [1, 2, 3]\n  ...   mean, update_op = tf.compat.v1.metrics.mean(values)\n  ...   global_init = tf.compat.v1.global_variables_initializer()\n  ...   local_init = tf.compat.v1.local_variables_initializer()\n  >>> sess = tf.compat.v1.Session(graph=g)\n  >>> sess.run([global_init, local_init])\n  >>> sess.run(update_op)\n  >>> sess.run(mean)\n  2.0\n\n\n  After:\n\n  >>> m = tf.keras.metrics.Mean()\n  >>> m.update_state([1, 2, 3])\n  >>> m.result().numpy()\n  2.0\n\n  ```python\n  # Used within Keras model\n  model.add_metric(tf.keras.metrics.Mean()(values))\n  ```\n\n  @end_compatibility\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.mean is not supported when eager execution is enabled.')
    with variable_scope.variable_scope(name, 'mean', (values, weights)):
        values = math_ops.cast(values, dtypes.float32)
        total = metric_variable([], dtypes.float32, name='total')
        count = metric_variable([], dtypes.float32, name='count')
        if weights is None:
            num_values = math_ops.cast(array_ops.size(values), dtypes.float32)
        else:
            (values, _, weights) = _remove_squeezable_dimensions(predictions=values, labels=None, weights=weights)
            weights = weights_broadcast_ops.broadcast_weights(math_ops.cast(weights, dtypes.float32), values)
            values = math_ops.multiply(values, weights)
            num_values = math_ops.reduce_sum(weights)
        update_total_op = state_ops.assign_add(total, math_ops.reduce_sum(values))
        with ops.control_dependencies([values]):
            update_count_op = state_ops.assign_add(count, num_values)

        def compute_mean(_, t, c):
            if False:
                print('Hello World!')
            return math_ops.div_no_nan(t, math_ops.maximum(c, 0), name='value')
        mean_t = _aggregate_across_replicas(metrics_collections, compute_mean, total, count)
        update_op = math_ops.div_no_nan(update_total_op, math_ops.maximum(update_count_op, 0), name='update_op')
        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)
        return (mean_t, update_op)

@tf_export(v1=['metrics.accuracy'])
def accuracy(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        i = 10
        return i + 15
    "Calculates how often `predictions` matches `labels`.\n\n  The `accuracy` function creates two local variables, `total` and\n  `count` that are used to compute the frequency with which `predictions`\n  matches `labels`. This frequency is ultimately returned as `accuracy`: an\n  idempotent operation that simply divides `total` by `count`.\n\n  For estimation of the metric over a stream of data, the function creates an\n  `update_op` operation that updates these variables and returns the `accuracy`.\n  Internally, an `is_correct` operation computes a `Tensor` with elements 1.0\n  where the corresponding elements of `predictions` and `labels` match and 0.0\n  otherwise. Then `update_op` increments `total` with the reduced sum of the\n  product of `weights` and `is_correct`, and it increments `count` with the\n  reduced sum of `weights`.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: The ground truth values, a `Tensor` whose shape matches\n      `predictions`.\n    predictions: The predicted values, a `Tensor` of any shape.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    metrics_collections: An optional list of collections that `accuracy` should\n      be added to.\n    updates_collections: An optional list of collections that `update_op` should\n      be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    accuracy: A `Tensor` representing the accuracy, the value of `total` divided\n      by `count`.\n    update_op: An operation that increments the `total` and `count` variables\n      appropriately and whose value matches `accuracy`.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, or if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      either `metrics_collections` or `updates_collections` are not a list or\n      tuple.\n    RuntimeError: If eager execution is enabled.\n\n  @compatibility(TF2)\n  `tf.compat.v1.metrics.accuracy` is not compatible with eager\n  execution or `tf.function`.\n  Please use `tf.keras.metrics.Accuracy` instead for TF2 migration. After\n  instantiating a `tf.keras.metrics.Accuracy` object, you can first call the\n  `update_state()` method to record the prediction/labels, and then call the\n  `result()` method to get the accuracy eagerly. You can also attach it to a\n  Keras model when calling the `compile` method. Please refer to [this\n  guide](https://www.tensorflow.org/guide/migrate#new-style_metrics_and_losses)\n  for more details.\n\n  #### Structural Mapping to Native TF2\n\n  Before:\n\n  ```python\n  accuracy, update_op = tf.compat.v1.metrics.accuracy(\n    labels=labels,\n    predictions=predictions,\n    weights=weights,\n    metrics_collections=metrics_collections,\n    update_collections=update_collections,\n    name=name)\n  ```\n\n  After:\n\n  ```python\n   m = tf.keras.metrics.Accuracy(\n     name=name,\n     dtype=None)\n\n   m.update_state(\n   y_true=labels,\n   y_pred=predictions,\n   sample_weight=weights)\n\n   accuracy = m.result()\n  ```\n\n  #### How to Map Arguments\n\n  | TF1 Arg Name          | TF2 Arg Name    | Note                       |\n  | :-------------------- | :-------------- | :------------------------- |\n  | `label`               | `y_true`        | In `update_state()` method |\n  | `predictions`         | `y_true`        | In `update_state()` method |\n  | `weights`             | `sample_weight` | In `update_state()` method |\n  | `metrics_collections` | Not supported   | Metrics should be tracked  |\n  :                       :                 : explicitly or with Keras   :\n  :                       :                 : APIs, for example,         :\n  :                       :                 : [add_metric][add_metric],  :\n  :                       :                 : instead of via collections :\n  | `updates_collections` | Not supported   | -                          |\n  | `name`                | `name`          | In constructor             |\n\n  [add_metric]:https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_metric\n\n\n  #### Before & After Usage Example\n\n  Before:\n\n  >>> g = tf.Graph()\n  >>> with g.as_default():\n  ...   logits = [1, 2, 3]\n  ...   labels = [0, 2, 3]\n  ...   acc, acc_op = tf.compat.v1.metrics.accuracy(logits, labels)\n  ...   global_init = tf.compat.v1.global_variables_initializer()\n  ...   local_init = tf.compat.v1.local_variables_initializer()\n  >>> sess = tf.compat.v1.Session(graph=g)\n  >>> sess.run([global_init, local_init])\n  >>> print(sess.run([acc, acc_op]))\n  [0.0, 0.66667]\n\n\n  After:\n\n  >>> m = tf.keras.metrics.Accuracy()\n  >>> m.update_state([1, 2, 3], [0, 2, 3])\n  >>> m.result().numpy()\n  0.66667\n\n  ```python\n  # Used within Keras model\n  model.compile(optimizer='sgd',\n                loss='mse',\n                metrics=[tf.keras.metrics.Accuracy()])\n  ```\n\n  @end_compatibility\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.accuracy is not supported when eager execution is enabled.')
    (predictions, labels, weights) = _remove_squeezable_dimensions(predictions=predictions, labels=labels, weights=weights)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    if labels.dtype != predictions.dtype:
        predictions = math_ops.cast(predictions, labels.dtype)
    is_correct = math_ops.cast(math_ops.equal(predictions, labels), dtypes.float32)
    return mean(is_correct, weights, metrics_collections, updates_collections, name or 'accuracy')

def _confusion_matrix_at_thresholds(labels, predictions, thresholds, weights=None, includes=None):
    if False:
        return 10
    "Computes true_positives, false_negatives, true_negatives, false_positives.\n\n  This function creates up to four local variables, `true_positives`,\n  `true_negatives`, `false_positives` and `false_negatives`.\n  `true_positive[i]` is defined as the total weight of values in `predictions`\n  above `thresholds[i]` whose corresponding entry in `labels` is `True`.\n  `false_negatives[i]` is defined as the total weight of values in `predictions`\n  at most `thresholds[i]` whose corresponding entry in `labels` is `True`.\n  `true_negatives[i]` is defined as the total weight of values in `predictions`\n  at most `thresholds[i]` whose corresponding entry in `labels` is `False`.\n  `false_positives[i]` is defined as the total weight of values in `predictions`\n  above `thresholds[i]` whose corresponding entry in `labels` is `False`.\n\n  For estimation of these metrics over a stream of data, for each metric the\n  function respectively creates an `update_op` operation that updates the\n  variable and returns its value.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: A `Tensor` whose shape matches `predictions`. Will be cast to\n      `bool`.\n    predictions: A floating point `Tensor` of arbitrary shape and whose values\n      are in the range `[0, 1]`.\n    thresholds: A python list or tuple of float thresholds in `[0, 1]`.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    includes: Tuple of keys to return, from 'tp', 'fn', 'tn', fp'. If `None`,\n        default to all four.\n\n  Returns:\n    values: Dict of variables of shape `[len(thresholds)]`. Keys are from\n        `includes`.\n    update_ops: Dict of operations that increments the `values`. Keys are from\n        `includes`.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, or if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      `includes` contains invalid keys.\n  "
    all_includes = ('tp', 'fn', 'tn', 'fp')
    if includes is None:
        includes = all_includes
    else:
        for include in includes:
            if include not in all_includes:
                raise ValueError(f'Invalid key: {include}')
    with ops.control_dependencies([check_ops.assert_greater_equal(predictions, math_ops.cast(0.0, dtype=predictions.dtype), message='predictions must be in [0, 1]'), check_ops.assert_less_equal(predictions, math_ops.cast(1.0, dtype=predictions.dtype), message='predictions must be in [0, 1]')]):
        (predictions, labels, weights) = _remove_squeezable_dimensions(predictions=math_ops.cast(predictions, dtypes.float32), labels=math_ops.cast(labels, dtype=dtypes.bool), weights=weights)
    num_thresholds = len(thresholds)
    predictions_2d = array_ops.reshape(predictions, [-1, 1])
    labels_2d = array_ops.reshape(math_ops.cast(labels, dtype=dtypes.bool), [1, -1])
    num_predictions = predictions_2d.get_shape().as_list()[0]
    if num_predictions is None:
        num_predictions = array_ops.shape(predictions_2d)[0]
    thresh_tiled = array_ops.tile(array_ops.expand_dims(array_ops.constant(thresholds), [1]), array_ops_stack.stack([1, num_predictions]))
    pred_is_pos = math_ops.greater(array_ops.tile(array_ops.transpose(predictions_2d), [num_thresholds, 1]), thresh_tiled)
    if 'fn' in includes or 'tn' in includes:
        pred_is_neg = math_ops.logical_not(pred_is_pos)
    label_is_pos = array_ops.tile(labels_2d, [num_thresholds, 1])
    if 'fp' in includes or 'tn' in includes:
        label_is_neg = math_ops.logical_not(label_is_pos)
    if weights is not None:
        weights = weights_broadcast_ops.broadcast_weights(math_ops.cast(weights, dtypes.float32), predictions)
        weights_tiled = array_ops.tile(array_ops.reshape(weights, [1, -1]), [num_thresholds, 1])
        thresh_tiled.get_shape().assert_is_compatible_with(weights_tiled.get_shape())
    else:
        weights_tiled = None
    values = {}
    update_ops = {}
    if 'tp' in includes:
        true_p = metric_variable([num_thresholds], dtypes.float32, name='true_positives')
        is_true_positive = math_ops.cast(math_ops.logical_and(label_is_pos, pred_is_pos), dtypes.float32)
        if weights_tiled is not None:
            is_true_positive *= weights_tiled
        update_ops['tp'] = state_ops.assign_add(true_p, math_ops.reduce_sum(is_true_positive, 1))
        values['tp'] = true_p
    if 'fn' in includes:
        false_n = metric_variable([num_thresholds], dtypes.float32, name='false_negatives')
        is_false_negative = math_ops.cast(math_ops.logical_and(label_is_pos, pred_is_neg), dtypes.float32)
        if weights_tiled is not None:
            is_false_negative *= weights_tiled
        update_ops['fn'] = state_ops.assign_add(false_n, math_ops.reduce_sum(is_false_negative, 1))
        values['fn'] = false_n
    if 'tn' in includes:
        true_n = metric_variable([num_thresholds], dtypes.float32, name='true_negatives')
        is_true_negative = math_ops.cast(math_ops.logical_and(label_is_neg, pred_is_neg), dtypes.float32)
        if weights_tiled is not None:
            is_true_negative *= weights_tiled
        update_ops['tn'] = state_ops.assign_add(true_n, math_ops.reduce_sum(is_true_negative, 1))
        values['tn'] = true_n
    if 'fp' in includes:
        false_p = metric_variable([num_thresholds], dtypes.float32, name='false_positives')
        is_false_positive = math_ops.cast(math_ops.logical_and(label_is_neg, pred_is_pos), dtypes.float32)
        if weights_tiled is not None:
            is_false_positive *= weights_tiled
        update_ops['fp'] = state_ops.assign_add(false_p, math_ops.reduce_sum(is_false_positive, 1))
        values['fp'] = false_p
    return (values, update_ops)

def _aggregate_variable(v, collections):
    if False:
        print('Hello World!')
    f = lambda distribution, value: distribution.extended.read_var(value)
    return _aggregate_across_replicas(collections, f, v)

@tf_export(v1=['metrics.auc'])
@deprecated(None, 'The value of AUC returned by this may race with the update so this is deprecated. Please use tf.keras.metrics.AUC instead.')
def auc(labels, predictions, weights=None, num_thresholds=200, metrics_collections=None, updates_collections=None, curve='ROC', name=None, summation_method='trapezoidal', thresholds=None):
    if False:
        return 10
    "Computes the approximate AUC via a Riemann sum.\n\n  The `auc` function creates four local variables, `true_positives`,\n  `true_negatives`, `false_positives` and `false_negatives` that are used to\n  compute the AUC. To discretize the AUC curve, a linearly spaced set of\n  thresholds is used to compute pairs of recall and precision values. The area\n  under the ROC-curve is therefore computed using the height of the recall\n  values by the false positive rate, while the area under the PR-curve is the\n  computed using the height of the precision values by the recall.\n\n  This value is ultimately returned as `auc`, an idempotent operation that\n  computes the area under a discretized curve of precision versus recall values\n  (computed using the aforementioned variables). The `num_thresholds` variable\n  controls the degree of discretization with larger numbers of thresholds more\n  closely approximating the true AUC. The quality of the approximation may vary\n  dramatically depending on `num_thresholds`.\n\n  For best results, `predictions` should be distributed approximately uniformly\n  in the range [0, 1] and not peaked around 0 or 1. The quality of the AUC\n  approximation may be poor if this is not the case. Setting `summation_method`\n  to 'minoring' or 'majoring' can help quantify the error in the approximation\n  by providing lower or upper bound estimate of the AUC. The `thresholds`\n  parameter can be used to manually specify thresholds which split the\n  predictions more evenly.\n\n  For estimation of the metric over a stream of data, the function creates an\n  `update_op` operation that updates these variables and returns the `auc`.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: A `Tensor` whose shape matches `predictions`. Will be cast to\n      `bool`.\n    predictions: A floating point `Tensor` of arbitrary shape and whose values\n      are in the range `[0, 1]`.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    num_thresholds: The number of thresholds to use when discretizing the roc\n      curve.\n    metrics_collections: An optional list of collections that `auc` should be\n      added to.\n    updates_collections: An optional list of collections that `update_op` should\n      be added to.\n    curve: Specifies the name of the curve to be computed, 'ROC' [default] or\n      'PR' for the Precision-Recall-curve.\n    name: An optional variable_scope name.\n    summation_method: Specifies the Riemann summation method used\n      (https://en.wikipedia.org/wiki/Riemann_sum): 'trapezoidal' [default] that\n      applies the trapezoidal rule; 'careful_interpolation', a variant of it\n      differing only by a more correct interpolation scheme for PR-AUC -\n      interpolating (true/false) positives but not the ratio that is precision;\n      'minoring' that applies left summation for increasing intervals and right\n      summation for decreasing intervals; 'majoring' that does the opposite.\n      Note that 'careful_interpolation' is strictly preferred to 'trapezoidal'\n      (to be deprecated soon) as it applies the same method for ROC, and a\n      better one (see Davis & Goadrich 2006 for details) for the PR curve.\n    thresholds: An optional list of floating point values to use as the\n      thresholds for discretizing the curve. If set, the `num_thresholds`\n      parameter is ignored. Values should be in [0, 1]. Endpoint thresholds\n      equal to {-epsilon, 1+epsilon} for a small positive epsilon value will be\n      automatically included with these to correctly handle predictions equal to\n       exactly 0 or 1.\n\n  Returns:\n    auc: A scalar `Tensor` representing the current area-under-curve.\n    update_op: An operation that increments the `true_positives`,\n      `true_negatives`, `false_positives` and `false_negatives` variables\n      appropriately and whose value matches `auc`.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, or if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      either `metrics_collections` or `updates_collections` are not a list or\n      tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.auc is not supported when eager execution is enabled.')
    with variable_scope.variable_scope(name, 'auc', (labels, predictions, weights)):
        if curve != 'ROC' and curve != 'PR':
            raise ValueError(f'Curve must be either ROC or PR. Curve {curve} is unknown.')
        kepsilon = 1e-07
        if thresholds is not None:
            thresholds = sorted(thresholds)
            num_thresholds = len(thresholds) + 2
        else:
            thresholds = [(i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)]
        thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]
        (values, update_ops) = _confusion_matrix_at_thresholds(labels, predictions, thresholds, weights)
        epsilon = 1e-06

        def interpolate_pr_auc(tp, fp, fn):
            if False:
                print('Hello World!')
            "Interpolation formula inspired by section 4 of (Davis et al., 2006).\n\n      Note here we derive & use a closed formula not present in the paper\n      - as follows:\n      Modeling all of TP (true positive weight),\n      FP (false positive weight) and their sum P = TP + FP (positive weight)\n      as varying linearly within each interval [A, B] between successive\n      thresholds, we get\n        Precision = (TP_A + slope * (P - P_A)) / P\n      with slope = dTP / dP = (TP_B - TP_A) / (P_B - P_A).\n      The area within the interval is thus (slope / total_pos_weight) times\n        int_A^B{Precision.dP} = int_A^B{(TP_A + slope * (P - P_A)) * dP / P}\n        int_A^B{Precision.dP} = int_A^B{slope * dP + intercept * dP / P}\n      where intercept = TP_A - slope * P_A = TP_B - slope * P_B, resulting in\n        int_A^B{Precision.dP} = TP_B - TP_A + intercept * log(P_B / P_A)\n      Bringing back the factor (slope / total_pos_weight) we'd put aside, we get\n         slope * [dTP + intercept *  log(P_B / P_A)] / total_pos_weight\n      where dTP == TP_B - TP_A.\n      Note that when P_A == 0 the above calculation simplifies into\n        int_A^B{Precision.dTP} = int_A^B{slope * dTP} = slope * (TP_B - TP_A)\n      which is really equivalent to imputing constant precision throughout the\n      first bucket having >0 true positives.\n\n      Args:\n        tp: true positive counts\n        fp: false positive counts\n        fn: false negative counts\n\n      Returns:\n        pr_auc: an approximation of the area under the P-R curve.\n\n      References:\n        The Relationship Between Precision-Recall and ROC Curves:\n          [Davis et al., 2006](https://dl.acm.org/citation.cfm?id=1143874)\n          ([pdf](https://www.biostat.wisc.edu/~page/rocpr.pdf))\n      "
            dtp = tp[:num_thresholds - 1] - tp[1:]
            p = tp + fp
            prec_slope = math_ops.div_no_nan(dtp, math_ops.maximum(p[:num_thresholds - 1] - p[1:], 0), name='prec_slope')
            intercept = tp[1:] - math_ops.multiply(prec_slope, p[1:])
            safe_p_ratio = array_ops.where(math_ops.logical_and(p[:num_thresholds - 1] > 0, p[1:] > 0), math_ops.div_no_nan(p[:num_thresholds - 1], math_ops.maximum(p[1:], 0), name='recall_relative_ratio'), array_ops.ones_like(p[1:]))
            return math_ops.reduce_sum(math_ops.div_no_nan(prec_slope * (dtp + intercept * math_ops.log(safe_p_ratio)), math_ops.maximum(tp[1:] + fn[1:], 0), name='pr_auc_increment'), name='interpolate_pr_auc')

        def compute_auc(tp, fn, tn, fp, name):
            if False:
                for i in range(10):
                    print('nop')
            'Computes the roc-auc or pr-auc based on confusion counts.'
            if curve == 'PR':
                if summation_method == 'trapezoidal':
                    logging.warning('Trapezoidal rule is known to produce incorrect PR-AUCs; please switch to "careful_interpolation" instead.')
                elif summation_method == 'careful_interpolation':
                    return interpolate_pr_auc(tp, fp, fn)
            rec = math_ops.divide(tp + epsilon, tp + fn + epsilon)
            if curve == 'ROC':
                fp_rate = math_ops.divide(fp, fp + tn + epsilon)
                x = fp_rate
                y = rec
            else:
                prec = math_ops.divide(tp + epsilon, tp + fp + epsilon)
                x = rec
                y = prec
            if summation_method in ('trapezoidal', 'careful_interpolation'):
                return math_ops.reduce_sum(math_ops.multiply(x[:num_thresholds - 1] - x[1:], (y[:num_thresholds - 1] + y[1:]) / 2.0), name=name)
            elif summation_method == 'minoring':
                return math_ops.reduce_sum(math_ops.multiply(x[:num_thresholds - 1] - x[1:], math_ops.minimum(y[:num_thresholds - 1], y[1:])), name=name)
            elif summation_method == 'majoring':
                return math_ops.reduce_sum(math_ops.multiply(x[:num_thresholds - 1] - x[1:], math_ops.maximum(y[:num_thresholds - 1], y[1:])), name=name)
            else:
                raise ValueError(f"Invalid summation_method: {summation_method} summation_method should be 'trapezoidal', 'careful_interpolation', 'minoring', or 'majoring'.")

        def compute_auc_value(_, values):
            if False:
                print('Hello World!')
            return compute_auc(values['tp'], values['fn'], values['tn'], values['fp'], 'value')
        auc_value = _aggregate_across_replicas(metrics_collections, compute_auc_value, values)
        update_op = compute_auc(update_ops['tp'], update_ops['fn'], update_ops['tn'], update_ops['fp'], 'update_op')
        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)
        return (auc_value, update_op)

@tf_export(v1=['metrics.mean_absolute_error'])
def mean_absolute_error(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        print('Hello World!')
    "Computes the mean absolute error between the labels and predictions.\n\n  The `mean_absolute_error` function creates two local variables,\n  `total` and `count` that are used to compute the mean absolute error. This\n  average is weighted by `weights`, and it is ultimately returned as\n  `mean_absolute_error`: an idempotent operation that simply divides `total` by\n  `count`.\n\n  For estimation of the metric over a stream of data, the function creates an\n  `update_op` operation that updates these variables and returns the\n  `mean_absolute_error`. Internally, an `absolute_errors` operation computes the\n  absolute value of the differences between `predictions` and `labels`. Then\n  `update_op` increments `total` with the reduced sum of the product of\n  `weights` and `absolute_errors`, and it increments `count` with the reduced\n  sum of `weights`\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: A `Tensor` of the same shape as `predictions`.\n    predictions: A `Tensor` of arbitrary shape.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    metrics_collections: An optional list of collections that\n      `mean_absolute_error` should be added to.\n    updates_collections: An optional list of collections that `update_op` should\n      be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    mean_absolute_error: A `Tensor` representing the current mean, the value of\n      `total` divided by `count`.\n    update_op: An operation that increments the `total` and `count` variables\n      appropriately and whose value matches `mean_absolute_error`.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, or if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      either `metrics_collections` or `updates_collections` are not a list or\n      tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.mean_absolute_error is not supported when eager execution is enabled.')
    (predictions, labels, weights) = _remove_squeezable_dimensions(predictions=predictions, labels=labels, weights=weights)
    absolute_errors = math_ops.abs(predictions - labels)
    return mean(absolute_errors, weights, metrics_collections, updates_collections, name or 'mean_absolute_error')

@tf_export(v1=['metrics.mean_cosine_distance'])
def mean_cosine_distance(labels, predictions, dim, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        print('Hello World!')
    "Computes the cosine distance between the labels and predictions.\n\n  The `mean_cosine_distance` function creates two local variables,\n  `total` and `count` that are used to compute the average cosine distance\n  between `predictions` and `labels`. This average is weighted by `weights`,\n  and it is ultimately returned as `mean_distance`, which is an idempotent\n  operation that simply divides `total` by `count`.\n\n  For estimation of the metric over a stream of data, the function creates an\n  `update_op` operation that updates these variables and returns the\n  `mean_distance`.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: A `Tensor` of arbitrary shape.\n    predictions: A `Tensor` of the same shape as `labels`.\n    dim: The dimension along which the cosine distance is computed.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension). Also,\n      dimension `dim` must be `1`.\n    metrics_collections: An optional list of collections that the metric\n      value variable should be added to.\n    updates_collections: An optional list of collections that the metric update\n      ops should be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    mean_distance: A `Tensor` representing the current mean, the value of\n      `total` divided by `count`.\n    update_op: An operation that increments the `total` and `count` variables\n      appropriately.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, or if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      either `metrics_collections` or `updates_collections` are not a list or\n      tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.mean_cosine_distance is not supported when eager execution is enabled.')
    (predictions, labels, weights) = _remove_squeezable_dimensions(predictions=predictions, labels=labels, weights=weights)
    radial_diffs = math_ops.multiply(predictions, labels)
    radial_diffs = math_ops.reduce_sum(radial_diffs, axis=[dim], keepdims=True)
    (mean_distance, update_op) = mean(radial_diffs, weights, None, None, name or 'mean_cosine_distance')
    mean_distance = math_ops.subtract(1.0, mean_distance)
    update_op = math_ops.subtract(1.0, update_op)
    if metrics_collections:
        ops.add_to_collections(metrics_collections, mean_distance)
    if updates_collections:
        ops.add_to_collections(updates_collections, update_op)
    return (mean_distance, update_op)

@tf_export(v1=['metrics.mean_per_class_accuracy'])
def mean_per_class_accuracy(labels, predictions, num_classes, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        i = 10
        return i + 15
    "Calculates the mean of the per-class accuracies.\n\n  Calculates the accuracy for each class, then takes the mean of that.\n\n  For estimation of the metric over a stream of data, the function creates an\n  `update_op` operation that updates the accuracy of each class and returns\n  them.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: A `Tensor` of ground truth labels with shape [batch size] and of\n      type `int32` or `int64`. The tensor will be flattened if its rank > 1.\n    predictions: A `Tensor` of prediction results for semantic labels, whose\n      shape is [batch size] and type `int32` or `int64`. The tensor will be\n      flattened if its rank > 1.\n    num_classes: The possible number of labels the prediction task can\n      have. This value must be provided, since two variables with shape =\n      [num_classes] will be allocated.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    metrics_collections: An optional list of collections that\n      `mean_per_class_accuracy'\n      should be added to.\n    updates_collections: An optional list of collections `update_op` should be\n      added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    mean_accuracy: A `Tensor` representing the mean per class accuracy.\n    update_op: An operation that updates the accuracy tensor.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, or if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      either `metrics_collections` or `updates_collections` are not a list or\n      tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.mean_per_class_accuracy is not supported when eager execution is enabled.')
    with variable_scope.variable_scope(name, 'mean_accuracy', (predictions, labels, weights)):
        labels = math_ops.cast(labels, dtypes.int64)
        if labels.get_shape().ndims > 1:
            labels = array_ops.reshape(labels, [-1])
        if predictions.get_shape().ndims > 1:
            predictions = array_ops.reshape(predictions, [-1])
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        total = metric_variable([num_classes], dtypes.float32, name='total')
        count = metric_variable([num_classes], dtypes.float32, name='count')
        ones = array_ops.ones([array_ops.size(labels)], dtypes.float32)
        if labels.dtype != predictions.dtype:
            predictions = math_ops.cast(predictions, labels.dtype)
        is_correct = math_ops.cast(math_ops.equal(predictions, labels), dtypes.float32)
        if weights is not None:
            if weights.get_shape().ndims > 1:
                weights = array_ops.reshape(weights, [-1])
            weights = math_ops.cast(weights, dtypes.float32)
            is_correct *= weights
            ones *= weights
        update_total_op = state_ops.scatter_add(total, labels, ones)
        update_count_op = state_ops.scatter_add(count, labels, is_correct)

        def compute_mean_accuracy(_, count, total):
            if False:
                return 10
            per_class_accuracy = math_ops.div_no_nan(count, math_ops.maximum(total, 0), name=None)
            mean_accuracy_v = math_ops.reduce_mean(per_class_accuracy, name='mean_accuracy')
            return mean_accuracy_v
        mean_accuracy_v = _aggregate_across_replicas(metrics_collections, compute_mean_accuracy, count, total)
        update_op = math_ops.div_no_nan(update_count_op, math_ops.maximum(update_total_op, 0), name='update_op')
        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)
        return (mean_accuracy_v, update_op)

@tf_export(v1=['metrics.mean_iou'])
def mean_iou(labels, predictions, num_classes, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        print('Hello World!')
    "Calculate per-step mean Intersection-Over-Union (mIOU).\n\n  Mean Intersection-Over-Union is a common evaluation metric for\n  semantic image segmentation, which first computes the IOU for each\n  semantic class and then computes the average over classes.\n  IOU is defined as follows:\n    IOU = true_positive / (true_positive + false_positive + false_negative).\n  The predictions are accumulated in a confusion matrix, weighted by `weights`,\n  and mIOU is then calculated from it.\n\n  For estimation of the metric over a stream of data, the function creates an\n  `update_op` operation that updates these variables and returns the `mean_iou`.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: A `Tensor` of ground truth labels with shape [batch size] and of\n      type `int32` or `int64`. The tensor will be flattened if its rank > 1.\n    predictions: A `Tensor` of prediction results for semantic labels, whose\n      shape is [batch size] and type `int32` or `int64`. The tensor will be\n      flattened if its rank > 1.\n    num_classes: The possible number of labels the prediction task can\n      have. This value must be provided, since a confusion matrix of\n      dimension = [num_classes, num_classes] will be allocated.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    metrics_collections: An optional list of collections that `mean_iou`\n      should be added to.\n    updates_collections: An optional list of collections `update_op` should be\n      added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    mean_iou: A `Tensor` representing the mean intersection-over-union.\n    update_op: An operation that increments the confusion matrix.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, or if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      either `metrics_collections` or `updates_collections` are not a list or\n      tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.mean_iou is not supported when eager execution is enabled.')
    with variable_scope.variable_scope(name, 'mean_iou', (predictions, labels, weights)):
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        (total_cm, update_op) = _streaming_confusion_matrix(labels, predictions, num_classes, weights)

        def compute_mean_iou(_, total_cm):
            if False:
                return 10
            'Compute the mean intersection-over-union via the confusion matrix.'
            sum_over_row = math_ops.cast(math_ops.reduce_sum(total_cm, 0), dtypes.float32)
            sum_over_col = math_ops.cast(math_ops.reduce_sum(total_cm, 1), dtypes.float32)
            cm_diag = math_ops.cast(array_ops.diag_part(total_cm), dtypes.float32)
            denominator = sum_over_row + sum_over_col - cm_diag
            num_valid_entries = math_ops.reduce_sum(math_ops.cast(math_ops.not_equal(denominator, 0), dtype=dtypes.float32))
            denominator = array_ops.where(math_ops.greater(denominator, 0), denominator, array_ops.ones_like(denominator))
            iou = math_ops.divide(cm_diag, denominator)
            result = array_ops.where(math_ops.greater(num_valid_entries, 0), math_ops.reduce_sum(iou, name='mean_iou') / num_valid_entries, 0)
            return result
        mean_iou_v = _aggregate_across_replicas(metrics_collections, compute_mean_iou, total_cm)
        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)
        return (mean_iou_v, update_op)

@tf_export(v1=['metrics.mean_relative_error'])
def mean_relative_error(labels, predictions, normalizer, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    "Computes the mean relative error by normalizing with the given values.\n\n  The `mean_relative_error` function creates two local variables,\n  `total` and `count` that are used to compute the mean relative absolute error.\n  This average is weighted by `weights`, and it is ultimately returned as\n  `mean_relative_error`: an idempotent operation that simply divides `total` by\n  `count`.\n\n  For estimation of the metric over a stream of data, the function creates an\n  `update_op` operation that updates these variables and returns the\n  `mean_reative_error`. Internally, a `relative_errors` operation divides the\n  absolute value of the differences between `predictions` and `labels` by the\n  `normalizer`. Then `update_op` increments `total` with the reduced sum of the\n  product of `weights` and `relative_errors`, and it increments `count` with the\n  reduced sum of `weights`.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: A `Tensor` of the same shape as `predictions`.\n    predictions: A `Tensor` of arbitrary shape.\n    normalizer: A `Tensor` of the same shape as `predictions`.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    metrics_collections: An optional list of collections that\n      `mean_relative_error` should be added to.\n    updates_collections: An optional list of collections that `update_op` should\n      be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    mean_relative_error: A `Tensor` representing the current mean, the value of\n      `total` divided by `count`.\n    update_op: An operation that increments the `total` and `count` variables\n      appropriately and whose value matches `mean_relative_error`.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, or if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      either `metrics_collections` or `updates_collections` are not a list or\n      tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.mean_relative_error is not supported when eager execution is enabled.')
    (predictions, labels, weights) = _remove_squeezable_dimensions(predictions=predictions, labels=labels, weights=weights)
    (predictions, normalizer) = confusion_matrix.remove_squeezable_dimensions(predictions, normalizer)
    predictions.get_shape().assert_is_compatible_with(normalizer.get_shape())
    relative_errors = array_ops.where(math_ops.equal(normalizer, 0.0), array_ops.zeros_like(labels), math_ops.divide(math_ops.abs(labels - predictions), normalizer))
    return mean(relative_errors, weights, metrics_collections, updates_collections, name or 'mean_relative_error')

@tf_export(v1=['metrics.mean_squared_error'])
def mean_squared_error(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        return 10
    "Computes the mean squared error between the labels and predictions.\n\n  The `mean_squared_error` function creates two local variables,\n  `total` and `count` that are used to compute the mean squared error.\n  This average is weighted by `weights`, and it is ultimately returned as\n  `mean_squared_error`: an idempotent operation that simply divides `total` by\n  `count`.\n\n  For estimation of the metric over a stream of data, the function creates an\n  `update_op` operation that updates these variables and returns the\n  `mean_squared_error`. Internally, a `squared_error` operation computes the\n  element-wise square of the difference between `predictions` and `labels`. Then\n  `update_op` increments `total` with the reduced sum of the product of\n  `weights` and `squared_error`, and it increments `count` with the reduced sum\n  of `weights`.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: A `Tensor` of the same shape as `predictions`.\n    predictions: A `Tensor` of arbitrary shape.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    metrics_collections: An optional list of collections that\n      `mean_squared_error` should be added to.\n    updates_collections: An optional list of collections that `update_op` should\n      be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    mean_squared_error: A `Tensor` representing the current mean, the value of\n      `total` divided by `count`.\n    update_op: An operation that increments the `total` and `count` variables\n      appropriately and whose value matches `mean_squared_error`.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, or if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      either `metrics_collections` or `updates_collections` are not a list or\n      tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.mean_squared_error is not supported when eager execution is enabled.')
    (predictions, labels, weights) = _remove_squeezable_dimensions(predictions=predictions, labels=labels, weights=weights)
    squared_error = math_ops.squared_difference(labels, predictions)
    return mean(squared_error, weights, metrics_collections, updates_collections, name or 'mean_squared_error')

@tf_export(v1=['metrics.mean_tensor'])
def mean_tensor(values, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        i = 10
        return i + 15
    "Computes the element-wise (weighted) mean of the given tensors.\n\n  In contrast to the `mean` function which returns a scalar with the\n  mean,  this function returns an average tensor with the same shape as the\n  input tensors.\n\n  The `mean_tensor` function creates two local variables,\n  `total_tensor` and `count_tensor` that are used to compute the average of\n  `values`. This average is ultimately returned as `mean` which is an idempotent\n  operation that simply divides `total` by `count`.\n\n  For estimation of the metric over a stream of data, the function creates an\n  `update_op` operation that updates these variables and returns the `mean`.\n  `update_op` increments `total` with the reduced sum of the product of `values`\n  and `weights`, and it increments `count` with the reduced sum of `weights`.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    values: A `Tensor` of arbitrary dimensions.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `values`, and must be broadcastable to `values` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `values` dimension).\n    metrics_collections: An optional list of collections that `mean`\n      should be added to.\n    updates_collections: An optional list of collections that `update_op`\n      should be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    mean: A float `Tensor` representing the current mean, the value of `total`\n      divided by `count`.\n    update_op: An operation that increments the `total` and `count` variables\n      appropriately and whose value matches `mean_value`.\n\n  Raises:\n    ValueError: If `weights` is not `None` and its shape doesn't match `values`,\n      or if either `metrics_collections` or `updates_collections` are not a list\n      or tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.mean_tensor is not supported when eager execution is enabled.')
    with variable_scope.variable_scope(name, 'mean', (values, weights)):
        values = math_ops.cast(values, dtypes.float32)
        total = metric_variable(values.get_shape(), dtypes.float32, name='total_tensor')
        count = metric_variable(values.get_shape(), dtypes.float32, name='count_tensor')
        num_values = array_ops.ones_like(values)
        if weights is not None:
            (values, _, weights) = _remove_squeezable_dimensions(predictions=values, labels=None, weights=weights)
            weights = weights_broadcast_ops.broadcast_weights(math_ops.cast(weights, dtypes.float32), values)
            values = math_ops.multiply(values, weights)
            num_values = math_ops.multiply(num_values, weights)
        update_total_op = state_ops.assign_add(total, values)
        with ops.control_dependencies([values]):
            update_count_op = state_ops.assign_add(count, num_values)
        compute_mean = lambda _, t, c: math_ops.div_no_nan(t, math_ops.maximum(c, 0), name='value')
        mean_t = _aggregate_across_replicas(metrics_collections, compute_mean, total, count)
        update_op = math_ops.div_no_nan(update_total_op, math_ops.maximum(update_count_op, 0), name='update_op')
        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)
        return (mean_t, update_op)

@tf_export(v1=['metrics.percentage_below'])
def percentage_below(values, threshold, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        i = 10
        return i + 15
    "Computes the percentage of values less than the given threshold.\n\n  The `percentage_below` function creates two local variables,\n  `total` and `count` that are used to compute the percentage of `values` that\n  fall below `threshold`. This rate is weighted by `weights`, and it is\n  ultimately returned as `percentage` which is an idempotent operation that\n  simply divides `total` by `count`.\n\n  For estimation of the metric over a stream of data, the function creates an\n  `update_op` operation that updates these variables and returns the\n  `percentage`.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    values: A numeric `Tensor` of arbitrary size.\n    threshold: A scalar threshold.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `values`, and must be broadcastable to `values` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `values` dimension).\n    metrics_collections: An optional list of collections that the metric\n      value variable should be added to.\n    updates_collections: An optional list of collections that the metric update\n      ops should be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    percentage: A `Tensor` representing the current mean, the value of `total`\n      divided by `count`.\n    update_op: An operation that increments the `total` and `count` variables\n      appropriately.\n\n  Raises:\n    ValueError: If `weights` is not `None` and its shape doesn't match `values`,\n      or if either `metrics_collections` or `updates_collections` are not a list\n      or tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.percentage_below is not supported when eager execution is enabled.')
    is_below_threshold = math_ops.cast(math_ops.less(values, threshold), dtypes.float32)
    return mean(is_below_threshold, weights, metrics_collections, updates_collections, name or 'percentage_below_threshold')

def _count_condition(values, weights=None, metrics_collections=None, updates_collections=None):
    if False:
        i = 10
        return i + 15
    "Sums the weights of cases where the given values are True.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    values: A `bool` `Tensor` of arbitrary size.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `values`, and must be broadcastable to `values` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `values` dimension).\n    metrics_collections: An optional list of collections that the metric\n      value variable should be added to.\n    updates_collections: An optional list of collections that the metric update\n      ops should be added to.\n\n  Returns:\n    value_tensor: A `Tensor` representing the current value of the metric.\n    update_op: An operation that accumulates the error from a batch of data.\n\n  Raises:\n    ValueError: If `weights` is not `None` and its shape doesn't match `values`,\n      or if either `metrics_collections` or `updates_collections` are not a list\n      or tuple.\n  "
    check_ops.assert_type(values, dtypes.bool)
    count = metric_variable([], dtypes.float32, name='count')
    values = math_ops.cast(values, dtypes.float32)
    if weights is not None:
        with ops.control_dependencies((check_ops.assert_rank_in(weights, (0, array_ops.rank(values))),)):
            weights = math_ops.cast(weights, dtypes.float32)
            values = math_ops.multiply(values, weights)
    value_tensor = _aggregate_variable(count, metrics_collections)
    update_op = state_ops.assign_add(count, math_ops.reduce_sum(values))
    if updates_collections:
        ops.add_to_collections(updates_collections, update_op)
    return (value_tensor, update_op)

@tf_export(v1=['metrics.false_negatives'])
def false_negatives(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    "Computes the total number of false negatives.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: The ground truth values, a `Tensor` whose dimensions must match\n      `predictions`. Will be cast to `bool`.\n    predictions: The predicted values, a `Tensor` of arbitrary dimensions. Will\n      be cast to `bool`.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    metrics_collections: An optional list of collections that the metric\n      value variable should be added to.\n    updates_collections: An optional list of collections that the metric update\n      ops should be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    value_tensor: A `Tensor` representing the current value of the metric.\n    update_op: An operation that accumulates the error from a batch of data.\n\n  Raises:\n    ValueError: If `weights` is not `None` and its shape doesn't match `values`,\n      or if either `metrics_collections` or `updates_collections` are not a list\n      or tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.false_negatives is not supported when eager execution is enabled.')
    with variable_scope.variable_scope(name, 'false_negatives', (predictions, labels, weights)):
        (predictions, labels, weights) = _remove_squeezable_dimensions(predictions=math_ops.cast(predictions, dtype=dtypes.bool), labels=math_ops.cast(labels, dtype=dtypes.bool), weights=weights)
        is_false_negative = math_ops.logical_and(math_ops.equal(labels, True), math_ops.equal(predictions, False))
        return _count_condition(is_false_negative, weights, metrics_collections, updates_collections)

@tf_export(v1=['metrics.false_negatives_at_thresholds'])
def false_negatives_at_thresholds(labels, predictions, thresholds, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        return 10
    "Computes false negatives at provided threshold values.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: A `Tensor` whose shape matches `predictions`. Will be cast to\n      `bool`.\n    predictions: A floating point `Tensor` of arbitrary shape and whose values\n      are in the range `[0, 1]`.\n    thresholds: A python list or tuple of float thresholds in `[0, 1]`.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    metrics_collections: An optional list of collections that `false_negatives`\n      should be added to.\n    updates_collections: An optional list of collections that `update_op` should\n      be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    false_negatives:  A float `Tensor` of shape `[len(thresholds)]`.\n    update_op: An operation that updates the `false_negatives` variable and\n      returns its current value.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, or if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      either `metrics_collections` or `updates_collections` are not a list or\n      tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.false_negatives_at_thresholds is not supported when eager execution is enabled.')
    with variable_scope.variable_scope(name, 'false_negatives', (predictions, labels, weights)):
        (values, update_ops) = _confusion_matrix_at_thresholds(labels, predictions, thresholds, weights=weights, includes=('fn',))
        fn_value = _aggregate_variable(values['fn'], metrics_collections)
        if updates_collections:
            ops.add_to_collections(updates_collections, update_ops['fn'])
        return (fn_value, update_ops['fn'])

@tf_export(v1=['metrics.false_positives'])
def false_positives(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        while True:
            i = 10
    "Sum the weights of false positives.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: The ground truth values, a `Tensor` whose dimensions must match\n      `predictions`. Will be cast to `bool`.\n    predictions: The predicted values, a `Tensor` of arbitrary dimensions. Will\n      be cast to `bool`.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    metrics_collections: An optional list of collections that the metric\n      value variable should be added to.\n    updates_collections: An optional list of collections that the metric update\n      ops should be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    value_tensor: A `Tensor` representing the current value of the metric.\n    update_op: An operation that accumulates the error from a batch of data.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, or if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      either `metrics_collections` or `updates_collections` are not a list or\n      tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.false_positives is not supported when eager execution is enabled.')
    with variable_scope.variable_scope(name, 'false_positives', (predictions, labels, weights)):
        (predictions, labels, weights) = _remove_squeezable_dimensions(predictions=math_ops.cast(predictions, dtype=dtypes.bool), labels=math_ops.cast(labels, dtype=dtypes.bool), weights=weights)
        is_false_positive = math_ops.logical_and(math_ops.equal(labels, False), math_ops.equal(predictions, True))
        return _count_condition(is_false_positive, weights, metrics_collections, updates_collections)

@tf_export(v1=['metrics.false_positives_at_thresholds'])
def false_positives_at_thresholds(labels, predictions, thresholds, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        while True:
            i = 10
    "Computes false positives at provided threshold values.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: A `Tensor` whose shape matches `predictions`. Will be cast to\n      `bool`.\n    predictions: A floating point `Tensor` of arbitrary shape and whose values\n      are in the range `[0, 1]`.\n    thresholds: A python list or tuple of float thresholds in `[0, 1]`.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    metrics_collections: An optional list of collections that `false_positives`\n      should be added to.\n    updates_collections: An optional list of collections that `update_op` should\n      be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    false_positives:  A float `Tensor` of shape `[len(thresholds)]`.\n    update_op: An operation that updates the `false_positives` variable and\n      returns its current value.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, or if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      either `metrics_collections` or `updates_collections` are not a list or\n      tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.false_positives_at_thresholds is not supported when eager execution is enabled.')
    with variable_scope.variable_scope(name, 'false_positives', (predictions, labels, weights)):
        (values, update_ops) = _confusion_matrix_at_thresholds(labels, predictions, thresholds, weights=weights, includes=('fp',))
        fp_value = _aggregate_variable(values['fp'], metrics_collections)
        if updates_collections:
            ops.add_to_collections(updates_collections, update_ops['fp'])
        return (fp_value, update_ops['fp'])

@tf_export(v1=['metrics.true_negatives'])
def true_negatives(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        return 10
    "Sum the weights of true_negatives.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: The ground truth values, a `Tensor` whose dimensions must match\n      `predictions`. Will be cast to `bool`.\n    predictions: The predicted values, a `Tensor` of arbitrary dimensions. Will\n      be cast to `bool`.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    metrics_collections: An optional list of collections that the metric\n      value variable should be added to.\n    updates_collections: An optional list of collections that the metric update\n      ops should be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    value_tensor: A `Tensor` representing the current value of the metric.\n    update_op: An operation that accumulates the error from a batch of data.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, or if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      either `metrics_collections` or `updates_collections` are not a list or\n      tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.true_negatives is not supported when eager execution is enabled.')
    with variable_scope.variable_scope(name, 'true_negatives', (predictions, labels, weights)):
        (predictions, labels, weights) = _remove_squeezable_dimensions(predictions=math_ops.cast(predictions, dtype=dtypes.bool), labels=math_ops.cast(labels, dtype=dtypes.bool), weights=weights)
        is_true_negative = math_ops.logical_and(math_ops.equal(labels, False), math_ops.equal(predictions, False))
        return _count_condition(is_true_negative, weights, metrics_collections, updates_collections)

@tf_export(v1=['metrics.true_negatives_at_thresholds'])
def true_negatives_at_thresholds(labels, predictions, thresholds, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        while True:
            i = 10
    "Computes true negatives at provided threshold values.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: A `Tensor` whose shape matches `predictions`. Will be cast to\n      `bool`.\n    predictions: A floating point `Tensor` of arbitrary shape and whose values\n      are in the range `[0, 1]`.\n    thresholds: A python list or tuple of float thresholds in `[0, 1]`.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    metrics_collections: An optional list of collections that `true_negatives`\n      should be added to.\n    updates_collections: An optional list of collections that `update_op` should\n      be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    true_negatives:  A float `Tensor` of shape `[len(thresholds)]`.\n    update_op: An operation that updates the `true_negatives` variable and\n      returns its current value.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, or if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      either `metrics_collections` or `updates_collections` are not a list or\n      tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.true_negatives_at_thresholds is not supported when eager execution is enabled.')
    with variable_scope.variable_scope(name, 'true_negatives', (predictions, labels, weights)):
        (values, update_ops) = _confusion_matrix_at_thresholds(labels, predictions, thresholds, weights=weights, includes=('tn',))
        tn_value = _aggregate_variable(values['tn'], metrics_collections)
        if updates_collections:
            ops.add_to_collections(updates_collections, update_ops['tn'])
        return (tn_value, update_ops['tn'])

@tf_export(v1=['metrics.true_positives'])
def true_positives(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        return 10
    "Sum the weights of true_positives.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: The ground truth values, a `Tensor` whose dimensions must match\n      `predictions`. Will be cast to `bool`.\n    predictions: The predicted values, a `Tensor` of arbitrary dimensions. Will\n      be cast to `bool`.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    metrics_collections: An optional list of collections that the metric\n      value variable should be added to.\n    updates_collections: An optional list of collections that the metric update\n      ops should be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    value_tensor: A `Tensor` representing the current value of the metric.\n    update_op: An operation that accumulates the error from a batch of data.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, or if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      either `metrics_collections` or `updates_collections` are not a list or\n      tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.true_positives is not supported when eager execution is enabled.')
    with variable_scope.variable_scope(name, 'true_positives', (predictions, labels, weights)):
        (predictions, labels, weights) = _remove_squeezable_dimensions(predictions=math_ops.cast(predictions, dtype=dtypes.bool), labels=math_ops.cast(labels, dtype=dtypes.bool), weights=weights)
        is_true_positive = math_ops.logical_and(math_ops.equal(labels, True), math_ops.equal(predictions, True))
        return _count_condition(is_true_positive, weights, metrics_collections, updates_collections)

@tf_export(v1=['metrics.true_positives_at_thresholds'])
def true_positives_at_thresholds(labels, predictions, thresholds, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    "Computes true positives at provided threshold values.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: A `Tensor` whose shape matches `predictions`. Will be cast to\n      `bool`.\n    predictions: A floating point `Tensor` of arbitrary shape and whose values\n      are in the range `[0, 1]`.\n    thresholds: A python list or tuple of float thresholds in `[0, 1]`.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    metrics_collections: An optional list of collections that `true_positives`\n      should be added to.\n    updates_collections: An optional list of collections that `update_op` should\n      be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    true_positives:  A float `Tensor` of shape `[len(thresholds)]`.\n    update_op: An operation that updates the `true_positives` variable and\n      returns its current value.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, or if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      either `metrics_collections` or `updates_collections` are not a list or\n      tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.true_positives_at_thresholds is not supported when eager execution is enabled.')
    with variable_scope.variable_scope(name, 'true_positives', (predictions, labels, weights)):
        (values, update_ops) = _confusion_matrix_at_thresholds(labels, predictions, thresholds, weights=weights, includes=('tp',))
        tp_value = _aggregate_variable(values['tp'], metrics_collections)
        if updates_collections:
            ops.add_to_collections(updates_collections, update_ops['tp'])
        return (tp_value, update_ops['tp'])

@tf_export(v1=['metrics.precision'])
def precision(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        return 10
    "Computes the precision of the predictions with respect to the labels.\n\n  The `precision` function creates two local variables,\n  `true_positives` and `false_positives`, that are used to compute the\n  precision. This value is ultimately returned as `precision`, an idempotent\n  operation that simply divides `true_positives` by the sum of `true_positives`\n  and `false_positives`.\n\n  For estimation of the metric over a stream of data, the function creates an\n  `update_op` operation that updates these variables and returns the\n  `precision`. `update_op` weights each prediction by the corresponding value in\n  `weights`.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: The ground truth values, a `Tensor` whose dimensions must match\n      `predictions`. Will be cast to `bool`.\n    predictions: The predicted values, a `Tensor` of arbitrary dimensions. Will\n      be cast to `bool`.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    metrics_collections: An optional list of collections that `precision` should\n      be added to.\n    updates_collections: An optional list of collections that `update_op` should\n      be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    precision: Scalar float `Tensor` with the value of `true_positives`\n      divided by the sum of `true_positives` and `false_positives`.\n    update_op: `Operation` that increments `true_positives` and\n      `false_positives` variables appropriately and whose value matches\n      `precision`.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, or if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      either `metrics_collections` or `updates_collections` are not a list or\n      tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.precision is not supported when eager execution is enabled.')
    with variable_scope.variable_scope(name, 'precision', (predictions, labels, weights)):
        (predictions, labels, weights) = _remove_squeezable_dimensions(predictions=math_ops.cast(predictions, dtype=dtypes.bool), labels=math_ops.cast(labels, dtype=dtypes.bool), weights=weights)
        (true_p, true_positives_update_op) = true_positives(labels, predictions, weights, metrics_collections=None, updates_collections=None, name=None)
        (false_p, false_positives_update_op) = false_positives(labels, predictions, weights, metrics_collections=None, updates_collections=None, name=None)

        def compute_precision(tp, fp, name):
            if False:
                print('Hello World!')
            return array_ops.where(math_ops.greater(tp + fp, 0), math_ops.divide(tp, tp + fp), 0, name)

        def once_across_replicas(_, true_p, false_p):
            if False:
                return 10
            return compute_precision(true_p, false_p, 'value')
        p = _aggregate_across_replicas(metrics_collections, once_across_replicas, true_p, false_p)
        update_op = compute_precision(true_positives_update_op, false_positives_update_op, 'update_op')
        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)
        return (p, update_op)

@tf_export(v1=['metrics.precision_at_thresholds'])
def precision_at_thresholds(labels, predictions, thresholds, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        i = 10
        return i + 15
    "Computes precision values for different `thresholds` on `predictions`.\n\n  The `precision_at_thresholds` function creates four local variables,\n  `true_positives`, `true_negatives`, `false_positives` and `false_negatives`\n  for various values of thresholds. `precision[i]` is defined as the total\n  weight of values in `predictions` above `thresholds[i]` whose corresponding\n  entry in `labels` is `True`, divided by the total weight of values in\n  `predictions` above `thresholds[i]` (`true_positives[i] / (true_positives[i] +\n  false_positives[i])`).\n\n  For estimation of the metric over a stream of data, the function creates an\n  `update_op` operation that updates these variables and returns the\n  `precision`.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: The ground truth values, a `Tensor` whose dimensions must match\n      `predictions`. Will be cast to `bool`.\n    predictions: A floating point `Tensor` of arbitrary shape and whose values\n      are in the range `[0, 1]`.\n    thresholds: A python list or tuple of float thresholds in `[0, 1]`.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    metrics_collections: An optional list of collections that `auc` should be\n      added to.\n    updates_collections: An optional list of collections that `update_op` should\n      be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    precision: A float `Tensor` of shape `[len(thresholds)]`.\n    update_op: An operation that increments the `true_positives`,\n      `true_negatives`, `false_positives` and `false_negatives` variables that\n      are used in the computation of `precision`.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, or if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      either `metrics_collections` or `updates_collections` are not a list or\n      tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.precision_at_thresholds is not supported when eager execution is enabled.')
    with variable_scope.variable_scope(name, 'precision_at_thresholds', (predictions, labels, weights)):
        (values, update_ops) = _confusion_matrix_at_thresholds(labels, predictions, thresholds, weights, includes=('tp', 'fp'))
        epsilon = 1e-07

        def compute_precision(tp, fp, name):
            if False:
                for i in range(10):
                    print('nop')
            return math_ops.divide(tp, epsilon + tp + fp, name='precision_' + name)

        def precision_across_replicas(_, values):
            if False:
                i = 10
                return i + 15
            return compute_precision(values['tp'], values['fp'], 'value')
        prec = _aggregate_across_replicas(metrics_collections, precision_across_replicas, values)
        update_op = compute_precision(update_ops['tp'], update_ops['fp'], 'update_op')
        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)
        return (prec, update_op)

@tf_export(v1=['metrics.recall'])
def recall(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        i = 10
        return i + 15
    "Computes the recall of the predictions with respect to the labels.\n\n  The `recall` function creates two local variables, `true_positives`\n  and `false_negatives`, that are used to compute the recall. This value is\n  ultimately returned as `recall`, an idempotent operation that simply divides\n  `true_positives` by the sum of `true_positives` and `false_negatives`.\n\n  For estimation of the metric over a stream of data, the function creates an\n  `update_op` that updates these variables and returns the `recall`. `update_op`\n  weights each prediction by the corresponding value in `weights`.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: The ground truth values, a `Tensor` whose dimensions must match\n      `predictions`. Will be cast to `bool`.\n    predictions: The predicted values, a `Tensor` of arbitrary dimensions. Will\n      be cast to `bool`.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    metrics_collections: An optional list of collections that `recall` should\n      be added to.\n    updates_collections: An optional list of collections that `update_op` should\n      be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    recall: Scalar float `Tensor` with the value of `true_positives` divided\n      by the sum of `true_positives` and `false_negatives`.\n    update_op: `Operation` that increments `true_positives` and\n      `false_negatives` variables appropriately and whose value matches\n      `recall`.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, or if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      either `metrics_collections` or `updates_collections` are not a list or\n      tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.recall is not supported is not supported when eager execution is enabled.')
    with variable_scope.variable_scope(name, 'recall', (predictions, labels, weights)):
        (predictions, labels, weights) = _remove_squeezable_dimensions(predictions=math_ops.cast(predictions, dtype=dtypes.bool), labels=math_ops.cast(labels, dtype=dtypes.bool), weights=weights)
        (true_p, true_positives_update_op) = true_positives(labels, predictions, weights, metrics_collections=None, updates_collections=None, name=None)
        (false_n, false_negatives_update_op) = false_negatives(labels, predictions, weights, metrics_collections=None, updates_collections=None, name=None)

        def compute_recall(true_p, false_n, name):
            if False:
                for i in range(10):
                    print('nop')
            return array_ops.where(math_ops.greater(true_p + false_n, 0), math_ops.divide(true_p, true_p + false_n), 0, name)

        def once_across_replicas(_, true_p, false_n):
            if False:
                return 10
            return compute_recall(true_p, false_n, 'value')
        rec = _aggregate_across_replicas(metrics_collections, once_across_replicas, true_p, false_n)
        update_op = compute_recall(true_positives_update_op, false_negatives_update_op, 'update_op')
        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)
        return (rec, update_op)

def _at_k_name(name, k=None, class_id=None):
    if False:
        for i in range(10):
            print('nop')
    if k is not None:
        name = '%s_at_%d' % (name, k)
    else:
        name = '%s_at_k' % name
    if class_id is not None:
        name = '%s_class%d' % (name, class_id)
    return name

def _select_class_id(ids, selected_id):
    if False:
        print('Hello World!')
    'Filter all but `selected_id` out of `ids`.\n\n  Args:\n    ids: `int64` `Tensor` or `SparseTensor` of IDs.\n    selected_id: Int id to select.\n\n  Returns:\n    `SparseTensor` of same dimensions as `ids`. This contains only the entries\n    equal to `selected_id`.\n  '
    ids = sparse_tensor.convert_to_tensor_or_sparse_tensor(ids)
    if isinstance(ids, sparse_tensor.SparseTensor):
        return sparse_ops.sparse_retain(ids, math_ops.equal(ids.values, selected_id))
    ids_shape = array_ops.shape(ids, out_type=dtypes.int64)
    ids_last_dim = array_ops.size(ids_shape) - 1
    filled_selected_id_shape = math_ops.reduced_shape(ids_shape, array_ops.reshape(ids_last_dim, [1]))
    filled_selected_id = array_ops.fill(filled_selected_id_shape, math_ops.cast(selected_id, dtypes.int64))
    result = sets.set_intersection(filled_selected_id, ids)
    return sparse_tensor.SparseTensor(indices=result.indices, values=result.values, dense_shape=ids_shape)

def _maybe_select_class_id(labels, predictions_idx, selected_id=None):
    if False:
        i = 10
        return i + 15
    'If class ID is specified, filter all other classes.\n\n  Args:\n    labels: `int64` `Tensor` or `SparseTensor` with shape\n      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of\n      target classes for the associated prediction. Commonly, N=1 and `labels`\n      has shape [batch_size, num_labels]. [D1, ... DN] must match\n      `predictions_idx`.\n    predictions_idx: `int64` `Tensor` of class IDs, with shape [D1, ... DN, k]\n      where N >= 1. Commonly, N=1 and `predictions_idx` has shape\n      [batch size, k].\n    selected_id: Int id to select.\n\n  Returns:\n    Tuple of `labels` and `predictions_idx`, possibly with classes removed.\n  '
    if selected_id is None:
        return (labels, predictions_idx)
    return (_select_class_id(labels, selected_id), _select_class_id(predictions_idx, selected_id))

def _sparse_true_positive_at_k(labels, predictions_idx, class_id=None, weights=None, name=None):
    if False:
        return 10
    'Calculates true positives for recall@k and precision@k.\n\n  If `class_id` is specified, calculate binary true positives for `class_id`\n      only.\n  If `class_id` is not specified, calculate metrics for `k` predicted vs\n      `n` label classes, where `n` is the 2nd dimension of `labels_sparse`.\n\n  Args:\n    labels: `int64` `Tensor` or `SparseTensor` with shape\n      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of\n      target classes for the associated prediction. Commonly, N=1 and `labels`\n      has shape [batch_size, num_labels]. [D1, ... DN] must match\n      `predictions_idx`.\n    predictions_idx: 1-D or higher `int64` `Tensor` with last dimension `k`,\n      top `k` predicted classes. For rank `n`, the first `n-1` dimensions must\n      match `labels`.\n    class_id: Class for which we want binary metrics.\n    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of\n      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all\n      dimensions must be either `1`, or the same as the corresponding `labels`\n      dimension).\n    name: Name of operation.\n\n  Returns:\n    A [D1, ... DN] `Tensor` of true positive counts.\n  '
    with ops.name_scope(name, 'true_positives', (predictions_idx, labels, weights)):
        (labels, predictions_idx) = _maybe_select_class_id(labels, predictions_idx, class_id)
        tp = sets.set_size(sets.set_intersection(predictions_idx, labels))
        tp = math_ops.cast(tp, dtypes.float64)
        if weights is not None:
            with ops.control_dependencies((weights_broadcast_ops.assert_broadcastable(weights, tp),)):
                weights = math_ops.cast(weights, dtypes.float64)
                tp = math_ops.multiply(tp, weights)
        return tp

def _streaming_sparse_true_positive_at_k(labels, predictions_idx, k=None, class_id=None, weights=None, name=None):
    if False:
        i = 10
        return i + 15
    'Calculates weighted per step true positives for recall@k and precision@k.\n\n  If `class_id` is specified, calculate binary true positives for `class_id`\n      only.\n  If `class_id` is not specified, calculate metrics for `k` predicted vs\n      `n` label classes, where `n` is the 2nd dimension of `labels`.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: `int64` `Tensor` or `SparseTensor` with shape\n      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of\n      target classes for the associated prediction. Commonly, N=1 and `labels`\n      has shape [batch_size, num_labels]. [D1, ... DN] must match\n      `predictions_idx`.\n    predictions_idx: 1-D or higher `int64` `Tensor` with last dimension `k`,\n      top `k` predicted classes. For rank `n`, the first `n-1` dimensions must\n      match `labels`.\n    k: Integer, k for @k metric. This is only used for default op name.\n    class_id: Class for which we want binary metrics.\n    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of\n      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all\n      dimensions must be either `1`, or the same as the corresponding `labels`\n      dimension).\n    name: Name of new variable, and namespace for other dependent ops.\n\n  Returns:\n    A tuple of `Variable` and update `Operation`.\n\n  Raises:\n    ValueError: If `weights` is not `None` and has an incompatible shape.\n  '
    with ops.name_scope(name, _at_k_name('true_positive', k, class_id=class_id), (predictions_idx, labels, weights)) as scope:
        tp = _sparse_true_positive_at_k(predictions_idx=predictions_idx, labels=labels, class_id=class_id, weights=weights)
        batch_total_tp = math_ops.cast(math_ops.reduce_sum(tp), dtypes.float64)
        var = metric_variable([], dtypes.float64, name=scope)
        return (var, state_ops.assign_add(var, batch_total_tp, name='update'))

def _sparse_false_negative_at_k(labels, predictions_idx, class_id=None, weights=None):
    if False:
        while True:
            i = 10
    'Calculates false negatives for recall@k.\n\n  If `class_id` is specified, calculate binary true positives for `class_id`\n      only.\n  If `class_id` is not specified, calculate metrics for `k` predicted vs\n      `n` label classes, where `n` is the 2nd dimension of `labels_sparse`.\n\n  Args:\n    labels: `int64` `Tensor` or `SparseTensor` with shape\n      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of\n      target classes for the associated prediction. Commonly, N=1 and `labels`\n      has shape [batch_size, num_labels]. [D1, ... DN] must match\n      `predictions_idx`.\n    predictions_idx: 1-D or higher `int64` `Tensor` with last dimension `k`,\n      top `k` predicted classes. For rank `n`, the first `n-1` dimensions must\n      match `labels`.\n    class_id: Class for which we want binary metrics.\n    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of\n      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all\n      dimensions must be either `1`, or the same as the corresponding `labels`\n      dimension).\n\n  Returns:\n    A [D1, ... DN] `Tensor` of false negative counts.\n  '
    with ops.name_scope(None, 'false_negatives', (predictions_idx, labels, weights)):
        (labels, predictions_idx) = _maybe_select_class_id(labels, predictions_idx, class_id)
        fn = sets.set_size(sets.set_difference(predictions_idx, labels, aminusb=False))
        fn = math_ops.cast(fn, dtypes.float64)
        if weights is not None:
            with ops.control_dependencies((weights_broadcast_ops.assert_broadcastable(weights, fn),)):
                weights = math_ops.cast(weights, dtypes.float64)
                fn = math_ops.multiply(fn, weights)
        return fn

def _streaming_sparse_false_negative_at_k(labels, predictions_idx, k, class_id=None, weights=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Calculates weighted per step false negatives for recall@k.\n\n  If `class_id` is specified, calculate binary true positives for `class_id`\n      only.\n  If `class_id` is not specified, calculate metrics for `k` predicted vs\n      `n` label classes, where `n` is the 2nd dimension of `labels`.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: `int64` `Tensor` or `SparseTensor` with shape\n      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of\n      target classes for the associated prediction. Commonly, N=1 and `labels`\n      has shape [batch_size, num_labels]. [D1, ... DN] must match\n      `predictions_idx`.\n    predictions_idx: 1-D or higher `int64` `Tensor` with last dimension `k`,\n      top `k` predicted classes. For rank `n`, the first `n-1` dimensions must\n      match `labels`.\n    k: Integer, k for @k metric. This is only used for default op name.\n    class_id: Class for which we want binary metrics.\n    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of\n      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all\n      dimensions must be either `1`, or the same as the corresponding `labels`\n      dimension).\n    name: Name of new variable, and namespace for other dependent ops.\n\n  Returns:\n    A tuple of `Variable` and update `Operation`.\n\n  Raises:\n    ValueError: If `weights` is not `None` and has an incompatible shape.\n  '
    with ops.name_scope(name, _at_k_name('false_negative', k, class_id=class_id), (predictions_idx, labels, weights)) as scope:
        fn = _sparse_false_negative_at_k(predictions_idx=predictions_idx, labels=labels, class_id=class_id, weights=weights)
        batch_total_fn = math_ops.cast(math_ops.reduce_sum(fn), dtypes.float64)
        var = metric_variable([], dtypes.float64, name=scope)
        return (var, state_ops.assign_add(var, batch_total_fn, name='update'))

@tf_export(v1=['metrics.recall_at_k'])
def recall_at_k(labels, predictions, k, class_id=None, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        while True:
            i = 10
    "Computes recall@k of the predictions with respect to sparse labels.\n\n  If `class_id` is specified, we calculate recall by considering only the\n      entries in the batch for which `class_id` is in the label, and computing\n      the fraction of them for which `class_id` is in the top-k `predictions`.\n  If `class_id` is not specified, we'll calculate recall as how often on\n      average a class among the labels of a batch entry is in the top-k\n      `predictions`.\n\n  `sparse_recall_at_k` creates two local variables,\n  `true_positive_at_<k>` and `false_negative_at_<k>`, that are used to compute\n  the recall_at_k frequency. This frequency is ultimately returned as\n  `recall_at_<k>`: an idempotent operation that simply divides\n  `true_positive_at_<k>` by total (`true_positive_at_<k>` +\n  `false_negative_at_<k>`).\n\n  For estimation of the metric over a stream of data, the function creates an\n  `update_op` operation that updates these variables and returns the\n  `recall_at_<k>`. Internally, a `top_k` operation computes a `Tensor`\n  indicating the top `k` `predictions`. Set operations applied to `top_k` and\n  `labels` calculate the true positives and false negatives weighted by\n  `weights`. Then `update_op` increments `true_positive_at_<k>` and\n  `false_negative_at_<k>` using these values.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: `int64` `Tensor` or `SparseTensor` with shape\n      [D1, ... DN, num_labels] or [D1, ... DN], where the latter implies\n      num_labels=1. N >= 1 and num_labels is the number of target classes for\n      the associated prediction. Commonly, N=1 and `labels` has shape\n      [batch_size, num_labels]. [D1, ... DN] must match `predictions`. Values\n      should be in range [0, num_classes), where num_classes is the last\n      dimension of `predictions`. Values outside this range always count\n      towards `false_negative_at_<k>`.\n    predictions: Float `Tensor` with shape [D1, ... DN, num_classes] where\n      N >= 1. Commonly, N=1 and predictions has shape [batch size, num_classes].\n      The final dimension contains the logit values for each class. [D1, ... DN]\n      must match `labels`.\n    k: Integer, k for @k metric.\n    class_id: Integer class ID for which we want binary metrics. This should be\n      in range [0, num_classes), where num_classes is the last dimension of\n      `predictions`. If class_id is outside this range, the method returns NAN.\n    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of\n      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all\n      dimensions must be either `1`, or the same as the corresponding `labels`\n      dimension).\n    metrics_collections: An optional list of collections that values should\n      be added to.\n    updates_collections: An optional list of collections that updates should\n      be added to.\n    name: Name of new update operation, and namespace for other dependent ops.\n\n  Returns:\n    recall: Scalar `float64` `Tensor` with the value of `true_positives` divided\n      by the sum of `true_positives` and `false_negatives`.\n    update_op: `Operation` that increments `true_positives` and\n      `false_negatives` variables appropriately, and whose value matches\n      `recall`.\n\n  Raises:\n    ValueError: If `weights` is not `None` and its shape doesn't match\n    `predictions`, or if either `metrics_collections` or `updates_collections`\n    are not a list or tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.recall_at_k is not supported when eager execution is enabled.')
    with ops.name_scope(name, _at_k_name('recall', k, class_id=class_id), (predictions, labels, weights)) as scope:
        (_, top_k_idx) = nn.top_k(predictions, k)
        return recall_at_top_k(labels=labels, predictions_idx=top_k_idx, k=k, class_id=class_id, weights=weights, metrics_collections=metrics_collections, updates_collections=updates_collections, name=scope)

@tf_export(v1=['metrics.recall_at_top_k'])
def recall_at_top_k(labels, predictions_idx, k=None, class_id=None, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        print('Hello World!')
    "Computes recall@k of top-k predictions with respect to sparse labels.\n\n  Differs from `recall_at_k` in that predictions must be in the form of top `k`\n  class indices, whereas `recall_at_k` expects logits. Refer to `recall_at_k`\n  for more details.\n\n  Args:\n    labels: `int64` `Tensor` or `SparseTensor` with shape\n      [D1, ... DN, num_labels] or [D1, ... DN], where the latter implies\n      num_labels=1. N >= 1 and num_labels is the number of target classes for\n      the associated prediction. Commonly, N=1 and `labels` has shape\n      [batch_size, num_labels]. [D1, ... DN] must match `predictions`. Values\n      should be in range [0, num_classes), where num_classes is the last\n      dimension of `predictions`. Values outside this range always count\n      towards `false_negative_at_<k>`.\n    predictions_idx: Integer `Tensor` with shape [D1, ... DN, k] where N >= 1.\n      Commonly, N=1 and predictions has shape [batch size, k]. The final\n      dimension contains the top `k` predicted class indices. [D1, ... DN] must\n      match `labels`.\n    k: Integer, k for @k metric. Only used for the default op name.\n    class_id: Integer class ID for which we want binary metrics. This should be\n      in range [0, num_classes), where num_classes is the last dimension of\n      `predictions`. If class_id is outside this range, the method returns NAN.\n    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of\n      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all\n      dimensions must be either `1`, or the same as the corresponding `labels`\n      dimension).\n    metrics_collections: An optional list of collections that values should\n      be added to.\n    updates_collections: An optional list of collections that updates should\n      be added to.\n    name: Name of new update operation, and namespace for other dependent ops.\n\n  Returns:\n    recall: Scalar `float64` `Tensor` with the value of `true_positives` divided\n      by the sum of `true_positives` and `false_negatives`.\n    update_op: `Operation` that increments `true_positives` and\n      `false_negatives` variables appropriately, and whose value matches\n      `recall`.\n\n  Raises:\n    ValueError: If `weights` is not `None` and its shape doesn't match\n    `predictions`, or if either `metrics_collections` or `updates_collections`\n    are not a list or tuple.\n  "
    with ops.name_scope(name, _at_k_name('recall', k, class_id=class_id), (predictions_idx, labels, weights)) as scope:
        labels = _maybe_expand_labels(labels, predictions_idx)
        top_k_idx = math_ops.cast(predictions_idx, dtypes.int64)
        (tp, tp_update) = _streaming_sparse_true_positive_at_k(predictions_idx=top_k_idx, labels=labels, k=k, class_id=class_id, weights=weights)
        (fn, fn_update) = _streaming_sparse_false_negative_at_k(predictions_idx=top_k_idx, labels=labels, k=k, class_id=class_id, weights=weights)

        def compute_recall(_, tp, fn):
            if False:
                print('Hello World!')
            return math_ops.divide(tp, math_ops.add(tp, fn), name=scope)
        metric = _aggregate_across_replicas(metrics_collections, compute_recall, tp, fn)
        update = math_ops.divide(tp_update, math_ops.add(tp_update, fn_update), name='update')
        if updates_collections:
            ops.add_to_collections(updates_collections, update)
        return (metric, update)

@tf_export(v1=['metrics.recall_at_thresholds'])
def recall_at_thresholds(labels, predictions, thresholds, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        print('Hello World!')
    "Computes various recall values for different `thresholds` on `predictions`.\n\n  The `recall_at_thresholds` function creates four local variables,\n  `true_positives`, `true_negatives`, `false_positives` and `false_negatives`\n  for various values of thresholds. `recall[i]` is defined as the total weight\n  of values in `predictions` above `thresholds[i]` whose corresponding entry in\n  `labels` is `True`, divided by the total weight of `True` values in `labels`\n  (`true_positives[i] / (true_positives[i] + false_negatives[i])`).\n\n  For estimation of the metric over a stream of data, the function creates an\n  `update_op` operation that updates these variables and returns the `recall`.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: The ground truth values, a `Tensor` whose dimensions must match\n      `predictions`. Will be cast to `bool`.\n    predictions: A floating point `Tensor` of arbitrary shape and whose values\n      are in the range `[0, 1]`.\n    thresholds: A python list or tuple of float thresholds in `[0, 1]`.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    metrics_collections: An optional list of collections that `recall` should be\n      added to.\n    updates_collections: An optional list of collections that `update_op` should\n      be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    recall: A float `Tensor` of shape `[len(thresholds)]`.\n    update_op: An operation that increments the `true_positives`,\n      `true_negatives`, `false_positives` and `false_negatives` variables that\n      are used in the computation of `recall`.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, or if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      either `metrics_collections` or `updates_collections` are not a list or\n      tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.recall_at_thresholds is not supported when eager execution is enabled.')
    with variable_scope.variable_scope(name, 'recall_at_thresholds', (predictions, labels, weights)):
        (values, update_ops) = _confusion_matrix_at_thresholds(labels, predictions, thresholds, weights, includes=('tp', 'fn'))
        epsilon = 1e-07

        def compute_recall(tp, fn, name):
            if False:
                i = 10
                return i + 15
            return math_ops.divide(tp, epsilon + tp + fn, name='recall_' + name)

        def recall_across_replicas(_, values):
            if False:
                return 10
            return compute_recall(values['tp'], values['fn'], 'value')
        rec = _aggregate_across_replicas(metrics_collections, recall_across_replicas, values)
        update_op = compute_recall(update_ops['tp'], update_ops['fn'], 'update_op')
        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)
        return (rec, update_op)

@tf_export(v1=['metrics.root_mean_squared_error'])
def root_mean_squared_error(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    "Computes the root mean squared error between the labels and predictions.\n\n  The `root_mean_squared_error` function creates two local variables,\n  `total` and `count` that are used to compute the root mean squared error.\n  This average is weighted by `weights`, and it is ultimately returned as\n  `root_mean_squared_error`: an idempotent operation that takes the square root\n  of the division of `total` by `count`.\n\n  For estimation of the metric over a stream of data, the function creates an\n  `update_op` operation that updates these variables and returns the\n  `root_mean_squared_error`. Internally, a `squared_error` operation computes\n  the element-wise square of the difference between `predictions` and `labels`.\n  Then `update_op` increments `total` with the reduced sum of the product of\n  `weights` and `squared_error`, and it increments `count` with the reduced sum\n  of `weights`.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: A `Tensor` of the same shape as `predictions`.\n    predictions: A `Tensor` of arbitrary shape.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    metrics_collections: An optional list of collections that\n      `root_mean_squared_error` should be added to.\n    updates_collections: An optional list of collections that `update_op` should\n      be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    root_mean_squared_error: A `Tensor` representing the current mean, the value\n      of `total` divided by `count`.\n    update_op: An operation that increments the `total` and `count` variables\n      appropriately and whose value matches `root_mean_squared_error`.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, or if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      either `metrics_collections` or `updates_collections` are not a list or\n      tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.root_mean_squared_error is not supported when eager execution is enabled.')
    (predictions, labels, weights) = _remove_squeezable_dimensions(predictions=predictions, labels=labels, weights=weights)
    (mse, update_mse_op) = mean_squared_error(labels, predictions, weights, None, None, name or 'root_mean_squared_error')
    once_across_replicas = lambda _, mse: math_ops.sqrt(mse)
    rmse = _aggregate_across_replicas(metrics_collections, once_across_replicas, mse)
    update_rmse_op = math_ops.sqrt(update_mse_op)
    if updates_collections:
        ops.add_to_collections(updates_collections, update_rmse_op)
    return (rmse, update_rmse_op)

@tf_export(v1=['metrics.sensitivity_at_specificity'])
def sensitivity_at_specificity(labels, predictions, specificity, weights=None, num_thresholds=200, metrics_collections=None, updates_collections=None, name=None):
    if False:
        return 10
    "Computes the specificity at a given sensitivity.\n\n  The `sensitivity_at_specificity` function creates four local\n  variables, `true_positives`, `true_negatives`, `false_positives` and\n  `false_negatives` that are used to compute the sensitivity at the given\n  specificity value. The threshold for the given specificity value is computed\n  and used to evaluate the corresponding sensitivity.\n\n  For estimation of the metric over a stream of data, the function creates an\n  `update_op` operation that updates these variables and returns the\n  `sensitivity`. `update_op` increments the `true_positives`, `true_negatives`,\n  `false_positives` and `false_negatives` counts with the weight of each case\n  found in the `predictions` and `labels`.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  For additional information about specificity and sensitivity, see the\n  following: https://en.wikipedia.org/wiki/Sensitivity_and_specificity\n\n  Args:\n    labels: The ground truth values, a `Tensor` whose dimensions must match\n      `predictions`. Will be cast to `bool`.\n    predictions: A floating point `Tensor` of arbitrary shape and whose values\n      are in the range `[0, 1]`.\n    specificity: A scalar value in range `[0, 1]`.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    num_thresholds: The number of thresholds to use for matching the given\n      specificity.\n    metrics_collections: An optional list of collections that `sensitivity`\n      should be added to.\n    updates_collections: An optional list of collections that `update_op` should\n      be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    sensitivity: A scalar `Tensor` representing the sensitivity at the given\n      `specificity` value.\n    update_op: An operation that increments the `true_positives`,\n      `true_negatives`, `false_positives` and `false_negatives` variables\n      appropriately and whose value matches `sensitivity`.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      `specificity` is not between 0 and 1, or if either `metrics_collections`\n      or `updates_collections` are not a list or tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.sensitivity_at_specificity is not supported when eager execution is enabled.')
    if specificity < 0 or specificity > 1:
        raise ValueError(f'`specificity` must be in the range [0, 1]. Currently, `specificity` got {specificity}.')
    with variable_scope.variable_scope(name, 'sensitivity_at_specificity', (predictions, labels, weights)):
        kepsilon = 1e-07
        thresholds = [(i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)]
        thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]
        (values, update_ops) = _confusion_matrix_at_thresholds(labels, predictions, thresholds, weights)

        def compute_sensitivity_at_specificity(tp, tn, fp, fn, name):
            if False:
                while True:
                    i = 10
            specificities = math_ops.divide(tn, tn + fp + kepsilon)
            tf_index = math_ops.argmin(math_ops.abs(specificities - specificity), 0)
            tf_index = math_ops.cast(tf_index, dtypes.int32)
            return math_ops.divide(tp[tf_index], tp[tf_index] + fn[tf_index] + kepsilon, name)

        def sensitivity_across_replicas(_, values):
            if False:
                print('Hello World!')
            return compute_sensitivity_at_specificity(values['tp'], values['tn'], values['fp'], values['fn'], 'value')
        sensitivity = _aggregate_across_replicas(metrics_collections, sensitivity_across_replicas, values)
        update_op = compute_sensitivity_at_specificity(update_ops['tp'], update_ops['tn'], update_ops['fp'], update_ops['fn'], 'update_op')
        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)
        return (sensitivity, update_op)

def _expand_and_tile(tensor, multiple, dim=0, name=None):
    if False:
        print('Hello World!')
    'Slice `tensor` shape in 2, then tile along the sliced dimension.\n\n  A new dimension is inserted in shape of `tensor` before `dim`, then values are\n  tiled `multiple` times along the new dimension.\n\n  Args:\n    tensor: Input `Tensor` or `SparseTensor`.\n    multiple: Integer, number of times to tile.\n    dim: Integer, dimension along which to tile.\n    name: Name of operation.\n\n  Returns:\n    `Tensor` result of expanding and tiling `tensor`.\n\n  Raises:\n    ValueError: if `multiple` is less than 1, or `dim` is not in\n    `[-rank(tensor), rank(tensor)]`.\n  '
    if multiple < 1:
        raise ValueError(f'Invalid argument multiple={multiple} for expand_and_tile  call. `multiple` must be an integer > 0')
    with ops.name_scope(name, 'expand_and_tile', (tensor, multiple, dim)) as scope:
        tensor = sparse_tensor.convert_to_tensor_or_sparse_tensor(tensor)
        if isinstance(tensor, sparse_tensor.SparseTensor):
            if dim < 0:
                expand_dims = array_ops.reshape(array_ops.size(tensor.dense_shape) + dim, [1])
            else:
                expand_dims = [dim]
            expanded_shape = array_ops.concat((array_ops.slice(tensor.dense_shape, [0], expand_dims), [1], array_ops.slice(tensor.dense_shape, expand_dims, [-1])), 0, name='expanded_shape')
            expanded = sparse_ops.sparse_reshape(tensor, shape=expanded_shape, name='expand')
            if multiple == 1:
                return expanded
            return sparse_ops.sparse_concat(dim - 1 if dim < 0 else dim, [expanded] * multiple, name=scope)
        expanded = array_ops.expand_dims(tensor, dim if dim >= 0 else dim - 1, name='expand')
        if multiple == 1:
            return expanded
        ones = array_ops.ones_like(array_ops.shape(tensor))
        tile_multiples = array_ops.concat((ones[:dim], (multiple,), ones[dim:]), 0, name='multiples')
        return array_ops.tile(expanded, tile_multiples, name=scope)

def _num_relevant(labels, k):
    if False:
        return 10
    'Computes number of relevant values for each row in labels.\n\n  For labels with shape [D1, ... DN, num_labels], this is the minimum of\n  `num_labels` and `k`.\n\n  Args:\n    labels: `int64` `Tensor` or `SparseTensor` with shape\n      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of\n      target classes for the associated prediction. Commonly, N=1 and `labels`\n      has shape [batch_size, num_labels].\n    k: Integer, k for @k metric.\n\n  Returns:\n    Integer `Tensor` of shape [D1, ... DN], where each value is the number of\n    relevant values for that row.\n\n  Raises:\n    ValueError: if inputs have invalid dtypes or values.\n  '
    if k < 1:
        raise ValueError(f'Invalid k={k}')
    with ops.name_scope(None, 'num_relevant', (labels,)) as scope:
        labels = sparse_tensor.convert_to_tensor_or_sparse_tensor(labels)
        if isinstance(labels, sparse_tensor.SparseTensor):
            return math_ops.minimum(sets.set_size(labels), k, name=scope)
        num_labels = math_ops.reduce_sum(array_ops.where_v2(math_ops.greater_equal(labels, 0), array_ops.ones_like(labels), array_ops.zeros_like(labels)), axis=-1)
        return math_ops.minimum(num_labels, k, name=scope)

def _sparse_average_precision_at_top_k(labels, predictions_idx):
    if False:
        return 10
    'Computes average precision@k of predictions with respect to sparse labels.\n\n  From en.wikipedia.org/wiki/Information_retrieval#Average_precision, formula\n  for each row is:\n\n    AveP = sum_{i=1...k} P_{i} * rel_{i} / num_relevant_items\n\n  A "row" is the elements in dimension [D1, ... DN] of `predictions_idx`,\n  `labels`, and the result `Tensors`. In the common case, this is [batch_size].\n  Each row of the results contains the average precision for that row.\n\n  Args:\n    labels: `int64` `Tensor` or `SparseTensor` with shape\n      [D1, ... DN, num_labels] or [D1, ... DN], where the latter implies\n      num_labels=1. N >= 1 and num_labels is the number of target classes for\n      the associated prediction. Commonly, N=1 and `labels` has shape\n      [batch_size, num_labels]. [D1, ... DN] must match `predictions_idx`.\n      Values should be non-negative. Negative values are ignored.\n    predictions_idx: Integer `Tensor` with shape [D1, ... DN, k] where N >= 1.\n      Commonly, N=1 and `predictions_idx` has shape [batch size, k]. The final\n      dimension must be set and contains the top `k` predicted class indices.\n      [D1, ... DN] must match `labels`. Values should be in range\n      [0, num_classes).\n\n  Returns:\n    `float64` `Tensor` of shape [D1, ... DN], where each value is the average\n    precision for that row.\n\n  Raises:\n    ValueError: if the last dimension of predictions_idx is not set.\n  '
    with ops.name_scope(None, 'average_precision', (predictions_idx, labels)) as scope:
        predictions_idx = math_ops.cast(predictions_idx, dtypes.int64, name='predictions_idx')
        if predictions_idx.get_shape().ndims == 0:
            raise ValueError('The rank of `predictions_idx` must be at least 1.')
        k = predictions_idx.get_shape().as_list()[-1]
        if k is None:
            raise ValueError('The last dimension of predictions_idx must be set. Currently, it is None.')
        labels = _maybe_expand_labels(labels, predictions_idx)
        predictions_idx_per_k = array_ops.expand_dims(predictions_idx, -1, name='predictions_idx_per_k')
        labels_per_k = _expand_and_tile(labels, multiple=k, dim=-1, name='labels_per_k')
        relevant_per_k = _sparse_true_positive_at_k(labels_per_k, predictions_idx_per_k, name='relevant_per_k')
        tp_per_k = math_ops.cumsum(relevant_per_k, axis=-1, name='tp_per_k')
        retrieved_per_k = math_ops.cumsum(array_ops.ones_like(relevant_per_k), axis=-1, name='retrieved_per_k')
        precision_per_k = math_ops.divide(math_ops.cast(tp_per_k, dtypes.float64), math_ops.cast(retrieved_per_k, dtypes.float64), name='precision_per_k')
        relevant_precision_per_k = math_ops.multiply(precision_per_k, math_ops.cast(relevant_per_k, dtypes.float64), name='relevant_precision_per_k')
        precision_sum = math_ops.reduce_sum(relevant_precision_per_k, axis=(-1,), name='precision_sum')
        num_relevant_items = math_ops.cast(_num_relevant(labels, k), dtypes.float64)
        return math_ops.divide(precision_sum, num_relevant_items, name=scope)

def _streaming_sparse_average_precision_at_top_k(labels, predictions_idx, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Computes average precision@k of predictions with respect to sparse labels.\n\n  `sparse_average_precision_at_top_k` creates two local variables,\n  `average_precision_at_<k>/total` and `average_precision_at_<k>/max`, that\n  are used to compute the frequency. This frequency is ultimately returned as\n  `average_precision_at_<k>`: an idempotent operation that simply divides\n  `average_precision_at_<k>/total` by `average_precision_at_<k>/max`.\n\n  For estimation of the metric over a stream of data, the function creates an\n  `update_op` operation that updates these variables and returns the\n  `precision_at_<k>`. Set operations applied to `top_k` and `labels` calculate\n  the true positives and false positives weighted by `weights`. Then `update_op`\n  increments `true_positive_at_<k>` and `false_positive_at_<k>` using these\n  values.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: `int64` `Tensor` or `SparseTensor` with shape\n      [D1, ... DN, num_labels] or [D1, ... DN], where the latter implies\n      num_labels=1. N >= 1 and num_labels is the number of target classes for\n      the associated prediction. Commonly, N=1 and `labels` has shape\n      [batch_size, num_labels]. [D1, ... DN] must match `predictions_idx`.\n      Values should be non-negative. Negative values are ignored.\n    predictions_idx: Integer `Tensor` with shape [D1, ... DN, k] where N >= 1.\n      Commonly, N=1 and `predictions_idx` has shape [batch size, k]. The final\n      dimension contains the top `k` predicted class indices. [D1, ... DN] must\n      match `labels`. Values should be in range [0, num_classes).\n    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of\n      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all\n      dimensions must be either `1`, or the same as the corresponding `labels`\n      dimension).\n    metrics_collections: An optional list of collections that values should\n      be added to.\n    updates_collections: An optional list of collections that updates should\n      be added to.\n    name: Name of new update operation, and namespace for other dependent ops.\n\n  Returns:\n    mean_average_precision: Scalar `float64` `Tensor` with the mean average\n      precision values.\n    update: `Operation` that increments variables appropriately, and whose\n      value matches `metric`.\n  '
    with ops.name_scope(name, 'average_precision_at_top_k', (predictions_idx, labels, weights)) as scope:
        average_precision = _sparse_average_precision_at_top_k(predictions_idx=predictions_idx, labels=labels)
        if weights is not None:
            weights = weights_broadcast_ops.broadcast_weights(math_ops.cast(weights, dtypes.float64), average_precision)
            average_precision = math_ops.multiply(average_precision, weights)
        with ops.name_scope(None, 'max', (average_precision,)) as max_scope:
            max_var = metric_variable([], dtypes.float64, name=max_scope)
            if weights is None:
                batch_max = math_ops.cast(array_ops.size(average_precision, name='batch_max'), dtypes.float64)
            else:
                batch_max = math_ops.reduce_sum(weights, name='batch_max')
            max_update = state_ops.assign_add(max_var, batch_max, name='update')
        with ops.name_scope(None, 'total', (average_precision,)) as total_scope:
            total_var = metric_variable([], dtypes.float64, name=total_scope)
            batch_total = math_ops.reduce_sum(average_precision, name='batch_total')
            total_update = state_ops.assign_add(total_var, batch_total, name='update')

        def precision_across_replicas(_, total_var, max_var):
            if False:
                while True:
                    i = 10
            return _safe_scalar_div(total_var, max_var, name='mean')
        mean_average_precision = _aggregate_across_replicas(metrics_collections, precision_across_replicas, total_var, max_var)
        update = _safe_scalar_div(total_update, max_update, name=scope)
        if updates_collections:
            ops.add_to_collections(updates_collections, update)
        return (mean_average_precision, update)

def _clean_out_of_range_indices(labels, num_classes):
    if False:
        i = 10
        return i + 15
    'Replaces large out-of-range labels by small out-of-range labels.\n\n  Replaces any value in `labels` that is greater or equal to `num_classes` by\n  -1. Do this conditionally for efficiency in case there are no such values.\n\n  Args:\n    labels: `int64` `Tensor` or `SparseTensor`.\n    num_classes: `int64` scalar `Tensor`.\n  Returns:\n    An `int64` `Tensor` or `SparseTensor` as `labels` with indices greater\n    or equal to num_classes replaced by -1.\n  '

    def _labels_is_sparse():
        if False:
            i = 10
            return i + 15
        'Returns true is `labels` is a sparse tensor.'
        return isinstance(labels, (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue))

    def _clean_out_of_range(values):
        if False:
            while True:
                i = 10
        'Replaces by -1 any large out-of-range `values`.'
        return array_ops.where_v2(math_ops.greater_equal(values, num_classes), -1 * array_ops.ones_like(values), values)

    def _clean_labels_out_of_range():
        if False:
            while True:
                i = 10
        'Replaces by -1 ane large out-of-range values in `labels`.'
        if _labels_is_sparse():
            return type(labels)(indices=labels.indices, values=_clean_out_of_range(labels.values), dense_shape=labels.dense_shape)
        else:
            return _clean_out_of_range(labels)
    max_labels = math_ops.reduce_max(labels.values if _labels_is_sparse() else labels)
    return cond.cond(math_ops.greater_equal(max_labels, num_classes), _clean_labels_out_of_range, lambda : labels)

@tf_export(v1=['metrics.sparse_average_precision_at_k'])
@deprecated(None, 'Use average_precision_at_k instead')
def sparse_average_precision_at_k(labels, predictions, k, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        while True:
            i = 10
    'Renamed to `average_precision_at_k`, please use that method instead.'
    return average_precision_at_k(labels=labels, predictions=predictions, k=k, weights=weights, metrics_collections=metrics_collections, updates_collections=updates_collections, name=name)

@tf_export(v1=['metrics.average_precision_at_k'])
def average_precision_at_k(labels, predictions, k, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        while True:
            i = 10
    'Computes average precision@k of predictions with respect to sparse labels.\n\n  `average_precision_at_k` creates two local variables,\n  `average_precision_at_<k>/total` and `average_precision_at_<k>/max`, that\n  are used to compute the frequency. This frequency is ultimately returned as\n  `average_precision_at_<k>`: an idempotent operation that simply divides\n  `average_precision_at_<k>/total` by `average_precision_at_<k>/max`.\n\n  For estimation of the metric over a stream of data, the function creates an\n  `update_op` operation that updates these variables and returns the\n  `precision_at_<k>`. Internally, a `top_k` operation computes a `Tensor`\n  indicating the top `k` `predictions`. Set operations applied to `top_k` and\n  `labels` calculate the true positives and false positives weighted by\n  `weights`. Then `update_op` increments `true_positive_at_<k>` and\n  `false_positive_at_<k>` using these values.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: `int64` `Tensor` or `SparseTensor` with shape\n      [D1, ... DN, num_labels] or [D1, ... DN], where the latter implies\n      num_labels=1. N >= 1 and num_labels is the number of target classes for\n      the associated prediction. Commonly, N=1 and `labels` has shape\n      [batch_size, num_labels]. [D1, ... DN] must match `predictions`. Values\n      should be in range [0, num_classes), where num_classes is the last\n      dimension of `predictions`. Values outside this range are ignored.\n    predictions: Float `Tensor` with shape [D1, ... DN, num_classes] where\n      N >= 1. Commonly, N=1 and `predictions` has shape\n      [batch size, num_classes]. The final dimension contains the logit values\n      for each class. [D1, ... DN] must match `labels`.\n    k: Integer, k for @k metric. This will calculate an average precision for\n      range `[1,k]`, as documented above.\n    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of\n      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all\n      dimensions must be either `1`, or the same as the corresponding `labels`\n      dimension).\n    metrics_collections: An optional list of collections that values should\n      be added to.\n    updates_collections: An optional list of collections that updates should\n      be added to.\n    name: Name of new update operation, and namespace for other dependent ops.\n\n  Returns:\n    mean_average_precision: Scalar `float64` `Tensor` with the mean average\n      precision values.\n    update: `Operation` that increments variables appropriately, and whose\n      value matches `metric`.\n\n  Raises:\n    ValueError: if k is invalid.\n    RuntimeError: If eager execution is enabled.\n  '
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.sparse_average_precision_at_k is not supported when eager execution is enabled.')
    if k < 1:
        raise ValueError(f'Invalid k={k}. `k` should be >= 1.')
    with ops.name_scope(name, _at_k_name('average_precision', k), (predictions, labels, weights)) as scope:
        (_, predictions_idx) = nn.top_k(predictions, k)
        labels = _clean_out_of_range_indices(labels, math_ops.cast(array_ops.shape(predictions)[-1], dtypes.int64))
        return _streaming_sparse_average_precision_at_top_k(labels=labels, predictions_idx=predictions_idx, weights=weights, metrics_collections=metrics_collections, updates_collections=updates_collections, name=scope)

def _sparse_false_positive_at_k(labels, predictions_idx, class_id=None, weights=None):
    if False:
        while True:
            i = 10
    'Calculates false positives for precision@k.\n\n  If `class_id` is specified, calculate binary true positives for `class_id`\n      only.\n  If `class_id` is not specified, calculate metrics for `k` predicted vs\n      `n` label classes, where `n` is the 2nd dimension of `labels_sparse`.\n\n  Args:\n    labels: `int64` `Tensor` or `SparseTensor` with shape\n      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of\n      target classes for the associated prediction. Commonly, N=1 and `labels`\n      has shape [batch_size, num_labels]. [D1, ... DN] must match\n      `predictions_idx`.\n    predictions_idx: 1-D or higher `int64` `Tensor` with last dimension `k`,\n      top `k` predicted classes. For rank `n`, the first `n-1` dimensions must\n      match `labels`.\n    class_id: Class for which we want binary metrics.\n    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of\n      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all\n      dimensions must be either `1`, or the same as the corresponding `labels`\n      dimension).\n\n  Returns:\n    A [D1, ... DN] `Tensor` of false positive counts.\n  '
    with ops.name_scope(None, 'false_positives', (predictions_idx, labels, weights)):
        (labels, predictions_idx) = _maybe_select_class_id(labels, predictions_idx, class_id)
        fp = sets.set_size(sets.set_difference(predictions_idx, labels, aminusb=True))
        fp = math_ops.cast(fp, dtypes.float64)
        if weights is not None:
            with ops.control_dependencies((weights_broadcast_ops.assert_broadcastable(weights, fp),)):
                weights = math_ops.cast(weights, dtypes.float64)
                fp = math_ops.multiply(fp, weights)
        return fp

def _streaming_sparse_false_positive_at_k(labels, predictions_idx, k=None, class_id=None, weights=None, name=None):
    if False:
        return 10
    'Calculates weighted per step false positives for precision@k.\n\n  If `class_id` is specified, calculate binary true positives for `class_id`\n      only.\n  If `class_id` is not specified, calculate metrics for `k` predicted vs\n      `n` label classes, where `n` is the 2nd dimension of `labels`.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: `int64` `Tensor` or `SparseTensor` with shape\n      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of\n      target classes for the associated prediction. Commonly, N=1 and `labels`\n      has shape [batch_size, num_labels]. [D1, ... DN] must match\n      `predictions_idx`.\n    predictions_idx: 1-D or higher `int64` `Tensor` with last dimension `k`,\n      top `k` predicted classes. For rank `n`, the first `n-1` dimensions must\n      match `labels`.\n    k: Integer, k for @k metric. This is only used for default op name.\n    class_id: Class for which we want binary metrics.\n    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of\n      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all\n      dimensions must be either `1`, or the same as the corresponding `labels`\n      dimension).\n    name: Name of new variable, and namespace for other dependent ops.\n\n  Returns:\n    A tuple of `Variable` and update `Operation`.\n\n  Raises:\n    ValueError: If `weights` is not `None` and has an incompatible shape.\n  '
    with ops.name_scope(name, _at_k_name('false_positive', k, class_id=class_id), (predictions_idx, labels, weights)) as scope:
        fp = _sparse_false_positive_at_k(predictions_idx=predictions_idx, labels=labels, class_id=class_id, weights=weights)
        batch_total_fp = math_ops.cast(math_ops.reduce_sum(fp), dtypes.float64)
        var = metric_variable([], dtypes.float64, name=scope)
        return (var, state_ops.assign_add(var, batch_total_fp, name='update'))

@tf_export(v1=['metrics.precision_at_top_k'])
def precision_at_top_k(labels, predictions_idx, k=None, class_id=None, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        return 10
    "Computes precision@k of the predictions with respect to sparse labels.\n\n  Differs from `sparse_precision_at_k` in that predictions must be in the form\n  of top `k` class indices, whereas `sparse_precision_at_k` expects logits.\n  Refer to `sparse_precision_at_k` for more details.\n\n  Args:\n    labels: `int64` `Tensor` or `SparseTensor` with shape\n      [D1, ... DN, num_labels] or [D1, ... DN], where the latter implies\n      num_labels=1. N >= 1 and num_labels is the number of target classes for\n      the associated prediction. Commonly, N=1 and `labels` has shape\n      [batch_size, num_labels]. [D1, ... DN] must match `predictions`. Values\n      should be in range [0, num_classes), where num_classes is the last\n      dimension of `predictions`. Values outside this range are ignored.\n    predictions_idx: Integer `Tensor` with shape [D1, ... DN, k] where\n      N >= 1. Commonly, N=1 and predictions has shape [batch size, k].\n      The final dimension contains the top `k` predicted class indices.\n      [D1, ... DN] must match `labels`.\n    k: Integer, k for @k metric. Only used for the default op name.\n    class_id: Integer class ID for which we want binary metrics. This should be\n      in range [0, num_classes], where num_classes is the last dimension of\n      `predictions`. If `class_id` is outside this range, the method returns\n      NAN.\n    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of\n      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all\n      dimensions must be either `1`, or the same as the corresponding `labels`\n      dimension).\n    metrics_collections: An optional list of collections that values should\n      be added to.\n    updates_collections: An optional list of collections that updates should\n      be added to.\n    name: Name of new update operation, and namespace for other dependent ops.\n\n  Returns:\n    precision: Scalar `float64` `Tensor` with the value of `true_positives`\n      divided by the sum of `true_positives` and `false_positives`.\n    update_op: `Operation` that increments `true_positives` and\n      `false_positives` variables appropriately, and whose value matches\n      `precision`.\n\n  Raises:\n    ValueError: If `weights` is not `None` and its shape doesn't match\n      `predictions`, or if either `metrics_collections` or `updates_collections`\n      are not a list or tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.precision_at_top_k is not supported when eager execution is enabled.')
    with ops.name_scope(name, _at_k_name('precision', k, class_id=class_id), (predictions_idx, labels, weights)) as scope:
        labels = _maybe_expand_labels(labels, predictions_idx)
        top_k_idx = math_ops.cast(predictions_idx, dtypes.int64)
        (tp, tp_update) = _streaming_sparse_true_positive_at_k(predictions_idx=top_k_idx, labels=labels, k=k, class_id=class_id, weights=weights)
        (fp, fp_update) = _streaming_sparse_false_positive_at_k(predictions_idx=top_k_idx, labels=labels, k=k, class_id=class_id, weights=weights)

        def precision_across_replicas(_, tp, fp):
            if False:
                print('Hello World!')
            return math_ops.divide(tp, math_ops.add(tp, fp), name=scope)
        metric = _aggregate_across_replicas(metrics_collections, precision_across_replicas, tp, fp)
        update = math_ops.divide(tp_update, math_ops.add(tp_update, fp_update), name='update')
        if updates_collections:
            ops.add_to_collections(updates_collections, update)
        return (metric, update)

@tf_export(v1=['metrics.sparse_precision_at_k'])
@deprecated(None, 'Use precision_at_k instead')
def sparse_precision_at_k(labels, predictions, k, class_id=None, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        return 10
    'Renamed to `precision_at_k`, please use that method instead.'
    return precision_at_k(labels=labels, predictions=predictions, k=k, class_id=class_id, weights=weights, metrics_collections=metrics_collections, updates_collections=updates_collections, name=name)

@tf_export(v1=['metrics.precision_at_k'])
def precision_at_k(labels, predictions, k, class_id=None, weights=None, metrics_collections=None, updates_collections=None, name=None):
    if False:
        print('Hello World!')
    "Computes precision@k of the predictions with respect to sparse labels.\n\n  If `class_id` is specified, we calculate precision by considering only the\n      entries in the batch for which `class_id` is in the top-k highest\n      `predictions`, and computing the fraction of them for which `class_id` is\n      indeed a correct label.\n  If `class_id` is not specified, we'll calculate precision as how often on\n      average a class among the top-k classes with the highest predicted values\n      of a batch entry is correct and can be found in the label for that entry.\n\n  `precision_at_k` creates two local variables,\n  `true_positive_at_<k>` and `false_positive_at_<k>`, that are used to compute\n  the precision@k frequency. This frequency is ultimately returned as\n  `precision_at_<k>`: an idempotent operation that simply divides\n  `true_positive_at_<k>` by total (`true_positive_at_<k>` +\n  `false_positive_at_<k>`).\n\n  For estimation of the metric over a stream of data, the function creates an\n  `update_op` operation that updates these variables and returns the\n  `precision_at_<k>`. Internally, a `top_k` operation computes a `Tensor`\n  indicating the top `k` `predictions`. Set operations applied to `top_k` and\n  `labels` calculate the true positives and false positives weighted by\n  `weights`. Then `update_op` increments `true_positive_at_<k>` and\n  `false_positive_at_<k>` using these values.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  Args:\n    labels: `int64` `Tensor` or `SparseTensor` with shape\n      [D1, ... DN, num_labels] or [D1, ... DN], where the latter implies\n      num_labels=1. N >= 1 and num_labels is the number of target classes for\n      the associated prediction. Commonly, N=1 and `labels` has shape\n      [batch_size, num_labels]. [D1, ... DN] must match `predictions`. Values\n      should be in range [0, num_classes), where num_classes is the last\n      dimension of `predictions`. Values outside this range are ignored.\n    predictions: Float `Tensor` with shape [D1, ... DN, num_classes] where\n      N >= 1. Commonly, N=1 and predictions has shape [batch size, num_classes].\n      The final dimension contains the logit values for each class. [D1, ... DN]\n      must match `labels`.\n    k: Integer, k for @k metric.\n    class_id: Integer class ID for which we want binary metrics. This should be\n      in range [0, num_classes], where num_classes is the last dimension of\n      `predictions`. If `class_id` is outside this range, the method returns\n      NAN.\n    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of\n      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all\n      dimensions must be either `1`, or the same as the corresponding `labels`\n      dimension).\n    metrics_collections: An optional list of collections that values should\n      be added to.\n    updates_collections: An optional list of collections that updates should\n      be added to.\n    name: Name of new update operation, and namespace for other dependent ops.\n\n  Returns:\n    precision: Scalar `float64` `Tensor` with the value of `true_positives`\n      divided by the sum of `true_positives` and `false_positives`.\n    update_op: `Operation` that increments `true_positives` and\n      `false_positives` variables appropriately, and whose value matches\n      `precision`.\n\n  Raises:\n    ValueError: If `weights` is not `None` and its shape doesn't match\n      `predictions`, or if either `metrics_collections` or `updates_collections`\n      are not a list or tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.sparse_precision_at_k is not supported when eager execution is enabled.')
    with ops.name_scope(name, _at_k_name('precision', k, class_id=class_id), (predictions, labels, weights)) as scope:
        (_, top_k_idx) = nn.top_k(predictions, k)
        return precision_at_top_k(labels=labels, predictions_idx=top_k_idx, k=k, class_id=class_id, weights=weights, metrics_collections=metrics_collections, updates_collections=updates_collections, name=scope)

@tf_export(v1=['metrics.specificity_at_sensitivity'])
def specificity_at_sensitivity(labels, predictions, sensitivity, weights=None, num_thresholds=200, metrics_collections=None, updates_collections=None, name=None):
    if False:
        while True:
            i = 10
    "Computes the specificity at a given sensitivity.\n\n  The `specificity_at_sensitivity` function creates four local\n  variables, `true_positives`, `true_negatives`, `false_positives` and\n  `false_negatives` that are used to compute the specificity at the given\n  sensitivity value. The threshold for the given sensitivity value is computed\n  and used to evaluate the corresponding specificity.\n\n  For estimation of the metric over a stream of data, the function creates an\n  `update_op` operation that updates these variables and returns the\n  `specificity`. `update_op` increments the `true_positives`, `true_negatives`,\n  `false_positives` and `false_negatives` counts with the weight of each case\n  found in the `predictions` and `labels`.\n\n  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.\n\n  For additional information about specificity and sensitivity, see the\n  following: https://en.wikipedia.org/wiki/Sensitivity_and_specificity\n\n  Args:\n    labels: The ground truth values, a `Tensor` whose dimensions must match\n      `predictions`. Will be cast to `bool`.\n    predictions: A floating point `Tensor` of arbitrary shape and whose values\n      are in the range `[0, 1]`.\n    sensitivity: A scalar value in range `[0, 1]`.\n    weights: Optional `Tensor` whose rank is either 0, or the same rank as\n      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must\n      be either `1`, or the same as the corresponding `labels` dimension).\n    num_thresholds: The number of thresholds to use for matching the given\n      sensitivity.\n    metrics_collections: An optional list of collections that `specificity`\n      should be added to.\n    updates_collections: An optional list of collections that `update_op` should\n      be added to.\n    name: An optional variable_scope name.\n\n  Returns:\n    specificity: A scalar `Tensor` representing the specificity at the given\n      `sensitivity` value.\n    update_op: An operation that increments the `true_positives`,\n      `true_negatives`, `false_positives` and `false_negatives` variables\n      appropriately and whose value matches `specificity`.\n\n  Raises:\n    ValueError: If `predictions` and `labels` have mismatched shapes, if\n      `weights` is not `None` and its shape doesn't match `predictions`, or if\n      `sensitivity` is not between 0 and 1, or if either `metrics_collections`\n      or `updates_collections` are not a list or tuple.\n    RuntimeError: If eager execution is enabled.\n  "
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.specificity_at_sensitivity is not supported when eager execution is enabled.')
    if sensitivity < 0 or sensitivity > 1:
        raise ValueError(f'`sensitivity` must be in the range [0, 1]. Currently, `sensitivity` is {sensitivity}.')
    with variable_scope.variable_scope(name, 'specificity_at_sensitivity', (predictions, labels, weights)):
        kepsilon = 1e-07
        thresholds = [(i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)]
        thresholds = [0.0 - kepsilon] + thresholds + [1.0 - kepsilon]
        (values, update_ops) = _confusion_matrix_at_thresholds(labels, predictions, thresholds, weights)

        def compute_specificity_at_sensitivity(tp, tn, fp, fn, name):
            if False:
                print('Hello World!')
            'Computes the specificity at the given sensitivity.\n\n      Args:\n        tp: True positives.\n        tn: True negatives.\n        fp: False positives.\n        fn: False negatives.\n        name: The name of the operation.\n\n      Returns:\n        The specificity using the aggregated values.\n      '
            sensitivities = math_ops.divide(tp, tp + fn + kepsilon)
            min_val = math_ops.reduce_min(math_ops.abs(sensitivities - sensitivity))
            indices_at_minval = math_ops.equal(math_ops.abs(sensitivities - sensitivity), min_val)
            indices_at_minval = math_ops.cast(indices_at_minval, dtypes.int64)
            indices_at_minval = math_ops.cumsum(indices_at_minval)
            tf_index = math_ops.argmax(indices_at_minval, 0)
            tf_index = math_ops.cast(tf_index, dtypes.int32)
            return math_ops.divide(tn[tf_index], tn[tf_index] + fp[tf_index] + kepsilon, name)

        def specificity_across_replicas(_, values):
            if False:
                print('Hello World!')
            return compute_specificity_at_sensitivity(values['tp'], values['tn'], values['fp'], values['fn'], 'value')
        specificity = _aggregate_across_replicas(metrics_collections, specificity_across_replicas, values)
        update_op = compute_specificity_at_sensitivity(update_ops['tp'], update_ops['tn'], update_ops['fp'], update_ops['fn'], 'update_op')
        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)
        return (specificity, update_op)