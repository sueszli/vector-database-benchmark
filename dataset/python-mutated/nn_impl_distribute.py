"""Implementation of Neural Net (NN) functions with distribution strategy."""
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import util as losses_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

@tf_export('nn.scale_regularization_loss')
@dispatch.add_dispatch_support
def scale_regularization_loss(regularization_loss):
    if False:
        print('Hello World!')
    'Scales the sum of the given regularization losses by number of replicas.\n\n  Usage with distribution strategy and custom training loop:\n\n  ```python\n  with strategy.scope():\n    def compute_loss(self, label, predictions):\n      per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(\n          labels, predictions)\n\n      # Compute loss that is scaled by sample_weight and by global batch size.\n      loss = tf.nn.compute_average_loss(\n          per_example_loss,\n          sample_weight=sample_weight,\n          global_batch_size=GLOBAL_BATCH_SIZE)\n\n      # Add scaled regularization losses.\n      loss += tf.nn.scale_regularization_loss(tf.nn.l2_loss(weights))\n      return loss\n  ```\n\n  Args:\n    regularization_loss: Regularization loss.\n\n  Returns:\n    Scalar loss value.\n  '
    if distribute_lib.has_strategy() and distribute_lib.in_cross_replica_context():
        raise RuntimeError('You are calling `scale_regularization_loss` in cross replica context, while it was expected to be called in replica context.')
    num_replicas = distribute_lib.get_strategy().num_replicas_in_sync
    return math_ops.reduce_sum(regularization_loss) / num_replicas

@tf_export('nn.compute_average_loss')
@dispatch.add_dispatch_support
def compute_average_loss(per_example_loss, sample_weight=None, global_batch_size=None):
    if False:
        i = 10
        return i + 15
    'Scales per-example losses with sample_weights and computes their average.\n\n  Usage with distribution strategy and custom training loop:\n\n  ```python\n  with strategy.scope():\n    def compute_loss(labels, predictions, sample_weight=None):\n\n      # If you are using a `Loss` class instead, set reduction to `NONE` so that\n      # we can do the reduction afterwards and divide by global batch size.\n      per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(\n          labels, predictions)\n\n      # Compute loss that is scaled by sample_weight and by global batch size.\n      return tf.nn.compute_average_loss(\n          per_example_loss,\n          sample_weight=sample_weight,\n          global_batch_size=GLOBAL_BATCH_SIZE)\n  ```\n\n  Args:\n    per_example_loss: Per-example loss.\n    sample_weight: Optional weighting for each example.\n    global_batch_size: Optional global batch size value. Defaults to (size of\n      first dimension of `losses`) * (number of replicas).\n\n  Returns:\n    Scalar loss value, obtained by summing the `per_example_loss` and dividing\n    by `global_batch_size`. If `global_batch_size` is zero, the result is zero.\n  '
    per_example_loss = ops.convert_to_tensor(per_example_loss)
    input_dtype = per_example_loss.dtype
    with losses_util.check_per_example_loss_rank(per_example_loss):
        if sample_weight is not None:
            sample_weight = ops.convert_to_tensor(sample_weight)
            per_example_loss = losses_util.scale_losses_by_sample_weight(per_example_loss, sample_weight)
        per_example_loss = math_ops.cast(per_example_loss, input_dtype)
        if global_batch_size is None:
            if distribute_lib.has_strategy() and distribute_lib.in_cross_replica_context():
                raise RuntimeError('You are calling `compute_average_loss` in cross replica context, while it was expected to be called in replica context.')
            num_replicas = distribute_lib.get_strategy().num_replicas_in_sync
            per_replica_batch_size = array_ops.shape_v2(per_example_loss)[0]
            global_batch_size = per_replica_batch_size * num_replicas
        check_ops.assert_scalar_v2(global_batch_size, message='global_batch_size must be scalar.')
        check_ops.assert_integer_v2(global_batch_size, message='global_batch_size must be an integer.')
        check_ops.assert_non_negative_v2(global_batch_size, message='global_batch_size must be non-negative.')
        loss = math_ops.reduce_sum(per_example_loss)
        global_batch_size = math_ops.cast(global_batch_size, input_dtype)
        return math_ops.div_no_nan(loss, global_batch_size)