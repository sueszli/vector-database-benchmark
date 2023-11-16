"""Optimizer that implements cross-shard gradient reduction for TPU."""
from tensorflow.python.framework import ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export

@tf_export(v1=['tpu.CrossShardOptimizer'])
class CrossShardOptimizer(optimizer.Optimizer):
    """An optimizer that averages gradients across TPU shards."""

    def __init__(self, opt, reduction=losses.Reduction.MEAN, name='CrossShardOptimizer', group_assignment=None):
        if False:
            while True:
                i = 10
        'Construct a new cross-shard optimizer.\n\n    Args:\n      opt: An existing `Optimizer` to encapsulate.\n      reduction: The reduction to apply to the shard losses.\n      name: Optional name prefix for the operations created when applying\n        gradients. Defaults to "CrossShardOptimizer".\n      group_assignment: Optional 2d int32 lists with shape\n        [num_groups, num_replicas_per_group] which describles how to apply\n        optimizer to subgroups.\n\n    Raises:\n      ValueError: If reduction is not a valid cross-shard reduction.\n    '
        accepted_reductions = (losses.Reduction.SUM, losses.Reduction.MEAN)
        if reduction not in accepted_reductions:
            raise ValueError(f'Argument `reduction` should be one of {accepted_reductions}. Received: {reduction}')
        if not isinstance(opt, optimizer.Optimizer):
            raise TypeError(f'CrossShardOptimizer only works with tf.training.Optimizer and not Keras Optimizer. Received: {opt}. If you are using TPUStrategy, Keras Optimizer will sum gradients across replicas.If you are using TPUEstimator, you may instead sum your gradients with:\n`grads = [tf.compat.v1.tpu.cross_replica_sum(g) for g in grads]`\nIf you want to average your gradients, rescale your loss with: `loss /= global_batch_size`')
        super(CrossShardOptimizer, self).__init__(False, name)
        self._opt = opt
        self._reduction = reduction
        self._group_assignment = group_assignment

    def _verify_and_get_subgroup_size(self, group_assignment, num_shards):
        if False:
            return 10
        'Verify group_assignment and get the subgroup size".\n\n    Args:\n      group_assignment: list of group ids for applying the optimizer\n        to subgroups.\n      num_shards: The number of TPU shards.\n\n    Returns:\n      The size of one subgroup in group_assignment.\n\n    Raises:\n      ValueError: If group_assignment is invalid.\n    '
        if not group_assignment:
            return None
        if not (isinstance(group_assignment, list) and all((isinstance(i, list) for i in group_assignment))):
            raise ValueError(f'Argument `group_assignment` must be a list of lists. Received: {group_assignment}')
        replica_ids = set()
        for g in group_assignment:
            for i in g:
                replica_ids.add(i)
        if set(range(num_shards)) != replica_ids:
            raise ValueError(f'Argument `group_assignment` must be a permutation of range({num_shards}). Received: {group_assignment}')
        subgroup_size_list = [len(group) for group in group_assignment]
        if all((subgroup_size_list[0] == size for size in subgroup_size_list)):
            return subgroup_size_list[0]
        else:
            raise ValueError(f'The size of each subgroup in `group_assignment` must be equal. Received: {group_assignment}')

    def compute_gradients(self, loss, var_list=None, **kwargs):
        if False:
            return 10
        'Compute gradients of "loss" for the variables in "var_list".\n\n    This simply wraps `compute_gradients()` from the real optimizer. The\n    gradients will be aggregated in `apply_gradients()` so that user can\n    modify the gradients like clipping with per replica global norm if needed.\n    The global norm with aggregated gradients can be bad as one replica\'s huge\n    gradients can hurt the gradients from other replicas.\n\n    When the CrossShardOptimizer is constructed with\n    `reduction == losses.Reduction.MEAN` (default), this function scales the\n    loss by `1.0 / num_shards` before computing the gradients. Assuming the\n    optimizer uses the default implementation of `compute_gradients()`, the\n    gradients of the scaled loss are scaled by `1.0 / num_shards` compared to\n    the gradients of the original loss. This scaling factor is important because\n    `apply_gradients()` sums gradients across shards, rather than averaging\n    them. However, the scaling factor must be taken into account when clipping\n    the norm of the gradients or performing other postprocessing.\n\n    Args:\n      loss: A Tensor containing the value to minimize.\n      var_list: Optional list or tuple of `tf.Variable` to update to minimize\n        `loss`.  Defaults to the list of variables collected in the graph\n        under the key `GraphKey.TRAINABLE_VARIABLES`.\n      **kwargs: Keyword arguments for compute_gradients().\n\n    Returns:\n      A list of (gradient, variable) pairs.\n\n    Raises:\n      ValueError: If not within a tpu_shard_context or group_assignment is\n        invalid.\n    '
        num_shards = tpu_function.get_tpu_context().number_of_shards
        if num_shards is None:
            logging.warning('CrossShardOptimizer should be used within a tpu_shard_context, but got unset number_of_shards. Assuming 1.')
            num_shards = 1
        subgroup_size = self._verify_and_get_subgroup_size(self._group_assignment, num_shards)
        if num_shards > 1 and self._reduction == losses.Reduction.MEAN:
            if self._group_assignment:
                scale = 1.0 / subgroup_size
            else:
                scale = 1.0 / num_shards
            loss *= scale
        return self._opt.compute_gradients(loss, var_list=var_list, **kwargs)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        if False:
            i = 10
            return i + 15
        'Apply gradients to variables.\n\n    Calls tpu_ops.cross_replica_sum() to sum gradient contributions across\n    replicas, and then applies the real optimizer.\n\n    Args:\n      grads_and_vars: List of (gradient, variable) pairs as returned by\n        compute_gradients().\n      global_step: Optional Variable to increment by one after the\n        variables have been updated.\n      name: Optional name for the returned operation.  Default to the\n        name passed to the Optimizer constructor.\n\n    Returns:\n      An `Operation` that applies the gradients. If `global_step` was not None,\n      that operation also increments `global_step`.\n\n    Raises:\n      ValueError: If the grads_and_vars is malformed.\n    '
        summed_grads_and_vars = []
        for (grad, var) in grads_and_vars:
            if grad is None:
                summed_grads_and_vars.append((grad, var))
            else:
                with ops.colocate_with(grad):
                    summed_grads_and_vars.append((tpu_ops.cross_replica_sum(grad, self._group_assignment), var))
        return self._opt.apply_gradients(summed_grads_and_vars, global_step, name)

    def get_slot(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Return a slot named "name" created for "var" by the Optimizer.\n\n    This simply wraps the get_slot() from the actual optimizer.\n\n    Args:\n      *args: Arguments for get_slot().\n      **kwargs: Keyword arguments for get_slot().\n\n    Returns:\n      The `Variable` for the slot if it was created, `None` otherwise.\n    '
        return self._opt.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Return a list of the names of slots created by the `Optimizer`.\n\n    This simply wraps the get_slot_names() from the actual optimizer.\n\n    Args:\n      *args: Arguments for get_slot().\n      **kwargs: Keyword arguments for get_slot().\n\n    Returns:\n      A list of strings.\n    '
        return self._opt.get_slot_names(*args, **kwargs)

    def variables(self):
        if False:
            while True:
                i = 10
        'Forwarding the variables from the underlying optimizer.'
        return self._opt.variables()