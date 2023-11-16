"""A class to store named variables and a scope operator to manage sharing."""
import copy
import enum
import functools
import sys
import threading
import traceback
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import resource_variables_toggle
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
__all__ = ['AUTO_REUSE', 'VariableScope', 'get_variable_scope', 'get_variable', 'get_local_variable', 'variable_scope', 'variable_op_scope', 'no_regularizer', 'VariableSynchronization', 'VariableAggregation']

class _PartitionInfo:
    """Holds partition info used by initializer functions."""
    __slots__ = ['_full_shape', '_var_offset']

    def __init__(self, full_shape, var_offset):
        if False:
            return 10
        'Constructor.\n\n    Args:\n      full_shape: Tuple or list of `int` indicating the full combined shape of\n        the partitioned variables.\n      var_offset: Tuple or list of `int` specifying offset of this partition\n        with respect to the full variable for each dimension.\n\n    Raises:\n      TypeError: If `full_shape` or `var_offset` is not a sequence.\n      ValueError: If `full_shape` or `var_offset` differ in length. If\n        `var_offset` exceeds `full_shape` in any dimension.\n    '
        if not isinstance(full_shape, (list, tuple)):
            raise TypeError('`full_shape` must be a sequence (like tuple or list) instead of ' + type(full_shape).__name__)
        if not isinstance(var_offset, (list, tuple)):
            raise TypeError('`var_offset` must be a sequence (like tuple or list) instead of ' + type(var_offset).__name__)
        if len(var_offset) != len(full_shape):
            raise ValueError('Expected equal length, but `var_offset` is of length {} while full_shape is of length {}.'.format(len(var_offset), len(full_shape)))
        for (offset, shape) in zip(var_offset, full_shape):
            if offset < 0 or offset >= shape:
                raise ValueError('Expected 0 <= offset < shape but found offset={}, shape={} for var_offset={}, full_shape={}'.format(offset, shape, var_offset, full_shape))
        self._full_shape = full_shape
        self._var_offset = var_offset

    @property
    def full_shape(self):
        if False:
            while True:
                i = 10
        return self._full_shape

    @property
    def var_offset(self):
        if False:
            i = 10
            return i + 15
        return self._var_offset

    def single_offset(self, shape):
        if False:
            print('Hello World!')
        'Returns the offset when the variable is partitioned in at most one dim.\n\n    Args:\n      shape: Tuple or list of `int` indicating the shape of one specific\n        variable partition.\n\n    Returns:\n      `int` representing the offset in the dimension along which the variable is\n       partitioned. Returns 0 if the variable is not being partitioned.\n\n    Raises:\n      ValueError: Depending on self.single_slice_dim().\n    '
        single_slice_dim = self.single_slice_dim(shape)
        if single_slice_dim is None:
            return 0
        return self.var_offset[single_slice_dim]

    def single_slice_dim(self, shape):
        if False:
            return 10
        "Returns the slice dim when the variable is partitioned only in one dim.\n\n    Args:\n      shape: Tuple or list of `int` indicating the shape of one specific\n        variable partition.\n\n    Returns:\n      `int` representing the dimension that the variable is partitioned in, or\n      `None` if the variable doesn't seem to be partitioned at all.\n\n    Raises:\n      TypeError: If `shape` is not a sequence.\n      ValueError: If `shape` is not the same length as `self.full_shape`. If\n        the variable is partitioned in more than one dimension.\n    "
        if not isinstance(shape, (tuple, list)):
            raise TypeError('`shape` must be a sequence (like tuple or list) instead of ' + type(shape).__name__)
        if len(shape) != len(self.full_shape):
            raise ValueError('Expected equal length, but received shape={} of length {} while self.full_shape={} is of length {}.'.format(shape, len(shape), self.full_shape, len(self.full_shape)))
        for i in range(len(shape)):
            if self.var_offset[i] + shape[i] > self.full_shape[i]:
                raise ValueError('With self.var_offset={}, a partition of shape={} would exceed self.full_shape={} in dimension {}.'.format(self.var_offset, shape, self.full_shape, i))
        slice_dim = None
        for i in range(len(shape)):
            if shape[i] == self.full_shape[i]:
                continue
            if slice_dim is not None:
                raise ValueError('Cannot use single_slice_dim() with shape={} and self.full_shape={} since slice dim could be either dimension {} or {}.'.format(shape, self.full_shape, i, slice_dim))
            slice_dim = i
        return slice_dim

class _ReuseMode(enum.Enum):
    """Mode for variable access within a variable scope."""
    AUTO_REUSE = 1
VariableSynchronization = variables.VariableSynchronization
VariableAggregation = variables.VariableAggregation
AUTO_REUSE = _ReuseMode.AUTO_REUSE
tf_export(v1=['AUTO_REUSE']).export_constant(__name__, 'AUTO_REUSE')
AUTO_REUSE.__doc__ = "\n@compatibility(TF2)\n`tf.compat.v1.AUTO_REUSE` is a legacy API that is a no-op when TF2 behaviors\nare enabled.\n\nIf you rely on `get_variable` and auto-reuse, see the\n[model mapping guide](https://www.tensorflow.org/guide/migrate/model_mapping)\nfor more info on how to migrate your code.\n\nNote: when you use the `tf.compat.v1.keras.utils.track_tf1_style_variables`\nAPI as described in the above guide, `get_variable` will always behave as if\n`v1.AUTO_REUSE` is set. Without the decorator, reuse will be ignored and new\nvariables will always be created, regardless of if they have already been\ncreated.\n@end_compatibility\n\nWhen passed in as the value for the `reuse` flag, `AUTO_REUSE` indicates that\nget_variable() should create the requested variable if it doesn't exist or, if\nit does exist, simply return it.\n"

def _needs_no_arguments(python_callable):
    if False:
        i = 10
        return i + 15
    'Returns true if the callable needs no arguments to call.'
    num_arguments = len(tf_inspect.getargspec(python_callable).args)
    if not tf_inspect.isfunction(python_callable) and (not isinstance(python_callable, functools.partial)):
        num_arguments -= 1
    return num_arguments == len(tf_inspect.getargspec(python_callable).defaults or [])

class _VariableStore:
    """Variable store that carries a number of named Variables.

  New variable names and new variables can be created; all stored
  variables are initialized with the initializer passed to __init__.

  Attributes:
    vars: a dictionary with string names (same as passed in GetVar) as keys and
      the corresponding TensorFlow Variables as values.
  """
    __slots__ = ['_vars', '_partitioned_vars', '_store_eager_variables']

    def __init__(self):
        if False:
            while True:
                i = 10
        'Create a variable store.'
        self._vars = {}
        self._partitioned_vars = {}
        self._store_eager_variables = False

    def get_variable(self, name, shape=None, dtype=dtypes.float32, initializer=None, regularizer=None, reuse=None, trainable=None, collections=None, caching_device=None, partitioner=None, validate_shape=True, use_resource=None, custom_getter=None, constraint=None, synchronization=VariableSynchronization.AUTO, aggregation=VariableAggregation.NONE):
        if False:
            while True:
                i = 10
        "Gets an existing variable with these parameters or create a new one.\n\n    If a variable with the given name is already stored, we return the stored\n    variable. Otherwise, we create a new one.\n\n    Set `reuse` to `True` when you only want to reuse existing Variables.\n    Set `reuse` to `False` when you only want to create new Variables.\n    Set `reuse` to None (the default) or tf.compat.v1.AUTO_REUSE when you want\n    variables to be created if they don't exist or returned if they do.\n\n    If initializer is `None` (the default), the default initializer passed in\n    the constructor is used. If that one is `None` too, we use a new\n    `glorot_uniform_initializer`. If initializer is a Tensor, we use\n    it as a value and derive the shape from the initializer.\n\n    If a partitioner is provided, a `PartitionedVariable` is returned.\n    Accessing this object as a `Tensor` returns the shards concatenated along\n    the partition axis.\n\n    Some useful partitioners are available.  See, e.g.,\n    `variable_axis_size_partitioner` and `min_max_variable_partitioner`.\n\n    Args:\n      name: The name of the new or existing variable.\n      shape: Shape of the new or existing variable.\n      dtype: Type of the new or existing variable (defaults to `DT_FLOAT`).\n      initializer: Initializer for the variable.\n      regularizer: A (Tensor -> Tensor or None) function; the result of applying\n        it on a newly created variable will be added to the collection\n        GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.\n      reuse: a Boolean, None, or tf.AUTO_REUSE. Controls reuse or creation of\n        variables. When eager execution is enabled  this argument is always\n        forced to be False.\n      trainable: If `True` also add the variable to the graph collection\n        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`). `trainable`\n        defaults to `True`, unless `synchronization` is set to `ON_READ`, in\n        which case it defaults to `False`.\n      collections: List of graph collections keys to add the `Variable` to.\n        Defaults to `[GraphKeys.GLOBAL_VARIABLES]` (see `tf.Variable`).\n      caching_device: Optional device string or function describing where the\n        Variable should be cached for reading.  Defaults to the Variable's\n        device.  If not `None`, caches on another device.  Typical use is to\n        cache on the device where the Ops using the `Variable` reside, to\n        deduplicate copying through `Switch` and other conditional statements.\n      partitioner: Optional callable that accepts a fully defined `TensorShape`\n        and dtype of the `Variable` to be created, and returns a list of\n        partitions for each axis (currently only one axis can be partitioned).\n      validate_shape: If False, allows the variable to be initialized with a\n        value of unknown shape. If True, the default, the shape of initial_value\n        must be known.\n      use_resource: If False, creates a regular Variable. If True, creates\n        instead an experimental ResourceVariable which has well-defined\n        semantics. Defaults to False (will later change to True). When eager\n        execution is enabled this argument is always forced to be true.\n      custom_getter: Callable that takes as a first argument the true getter,\n        and allows overwriting the internal get_variable method. The signature\n        of `custom_getter` should match that of this method,\n        but the most future-proof version will allow for changes: `def\n          custom_getter(getter, *args, **kwargs)`.  Direct access to\n        all `get_variable` parameters is also allowed: `def\n          custom_getter(getter, name, *args, **kwargs)`.  A simple identity\n        custom getter that simply creates variables with modified names is:\n          ```python\n        def custom_getter(getter, name, *args, **kwargs): return getter(name +\n          '_suffix', *args, **kwargs) ```\n      constraint: An optional projection function to be applied to the variable\n        after being updated by an `Optimizer` (e.g. used to implement norm\n        constraints or value constraints for layer weights). The function must\n        take as input the unprojected Tensor representing the value of the\n        variable and return the Tensor for the projected value (which must have\n        the same shape). Constraints are not safe to use when doing asynchronous\n        distributed training.\n      synchronization: Indicates when a distributed a variable will be\n        aggregated. Accepted values are constants defined in the class\n        `tf.VariableSynchronization`. By default the synchronization is set to\n        `AUTO` and the current `DistributionStrategy` chooses when to\n        synchronize.\n      aggregation: Indicates how a distributed variable will be aggregated.\n        Accepted values are constants defined in the class\n        `tf.VariableAggregation`.\n\n    Returns:\n      The created or existing `Variable` (or `PartitionedVariable`, if a\n      partitioner was used).\n\n    Raises:\n      ValueError: when creating a new variable and shape is not declared,\n        when reusing a variable and specifying a conflicting shape,\n        or when violating reuse during variable creation.\n      RuntimeError: when eager execution is enabled and not called from an\n        EagerVariableStore.\n    "
        if custom_getter is not None and (not callable(custom_getter)):
            raise ValueError('Passed a custom_getter which is not callable: %s' % custom_getter)
        with ops.init_scope():
            if context.executing_eagerly():
                use_resource = True
        if context.executing_eagerly():
            if not self._store_eager_variables and reuse:
                raise RuntimeError('When eager execution is enabled variable reuse is only supported when an EagerVariableStore is active. See the documentation on EagerVariableStore for example usage.')
            if self._store_eager_variables:
                reuse = AUTO_REUSE
        try:
            dtype = dtype.base_dtype
        except AttributeError:
            pass

        def _true_getter(name, shape=None, dtype=dtypes.float32, initializer=None, regularizer=None, reuse=None, trainable=None, collections=None, caching_device=None, partitioner=None, validate_shape=True, use_resource=None, constraint=None, synchronization=VariableSynchronization.AUTO, aggregation=VariableAggregation.NONE):
            if False:
                return 10
            is_scalar = shape is not None and isinstance(shape, collections_abc.Sequence) and (not shape)
            if partitioner is not None and (not is_scalar):
                if not callable(partitioner):
                    raise ValueError('Partitioner must be callable, but received: %s' % partitioner)
                with ops.name_scope(None):
                    return self._get_partitioned_variable(name=name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer, reuse=reuse, trainable=trainable, collections=collections, caching_device=caching_device, partitioner=partitioner, validate_shape=validate_shape, use_resource=use_resource, constraint=constraint, synchronization=synchronization, aggregation=aggregation)
            if reuse is True and partitioner is None and (name in self._partitioned_vars):
                return self._get_partitioned_variable(name=name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer, reuse=reuse, trainable=trainable, collections=collections, caching_device=caching_device, partitioner=None, validate_shape=validate_shape, use_resource=use_resource, constraint=constraint, synchronization=synchronization, aggregation=aggregation)
            if '%s/part_0' % name in self._vars:
                raise ValueError('No partitioner was provided, but a partitioned version of the variable was found: %s/part_0. Perhaps a variable of the same name was already created with partitioning?' % name)
            return self._get_single_variable(name=name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer, reuse=reuse, trainable=trainable, collections=collections, caching_device=caching_device, validate_shape=validate_shape, use_resource=use_resource, constraint=constraint, synchronization=synchronization, aggregation=aggregation)
        (synchronization, aggregation, trainable) = variables.validate_synchronization_aggregation_trainable(synchronization, aggregation, trainable, name)
        if custom_getter is not None:
            custom_getter_kwargs = {'getter': _true_getter, 'name': name, 'shape': shape, 'dtype': dtype, 'initializer': initializer, 'regularizer': regularizer, 'reuse': reuse, 'trainable': trainable, 'collections': collections, 'caching_device': caching_device, 'partitioner': partitioner, 'validate_shape': validate_shape, 'use_resource': use_resource, 'synchronization': synchronization, 'aggregation': aggregation}
            if 'constraint' in function_utils.fn_args(custom_getter) or function_utils.has_kwargs(custom_getter):
                custom_getter_kwargs['constraint'] = constraint
            return custom_getter(**custom_getter_kwargs)
        else:
            return _true_getter(name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer, reuse=reuse, trainable=trainable, collections=collections, caching_device=caching_device, partitioner=partitioner, validate_shape=validate_shape, use_resource=use_resource, constraint=constraint, synchronization=synchronization, aggregation=aggregation)

    def _get_partitioned_variable(self, name, partitioner, shape=None, dtype=dtypes.float32, initializer=None, regularizer=None, reuse=None, trainable=None, collections=None, caching_device=None, validate_shape=True, use_resource=None, constraint=None, synchronization=VariableSynchronization.AUTO, aggregation=VariableAggregation.NONE):
        if False:
            for i in range(10):
                print('nop')
        "Gets or creates a sharded variable list with these parameters.\n\n    The `partitioner` must be a callable that accepts a fully defined\n    `TensorShape` and returns a sequence of integers (the `partitions`).\n    These integers describe how to partition the given sharded `Variable`\n    along the given dimension.  That is, `partitions[1] = 3` means split\n    the `Variable` into 3 shards along dimension 1.  Currently, sharding along\n    only one axis is supported.\n\n    If the list of variables with the given name (prefix) is already stored,\n    we return the stored variables. Otherwise, we create a new one.\n\n    Set `reuse` to `True` when you only want to reuse existing Variables.\n    Set `reuse` to `False` when you only want to create new Variables.\n    Set `reuse` to None (the default) or tf.compat.v1.AUTO_REUSE when you want\n    variables to be created if they don't exist or returned if they do.\n\n    If initializer is `None` (the default), the default initializer passed in\n    the constructor is used. If that one is `None` too, we use a new\n    `glorot_uniform_initializer`. If initializer is a Tensor, we use\n    it as a value and derive the shape from the initializer.\n\n    If the initializer is a callable, then it will be called for each\n    shard.  Otherwise the initializer should match the shape of the entire\n    sharded Variable, and it will be sliced accordingly for each shard.\n\n    Some useful partitioners are available.  See, e.g.,\n    `variable_axis_size_partitioner` and `min_max_variable_partitioner`.\n\n    Args:\n      name: the name of the new or existing sharded variable.\n      partitioner: Optional callable that accepts a fully defined `TensorShape`\n        and `dtype` of the Variable to be created, and returns a list of\n        partitions for each axis (currently only one axis can be partitioned).\n      shape: shape of the new or existing sharded variable.\n      dtype: type of the new or existing sharded variable (defaults to\n        `DT_FLOAT`).\n      initializer: initializer for the sharded variable.\n      regularizer: a (Tensor -> Tensor or None) function; the result of applying\n        it on a newly created variable will be added to the collection\n        GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.\n      reuse: a Boolean, None, or tf.AUTO_REUSE. Controls reuse or creation of\n        variables.\n      trainable: If `True` also add the variable to the graph collection\n        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).\n      collections: List of graph collections keys to add the Variable to.\n        Defaults to `[GraphKeys.GLOBAL_VARIABLES]` (see `tf.Variable`).\n      caching_device: Optional device string or function describing where the\n        Variable should be cached for reading.  Defaults to the Variable's\n        device.  If not `None`, caches on another device.  Typical use is to\n        cache on the device where the Ops using the Variable reside, to\n        deduplicate copying through `Switch` and other conditional statements.\n      validate_shape: If False, allows the variable to be initialized with a\n        value of unknown shape. If True, the default, the shape of initial_value\n        must be known.\n      use_resource: If False, creates a regular Variable. If True, creates an\n        experimental ResourceVariable which has well-defined semantics. Defaults\n        to False (will later change to True).\n      constraint: An optional projection function to be applied to the variable\n        after being updated by an `Optimizer` (e.g. used to implement norm\n        constraints or value constraints for layer weights). The function must\n        take as input the unprojected Tensor representing the value of the\n        variable and return the Tensor for the projected value (which must have\n        the same shape). Constraints are not safe to use when doing asynchronous\n        distributed training.\n      synchronization: Indicates when a distributed a variable will be\n        aggregated. Accepted values are constants defined in the class\n        `tf.VariableSynchronization`. By default the synchronization is set to\n        `AUTO` and the current `DistributionStrategy` chooses when to\n        synchronize.\n      aggregation: Indicates how a distributed variable will be aggregated.\n        Accepted values are constants defined in the class\n        `tf.VariableAggregation`.\n\n    Returns:\n      A `PartitionedVariable` object.\n\n    Raises:\n      ValueError: when creating a new variable and shape is not declared,\n        when reusing a variable and specifying a conflicting shape,\n        when violating reuse during variable creation, or if an existing\n        sharded variable exists for the given name but with different sharding.\n    "
        initializing_from_value = initializer is not None and isinstance(initializer, tensor.Tensor)
        if name in self._vars:
            raise ValueError('A partitioner was provided, but an unpartitioned version of the variable was found: %s.  Perhaps a variable of the same name was already created without partitioning?' % name)
        shape = tensor_shape.as_shape(shape)
        if initializing_from_value:
            shape = shape.merge_with(initializer.get_shape())
        partitions = None
        if not reuse or partitioner:
            partitions = _call_partitioner(partitioner, shape, dtype)
        if name in self._partitioned_vars:
            if reuse is False:
                raise ValueError('Partitioned variable with name %s already exists. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?' % name)
            existing_var = self._partitioned_vars[name]
            if not shape.is_compatible_with(existing_var.get_shape()):
                raise ValueError('Trying to reuse partitioned variable %s, but specified shape %s and found shape %s.' % (name, shape, existing_var.get_shape()))
            if not dtype.is_compatible_with(existing_var.dtype):
                raise ValueError('Trying to reuse partitioned variable %s, but specified dtype %s and found dtype %s.' % (name, dtype.name, existing_var.dtype.name))
            if partitions is not None and existing_var._get_partitions() != partitions:
                raise ValueError('Trying to reuse partitioned variable %s, but specified partitions %s and found partitions %s.' % (name, partitions, existing_var._get_partitions()))
            return existing_var
        if reuse is True:
            raise ValueError('PartitionedVariable %s does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=False or reuse=tf.AUTO_REUSE in VarScope?' % name)
        (slice_dim, num_slices) = _get_slice_dim_and_num_slices(partitions)
        if '%s/part_0' % name in self._vars:
            if '%s/part_%d' % (name, num_slices - 1) not in self._vars:
                raise ValueError('Partitioner returned a different partitioning than what was already found.  Partitioner returned %d shards, and shard %s/part_0 was found, but %s/part_%d was not.' % (num_slices, name, name, num_slices - 1))
            if '%s/part_%d' % (name, num_slices) in self._vars:
                raise ValueError('Partitioner returned a different partitioning than what was already found.  Partitioner returned %d shards, and shard %s/part_0 was found, but so was the extra shard %s/part_%d.' % (num_slices, name, name, num_slices))
        vs = []
        for (i, (var_offset, var_shape)) in enumerate(_iter_slices(shape.as_list(), num_slices, slice_dim)):
            partition_info = _PartitionInfo(full_shape=shape.as_list(), var_offset=var_offset)
            var_full_name = '%s/part_%d' % (name, i)
            with ops.name_scope(var_full_name + '/PartitionedInitializer', skip_on_eager=False):
                if initializer is None:
                    (init, initializing_from_value) = self._get_default_initializer(name=name, shape=shape, dtype=dtype)
                    if initializing_from_value:
                        init_shape = None
                    else:
                        init_shape = var_shape
                elif callable(initializer):
                    init = initializer
                    init_shape = var_shape
                elif isinstance(initializer, tensor.Tensor):
                    init = array_ops.slice(initializer, var_offset, var_shape)
                    dtype = init.dtype.base_dtype
                    init_shape = None
                else:
                    init = ops.convert_to_tensor(initializer, dtype=dtype)
                    init = array_ops.slice(init, var_offset, var_shape)
                    init_shape = None
            with ops.name_scope(None):
                var = self._get_single_variable(name=var_full_name, shape=init_shape, dtype=dtype, initializer=init, partition_info=partition_info, regularizer=regularizer, reuse=reuse, trainable=trainable, collections=collections, caching_device=caching_device, validate_shape=validate_shape, use_resource=use_resource, constraint=constraint, synchronization=synchronization, aggregation=aggregation)
            var._set_save_slice_info(variables.Variable.SaveSliceInfo(name, shape.as_list(), var_offset, var_shape))
            vs.append(var)
        partitioned_var = variables.PartitionedVariable(name=name, shape=shape, dtype=dtype, variable_list=vs, partitions=partitions)
        if not context.executing_eagerly() or self._store_eager_variables:
            self._partitioned_vars[name] = partitioned_var
        return partitioned_var

    def _get_single_variable(self, name, shape=None, dtype=dtypes.float32, initializer=None, regularizer=None, partition_info=None, reuse=None, trainable=None, collections=None, caching_device=None, validate_shape=True, use_resource=None, constraint=None, synchronization=VariableSynchronization.AUTO, aggregation=VariableAggregation.NONE):
        if False:
            return 10
        'Get or create a single Variable (e.g.\n\n    a shard or entire variable).\n\n    See the documentation of get_variable above (ignore partitioning components)\n    for details.\n\n    Args:\n      name: see get_variable.\n      shape: see get_variable.\n      dtype: see get_variable.\n      initializer: see get_variable.\n      regularizer: see get_variable.\n      partition_info: _PartitionInfo object.\n      reuse: see get_variable.\n      trainable: see get_variable.\n      collections: see get_variable.\n      caching_device: see get_variable.\n      validate_shape: see get_variable.\n      use_resource: see get_variable.\n      constraint: see get_variable.\n      synchronization: see get_variable.\n      aggregation: see get_variable.\n\n    Returns:\n      A Variable.  See documentation of get_variable above.\n\n    Raises:\n      ValueError: See documentation of get_variable above.\n    '
        initializing_from_value = False
        if initializer is not None and (not callable(initializer)):
            initializing_from_value = True
        if shape is not None and initializing_from_value:
            raise ValueError('If initializer is a constant, do not specify shape.')
        dtype = dtypes.as_dtype(dtype)
        if shape is not None:
            shape = tensor_shape.as_shape(shape)
        if name in self._vars:
            if reuse is False:
                var = self._vars[name]
                err_msg = 'Variable %s already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?' % name
                if isinstance(var, resource_variable_ops.ResourceVariable):
                    raise ValueError(err_msg)
                tb = var.op.traceback[::-1]
                tb = [x for x in tb if 'tensorflow/python' not in x[0]][:5]
                raise ValueError('%s Originally defined at:\n\n%s' % (err_msg, ''.join(traceback.format_list(tb))))
            found_var = self._vars[name]
            if shape is not None and (not shape.is_compatible_with(found_var.get_shape())):
                raise ValueError('Trying to share variable %s, but specified shape %s and found shape %s.' % (name, shape, found_var.get_shape()))
            if not dtype.is_compatible_with(found_var.dtype):
                dtype_str = dtype.name
                found_type_str = found_var.dtype.name
                raise ValueError('Trying to share variable %s, but specified dtype %s and found dtype %s.' % (name, dtype_str, found_type_str))
            return found_var
        if reuse is True:
            raise ValueError('Variable %s does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=tf.AUTO_REUSE in VarScope?' % name)
        if initializer is None:
            if shape is None:
                raise ValueError(f'Variable {name} did not get an initializer, so its `shape` argument must be specified.')
            (initializer, initializing_from_value) = self._get_default_initializer(name=name, shape=shape, dtype=dtype)
        with ops.init_scope():
            if initializing_from_value:
                init_val = initializer
                variable_dtype = None
            else:
                if tf_inspect.isclass(initializer):
                    initializer = initializer()
                if shape is not None and shape.is_fully_defined():
                    if 'partition_info' in tf_inspect.getargspec(initializer).args:
                        init_val = functools.partial(initializer, shape.as_list(), dtype=dtype, partition_info=partition_info)
                    else:
                        init_val = functools.partial(initializer, shape.as_list(), dtype=dtype)
                    variable_dtype = dtype.base_dtype
                elif _needs_no_arguments(initializer):
                    init_val = initializer
                    variable_dtype = None
                else:
                    raise ValueError("The initializer passed is not valid. It should be a callable with no arguments and the shape should not be provided or an instance of `tf.keras.initializers.*' and `shape` should be fully defined.")
        if use_resource is None:
            use_resource = resource_variables_toggle.resource_variables_enabled()
        v = _variable_v1(initial_value=init_val, name=name, trainable=trainable, collections=collections, caching_device=caching_device, dtype=variable_dtype, validate_shape=validate_shape, constraint=constraint, use_resource=use_resource, synchronization=synchronization, aggregation=aggregation, shape=shape)
        if context.executing_eagerly() and self._store_eager_variables:
            if collections:
                ops.add_to_collections(collections, v)
            else:
                ops.add_to_collection(ops.GraphKeys.GLOBAL_VARIABLES, v)
            if trainable:
                ops.add_to_collection(ops.GraphKeys.TRAINABLE_VARIABLES, v)
        if not context.executing_eagerly() or self._store_eager_variables:
            self._vars[name] = v
        logging.vlog(1, 'Created variable %s with shape %s and init %s', v.name, format(shape), initializer)
        if regularizer:

            def make_regularizer_op():
                if False:
                    while True:
                        i = 10
                with ops.colocate_with(v):
                    with ops.name_scope(name + '/Regularizer/'):
                        return regularizer(v)
            if regularizer(v) is not None:
                lazy_eval_tensor = _LazyEvalTensor(make_regularizer_op)
                ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, lazy_eval_tensor)
        return v

    def _get_default_initializer(self, name, shape=None, dtype=dtypes.float32):
        if False:
            print('Hello World!')
        'Provide a default initializer and a corresponding value.\n\n    Args:\n      name: see get_variable.\n      shape: see get_variable.\n      dtype: see get_variable.\n\n    Returns:\n      initializer and initializing_from_value. See get_variable above.\n\n    Raises:\n      ValueError: When giving unsupported dtype.\n    '
        del shape
        if dtype.is_floating:
            initializer = init_ops.glorot_uniform_initializer()
            initializing_from_value = False
        elif dtype.is_integer or dtype.is_unsigned or dtype.is_bool or (dtype == dtypes.string):
            initializer = init_ops.zeros_initializer()
            initializing_from_value = False
        else:
            raise ValueError('An initializer for variable %s of %s is required' % (name, dtype.base_dtype))
        return (initializer, initializing_from_value)

class _LazyEvalTensor(core.Tensor):
    """A Tensor-like object that only evaluates its thunk when used."""

    def __init__(self, thunk):
        if False:
            for i in range(10):
                print('nop')
        'Initializes a _LazyEvalTensor object.\n\n    Args:\n      thunk: A callable. A thunk which computes the value of the tensor.\n    '
        self._thunk = thunk
        self._master_tensor = thunk()

    def _as_tensor(self, dtype=None, name=None, as_ref=False):
        if False:
            for i in range(10):
                print('nop')
        del name
        assert not as_ref
        assert dtype in [None, self.dtype]
        return self._thunk()

def _make_master_property(name):
    if False:
        for i in range(10):
            print('nop')

    @property
    def prop(self):
        if False:
            while True:
                i = 10
        return getattr(self._master_tensor, name)
    return prop
_master_property_list = ('device', 'dtype', 'graph', 'name', 'op', 'shape', 'value_index')
for _name in _master_property_list:
    setattr(_LazyEvalTensor, _name, _make_master_property(_name))

def _make_master_method(name):
    if False:
        print('Hello World!')

    def method(self, *args, **kwargs):
        if False:
            return 10
        return getattr(self._master_tensor, name)(*args, **kwargs)
    return method
_master_method_list = ('get_shape', '__str__', 'shape_as_list')
for _name in _master_method_list:
    setattr(_LazyEvalTensor, _name, _make_master_method(_name))

def _make_op_method(name):
    if False:
        return 10

    def method(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self._as_tensor(), name)(*args, **kwargs)
    return method
_op_list = ('__abs__', '__add__', '__and__', '__bool__', '__div__', '__eq__', '__floordiv__', '__ge__', '__getitem__', '__gt__', '__invert__', '__iter__', '__le__', '__len__', '__lt__', '__matmul__', '__mod__', '__mul__', '__ne__', '__neg__', '__nonzero__', '__or__', '__pow__', '__radd__', '__rand__', '__rdiv__', '__rfloordiv__', '__rmatmul__', '__rmod__', '__rmul__', '__ror__', '__rpow__', '__rsub__', '__rtruediv__', '__rxor__', '__sub__', '__truediv__', '__xor__', 'eval', 'numpy')
for _name in _op_list:
    setattr(_LazyEvalTensor, _name, _make_op_method(_name))
tensor_conversion_registry.register_tensor_conversion_function(_LazyEvalTensor, lambda val, dtype, name, as_ref: val._as_tensor(dtype, name, as_ref))
session.register_session_run_conversion_functions(_LazyEvalTensor, lambda fetch: ([fetch._master_tensor], lambda fetched_vals: fetched_vals[0]))

@tf_export(v1=['no_regularizer'])
def no_regularizer(_):
    if False:
        for i in range(10):
            print('nop')
    'Use this function to prevent regularization of variables.'
    return None

@tf_export(v1=['VariableScope'])
class VariableScope:
    """Variable scope object to carry defaults to provide to `get_variable`.

  Many of the arguments we need for `get_variable` in a variable store are most
  easily handled with a context. This object is used for the defaults.

  Attributes:
    name: name of the current scope, used as prefix in get_variable.
    initializer: default initializer passed to get_variable.
    regularizer: default regularizer passed to get_variable.
    reuse: Boolean, None, or tf.compat.v1.AUTO_REUSE, setting the reuse in
      get_variable. When eager execution is enabled this argument is always
      forced to be False.
    caching_device: string, callable, or None: the caching device passed to
      get_variable.
    partitioner: callable or `None`: the partitioner passed to `get_variable`.
    custom_getter: default custom getter passed to get_variable.
    name_scope: The name passed to `tf.name_scope`.
    dtype: default type passed to get_variable (defaults to DT_FLOAT).
    use_resource: if False, create a normal Variable; if True create an
      experimental ResourceVariable with well-defined semantics. Defaults to
      False (will later change to True). When eager execution is enabled this
      argument is always forced to be True.
    constraint: An optional projection function to be applied to the variable
      after being updated by an `Optimizer` (e.g. used to implement norm
      constraints or value constraints for layer weights). The function must
      take as input the unprojected Tensor representing the value of the
      variable and return the Tensor for the projected value (which must have
      the same shape). Constraints are not safe to use when doing asynchronous
      distributed training.
  """

    def __init__(self, reuse, name='', initializer=None, regularizer=None, caching_device=None, partitioner=None, custom_getter=None, name_scope='', dtype=dtypes.float32, use_resource=None, constraint=None):
        if False:
            while True:
                i = 10
        'Creates a new VariableScope with the given properties.'
        self._name = name
        self._initializer = initializer
        self._regularizer = regularizer
        self._reuse = reuse
        self._caching_device = caching_device
        self._partitioner = partitioner
        self._custom_getter = custom_getter
        self._name_scope = name_scope
        self._dtype = dtype
        self._use_resource = use_resource
        self._constraint = constraint
        if context.executing_eagerly():
            if self._caching_device is not None:
                raise NotImplementedError('Caching devices is not yet supported when eager execution is enabled.')
            self._reuse = AUTO_REUSE
            self._use_resource = True

    @property
    def name(self):
        if False:
            while True:
                i = 10
        return self._name

    @property
    def original_name_scope(self):
        if False:
            i = 10
            return i + 15
        return self._name_scope

    @property
    def reuse(self):
        if False:
            print('Hello World!')
        return self._reuse

    @property
    def initializer(self):
        if False:
            while True:
                i = 10
        return self._initializer

    @property
    def dtype(self):
        if False:
            while True:
                i = 10
        return self._dtype

    @property
    def use_resource(self):
        if False:
            while True:
                i = 10
        return self._use_resource

    @property
    def regularizer(self):
        if False:
            print('Hello World!')
        return self._regularizer

    @property
    def caching_device(self):
        if False:
            print('Hello World!')
        return self._caching_device

    @property
    def partitioner(self):
        if False:
            print('Hello World!')
        return self._partitioner

    @property
    def custom_getter(self):
        if False:
            while True:
                i = 10
        return self._custom_getter

    @property
    def constraint(self):
        if False:
            print('Hello World!')
        return self._constraint

    def reuse_variables(self):
        if False:
            i = 10
            return i + 15
        'Reuse variables in this scope.'
        self._reuse = True

    def set_initializer(self, initializer):
        if False:
            print('Hello World!')
        'Set initializer for this scope.'
        self._initializer = initializer

    def set_dtype(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        'Set data type for this scope.'
        self._dtype = dtype

    def set_use_resource(self, use_resource):
        if False:
            print('Hello World!')
        'Sets whether to use ResourceVariables for this scope.'
        if context.executing_eagerly() and (not use_resource):
            raise ValueError('When eager execution is enabled, use_resource cannot be set to false.')
        self._use_resource = use_resource

    def set_regularizer(self, regularizer):
        if False:
            return 10
        'Set regularizer for this scope.'
        self._regularizer = regularizer

    def set_caching_device(self, caching_device):
        if False:
            for i in range(10):
                print('nop')
        'Set caching_device for this scope.'
        if context.executing_eagerly():
            raise NotImplementedError('Caching devices are not yet supported when eager execution is enabled.')
        self._caching_device = caching_device

    def set_partitioner(self, partitioner):
        if False:
            return 10
        'Set partitioner for this scope.'
        self._partitioner = partitioner

    def set_custom_getter(self, custom_getter):
        if False:
            return 10
        'Set custom getter for this scope.'
        self._custom_getter = custom_getter

    def get_collection(self, name):
        if False:
            for i in range(10):
                print('nop')
        "Get this scope's variables."
        scope = self._name + '/' if self._name else ''
        return ops.get_collection(name, scope)

    def trainable_variables(self):
        if False:
            while True:
                i = 10
        "Get this scope's trainable variables."
        return self.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)

    def global_variables(self):
        if False:
            return 10
        "Get this scope's global variables."
        return self.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

    def local_variables(self):
        if False:
            while True:
                i = 10
        "Get this scope's local variables."
        return self.get_collection(ops.GraphKeys.LOCAL_VARIABLES)

    def get_variable(self, var_store, name, shape=None, dtype=None, initializer=None, regularizer=None, reuse=None, trainable=None, collections=None, caching_device=None, partitioner=None, validate_shape=True, use_resource=None, custom_getter=None, constraint=None, synchronization=VariableSynchronization.AUTO, aggregation=VariableAggregation.NONE):
        if False:
            while True:
                i = 10
        'Gets an existing variable with this name or create a new one.'
        if regularizer is None:
            regularizer = self._regularizer
        if caching_device is None:
            caching_device = self._caching_device
        if partitioner is None:
            partitioner = self._partitioner
        if custom_getter is None:
            custom_getter = self._custom_getter
        if context.executing_eagerly():
            reuse = False
            use_resource = True
        else:
            if reuse is None:
                reuse = self._reuse
            if use_resource is None:
                use_resource = self._use_resource
        full_name = self.name + '/' + name if self.name else name
        with ops.name_scope(None, skip_on_eager=False):
            if dtype is not None and initializer is not None and (not callable(initializer)):
                init_dtype = ops.convert_to_tensor(initializer).dtype.base_dtype
                if init_dtype != dtype:
                    raise ValueError("Initializer type '%s' and explicit dtype '%s' don't match." % (init_dtype, dtype))
            if initializer is None:
                initializer = self._initializer
            if constraint is None:
                constraint = self._constraint
            if dtype is None:
                dtype = self._dtype
            return var_store.get_variable(full_name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer, reuse=reuse, trainable=trainable, collections=collections, caching_device=caching_device, partitioner=partitioner, validate_shape=validate_shape, use_resource=use_resource, custom_getter=custom_getter, constraint=constraint, synchronization=synchronization, aggregation=aggregation)

    def _get_partitioned_variable(self, var_store, name, shape=None, dtype=None, initializer=None, regularizer=None, trainable=None, collections=None, caching_device=None, partitioner=None, validate_shape=True, use_resource=None, constraint=None, synchronization=VariableSynchronization.AUTO, aggregation=VariableAggregation.NONE):
        if False:
            print('Hello World!')
        'Gets an existing variable with this name or create a new one.'
        if initializer is None:
            initializer = self._initializer
        if regularizer is None:
            regularizer = self._regularizer
        if constraint is None:
            constraint = self._constraint
        if caching_device is None:
            caching_device = self._caching_device
        if partitioner is None:
            partitioner = self._partitioner
        if dtype is None:
            dtype = self._dtype
        if use_resource is None:
            use_resource = self._use_resource
        if self._custom_getter is not None:
            raise ValueError("Private access to _get_partitioned_variable is not allowed when a custom getter is set.  Current custom getter: %s.  It is likely that you're using create_partitioned_variables.  If so, consider instead using get_variable with a non-empty partitioner parameter instead." % self._custom_getter)
        if partitioner is None:
            raise ValueError('No partitioner was specified')
        full_name_list = []
        if self.name:
            full_name_list.append(self.name)
        if name:
            full_name_list.append(name)
        full_name = '/'.join(full_name_list)
        with ops.name_scope(None, skip_on_eager=False):
            return var_store._get_partitioned_variable(full_name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer, reuse=self.reuse, trainable=trainable, collections=collections, caching_device=caching_device, partitioner=partitioner, validate_shape=validate_shape, use_resource=use_resource, constraint=constraint, synchronization=synchronization, aggregation=aggregation)
_VARSTORE_KEY = ('__variable_store',)
_VARSCOPESTORE_KEY = ('__varscope',)

class _VariableScopeStore(threading.local):
    """A thread local store for the current variable scope and scope counts."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(_VariableScopeStore, self).__init__()
        self.current_scope = VariableScope(False)
        self.variable_scopes_count = {}

    def open_variable_scope(self, scope_name):
        if False:
            return 10
        if scope_name in self.variable_scopes_count:
            self.variable_scopes_count[scope_name] += 1
        else:
            self.variable_scopes_count[scope_name] = 1

    def close_variable_subscopes(self, scope_name):
        if False:
            return 10
        if scope_name is None:
            for k in self.variable_scopes_count:
                self.variable_scopes_count[k] = 0
        else:
            startswith_check = scope_name + '/'
            startswith_len = len(startswith_check)
            for k in self.variable_scopes_count:
                if k[:startswith_len] == startswith_check:
                    self.variable_scopes_count[k] = 0

    def variable_scope_count(self, scope_name):
        if False:
            while True:
                i = 10
        return self.variable_scopes_count.get(scope_name, 0)

def get_variable_scope_store():
    if False:
        print('Hello World!')
    'Returns the variable scope store for current thread.'
    scope_store = ops.get_collection(_VARSCOPESTORE_KEY)
    if not scope_store:
        scope_store = _VariableScopeStore()
        ops.add_to_collection(_VARSCOPESTORE_KEY, scope_store)
    else:
        scope_store = scope_store[0]
    return scope_store

@tf_export(v1=['get_variable_scope'])
def get_variable_scope():
    if False:
        i = 10
        return i + 15
    "Returns the current variable scope.\n\n  @compatibility(TF2)\n  Although it is a legacy `compat.v1` api,\n  `tf.compat.v1.get_variable` is compatible with eager\n  execution and `tf.function`\n\n  However, to maintain variable-scope based variable reuse\n  you will need to combine it with\n  `tf.compat.v1.keras.utils.track_tf1_style_variables`. (Though\n  it will behave as if reuse is always set to `tf.compat.v1.AUTO_REUSE`.)\n\n  See the\n  [migration guide](https://www.tensorflow.org/guide/migrate/model_mapping)\n  for more info.\n\n  The TF2 equivalent, if you are just trying to track\n  variable name prefixes and not control `get_variable`-based variable reuse,\n  would be to use `tf.name_scope` and capture the output of opening the\n  scope (which represents the current name prefix).\n\n  For example:\n  ```python\n  x = tf.name_scope('foo') as current_scope:\n    ...\n  ```\n  @end_compatibility\n  "
    return get_variable_scope_store().current_scope

def _get_default_variable_store():
    if False:
        print('Hello World!')
    store = ops.get_collection(_VARSTORE_KEY)
    if store:
        return store[0]
    store = _VariableStore()
    ops.add_to_collection(_VARSTORE_KEY, store)
    return store

@tf_contextlib.contextmanager
def with_variable_store(store):
    if False:
        i = 10
        return i + 15
    store_collection = ops.get_collection_ref(_VARSTORE_KEY)
    old = list(store_collection)
    store_collection[:] = [store]
    try:
        yield
    finally:
        store_collection[:] = old

class EagerVariableStore:
    """Wrapper allowing functional layers to be used with eager execution.

  When eager execution is enabled Variables get deleted when they go out of
  scope, and are not stored in global collections by default. A lot of code
  (mostly the functional layers in tf.layers) assumes that variables are kept in
  a global list.

  EagerVariableStore can be used in conjunction with this code to make it
  eager-friendly. For example, to create a dense layer, use:

  ```
    container = tfe.EagerVariableStore()
    for input in dataset_iterator:
      with container.as_default():
        x = tf.compat.v1.layers.dense(input, name="l1")
    print(container.variables)  # Should print the variables used in the layer.
  ```
  """

    def __init__(self, store=None):
        if False:
            i = 10
            return i + 15
        if store is not None:
            if not store._store_eager_variables:
                raise ValueError('Cannot construct EagerVariableStore from a VariableStore object that does not hold eager variables.')
            self._store = store
        else:
            self._store = _VariableStore()
        self._store._store_eager_variables = True

    def as_default(self):
        if False:
            i = 10
            return i + 15
        return with_variable_store(self._store)

    def variables(self):
        if False:
            i = 10
            return i + 15
        return sorted(self._store._vars.values(), key=lambda x: x.name)

    def trainable_variables(self):
        if False:
            return 10
        return sorted([x for x in self._store._vars.values() if x.trainable], key=lambda x: x.name)

    def non_trainable_variables(self):
        if False:
            for i in range(10):
                print('nop')
        return sorted([x for x in self._store._vars.values() if not x.trainable], key=lambda x: x.name)

    def copy(self):
        if False:
            i = 10
            return i + 15
        'Copy this variable store and all of its contents.\n\n    Variables contained in this store will be copied over to the new variable\n    store, meaning that they can be modified without affecting the variables in\n    this store.\n\n    Returns:\n      A new EagerVariableStore instance containing copied variables.\n    '
        new_store = EagerVariableStore()
        for (key, var) in self._store._vars.items():
            try:
                index = var.name.index(':')
            except ValueError:
                stripped_var_name = var.name
            else:
                stripped_var_name = var.name[:index]
            new_var = resource_variable_ops.ResourceVariable(var.read_value(), name=stripped_var_name, trainable=var.trainable)
            new_store._store._vars[key] = new_var
        return new_store

@tf_export(v1=['get_variable'])
def get_variable(name, shape=None, dtype=None, initializer=None, regularizer=None, trainable=None, collections=None, caching_device=None, partitioner=None, validate_shape=True, use_resource=None, custom_getter=None, constraint=None, synchronization=VariableSynchronization.AUTO, aggregation=VariableAggregation.NONE):
    if False:
        for i in range(10):
            print('nop')
    return get_variable_scope().get_variable(_get_default_variable_store(), name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer, trainable=trainable, collections=collections, caching_device=caching_device, partitioner=partitioner, validate_shape=validate_shape, use_resource=use_resource, custom_getter=custom_getter, constraint=constraint, synchronization=synchronization, aggregation=aggregation)
get_variable_or_local_docstring = '%s\n\n@compatibility(TF2)\nAlthough it is a legacy `compat.v1` api,\n`tf.compat.v1.get_variable` is mostly compatible with eager\nexecution and `tf.function` but only if you combine it with the\n`tf.compat.v1.keras.utils.track_tf1_style_variables` decorator. (Though\nit will behave as if reuse is always set to `AUTO_REUSE`.)\n\nSee the\n[model migration guide](https://www.tensorflow.org/guide/migrate/model_mapping)\nfor more info.\n\nIf you do not combine it with\n`tf.compat.v1.keras.utils.track_tf1_style_variables`, `get_variable` will create\na brand new variable every single time it is called and will never reuse\nvariables, regardless of variable names or `reuse` arguments.\n\nThe TF2 equivalent of this symbol would be `tf.Variable`, but note\nthat when using `tf.Variable` you must make sure you track your variables\n(and regularizer arguments) either manually or via `tf.Module` or\n`tf.keras.layers.Layer` mechanisms.\n\nA section of the\n[migration guide](https://www.tensorflow.org/guide/migrate/model_mapping#incremental_migration_to_native_tf2)\nprovides more details on incrementally migrating these usages to `tf.Variable`\nas well.\n\nNote: The `partitioner` arg is not compatible with TF2 behaviors even when\nusing `tf.compat.v1.keras.utils.track_tf1_style_variables`. It can be replaced\nby using `ParameterServerStrategy` and its partitioners. See the\n[multi-gpu migration guide](https://www.tensorflow.org/guide/migrate/multi_worker_cpu_gpu_training)\nand the ParameterServerStrategy guides it references for more info.\n@end_compatibility\n\n%sThis function prefixes the name with the current variable scope\nand performs reuse checks. See the\n[Variable Scope How To](https://tensorflow.org/guide/variables)\nfor an extensive description of how reusing works. Here is a basic example:\n\n```python\ndef foo():\n  with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):\n    v = tf.get_variable("v", [1])\n  return v\n\nv1 = foo()  # Creates v.\nv2 = foo()  # Gets the same, existing v.\nassert v1 == v2\n```\n\nIf initializer is `None` (the default), the default initializer passed in\nthe variable scope will be used. If that one is `None` too, a\n`glorot_uniform_initializer` will be used. The initializer can also be\na Tensor, in which case the variable is initialized to this value and shape.\n\nSimilarly, if the regularizer is `None` (the default), the default regularizer\npassed in the variable scope will be used (if that is `None` too,\nthen by default no regularization is performed).\n\nIf a partitioner is provided, a `PartitionedVariable` is returned.\nAccessing this object as a `Tensor` returns the shards concatenated along\nthe partition axis.\n\nSome useful partitioners are available.  See, e.g.,\n`variable_axis_size_partitioner` and `min_max_variable_partitioner`.\n\nArgs:\n  name: The name of the new or existing variable.\n  shape: Shape of the new or existing variable.\n  dtype: Type of the new or existing variable (defaults to `DT_FLOAT`).\n  initializer: Initializer for the variable if one is created. Can either be\n    an initializer object or a Tensor. If it\'s a Tensor, its shape must be known\n    unless validate_shape is False.\n  regularizer: A (Tensor -> Tensor or None) function; the result of\n    applying it on a newly created variable will be added to the collection\n    `tf.GraphKeys.REGULARIZATION_LOSSES` and can be used for regularization.\n  %scollections: List of graph collections keys to add the Variable to.\n    Defaults to `[%s]` (see `tf.Variable`).\n  caching_device: Optional device string or function describing where the\n    Variable should be cached for reading.  Defaults to the Variable\'s\n    device.  If not `None`, caches on another device.  Typical use is to\n    cache on the device where the Ops using the Variable reside, to\n    deduplicate copying through `Switch` and other conditional statements.\n  partitioner: Optional callable that accepts a fully defined `TensorShape`\n    and `dtype` of the Variable to be created, and returns a list of\n    partitions for each axis (currently only one axis can be partitioned).\n  validate_shape: If False, allows the variable to be initialized with a\n      value of unknown shape. If True, the default, the shape of initial_value\n      must be known. For this to be used the initializer must be a Tensor and\n      not an initializer object.\n  use_resource: If False, creates a regular Variable. If true, creates an\n    experimental ResourceVariable instead with well-defined semantics.\n    Defaults to False (will later change to True). When eager execution is\n    enabled this argument is always forced to be True.\n  custom_getter: Callable that takes as a first argument the true getter, and\n    allows overwriting the internal get_variable method.\n    The signature of `custom_getter` should match that of this method,\n    but the most future-proof version will allow for changes:\n    `def custom_getter(getter, *args, **kwargs)`.  Direct access to\n    all `get_variable` parameters is also allowed:\n    `def custom_getter(getter, name, *args, **kwargs)`.  A simple identity\n    custom getter that simply creates variables with modified names is:\n    ```python\n    def custom_getter(getter, name, *args, **kwargs):\n      return getter(name + \'_suffix\', *args, **kwargs)\n    ```\n  constraint: An optional projection function to be applied to the variable\n    after being updated by an `Optimizer` (e.g. used to implement norm\n    constraints or value constraints for layer weights). The function must\n    take as input the unprojected Tensor representing the value of the\n    variable and return the Tensor for the projected value\n    (which must have the same shape). Constraints are not safe to\n    use when doing asynchronous distributed training.\n  synchronization: Indicates when a distributed a variable will be\n    aggregated. Accepted values are constants defined in the class\n    `tf.VariableSynchronization`. By default the synchronization is set to\n    `AUTO` and the current `DistributionStrategy` chooses\n    when to synchronize.\n  aggregation: Indicates how a distributed variable will be aggregated.\n    Accepted values are constants defined in the class\n    `tf.VariableAggregation`.\n\nReturns:\n  The created or existing `Variable` (or `PartitionedVariable`, if a\n  partitioner was used).\n\nRaises:\n  ValueError: when creating a new variable and shape is not declared,\n    when violating reuse during variable creation, or when `initializer` dtype\n    and `dtype` don\'t match. Reuse is set inside `variable_scope`.\n'
get_variable.__doc__ = get_variable_or_local_docstring % ('Gets an existing variable with these parameters or create a new one.', '', 'trainable: If `True` also add the variable to the graph collection\n    `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).\n  ', 'GraphKeys.GLOBAL_VARIABLES')

@tf_export(v1=['get_local_variable'])
def get_local_variable(name, shape=None, dtype=None, initializer=None, regularizer=None, trainable=False, collections=None, caching_device=None, partitioner=None, validate_shape=True, use_resource=None, custom_getter=None, constraint=None, synchronization=VariableSynchronization.AUTO, aggregation=VariableAggregation.NONE):
    if False:
        return 10
    if collections:
        collections += [ops.GraphKeys.LOCAL_VARIABLES]
    else:
        collections = [ops.GraphKeys.LOCAL_VARIABLES]
    return get_variable(name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer, trainable=False, collections=collections, caching_device=caching_device, partitioner=partitioner, validate_shape=validate_shape, use_resource=use_resource, synchronization=synchronization, aggregation=aggregation, custom_getter=custom_getter, constraint=constraint)
get_local_variable.__doc__ = get_variable_or_local_docstring % ('Gets an existing *local* variable or creates a new one.', 'Behavior is the same as in `get_variable`, except that variables are\nadded to the `LOCAL_VARIABLES` collection and `trainable` is set to\n`False`.\n', '', 'GraphKeys.LOCAL_VARIABLES')

def _get_partitioned_variable(name, shape=None, dtype=None, initializer=None, regularizer=None, trainable=True, collections=None, caching_device=None, partitioner=None, validate_shape=True, use_resource=None, constraint=None, synchronization=VariableSynchronization.AUTO, aggregation=VariableAggregation.NONE):
    if False:
        for i in range(10):
            print('nop')
    "Gets or creates a sharded variable list with these parameters.\n\n  The `partitioner` must be a callable that accepts a fully defined\n  `TensorShape` and returns a sequence of integers (the `partitions`).\n  These integers describe how to partition the given sharded `Variable`\n  along the given dimension.  That is, `partitions[1] = 3` means split\n  the `Variable` into 3 shards along dimension 1.  Currently, sharding along\n  only one axis is supported.\n\n  If the list of variables with the given name (prefix) is already stored,\n  we return the stored variables. Otherwise, we create a new one.\n\n  If initializer is `None` (the default), the default initializer passed in\n  the constructor is used. If that one is `None` too, we use a new\n  `glorot_uniform_initializer`. If initializer is a Tensor, we use\n  it as a value and derive the shape from the initializer.\n\n  If the initializer is a callable, then it will be called for each\n  shard.  Otherwise the initializer should match the shape of the entire\n  sharded Variable, and it will be sliced accordingly for each shard.\n\n  Some useful partitioners are available.  See, e.g.,\n  `variable_axis_size_partitioner` and `min_max_variable_partitioner`.\n\n  Args:\n    name: The name of the new or existing variable.\n    shape: Shape of the new or existing variable.\n    dtype: Type of the new or existing variable (defaults to `DT_FLOAT`).\n    initializer: Initializer for the variable if one is created.\n    regularizer: A (Tensor -> Tensor or None) function; the result of applying\n      it on a newly created variable will be added to the collection\n      GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.\n    trainable: If `True` also add the variable to the graph collection\n      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).\n    collections: List of graph collections keys to add the Variable to. Defaults\n      to `[GraphKeys.GLOBAL_VARIABLES]` (see `tf.Variable`).\n    caching_device: Optional device string or function describing where the\n      Variable should be cached for reading.  Defaults to the Variable's device.\n      If not `None`, caches on another device.  Typical use is to cache on the\n      device where the Ops using the Variable reside, to deduplicate copying\n      through `Switch` and other conditional statements.\n    partitioner: Optional callable that accepts a fully defined `TensorShape`\n      and `dtype` of the Variable to be created, and returns a list of\n      partitions for each axis (currently only one axis can be partitioned).\n    validate_shape: If False, allows the variable to be initialized with a value\n      of unknown shape. If True, the default, the shape of initial_value must be\n      known.\n    use_resource: If False, creates a regular Variable. If True, creates an\n      experimental ResourceVariable instead which has well-defined semantics.\n      Defaults to False (will later change to True).\n    constraint: An optional projection function to be applied to the variable\n      after being updated by an `Optimizer` (e.g. used to implement norm\n      constraints or value constraints for layer weights). The function must\n      take as input the unprojected Tensor representing the value of the\n      variable and return the Tensor for the projected value (which must have\n      the same shape). Constraints are not safe to use when doing asynchronous\n      distributed training.\n    synchronization: Indicates when a distributed a variable will be aggregated.\n      Accepted values are constants defined in the class\n      `tf.VariableSynchronization`. By default the synchronization is set to\n      `AUTO` and the current `DistributionStrategy` chooses when to synchronize.\n    aggregation: Indicates how a distributed variable will be aggregated.\n      Accepted values are constants defined in the class\n      `tf.VariableAggregation`.\n\n  Returns:\n    A tuple `(shards, partitions)` where `shards` is the list of `Variable`\n    shards and `partitions` is the output of the partitioner on the input\n    shape.\n\n  Raises:\n    ValueError: when creating a new variable and shape is not declared,\n      or when violating reuse during variable creation. Reuse is set inside\n      `variable_scope`.\n  "
    scope = get_variable_scope()
    if scope.custom_getter is not None:
        raise ValueError("Private access to _get_partitioned_variable is not allowed when a custom getter is set.  Current custom getter: %s.  It is likely that you're using create_partitioned_variables.  If so, consider instead using get_variable with a non-empty partitioner parameter instead." % scope.custom_getter)
    return scope._get_partitioned_variable(_get_default_variable_store(), name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer, trainable=trainable, collections=collections, caching_device=caching_device, partitioner=partitioner, validate_shape=validate_shape, use_resource=use_resource, constraint=constraint, synchronization=synchronization, aggregation=aggregation)

class _pure_variable_scope:
    """A context for the variable_scope, see `variable_scope` for docs."""

    def __init__(self, name_or_scope, reuse=None, initializer=None, regularizer=None, caching_device=None, partitioner=None, custom_getter=None, old_name_scope=None, dtype=dtypes.float32, use_resource=None, constraint=None):
        if False:
            i = 10
            return i + 15
        "Creates a context for the variable_scope, see `variable_scope` for docs.\n\n    Note: this does not create a name scope.\n\n    Args:\n      name_or_scope: `string` or `VariableScope`: the scope to open.\n      reuse: `True` or None, or tf.compat.v1.AUTO_REUSE; if `None`, we inherit\n        the parent scope's reuse flag.\n      initializer: default initializer for variables within this scope.\n      regularizer: default regularizer for variables within this scope.\n      caching_device: default caching device for variables within this scope.\n      partitioner: default partitioner for variables within this scope.\n      custom_getter: default custom getter for variables within this scope.\n      old_name_scope: the original name scope when re-entering a variable scope.\n      dtype: type of the variables within this scope (defaults to `DT_FLOAT`).\n      use_resource: If False, variables in this scope will be regular Variables.\n        If True, experimental ResourceVariables will be creates instead, with\n        well-defined semantics. Defaults to False (will later change to True).\n      constraint: An optional projection function to be applied to the variable\n        after being updated by an `Optimizer` (e.g. used to implement norm\n        constraints or value constraints for layer weights). The function must\n        take as input the unprojected Tensor representing the value of the\n        variable and return the Tensor for the projected value (which must have\n        the same shape). Constraints are not safe to use when doing asynchronous\n        distributed training.\n    "
        self._name_or_scope = name_or_scope
        self._reuse = reuse
        self._initializer = initializer
        self._regularizer = regularizer
        self._caching_device = caching_device
        self._partitioner = partitioner
        self._custom_getter = custom_getter
        self._old_name_scope = old_name_scope
        self._dtype = dtype
        self._use_resource = use_resource
        self._constraint = constraint
        self._var_store = _get_default_variable_store()
        self._var_scope_store = get_variable_scope_store()
        self._last_variable_scope_object = None
        if isinstance(self._name_or_scope, VariableScope):
            self._new_name = self._name_or_scope.name
            name_scope = self._name_or_scope._name_scope
            variable_scope_object = VariableScope(self._name_or_scope.reuse if not self._reuse else self._reuse, name=self._new_name, initializer=self._name_or_scope.initializer, regularizer=self._name_or_scope.regularizer, caching_device=self._name_or_scope.caching_device, partitioner=self._name_or_scope.partitioner, dtype=self._name_or_scope.dtype, custom_getter=self._name_or_scope.custom_getter, name_scope=name_scope, use_resource=self._name_or_scope.use_resource, constraint=self._constraint)
            if self._initializer is not None:
                variable_scope_object.set_initializer(self._initializer)
            if self._regularizer is not None:
                variable_scope_object.set_regularizer(self._regularizer)
            if self._caching_device is not None:
                variable_scope_object.set_caching_device(self._caching_device)
            if self._partitioner is not None:
                variable_scope_object.set_partitioner(self._partitioner)
            if self._custom_getter is not None:
                variable_scope_object.set_custom_getter(_maybe_wrap_custom_getter(self._custom_getter, self._name_or_scope.custom_getter))
            if self._dtype is not None:
                variable_scope_object.set_dtype(self._dtype)
            if self._use_resource is not None:
                variable_scope_object.set_use_resource(self._use_resource)
            self._cached_variable_scope_object = variable_scope_object

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        'Begins the scope block.\n\n    Returns:\n      A VariableScope.\n    Raises:\n      ValueError: when trying to reuse within a create scope, or create within\n        a reuse scope, or if reuse is not `None` or `True`.\n      TypeError: when the types of some arguments are not appropriate.\n    '
        self._old = self._var_scope_store.current_scope
        if isinstance(self._name_or_scope, VariableScope):
            self._var_scope_store.open_variable_scope(self._new_name)
            self._old_subscopes = copy.copy(self._var_scope_store.variable_scopes_count)
            variable_scope_object = self._cached_variable_scope_object
        else:
            self._new_name = self._old.name + '/' + self._name_or_scope if self._old.name else self._name_or_scope
            self._reuse = self._reuse or self._old.reuse
            if self._old_name_scope is None:
                name_scope = self._name_or_scope
            else:
                name_scope = self._old_name_scope
            variable_scope_object = VariableScope(self._reuse, name=self._new_name, initializer=self._old.initializer, regularizer=self._old.regularizer, caching_device=self._old.caching_device, partitioner=self._old.partitioner, dtype=self._old.dtype, use_resource=self._old.use_resource, custom_getter=self._old.custom_getter, name_scope=name_scope, constraint=self._constraint)
            if self._initializer is not None:
                variable_scope_object.set_initializer(self._initializer)
            if self._regularizer is not None:
                variable_scope_object.set_regularizer(self._regularizer)
            if self._caching_device is not None:
                variable_scope_object.set_caching_device(self._caching_device)
            if self._partitioner is not None:
                variable_scope_object.set_partitioner(self._partitioner)
            if self._custom_getter is not None:
                variable_scope_object.set_custom_getter(_maybe_wrap_custom_getter(self._custom_getter, self._old.custom_getter))
            if self._dtype is not None:
                variable_scope_object.set_dtype(self._dtype)
            if self._use_resource is not None:
                variable_scope_object.set_use_resource(self._use_resource)
            self._var_scope_store.open_variable_scope(self._new_name)
        self._var_scope_store.current_scope = variable_scope_object
        self._last_variable_scope_object = variable_scope_object
        return variable_scope_object

    def __exit__(self, type_arg, value_arg, traceback_arg):
        if False:
            while True:
                i = 10
        if self._var_scope_store.current_scope is not self._last_variable_scope_object:
            raise RuntimeError('Improper nesting of variable_scope.')
        if isinstance(self._name_or_scope, VariableScope):
            self._var_scope_store.variable_scopes_count = self._old_subscopes
        else:
            self._var_scope_store.close_variable_subscopes(self._new_name)
        self._var_scope_store.current_scope = self._old

def _maybe_wrap_custom_getter(custom_getter, old_getter):
    if False:
        return 10
    'Wrap a call to a custom_getter to use the old_getter internally.'
    if old_getter is None:
        return custom_getter

    def wrapped_custom_getter(getter, *args, **kwargs):
        if False:
            while True:
                i = 10
        return custom_getter(functools.partial(old_getter, getter), *args, **kwargs)
    return wrapped_custom_getter

def _get_unique_variable_scope(prefix):
    if False:
        print('Hello World!')
    'Get a name with the given prefix unique in the current variable scope.'
    var_scope_store = get_variable_scope_store()
    current_scope = get_variable_scope()
    name = current_scope.name + '/' + prefix if current_scope.name else prefix
    if var_scope_store.variable_scope_count(name) == 0:
        return prefix
    idx = 1
    while var_scope_store.variable_scope_count(name + '_%d' % idx) > 0:
        idx += 1
    return prefix + '_%d' % idx

@tf_export(v1=['variable_scope'])
class variable_scope:
    """A context manager for defining ops that creates variables (layers).

  @compatibility(TF2)
  Although it is a legacy `compat.v1` api,
  `tf.compat.v1.variable_scope` is mostly compatible with eager
  execution and `tf.function` as long as you combine it with the
  `tf.compat.v1.keras.utils.track_tf1_style_variables` decorator (though
  it will behave as if reuse is always set to `AUTO_REUSE`.)

  See the
  [model migration guide](
      https://www.tensorflow.org/guide/migrate/model_mapping)
  for more info on
  migrating code that relies on `variable_scope`-based variable reuse.

  When you use it with eager execution enabled but without
  `tf.compat.v1.keras.utils.track_tf1_style_variables`,
  `tf.compat.v1.variable_scope` will still be able to prefix the names
  of variables created within the scope but it will not enable variable reuse
  or error-raising checks around variable reuse (`get_variable` calls within
  it would always create new variables).

  Once you have switched away from `get_variable`-based variable reuse
  mechanisms, to switch to TF2 APIs you can just use
  `tf.name_scope` to prefix variable names.
  @end_compatibility

  This context manager validates that the (optional) `values` are from the same
  graph, ensures that graph is the default graph, and pushes a name scope and a
  variable scope.

  If `name_or_scope` is not None, it is used as is. If `name_or_scope` is None,
  then `default_name` is used.  In that case, if the same name has been
  previously used in the same scope, it will be made unique by appending `_N`
  to it.

  Variable scope allows you to create new variables and to share already created
  ones while providing checks to not create or share by accident. For details,
  see the [Variable Scope How To](https://tensorflow.org/guide/variables), here
  we present only a few basic examples.

  The Variable Scope works as expected when the Eager Execution is Disabled.

  ```python
  tf.compat.v1.disable_eager_execution()
  ```

  Simple example of how to create a new variable:

  ```python
  with tf.compat.v1.variable_scope("foo"):
      with tf.compat.v1.variable_scope("bar"):
          v = tf.compat.v1.get_variable("v", [1])
          assert v.name == "foo/bar/v:0"
  ```

  Simple example of how to reenter a premade variable scope safely:

  ```python
  with tf.compat.v1.variable_scope("foo") as vs:
    pass

  # Re-enter the variable scope.
  with tf.compat.v1.variable_scope(vs,
                         auxiliary_name_scope=False) as vs1:
    # Restore the original name_scope.
    with tf.name_scope(vs1.original_name_scope):
        v = tf.compat.v1.get_variable("v", [1])
        assert v.name == "foo/v:0"
        c = tf.constant([1], name="c")
        assert c.name == "foo/c:0"
  ```

  Keep in mind that the counters for `default_name` are discarded once the
  parent scope is exited. Therefore when the code re-enters the scope (for
  instance by saving it), all nested default_name counters will be restarted.

  For instance:

  ```python
  with tf.compat.v1.variable_scope("foo") as vs:
    with tf.compat.v1.variable_scope(None, default_name="bar"):
      v = tf.compat.v1.get_variable("a", [1])
      assert v.name == "foo/bar/a:0", v.name
    with tf.compat.v1.variable_scope(None, default_name="bar"):
      v = tf.compat.v1.get_variable("b", [1])
      assert v.name == "foo/bar_1/b:0"

  with tf.compat.v1.variable_scope(vs):
    with tf.compat.v1.variable_scope(None, default_name="bar"):
      v = tf.compat.v1.get_variable("c", [1])
      assert v.name == "foo/bar/c:0"   # Uses bar instead of bar_2!
  ```

  Basic example of sharing a variable AUTO_REUSE:

  ```python
  def foo():
    with tf.compat.v1.variable_scope("foo", reuse=tf.compat.v1.AUTO_REUSE):
      v = tf.compat.v1.get_variable("v", [1])
    return v

  v1 = foo()  # Creates v.
  v2 = foo()  # Gets the same, existing v.
  assert v1 == v2
  ```

  Basic example of sharing a variable with reuse=True:

  ```python
  with tf.compat.v1.variable_scope("foo"):
      v = tf.compat.v1.get_variable("v", [1])
  with tf.compat.v1.variable_scope("foo", reuse=True):
      v1 = tf.compat.v1.get_variable("v", [1])
  assert v1 == v
  ```

  Sharing a variable by capturing a scope and setting reuse:

  ```python
  with tf.compat.v1.variable_scope("foo") as scope:
      v = tf.compat.v1.get_variable("v", [1])
      scope.reuse_variables()
      v1 = tf.compat.v1.get_variable("v", [1])
  assert v1 == v
  ```

  To prevent accidental sharing of variables, we raise an exception when getting
  an existing variable in a non-reusing scope.

  ```python
  with tf.compat.v1.variable_scope("foo"):
      v = tf.compat.v1.get_variable("v", [1])
      v1 = tf.compat.v1.get_variable("v", [1])
      #  Raises ValueError("... v already exists ...").
  ```

  Similarly, we raise an exception when trying to get a variable that does not
  exist in reuse mode.

  ```python
  with tf.compat.v1.variable_scope("foo", reuse=True):
      v = tf.compat.v1.get_variable("v", [1])
      #  Raises ValueError("... v does not exists ...").
  ```

  Note that the `reuse` flag is inherited: if we open a reusing scope, then all
  its sub-scopes become reusing as well.

  A note about name scoping: Setting `reuse` does not impact the naming of other
  ops such as mult. See related discussion on
  [github#6189](https://github.com/tensorflow/tensorflow/issues/6189)

  Note that up to and including version 1.0, it was allowed (though explicitly
  discouraged) to pass False to the reuse argument, yielding undocumented
  behaviour slightly different from None. Starting at 1.1.0 passing None and
  False as reuse has exactly the same effect.

  A note about using variable scopes in multi-threaded environment: Variable
  scopes are thread local, so one thread will not see another thread's current
  scope. Also, when using `default_name`, unique scopes names are also generated
  only on a per thread basis. If the same name was used within a different
  thread, that doesn't prevent a new thread from creating the same scope.
  However, the underlying variable store is shared across threads (within the
  same graph). As such, if another thread tries to create a new variable with
  the same name as a variable created by a previous thread, it will fail unless
  reuse is True.

  Further, each thread starts with an empty variable scope. So if you wish to
  preserve name prefixes from a scope from the main thread, you should capture
  the main thread's scope and re-enter it in each thread. For e.g.

  ```
  main_thread_scope = variable_scope.get_variable_scope()

  # Thread's target function:
  def thread_target_fn(captured_scope):
    with variable_scope.variable_scope(captured_scope):
      # .... regular code for this thread


  thread = threading.Thread(target=thread_target_fn, args=(main_thread_scope,))
  ```
  """

    def __init__(self, name_or_scope, default_name=None, values=None, initializer=None, regularizer=None, caching_device=None, partitioner=None, custom_getter=None, reuse=None, dtype=None, use_resource=None, constraint=None, auxiliary_name_scope=True):
        if False:
            i = 10
            return i + 15
        "Initialize the context manager.\n\n    Args:\n      name_or_scope: `string` or `VariableScope`: the scope to open.\n      default_name: The default name to use if the `name_or_scope` argument is\n        `None`, this name will be uniquified. If name_or_scope is provided it\n        won't be used and therefore it is not required and can be None.\n      values: The list of `Tensor` arguments that are passed to the op function.\n      initializer: default initializer for variables within this scope.\n      regularizer: default regularizer for variables within this scope.\n      caching_device: default caching device for variables within this scope.\n      partitioner: default partitioner for variables within this scope.\n      custom_getter: default custom getter for variables within this scope.\n      reuse: `True`, None, or tf.compat.v1.AUTO_REUSE; if `True`, we go into\n        reuse mode for this scope as well as all sub-scopes; if\n        tf.compat.v1.AUTO_REUSE, we create variables if they do not exist, and\n        return them otherwise; if None, we inherit the parent scope's reuse\n        flag. When eager execution is enabled, new variables are always created\n        unless an EagerVariableStore or template is currently active.\n      dtype: type of variables created in this scope (defaults to the type in\n        the passed scope, or inherited from parent scope).\n      use_resource: If False, all variables will be regular Variables. If True,\n        experimental ResourceVariables with well-defined semantics will be used\n        instead. Defaults to False (will later change to True). When eager\n        execution is enabled this argument is always forced to be True.\n      constraint: An optional projection function to be applied to the variable\n        after being updated by an `Optimizer` (e.g. used to implement norm\n        constraints or value constraints for layer weights). The function must\n        take as input the unprojected Tensor representing the value of the\n        variable and return the Tensor for the projected value (which must have\n        the same shape). Constraints are not safe to use when doing asynchronous\n        distributed training.\n      auxiliary_name_scope: If `True`, we create an auxiliary name scope with\n        the scope. If `False`, we don't create it. Note that the argument is not\n        inherited, and it only takes effect for once when creating. You should\n        only use it for re-entering a premade variable scope.\n\n    Returns:\n      A scope that can be captured and reused.\n\n    Raises:\n      ValueError: when trying to reuse within a create scope, or create within\n        a reuse scope.\n      TypeError: when the types of some arguments are not appropriate.\n    "
        self._name_or_scope = name_or_scope
        self._default_name = default_name
        self._values = values
        self._initializer = initializer
        self._regularizer = regularizer
        self._caching_device = caching_device
        self._partitioner = partitioner
        self._custom_getter = custom_getter
        self._reuse = reuse
        self._dtype = dtype
        self._use_resource = use_resource
        self._constraint = constraint
        if self._default_name is None and self._name_or_scope is None:
            raise TypeError('If default_name is None then name_or_scope is required')
        if self._reuse is False:
            self._reuse = None
        if not (self._reuse is True or self._reuse is None or self._reuse is AUTO_REUSE):
            raise ValueError('The reuse parameter must be True or False or None.')
        if self._values is None:
            self._values = []
        self._in_graph_mode = not context.executing_eagerly()
        if self._in_graph_mode:
            self._graph = ops._get_graph_from_inputs(self._values)
        self._cached_pure_variable_scope = None
        self._current_name_scope = None
        if not isinstance(auxiliary_name_scope, bool):
            raise TypeError('The auxiliary_name_scope must be `True` or `False`, while get {}'.format(auxiliary_name_scope))
        self._auxiliary_name_scope = auxiliary_name_scope

    def __enter__(self):
        if False:
            print('Hello World!')
        if ops.get_default_graph().building_function:
            self._building_function = True
        else:
            self._building_function = False
        if self._in_graph_mode and (not self._building_function):
            self._graph_context_manager = self._graph.as_default()
            self._graph_context_manager.__enter__()
        if self._cached_pure_variable_scope is not None:
            if self._current_name_scope is not None:
                self._current_name_scope.__enter__()
            return self._cached_pure_variable_scope.__enter__()
        try:
            return self._enter_scope_uncached()
        except:
            if self._in_graph_mode and (not self._building_function) and (self._graph_context_manager is not None):
                self._graph_context_manager.__exit__(*sys.exc_info())
            raise

    def _enter_scope_uncached(self):
        if False:
            return 10
        'Enters the context manager when there is no cached scope yet.\n\n    Returns:\n      The entered variable scope.\n\n    Raises:\n      TypeError: A wrong type is passed as `scope` at __init__().\n      ValueError: `reuse` is incorrectly set at __init__().\n    '
        if self._auxiliary_name_scope:
            current_name_scope = None
        else:
            name_scope = ops.get_name_scope()
            if name_scope:
                name_scope += '/'
                current_name_scope = ops.name_scope(name_scope, skip_on_eager=False)
            else:
                current_name_scope = ops.name_scope(name_scope, skip_on_eager=False)
        if self._name_or_scope is not None:
            if not isinstance(self._name_or_scope, (VariableScope, str)):
                raise TypeError('VariableScope: name_or_scope must be a string or VariableScope.')
            if isinstance(self._name_or_scope, str):
                name_scope = self._name_or_scope
            else:
                name_scope = self._name_or_scope.name.split('/')[-1]
            if name_scope or current_name_scope:
                current_name_scope = current_name_scope or ops.name_scope(name_scope, skip_on_eager=False)
                try:
                    current_name_scope_name = current_name_scope.__enter__()
                except:
                    current_name_scope.__exit__(*sys.exc_info())
                    raise
                self._current_name_scope = current_name_scope
                if isinstance(self._name_or_scope, str):
                    old_name_scope = current_name_scope_name
                else:
                    old_name_scope = self._name_or_scope.original_name_scope
                pure_variable_scope = _pure_variable_scope(self._name_or_scope, reuse=self._reuse, initializer=self._initializer, regularizer=self._regularizer, caching_device=self._caching_device, partitioner=self._partitioner, custom_getter=self._custom_getter, old_name_scope=old_name_scope, dtype=self._dtype, use_resource=self._use_resource, constraint=self._constraint)
                try:
                    entered_pure_variable_scope = pure_variable_scope.__enter__()
                except:
                    pure_variable_scope.__exit__(*sys.exc_info())
                    raise
                self._cached_pure_variable_scope = pure_variable_scope
                return entered_pure_variable_scope
            else:
                self._current_name_scope = None
                pure_variable_scope = _pure_variable_scope(self._name_or_scope, reuse=self._reuse, initializer=self._initializer, regularizer=self._regularizer, caching_device=self._caching_device, partitioner=self._partitioner, custom_getter=self._custom_getter, dtype=self._dtype, use_resource=self._use_resource, constraint=self._constraint)
                try:
                    entered_pure_variable_scope = pure_variable_scope.__enter__()
                except:
                    pure_variable_scope.__exit__(*sys.exc_info())
                    raise
                self._cached_pure_variable_scope = pure_variable_scope
                return entered_pure_variable_scope
        else:
            if self._reuse:
                raise ValueError('reuse=True cannot be used without a name_or_scope')
            current_name_scope = current_name_scope or ops.name_scope(self._default_name, skip_on_eager=False)
            try:
                current_name_scope_name = current_name_scope.__enter__()
            except:
                current_name_scope.__exit__(*sys.exc_info())
                raise
            self._current_name_scope = current_name_scope
            unique_default_name = _get_unique_variable_scope(self._default_name)
            pure_variable_scope = _pure_variable_scope(unique_default_name, initializer=self._initializer, regularizer=self._regularizer, caching_device=self._caching_device, partitioner=self._partitioner, custom_getter=self._custom_getter, old_name_scope=current_name_scope_name, dtype=self._dtype, use_resource=self._use_resource, constraint=self._constraint)
            try:
                entered_pure_variable_scope = pure_variable_scope.__enter__()
            except:
                pure_variable_scope.__exit__(*sys.exc_info())
                raise
            self._cached_pure_variable_scope = pure_variable_scope
            return entered_pure_variable_scope

    def __exit__(self, type_arg, value_arg, traceback_arg):
        if False:
            print('Hello World!')
        try:
            self._cached_pure_variable_scope.__exit__(type_arg, value_arg, traceback_arg)
        finally:
            try:
                if self._current_name_scope:
                    self._current_name_scope.__exit__(type_arg, value_arg, traceback_arg)
            finally:
                if self._in_graph_mode and (not self._building_function):
                    self._graph_context_manager.__exit__(type_arg, value_arg, traceback_arg)

@tf_export(v1=['variable_op_scope'])
@tf_contextlib.contextmanager
def variable_op_scope(values, name_or_scope, default_name=None, initializer=None, regularizer=None, caching_device=None, partitioner=None, custom_getter=None, reuse=None, dtype=None, use_resource=None, constraint=None):
    if False:
        return 10
    'Deprecated: context manager for defining an op that creates variables.'
    logging.warn('tf.variable_op_scope(values, name, default_name) is deprecated, use tf.variable_scope(name, default_name, values)')
    with variable_scope(name_or_scope, default_name=default_name, values=values, initializer=initializer, regularizer=regularizer, caching_device=caching_device, partitioner=partitioner, custom_getter=custom_getter, reuse=reuse, dtype=dtype, use_resource=use_resource, constraint=constraint) as scope:
        yield scope

def _call_partitioner(partitioner, shape, dtype):
    if False:
        i = 10
        return i + 15
    'Call partitioner validating its inputs/output.\n\n  Args:\n    partitioner: a function mapping `Tensor` shape and dtype to a list of\n      partitions.\n    shape: shape of the `Tensor` to partition, must have at least two\n      dimensions.\n    dtype: dtype of the elements in the `Tensor`.\n\n  Returns:\n    A list with elements >=1 and exactly one >1. The index of that\n    element corresponds to the partitioning axis.\n  '
    if not shape.is_fully_defined():
        raise ValueError('Shape of a new partitioned variable must be fully defined, but instead was %s.' % (shape,))
    if shape.ndims < 1:
        raise ValueError('A partitioned Variable must have rank at least 1, shape: %s' % shape)
    slicing = partitioner(shape=shape, dtype=dtype)
    if not isinstance(slicing, collections_abc.Sequence):
        raise ValueError('Partitioner must return a sequence, but saw: %s' % slicing)
    if len(slicing) != shape.ndims:
        raise ValueError("Partitioner returned a partition list that does not match the Variable's rank: %s vs. %s" % (slicing, shape))
    if any((p < 1 for p in slicing)):
        raise ValueError('Partitioner returned zero partitions for some axes: %s' % slicing)
    if sum((p > 1 for p in slicing)) > 1:
        raise ValueError('Can only slice a variable along one dimension: shape: %s, partitioning: %s' % (shape, slicing))
    return slicing

def _get_slice_dim_and_num_slices(slicing):
    if False:
        return 10
    'Get slicing dimension and number of slices from the partitioner output.'
    for (slice_dim, num_slices) in enumerate(slicing):
        if num_slices > 1:
            break
    else:
        slice_dim = 0
        num_slices = 1
    return (slice_dim, num_slices)

def _iter_slices(full_shape, num_slices, slice_dim):
    if False:
        for i in range(10):
            print('nop')
    'Slices a given a shape along the specified dimension.'
    num_slices_with_excess = full_shape[slice_dim] % num_slices
    offset = [0] * len(full_shape)
    min_slice_len = full_shape[slice_dim] // num_slices
    for i in range(num_slices):
        shape = full_shape[:]
        shape[slice_dim] = min_slice_len + bool(i < num_slices_with_excess)
        yield (offset[:], shape)
        offset[slice_dim] += shape[slice_dim]

def _make_getter(captured_getter, captured_previous):
    if False:
        i = 10
        return i + 15
    'Gets around capturing loop variables in python being broken.'
    return lambda **kwargs: captured_getter(captured_previous, **kwargs)
_variable_v1 = None

def set_variable_v1(variable_v1):
    if False:
        return 10
    'Sets a reference to variable_v1.VariableV1.'
    global _variable_v1
    _variable_v1 = variable_v1

@tf_export(v1=['variable_creator_scope'])
@tf_contextlib.contextmanager
def variable_creator_scope_v1(variable_creator):
    if False:
        for i in range(10):
            print('nop')
    "Scope which defines a variable creation function to be used by variable().\n\n  variable_creator is expected to be a function with the following signature:\n\n  ```\n    def variable_creator(next_creator, **kwargs)\n  ```\n\n  The creator is supposed to eventually call the next_creator to create a\n  variable if it does want to create a variable and not call Variable or\n  ResourceVariable directly. This helps make creators composable. A creator may\n  choose to create multiple variables, return already existing variables, or\n  simply register that a variable was created and defer to the next creators in\n  line. Creators can also modify the keyword arguments seen by the next\n  creators.\n\n  Custom getters in the variable scope will eventually resolve down to these\n  custom creators when they do create variables.\n\n  The valid keyword arguments in kwds are:\n\n   * initial_value: A `Tensor`, or Python object convertible to a `Tensor`,\n        which is the initial value for the Variable. The initial value must have\n        a shape specified unless `validate_shape` is set to False. Can also be a\n        callable with no argument that returns the initial value when called. In\n        that case, `dtype` must be specified. (Note that initializer functions\n        from init_ops.py must first be bound to a shape before being used here.)\n   * trainable: If `True`, the default, also adds the variable to the graph\n        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as\n        the default list of variables to use by the `Optimizer` classes.\n        `trainable` defaults to `True`, unless `synchronization` is\n        set to `ON_READ`, in which case it defaults to `False`.\n   * collections: List of graph collections keys. The new variable is added to\n        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.\n   * validate_shape: If `False`, allows the variable to be initialized with a\n        value of unknown shape. If `True`, the default, the shape of\n        `initial_value` must be known.\n   * caching_device: Optional device string describing where the Variable\n        should be cached for reading.  Defaults to the Variable's device.\n        If not `None`, caches on another device.  Typical use is to cache\n        on the device where the Ops using the Variable reside, to deduplicate\n        copying through `Switch` and other conditional statements.\n   * name: Optional name for the variable. Defaults to `'Variable'` and gets\n        uniquified automatically.\n   * dtype: If set, initial_value will be converted to the given type.\n        If `None`, either the datatype will be kept (if `initial_value` is\n        a Tensor), or `convert_to_tensor` will decide.\n   * constraint: A constraint function to be applied to the variable after\n        updates by some algorithms.\n   * use_resource: if True, a ResourceVariable is always created.\n   * synchronization: Indicates when a distributed a variable will be\n        aggregated. Accepted values are constants defined in the class\n        `tf.VariableSynchronization`. By default the synchronization is set to\n        `AUTO` and the current `DistributionStrategy` chooses\n        when to synchronize.\n   * aggregation: Indicates how a distributed variable will be aggregated.\n        Accepted values are constants defined in the class\n        `tf.VariableAggregation`.\n\n  This set may grow over time, so it's important the signature of creators is as\n  mentioned above.\n\n  Args:\n    variable_creator: the passed creator\n\n  Yields:\n    A scope in which the creator is active\n  "
    with ops.get_default_graph()._variable_creator_scope(variable_creator):
        yield

@tf_export('variable_creator_scope', v1=[])
@tf_contextlib.contextmanager
def variable_creator_scope(variable_creator):
    if False:
        print('Hello World!')
    "Scope which defines a variable creation function to be used by variable().\n\n  variable_creator is expected to be a function with the following signature:\n\n  ```\n    def variable_creator(next_creator, **kwargs)\n  ```\n\n  The creator is supposed to eventually call the next_creator to create a\n  variable if it does want to create a variable and not call Variable or\n  ResourceVariable directly. This helps make creators composable. A creator may\n  choose to create multiple variables, return already existing variables, or\n  simply register that a variable was created and defer to the next creators in\n  line. Creators can also modify the keyword arguments seen by the next\n  creators.\n\n  Custom getters in the variable scope will eventually resolve down to these\n  custom creators when they do create variables.\n\n  The valid keyword arguments in kwds are:\n\n   * initial_value: A `Tensor`, or Python object convertible to a `Tensor`,\n        which is the initial value for the Variable. The initial value must have\n        a shape specified unless `validate_shape` is set to False. Can also be a\n        callable with no argument that returns the initial value when called. In\n        that case, `dtype` must be specified. (Note that initializer functions\n        from init_ops.py must first be bound to a shape before being used here.)\n   * trainable: If `True`, the default, GradientTapes automatically watch\n        uses of this Variable.\n   * validate_shape: If `False`, allows the variable to be initialized with a\n        value of unknown shape. If `True`, the default, the shape of\n        `initial_value` must be known.\n   * caching_device: Optional device string describing where the Variable\n        should be cached for reading.  Defaults to the Variable's device.\n        If not `None`, caches on another device.  Typical use is to cache\n        on the device where the Ops using the Variable reside, to deduplicate\n        copying through `Switch` and other conditional statements.\n   * name: Optional name for the variable. Defaults to `'Variable'` and gets\n        uniquified automatically.\n      dtype: If set, initial_value will be converted to the given type.\n        If `None`, either the datatype will be kept (if `initial_value` is\n        a Tensor), or `convert_to_tensor` will decide.\n   * constraint: A constraint function to be applied to the variable after\n        updates by some algorithms.\n   * synchronization: Indicates when a distributed a variable will be\n        aggregated. Accepted values are constants defined in the class\n        `tf.VariableSynchronization`. By default the synchronization is set to\n        `AUTO` and the current `DistributionStrategy` chooses\n        when to synchronize.\n   * aggregation: Indicates how a distributed variable will be aggregated.\n        Accepted values are constants defined in the class\n        `tf.VariableAggregation`.\n\n  This set may grow over time, so it's important the signature of creators is as\n  mentioned above.\n\n  Args:\n    variable_creator: the passed creator\n\n  Yields:\n    A scope in which the creator is active\n  "
    with ops.get_default_graph()._variable_creator_scope(variable_creator):
        yield