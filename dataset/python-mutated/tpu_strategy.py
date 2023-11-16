"""TPU Strategy."""
import atexit
import collections
import contextlib
import copy
import functools
import weakref
from absl import logging
import numpy as np
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import tpu_replicated_variable
from tensorflow.python.distribute import tpu_util
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver as tpu_cluster_resolver_lib
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import save_context
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_hardware_feature
from tensorflow.python.tpu import training_loop
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export
from tensorflow.python.util import tf_inspect
_XLA_OP_BY_OP_INPUTS_LIMIT = 200
_EXPERIMENTAL_TPU_BATCH_VARIABLE_INITIALIZATION = False

def enable_batch_variable_initialization():
    if False:
        for i in range(10):
            print('nop')
    'Whether to batch variable initialization in tf.function.'
    return _EXPERIMENTAL_TPU_BATCH_VARIABLE_INITIALIZATION and context.executing_eagerly() and (not save_context.in_save_context())

@contextlib.contextmanager
def maybe_init_scope():
    if False:
        i = 10
        return i + 15
    if ops.executing_eagerly_outside_functions():
        yield
    else:
        with ops.init_scope():
            yield

def validate_run_function(fn):
    if False:
        while True:
            i = 10
    'Validate the function passed into strategy.run.'
    if context.executing_eagerly() and (not isinstance(fn, def_function.Function)) and (not isinstance(fn, function.ConcreteFunction)) and (not (callable(fn) and isinstance(fn.__call__, def_function.Function))):
        raise NotImplementedError('TPUStrategy.run(fn, ...) does not support pure eager execution. please make sure the function passed into `strategy.run` is a `tf.function` or `strategy.run` is called inside a `tf.function` if eager behavior is enabled.')

def _maybe_partial_apply_variables(fn, args, kwargs):
    if False:
        i = 10
        return i + 15
    "Inspects arguments to partially apply any DistributedVariable.\n\n  This avoids an automatic cast of the current variable value to tensor.\n\n  Note that a variable may be captured implicitly with Python scope instead of\n  passing it to run(), but supporting run() keeps behavior consistent\n  with MirroredStrategy.\n\n  Since positional arguments must be applied from left to right, this function\n  does some tricky function inspection to move variable positional arguments\n  into kwargs. As a result of this, we can't support passing Variables as *args,\n  nor as args to functions which combine both explicit positional arguments and\n  *args.\n\n  Args:\n    fn: The function to run, as passed to run().\n    args: Positional arguments to fn, as passed to run().\n    kwargs: Keyword arguments to fn, as passed to run().\n\n  Returns:\n    A tuple of the function (possibly wrapped), args, kwargs (both\n    possibly filtered, with members of args possibly moved to kwargs).\n    If no variables are found, this function is a noop.\n\n  Raises:\n    ValueError: If the function signature makes unsupported use of *args, or if\n      too many arguments are passed.\n  "

    def is_distributed_var(x):
        if False:
            for i in range(10):
                print('nop')
        flat = nest.flatten(x)
        return flat and isinstance(flat[0], values.DistributedVariable)
    var_kwargs = {}
    nonvar_kwargs = {}
    if kwargs:
        var_kwargs = {k: v for (k, v) in kwargs.items() if is_distributed_var(v)}
    if var_kwargs:
        nonvar_kwargs = {k: v for (k, v) in kwargs.items() if not is_distributed_var(v)}
    positional_args = []
    index_of_star_args = None
    for (i, p) in enumerate(tf_inspect.signature(fn).parameters.values()):
        if i == 0 and p.name == 'self':
            continue
        if p.kind == tf_inspect.Parameter.POSITIONAL_OR_KEYWORD:
            positional_args.append(p.name)
        elif p.kind == tf_inspect.Parameter.VAR_POSITIONAL:
            index_of_star_args = i
        elif p.kind == tf_inspect.Parameter.POSITIONAL_ONLY:
            if var_kwargs or any((is_distributed_var(a) for a in args)):
                raise ValueError(f'Mixing Variables and positional-only parameters not supported by TPUStrategy. Received {len(var_kwargs)} DistributedVariables in **kwargs and {sum((is_distributed_var(a) for a in args))} in *args, expected zero for both.')
            return (fn, args, kwargs)
    star_args = []
    have_seen_var_arg = False
    for (i, a) in enumerate(args):
        if is_distributed_var(a):
            if index_of_star_args is not None and i >= index_of_star_args:
                raise ValueError('TPUStrategy.run() cannot handle Variables passed to *args. Either name the function argument, or capture the Variable implicitly.')
            if len(positional_args) <= i:
                raise ValueError('Too many positional arguments passed to call to TPUStrategy.run().')
            var_kwargs[positional_args[i]] = a
            have_seen_var_arg = True
        else:
            if index_of_star_args is not None and i >= index_of_star_args:
                if have_seen_var_arg:
                    raise ValueError('TPUStrategy.run() cannot handle both Variables and a mix of positional args and *args. Either remove the *args, or capture the Variable implicitly.')
                else:
                    star_args.append(a)
                    continue
            if len(positional_args) <= i:
                raise ValueError('Too many positional arguments passed to call to TPUStrategy.run().')
            nonvar_kwargs[positional_args[i]] = a
    if var_kwargs:
        return (functools.partial(fn, **var_kwargs), star_args, nonvar_kwargs)
    return (fn, args, kwargs)

@tf_export.tf_export('distribute.TPUStrategy', v1=[])
class TPUStrategyV2(distribute_lib.Strategy):
    """Synchronous training on TPUs and TPU Pods.

  To construct a TPUStrategy object, you need to run the
  initialization code as below:

  >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
  >>> tf.config.experimental_connect_to_cluster(resolver)
  >>> tf.tpu.experimental.initialize_tpu_system(resolver)
  >>> strategy = tf.distribute.TPUStrategy(resolver)

  While using distribution strategies, the variables created within the
  strategy's scope will be replicated across all the replicas and can be kept in
  sync using all-reduce algorithms.

  To run TF2 programs on TPUs, you can either use `.compile` and
  `.fit` APIs in `tf.keras` with TPUStrategy, or write your own customized
  training loop by calling `strategy.run` directly. Note that
  TPUStrategy doesn't support pure eager execution, so please make sure the
  function passed into `strategy.run` is a `tf.function` or
  `strategy.run` is called inside a `tf.function` if eager
  behavior is enabled. See more details in https://www.tensorflow.org/guide/tpu.

  `distribute_datasets_from_function` and
  `experimental_distribute_dataset` APIs can be used to distribute the dataset
  across the TPU workers when writing your own training loop. If you are using
  `fit` and `compile` methods available in `tf.keras.Model`, then Keras will
  handle the distribution for you.

  An example of writing customized training loop on TPUs:

  >>> with strategy.scope():
  ...   model = tf.keras.Sequential([
  ...     tf.keras.layers.Dense(2, input_shape=(5,)),
  ...   ])
  ...   optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

  >>> def dataset_fn(ctx):
  ...   x = np.random.random((2, 5)).astype(np.float32)
  ...   y = np.random.randint(2, size=(2, 1))
  ...   dataset = tf.data.Dataset.from_tensor_slices((x, y))
  ...   return dataset.repeat().batch(1, drop_remainder=True)
  >>> dist_dataset = strategy.distribute_datasets_from_function(
  ...     dataset_fn)
  >>> iterator = iter(dist_dataset)

  >>> @tf.function()
  ... def train_step(iterator):
  ...
  ...   def step_fn(inputs):
  ...     features, labels = inputs
  ...     with tf.GradientTape() as tape:
  ...       logits = model(features, training=True)
  ...       loss = tf.keras.losses.sparse_categorical_crossentropy(
  ...           labels, logits)
  ...
  ...     grads = tape.gradient(loss, model.trainable_variables)
  ...     optimizer.apply_gradients(zip(grads, model.trainable_variables))
  ...
  ...   strategy.run(step_fn, args=(next(iterator),))

  >>> train_step(iterator)

  For the advanced use cases like model parallelism, you can set
  `experimental_device_assignment` argument when creating TPUStrategy to specify
  number of replicas and number of logical devices. Below is an example to
  initialize TPU system with 2 logical devices and 1 replica.

  >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
  >>> tf.config.experimental_connect_to_cluster(resolver)
  >>> topology = tf.tpu.experimental.initialize_tpu_system(resolver)
  >>> device_assignment = tf.tpu.experimental.DeviceAssignment.build(
  ...     topology,
  ...     computation_shape=[1, 1, 1, 2],
  ...     num_replicas=1)
  >>> strategy = tf.distribute.TPUStrategy(
  ...     resolver, experimental_device_assignment=device_assignment)

  Then you can run a `tf.add` operation only on logical device 0.

  >>> @tf.function()
  ... def step_fn(inputs):
  ...   features, _ = inputs
  ...   output = tf.add(features, features)
  ...
  ...   # Add operation will be executed on logical device 0.
  ...   output = strategy.experimental_assign_to_logical_device(output, 0)
  ...   return output
  >>> dist_dataset = strategy.distribute_datasets_from_function(
  ...     dataset_fn)
  >>> iterator = iter(dist_dataset)
  >>> strategy.run(step_fn, args=(next(iterator),))

  `experimental_spmd_xla_partitioning` enables the experimental XLA SPMD feature
  for model parallelism. This flag can reduce the compilation time and HBM
  requirements. When running in this mode, every input tensor must either be
  partitioned (via `strategy.experimental_split_to_logical_devices`) or fully
  replicated (via `strategy.experimental_replicate_to_logical_devices`) to all
  logical devices. And calling `strategy.experimental_assign_to_logical_device`
  will result in a ValueError in this mode.
  """

    def __init__(self, tpu_cluster_resolver=None, experimental_device_assignment=None, experimental_spmd_xla_partitioning=False):
        if False:
            i = 10
            return i + 15
        'Synchronous training in TPU donuts or Pods.\n\n    Args:\n      tpu_cluster_resolver: A\n        `tf.distribute.cluster_resolver.TPUClusterResolver` instance, which\n        provides information about the TPU cluster. If None, it will assume\n        running on a local TPU worker.\n      experimental_device_assignment: Optional\n        `tf.tpu.experimental.DeviceAssignment` to specify the placement of\n        replicas on the TPU cluster.\n      experimental_spmd_xla_partitioning: If True, enable the SPMD (Single\n        Program Multiple Data) mode in XLA compiler. This flag only affects the\n        performance of XLA compilation and the HBM requirement of the compiled\n        TPU program. Ceveat: if this flag is True, calling\n        `tf.distribute.TPUStrategy.experimental_assign_to_logical_device` will\n        result in a ValueError.\n    '
        super().__init__(TPUExtended(self, tpu_cluster_resolver, device_assignment=experimental_device_assignment, use_spmd_for_xla_partitioning=experimental_spmd_xla_partitioning))
        distribute_lib.distribution_strategy_gauge.get_cell('V2').set('TPUStrategy')
        distribute_lib.distribution_strategy_replica_gauge.get_cell('num_workers').set(self.extended.num_hosts)
        distribute_lib.distribution_strategy_replica_gauge.get_cell('num_replicas_per_worker').set(self.extended.num_replicas_per_host)
        self._enable_packed_variable_in_eager_mode = True

    def run(self, fn, args=(), kwargs=None, options=None):
        if False:
            print('Hello World!')
        "Run the computation defined by `fn` on each TPU replica.\n\n    Executes ops specified by `fn` on each replica. If `args` or `kwargs` have\n    `tf.distribute.DistributedValues`, such as those produced by a\n    `tf.distribute.DistributedDataset` from\n    `tf.distribute.Strategy.experimental_distribute_dataset` or\n    `tf.distribute.Strategy.distribute_datasets_from_function`,\n    when `fn` is executed on a particular replica, it will be executed with the\n    component of `tf.distribute.DistributedValues` that correspond to that\n    replica.\n\n    `fn` may call `tf.distribute.get_replica_context()` to access members such\n    as `all_reduce`.\n\n    All arguments in `args` or `kwargs` should either be nest of tensors or\n    `tf.distribute.DistributedValues` containing tensors or composite tensors.\n\n    Example usage:\n\n    >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')\n    >>> tf.config.experimental_connect_to_cluster(resolver)\n    >>> tf.tpu.experimental.initialize_tpu_system(resolver)\n    >>> strategy = tf.distribute.TPUStrategy(resolver)\n    >>> @tf.function\n    ... def run():\n    ...   def value_fn(value_context):\n    ...     return value_context.num_replicas_in_sync\n    ...   distributed_values = (\n    ...       strategy.experimental_distribute_values_from_function(value_fn))\n    ...   def replica_fn(input):\n    ...     return input * 2\n    ...   return strategy.run(replica_fn, args=(distributed_values,))\n    >>> result = run()\n\n    Args:\n      fn: The function to run. The output must be a `tf.nest` of `Tensor`s.\n      args: (Optional) Positional arguments to `fn`.\n      kwargs: (Optional) Keyword arguments to `fn`.\n      options: (Optional) An instance of `tf.distribute.RunOptions` specifying\n        the options to run `fn`.\n\n    Returns:\n      Merged return value of `fn` across replicas. The structure of the return\n      value is the same as the return value from `fn`. Each element in the\n      structure can either be `tf.distribute.DistributedValues`, `Tensor`\n      objects, or `Tensor`s (for example, if running on a single replica).\n    "
        validate_run_function(fn)
        (fn, args, kwargs) = _maybe_partial_apply_variables(fn, args, kwargs)
        fn = autograph.tf_convert(fn, autograph_ctx.control_status_ctx())
        options = options or distribute_lib.RunOptions()
        return self.extended.tpu_run(fn, args, kwargs, options)

    @property
    def cluster_resolver(self):
        if False:
            print('Hello World!')
        'Returns the cluster resolver associated with this strategy.\n\n    `tf.distribute.TPUStrategy` provides the associated\n    `tf.distribute.cluster_resolver.ClusterResolver`. If the user provides one\n    in `__init__`, that instance is returned; if the user does not, a default\n    `tf.distribute.cluster_resolver.TPUClusterResolver` is provided.\n    '
        return self.extended._tpu_cluster_resolver

    def experimental_assign_to_logical_device(self, tensor, logical_device_id):
        if False:
            return 10
        "Adds annotation that `tensor` will be assigned to a logical device.\n\n    This adds an annotation to `tensor` specifying that operations on\n    `tensor` will be invoked on logical core device id `logical_device_id`.\n    When model parallelism is used, the default behavior is that all ops\n    are placed on zero-th logical device.\n\n    ```python\n\n    # Initializing TPU system with 2 logical devices and 4 replicas.\n    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')\n    tf.config.experimental_connect_to_cluster(resolver)\n    topology = tf.tpu.experimental.initialize_tpu_system(resolver)\n    device_assignment = tf.tpu.experimental.DeviceAssignment.build(\n        topology,\n        computation_shape=[1, 1, 1, 2],\n        num_replicas=4)\n    strategy = tf.distribute.TPUStrategy(\n        resolver, experimental_device_assignment=device_assignment)\n    iterator = iter(inputs)\n\n    @tf.function()\n    def step_fn(inputs):\n      output = tf.add(inputs, inputs)\n\n      # Add operation will be executed on logical device 0.\n      output = strategy.experimental_assign_to_logical_device(output, 0)\n      return output\n\n    strategy.run(step_fn, args=(next(iterator),))\n    ```\n\n    Args:\n      tensor: Input tensor to annotate.\n      logical_device_id: Id of the logical core to which the tensor will be\n        assigned.\n\n    Raises:\n      ValueError: The logical device id presented is not consistent with total\n      number of partitions specified by the device assignment or the TPUStrategy\n      is constructed with `experimental_spmd_xla_partitioning=True`.\n\n    Returns:\n      Annotated tensor with identical value as `tensor`.\n    "
        if self.extended._use_spmd_for_xla_partitioning:
            raise ValueError('Cannot assign a tensor to a logical device in SPMD mode. To disable SPMD, Please construct the TPUStrategy with `experimental_spmd_xla_partitioning=False`')
        num_logical_devices_per_replica = self.extended._tpu_devices.shape[1]
        if logical_device_id < 0 or logical_device_id >= num_logical_devices_per_replica:
            raise ValueError('`logical_core_id` to assign must be lower then total number of logical devices per replica. Received logical device id {} but there are only total of {} logical devices in replica.'.format(logical_device_id, num_logical_devices_per_replica))
        return xla_sharding.assign_device(tensor, logical_device_id, use_sharding_op=True)

    def experimental_split_to_logical_devices(self, tensor, partition_dimensions):
        if False:
            while True:
                i = 10
        "Adds annotation that `tensor` will be split across logical devices.\n\n    This adds an annotation to tensor `tensor` specifying that operations on\n    `tensor` will be split among multiple logical devices. Tensor `tensor` will\n    be split across dimensions specified by `partition_dimensions`.\n    The dimensions of `tensor` must be divisible by corresponding value in\n    `partition_dimensions`.\n\n    For example, for system with 8 logical devices, if `tensor` is an image\n    tensor with shape (batch_size, width, height, channel) and\n    `partition_dimensions` is [1, 2, 4, 1], then `tensor` will be split\n    2 in width dimension and 4 way in height dimension and the split\n    tensor values will be fed into 8 logical devices.\n\n    ```python\n    # Initializing TPU system with 8 logical devices and 1 replica.\n    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')\n    tf.config.experimental_connect_to_cluster(resolver)\n    topology = tf.tpu.experimental.initialize_tpu_system(resolver)\n    device_assignment = tf.tpu.experimental.DeviceAssignment.build(\n        topology,\n        computation_shape=[1, 2, 2, 2],\n        num_replicas=1)\n    # Construct the TPUStrategy. Since we are going to split the image across\n    # logical devices, here we set `experimental_spmd_xla_partitioning=True`\n    # so that the partitioning can be compiled in SPMD mode, which usually\n    # results in faster compilation and smaller HBM requirement if the size of\n    # input and activation tensors are much bigger than that of the model\n    # parameters. Note that this flag is suggested but not a hard requirement\n    # for `experimental_split_to_logical_devices`.\n    strategy = tf.distribute.TPUStrategy(\n        resolver, experimental_device_assignment=device_assignment,\n        experimental_spmd_xla_partitioning=True)\n\n    iterator = iter(inputs)\n\n    @tf.function()\n    def step_fn(inputs):\n      inputs = strategy.experimental_split_to_logical_devices(\n        inputs, [1, 2, 4, 1])\n\n      # model() function will be executed on 8 logical devices with `inputs`\n      # split 2 * 4  ways.\n      output = model(inputs)\n      return output\n\n    strategy.run(step_fn, args=(next(iterator),))\n    ```\n    Args:\n      tensor: Input tensor to annotate.\n      partition_dimensions: An unnested list of integers with the size equal to\n        rank of `tensor` specifying how `tensor` will be partitioned. The\n        product of all elements in `partition_dimensions` must be equal to the\n        total number of logical devices per replica.\n\n    Raises:\n      ValueError: 1) If the size of partition_dimensions does not equal to rank\n        of `tensor` or 2) if product of elements of `partition_dimensions` does\n        not match the number of logical devices per replica defined by the\n        implementing DistributionStrategy's device specification or\n        3) if a known size of `tensor` is not divisible by corresponding\n        value in `partition_dimensions`.\n\n    Returns:\n      Annotated tensor with identical value as `tensor`.\n    "
        num_logical_devices_per_replica = self.extended._tpu_devices.shape[1]
        num_partition_splits = np.prod(partition_dimensions)
        input_shape = tensor.shape
        tensor_rank = len(input_shape)
        if tensor_rank != len(partition_dimensions):
            raise ValueError('Length of `partition_dimensions` must equal to the rank of `tensor.shape` ({}). Received len(partition_dimensions)={}.'.format(tensor_rank, len(partition_dimensions)))
        for (dim_index, dim_size) in enumerate(input_shape):
            if dim_size is None:
                continue
            split_size = partition_dimensions[dim_index]
            if dim_size % split_size != 0:
                raise ValueError('Tensor shape at `partition_dimensions[{}]` must be divisible by corresponding value specified by `partition_dimensions` ({}). Received: {}.'.format(dim_index, split_size, dim_size))
        if num_partition_splits != num_logical_devices_per_replica:
            raise ValueError('The product of `partition_dimensions` should be the same as the number of logical devices (={}). Received `partition_dimensions`={},and their product is {}.'.format(num_logical_devices_per_replica, partition_dimensions, num_partition_splits))
        tile_assignment = np.arange(num_partition_splits).reshape(partition_dimensions)
        return xla_sharding.tile(tensor, tile_assignment, use_sharding_op=True)

    def experimental_replicate_to_logical_devices(self, tensor):
        if False:
            while True:
                i = 10
        "Adds annotation that `tensor` will be replicated to all logical devices.\n\n    This adds an annotation to tensor `tensor` specifying that operations on\n    `tensor` will be invoked on all logical devices.\n\n    ```python\n    # Initializing TPU system with 2 logical devices and 4 replicas.\n    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')\n    tf.config.experimental_connect_to_cluster(resolver)\n    topology = tf.tpu.experimental.initialize_tpu_system(resolver)\n    device_assignment = tf.tpu.experimental.DeviceAssignment.build(\n        topology,\n        computation_shape=[1, 1, 1, 2],\n        num_replicas=4)\n    strategy = tf.distribute.TPUStrategy(\n        resolver, experimental_device_assignment=device_assignment)\n\n    iterator = iter(inputs)\n\n    @tf.function()\n    def step_fn(inputs):\n      images, labels = inputs\n      images = strategy.experimental_split_to_logical_devices(\n        inputs, [1, 2, 4, 1])\n\n      # model() function will be executed on 8 logical devices with `inputs`\n      # split 2 * 4  ways.\n      output = model(inputs)\n\n      # For loss calculation, all logical devices share the same logits\n      # and labels.\n      labels = strategy.experimental_replicate_to_logical_devices(labels)\n      output = strategy.experimental_replicate_to_logical_devices(output)\n      loss = loss_fn(labels, output)\n\n      return loss\n\n    strategy.run(step_fn, args=(next(iterator),))\n    ```\n    Args:\n      tensor: Input tensor to annotate.\n\n    Returns:\n      Annotated tensor with identical value as `tensor`.\n    "
        return xla_sharding.replicate(tensor, use_sharding_op=True)

@tf_export.tf_export('distribute.experimental.TPUStrategy', v1=[])
@deprecation.deprecated_endpoints('distribute.experimental.TPUStrategy')
class TPUStrategy(distribute_lib.Strategy):
    """Synchronous training on TPUs and TPU Pods.

  To construct a TPUStrategy object, you need to run the
  initialization code as below:

  >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
  >>> tf.config.experimental_connect_to_cluster(resolver)
  >>> tf.tpu.experimental.initialize_tpu_system(resolver)
  >>> strategy = tf.distribute.experimental.TPUStrategy(resolver)

  While using distribution strategies, the variables created within the
  strategy's scope will be replicated across all the replicas and can be kept in
  sync using all-reduce algorithms.

  To run TF2 programs on TPUs, you can either use `.compile` and
  `.fit` APIs in `tf.keras` with TPUStrategy, or write your own customized
  training loop by calling `strategy.run` directly. Note that
  TPUStrategy doesn't support pure eager execution, so please make sure the
  function passed into `strategy.run` is a `tf.function` or
  `strategy.run` is called inside a `tf.function` if eager
  behavior is enabled.
  """

    def __init__(self, tpu_cluster_resolver=None, device_assignment=None):
        if False:
            print('Hello World!')
        'Synchronous training in TPU donuts or Pods.\n\n    Args:\n      tpu_cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,\n        which provides information about the TPU cluster.\n      device_assignment: Optional `tf.tpu.experimental.DeviceAssignment` to\n        specify the placement of replicas on the TPU cluster.\n    '
        logging.warning('`tf.distribute.experimental.TPUStrategy` is deprecated, please use the non-experimental symbol `tf.distribute.TPUStrategy` instead.')
        super().__init__(TPUExtended(self, tpu_cluster_resolver, device_assignment=device_assignment))
        distribute_lib.distribution_strategy_gauge.get_cell('V2').set('TPUStrategy')
        distribute_lib.distribution_strategy_replica_gauge.get_cell('num_workers').set(self.extended.num_hosts)
        distribute_lib.distribution_strategy_replica_gauge.get_cell('num_replicas_per_worker').set(self.extended.num_replicas_per_host)
        self._enable_packed_variable_in_eager_mode = True

    def run(self, fn, args=(), kwargs=None, options=None):
        if False:
            print('Hello World!')
        'See base class.'
        validate_run_function(fn)
        (fn, args, kwargs) = _maybe_partial_apply_variables(fn, args, kwargs)
        fn = autograph.tf_convert(fn, autograph_ctx.control_status_ctx())
        options = options or distribute_lib.RunOptions()
        return self.extended.tpu_run(fn, args, kwargs, options)

    @property
    def cluster_resolver(self):
        if False:
            return 10
        'Returns the cluster resolver associated with this strategy.\n\n    `tf.distribute.experimental.TPUStrategy` provides the\n    associated `tf.distribute.cluster_resolver.ClusterResolver`. If the user\n    provides one in `__init__`, that instance is returned; if the user does\n    not, a default\n    `tf.distribute.cluster_resolver.TPUClusterResolver` is provided.\n    '
        return self.extended._tpu_cluster_resolver

@tf_export.tf_export(v1=['distribute.experimental.TPUStrategy'])
class TPUStrategyV1(distribute_lib.StrategyV1):
    """TPU distribution strategy implementation."""

    def __init__(self, tpu_cluster_resolver=None, steps_per_run=None, device_assignment=None):
        if False:
            i = 10
            return i + 15
        'Initializes the TPUStrategy object.\n\n    Args:\n      tpu_cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,\n          which provides information about the TPU cluster.\n      steps_per_run: Number of steps to run on device before returning to the\n          host. Note that this can have side-effects on performance, hooks,\n          metrics, summaries etc.\n          This parameter is only used when Distribution Strategy is used with\n          estimator or keras.\n      device_assignment: Optional `tf.tpu.experimental.DeviceAssignment` to\n          specify the placement of replicas on the TPU cluster. Currently only\n          supports the usecase of using a single core within a TPU cluster.\n    '
        super().__init__(TPUExtended(self, tpu_cluster_resolver, steps_per_run, device_assignment))
        distribute_lib.distribution_strategy_gauge.get_cell('V1').set('TPUStrategy')
        distribute_lib.distribution_strategy_replica_gauge.get_cell('num_workers').set(self.extended.num_hosts)
        distribute_lib.distribution_strategy_replica_gauge.get_cell('num_replicas_per_worker').set(self.extended.num_replicas_per_host)
        self._enable_packed_variable_in_eager_mode = True

    @property
    def steps_per_run(self):
        if False:
            for i in range(10):
                print('nop')
        'DEPRECATED: use .extended.steps_per_run instead.'
        return self._extended.steps_per_run

    def run(self, fn, args=(), kwargs=None, options=None):
        if False:
            while True:
                i = 10
        'Run `fn` on each replica, with the given arguments.\n\n    Executes ops specified by `fn` on each replica. If `args` or `kwargs` have\n    "per-replica" values, such as those produced by a "distributed `Dataset`",\n    when `fn` is executed on a particular replica, it will be executed with the\n    component of those "per-replica" values that correspond to that replica.\n\n    `fn` may call `tf.distribute.get_replica_context()` to access members such\n    as `all_reduce`.\n\n    All arguments in `args` or `kwargs` should either be nest of tensors or\n    per-replica objects containing tensors or composite tensors.\n\n    Users can pass strategy specific options to `options` argument. An example\n    to enable bucketizing dynamic shapes in `TPUStrategy.run`\n    is:\n\n    >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=\'\')\n    >>> tf.config.experimental_connect_to_cluster(resolver)\n    >>> tf.tpu.experimental.initialize_tpu_system(resolver)\n    >>> strategy = tf.distribute.experimental.TPUStrategy(resolver)\n\n    >>> options = tf.distribute.RunOptions(\n    ...     experimental_bucketizing_dynamic_shape=True)\n\n    >>> dataset = tf.data.Dataset.range(\n    ...    strategy.num_replicas_in_sync, output_type=dtypes.float32).batch(\n    ...        strategy.num_replicas_in_sync, drop_remainder=True)\n    >>> input_iterator = iter(strategy.experimental_distribute_dataset(dataset))\n\n    >>> @tf.function()\n    ... def step_fn(inputs):\n    ...  output = tf.reduce_sum(inputs)\n    ...  return output\n\n    >>> strategy.run(step_fn, args=(next(input_iterator),), options=options)\n\n    Args:\n      fn: The function to run. The output must be a `tf.nest` of `Tensor`s.\n      args: (Optional) Positional arguments to `fn`.\n      kwargs: (Optional) Keyword arguments to `fn`.\n      options: (Optional) An instance of `tf.distribute.RunOptions` specifying\n        the options to run `fn`.\n\n    Returns:\n      Merged return value of `fn` across replicas. The structure of the return\n      value is the same as the return value from `fn`. Each element in the\n      structure can either be "per-replica" `Tensor` objects or `Tensor`s\n      (for example, if running on a single replica).\n    '
        validate_run_function(fn)
        (fn, args, kwargs) = _maybe_partial_apply_variables(fn, args, kwargs)
        fn = autograph.tf_convert(fn, autograph_ctx.control_status_ctx())
        options = options or distribute_lib.RunOptions()
        return self.extended.tpu_run(fn, args, kwargs, options)

class TPUExtended(distribute_lib.StrategyExtendedV1):
    """Implementation of TPUStrategy."""

    def __init__(self, container_strategy, tpu_cluster_resolver=None, steps_per_run=None, device_assignment=None, use_spmd_for_xla_partitioning=False):
        if False:
            while True:
                i = 10
        super().__init__(container_strategy)
        if tpu_cluster_resolver is None:
            tpu_cluster_resolver = tpu_cluster_resolver_lib.TPUClusterResolver('')
        if steps_per_run is None:
            steps_per_run = 1
        self._tpu_function_cache = weakref.WeakKeyDictionary()
        self._tpu_cluster_resolver = tpu_cluster_resolver
        self._tpu_metadata = self._tpu_cluster_resolver.get_tpu_system_metadata()
        self._device_assignment = device_assignment
        tpu_devices_flat = [d.name for d in self._tpu_metadata.devices if 'device:TPU:' in d.name]
        if device_assignment is None:
            self._tpu_devices = np.array([[d] for d in tpu_devices_flat], dtype=object)
        else:
            job_name = device_spec.DeviceSpecV2.from_string(tpu_devices_flat[0]).job
            tpu_devices = []
            for replica_id in range(device_assignment.num_replicas):
                replica_devices = []
                for logical_core in range(device_assignment.num_cores_per_replica):
                    replica_devices.append(device_util.canonicalize(device_assignment.tpu_device(replica=replica_id, logical_core=logical_core, job=job_name)))
                tpu_devices.append(replica_devices)
            self._tpu_devices = np.array(tpu_devices, dtype=object)
        self._host_device = device_util.get_host_for_device(self._tpu_devices[0][0])
        self._device_input_worker_devices = collections.OrderedDict()
        self._host_input_worker_devices = collections.OrderedDict()
        for tpu_device in self._tpu_devices[:, 0]:
            host_device = device_util.get_host_for_device(tpu_device)
            self._device_input_worker_devices.setdefault(host_device, [])
            self._device_input_worker_devices[host_device].append(tpu_device)
            self._host_input_worker_devices.setdefault(host_device, [])
            self._host_input_worker_devices[host_device].append(host_device)
        self.steps_per_run = steps_per_run
        self._require_static_shapes = True
        self.experimental_enable_get_next_as_optional = True
        self._logical_device_stack = [0]
        if context.executing_eagerly():
            atexit.register(context.async_wait)
        self._use_var_policy = not use_spmd_for_xla_partitioning
        self._use_spmd_for_xla_partitioning = use_spmd_for_xla_partitioning
        self._using_custom_device = False
        devices = self._tpu_devices[:, self._logical_device_stack[-1]]
        for d in devices:
            if context.is_custom_device(d):
                self._using_custom_device = True
                break
        self._enable_data_reorder = False

    def _get_replica_order(self):
        if False:
            for i in range(10):
                print('nop')
        "Get the replica order based on the tpu device order.\n\n    For example, if the tpu_devices are:\n    '/job:worker/replica:0/task:0/device:TPU:0',\n    '/job:worker/replica:0/task:0/device:TPU:2',\n    '/job:worker/replica:0/task:1/device:TPU:0',\n    '/job:worker/replica:0/task:1/device:TPU:2',\n    '/job:worker/replica:0/task:1/device:TPU:6',\n    '/job:worker/replica:0/task:1/device:TPU:4',\n    '/job:worker/replica:0/task:0/device:TPU:6',\n    '/job:worker/replica:0/task:0/device:TPU:4',\n\n    the returned replica order will be:\n    [0, 1, 7, 6, 2, 3, 5, 4]\n\n    This replica order will be used to reorder the data returned by the\n    iterators,\n    so that they can be placed on the same node as their computation graphs.\n\n    Returns:\n      A list containing the order ids of corresponding TPU devices.\n    "
        if not self._enable_data_reorder:
            return None
        tpu_devices = self._tpu_devices[:, 0]
        devices_with_ids = []
        for (i, tpu_device) in enumerate(tpu_devices):
            spec = tf_device.DeviceSpec.from_string(tpu_device)
            devices_with_ids.append(((spec.job, spec.replica, spec.device_type, spec.task, spec.device_index), i))
        return [i for (_, i) in sorted(devices_with_ids)]

    def _validate_colocate_with_variable(self, colocate_with_variable):
        if False:
            i = 10
            return i + 15
        distribute_utils.validate_colocate(colocate_with_variable, self)

    def _make_dataset_iterator(self, dataset):
        if False:
            for i in range(10):
                print('nop')
        'Make iterators for each of the TPU hosts.'
        input_workers = input_lib.InputWorkers(tuple(self._device_input_worker_devices.items()))
        return input_lib_v1.DatasetIterator(dataset, input_workers, self._container_strategy(), num_replicas_in_sync=self._num_replicas_in_sync)

    def _make_input_fn_iterator(self, input_fn, replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
        if False:
            while True:
                i = 10
        input_contexts = []
        input_workers = input_lib.InputWorkers(tuple(self._device_input_worker_devices.items()))
        num_workers = input_workers.num_workers
        for i in range(num_workers):
            input_contexts.append(distribute_lib.InputContext(num_input_pipelines=num_workers, input_pipeline_id=i, num_replicas_in_sync=self._num_replicas_in_sync))
        return input_lib_v1.InputFunctionIterator(input_fn, input_workers, input_contexts, self._container_strategy())

    def _experimental_make_numpy_dataset(self, numpy_input, session):
        if False:
            return 10
        return numpy_dataset.one_host_numpy_dataset(numpy_input, numpy_dataset.SingleDevice(self._host_device), session)

    def _get_input_workers(self, options):
        if False:
            for i in range(10):
                print('nop')
        if not options or options.experimental_fetch_to_device:
            return input_lib.InputWorkers(tuple(self._device_input_worker_devices.items()))
        else:
            return input_lib.InputWorkers(tuple(self._host_input_worker_devices.items()))

    def _check_spec(self, element_spec):
        if False:
            i = 10
            return i + 15
        if isinstance(element_spec, values.PerReplicaSpec):
            element_spec = element_spec._component_specs
        specs = nest.flatten_with_joined_string_paths(element_spec)
        for (path, spec) in specs:
            if isinstance(spec, (sparse_tensor.SparseTensorSpec, ragged_tensor.RaggedTensorSpec)):
                raise ValueError('Found tensor {} with spec {}. TPUStrategy does not support distributed datasets with device prefetch when using sparse or ragged tensors. If you intend to use sparse or ragged tensors, please pass a tf.distribute.InputOptions object with experimental_fetch_to_device set to False to your dataset distribution function.'.format(path, type(spec)))

    def _experimental_distribute_dataset(self, dataset, options):
        if False:
            while True:
                i = 10
        if options and options.experimental_replication_mode == distribute_lib.InputReplicationMode.PER_REPLICA:
            raise NotImplementedError('InputReplicationMode.PER_REPLICA is only supported in `experimental_distribute_datasets_from_function`.')
        if options is None or options.experimental_fetch_to_device:
            self._check_spec(dataset.element_spec)
        return input_util.get_distributed_dataset(dataset, self._get_input_workers(options), self._container_strategy(), num_replicas_in_sync=self._num_replicas_in_sync, options=options, replica_order=self._get_replica_order())

    def _distribute_datasets_from_function(self, dataset_fn, options):
        if False:
            i = 10
            return i + 15
        if options and options.experimental_replication_mode == distribute_lib.InputReplicationMode.PER_REPLICA:
            raise NotImplementedError('InputReplicationMode.PER_REPLICA is only supported in  `experimental_distribute_datasets_from_function` of tf.distribute.MirroredStrategy')
        input_workers = self._get_input_workers(options)
        input_contexts = []
        num_workers = input_workers.num_workers
        for i in range(num_workers):
            input_contexts.append(distribute_lib.InputContext(num_input_pipelines=num_workers, input_pipeline_id=i, num_replicas_in_sync=self._num_replicas_in_sync))
        distributed_dataset = input_util.get_distributed_datasets_from_function(dataset_fn, input_workers, input_contexts, self._container_strategy(), options=options, replica_order=self._get_replica_order())
        if options is None or options.experimental_fetch_to_device:
            self._check_spec(distributed_dataset.element_spec)
        return distributed_dataset

    def _experimental_distribute_values_from_function(self, value_fn):
        if False:
            while True:
                i = 10
        per_replica_values = []
        for replica_id in range(self._num_replicas_in_sync):
            per_replica_values.append(value_fn(distribute_lib.ValueContext(replica_id, self._num_replicas_in_sync)))
        return distribute_utils.regroup(per_replica_values, always_wrap=True)

    def _experimental_run_steps_on_iterator(self, fn, multi_worker_iterator, iterations, initial_loop_values=None):
        if False:
            return 10
        if initial_loop_values is None:
            initial_loop_values = {}
        initial_loop_values = nest.flatten(initial_loop_values)
        ctx = input_lib.MultiStepContext()

        def run_fn(inputs):
            if False:
                i = 10
                return i + 15
            'Single step on the TPU device.'
            fn_result = fn(ctx, inputs)
            flat_last_step_outputs = nest.flatten(ctx.last_step_outputs)
            if flat_last_step_outputs:
                with ops.control_dependencies([fn_result]):
                    return [array_ops.identity(f) for f in flat_last_step_outputs]
            else:
                return fn_result
        self._outer_control_flow_context = ops.get_default_graph()._get_control_flow_context()

        def rewrite_fn(*args):
            if False:
                i = 10
                return i + 15
            'The rewritten step fn running on TPU.'
            del args
            per_replica_inputs = multi_worker_iterator.get_next()
            replicate_inputs = []
            for replica_id in range(self._num_replicas_in_sync):
                select_replica = lambda x: distribute_utils.select_replica(replica_id, x)
                replicate_inputs.append((nest.map_structure(select_replica, per_replica_inputs),))
            replicate_outputs = tpu.replicate(run_fn, replicate_inputs, device_assignment=self._device_assignment, xla_options=tpu.XLAOptions(use_spmd_for_xla_partitioning=self._use_spmd_for_xla_partitioning))
            if isinstance(replicate_outputs[0], list):
                replicate_outputs = nest.flatten(replicate_outputs)
            return replicate_outputs
        assert isinstance(initial_loop_values, list)
        initial_loop_values = initial_loop_values * self._num_replicas_in_sync
        with ops.device(self._host_device):
            if self.steps_per_run == 1:
                replicate_outputs = rewrite_fn()
            else:
                replicate_outputs = training_loop.repeat(iterations, rewrite_fn, initial_loop_values)
        del self._outer_control_flow_context
        ctx.run_op = control_flow_ops.group(replicate_outputs)
        if isinstance(replicate_outputs, list):
            last_step_tensor_outputs = [x for x in replicate_outputs if not isinstance(x, ops.Operation)]
            output_num = len(last_step_tensor_outputs) // self._num_replicas_in_sync
            last_step_tensor_outputs = [last_step_tensor_outputs[i::output_num] for i in range(output_num)]
        else:
            last_step_tensor_outputs = []
        _set_last_step_outputs(ctx, last_step_tensor_outputs)
        return ctx

    def _call_for_each_replica(self, fn, args, kwargs):
        if False:
            print('Hello World!')
        with _TPUReplicaContext(self._container_strategy()):
            return fn(*args, **kwargs)

    @contextlib.contextmanager
    def experimental_logical_device(self, logical_device_id):
        if False:
            return 10
        'Places variables and ops on the specified logical device.'
        num_logical_devices_per_replica = self._tpu_devices.shape[1]
        if logical_device_id >= num_logical_devices_per_replica:
            raise ValueError('`logical_device_id` not in range (was {}, but there are only {} logical devices per replica).'.format(logical_device_id, num_logical_devices_per_replica))
        self._logical_device_stack.append(logical_device_id)
        try:
            if tpu_util.enclosing_tpu_context() is None:
                yield
            else:
                with ops.device(tpu.core(logical_device_id)):
                    yield
        finally:
            self._logical_device_stack.pop()

    def _experimental_initialize_system(self):
        if False:
            while True:
                i = 10
        'Experimental method added to be used by Estimator.\n\n    This is a private method only to be used by Estimator. Other frameworks\n    should directly be calling `tf.tpu.experimental.initialize_tpu_system`\n    '
        tpu_cluster_resolver_lib.initialize_tpu_system(self._tpu_cluster_resolver)

    def _create_variable(self, next_creator, **kwargs):
        if False:
            while True:
                i = 10
        'Create a TPUMirroredVariable. See `DistributionStrategy.scope`.'
        if kwargs.pop('skip_mirrored_creator', False):
            return next_creator(**kwargs)
        custom_tpu_variable_creator = kwargs.pop('custom_tpu_variable_creator', None)
        if custom_tpu_variable_creator is not None:
            return custom_tpu_variable_creator(next_creator, **kwargs)
        colocate_with = kwargs.pop('colocate_with', None)
        if colocate_with is None:
            devices = self._tpu_devices[:, self._logical_device_stack[-1]]
        elif isinstance(colocate_with, numpy_dataset.SingleDevice):
            with ops.device(colocate_with.device):
                return next_creator(**kwargs)
        else:
            devices = colocate_with._devices
        (num_replicas, num_cores_per_replica) = self._tpu_devices.shape

        def _create_mirrored_tpu_variables(**kwargs):
            if False:
                print('Hello World!')
            'Returns a list of `tf.Variable`s.\n\n      The list contains `number_replicas` `tf.Variable`s and can be used to\n      initialize a `TPUMirroredVariable`.\n\n      Args:\n        **kwargs: the keyword arguments for creating a variable\n      '
            initial_value = None
            value_list = []
            for (i, d) in enumerate(devices):
                with ops.device(d):
                    if i == 0:
                        initial_value = kwargs['initial_value']
                        with maybe_init_scope():
                            initial_value = initial_value() if callable(initial_value) else initial_value
                    if i > 0:
                        var0name = value_list[0].name.split(':')[0]
                        kwargs['name'] = '%s/replica_%d/' % (var0name, i)
                    kwargs['initial_value'] = initial_value
                    with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
                        v = next_creator(**kwargs)
                    assert not isinstance(v, tpu_values.TPUMirroredVariable)
                    value_list.append(v)
            return value_list

        def _create_mirrored_tpu_replicated_variables(**kwargs):
            if False:
                while True:
                    i = 10
            'Returns a list of `TPUReplicatedVariable`s.\n\n      The list consists of `num_replicas` `TPUReplicatedVariable`s and can be\n      used to initialize a `TPUMirroredVariable`. Each `TPUReplicatedVariable`\n      contains a list of `tf.Variable`s which are replicated to\n      `num_cores_per_replica` logical cores to enable XLA SPMD compilation.\n\n      Args:\n        **kwargs: the keyword arguments for creating a variable\n      '
            initial_value = kwargs['initial_value']
            with maybe_init_scope():
                initial_value = initial_value() if callable(initial_value) else initial_value
            mirrored_replicated_var_list = []
            for replica_id in range(num_replicas):
                replicated_var_list = []
                for logic_core_id in range(num_cores_per_replica):
                    with ops.device(self._tpu_devices[replica_id][logic_core_id]):
                        kwargs['initial_value'] = initial_value
                        v = next_creator(**kwargs)
                    replicated_var_list.append(v)
                replica_name = '{}/r:{}'.format(kwargs['name'], replica_id)
                tpu_replicated_var = tpu_replicated_variable.TPUReplicatedVariable(variables=replicated_var_list, name=replica_name)
                mirrored_replicated_var_list.append(tpu_replicated_var)
            return mirrored_replicated_var_list

        def uninitialized_variable_creator(**kwargs):
            if False:
                return 10
            uninitialized_variable = tpu_util.TPUUninitializedVariable(**kwargs)
            self.lazy_variable_tracker.add_uninitialized_var(uninitialized_variable)
            setattr(uninitialized_variable, '_lazy_scope', self.lazy_variable_tracker)
            return uninitialized_variable

        def _create_uninitialized_mirrored_tpu_variables(**kwargs):
            if False:
                for i in range(10):
                    print('nop')
            'Returns a list of `tf.Variable`s.\n\n      The list contains `number_replicas` `tf.Variable`s and can be used to\n      initialize a `TPUMirroredVariable`.\n\n      Args:\n        **kwargs: the keyword arguments for creating a variable\n      '
            if kwargs.get('initial_value', None) is None:
                return _create_mirrored_tpu_variables(**kwargs)
            value_list = []
            initial_value = None
            for (i, d) in enumerate(devices):
                with ops.device(d):
                    if i == 0:
                        initial_value = kwargs.get('initial_value', None)
                        with maybe_init_scope():
                            if initial_value is not None:
                                if callable(initial_value):
                                    initial_value = initial_value()
                                initial_value = ops.convert_to_tensor(initial_value, dtype=kwargs.get('dtype', None))
                    if i > 0:
                        var0name = value_list[0].name.split(':')[0]
                        kwargs['name'] = '%s/replica_%d/' % (var0name, i)
                    kwargs['initial_value'] = initial_value
                    if kwargs.get('dtype', None) is None:
                        kwargs['dtype'] = kwargs['initial_value'].dtype
                    if kwargs.get('shape', None) is None:
                        kwargs['shape'] = kwargs['initial_value'].shape
                    with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
                        v = uninitialized_variable_creator(**kwargs)
                    assert not isinstance(v, tpu_values.TPUMirroredVariable)
                    value_list.append(v)
            return value_list

        def _create_uninitialized_mirrored_tpu_replicated_variables(**kwargs):
            if False:
                while True:
                    i = 10
            'Returns a list of `TPUReplicatedVariable`s.\n\n      The list consists of `num_replicas` `TPUReplicatedVariable`s and can be\n      used to initialize a `TPUMirroredVariable`. Each `TPUReplicatedVariable`\n      contains a list of `tf.Variable`s which are replicated to\n      `num_cores_per_replica` logical cores to enable XLA SPMD compilation.\n\n      Args:\n        **kwargs: the keyword arguments for creating a variable\n      '
            dtype = kwargs.get('dtype', None)
            shape = kwargs.get('shape', None)
            initial_value = kwargs.get('initial_value', None)
            if initial_value is None:
                return _create_mirrored_tpu_replicated_variables(**kwargs)
            with maybe_init_scope():
                if initial_value is not None:
                    if callable(initial_value):
                        initial_value = initial_value()
                    initial_value = ops.convert_to_tensor(initial_value, dtype=dtype)
                    kwargs['initial_value'] = initial_value
                    if dtype is None:
                        kwargs['dtype'] = kwargs['initial_value'].dtype
                    if shape is None:
                        kwargs['shape'] = kwargs['initial_value'].shape
            mirrored_replicated_var_list = []
            for replica_id in range(num_replicas):
                replicated_var_list = []
                for logic_core_id in range(num_cores_per_replica):
                    with ops.device(self._tpu_devices[replica_id][logic_core_id]):
                        v = uninitialized_variable_creator(**kwargs)
                    replicated_var_list.append(v)
                replica_name = '{}/r:{}'.format(kwargs['name'], replica_id)
                tpu_replicated_var = tpu_replicated_variable.TPUReplicatedVariable(variables=replicated_var_list, name=replica_name)
                mirrored_replicated_var_list.append(tpu_replicated_var)
            return mirrored_replicated_var_list
        if not self._using_custom_device and enable_batch_variable_initialization():
            if self._use_spmd_for_xla_partitioning and num_cores_per_replica > 1:
                real_creator = _create_uninitialized_mirrored_tpu_replicated_variables
            else:
                real_creator = _create_uninitialized_mirrored_tpu_variables
            kwargs['experimental_batch_initialization'] = True
        elif self._use_spmd_for_xla_partitioning and num_cores_per_replica > 1:
            real_creator = _create_mirrored_tpu_replicated_variables
        else:
            real_creator = _create_mirrored_tpu_variables
        mirrored_variable = distribute_utils.create_mirrored_variable(self._container_strategy(), real_creator, distribute_utils.TPU_VARIABLE_CLASS_MAPPING, distribute_utils.TPU_VARIABLE_POLICY_MAPPING, **kwargs)
        if not self._using_custom_device and enable_batch_variable_initialization():
            setattr(mirrored_variable, '_lazy_scope', self.lazy_variable_tracker)
        return mirrored_variable

    @property
    def lazy_variable_tracker(self):
        if False:
            while True:
                i = 10
        if not getattr(self, '_lazy_variable_tracker', None):
            self._lazy_variable_tracker = tpu_util.LazyVariableTracker()
        return self._lazy_variable_tracker

    def _resource_creator_scope(self):
        if False:
            print('Hello World!')

        def lookup_creator(next_creator, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            host_to_table = collections.OrderedDict()
            for host_device in self._device_input_worker_devices.keys():
                with ops.device(host_device):
                    host_to_table[host_device] = next_creator(*args, **kwargs)
            return values.PerWorkerResource(self._container_strategy(), host_to_table)
        return ops.resource_creator_scope('StaticHashTable', lookup_creator)

    def _gather_to_implementation(self, value, destinations, axis, options):
        if False:
            print('Hello World!')
        if not isinstance(value, values.DistributedValues):
            return value
        value_list = list(value.values)
        if isinstance(value, values.DistributedVariable) and value._packed_variable is not None:
            value_list = list((value._packed_variable.on_device(d) for d in value._packed_variable.devices))
        if len(value.values) <= _XLA_OP_BY_OP_INPUTS_LIMIT:
            output = array_ops.concat(value_list, axis=axis)
        else:
            output = array_ops.concat(value_list[:_XLA_OP_BY_OP_INPUTS_LIMIT], axis=axis)
            for i in range(_XLA_OP_BY_OP_INPUTS_LIMIT, len(value_list), _XLA_OP_BY_OP_INPUTS_LIMIT - 1):
                output = array_ops.concat([output] + value_list[i:i + _XLA_OP_BY_OP_INPUTS_LIMIT - 1], axis=axis)
        output = self._broadcast_output(destinations, output)
        return output

    def _broadcast_output(self, destinations, output):
        if False:
            return 10
        devices = cross_device_ops_lib.get_devices_from(destinations)
        if len(devices) == 1:
            dest_canonical = device_util.canonicalize(devices[0])
            host_canonical = device_util.canonicalize(self._host_device)
            if dest_canonical != host_canonical:
                with ops.device(dest_canonical):
                    output = array_ops.identity(output)
        else:
            output = cross_device_ops_lib.simple_broadcast(output, destinations)
        return output

    def _reduce_to(self, reduce_op, value, destinations, options):
        if False:
            for i in range(10):
                print('nop')
        if (isinstance(value, values.DistributedValues) or tensor_util.is_tf_type(value)) and tpu_util.enclosing_tpu_context() is not None:
            if reduce_op == reduce_util.ReduceOp.MEAN:
                value = math_ops.scalar_mul(1.0 / self._num_replicas_in_sync, value)
            elif reduce_op != reduce_util.ReduceOp.SUM:
                raise NotImplementedError(f'`reduce_op`={reduce_op} is not supported. Currently we only support ReduceOp.SUM and ReduceOp.MEAN in TPUStrategy.')
            return tpu_ops.cross_replica_sum(value)
        if not isinstance(value, values.DistributedValues):
            return cross_device_ops_lib.reduce_non_distributed_value(reduce_op, value, destinations, self._num_replicas_in_sync)
        value_list = value.values
        if isinstance(value, values.DistributedVariable) and value._packed_variable is not None:
            value_list = tuple((value._packed_variable.on_device(d) for d in value._packed_variable.devices))
        if len(value.values) <= _XLA_OP_BY_OP_INPUTS_LIMIT:
            output = math_ops.add_n(value_list)
        else:
            output = array_ops.zeros_like(value_list[0], dtype=value_list[0].dtype)
            for i in range(0, len(value_list), _XLA_OP_BY_OP_INPUTS_LIMIT):
                output += math_ops.add_n(value_list[i:i + _XLA_OP_BY_OP_INPUTS_LIMIT])
        if reduce_op == reduce_util.ReduceOp.MEAN:
            output *= 1.0 / len(value_list)
        output = self._broadcast_output(destinations, output)
        return output

    def _update(self, var, fn, args, kwargs, group):
        if False:
            i = 10
            return i + 15
        assert isinstance(var, tpu_values.TPUVariableMixin) or isinstance(var, resource_variable_ops.BaseResourceVariable)
        if tpu_util.enclosing_tpu_context() is not None:
            if group:
                return fn(var, *args, **kwargs)
            else:
                return (fn(var, *args, **kwargs),)
        packed_var = var._packed_variable
        if packed_var is not None and (not context.executing_eagerly()):
            if group:
                return fn(packed_var, *args, **kwargs)
            else:
                return (fn(packed_var, *args, **kwargs),)
        updates = []
        values_and_devices = []
        if packed_var is not None:
            for device in packed_var.devices:
                values_and_devices.append((packed_var, device))
        else:
            for value in var.values:
                values_and_devices.append((value, value.device))
        if var.synchronization != variables_lib.VariableSynchronization.ON_READ and var.aggregation != variables_lib.VariableAggregation.NONE:
            distribute_utils.assert_mirrored(args)
            distribute_utils.assert_mirrored(kwargs)
        for (i, value_and_device) in enumerate(values_and_devices):
            value = value_and_device[0]
            device = value_and_device[1]
            name = 'update_%d' % i
            with ops.device(device), distribute_lib.UpdateContext(i), ops.name_scope(name):
                updates.append(fn(value, *distribute_utils.select_replica(i, args), **distribute_utils.select_replica(i, kwargs)))
        return distribute_utils.update_regroup(self, updates, group)

    def read_var(self, var):
        if False:
            i = 10
            return i + 15
        assert isinstance(var, tpu_values.TPUVariableMixin) or isinstance(var, resource_variable_ops.BaseResourceVariable)
        return var.read_value()

    def value_container(self, value):
        if False:
            i = 10
            return i + 15
        return value

    def _broadcast_to(self, tensor, destinations):
        if False:
            for i in range(10):
                print('nop')
        del destinations
        if isinstance(tensor, (float, int)):
            return tensor
        if tpu_util.enclosing_tpu_context() is not None:
            broadcast_tensor = [tensor for _ in range(self._num_replicas_in_sync)]
            result = tpu_ops.all_to_all(broadcast_tensor, concat_dimension=0, split_dimension=0, split_count=self._num_replicas_in_sync)
            return result[0]
        return tensor

    @property
    def num_hosts(self):
        if False:
            i = 10
            return i + 15
        if self._device_assignment is None:
            return self._tpu_metadata.num_hosts
        return len(set([self._device_assignment.host_device(r) for r in range(self._device_assignment.num_replicas)]))

    @property
    def num_replicas_per_host(self):
        if False:
            while True:
                i = 10
        if self._device_assignment is None:
            return self._tpu_metadata.num_of_cores_per_host
        max_models_per_host = self._tpu_metadata.num_of_cores_per_host // self._device_assignment.num_cores_per_replica
        return min(self._device_assignment.num_replicas, max_models_per_host)

    @property
    def _num_replicas_in_sync(self):
        if False:
            for i in range(10):
                print('nop')
        if self._device_assignment is None:
            return self._tpu_metadata.num_cores
        return self._device_assignment.num_replicas

    @property
    def experimental_between_graph(self):
        if False:
            while True:
                i = 10
        return False

    @property
    def experimental_should_init(self):
        if False:
            i = 10
            return i + 15
        return True

    @property
    def should_checkpoint(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    @property
    def should_save_summary(self):
        if False:
            return 10
        return True

    @property
    def worker_devices(self):
        if False:
            print('Hello World!')
        return tuple(self._tpu_devices[:, self._logical_device_stack[-1]])

    @property
    def parameter_devices(self):
        if False:
            print('Hello World!')
        return self.worker_devices

    @property
    def tpu_hardware_feature(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the `tf.tpu.experimental.HardwareFeature` class.'
        return tpu_hardware_feature.HardwareFeature(self._tpu_cluster_resolver.tpu_hardware_feature)

    def non_slot_devices(self, var_list):
        if False:
            print('Hello World!')
        return self._host_device

    def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
        if False:
            i = 10
            return i + 15
        del colocate_with
        with ops.device(self._host_device), distribute_lib.UpdateContext(None):
            result = fn(*args, **kwargs)
            if group:
                return result
            else:
                return nest.map_structure(self._local_results, result)

    def _configure(self, session_config=None, cluster_spec=None, task_type=None, task_id=None):
        if False:
            i = 10
            return i + 15
        del cluster_spec, task_type, task_id
        if session_config:
            session_config.CopyFrom(self._update_config_proto(session_config))

    def _update_config_proto(self, config_proto):
        if False:
            print('Hello World!')
        updated_config = copy.deepcopy(config_proto)
        updated_config.isolate_session_state = True
        cluster_spec = self._tpu_cluster_resolver.cluster_spec()
        if cluster_spec:
            updated_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
        return updated_config

    @property
    def _global_batch_size(self):
        if False:
            return 10
        '`make_dataset_iterator` and `make_numpy_iterator` use global batch size.\n\n    `make_input_fn_iterator` assumes per-replica batching.\n\n    Returns:\n      Boolean.\n    '
        return True

    def tpu_run(self, fn, args, kwargs, options=None):
        if False:
            i = 10
            return i + 15
        func = self._tpu_function_creator(fn, options)
        return func(args, kwargs)

    def _tpu_function_creator(self, fn, options):
        if False:
            while True:
                i = 10
        if context.executing_eagerly() and fn in self._tpu_function_cache:
            return self._tpu_function_cache[fn]
        strategy = self._container_strategy()

        def tpu_function(args, kwargs):
            if False:
                i = 10
                return i + 15
            'TF Function used to replicate the user computation.'
            logging.vlog(1, '`TPUStrategy.run` is called with [args: %s] [kwargs: %s]', args, kwargs)
            if kwargs is None:
                kwargs = {}
            result = [[]]

            def replicated_fn(replica_id, replica_args, replica_kwargs):
                if False:
                    return 10
                'Wraps user function to provide replica ID and `Tensor` inputs.'
                with _TPUReplicaContext(strategy, replica_id_in_sync_group=replica_id):
                    result[0] = fn(*replica_args, **replica_kwargs)
                return result[0]
            replicate_inputs = []
            for i in range(strategy.num_replicas_in_sync):
                replicate_inputs.append([constant_op.constant(i, dtype=dtypes.int32), distribute_utils.select_replica(i, args), distribute_utils.select_replica(i, kwargs)])
            if options.experimental_enable_dynamic_batch_size and replicate_inputs:
                maximum_shapes = []
                flattened_list = nest.flatten(replicate_inputs[0])
                for input_tensor in flattened_list:
                    if tensor_util.is_tf_type(input_tensor):
                        rank = input_tensor.shape.rank
                    else:
                        rank = np.ndim(input_tensor)
                    if rank is None:
                        raise ValueError('input tensor {} to TPUStrategy.run() has unknown rank, which is not allowed'.format(input_tensor))
                    maximum_shape = tensor_shape.TensorShape([None] * rank)
                    maximum_shapes.append(maximum_shape)
                maximum_shapes = nest.pack_sequence_as(replicate_inputs[0], maximum_shapes)
            else:
                maximum_shapes = None
            if options.experimental_bucketizing_dynamic_shape:
                padding_spec = tpu.PaddingSpec.POWER_OF_TWO
            else:
                padding_spec = None
            with strategy.scope():
                xla_options = options.experimental_xla_options or tpu.XLAOptions(use_spmd_for_xla_partitioning=self._use_spmd_for_xla_partitioning)
                replicate_outputs = tpu.replicate(replicated_fn, replicate_inputs, device_assignment=self._device_assignment, maximum_shapes=maximum_shapes, padding_spec=padding_spec, xla_options=xla_options)
            filter_ops = lambda x: [o for o in x if not isinstance(o, ops.Operation)]
            if isinstance(result[0], list):
                result[0] = filter_ops(result[0])
            if result[0] is None or isinstance(result[0], ops.Operation):
                replicate_outputs = [None] * len(replicate_outputs)
            else:
                replicate_outputs = [nest.pack_sequence_as(result[0], filter_ops(nest.flatten(output))) for output in replicate_outputs]
            return distribute_utils.regroup(replicate_outputs)
        if context.executing_eagerly():
            tpu_function = def_function.function(tpu_function)
            self._tpu_function_cache[fn] = tpu_function
        return tpu_function

    def _in_multi_worker_mode(self):
        if False:
            print('Hello World!')
        'Whether this strategy indicates working in multi-worker settings.'
        return False

    def _get_local_replica_id(self, replica_id_in_sync_group):
        if False:
            i = 10
            return i + 15
        return replica_id_in_sync_group

def _make_axis_nonnegative(axis, rank):
    if False:
        while True:
            i = 10
    if isinstance(axis, int):
        if axis >= 0:
            return axis
        else:
            return axis + rank
    else:
        return array_ops.where_v2(math_ops.greater_equal(axis, 0), axis, axis + rank)
_DTYPES_SUPPORTED_BY_CROSS_REPLICA_SUM = (dtypes.bfloat16, dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32, dtypes.uint32)

class _TPUReplicaContext(distribute_lib.ReplicaContext):
    """Replication Context class for TPU Strategy."""

    def __init__(self, strategy, replica_id_in_sync_group=0):
        if False:
            for i in range(10):
                print('nop')
        distribute_lib.ReplicaContext.__init__(self, strategy, replica_id_in_sync_group=replica_id_in_sync_group)

    @property
    def devices(self):
        if False:
            return 10
        distribute_lib.require_replica_context(self)
        ds = self._strategy
        replica_id = tensor_util.constant_value(self.replica_id_in_sync_group)
        if replica_id is None:
            return (tpu.core(0),)
        else:
            return (ds.extended.worker_devices[replica_id],)

    def experimental_logical_device(self, logical_device_id):
        if False:
            for i in range(10):
                print('nop')
        'Places variables and ops on the specified logical device.'
        return self.strategy.extended.experimental_logical_device(logical_device_id)

    def _compute_all_gather_output_shape(self, value_shape, value_rank, axis):
        if False:
            print('Hello World!')
        if isinstance(value_rank, int):
            output_shape = list(value_shape)
            output_shape[axis] *= self.num_replicas_in_sync
        else:
            output_shape = array_ops.where_v2(math_ops.equal(math_ops.range(value_rank), axis), value_shape * context.num_replicas_in_sync, value_shape)
        return output_shape

    def all_gather(self, value, axis, experimental_hints=None):
        if False:
            i = 10
            return i + 15
        del experimental_hints
        for v in nest.flatten(value):
            if isinstance(v, indexed_slices.IndexedSlices):
                raise NotImplementedError('all_gather does not support IndexedSlices')

        def _all_gather_tensor(value, axis):
            if False:
                print('Hello World!')
            value = ops.convert_to_tensor(value)
            if value.shape.rank is None:
                value_rank = array_ops.rank(value)
                value_shape = array_ops.shape(value)
            else:
                value_rank = value.shape.rank
                value_shape = value.shape.as_list()
                value_shape_tensor = array_ops.shape(value)
                for i in range(len(value_shape)):
                    if value_shape[i] is None:
                        value_shape[i] = value_shape_tensor[i]
            axis = _make_axis_nonnegative(axis, value_rank)
            if isinstance(value_rank, int):
                replica_broadcast_shape = [1] * (value_rank + 1)
                replica_broadcast_shape[axis] = self.num_replicas_in_sync
            else:
                replica_broadcast_shape = array_ops.where_v2(math_ops.equal(math_ops.range(value_rank + 1), axis), self.num_replicas_in_sync, 1)
            output_shape = self._compute_all_gather_output_shape(value_shape, value_rank, axis)
            if value.dtype in _DTYPES_SUPPORTED_BY_CROSS_REPLICA_SUM:
                replica_id_mask = array_ops.one_hot(self.replica_id_in_sync_group, self.num_replicas_in_sync)
                replica_id_mask = array_ops.reshape(replica_id_mask, replica_broadcast_shape)
                replica_id_mask = math_ops.cast(replica_id_mask, value.dtype)
                gathered_value = array_ops.expand_dims(value, axis) * replica_id_mask
                gathered_value = self.all_reduce(reduce_util.ReduceOp.SUM, gathered_value)
                return array_ops.reshape(gathered_value, output_shape)
            else:
                inputs = array_ops.expand_dims(value, axis=axis)
                inputs = array_ops.tile(inputs, replica_broadcast_shape)
                unordered_output = tpu_ops.all_to_all(inputs, concat_dimension=axis, split_dimension=axis, split_count=self.num_replicas_in_sync)
                concat_replica_id = array_ops.reshape(self.replica_id_in_sync_group, [1])
                concat_replica_id = array_ops.tile(concat_replica_id, [self.num_replicas_in_sync])
                xla_to_replica_context_id = tpu_ops.all_to_all(concat_replica_id, concat_dimension=0, split_dimension=0, split_count=self.num_replicas_in_sync)
                replica_context_to_xla_id = math_ops.argmax(array_ops.one_hot(xla_to_replica_context_id, self.num_replicas_in_sync), axis=0)
                sorted_with_extra_dim = array_ops.gather(unordered_output, replica_context_to_xla_id, axis=axis)
                return array_ops.reshape(sorted_with_extra_dim, output_shape)
        ys = [_all_gather_tensor(t, axis=axis) for t in nest.flatten(value)]
        return nest.pack_sequence_as(value, ys)

def _set_last_step_outputs(ctx, last_step_tensor_outputs):
    if False:
        print('Hello World!')
    'Sets the last step outputs on the given context.'
    last_step_tensor_outputs_dict = nest.pack_sequence_as(ctx.last_step_outputs, last_step_tensor_outputs)
    for (name, reduce_op) in ctx._last_step_outputs_reduce_ops.items():
        output = last_step_tensor_outputs_dict[name]
        if reduce_op is None:
            last_step_tensor_outputs_dict[name] = values.PerReplica(output)
        else:
            last_step_tensor_outputs_dict[name] = output[0]
    ctx._set_last_step_outputs(last_step_tensor_outputs_dict)