"""A tf.distribute.Strategy for running on a single device."""
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

@tf_export('distribute.OneDeviceStrategy', v1=[])
class OneDeviceStrategy(distribute_lib.Strategy):
    """A distribution strategy for running on a single device.

  Using this strategy will place any variables created in its scope on the
  specified device. Input distributed through this strategy will be
  prefetched to the specified device. Moreover, any functions called via
  `strategy.run` will also be placed on the specified device
  as well.

  Typical usage of this strategy could be testing your code with the
  tf.distribute.Strategy API before switching to other strategies which
  actually distribute to multiple devices/machines.

  For example:
  ```
  strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

  with strategy.scope():
    v = tf.Variable(1.0)
    print(v.device)  # /job:localhost/replica:0/task:0/device:GPU:0

  def step_fn(x):
    return x * 2

  result = 0
  for i in range(10):
    result += strategy.run(step_fn, args=(i,))
  print(result)  # 90
  ```
  """

    def __init__(self, device):
        if False:
            for i in range(10):
                print('nop')
        'Creates a `OneDeviceStrategy`.\n\n    Args:\n      device: Device string identifier for the device on which the variables\n        should be placed. See class docs for more details on how the device is\n        used. Examples: "/cpu:0", "/gpu:0", "/device:CPU:0", "/device:GPU:0"\n    '
        super(OneDeviceStrategy, self).__init__(OneDeviceExtended(self, device))
        distribute_lib.distribution_strategy_gauge.get_cell('V2').set('OneDeviceStrategy')

    def experimental_distribute_dataset(self, dataset, options=None):
        if False:
            i = 10
            return i + 15
        'Distributes a tf.data.Dataset instance provided via dataset.\n\n    In this case, there is only one device, so this is only a thin wrapper\n    around the input dataset. It will, however, prefetch the input data to the\n    specified device. The returned distributed dataset can be iterated over\n    similar to how regular datasets can.\n\n    NOTE: Currently, the user cannot add any more transformations to a\n    distributed dataset.\n\n    Example:\n    ```\n    strategy = tf.distribute.OneDeviceStrategy()\n    dataset = tf.data.Dataset.range(10).batch(2)\n    dist_dataset = strategy.experimental_distribute_dataset(dataset)\n    for x in dist_dataset:\n      print(x)  # [0, 1], [2, 3],...\n    ```\n    Args:\n      dataset: `tf.data.Dataset` to be prefetched to device.\n      options: `tf.distribute.InputOptions` used to control options on how this\n        dataset is distributed.\n    Returns:\n      A "distributed `Dataset`" that the caller can iterate over.\n    '
        return super(OneDeviceStrategy, self).experimental_distribute_dataset(dataset, options)

    def distribute_datasets_from_function(self, dataset_fn, options=None):
        if False:
            while True:
                i = 10
        'Distributes `tf.data.Dataset` instances created by calls to `dataset_fn`.\n\n    `dataset_fn` will be called once for each worker in the strategy. In this\n    case, we only have one worker and one device so `dataset_fn` is called\n    once.\n\n    The `dataset_fn` should take an `tf.distribute.InputContext` instance where\n    information about batching and input replication can be accessed:\n\n    ```\n    def dataset_fn(input_context):\n      batch_size = input_context.get_per_replica_batch_size(global_batch_size)\n      d = tf.data.Dataset.from_tensors([[1.]]).repeat().batch(batch_size)\n      return d.shard(\n          input_context.num_input_pipelines, input_context.input_pipeline_id)\n\n    inputs = strategy.distribute_datasets_from_function(dataset_fn)\n\n    for batch in inputs:\n      replica_results = strategy.run(replica_fn, args=(batch,))\n    ```\n\n    IMPORTANT: The `tf.data.Dataset` returned by `dataset_fn` should have a\n    per-replica batch size, unlike `experimental_distribute_dataset`, which uses\n    the global batch size.  This may be computed using\n    `input_context.get_per_replica_batch_size`.\n\n    Args:\n      dataset_fn: A function taking a `tf.distribute.InputContext` instance and\n        returning a `tf.data.Dataset`.\n      options: `tf.distribute.InputOptions` used to control options on how this\n        dataset is distributed.\n\n    Returns:\n      A "distributed `Dataset`", which the caller can iterate over like regular\n      datasets.\n    '
        return super(OneDeviceStrategy, self).distribute_datasets_from_function(dataset_fn, options)

    def experimental_local_results(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Returns the list of all local per-replica values contained in `value`.\n\n    In `OneDeviceStrategy`, the `value` is always expected to be a single\n    value, so the result is just the value in a tuple.\n\n    Args:\n      value: A value returned by `experimental_run()`, `run()`,\n        `extended.call_for_each_replica()`, or a variable created in `scope`.\n\n    Returns:\n      A tuple of values contained in `value`. If `value` represents a single\n      value, this returns `(value,).`\n    '
        return super(OneDeviceStrategy, self).experimental_local_results(value)

    def run(self, fn, args=(), kwargs=None, options=None):
        if False:
            print('Hello World!')
        'Run `fn` on each replica, with the given arguments.\n\n    In `OneDeviceStrategy`, `fn` is simply called within a device scope for the\n    given device, with the provided arguments.\n\n    Args:\n      fn: The function to run. The output must be a `tf.nest` of `Tensor`s.\n      args: (Optional) Positional arguments to `fn`.\n      kwargs: (Optional) Keyword arguments to `fn`.\n      options: (Optional) An instance of `tf.distribute.RunOptions` specifying\n        the options to run `fn`.\n\n    Returns:\n      Return value from running `fn`.\n    '
        return super(OneDeviceStrategy, self).run(fn, args, kwargs, options)

    def reduce(self, reduce_op, value, axis):
        if False:
            for i in range(10):
                print('nop')
        'Reduce `value` across replicas.\n\n    In `OneDeviceStrategy`, there is only one replica, so if axis=None, value\n    is simply returned. If axis is specified as something other than None,\n    such as axis=0, value is reduced along that axis and returned.\n\n    Example:\n    ```\n    t = tf.range(10)\n\n    result = strategy.reduce(tf.distribute.ReduceOp.SUM, t, axis=None).numpy()\n    # result: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]\n\n    result = strategy.reduce(tf.distribute.ReduceOp.SUM, t, axis=0).numpy()\n    # result: 45\n    ```\n\n    Args:\n      reduce_op: A `tf.distribute.ReduceOp` value specifying how values should\n        be combined.\n      value: A "per replica" value, e.g. returned by `run` to\n        be combined into a single tensor.\n      axis: Specifies the dimension to reduce along within each\n        replica\'s tensor. Should typically be set to the batch dimension, or\n        `None` to only reduce across replicas (e.g. if the tensor has no batch\n        dimension).\n\n    Returns:\n      A `Tensor`.\n    '
        return super(OneDeviceStrategy, self).reduce(reduce_op, value, axis)

    def scope(self):
        if False:
            while True:
                i = 10
        'Returns a context manager selecting this Strategy as current.\n\n    Inside a `with strategy.scope():` code block, this thread\n    will use a variable creator set by `strategy`, and will\n    enter its "cross-replica context".\n\n    In `OneDeviceStrategy`, all variables created inside `strategy.scope()`\n    will be on `device` specified at strategy construction time.\n    See example in the docs for this class.\n\n    Returns:\n      A context manager to use for creating variables with this strategy.\n    '
        return super(OneDeviceStrategy, self).scope()

@tf_export(v1=['distribute.OneDeviceStrategy'])
class OneDeviceStrategyV1(distribute_lib.StrategyV1):
    __doc__ = OneDeviceStrategy.__doc__.replace('For example:\n  ```', 'For example:\n  ```\n  tf.enable_eager_execution()')

    def __init__(self, device):
        if False:
            print('Hello World!')
        super(OneDeviceStrategyV1, self).__init__(OneDeviceExtended(self, device))
        distribute_lib.distribution_strategy_gauge.get_cell('V1').set('OneDeviceStrategy')
    __init__.__doc__ = OneDeviceStrategy.__init__.__doc__

class OneDeviceExtended(distribute_lib.StrategyExtendedV1):
    """Implementation of OneDeviceStrategy."""

    def __init__(self, container_strategy, device):
        if False:
            for i in range(10):
                print('nop')
        super(OneDeviceExtended, self).__init__(container_strategy)
        self._device = device_util.resolve(device)
        self._input_device = device_util.get_host_for_device(self._device)

    def _input_workers_with_options(self, options=None):
        if False:
            for i in range(10):
                print('nop')
        if not options or options.experimental_fetch_to_device:
            return input_lib.InputWorkers([(self._input_device, (self._device,))])
        else:
            return input_lib.InputWorkers([(self._input_device, (self._input_device,))])

    @property
    def _input_workers(self):
        if False:
            return 10
        return self._input_workers_with_options()

    def _create_variable(self, next_creator, **kwargs):
        if False:
            i = 10
            return i + 15
        colocate_with = kwargs.pop('colocate_with', None)
        if colocate_with is None:
            with ops.device(self._device):
                return next_creator(**kwargs)
        elif isinstance(colocate_with, numpy_dataset.SingleDevice):
            with ops.device(colocate_with.device):
                return next_creator(**kwargs)
        else:
            with ops.colocate_with(colocate_with):
                return next_creator(**kwargs)

    def _validate_colocate_with_variable(self, colocate_with_variable):
        if False:
            print('Hello World!')
        distribute_utils.validate_colocate(colocate_with_variable, self)

    def _make_dataset_iterator(self, dataset):
        if False:
            i = 10
            return i + 15
        'Make iterator from dataset without splitting the batch.'
        return input_lib_v1.DatasetIterator(dataset, self._input_workers, self._container_strategy())

    def _make_input_fn_iterator(self, input_fn, replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
        if False:
            for i in range(10):
                print('nop')
        return input_lib_v1.InputFunctionIterator(input_fn, self._input_workers, [distribute_lib.InputContext()], self._container_strategy())

    def _experimental_make_numpy_dataset(self, numpy_input, session):
        if False:
            while True:
                i = 10
        return numpy_dataset.one_host_numpy_dataset(numpy_input, numpy_dataset.SingleDevice(self._input_device), session)

    def _broadcast_to(self, tensor, destinations):
        if False:
            while True:
                i = 10
        del destinations
        return tensor

    def _experimental_distribute_dataset(self, dataset, options):
        if False:
            print('Hello World!')
        if options and options.experimental_replication_mode == distribute_lib.InputReplicationMode.PER_REPLICA:
            raise NotImplementedError('InputReplicationMode.PER_REPLICA is only supported in  `experimental_distribute_datasets_from_function`.')
        return input_util.get_distributed_dataset(dataset, self._input_workers_with_options(options), self._container_strategy(), options=options)

    def _distribute_datasets_from_function(self, dataset_fn, options):
        if False:
            i = 10
            return i + 15
        if options and options.experimental_replication_mode == distribute_lib.InputReplicationMode.PER_REPLICA:
            raise NotImplementedError('InputReplicationMode.PER_REPLICA is only supported in `experimental_distribute_datasets_from_function` of tf.distribute.MirroredStrategy')
        return input_util.get_distributed_datasets_from_function(dataset_fn, self._input_workers_with_options(options), [distribute_lib.InputContext()], self._container_strategy(), options=options)

    def _experimental_distribute_values_from_function(self, value_fn):
        if False:
            print('Hello World!')
        return value_fn(distribute_lib.ValueContext())

    def _experimental_run_steps_on_iterator(self, fn, iterator, iterations, initial_loop_values=None):
        if False:
            for i in range(10):
                print('nop')
        if initial_loop_values is None:
            initial_loop_values = {}
        initial_loop_values = nest.flatten(initial_loop_values)
        ctx = input_lib.MultiStepContext()

        def body(i, *args):
            if False:
                print('Hello World!')
            'A wrapper around `fn` to create the while loop body.'
            del args
            fn_result = fn(ctx, iterator.get_next())
            flat_last_step_outputs = nest.flatten(ctx.last_step_outputs)
            with ops.control_dependencies([fn_result]):
                return [i + 1] + flat_last_step_outputs
        self._outer_control_flow_context = ops.get_default_graph()._get_control_flow_context()
        cond = lambda i, *args: i < iterations
        i = constant_op.constant(0)
        loop_result = while_loop.while_loop(cond, body, [i] + initial_loop_values, name='', parallel_iterations=1, back_prop=False, swap_memory=False, return_same_structure=True)
        del self._outer_control_flow_context
        ctx.run_op = control_flow_ops.group(loop_result)
        last_step_tensor_outputs = loop_result[1:]
        last_step_tensor_outputs_dict = nest.pack_sequence_as(ctx.last_step_outputs, last_step_tensor_outputs)
        ctx._set_last_step_outputs(last_step_tensor_outputs_dict)
        return ctx

    def _call_for_each_replica(self, fn, args, kwargs):
        if False:
            print('Hello World!')
        strategy = self._container_strategy()
        with ops.device(self._device), _OneDeviceReplicaContext(strategy):
            return fn(*args, **kwargs)

    def _reduce_to(self, reduce_op, value, destinations, options):
        if False:
            for i in range(10):
                print('nop')
        del reduce_op, destinations, options
        return value

    def _gather_to_implementation(self, value, destinations, axis, options):
        if False:
            print('Hello World!')
        del destinations, axis, options
        return value

    def _update(self, var, fn, args, kwargs, group):
        if False:
            for i in range(10):
                print('nop')
        return self._update_non_slot(var, fn, (var,) + tuple(args), kwargs, group)

    def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
        if False:
            for i in range(10):
                print('nop')
        del colocate_with
        with ops.device(self._device), distribute_lib.UpdateContext(self._device):
            result = fn(*args, **kwargs)
            if group:
                return result
            else:
                return nest.map_structure(self._local_results, result)

    def read_var(self, replica_local_var):
        if False:
            i = 10
            return i + 15
        'Read the aggregate value of a replica-local variable.'
        return array_ops.identity(replica_local_var)

    def _local_results(self, value):
        if False:
            while True:
                i = 10
        return (value,)

    def value_container(self, value):
        if False:
            while True:
                i = 10
        return value

    def _in_multi_worker_mode(self):
        if False:
            for i in range(10):
                print('nop')
        'Whether this strategy indicates working in multi-worker settings.'
        return False

    @property
    def _num_replicas_in_sync(self):
        if False:
            while True:
                i = 10
        return 1

    @property
    def worker_devices(self):
        if False:
            while True:
                i = 10
        return (self._device,)

    @property
    def parameter_devices(self):
        if False:
            return 10
        return (self._device,)

    def non_slot_devices(self, var_list):
        if False:
            while True:
                i = 10
        del var_list
        return (self._device,)

    @property
    def experimental_should_init(self):
        if False:
            return 10
        return True

    @property
    def experimental_between_graph(self):
        if False:
            while True:
                i = 10
        return False

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
    def _global_batch_size(self):
        if False:
            i = 10
            return i + 15
        'Global and per-replica batching are equivalent for OneDeviceStrategy.'
        return True

    @property
    def _support_per_replica_values(self):
        if False:
            i = 10
            return i + 15
        return False

    def _get_local_replica_id(self, replica_id_in_sync_group):
        if False:
            print('Hello World!')
        return replica_id_in_sync_group

class _OneDeviceReplicaContext(distribute_lib.ReplicaContext):
    """ReplicaContext for OneDeviceStrategy."""

    def __init__(self, strategy):
        if False:
            return 10
        distribute_lib.ReplicaContext.__init__(self, strategy, replica_id_in_sync_group=0)

    @property
    def devices(self):
        if False:
            for i in range(10):
                print('nop')
        return self._strategy.extended.worker_devices