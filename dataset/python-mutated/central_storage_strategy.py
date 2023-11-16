"""Class implementing a single machine parameter server strategy."""
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.util.tf_export import tf_export

@tf_export('distribute.experimental.CentralStorageStrategy', v1=[])
class CentralStorageStrategy(distribute_lib.Strategy):
    """A one-machine strategy that puts all variables on a single device.

  Variables are assigned to local CPU or the only GPU. If there is more
  than one GPU, compute operations (other than variable update operations)
  will be replicated across all GPUs.

  For Example:
  ```
  strategy = tf.distribute.experimental.CentralStorageStrategy()
  # Create a dataset
  ds = tf.data.Dataset.range(5).batch(2)
  # Distribute that dataset
  dist_dataset = strategy.experimental_distribute_dataset(ds)

  with strategy.scope():
    @tf.function
    def train_step(val):
      return val + 1

    # Iterate over the distributed dataset
    for x in dist_dataset:
      # process dataset elements
      strategy.run(train_step, args=(x,))
  ```
  """

    def __init__(self, compute_devices=None, parameter_device=None):
        if False:
            while True:
                i = 10
        extended = parameter_server_strategy.ParameterServerStrategyExtended(self, compute_devices=compute_devices, parameter_device=parameter_device)
        'Initializes the strategy with optional device strings.\n\n    Args:\n    compute_devices: an optional list of strings for device to replicate models\n      on. If this is not provided, all local GPUs will be used; if there is no\n      GPU, local CPU will be used.\n    parameter_device: an optional device string for which device to put\n      variables on. The default one is CPU or GPU if there is only one.\n    '
        super(CentralStorageStrategy, self).__init__(extended)
        distribute_lib.distribution_strategy_gauge.get_cell('V2').set('CentralStorageStrategy')

    @classmethod
    def _from_num_gpus(cls, num_gpus):
        if False:
            while True:
                i = 10
        return cls(device_util.local_devices_from_num_gpus(num_gpus))

    def experimental_distribute_dataset(self, dataset, options=None):
        if False:
            for i in range(10):
                print('nop')
        'Distributes a tf.data.Dataset instance provided via dataset.\n\n    The returned dataset is a wrapped strategy dataset which creates a\n    multidevice iterator under the hood. It prefetches the input data to the\n    specified devices on the worker. The returned distributed dataset can be\n    iterated over similar to how regular datasets can.\n\n    NOTE: Currently, the user cannot add any more transformations to a\n    distributed dataset.\n\n    For Example:\n    ```\n    strategy = tf.distribute.CentralStorageStrategy()  # with 1 CPU and 1 GPU\n    dataset = tf.data.Dataset.range(10).batch(2)\n    dist_dataset = strategy.experimental_distribute_dataset(dataset)\n    for x in dist_dataset:\n      print(x)  # Prints PerReplica values [0, 1], [2, 3],...\n\n    ```\n    Args:\n      dataset: `tf.data.Dataset` to be prefetched to device.\n      options: `tf.distribute.InputOptions` used to control options on how this\n        dataset is distributed.\n\n    Returns:\n      A "distributed `Dataset`" that the caller can iterate over.\n    '
        if options and options.experimental_replication_moden == distribute_lib.InputReplicationMode.PER_REPLICA:
            raise NotImplementedError('InputReplicationMode.PER_REPLICA is only supported in `experimental_distribute_datasets_from_function`.')
        return super(CentralStorageStrategy, self).experimental_distribute_dataset(dataset, options)

    def experimental_local_results(self, value):
        if False:
            print('Hello World!')
        'Returns the list of all local per-replica values contained in `value`.\n\n    In `CentralStorageStrategy` there is a single worker so the value returned\n    will be all the values on that worker.\n\n    Args:\n      value: A value returned by `run()`, `extended.call_for_each_replica()`,\n      or a variable created in `scope`.\n\n    Returns:\n      A tuple of values contained in `value`. If `value` represents a single\n      value, this returns `(value,).`\n    '
        return super(CentralStorageStrategy, self).experimental_local_results(value)

    def run(self, fn, args=(), kwargs=None, options=None):
        if False:
            return 10
        'Run `fn` on each replica, with the given arguments.\n\n    In `CentralStorageStrategy`, `fn` is  called on each of the compute\n    replicas, with the provided "per replica" arguments specific to that device.\n\n    Args:\n      fn: The function to run. The output must be a `tf.nest` of `Tensor`s.\n      args: (Optional) Positional arguments to `fn`.\n      kwargs: (Optional) Keyword arguments to `fn`.\n      options: (Optional) An instance of `tf.distribute.RunOptions` specifying\n        the options to run `fn`.\n\n    Returns:\n      Return value from running `fn`.\n    '
        return super(CentralStorageStrategy, self).run(fn, args, kwargs, options)

    def reduce(self, reduce_op, value, axis):
        if False:
            i = 10
            return i + 15
        'Reduce `value` across replicas.\n\n    Given a per-replica value returned by `run`, say a\n    per-example loss, the batch will be divided across all the replicas. This\n    function allows you to aggregate across replicas and optionally also across\n    batch elements.  For example, if you have a global batch size of 8 and 2\n    replicas, values for examples `[0, 1, 2, 3]` will be on replica 0 and\n    `[4, 5, 6, 7]` will be on replica 1. By default, `reduce` will just\n    aggregate across replicas, returning `[0+4, 1+5, 2+6, 3+7]`. This is useful\n    when each replica is computing a scalar or some other value that doesn\'t\n    have a "batch" dimension (like a gradient). More often you will want to\n    aggregate across the global batch, which you can get by specifying the batch\n    dimension as the `axis`, typically `axis=0`. In this case it would return a\n    scalar `0+1+2+3+4+5+6+7`.\n\n    If there is a last partial batch, you will need to specify an axis so\n    that the resulting shape is consistent across replicas. So if the last\n    batch has size 6 and it is divided into [0, 1, 2, 3] and [4, 5], you\n    would get a shape mismatch unless you specify `axis=0`. If you specify\n    `tf.distribute.ReduceOp.MEAN`, using `axis=0` will use the correct\n    denominator of 6. Contrast this with computing `reduce_mean` to get a\n    scalar value on each replica and this function to average those means,\n    which will weigh some values `1/8` and others `1/4`.\n\n    For Example:\n    ```\n    strategy = tf.distribute.experimental.CentralStorageStrategy(\n        compute_devices=[\'CPU:0\', \'GPU:0\'], parameter_device=\'CPU:0\')\n    ds = tf.data.Dataset.range(10)\n    # Distribute that dataset\n    dist_dataset = strategy.experimental_distribute_dataset(ds)\n\n    with strategy.scope():\n      @tf.function\n      def train_step(val):\n        # pass through\n        return val\n\n      # Iterate over the distributed dataset\n      for x in dist_dataset:\n        result = strategy.run(train_step, args=(x,))\n\n    result = strategy.reduce(tf.distribute.ReduceOp.SUM, result,\n                             axis=None).numpy()\n    # result: array([ 4,  6,  8, 10])\n\n    result = strategy.reduce(tf.distribute.ReduceOp.SUM, result, axis=0).numpy()\n    # result: 28\n    ```\n\n    Args:\n      reduce_op: A `tf.distribute.ReduceOp` value specifying how values should\n        be combined.\n      value: A "per replica" value, e.g. returned by `run` to\n        be combined into a single tensor.\n      axis: Specifies the dimension to reduce along within each\n        replica\'s tensor. Should typically be set to the batch dimension, or\n        `None` to only reduce across replicas (e.g. if the tensor has no batch\n        dimension).\n\n    Returns:\n      A `Tensor`.\n    '
        return super(CentralStorageStrategy, self).reduce(reduce_op, value, axis)

@tf_export(v1=['distribute.experimental.CentralStorageStrategy'])
class CentralStorageStrategyV1(distribute_lib.StrategyV1):
    __doc__ = CentralStorageStrategy.__doc__

    def __init__(self, compute_devices=None, parameter_device=None):
        if False:
            i = 10
            return i + 15
        super(CentralStorageStrategyV1, self).__init__(parameter_server_strategy.ParameterServerStrategyExtended(self, compute_devices=compute_devices, parameter_device=parameter_device))
        distribute_lib.distribution_strategy_gauge.get_cell('V1').set('CentralStorageStrategy')
    __init__.__doc__ = CentralStorageStrategy.__init__.__doc__