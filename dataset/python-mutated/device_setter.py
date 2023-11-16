"""Device function for replicated training."""
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
from tensorflow.python.util.tf_export import tf_export
STANDARD_PS_OPS = ('Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable', 'MutableHashTableV2', 'MutableHashTableOfTensors', 'MutableHashTableOfTensorsV2', 'MutableDenseHashTable', 'MutableDenseHashTableV2', 'VarHandleOp', 'BoostedTreesEnsembleResourceHandleOp', 'BoostedTreesQuantileStreamResourceHandleOp', 'ResourceConditionalAccumulator', 'DecisionTreeResource')

class _RoundRobinStrategy:
    """Returns the next ps task index for placement in round-robin order.

  This class is not to be used directly by users.  See instead
  `replica_device_setter()` below.
  """

    def __init__(self, num_tasks):
        if False:
            print('Hello World!')
        'Create a new `_RoundRobinStrategy`.\n\n    Args:\n      num_tasks: Number of ps tasks to cycle among.\n    '
        self._num_tasks = num_tasks
        self._next_task = 0

    def __call__(self, unused_op):
        if False:
            print('Hello World!')
        'Choose a ps task index for the given `Operation`.\n\n    Args:\n      unused_op: An `Operation` to be placed on ps.\n\n    Returns:\n      The next ps task index to use for the `Operation`. Returns the next\n      index, in the range `[offset, offset + num_tasks)`.\n    '
        task = self._next_task
        self._next_task = (self._next_task + 1) % self._num_tasks
        return task

class _ReplicaDeviceChooser:
    """Class to choose devices for Ops in a replicated training setup.

  This class is not to be used directly by users.  See instead
  `replica_device_setter()` below.
  """

    def __init__(self, ps_tasks, ps_device, worker_device, merge_devices, ps_ops, ps_strategy):
        if False:
            i = 10
            return i + 15
        'Create a new `_ReplicaDeviceChooser`.\n\n    Args:\n      ps_tasks: Number of tasks in the `ps` job.\n      ps_device: String.  Name of the `ps` job.\n      worker_device: String.  Name of the `worker` job.\n      merge_devices: Boolean. Set to True to allow merging of device specs.\n      ps_ops: List of strings representing `Operation` types that need to be\n        placed on `ps` devices.\n      ps_strategy: A callable invoked for every ps `Operation` (i.e. matched by\n        `ps_ops`), that takes the `Operation` and returns the ps task index to\n        use.\n    '
        self._ps_tasks = ps_tasks
        self._ps_device = ps_device
        self._worker_device = worker_device
        self._merge_devices = merge_devices
        self._ps_ops = ps_ops
        self._ps_strategy = ps_strategy

    def device_function(self, op):
        if False:
            for i in range(10):
                print('nop')
        'Choose a device for `op`.\n\n    Args:\n      op: an `Operation`.\n\n    Returns:\n      The device to use for the `Operation`.\n    '
        if not self._merge_devices and op.device:
            return op.device
        current_device = pydev.DeviceSpec.from_string(op.device or '')
        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if self._ps_tasks and self._ps_device and (node_def.op in self._ps_ops):
            ps_device = pydev.DeviceSpec.from_string(self._ps_device)
            (current_job, ps_job) = (current_device.job, ps_device.job)
            if ps_job and (not current_job or current_job == ps_job):
                ps_device = ps_device.replace(task=self._ps_strategy(op))
            ps_device = ps_device.make_merged_spec(current_device)
            return ps_device.to_string()
        worker_device = pydev.DeviceSpec.from_string(self._worker_device or '')
        worker_device = worker_device.make_merged_spec(current_device)
        return worker_device.to_string()

@tf_export(v1=['train.replica_device_setter'])
def replica_device_setter(ps_tasks=0, ps_device='/job:ps', worker_device='/job:worker', merge_devices=True, cluster=None, ps_ops=None, ps_strategy=None):
    if False:
        return 10
    'Return a `device function` to use when building a Graph for replicas.\n\n  Device Functions are used in `with tf.device(device_function):` statement to\n  automatically assign devices to `Operation` objects as they are constructed,\n  Device constraints are added from the inner-most context first, working\n  outwards. The merging behavior adds constraints to fields that are yet unset\n  by a more inner context. Currently the fields are (job, task, cpu/gpu).\n\n  If `cluster` is `None`, and `ps_tasks` is 0, the returned function is a no-op.\n  Otherwise, the value of `ps_tasks` is derived from `cluster`.\n\n  By default, only Variable ops are placed on ps tasks, and the placement\n  strategy is round-robin over all ps tasks. A custom `ps_strategy` may be used\n  to do more intelligent placement, such as\n  `tf.contrib.training.GreedyLoadBalancingStrategy`.\n\n  For example,\n\n  ```python\n  # To build a cluster with two ps jobs on hosts ps0 and ps1, and 3 worker\n  # jobs on hosts worker0, worker1 and worker2.\n  cluster_spec = {\n      "ps": ["ps0:2222", "ps1:2222"],\n      "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}\n  with\n  tf.compat.v1.device(tf.compat.v1.train.replica_device_setter(cluster=cluster_spec)):\n    # Build your graph\n    v1 = tf.Variable(...)  # assigned to /job:ps/task:0\n    v2 = tf.Variable(...)  # assigned to /job:ps/task:1\n    v3 = tf.Variable(...)  # assigned to /job:ps/task:0\n  # Run compute\n  ```\n\n  Args:\n    ps_tasks: Number of tasks in the `ps` job.  Ignored if `cluster` is\n      provided.\n    ps_device: String.  Device of the `ps` job.  If empty no `ps` job is used.\n      Defaults to `ps`.\n    worker_device: String.  Device of the `worker` job.  If empty no `worker`\n      job is used.\n    merge_devices: `Boolean`. If `True`, merges or only sets a device if the\n      device constraint is completely unset. merges device specification rather\n      than overriding them.\n    cluster: `ClusterDef` proto or `ClusterSpec`.\n    ps_ops: List of strings representing `Operation` types that need to be\n      placed on `ps` devices.  If `None`, defaults to `STANDARD_PS_OPS`.\n    ps_strategy: A callable invoked for every ps `Operation` (i.e. matched by\n      `ps_ops`), that takes the `Operation` and returns the ps task index to\n      use.  If `None`, defaults to a round-robin strategy across all `ps`\n      devices.\n\n  Returns:\n    A function to pass to `tf.device()`.\n\n  Raises:\n    TypeError if `cluster` is not a dictionary or `ClusterDef` protocol buffer,\n    or if `ps_strategy` is provided but not a callable.\n  '
    if cluster is not None:
        if isinstance(cluster, server_lib.ClusterSpec):
            cluster_spec = cluster.as_dict()
        else:
            cluster_spec = server_lib.ClusterSpec(cluster).as_dict()
        ps_job_name = pydev.DeviceSpec.from_string(ps_device).job
        if ps_job_name not in cluster_spec or cluster_spec[ps_job_name] is None:
            return None
        ps_tasks = len(cluster_spec[ps_job_name])
    if ps_tasks == 0:
        return None
    if ps_ops is None:
        ps_ops = list(STANDARD_PS_OPS)
    if not merge_devices:
        logging.warning('DEPRECATION: It is recommended to set merge_devices=true in replica_device_setter')
    if ps_strategy is None:
        ps_strategy = _RoundRobinStrategy(ps_tasks)
    if not callable(ps_strategy):
        raise TypeError('ps_strategy must be callable')
    chooser = _ReplicaDeviceChooser(ps_tasks, ps_device, worker_device, merge_devices, ps_ops, ps_strategy)
    return chooser.device_function