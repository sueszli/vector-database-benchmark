"""The execution context for ClusterCoordinator."""
import contextlib
import threading
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
_dispatch_context = threading.local()

def get_current_dispatch_context():
    if False:
        print('Hello World!')
    try:
        return _dispatch_context.current
    except AttributeError:
        return None

@contextlib.contextmanager
def with_dispatch_context(worker_obj):
    if False:
        while True:
            i = 10
    previous_context = getattr(_dispatch_context, 'current', None)
    _dispatch_context.current = DispatchContext(worker_obj)
    yield
    _dispatch_context.current = previous_context

class DispatchContext(object):
    """Context entered when executing a closure on a given worker."""

    def __init__(self, worker_obj):
        if False:
            print('Hello World!')
        self._worker = worker_obj
        self._worker_index = worker_obj.worker_index

    @property
    def worker(self):
        if False:
            return 10
        return self._worker

    @property
    def worker_index(self):
        if False:
            for i in range(10):
                print('nop')
        return self._worker_index

    def maybe_get_remote_value(self, ret):
        if False:
            for i in range(10):
                print('nop')
        return maybe_get_remote_value(ret)

def maybe_get_remote_value(val):
    if False:
        for i in range(10):
            print('nop')
    'Gets the value of `val` if it is a `RemoteValue`.'
    if isinstance(val, remote_value.RemoteValue):
        error = val._get_error()
        if error:
            raise AssertionError("RemoteValue doesn't have a value because it has error %r:%s" % (error, error))
        elif val._status is not remote_value.RemoteValueStatus.READY:
            raise AssertionError('The input RemoteValue has not been executed.')
        else:
            return val._get_values()
    else:
        return val

@tf_export('distribute.coordinator.experimental_get_current_worker_index', v1=[])
def get_current_worker_index():
    if False:
        print('Hello World!')
    'Returns the current worker index, when called within a worker closure.\n\n  Some parameter server training workloads may require the worker to know its\n  index, for example for data sharding for reduced-variance training.\n\n  This method may be used within a `tf.function` that is executed on a worker.\n  That is, either a `dataset_fn` that runs via\n  `ClusterCoordinator.create_per_worker_dataset`, or any other function\n  scheduled via `ClusterCoordinator.schedule`.\n\n  Example (sharding data by worker):\n\n  ```python\n  strategy = tf.distribute.ParameterServerStrategy(\n      cluster_resolver=...)\n  coordinator = (\n      tf.distribute.coordinator.ClusterCoordinator(strategy))\n\n  def dataset_fn(context):\n    dataset = tf.data.Dataset.range(10)\n    worker_index = (\n        tf.distribute.coordinator.experimental_get_current_worker_index()\n    )\n    dataset = dataset.shard(\n        num_shards=num_workers,\n        index=worker_index,\n    )\n    return dataset\n\n  @tf.function\n  def per_worker_dataset_fn():\n    return strategy.distribute_datasets_from_function(dataset_fn)\n\n  per_worker_dataset = coordinator.create_per_worker_dataset(\n      per_worker_dataset_fn)\n  ```\n\n  Raises:\n    RuntimeError: if called from outside a `tf.function` or outside of a remote\n      closure execution context (that is, on a non-worker machine).\n  '
    msg = 'Cannot retrieve the worker index. `get_worker_idx_and_num_workers` should be called from within a tf.function being executed on a worker. This method should only be called from either a dataset_fn that is passed into `ClusterCoordinator.create_per_worker_dataset`, or a tf.function that is passed into `ClusterCoordinator.schedule`.'
    if not ops.inside_function():
        raise RuntimeError(msg)

    def call_time_worker_index():
        if False:
            while True:
                i = 10
        dispatch_context = get_current_dispatch_context()
        if not dispatch_context:
            raise RuntimeError(msg)
        return dispatch_context.worker_index
    worker_index = ops.get_default_graph().capture_call_time_value(call_time_worker_index, tensor.TensorSpec([], dtype=dtypes.int64))
    worker_index.op._set_attr('_user_specified_name', attr_value_pb2.AttrValue(s=compat.as_bytes('worker_index')))
    return worker_index