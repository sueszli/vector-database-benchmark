"""Tensor Handle Operations."""
import numpy as np
from tensorflow.core.framework import resource_handle_pb2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export

def encode_resource_handle(resource_handle):
    if False:
        return 10
    'Encode a ResourceHandle proto as custom numpy struct type.'
    return np.asarray(bytearray(resource_handle.SerializeToString()), dtype=dtypes.np_resource)

class TensorHandle:
    """Represents a handle for a live tensor in a session."""

    def __init__(self, handle, dtype, session):
        if False:
            print('Hello World!')
        'Constructs a new tensor handle.\n\n    A tensor handle for a persistent tensor is a python string\n    that has the form of "tensor_name;unique_id;device_name".\n\n    Args:\n      handle: A tensor handle.\n      dtype: The data type of the tensor represented by `handle`.\n      session: The session in which the tensor is produced.\n    '
        self._handle = compat.as_str_any(handle)
        self._resource_handle = None
        self._dtype = dtype
        self._session = session
        self._auto_gc_enabled = True

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        if self._auto_gc_enabled:
            self._session._register_dead_handle(self.handle)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._handle

    def _get_resource_handle(self):
        if False:
            while True:
                i = 10
        'The ResourceHandle representation of this handle.'
        if not self._resource_handle:
            self._resource_handle = resource_handle_pb2.ResourceHandleProto()
            self._resource_handle.device = self._handle.split(';')[-1]
            self._resource_handle.container = pywrap_tf_session.TENSOR_HANDLE_KEY
            self._resource_handle.name = self._handle
        return self._resource_handle

    def to_numpy_array(self):
        if False:
            for i in range(10):
                print('nop')
        'Convert a TensorHandle object to a feedable numpy value.\n\n    Returns:\n      A numpy array of a custom struct type that can be used as a feed value\n      to run().\n    '
        return encode_resource_handle(self._get_resource_handle())

    @property
    def handle(self):
        if False:
            return 10
        'The string representation of this handle.'
        return self._handle

    def eval(self):
        if False:
            print('Hello World!')
        'Return the value of the tensor represented by this handle.'
        if not self._auto_gc_enabled:
            raise TypeError('Persistent tensor %s may have already been deleted.' % self.handle)
        (holder, reader) = _get_handle_reader(self._session.graph, self._handle, self._dtype)
        return self._session.run(reader, feed_dict={holder: self._handle})

    def delete(self):
        if False:
            i = 10
            return i + 15
        'Force the deletion of this persistent tensor.'
        if not self._auto_gc_enabled:
            raise TypeError('Persistent tensor %s may have already been deleted.' % self.handle)
        self._auto_gc_enabled = False
        (holder, deleter) = _get_handle_deleter(self._session.graph, 0, self._handle)
        self._session.run(deleter, feed_dict={holder: self.handle})

    def get_raw_handle(self):
        if False:
            print('Hello World!')
        'Return the raw handle of the tensor.\n\n    Note that the method disables the automatic garbage collection of this\n    persistent tensor. The caller is now responsible for managing the life\n    time of the tensor.\n    '
        self._auto_gc_enabled = False
        return self._handle

    @staticmethod
    def _get_device_name(handle):
        if False:
            return 10
        'The device name encoded in the handle.'
        handle_str = compat.as_str_any(handle)
        return pydev.canonical_name(handle_str.split(';')[-1])

    @staticmethod
    def _get_reader_key(handle):
        if False:
            for i in range(10):
                print('nop')
        'The graph key for reader.'
        handle_parts = str(handle).split(';')
        return handle_parts[0] + ';' + handle_parts[-1]

    @staticmethod
    def _get_mover_key(feeder, handle):
        if False:
            i = 10
            return i + 15
        'The graph key for mover.'
        return feeder.op.name + ';' + TensorHandle._get_reader_key(handle)

@tf_export(v1=['get_session_handle'])
def get_session_handle(data, name=None):
    if False:
        return 10
    'Return the handle of `data`.\n\n  This is EXPERIMENTAL and subject to change.\n\n  Keep `data` "in-place" in the runtime and create a handle that can be\n  used to retrieve `data` in a subsequent run().\n\n  Combined with `get_session_tensor`, we can keep a tensor produced in\n  one run call in place, and use it as the input in a future run call.\n\n  Args:\n    data: A tensor to be stored in the session.\n    name: Optional name prefix for the return tensor.\n\n  Returns:\n    A scalar string tensor representing a unique handle for `data`.\n\n  Raises:\n    TypeError: if `data` is not a Tensor.\n\n  Example:\n\n  ```python\n  c = tf.multiply(a, b)\n  h = tf.compat.v1.get_session_handle(c)\n  h = sess.run(h)\n\n  p, a = tf.compat.v1.get_session_tensor(h.handle, tf.float32)\n  b = tf.multiply(a, 10)\n  c = sess.run(b, feed_dict={p: h.handle})\n  ```\n\n  '
    if not isinstance(data, tensor_lib.Tensor):
        raise TypeError('`data` must be of type Tensor.')
    with ops.colocate_with(data):
        return gen_data_flow_ops.get_session_handle(data, name=name)

@tf_export(v1=['get_session_tensor'])
def get_session_tensor(handle, dtype, name=None):
    if False:
        print('Hello World!')
    'Get the tensor of type `dtype` by feeding a tensor handle.\n\n  This is EXPERIMENTAL and subject to change.\n\n  Get the value of the tensor from a tensor handle. The tensor\n  is produced in a previous run() and stored in the state of the\n  session.\n\n  Args:\n    handle: The string representation of a persistent tensor handle.\n    dtype: The type of the output tensor.\n    name: Optional name prefix for the return tensor.\n\n  Returns:\n    A pair of tensors. The first is a placeholder for feeding a\n    tensor handle and the second is the tensor in the session state\n    keyed by the tensor handle.\n\n  Example:\n\n  ```python\n  c = tf.multiply(a, b)\n  h = tf.compat.v1.get_session_handle(c)\n  h = sess.run(h)\n\n  p, a = tf.compat.v1.get_session_tensor(h.handle, tf.float32)\n  b = tf.multiply(a, 10)\n  c = sess.run(b, feed_dict={p: h.handle})\n  ```\n\n  '
    handle_device = TensorHandle._get_device_name(handle)
    with ops.device(handle_device):
        holder = array_ops.placeholder(dtypes.string)
        _register_handle_feeder(holder.graph, holder, dtype)
        tensor = gen_data_flow_ops.get_session_tensor(holder, dtype, name=name)
    return (holder, tensor)

@tf_export(v1=['delete_session_tensor'])
def delete_session_tensor(handle, name=None):
    if False:
        return 10
    'Delete the tensor for the given tensor handle.\n\n  This is EXPERIMENTAL and subject to change.\n\n  Delete the tensor of a given tensor handle. The tensor is produced\n  in a previous run() and stored in the state of the session.\n\n  Args:\n    handle: The string representation of a persistent tensor handle.\n    name: Optional name prefix for the return tensor.\n\n  Returns:\n    A pair of graph elements. The first is a placeholder for feeding a\n    tensor handle and the second is a deletion operation.\n  '
    handle_device = TensorHandle._get_device_name(handle)
    with ops.device(handle_device):
        holder = array_ops.placeholder(dtypes.string)
        deleter = gen_data_flow_ops.delete_session_tensor(holder, name=name)
    return (holder, deleter)

def _register_handle_feeder(graph, feeder, dtype):
    if False:
        return 10
    graph._handle_feeders[feeder.op.name] = dtype

def _get_handle_feeder(graph, feeder):
    if False:
        print('Hello World!')
    return graph._handle_feeders.get(feeder.op.name)

def _get_handle_reader(graph, handle, dtype):
    if False:
        while True:
            i = 10
    'Return a read subgraph for this handle.'
    graph_key = TensorHandle._get_reader_key(handle)
    result = graph._handle_readers.get(graph_key)
    if result is None:
        handle_device = TensorHandle._get_device_name(handle)
        with graph.as_default(), graph.device(handle_device):
            holder = array_ops.placeholder(dtypes.string)
            _register_handle_feeder(holder.graph, holder, dtype)
            reader = gen_data_flow_ops.get_session_tensor(holder, dtype)
        result = (holder, reader)
        graph._handle_readers[graph_key] = result
    return result

def _get_handle_mover(graph, feeder, handle):
    if False:
        for i in range(10):
            print('nop')
    'Return a move subgraph for this pair of feeder and handle.'
    dtype = _get_handle_feeder(graph, feeder)
    if dtype is None:
        return None
    handle_device = TensorHandle._get_device_name(handle)
    if feeder.op.device == handle_device:
        return None
    graph_key = TensorHandle._get_mover_key(feeder, handle)
    result = graph._handle_movers.get(graph_key)
    if result is None:
        (holder, reader) = _get_handle_reader(graph, handle, dtype)
        with graph.as_default(), graph.device(feeder.op.device):
            mover = gen_data_flow_ops.get_session_handle(reader)
        result = (holder, mover)
        graph._handle_movers[graph_key] = result
    return result

def _get_handle_deleter(graph, deleter_key, handle):
    if False:
        while True:
            i = 10
    'Return a deletion subgraph for this handle.'
    result = graph._handle_deleters.get(deleter_key)
    if result is None:
        handle_device = TensorHandle._get_device_name(handle)
        with graph.as_default(), graph.device(handle_device):
            holder = array_ops.placeholder(dtypes.string)
            deleter = gen_data_flow_ops.delete_session_tensor(holder)
        result = (holder, deleter)
        graph._handle_deleters[deleter_key] = result
    return result