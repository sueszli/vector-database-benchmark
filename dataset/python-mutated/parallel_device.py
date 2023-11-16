"""Utility for eagerly executing operations in parallel on multiple devices."""
import threading
import weakref
from tensorflow.python import _pywrap_parallel_device
from tensorflow.python.distribute import device_util
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
_next_device_number = 0
_next_device_number_lock = threading.Lock()
_all_parallel_devices = weakref.WeakValueDictionary()

def unpack(tensor):
    if False:
        i = 10
        return i + 15
    "Finds `tensor`'s parallel device and unpacks its components."
    parallel_device = _all_parallel_devices.get(tensor.device, None)
    if parallel_device is None:
        raise ValueError('{} is not a parallel device'.format(tensor.device))
    return parallel_device.unpack(tensor)

class ParallelDevice(object):
    """A device which executes operations in parallel."""

    def __init__(self, components):
        if False:
            for i in range(10):
                print('nop')
        'Creates a device which executes operations in parallel on `components`.\n\n    Args:\n      components: A list of device names. Each operation executed on the\n        returned device executes on these component devices.\n\n    Returns:\n      A string with the name of the newly created device.\n    '
        global _next_device_number, _next_device_number_lock
        self.components = tuple((device_util.canonicalize(d) for d in components))
        if not self.components:
            raise ValueError('ParallelDevice requires at least one component.')
        ctx = context.context()
        with _next_device_number_lock:
            self._name = '{}/device:CUSTOM:{}'.format(ctx.host_address_space(), _next_device_number)
            _next_device_number += 1
        (device, device_info) = _pywrap_parallel_device.GetParallelDeviceCapsules(self._name, self.components)
        context.register_custom_device(device, self._name, device_info)
        self._device_ids = None
        self._device_scope = None
        _all_parallel_devices[self._name] = self

    def _pack_tensor(self, *tensors):
        if False:
            while True:
                i = 10
        'Helper to pack plain-old-tensors, not structures or composites.'
        for tensor in tensors:
            if not isinstance(tensor, (tensor_lib.Tensor, composite_tensor.CompositeTensor, variables.Variable)):
                raise ValueError('Every component to pack onto the ParallelDevice must already be a tensor, got {}. Consider running `tf.constant` or `tf.convert_to_tensor` first on literal values.'.format(tensors))
        with ops.device(self._name):
            return tpu_ops.tpu_replicated_input(inputs=tensors)

    def pack(self, tensors):
        if False:
            return 10
        'Create a tensor on the parallel device from a sequence of tensors.\n\n    Args:\n      tensors: A list of tensors, one per device in `self.components`. The list\n        can contain composite tensors and nests (lists, dicts, etc. supported by\n        `tf.nest`) with the same structure for each device, but every component\n        of nests must already be a `tf.Tensor` or composite. Passing\n        `tf.Variable` objects reads their value, it does not share a mutable\n        reference between the packed and unpacked forms.\n\n    Returns:\n      A tensor placed on the ParallelDevice. For nested structures, returns a\n      single structure containing tensors placed on the ParallelDevice (same\n      structure as each component of `tensors`).\n\n    Raises:\n      ValueError: If the length of `tensors` does not match the number of\n        component devices, or if there are non-tensor inputs.\n\n    '
        self._assert_eager()
        if len(tensors) != len(self.components):
            raise ValueError('Creating a parallel tensor requires one tensor per component. Got {} but was expecting {}.'.format(len(tensors), len(self.components)))
        with ops.device(None):
            tensors = variable_utils.convert_variables_to_tensors(tensors)
        return nest.map_structure(self._pack_tensor, *tensors, expand_composites=True)

    def _unpack_tensor(self, parallel_tensor):
        if False:
            i = 10
            return i + 15
        'Helper to unpack a single tensor.'
        if not isinstance(parallel_tensor, (tensor_lib.Tensor, composite_tensor.CompositeTensor, variables.Variable)):
            raise ValueError('Expected a tensor, got {}.'.format(parallel_tensor))
        with ops.device(self._name):
            return tpu_ops.tpu_replicated_output(parallel_tensor, num_replicas=len(self.components))

    def unpack(self, parallel_tensor):
        if False:
            return 10
        'Unpack a parallel tensor into its components.\n\n    Args:\n      parallel_tensor: A tensor, composite tensor, or `tf.nest` of such placed\n        on the ParallelDevice. Passing `tf.Variable` objects reads their value,\n        it does not share a mutable reference between the packed and unpacked\n        forms.\n\n    Returns:\n      A list with the same length as `self.components` each with the same\n      structure as `parallel_tensor`, containing component tensors.\n\n    '
        self._assert_eager()
        unpacked_components = [[] for _ in range(len(self.components))]
        with ops.device(self._name):
            parallel_tensor = variable_utils.convert_variables_to_tensors(parallel_tensor)
        for tensor in nest.flatten(parallel_tensor, expand_composites=True):
            for (accumulator, unpacked_tensor) in zip(unpacked_components, self._unpack_tensor(tensor)):
                accumulator.append(unpacked_tensor)
        return [nest.pack_sequence_as(parallel_tensor, unpacked, expand_composites=True) for unpacked in unpacked_components]

    @property
    def device_ids(self):
        if False:
            print('Hello World!')
        'A parallel tensor with scalar integers numbering component devices.\n\n    Each device ID is placed on its corresponding device, in the same order as\n    the `components` constructor argument.\n\n    Returns:\n      A parallel tensor containing 0 on the first device, 1 on the second, etc.\n    '
        if self._device_ids is None:
            with ops.init_scope():
                device_ids_list = []
                for (index, device) in enumerate(self.components):
                    with ops.device(device):
                        device_ids_list.append(array_ops.identity(constant_op.constant(index)))
                self._device_ids = self.pack(device_ids_list)
        return self._device_ids

    def _assert_eager(self):
        if False:
            return 10
        'Verifies that tracing is not active.'
        if not context.executing_eagerly():
            raise NotImplementedError('ParallelDevice is currently not supported inside `tf.function`. It can however run calls to a `tf.function` in parallel:\n\nwith ParallelDevice() as p:\n  f()')

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        'Runs ops in parallel, makes variables which save independent buffers.'
        if self._device_scope is not None:
            raise AssertionError('Re-entered a ParallelDevice scope without first exiting it.')
        self._assert_eager()
        self._device_scope = ops.device(self._name)
        self._device_scope.__enter__()
        return self

    def __exit__(self, typ, exc, tb):
        if False:
            i = 10
            return i + 15
        self._device_scope.__exit__(typ, exc, tb)
        self._device_scope = None