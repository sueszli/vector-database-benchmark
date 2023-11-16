"""Propagates information about tensor layouts across operations."""
import contextlib
import logging
import threading
from typing import Any, List, Sequence, Set
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python import _pywrap_dtensor_device
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.util import _pywrap_utils
_next_device_number = 0
_next_device_number_lock = threading.Lock()

class DTensorDevice(object):
    """Wraps a custom device which attempts to propagate tensor layouts."""

    def __init__(self, meshes: List[layout_lib.Mesh], is_async=True, in_flight_nodes_limit=8):
        if False:
            for i in range(10):
                print('nop')
        'Create a new DTensorDevice which executes ops on `underlying_device`.\n\n    Args:\n      meshes: A list of `Mesh` objects indicating groups of devices to execute\n        on. These may also be registered lazily.\n      is_async: Indicates whether DTensor operations on this client will return\n        immediately (with "non-ready" handles) or block until executed. This is\n        on by default and is exposed as an option for ease of debugging.\n      in_flight_nodes_limit: Indicates the limit of in-flight nodes before\n        enqueueing of async operations to DTensorDevice is blocked. This limit\n        is per mesh. 0 for no limits from DTensor. Default is 8.\n    '
        if any((not isinstance(mesh, layout_lib.Mesh) for mesh in meshes)):
            raise TypeError('Expected a flat list of Mesh objects, got {}'.format(meshes))
        global _next_device_number
        ctx = context.context()
        with _next_device_number_lock:
            self.name = '{}/device:CUSTOM:{}'.format(ctx.host_address_space(), _next_device_number)
            _next_device_number += 1
        (device, device_info) = _pywrap_dtensor_device.Allocate(self.name, is_async, in_flight_nodes_limit)
        context.register_custom_device(device, self.name, device_info)
        self._device_info = device_info
        self._current_output_layout = None
        self._current_default_mesh = None
        self._meshes = set()
        self._mesh_lock = threading.Lock()
        for mesh in meshes:
            self._register_mesh(mesh)

    def _create_host_array(self, shape, host_id):
        if False:
            print('Hello World!')
        'Returns ID and device lists that can be used to create a host mesh.'
        num_global_devices = np.prod(shape)
        global_device_ids = np.arange(num_global_devices).reshape(shape)
        local_device_list = [tf_device.DeviceSpec(job=config.full_job_name(), device_type='CPU', device_index=0)]
        num_local_devices = len(local_device_list)
        local_device_ids = [x + host_id * num_local_devices for x in range(num_local_devices)]
        return (global_device_ids, local_device_ids, local_device_list)

    def _register_mesh(self, mesh: layout_lib.Mesh):
        if False:
            return 10
        'Idempotently register `mesh` with the dtensor device.'
        with self._mesh_lock:
            if mesh not in self._meshes:
                _pywrap_dtensor_device.AddMesh(self._device_info, mesh.to_string(), False)
                self._meshes.add(mesh)
                if mesh.device_type().upper() == 'TPU':
                    logging.info('Registering virtual 1:1 mapped host mesh %s for mesh %s', mesh.host_mesh().to_string(), mesh.to_string())
                    _pywrap_dtensor_device.AddMesh(self._device_info, mesh.host_mesh().to_string(), True)
                    self._meshes.add(mesh.host_mesh())

    @property
    def meshes(self) -> Set[layout_lib.Mesh]:
        if False:
            print('Hello World!')
        return self._meshes

    def pack(self, tensors: Sequence[Any], layout: layout_lib.Layout) -> Any:
        if False:
            return 10
        'Packs tensors into a DTensor handle on this DTensor device.\n\n    Packing and unpacking are inverse operations:\n\n    ```\n    * unpack(pack(tensors)) == tensors\n    * pack(unpack(dtensor)) == dtensor\n    ```\n\n    Refer to `dtensor.pack` for more information.\n\n    Args:\n      tensors: The list of tensors to pack into a DTensor.\n      layout: The layout of the DTensor to be created.\n\n    Returns:\n      A DTensor created from the individual component tensors.\n\n    Raises:\n      RuntimeError: When not called eagerly.\n    '
        if not context.executing_eagerly():
            raise RuntimeError('`pack` must be called eagerly.')
        self._register_mesh(layout.mesh)
        with ops.device(self.name):
            if all((isinstance(t, sparse_tensor.SparseTensor) for t in tensors)):
                if not all((t.shape == tensors[0].shape for t in tensors)):
                    raise TypeError('All input SparseTensors to Pack must be same shape.')
                is_sparse = True
                tensors = [t.indices for t in tensors] + [t.values for t in tensors] + [ops.convert_to_tensor(t.shape, dtype=dtypes.int64) for t in tensors]
            elif any((isinstance(t, sparse_tensor.SparseTensor) for t in tensors)):
                raise TypeError('Cannot Pack SparseTensors with Tensors.')
            else:
                is_sparse = False
            try:
                return _pywrap_dtensor_device.Pack(context.context()._handle, tensors, layout.to_string(), self._device_info, is_sparse)
            except core._NotOkStatusException as e:
                raise core._status_to_exception(e) from None

    def unpack(self, dtensor: Any) -> Sequence[Any]:
        if False:
            i = 10
            return i + 15
        'Unpacks a DTensor handle on this DTensor device.\n\n    Packing and unpacking are inverse operations:\n\n    ```\n    * unpack(pack(tensors)) == tensors\n    * pack(unpack(dtensor)) == dtensor\n    ```\n\n    Refer to `dtensor.unpack` for more information.\n\n    Args:\n      dtensor: The DTensor to unpack.\n\n    Returns:\n      The raw underlying tensor components of the DTensor.\n\n    Raises:\n      RuntimeError: When not called eagerly.\n    '
        if not context.executing_eagerly():
            raise RuntimeError('`unpack` must be called eagerly.')
        try:
            tensors = _pywrap_dtensor_device.Unpack(context.context()._handle, dtensor, self._device_info)
        except core._NotOkStatusException as e:
            raise core._status_to_exception(e) from None
        is_sparse = _pywrap_dtensor_device.IsSparseDTensor(context.context()._handle, dtensor, self._device_info)
        if is_sparse:
            result = []
            for i in range(len(tensors) // 3):
                result.append(sparse_tensor.SparseTensor(tensors[i], tensors[i + len(tensors) // 3], tensors[i + 2 * len(tensors) // 3]))
            return result
        else:
            return tensors

    def fetch_layout(self, dtensor: Any) -> layout_lib.Layout:
        if False:
            print('Hello World!')
        'Fetches the layout of the DTensor.\n\n    Args:\n      dtensor: The DTensor whose layout is to be fetched.\n\n    Returns:\n      The `Layout` of this DTensor.\n\n    Raises:\n      RuntimeError: When not called eagerly.\n    '
        if not context.executing_eagerly():
            raise RuntimeError('`fetch_layout` must be called eagerly.')
        if _pywrap_utils.IsVariable(dtensor):
            dtensor = dtensor.read_value()
        try:
            layout_string = _pywrap_dtensor_device.FetchLayout(context.context()._handle, dtensor, self._device_info)
        except core._NotOkStatusException as e:
            raise core._status_to_exception(e) from None
        if layout_string is None:
            return None
        return layout_lib.Layout.from_string(layout_string)

    def is_dtensor(self, tensor: Any) -> bool:
        if False:
            return 10
        'Check whether the input tensor is a DTensor.\n\n    In Python, a DTensor has the same type as a `tf.Tensor`. This method will\n    let you check and handle the tensor differently if a tf.Tensor is a DTensor.\n\n    Args:\n      tensor: an object to be checked.\n\n    Returns:\n      bool, True if the given tensor is a DTensor.\n\n    Raises:\n      RuntimeError: When not called eagerly.\n    '
        if not context.executing_eagerly():
            raise RuntimeError('`is_dtensor` must be called eagerly.')
        if not tensor_util.is_tensor(tensor):
            return False
        if _pywrap_utils.IsVariable(tensor):
            tensor = tensor._handle
        return _pywrap_dtensor_device.IsDTensor(context.context()._handle, tensor, self._device_info)

    def set_tpu_core_ids(self, mesh_name, tpu_core_ids):
        if False:
            print('Hello World!')
        'Sets the singleton global device ID-to-physical core ID map.\n\n    Args:\n      mesh_name: The name of a mesh. If empty, set the default mapping.\n      tpu_core_ids: TPU core IDs sorted by TF task/device ordinal.\n    '
        _pywrap_dtensor_device.SetTPUCoreIDs(self._device_info, mesh_name, tpu_core_ids)

    def clear_tpu_core_ids(self):
        if False:
            return 10
        _pywrap_dtensor_device.ClearTPUCoreIDs(self._device_info)

    def tpu_core_ids_to_locations(self, tpu_core_ids):
        if False:
            while True:
                i = 10
        'Translates TPU core IDs to TPU core locations.\n\n    Args:\n      tpu_core_ids: A list of TPU core IDs. Each one is an unsigned integer.\n\n    Returns:\n      A list of corresponding TPU core locations.\n    '
        return _pywrap_dtensor_device.TPUCoreIDsToLocations(context.context()._handle, self._device_info, tpu_core_ids)

    def tpu_core_locations_to_ids(self, tpu_core_locations):
        if False:
            for i in range(10):
                print('nop')
        'Translates TPU core locations to TPU core IDs.\n\n    Args:\n      tpu_core_locations: A list of TPU core locations. Each one is a list of\n        four unsigned integers, [x, y, z, core].\n\n    Returns:\n      A list of corresponding TPU core IDs.\n    '
        return _pywrap_dtensor_device.TPUCoreLocationsToIDs(context.context()._handle, self._device_info, tpu_core_locations)

    def _get_stats(self):
        if False:
            return 10
        "Returns the number of cache hit and miss for function compilation.\n\n    Returns:\n      A dictionary.\n        'miss': number of cache misses;\n        'hit': number of cache hits; and\n        'size': size of cache;\n      miss count.\n    "
        return _pywrap_dtensor_device.GetStats(context.context()._handle, self._device_info)

    def set_iterator_element_layouts(self, iterator_resource_dtensor, layouts: List[layout_lib.Layout]):
        if False:
            i = 10
            return i + 15
        'Sets the element layouts on an iterator resource tensor.\n\n    Args:\n      iterator_resource_dtensor: a DTensor created by packing the individiual\n        iterator resource tensors.\n      layouts: the flattened list of layouts to be applied to the elements\n        emitted by the iterator resource DTensor.\n    '
        _pywrap_dtensor_device.SetIteratorElementLayouts(context.context()._handle, iterator_resource_dtensor, [layout.to_string() for layout in layouts], self._device_info)

    @contextlib.contextmanager
    def _experimental_default_mesh(self, mesh: layout_lib.Mesh):
        if False:
            for i in range(10):
                print('nop')
        'Sets a default mesh for all ops in the scope.\n\n    Note: This is an internal helper method, which is not user facing api.\n\n    Useful for requesting a specific mesh for ops which would have no inferred\n    layout, e.g. tf.zeros.\n\n    Args:\n      mesh: A Mesh to be used for ops without Mesh.\n\n    Yields:\n      Nothing.\n    '
        previous_default = self._current_default_mesh
        self._register_mesh(mesh)
        _pywrap_dtensor_device.ExperimentalSetDefaultMesh(self._device_info, mesh.to_string().encode('utf-8'))
        self._current_default_mesh = mesh
        yield
        _pywrap_dtensor_device.ExperimentalClearDefaultMesh(self._device_info)
        if previous_default:
            _pywrap_dtensor_device.ExperimentalSetDefaultMesh(self._device_info, previous_default.to_string().encode('utf-8'))
        self._current_default_mesh = previous_default

    @contextlib.contextmanager
    def _default_layout(self, layout: layout_lib.Layout):
        if False:
            while True:
                i = 10
        'Sets a default output layout for all ops in the scope.\n\n    Note: This is an internal helper method, which is not user facing api.\n\n    Useful for requesting a specific layout for ops which would have no inferred\n    layout, e.g. tf.zeros.\n\n    Caveats:\n\n    - Currently only affects the first output of an op. For Op with multiple\n      outputs, this does not support yet.\n\n    - All Ops in the scope will be attached with the same layout. This might not\n      be valid as the rank is different. The current suggestion is: Try to wrap\n      the raw op wheneven possible.\n\n    Args:\n      layout: A Layout for the outputs of all operations in this scope.\n\n    Yields:\n      Nothing.\n    '
        previous_default = None
        previous_graph_size = None
        graph = None
        self._register_mesh(layout.mesh)
        try:
            previous_default = self._current_output_layout
            self._current_output_layout = layout.to_string().encode('utf-8')
            _pywrap_dtensor_device.ExperimentalSetDefaultLayout(self._device_info, self._current_output_layout)
            if context.executing_eagerly():
                with ops.device(self.name):
                    yield
            else:
                graph = ops.get_default_graph()
                previous_graph_size = len(graph.get_operations())
                yield
        finally:
            if graph is not None:
                for operation in graph.get_operations()[previous_graph_size:]:
                    operation._set_attr('_layout', attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(s=[self._current_output_layout])))
                    operation._set_attr('_mesh', attr_value_pb2.AttrValue(s=layout.mesh.to_string().encode('utf-8')))
            self._current_output_layout = previous_default
            if self._current_output_layout is None:
                _pywrap_dtensor_device.ExperimentalClearDefaultLayout(self._device_info)
            else:
                _pywrap_dtensor_device.ExperimentalSetDefaultLayout(self._device_info, self._current_output_layout.decode('utf-8'))