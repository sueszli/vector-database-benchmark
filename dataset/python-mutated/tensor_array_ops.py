"""TensorArray: a dynamically sized array of Tensors."""
import contextlib
import traceback
import weakref
import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import trace
from tensorflow.python.util import tf_should_use
from tensorflow.python.util.tf_export import tf_export

class _GraphTensorArray:
    """Graph-mode implementation of TensorArray."""

    def __init__(self, dtype, size=None, dynamic_size=None, clear_after_read=None, tensor_array_name=None, handle=None, flow=None, infer_shape=True, element_shape=None, colocate_with_first_write_call=True, name=None):
        if False:
            return 10
        'Constructs a graph mode TensorArray.\n\n    Args:\n      dtype: (required) data type of the TensorArray.\n      size: (optional) int32 scalar `Tensor`: the size of the TensorArray.\n        Required if handle is not provided.\n      dynamic_size: (optional) Python bool: If true, writes to the TensorArray\n        can grow the TensorArray past its initial size.  Default: False.\n      clear_after_read: Boolean (optional, default: True).  If True, clear\n        TensorArray values after reading them.  This disables read-many\n        semantics, but allows early release of memory.\n      tensor_array_name: (optional) Python string: the name of the TensorArray.\n        This is used when creating the TensorArray handle.  If this value is\n        set, handle should be None.\n      handle: (optional) A `Tensor` handle to an existing TensorArray.  If this\n        is set, tensor_array_name should be None. Only supported in graph mode.\n      flow: (optional) A float `Tensor` scalar coming from an existing\n        `TensorArray.flow`. Only supported in graph mode.\n      infer_shape: (optional, default: True) If True, shape inference is\n        enabled.  In this case, all elements must have the same shape.\n      element_shape: (optional, default: None) A `TensorShape` object specifying\n        the shape constraints of each of the elements of the TensorArray. Need\n        not be fully defined.\n      colocate_with_first_write_call: If `True`, the TensorArray will be\n        colocated on the same device as the Tensor used on its first write\n        (write operations include `write`, `unstack`, and `split`).  If `False`,\n        the TensorArray will be placed on the device determined by the device\n        context available during its initialization.\n      name: A name for the operation (optional).\n\n    Raises:\n      ValueError: if both handle and tensor_array_name are provided.\n      TypeError: if handle is provided but is not a Tensor.\n    '
        if handle is not None and tensor_array_name:
            raise ValueError('Cannot provide both `handle` and `tensor_array_name` arguments at the same time.')
        if handle is not None and (not isinstance(handle, tensor_lib.Tensor)):
            raise TypeError(f'Expected `handle` to be a Tensor, but got `{handle}` of type `{type(handle)}` instead.')
        if handle is None and size is None:
            raise ValueError('Argument `size` must be provided if handle is not provided.')
        if handle is not None and size is not None:
            raise ValueError('Cannot provide both a `handle` and `size` arguments at the same time.')
        if handle is not None and element_shape is not None:
            raise ValueError('Cannot provide both `handle` and `element_shape` arguments at the same time.')
        if handle is not None and dynamic_size is not None:
            raise ValueError('Cannot provide both `handle` and `dynamic_size` arguments at the same time.')
        if handle is not None and clear_after_read is not None:
            raise ValueError('Cannot provide both `handle` and `clear_after_read` arguments at the same time.')
        if clear_after_read is None:
            clear_after_read = True
        self._dynamic_size = dynamic_size or False
        self._dtype = dtypes.as_dtype(dtype).base_dtype
        self._colocate_with_first_write_call = colocate_with_first_write_call
        if colocate_with_first_write_call:
            self._colocate_with = []
        else:
            self._colocate_with = None
        self._element_shape = [tensor_shape.as_shape(element_shape)]
        self._infer_shape = infer_shape
        self._size = size
        with ops.name_scope(name, 'TensorArray', [handle, size, flow]) as scope:
            if handle is not None:
                self._handle = handle
                if flow is None:
                    raise ValueError('flow must not be None if handle is not None.')
                self._flow = flow
            else:

                def create():
                    if False:
                        return 10
                    'Create the TensorArray op.'
                    return gen_data_flow_ops.tensor_array_v3(dtype=dtype, size=size, element_shape=element_shape, identical_element_shapes=infer_shape, dynamic_size=self._dynamic_size, clear_after_read=clear_after_read, tensor_array_name=tensor_array_name, name=scope)
                if colocate_with_first_write_call:
                    with ops.device(None), ops.colocate_with(None, ignore_existing=True):
                        (self._handle, self._flow) = create()
                else:
                    (self._handle, self._flow) = create()

    @property
    def flow(self):
        if False:
            print('Hello World!')
        return self._flow

    @property
    def dtype(self):
        if False:
            for i in range(10):
                print('nop')
        return self._dtype

    @property
    def handle(self):
        if False:
            for i in range(10):
                print('nop')
        return self._handle

    @property
    def element_shape(self):
        if False:
            for i in range(10):
                print('nop')
        return self._element_shape[0]

    def _check_element_shape(self, shape):
        if False:
            i = 10
            return i + 15
        'Changes the element shape of the array given a shape to merge with.\n\n    Args:\n      shape: A `TensorShape` object to merge with.\n\n    Raises:\n      ValueError: if the provided shape is incompatible with the current\n          element shape of the `TensorArray`.\n    '
        if not shape.is_compatible_with(self.element_shape):
            raise ValueError('Inconsistent shapes: saw %s but expected %s ' % (shape, self.element_shape))
        if self._infer_shape:
            self._element_shape[0] = self.element_shape.merge_with(shape)

    @contextlib.contextmanager
    def _maybe_colocate_with(self, value):
        if False:
            i = 10
            return i + 15
        'Colocate operations with an internal colocation group or `value`.\n\n    Args:\n      value: `Tensor`, the tensor to try to colocate with.\n\n    Yields:\n      Does not yield anything, but the new context is a colocation context.\n\n    If no internal colocation group is set, colocate with `value` and set\n    the internal colocation group to be value.\n    '
        if not self._colocate_with_first_write_call:
            yield
        else:
            if not self._colocate_with:
                self._colocate_with.append(value)
            with ops.colocate_with(self._colocate_with[0]):
                yield

    def identity(self):
        if False:
            for i in range(10):
                print('nop')
        'See TensorArray.'
        flow = array_ops.identity(self._flow)
        return build_ta_with_new_flow(self, flow)

    def grad(self, source, flow=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        'See TensorArray.'
        if flow is None:
            flow = self.flow
        with ops.name_scope(name, 'TensorArrayGrad', [self._handle]):
            with ops.colocate_with(self._handle):
                (g_handle, unused_flow) = gen_data_flow_ops.tensor_array_grad_v3(handle=self._handle, source=source, flow_in=flow, name=name)
                with ops.control_dependencies([g_handle]):
                    flow = array_ops.identity(flow, name='gradient_flow')
                g = TensorArray(dtype=self._dtype, handle=g_handle, flow=flow, infer_shape=self._infer_shape, colocate_with_first_write_call=False)
                g._implementation._element_shape = self._element_shape
                return g

    def read(self, index, name=None):
        if False:
            while True:
                i = 10
        'See TensorArray.'
        value = gen_data_flow_ops.tensor_array_read_v3(handle=self._handle, index=index, flow_in=self._flow, dtype=self._dtype, name=name)
        if self._element_shape:
            value.set_shape(self._element_shape[0].dims)
        return value

    def write(self, index, value, name=None):
        if False:
            return 10
        'See TensorArray.'
        with ops.name_scope(name, 'TensorArrayWrite', [self._handle, index, value]):
            value = ops.convert_to_tensor(value, preferred_dtype=self._dtype, name='value')
            _check_dtypes(value, self._dtype)
            self._check_element_shape(value.shape)
            with self._maybe_colocate_with(value):
                flow_out = gen_data_flow_ops.tensor_array_write_v3(handle=self._handle, index=index, value=value, flow_in=self._flow, name=name)
            return build_ta_with_new_flow(self, flow_out)

    def stack(self, name=None):
        if False:
            return 10
        'See TensorArray.'
        with ops.colocate_with(self._handle):
            with ops.name_scope(name, 'TensorArrayStack', [self._handle]):
                value = self.gather(math_ops.range(0, self.size()), name=name)
                if self.element_shape and (not self._dynamic_size) and (self._size is not None):
                    value.set_shape([tensor_util.constant_value(self._size)] + self.element_shape.dims)
                return value

    def gather(self, indices, name=None):
        if False:
            i = 10
            return i + 15
        'See TensorArray.'
        if self._element_shape:
            element_shape = self._element_shape[0]
        else:
            element_shape = tensor_shape.unknown_shape(None)
        value = gen_data_flow_ops.tensor_array_gather_v3(handle=self._handle, indices=indices, flow_in=self._flow, dtype=self._dtype, name=name, element_shape=element_shape)
        if self.element_shape:
            value.set_shape([None] + self.element_shape.dims)
        return value

    def concat(self, name=None):
        if False:
            while True:
                i = 10
        'See TensorArray.'
        (value, _) = gen_data_flow_ops.tensor_array_concat_v3(handle=self._handle, flow_in=self._flow, dtype=self._dtype, name=name, element_shape_except0=self.element_shape[1:])
        if self.element_shape:
            dim0 = None
            if self._infer_shape:
                size = tensor_util.constant_value(self.size())
                if size is not None and self.element_shape[0] is not None:
                    dim0 = size * self.element_shape[0]
            value.set_shape([dim0] + self.element_shape.dims[1:])
        return value

    @tf_should_use.should_use_result
    def unstack(self, value, name=None):
        if False:
            for i in range(10):
                print('nop')
        'See TensorArray.'
        with ops.name_scope(name, 'TensorArrayUnstack', [self._handle, value]):
            num_elements = array_ops.shape(value)[0]
            return self.scatter(indices=math_ops.range(0, num_elements), value=value, name=name)

    @tf_should_use.should_use_result
    def scatter(self, indices, value, name=None):
        if False:
            return 10
        'See TensorArray.'
        with ops.name_scope(name, 'TensorArrayScatter', [self._handle, value, indices]):
            value = ops.convert_to_tensor(value, preferred_dtype=self._dtype, name='value')
            _check_dtypes(value, self._dtype)
            if not context.executing_eagerly():
                self._check_element_shape(value.shape[1:])
            with self._maybe_colocate_with(value):
                flow_out = gen_data_flow_ops.tensor_array_scatter_v3(handle=self._handle, indices=indices, value=value, flow_in=self._flow, name=name)
            return build_ta_with_new_flow(self, flow_out)

    @tf_should_use.should_use_result
    def split(self, value, lengths, name=None):
        if False:
            return 10
        'See TensorArray.'
        with ops.name_scope(name, 'TensorArraySplit', [self._handle, value, lengths]):
            value = ops.convert_to_tensor(value, dtype=self._dtype, name='value')
            with self._maybe_colocate_with(value):
                lengths_64 = math_ops.cast(lengths, dtypes.int64)
                if not context.executing_eagerly():
                    clengths = tensor_util.constant_value(lengths_64)
                    if value.shape.dims is not None and clengths is not None:
                        if clengths.shape and clengths.max() == clengths.min():
                            self._check_element_shape(tensor_shape.TensorShape([clengths[0]]).concatenate(value.shape[1:]))
                flow_out = gen_data_flow_ops.tensor_array_split_v3(handle=self._handle, value=value, lengths=lengths_64, flow_in=self._flow, name=name)
            return build_ta_with_new_flow(self, flow_out)

    def size(self, name=None):
        if False:
            return 10
        'See TensorArray.'
        if not self._dynamic_size and self._size is not None:
            return ops.convert_to_tensor(self._size, dtype=dtypes.int32)
        else:
            return gen_data_flow_ops.tensor_array_size_v3(handle=self._handle, flow_in=self.flow, name=name)

    @tf_should_use.should_use_result
    def close(self, name=None):
        if False:
            i = 10
            return i + 15
        'See TensorArray.'
        return gen_data_flow_ops.tensor_array_close_v3(handle=self._handle, name=name)

class _GraphTensorArrayV2:
    """Graph-mode implementation of TensorArray backed by TensorLists.

  The backing tensor of this TensorArray is a TensorList variant tensor which is
  stored in the `flow`. The `handle` is always none here. The reason we use the
  `flow` field and not the `handle` field is to ensure backwards compatibility
  with legacy control flow.
  """

    def __init__(self, dtype, size=None, dynamic_size=None, clear_after_read=None, tensor_array_name=None, handle=None, flow=None, infer_shape=True, element_shape=None, colocate_with_first_write_call=True, name=None):
        if False:
            print('Hello World!')
        'Constructs a graph mode TensorArray.\n\n    Args:\n      dtype: (required) data type of the TensorArray.\n      size: (optional) int32 scalar `Tensor`: the size of the TensorArray.\n        Required if flow is not provided.\n      dynamic_size: (optional) Python bool: If true, writes to the TensorArray\n        can grow the TensorArray past its initial size.  Default: False.\n      clear_after_read: (optional) unused. Not supported in TensorLists.\n      tensor_array_name: (optional) unused.\n      handle: (optional) Must always be None.\n      flow: (optional) A variant `Tensor` scalar for a TensorList.\n      infer_shape: (optional, default: True) If True, shape inference is\n        enabled.  In this case, all elements must have the same shape.\n      element_shape: (optional, default: None) A `TensorShape` object specifying\n        the shape constraints of each of the elements of the TensorArray. Need\n        not be fully defined.\n      colocate_with_first_write_call: (optional). unused.\n      name: (optional) A name for the operation.\n\n    Raises:\n      ValueError: if both handle and tensor_array_name are provided.\n      TypeError: if handle is provided but is not a Tensor.\n    '
        assert handle is None
        del handle
        del clear_after_read
        del tensor_array_name
        del colocate_with_first_write_call
        self._dynamic_size = dynamic_size
        self._size = size
        if flow is not None and (not isinstance(flow, tensor_lib.Tensor) or flow.dtype != dtypes.variant):
            raise TypeError(f'Expected `flow` to be a variant tensor, but received `{flow.dtype}` instead.')
        if flow is None and size is None:
            raise ValueError('Argument `size` must be provided if argument `flow` is not provided.')
        if flow is not None and size is not None:
            raise ValueError('Cannot provide both `flow` and `size` arguments at the same time.')
        if flow is not None and element_shape is not None:
            raise ValueError('Cannot provide both `flow` and `element_shape` argumentsat the same time.')
        self._dtype = dtypes.as_dtype(dtype).base_dtype
        self._element_shape = [tensor_shape.as_shape(element_shape)]
        self._infer_shape = infer_shape
        with ops.name_scope(name, 'TensorArrayV2', [size, flow]) as scope:
            if flow is None:
                self._flow = list_ops.tensor_list_reserve(element_shape=element_shape, num_elements=size, element_dtype=dtype, name=scope)
            else:
                self._flow = flow
        self._colocate_with_first_write_call = None
        self._colocate_with = None

    @property
    def flow(self):
        if False:
            print('Hello World!')
        return self._flow

    @property
    def dtype(self):
        if False:
            for i in range(10):
                print('nop')
        return self._dtype

    @property
    def element_shape(self):
        if False:
            print('Hello World!')
        return self._element_shape[0]

    @property
    def handle(self):
        if False:
            for i in range(10):
                print('nop')
        return None

    def _check_element_shape(self, shape):
        if False:
            print('Hello World!')
        'Changes the element shape of the array given a shape to merge with.\n\n    Args:\n      shape: A `TensorShape` object to merge with.\n\n    Raises:\n      ValueError: if the provided shape is incompatible with the current\n          element shape of the `TensorArray`.\n    '
        if not shape.is_compatible_with(self.element_shape):
            raise ValueError('Inconsistent shapes: saw %s but expected %s ' % (shape, self.element_shape))
        if self._infer_shape:
            self._element_shape[0] = self.element_shape.merge_with(shape)

    def identity(self):
        if False:
            i = 10
            return i + 15
        'See TensorArray.'
        flow = array_ops.identity(self._flow)
        return build_ta_with_new_flow(self, flow)

    def grad(self, source, flow=None, name=None):
        if False:
            i = 10
            return i + 15
        'Not supported.'
        raise NotImplementedError()

    def read(self, index, name=None):
        if False:
            while True:
                i = 10
        'See TensorArray.'
        with ops.name_scope(name, 'TensorArrayV2Read', [self._flow, index]):
            value = list_ops.tensor_list_get_item(input_handle=self._flow, index=index, element_dtype=self._dtype, element_shape=self.element_shape, name=name)
            return value

    def write(self, index, value, name=None):
        if False:
            i = 10
            return i + 15
        'See TensorArray.'
        with ops.name_scope(name, 'TensorArrayV2Write', [self._flow, index, value]):
            value = ops.convert_to_tensor(value, preferred_dtype=self._dtype, name='value')
            _check_dtypes(value, self._dtype)
            self._check_element_shape(value.shape)
            flow_out = list_ops.tensor_list_set_item(input_handle=self._flow, index=index, item=value, resize_if_index_out_of_bounds=self._dynamic_size, name=name)
            return build_ta_with_new_flow(self, flow_out)

    def stack(self, name=None):
        if False:
            while True:
                i = 10
        'See TensorArray.'
        with ops.name_scope(name, 'TensorArrayV2Stack', [self._flow]):
            if not self._dynamic_size and self._size is not None:
                ta_size = tensor_util.constant_value(self._size)
            else:
                ta_size = -1
            value = list_ops.tensor_list_stack(input_handle=self._flow, element_dtype=self._dtype, num_elements=ta_size, element_shape=self.element_shape)
            return value

    def gather(self, indices, name=None):
        if False:
            print('Hello World!')
        'See TensorArray.'
        value = list_ops.tensor_list_gather(input_handle=self._flow, indices=indices, element_dtype=self._dtype, element_shape=self.element_shape, name=name)
        return value

    def concat(self, name=None):
        if False:
            return 10
        'See TensorArray.'
        if self.element_shape:
            element_shape = [None] + self.element_shape.dims[1:]
        else:
            element_shape = None
        value = list_ops.tensor_list_concat(input_handle=self._flow, element_dtype=self._dtype, element_shape=element_shape, name=name)
        return value

    @tf_should_use.should_use_result
    def unstack(self, value, name=None):
        if False:
            i = 10
            return i + 15
        'See TensorArray.'
        with ops.name_scope(name, 'TensorArrayUnstack', [self._flow, value]):
            value = ops.convert_to_tensor(value, preferred_dtype=self._dtype, name='value')
            _check_dtypes(value, self._dtype)
            self._check_element_shape(value.shape[1:])
            flow_out = list_ops.tensor_list_from_tensor(tensor=value, element_shape=value.shape[1:])
            return build_ta_with_new_flow(self, flow_out)

    @tf_should_use.should_use_result
    def scatter(self, indices, value, name=None):
        if False:
            i = 10
            return i + 15
        'See TensorArray.'
        with ops.name_scope(name, 'TensorArrayScatter', [self._flow, value, indices]):
            value = ops.convert_to_tensor(value, preferred_dtype=self._dtype, name='value')
            _check_dtypes(value, self._dtype)
            self._check_element_shape(value.shape[1:])
            flow_out = list_ops.tensor_list_scatter(tensor=value, indices=indices, element_shape=self.element_shape, input_handle=self._flow)
            return build_ta_with_new_flow(self, flow_out)

    @tf_should_use.should_use_result
    def split(self, value, lengths, name=None):
        if False:
            i = 10
            return i + 15
        'See TensorArray.'
        with ops.name_scope(name, 'TensorArraySplit', [self._flow, value, lengths]):
            value = ops.convert_to_tensor(value, preferred_dtype=self._dtype, name='value')
            _check_dtypes(value, self._dtype)
            lengths_64 = math_ops.cast(lengths, dtypes.int64)
            if not context.executing_eagerly():
                clengths = tensor_util.constant_value(lengths_64)
                if value.shape.dims is not None and clengths is not None:
                    if clengths.shape and clengths.max() == clengths.min():
                        self._check_element_shape(tensor_shape.TensorShape([clengths[0]]).concatenate(value.shape[1:]))
            flow_out = list_ops.tensor_list_split(tensor=value, lengths=lengths_64, element_shape=self.element_shape, name=name)
            return build_ta_with_new_flow(self, flow_out)

    def size(self, name=None):
        if False:
            i = 10
            return i + 15
        'See TensorArray.'
        if not self._dynamic_size and self._size is not None:
            return ops.convert_to_tensor(self._size, dtype=dtypes.int32)
        else:
            return list_ops.tensor_list_length(input_handle=self._flow, name=name)

    def close(self, name=None):
        if False:
            i = 10
            return i + 15
        'See TensorArray.'
        return gen_control_flow_ops.no_op(name=name)

class _EagerTensorArray:
    """Eager-compatible implementation of TensorArray."""

    def __init__(self, dtype, size=None, dynamic_size=None, clear_after_read=None, tensor_array_name=None, handle=None, flow=None, infer_shape=True, element_shape=None, colocate_with_first_write_call=True, name=None):
        if False:
            i = 10
            return i + 15
        'Constructs a TensorArray compatible with eager execution.\n\n    Args:\n      dtype: (required) data type of the TensorArray.\n      size: (optional) int32 scalar `Tensor`: the size of the TensorArray.\n        Required if handle is not provided.\n      dynamic_size: (optional) Python bool: If true, writes to the TensorArray\n        can grow the TensorArray past its initial size.  Default: False.\n      clear_after_read: Boolean (optional, default: True).  If True, clear\n        TensorArray values after reading them.  This disables read-many\n        semantics, but allows early release of memory.\n      tensor_array_name: unused.\n      handle: unsupported.\n      flow: unsupported.\n      infer_shape: used for error checking, same semantics as TensorArray.\n      element_shape: used for error checking, same semantics as TensorArray.\n      colocate_with_first_write_call: unsupported.\n      name: unsupported.\n\n    Raises:\n      ValueError: handle or flow are supplied, or if size is not supplied.\n    '
        del (flow, tensor_array_name, name)
        if handle is not None:
            raise ValueError('TensorArray handles are not supported when eager execution is enabled.')
        if size is None:
            raise ValueError('Size must be declared for TensorArrays when eager execution is enabled.')
        self._handle = None
        self._flow = constant_op.constant(0, dtype=dtypes.int32)
        self._infer_shape = infer_shape
        self._element_shape = tensor_shape.as_shape(element_shape)
        self._colocate_with_first_write_call = colocate_with_first_write_call
        self._dtype = dtypes.as_dtype(dtype).base_dtype
        self._dynamic_size = dynamic_size or False
        self._clear_after_read = True if clear_after_read is None else clear_after_read
        self._previously_read_indices = []
        if isinstance(size, ops.EagerTensor):
            size = size.numpy()
        self._tensor_array = [None for _ in range(size)]

    @property
    def flow(self):
        if False:
            i = 10
            return i + 15
        'For compatibility; flows are not meaningful when eager is enabled.'
        return self._flow

    @property
    def dtype(self):
        if False:
            for i in range(10):
                print('nop')
        return self._dtype

    @property
    def handle(self):
        if False:
            print('Hello World!')
        'For compatibility; handles are not meaningful when eager is enabled.'
        return self._handle

    @property
    def element_shape(self):
        if False:
            for i in range(10):
                print('nop')
        return self._element_shape

    def identity(self):
        if False:
            while True:
                i = 10
        'See TensorArray.'
        return self.parent()

    def grad(self, source, flow=None, name=None):
        if False:
            print('Hello World!')
        raise NotImplementedError("TensorArray.grad is not supported when executing eagerly; eager's gradient implementation does not use/need this function to compute gradients of operations that use TensorArrays.")

    def read(self, index, name=None):
        if False:
            i = 10
            return i + 15
        'See TensorArray.'
        del name
        if isinstance(index, ops.EagerTensor):
            index = index.numpy()
        if index < 0:
            raise errors_impl.OutOfRangeError(None, None, 'Reading from negative indices (index %d) is not allowed.' % index)
        if index >= len(self._tensor_array):
            raise errors_impl.OutOfRangeError(None, None, 'Tried to read from index %d but array size is: %d ' % (index, len(self._tensor_array)))
        tensor = self._tensor_array[index]
        if tensor is None:
            if index in self._previously_read_indices:
                raise errors_impl.InvalidArgumentError(None, None, 'Could not read index %d twice because it was cleared after a previous read (perhaps try setting clear_after_read = false?)' % index)
            else:
                tensor = self._maybe_zero(index)
        if self._clear_after_read:
            self._tensor_array[index] = None
            self._previously_read_indices.append(index)
        return tensor

    def _write(self, index, value):
        if False:
            while True:
                i = 10
        'Writes `value` into index named by `index`.\n\n    Args:\n      index: 0-D.  int32 scalar with the index to write to.\n      value: N-D.  Tensor of type `dtype`.  The `Tensor` to write to `index`.\n\n    Raises:\n      errors_impl.InvalidArgumentError: `value` dtype does not match dtype.\n      errors_impl.OutOfRangeError: `index` is out of bounds.\n      ValueError: shape of `value` is not consistent with inferred shape.\n    '
        if isinstance(index, ops.EagerTensor):
            index = index.numpy()
        if index < 0:
            raise errors_impl.OutOfRangeError(None, None, 'Writing to negative indices (index %d) is not allowed.' % index)
        size = len(self._tensor_array)
        if index >= size:
            if not self._dynamic_size:
                raise errors_impl.OutOfRangeError(None, None, 'Tried to write to index %d but array is not resizeable and size is: %d ' % (index, size))
            self._tensor_array.extend((None for _ in range(index - size + 1)))
        if not isinstance(value, ops.EagerTensor):
            value = ops.convert_to_tensor(value, preferred_dtype=self._dtype, name='value')
        if self._dtype != value.dtype:
            raise errors_impl.InvalidArgumentError(None, None, 'TensorArray dtype is %s but Op is trying to write dtype %s ' % (self._dtype.name, value.dtype.name))
        if not self._element_shape.is_compatible_with(value.shape):
            raise ValueError('Incompatible shape for value (%s), expected (%s)' % (value.shape, self._element_shape))
        if self._infer_shape:
            self._element_shape = self._element_shape.merge_with(value.shape)
        self._tensor_array[index] = value

    def write(self, index, value, name=None):
        if False:
            while True:
                i = 10
        'See TensorArray.'
        del name
        self._write(index, value)
        return self.parent()

    def _maybe_zero(self, ix):
        if False:
            for i in range(10):
                print('nop')
        val = self._tensor_array[ix]
        if val is None:
            val = self._tensor_array[ix] = array_ops.zeros(shape=self._element_shape, dtype=self._dtype)
        return val

    def stack(self, name=None):
        if False:
            while True:
                i = 10
        'See TensorArray.'
        if self._tensor_array:
            for ix in range(len(self._tensor_array)):
                self._maybe_zero(ix)
        if not self._tensor_array and self._element_shape.is_fully_defined():
            return ops.convert_to_tensor(np.ndarray([0] + self._element_shape), name=name, dtype=self._dtype)
        else:
            return ops.convert_to_tensor(self._tensor_array, name=name, dtype=self._dtype)

    def gather(self, indices, name=None):
        if False:
            while True:
                i = 10
        'See TensorArray.'
        del name
        if isinstance(indices, ops.EagerTensor):
            indices = indices.numpy()
        return array_ops_stack.stack([self._maybe_zero(i) for i in indices])

    def concat(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        'See TensorArray.'
        try:
            return array_ops.concat([self._maybe_zero(ix) for ix in range(len(self._tensor_array))], 0, name=name)
        except errors_impl.OpError:
            shapes = [t.shape for t in self._tensor_array]
            ndims = [s.ndims for s in shapes]
            if 0 in ndims:
                idx = ndims.index(0)
                raise errors_impl.InvalidArgumentError(None, None, 'Concat saw a scalar shape at index %d but requires at least vectors.' % idx)
            else:
                raise

    def unstack(self, value, name=None):
        if False:
            return 10
        'See TensorArray.'
        tensors = array_ops_stack.unstack(value, name=name)
        if len(tensors) > len(self._tensor_array) and (not self._dynamic_size):
            raise ValueError('Cannot unstack %d tensors into a TensorArray of static size %d ' % (len(tensors), len(self._tensor_array)))
        self._tensor_array = tensors
        return self.parent()

    def scatter(self, indices, value, name=None):
        if False:
            i = 10
            return i + 15
        'See TensorArray.'
        del name
        if isinstance(indices, ops.EagerTensor):
            indices = indices.numpy()
        for (index, val) in zip(indices, array_ops_stack.unstack(value)):
            self._write(index, val)
        return self.parent()

    def split(self, value, lengths, name=None):
        if False:
            i = 10
            return i + 15
        'See TensorArray.'
        value = ops.convert_to_tensor(value, preferred_dtype=self._dtype, name='value')
        _check_dtypes(value, self._dtype)
        lengths = ops.convert_to_tensor(lengths)
        sum_lengths = math_ops.reduce_sum(lengths)
        if lengths.shape.ndims != 1:
            raise errors_impl.InvalidArgumentError(None, None, 'Expected lengths to be a vector, received shape: %s ' % lengths.shape.as_list())
        elif value.shape.ndims == 0:
            raise errors_impl.InvalidArgumentError(None, None, 'Expected value to be at least a vector, but received shape: %s ' % value.shape.as_list())
        elif sum_lengths.numpy() != value.shape.as_list()[0]:
            raise errors_impl.InvalidArgumentError(None, None, "Expected sum of lengths to be equal to values.shape[0], but sum of lengths is %d and value's shape is: %s " % (sum_lengths.numpy(), value.shape.as_list()))
        elif not self._dynamic_size and lengths.shape[0] != len(self._tensor_array):
            raise errors_impl.InvalidArgumentError(None, None, "TensorArray's size is not equal to the size of lengths (%d vs. %d), and the TensorArray is not marked as dynamically resizeable." % (len(self._tensor_array), lengths.shape[0]))
        else:
            self._tensor_array = array_ops.split(value, lengths, name=name)
            return self.parent()

    def size(self, name=None):
        if False:
            i = 10
            return i + 15
        'See TensorArray.'
        del name
        return constant_op.constant(len(self._tensor_array))

    def close(self, name=None):
        if False:
            return 10
        del name
        del self._tensor_array[:]

@tf_export('TensorArray')
class TensorArray:
    """Class wrapping dynamic-sized, per-time-step, Tensor arrays.

  This class is meant to be used with dynamic iteration primitives such as
  `while_loop` and `map_fn`.  It supports gradient back-propagation via special
  "flow" control flow dependencies.

  Note that although the array can be read multiple times and positions can be
  overwritten, behavior may be undefined when storing multiple references to
  the same array and clear_after_read is False. In particular, avoid using
  methods like concat() to convert an intermediate TensorArray to a Tensor,
  then further modifying the TensorArray, particularly if you need to backprop
  through it later.

  Example 1: Plain reading and writing.

  >>> ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
  >>> ta = ta.write(0, 10)
  >>> ta = ta.write(1, 20)
  >>> ta = ta.write(2, 30)
  >>>
  >>> ta.read(0)
  <tf.Tensor: shape=(), dtype=float32, numpy=10.0>
  >>> ta.read(1)
  <tf.Tensor: shape=(), dtype=float32, numpy=20.0>
  >>> ta.read(2)
  <tf.Tensor: shape=(), dtype=float32, numpy=30.0>
  >>> ta.stack()
  <tf.Tensor: shape=(3,), dtype=float32, numpy=array([10., 20., 30.],
  dtype=float32)>

  Example 2: Fibonacci sequence algorithm that writes in a loop then returns.

  >>> @tf.function
  ... def fibonacci(n):
  ...   ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
  ...   ta = ta.unstack([0., 1.])
  ...
  ...   for i in range(2, n):
  ...     ta = ta.write(i, ta.read(i - 1) + ta.read(i - 2))
  ...
  ...   return ta.stack()
  >>>
  >>> fibonacci(7)
  <tf.Tensor: shape=(7,), dtype=float32,
  numpy=array([0., 1., 1., 2., 3., 5., 8.], dtype=float32)>

  Example 3: A simple loop interacting with a `tf.Variable`.

  >>> v = tf.Variable(1)
  >>> @tf.function
  ... def f(x):
  ...   ta = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
  ...   for i in tf.range(x):
  ...     v.assign_add(i)
  ...     ta = ta.write(i, v)
  ...   return ta.stack()
  >>> f(5)
  <tf.Tensor: shape=(5,), dtype=int32, numpy=array([ 1,  2,  4,  7, 11],
  dtype=int32)>
  """

    def __init__(self, dtype, size=None, dynamic_size=None, clear_after_read=None, tensor_array_name=None, handle=None, flow=None, infer_shape=True, element_shape=None, colocate_with_first_write_call=True, name=None):
        if False:
            i = 10
            return i + 15
        'Construct a new TensorArray or wrap an existing TensorArray handle.\n\n    A note about the parameter `name`:\n\n    The name of the `TensorArray` (even if passed in) is uniquified: each time\n    a new `TensorArray` is created at runtime it is assigned its own name for\n    the duration of the run.  This avoids name collisions if a `TensorArray`\n    is created within a `while_loop`.\n\n    Args:\n      dtype: (required) data type of the TensorArray.\n      size: (optional) int32 scalar `Tensor`: the size of the TensorArray.\n        Required if handle is not provided.\n      dynamic_size: (optional) Python bool: If true, writes to the TensorArray\n        can grow the TensorArray past its initial size.  Default: False.\n      clear_after_read: Boolean (optional, default: True).  If True, clear\n        TensorArray values after reading them.  This disables read-many\n        semantics, but allows early release of memory.\n      tensor_array_name: (optional) Python string: the name of the TensorArray.\n        This is used when creating the TensorArray handle.  If this value is\n        set, handle should be None.\n      handle: (optional) A `Tensor` handle to an existing TensorArray.  If this\n        is set, tensor_array_name should be None. Only supported in graph mode.\n      flow: (optional) A float `Tensor` scalar coming from an existing\n        `TensorArray.flow`. Only supported in graph mode.\n      infer_shape: (optional, default: True) If True, shape inference is\n        enabled.  In this case, all elements must have the same shape.\n      element_shape: (optional, default: None) A `TensorShape` object specifying\n        the shape constraints of each of the elements of the TensorArray. Need\n        not be fully defined.\n      colocate_with_first_write_call: If `True`, the TensorArray will be\n        colocated on the same device as the Tensor used on its first write\n        (write operations include `write`, `unstack`, and `split`).  If `False`,\n        the TensorArray will be placed on the device determined by the device\n        context available during its initialization.\n      name: A name for the operation (optional).\n\n    Raises:\n      ValueError: if both handle and tensor_array_name are provided.\n      TypeError: if handle is provided but is not a Tensor.\n    '
        if context.executing_eagerly() and (flow is None or flow.dtype != dtypes.variant):
            implementation = _EagerTensorArray
        elif flow is not None and flow.dtype == dtypes.variant or control_flow_util.EnableControlFlowV2(ops.get_default_graph()):
            implementation = _GraphTensorArrayV2
        else:
            implementation = _GraphTensorArray
        self._implementation = implementation(dtype, size=size, dynamic_size=dynamic_size, clear_after_read=clear_after_read, tensor_array_name=tensor_array_name, handle=handle, flow=flow, infer_shape=infer_shape, element_shape=element_shape, colocate_with_first_write_call=colocate_with_first_write_call, name=name)
        self._implementation.parent = weakref.ref(self)

    @property
    def flow(self):
        if False:
            return 10
        'The flow `Tensor` forcing ops leading to this TensorArray state.'
        return self._implementation._flow

    @property
    def dtype(self):
        if False:
            i = 10
            return i + 15
        'The data type of this TensorArray.'
        return self._implementation._dtype

    @property
    def handle(self):
        if False:
            for i in range(10):
                print('nop')
        'The reference to the TensorArray.'
        return self._implementation.handle

    @property
    def element_shape(self):
        if False:
            return 10
        'The `tf.TensorShape` of elements in this TensorArray.'
        return self._implementation.element_shape

    @property
    def dynamic_size(self):
        if False:
            while True:
                i = 10
        'Python bool; if `True` the TensorArray can grow dynamically.'
        return self._implementation._dynamic_size

    @property
    def _infer_shape(self):
        if False:
            return 10
        return self._implementation._infer_shape

    def identity(self):
        if False:
            print('Hello World!')
        'Returns a TensorArray with the same content and properties.\n\n    Returns:\n      A new TensorArray object with flow that ensures the control dependencies\n      from the contexts will become control dependencies for writes, reads, etc.\n      Use this object for all subsequent operations.\n    '
        return self._implementation.identity()

    def grad(self, source, flow=None, name=None):
        if False:
            while True:
                i = 10
        return self._implementation.grad(source, flow=flow, name=name)

    def read(self, index, name=None):
        if False:
            return 10
        'Read the value at location `index` in the TensorArray.\n\n    Args:\n      index: 0-D.  int32 tensor with the index to read from.\n      name: A name for the operation (optional).\n\n    Returns:\n      The tensor at index `index`.\n    '
        return self._implementation.read(index, name=name)

    @tf_should_use.should_use_result(warn_in_eager=True)
    def write(self, index, value, name=None):
        if False:
            while True:
                i = 10
        'Write `value` into index `index` of the TensorArray.\n\n    Args:\n      index: 0-D.  int32 scalar with the index to write to.\n      value: N-D.  Tensor of type `dtype`.  The Tensor to write to this index.\n      name: A name for the operation (optional).\n\n    Returns:\n      A new TensorArray object with flow that ensures the write occurs.\n      Use this object for all subsequent operations.\n\n    Raises:\n      ValueError: if there are more writers than specified.\n    '
        return self._implementation.write(index, value, name=name)

    def stack(self, name=None):
        if False:
            i = 10
            return i + 15
        'Return the values in the TensorArray as a stacked `Tensor`.\n\n    All of the values must have been written and their shapes must all match.\n    If input shapes have rank-`R`, then output shape will have rank-`(R+1)`.\n\n    For example:\n\n\n    >>> ta = tf.TensorArray(tf.int32, size=3)\n    >>> ta = ta.write(0, tf.constant([1, 2]))\n    >>> ta = ta.write(1, tf.constant([3, 4]))\n    >>> ta = ta.write(2, tf.constant([5, 6]))\n    >>> ta.stack()\n    <tf.Tensor: shape=(3, 2), dtype=int32, numpy=\n    array([[1, 2],\n           [3, 4],\n           [5, 6]], dtype=int32)>\n\n\n    Args:\n      name: A name for the operation (optional).\n\n    Returns:\n      All the tensors in the TensorArray stacked into one tensor.\n    '
        return self._implementation.stack(name=name)

    def gather(self, indices, name=None):
        if False:
            print('Hello World!')
        'Return selected values in the TensorArray as a packed `Tensor`.\n\n    All of selected values must have been written and their shapes\n    must all match.\n\n    Args:\n      indices: A `1-D` `Tensor` taking values in `[0, max_value)`.  If the\n        `TensorArray` is not dynamic, `max_value=size()`.\n      name: A name for the operation (optional).\n\n    Returns:\n      The tensors in the `TensorArray` selected by `indices`, packed into one\n      tensor.\n    '
        return self._implementation.gather(indices, name=name)

    def concat(self, name=None):
        if False:
            i = 10
            return i + 15
        'Return the values in the TensorArray as a concatenated `Tensor`.\n\n    All of the values must have been written, their ranks must match, and\n    and their shapes must all match for all dimensions except the first.\n\n    Args:\n      name: A name for the operation (optional).\n\n    Returns:\n      All the tensors in the TensorArray concatenated into one tensor.\n    '
        return self._implementation.concat(name=name)

    @tf_should_use.should_use_result
    def unstack(self, value, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Unstack the values of a `Tensor` in the TensorArray.\n\n    If input value shapes have rank-`R`, then the output TensorArray will\n    contain elements whose shapes are rank-`(R-1)`.\n\n    Args:\n      value: (N+1)-D.  Tensor of type `dtype`.  The Tensor to unstack.\n      name: A name for the operation (optional).\n\n    Returns:\n      A new TensorArray object with flow that ensures the unstack occurs.\n      Use this object for all subsequent operations.\n\n    Raises:\n      ValueError: if the shape inference fails.\n    '
        return self._implementation.unstack(value, name=name)

    @tf_should_use.should_use_result
    def scatter(self, indices, value, name=None):
        if False:
            return 10
        'Scatter the values of a `Tensor` in specific indices of a `TensorArray`.\n\n    Args:\n      indices: A `1-D` `Tensor` taking values in `[0, max_value)`.  If the\n        `TensorArray` is not dynamic, `max_value=size()`.\n      value: (N+1)-D.  Tensor of type `dtype`.  The Tensor to unpack.\n      name: A name for the operation (optional).\n\n    Returns:\n      A new TensorArray object with flow that ensures the scatter occurs.\n      Use this object for all subsequent operations.\n\n    Raises:\n      ValueError: if the shape inference fails.\n    '
        return self._implementation.scatter(indices, value, name=name)

    @tf_should_use.should_use_result
    def split(self, value, lengths, name=None):
        if False:
            return 10
        'Split the values of a `Tensor` into the TensorArray.\n\n    Args:\n      value: (N+1)-D.  Tensor of type `dtype`.  The Tensor to split.\n      lengths: 1-D.  int32 vector with the lengths to use when splitting `value`\n        along its first dimension.\n      name: A name for the operation (optional).\n\n    Returns:\n      A new TensorArray object with flow that ensures the split occurs.\n      Use this object for all subsequent operations.\n\n    Raises:\n      ValueError: if the shape inference fails.\n    '
        return self._implementation.split(value, lengths, name=name)

    def size(self, name=None):
        if False:
            i = 10
            return i + 15
        'Return the size of the TensorArray.'
        return self._implementation.size(name=name)

    @tf_should_use.should_use_result
    def close(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Close the current TensorArray.'
        return self._implementation.close(name=name)

    def __tf_tracing_type__(self, _):
        if False:
            i = 10
            return i + 15
        return TensorArrayTraceType(self)

def build_ta_with_new_flow(old_ta, flow):
    if False:
        for i in range(10):
            print('nop')
    'Builds a TensorArray with a new `flow` tensor.'
    impl = old_ta._implementation if isinstance(old_ta, TensorArray) else old_ta
    if not context.executing_eagerly():
        if not isinstance(impl, _GraphTensorArrayV2) and control_flow_util.EnableControlFlowV2(ops.get_default_graph()):
            raise NotImplementedError('Attempting to build a graph-mode TF2-style TensorArray from either an eager-mode TensorArray or a TF1-style TensorArray.  This is not currently supported.  You may be attempting to capture a TensorArray inside a tf.function or tf.data map function. Instead, construct a new TensorArray inside the function.')
    new_ta = TensorArray(dtype=impl.dtype, handle=impl.handle, flow=flow, infer_shape=impl._infer_shape, colocate_with_first_write_call=impl._colocate_with_first_write_call)
    new_impl = new_ta._implementation
    new_impl._dynamic_size = impl._dynamic_size
    new_impl._size = impl._size
    new_impl._colocate_with = impl._colocate_with
    new_impl._element_shape = impl._element_shape
    return new_ta

def _check_dtypes(value, dtype):
    if False:
        return 10
    if value.dtype != dtype:
        logging.error('Error: Input value {} has dtype {}, but expected dtype {}.  This leads to undefined behavior and will be an error in future versions of TensorFlow.  Traceback:\n{}'.format(value, str(value.dtype), str(dtype), ''.join(traceback.format_stack())))

@tf_export('TensorArraySpec')
@type_spec_registry.register('tf.TensorArraySpec')
class TensorArraySpec(type_spec.TypeSpec):
    """Type specification for a `tf.TensorArray`."""
    __slots__ = ['_element_shape', '_dtype', '_dynamic_size', '_infer_shape']
    value_type = property(lambda self: TensorArray)

    def __init__(self, element_shape=None, dtype=dtypes.float32, dynamic_size=False, infer_shape=True):
        if False:
            while True:
                i = 10
        'Constructs a type specification for a `tf.TensorArray`.\n\n    Args:\n      element_shape: The shape of each element in the `TensorArray`.\n      dtype: Data type of the `TensorArray`.\n      dynamic_size: Whether the `TensorArray` can grow past its initial size.\n      infer_shape: Whether shape inference is enabled.\n    '
        self._element_shape = tensor_shape.as_shape(element_shape)
        self._dtype = dtypes.as_dtype(dtype)
        self._dynamic_size = dynamic_size
        self._infer_shape = infer_shape

    def is_subtype_of(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, TensorArraySpec) and self._dtype == other._dtype and (self._dynamic_size == other._dynamic_size)

    def most_specific_common_supertype(self, others):
        if False:
            while True:
                i = 10
        'Returns the most specific supertype of `self` and `others`.\n\n    Args:\n      others: A Sequence of `TypeSpec`.\n\n    Returns `None` if a supertype does not exist.\n    '
        if not all((isinstance(other, TensorArraySpec) for other in others)):
            return False
        common_shape = self._element_shape.most_specific_common_supertype((other._element_shape for other in others))
        if common_shape is None:
            return None
        if not all((self._dtype == other._dtype for other in others)):
            return None
        if not all((self._dynamic_size == other._dynamic_size for other in others)):
            return None
        infer_shape = self._infer_shape and all((other._infer_shape for other in others))
        return TensorArraySpec(common_shape, self._dtype, self._dynamic_size, infer_shape)

    def is_compatible_with(self, other):
        if False:
            print('Hello World!')
        if not isinstance(other, type_spec.TypeSpec):
            other = type_spec.type_spec_from_value(other)
        return isinstance(other, TensorArraySpec) and self._dtype.is_compatible_with(other._dtype) and self._element_shape.is_compatible_with(other._element_shape) and (self._dynamic_size == other._dynamic_size)

    def _serialize(self):
        if False:
            return 10
        return (self._element_shape, self._dtype, self._dynamic_size, self._infer_shape)

    @property
    def _component_specs(self):
        if False:
            i = 10
            return i + 15
        return [tensor_lib.TensorSpec([], dtypes.variant)]

    def _to_components(self, value):
        if False:
            print('Hello World!')
        if not isinstance(value, TensorArray):
            raise TypeError('Expected value to be a TensorArray, but got: `{}`'.format(type(value)))
        if value.flow is not None and value.flow.dtype == dtypes.variant:
            return [value.flow]
        else:
            with ops.name_scope('convert_tensor_array'):
                flow = list_ops.tensor_list_from_tensor(tensor=value.stack(), element_shape=value.element_shape)
            return [flow]

    def _from_components(self, tensor_list):
        if False:
            print('Hello World!')
        ret = TensorArray(dtype=self._dtype, flow=tensor_list[0], dynamic_size=self._dynamic_size, infer_shape=self._infer_shape)
        ret._implementation._element_shape = [self._element_shape]
        return ret

    @staticmethod
    def from_value(value):
        if False:
            return 10
        if not isinstance(value, TensorArray):
            raise TypeError('Expected value to be a TensorArray, but got: `{}`'.format(type(value)))
        return TensorArraySpec(dtype=value.dtype, element_shape=value.element_shape, dynamic_size=value.dynamic_size, infer_shape=value._infer_shape)

    def _to_legacy_output_types(self):
        if False:
            i = 10
            return i + 15
        return self._dtype

    def _to_legacy_output_shapes(self):
        if False:
            i = 10
            return i + 15
        return tensor_shape.TensorShape([self._dynamic_size, self._infer_shape]).concatenate(self._element_shape)

    def _to_legacy_output_classes(self):
        if False:
            i = 10
            return i + 15
        return TensorArray
nested_structure_coder.register_codec(nested_structure_coder.BuiltInTypeSpecCodec(TensorArraySpec, struct_pb2.TypeSpecProto.TENSOR_ARRAY_SPEC))

class TensorArrayTraceType(trace.TraceType):
    """Represents TraceType of TensorArray."""

    def __init__(self, value):
        if False:
            i = 10
            return i + 15
        self._value = value

    def is_subtype_of(self, other):
        if False:
            return 10
        return self == other

    def most_specific_common_supertype(self, types):
        if False:
            for i in range(10):
                print('nop')
        return self if all((self == other for other in types)) else None

    def placeholder_value(self, placeholder_context):
        if False:
            print('Hello World!')
        return self._value

    def flatten(self):
        if False:
            while True:
                i = 10
        return [tensor_lib.TensorSpec([], dtypes.variant)]

    def from_tensors(self, tensors):
        if False:
            return 10
        return next(tensors)

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if not isinstance(other, trace.TraceType):
            return NotImplemented
        if not isinstance(other, TensorArrayTraceType):
            return False
        return self._value is other._value

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return id(self._value)

    def __repr__(self):
        if False:
            return 10
        return f'{self.__class__.__name__}(value={self._value!r})'
type_spec.register_type_spec_from_value_converter(TensorArray, TensorArraySpec.from_value, allow_subclass=True)