"""DTensor variable and saveable."""
import functools
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util.tf_export import tf_export

class DSaveSpec(saveable_object.SaveSpec):
    """DTensor SaveSpec that additionaly captures global_shape and layout."""

    def __init__(self, tensor, slice_spec, name, global_shape, layout, dtype=None, device=None):
        if False:
            return 10
        super().__init__(tensor=tensor, slice_spec=slice_spec, name=name, dtype=dtype, device=device)
        self.global_shape = global_shape
        self.layout = layout

class _DVariableSaveable(saveable_object.SaveableObject):
    """Class for defining how to save/restore DTensor variable."""

    def __init__(self, dvariable, name):
        if False:
            while True:
                i = 10
        with ops.device(dvariable.device):
            original_layout = api.fetch_layout(dvariable)
        self._original_layout = original_layout
        self._dvariable = dvariable

        def pack(tensors, layout):
            if False:
                return 10
            with ops.device(dvariable.device):
                return api.pack(tensors, layout)
        host_layout = layout_lib.Layout(original_layout.sharding_specs, original_layout.mesh.host_mesh())

        def get_host_dtensor():
            if False:
                print('Hello World!')
            if original_layout.mesh.device_type().upper() != 'CPU':
                if context.executing_eagerly():
                    host_dtensor = api.pack(api.unpack(dvariable.read_value()), host_layout)
                else:
                    host_dtensor = api.copy_to_mesh(dvariable.read_value(), host_layout)
            else:
                host_dtensor = dvariable.read_value()
            return math_ops.cast(host_dtensor, dtypes.bfloat16) if self.should_cast(host_dtensor) else host_dtensor
        num_local_devices = original_layout.mesh.num_local_devices()
        super(_DVariableSaveable, self).__init__(None, [DSaveSpec(tensor=get_host_dtensor, slice_spec=pack([''] * num_local_devices, layout_lib.Layout.replicated(original_layout.mesh.host_mesh(), rank=0)), name=pack([name] * num_local_devices, layout_lib.Layout.replicated(original_layout.mesh.host_mesh(), rank=0)), global_shape=dvariable.shape, layout=host_layout.to_string(), dtype=dtypes.bfloat16 if self.should_cast(dvariable) else dvariable.dtype, device=dvariable.device)], name)

    def should_cast(self, v):
        if False:
            i = 10
            return i + 15
        'Returns True if v has float32 dtype and is intructed to save as bf16.\n\n    Args:\n      v : The variable that determines whether to cast.\n\n    Returns:\n      True if current savable DVariable is instructed to save as bfloat16 and\n        the variable has dtype float32.\n    '
        return self._dvariable.save_as_bf16 and v.dtype == dtypes.float32

    def restore(self, restored_tensors, restored_shapes):
        if False:
            i = 10
            return i + 15
        'Restores the same value into all variables.'
        (tensor,) = restored_tensors

        @def_function.function
        def _restore(t):
            if False:
                return 10
            with ops.device(self._dvariable.device):
                return api.copy_to_mesh(t, self._original_layout)
        if self._original_layout.mesh.device_type().upper() != 'CPU':
            tensor = _restore(tensor)
        return self._dvariable.assign(math_ops.cast(tensor, dtype=self._dvariable.dtype) if self._dvariable.save_as_bf16 else tensor)

@tf_export('experimental.dtensor.DVariable', v1=[])
class DVariable(resource_variable_ops.ResourceVariable):
    """A replacement for tf.Variable which follows initial value placement.

    The class also handles restore/save operations in DTensor. Note that,
    DVariable may fall back to normal tf.Variable at this moment if
    `initial_value` is not a DTensor.
  """

    def __init__(self, initial_value, *args, dtype=None, **kwargs):
        if False:
            print('Hello World!')
        'Overrides tf.Variable to fix VarHandleOp placements.'
        layout = kwargs.pop('layout', None)
        shape = kwargs.get('shape', None)
        if callable(initial_value):
            unwrapped = initial_value
            if issubclass(type(initial_value), functools.partial):
                unwrapped = initial_value.func
            if issubclass(type(unwrapped), trackable.CheckpointInitialValueCallable):
                if not shape or not layout:
                    raise ValueError('Expected shape and layout to be not None.')
                initial_value = api.call_with_layout(initial_value, layout, shard_info=trackable.ShardInfo(shape=shape, offset=[0] * len(shape)))
            else:
                initial_value = initial_value()
        if isinstance(initial_value, trackable.CheckpointInitialValue):
            initial_value = initial_value.wrapped_value
        initial_value = ops.convert_to_tensor(initial_value, dtype=dtype)
        variable_device = initial_value.device
        self._save_as_bf16 = False
        with ops.device(variable_device):
            if context.executing_eagerly():
                if api.is_dtensor(initial_value):
                    value_layout = api.fetch_layout(initial_value)
                    if layout is not None and layout != value_layout:
                        raise errors_impl.InvalidArgumentError(None, None, f'Conflicting layout are provided for initial value layout ({value_layout}) and variable ({layout}).')
                    layout = value_layout
                elif layout is not None:
                    initial_value = api.relayout(initial_value, layout)
                else:
                    raise errors_impl.InvalidArgumentError(None, None, 'Neither layout nor DTensor initial value are provided.')
                self.layout = layout
                with api.default_mesh(layout.mesh):
                    super(DVariable, self).__init__(initial_value, *args, dtype=dtype, **kwargs)
            else:
                if layout is not None:
                    initial_value = api.relayout(initial_value, layout)
                super(DVariable, self).__init__(initial_value, *args, dtype=dtype, **kwargs)

    @property
    def save_as_bf16(self):
        if False:
            i = 10
            return i + 15
        return self._save_as_bf16

    @save_as_bf16.setter
    def save_as_bf16(self, save_as_bf16):
        if False:
            for i in range(10):
                print('nop')
        'Enables saving float32 as bfloat16.'
        self._save_as_bf16 = save_as_bf16 and self.dtype == dtypes.float32

    def _gather_saveables_for_checkpoint(self):
        if False:
            i = 10
            return i + 15
        return {trackable.VARIABLE_VALUE_KEY: functools.partial(_DVariableSaveable, self)}