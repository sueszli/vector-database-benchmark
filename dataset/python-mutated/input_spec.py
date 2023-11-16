"""Contains the InputSpec class."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

@tf_export(v1=['layers.InputSpec'])
class InputSpec(object):
    """Specifies the rank, dtype and shape of every input to a layer.

  Layers can expose (if appropriate) an `input_spec` attribute:
  an instance of `InputSpec`, or a nested structure of `InputSpec` instances
  (one per input tensor). These objects enable the layer to run input
  compatibility checks for input structure, input rank, input shape, and
  input dtype.

  A None entry in a shape is compatible with any dimension,
  a None shape is compatible with any shape.

  Args:
    dtype: Expected DataType of the input.
    shape: Shape tuple, expected shape of the input
      (may include None for unchecked axes). Includes the batch size.
    ndim: Integer, expected rank of the input.
    max_ndim: Integer, maximum rank of the input.
    min_ndim: Integer, minimum rank of the input.
    axes: Dictionary mapping integer axes to
      a specific dimension value.
    allow_last_axis_squeeze: If True, then allow inputs of rank N+1 as long
      as the last axis of the input is 1, as well as inputs of rank N-1
      as long as the last axis of the spec is 1.
    name: Expected key corresponding to this input when passing data as
      a dictionary.

  Example:

  ```python
  class MyLayer(Layer):
      def __init__(self):
          super(MyLayer, self).__init__()
          # The layer will accept inputs with shape (?, 28, 28) & (?, 28, 28, 1)
          # and raise an appropriate error message otherwise.
          self.input_spec = InputSpec(
              shape=(None, 28, 28, 1),
              allow_last_axis_squeeze=True)
  ```
  """

    def __init__(self, dtype=None, shape=None, ndim=None, max_ndim=None, min_ndim=None, axes=None, allow_last_axis_squeeze=False, name=None):
        if False:
            while True:
                i = 10
        self.dtype = dtypes.as_dtype(dtype).name if dtype is not None else None
        shape = tensor_shape.TensorShape(shape)
        if shape.rank is None:
            shape = None
        else:
            shape = tuple(shape.as_list())
        if shape is not None:
            self.ndim = len(shape)
            self.shape = shape
        else:
            self.ndim = ndim
            self.shape = None
        self.max_ndim = max_ndim
        self.min_ndim = min_ndim
        self.name = name
        self.allow_last_axis_squeeze = allow_last_axis_squeeze
        try:
            axes = axes or {}
            self.axes = {int(k): axes[k] for k in axes}
        except (ValueError, TypeError):
            raise TypeError('The keys in axes must be integers.')
        if self.axes and (self.ndim is not None or self.max_ndim is not None):
            max_dim = (self.ndim if self.ndim else self.max_ndim) - 1
            max_axis = max(self.axes)
            if max_axis > max_dim:
                raise ValueError('Axis {} is greater than the maximum allowed value: {}'.format(max_axis, max_dim))

    def __repr__(self):
        if False:
            print('Hello World!')
        spec = ['dtype=' + str(self.dtype) if self.dtype else '', 'shape=' + str(self.shape) if self.shape else '', 'ndim=' + str(self.ndim) if self.ndim else '', 'max_ndim=' + str(self.max_ndim) if self.max_ndim else '', 'min_ndim=' + str(self.min_ndim) if self.min_ndim else '', 'axes=' + str(self.axes) if self.axes else '']
        return 'InputSpec(%s)' % ', '.join((x for x in spec if x))

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        return {'dtype': self.dtype, 'shape': self.shape, 'ndim': self.ndim, 'max_ndim': self.max_ndim, 'min_ndim': self.min_ndim, 'axes': self.axes}

    @classmethod
    def from_config(cls, config):
        if False:
            for i in range(10):
                print('nop')
        return cls(**config)

def to_tensor_shape(spec):
    if False:
        print('Hello World!')
    "Returns a tf.TensorShape object that matches the shape specifications.\n\n  If the InputSpec's shape or ndim is defined, this method will return a fully\n  or partially-known shape. Otherwise, the returned TensorShape is None.\n\n  Args:\n    spec: an InputSpec object.\n\n  Returns:\n    a tf.TensorShape object\n  "
    if spec.ndim is None and spec.shape is None:
        return tensor_shape.TensorShape(None)
    elif spec.shape is not None:
        return tensor_shape.TensorShape(spec.shape)
    else:
        shape = [None] * spec.ndim
        for a in spec.axes:
            shape[a] = spec.axes[a]
        return tensor_shape.TensorShape(shape)

def assert_input_compatibility(input_spec, inputs, layer_name):
    if False:
        while True:
            i = 10
    'Checks compatibility between the layer and provided inputs.\n\n  This checks that the tensor(s) `inputs` verify the input assumptions\n  of a layer (if any). If not, a clear and actional exception gets raised.\n\n  Args:\n      input_spec: An InputSpec instance, list of InputSpec instances, a nested\n          structure of InputSpec instances, or None.\n      inputs: Input tensor, list of input tensors, or a nested structure of\n          input tensors.\n      layer_name: String, name of the layer (for error message formatting).\n\n  Raises:\n      ValueError: in case of mismatch between\n          the provided inputs and the expectations of the layer.\n  '
    if not input_spec:
        return
    input_spec = nest.flatten(input_spec)
    if isinstance(inputs, dict):
        names = [spec.name for spec in input_spec]
        if all(names):
            list_inputs = []
            for name in names:
                if name not in inputs:
                    raise ValueError('Missing data for input "%s". You passed a data dictionary with keys %s. Expected the following keys: %s' % (name, list(inputs.keys()), names))
                list_inputs.append(inputs[name])
            inputs = list_inputs
    inputs = nest.flatten(inputs)
    for x in inputs:
        if not hasattr(x, 'shape'):
            raise TypeError('Inputs to a layer should be tensors. Got: %s' % (x,))
    if len(inputs) != len(input_spec):
        raise ValueError('Layer ' + layer_name + ' expects ' + str(len(input_spec)) + ' input(s), but it received ' + str(len(inputs)) + ' input tensors. Inputs received: ' + str(inputs))
    for (input_index, (x, spec)) in enumerate(zip(inputs, input_spec)):
        if spec is None:
            continue
        shape = tensor_shape.TensorShape(x.shape)
        if shape.rank is None:
            return
        if spec.ndim is not None and (not spec.allow_last_axis_squeeze):
            ndim = shape.rank
            if ndim != spec.ndim:
                raise ValueError('Input ' + str(input_index) + ' of layer ' + layer_name + ' is incompatible with the layer: expected ndim=' + str(spec.ndim) + ', found ndim=' + str(ndim) + '. Full shape received: ' + str(tuple(shape)))
        if spec.max_ndim is not None:
            ndim = x.shape.rank
            if ndim is not None and ndim > spec.max_ndim:
                raise ValueError('Input ' + str(input_index) + ' of layer ' + layer_name + ' is incompatible with the layer: expected max_ndim=' + str(spec.max_ndim) + ', found ndim=' + str(ndim))
        if spec.min_ndim is not None:
            ndim = x.shape.rank
            if ndim is not None and ndim < spec.min_ndim:
                raise ValueError('Input ' + str(input_index) + ' of layer ' + layer_name + ' is incompatible with the layer: : expected min_ndim=' + str(spec.min_ndim) + ', found ndim=' + str(ndim) + '. Full shape received: ' + str(tuple(shape)))
        if spec.dtype is not None:
            if x.dtype.name != spec.dtype:
                raise ValueError('Input ' + str(input_index) + ' of layer ' + layer_name + ' is incompatible with the layer: expected dtype=' + str(spec.dtype) + ', found dtype=' + str(x.dtype))
        shape_as_list = shape.as_list()
        if spec.axes:
            for (axis, value) in spec.axes.items():
                if hasattr(value, 'value'):
                    value = value.value
                if value is not None and shape_as_list[int(axis)] not in {value, None}:
                    raise ValueError('Input ' + str(input_index) + ' of layer ' + layer_name + ' is incompatible with the layer: expected axis ' + str(axis) + ' of input shape to have value ' + str(value) + ' but received input with shape ' + display_shape(x.shape))
        if spec.shape is not None and shape.rank is not None:
            spec_shape = spec.shape
            if spec.allow_last_axis_squeeze:
                if shape_as_list and shape_as_list[-1] == 1:
                    shape_as_list = shape_as_list[:-1]
                if spec_shape and spec_shape[-1] == 1:
                    spec_shape = spec_shape[:-1]
            for (spec_dim, dim) in zip(spec_shape, shape_as_list):
                if spec_dim is not None and dim is not None:
                    if spec_dim != dim:
                        raise ValueError('Input ' + str(input_index) + ' is incompatible with layer ' + layer_name + ': expected shape=' + str(spec.shape) + ', found shape=' + display_shape(x.shape))

def display_shape(shape):
    if False:
        for i in range(10):
            print('nop')
    return str(tuple(shape.as_list()))

def to_tensor_spec(input_spec, default_dtype=None):
    if False:
        return 10
    'Converts a Keras InputSpec object to a TensorSpec.'
    default_dtype = default_dtype or backend.floatx()
    if isinstance(input_spec, InputSpec):
        dtype = input_spec.dtype or default_dtype
        return tensor_spec.TensorSpec(to_tensor_shape(input_spec), dtype)
    return tensor_spec.TensorSpec(None, default_dtype)