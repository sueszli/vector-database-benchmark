"""Module implementing RNN Cells.

This module provides a number of basic commonly used RNN cells, such as LSTM
(Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number of
operators that allow adding dropouts, projections, or embeddings for inputs.
Constructing multi-layer cells is supported by the class `MultiRNNCell`, or by
calling the `rnn` ops several times.
"""
import collections
import warnings
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.layers.legacy_rnn import rnn_cell_wrapper_impl
from tensorflow.python.keras.legacy_tf_layers import base as base_layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
_BIAS_VARIABLE_NAME = 'bias'
_WEIGHTS_VARIABLE_NAME = 'kernel'
ASSERT_LIKE_RNNCELL_ERROR_REGEXP = 'is not an RNNCell'

def _hasattr(obj, attr_name):
    if False:
        i = 10
        return i + 15
    try:
        getattr(obj, attr_name)
    except AttributeError:
        return False
    else:
        return True

def assert_like_rnncell(cell_name, cell):
    if False:
        while True:
            i = 10
    'Raises a TypeError if cell is not like an RNNCell.\n\n  NOTE: Do not rely on the error message (in particular in tests) which can be\n  subject to change to increase readability. Use\n  ASSERT_LIKE_RNNCELL_ERROR_REGEXP.\n\n  Args:\n    cell_name: A string to give a meaningful error referencing to the name of\n      the functionargument.\n    cell: The object which should behave like an RNNCell.\n\n  Raises:\n    TypeError: A human-friendly exception.\n  '
    conditions = [_hasattr(cell, 'output_size'), _hasattr(cell, 'state_size'), _hasattr(cell, 'get_initial_state') or _hasattr(cell, 'zero_state'), callable(cell)]
    errors = ["'output_size' property is missing", "'state_size' property is missing", "either 'zero_state' or 'get_initial_state' method is required", 'is not callable']
    if not all(conditions):
        errors = [error for (error, cond) in zip(errors, conditions) if not cond]
        raise TypeError('The argument {!r} ({}) is not an RNNCell: {}.'.format(cell_name, cell, ', '.join(errors)))

def _concat(prefix, suffix, static=False):
    if False:
        while True:
            i = 10
    'Concat that enables int, Tensor, or TensorShape values.\n\n  This function takes a size specification, which can be an integer, a\n  TensorShape, or a Tensor, and converts it into a concatenated Tensor\n  (if static = False) or a list of integers (if static = True).\n\n  Args:\n    prefix: The prefix; usually the batch size (and/or time step size).\n      (TensorShape, int, or Tensor.)\n    suffix: TensorShape, int, or Tensor.\n    static: If `True`, return a python list with possibly unknown dimensions.\n      Otherwise return a `Tensor`.\n\n  Returns:\n    shape: the concatenation of prefix and suffix.\n\n  Raises:\n    ValueError: if `suffix` is not a scalar or vector (or TensorShape).\n    ValueError: if prefix or suffix was `None` and asked for dynamic\n      Tensors out.\n  '
    if isinstance(prefix, tensor.Tensor):
        p = prefix
        p_static = tensor_util.constant_value(prefix)
        if p.shape.ndims == 0:
            p = array_ops.expand_dims(p, 0)
        elif p.shape.ndims != 1:
            raise ValueError('prefix tensor must be either a scalar or vector, but saw tensor: %s' % p)
    else:
        p = tensor_shape.TensorShape(prefix)
        p_static = p.as_list() if p.ndims is not None else None
        p = constant_op.constant(p.as_list(), dtype=dtypes.int32) if p.is_fully_defined() else None
    if isinstance(suffix, tensor.Tensor):
        s = suffix
        s_static = tensor_util.constant_value(suffix)
        if s.shape.ndims == 0:
            s = array_ops.expand_dims(s, 0)
        elif s.shape.ndims != 1:
            raise ValueError('suffix tensor must be either a scalar or vector, but saw tensor: %s' % s)
    else:
        s = tensor_shape.TensorShape(suffix)
        s_static = s.as_list() if s.ndims is not None else None
        s = constant_op.constant(s.as_list(), dtype=dtypes.int32) if s.is_fully_defined() else None
    if static:
        shape = tensor_shape.TensorShape(p_static).concatenate(s_static)
        shape = shape.as_list() if shape.ndims is not None else None
    else:
        if p is None or s is None:
            raise ValueError('Provided a prefix or suffix of None: %s and %s' % (prefix, suffix))
        shape = array_ops.concat((p, s), 0)
    return shape

def _zero_state_tensors(state_size, batch_size, dtype):
    if False:
        return 10
    'Create tensors of zeros based on state_size, batch_size, and dtype.'

    def get_state_shape(s):
        if False:
            print('Hello World!')
        'Combine s with batch_size to get a proper tensor shape.'
        c = _concat(batch_size, s)
        size = array_ops.zeros(c, dtype=dtype)
        if not context.executing_eagerly():
            c_static = _concat(batch_size, s, static=True)
            size.set_shape(c_static)
        return size
    return nest.map_structure(get_state_shape, state_size)

@tf_export(v1=['nn.rnn_cell.RNNCell'])
class RNNCell(base_layer.Layer):
    """Abstract object representing an RNN cell.

  Every `RNNCell` must have the properties below and implement `call` with
  the signature `(output, next_state) = call(input, state)`.  The optional
  third input argument, `scope`, is allowed for backwards compatibility
  purposes; but should be left off for new subclasses.

  This definition of cell differs from the definition used in the literature.
  In the literature, 'cell' refers to an object with a single scalar output.
  This definition refers to a horizontal array of such units.

  An RNN cell, in the most abstract setting, is anything that has
  a state and performs some operation that takes a matrix of inputs.
  This operation results in an output matrix with `self.output_size` columns.
  If `self.state_size` is an integer, this operation also results in a new
  state matrix with `self.state_size` columns.  If `self.state_size` is a
  (possibly nested tuple of) TensorShape object(s), then it should return a
  matching structure of Tensors having shape `[batch_size].concatenate(s)`
  for each `s` in `self.batch_size`.
  """

    def __init__(self, trainable=True, name=None, dtype=None, **kwargs):
        if False:
            return 10
        super(RNNCell, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self._is_tf_rnn_cell = True

    def __call__(self, inputs, state, scope=None):
        if False:
            for i in range(10):
                print('nop')
        'Run this RNN cell on inputs, starting from the given state.\n\n    Args:\n      inputs: `2-D` tensor with shape `[batch_size, input_size]`.\n      state: if `self.state_size` is an integer, this should be a `2-D Tensor`\n        with shape `[batch_size, self.state_size]`.  Otherwise, if\n        `self.state_size` is a tuple of integers, this should be a tuple with\n        shapes `[batch_size, s] for s in self.state_size`.\n      scope: VariableScope for the created subgraph; defaults to class name.\n\n    Returns:\n      A pair containing:\n\n      - Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.\n      - New state: Either a single `2-D` tensor, or a tuple of tensors matching\n        the arity and shapes of `state`.\n    '
        if scope is not None:
            with vs.variable_scope(scope, custom_getter=self._rnn_get_variable) as scope:
                return super(RNNCell, self).__call__(inputs, state, scope=scope)
        else:
            scope_attrname = 'rnncell_scope'
            scope = getattr(self, scope_attrname, None)
            if scope is None:
                scope = vs.variable_scope(vs.get_variable_scope(), custom_getter=self._rnn_get_variable)
                setattr(self, scope_attrname, scope)
            with scope:
                return super(RNNCell, self).__call__(inputs, state)

    def _rnn_get_variable(self, getter, *args, **kwargs):
        if False:
            print('Hello World!')
        variable = getter(*args, **kwargs)
        if ops.executing_eagerly_outside_functions():
            trainable = variable.trainable
        else:
            trainable = variable in tf_variables.trainable_variables() or (base_layer_utils.is_split_variable(variable) and list(variable)[0] in tf_variables.trainable_variables())
        if trainable and all((variable is not v for v in self._trainable_weights)):
            self._trainable_weights.append(variable)
        elif not trainable and all((variable is not v for v in self._non_trainable_weights)):
            self._non_trainable_weights.append(variable)
        return variable

    @property
    def state_size(self):
        if False:
            i = 10
            return i + 15
        'size(s) of state(s) used by this cell.\n\n    It can be represented by an Integer, a TensorShape or a tuple of Integers\n    or TensorShapes.\n    '
        raise NotImplementedError('Abstract method')

    @property
    def output_size(self):
        if False:
            return 10
        'Integer or TensorShape: size of outputs produced by this cell.'
        raise NotImplementedError('Abstract method')

    def build(self, _):
        if False:
            while True:
                i = 10
        pass

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if False:
            i = 10
            return i + 15
        if inputs is not None:
            inputs = tensor_conversion.convert_to_tensor_v2_with_dispatch(inputs, name='inputs')
            if batch_size is not None:
                if tensor_util.is_tf_type(batch_size):
                    static_batch_size = tensor_util.constant_value(batch_size, partial=True)
                else:
                    static_batch_size = batch_size
                if inputs.shape.dims[0].value != static_batch_size:
                    raise ValueError('batch size from input tensor is different from the input param. Input tensor batch: {}, batch_size: {}'.format(inputs.shape.dims[0].value, batch_size))
            if dtype is not None and inputs.dtype != dtype:
                raise ValueError('dtype from input tensor is different from the input param. Input tensor dtype: {}, dtype: {}'.format(inputs.dtype, dtype))
            batch_size = inputs.shape.dims[0].value or array_ops.shape(inputs)[0]
            dtype = inputs.dtype
        if batch_size is None or dtype is None:
            raise ValueError('batch_size and dtype cannot be None while constructing initial state: batch_size={}, dtype={}'.format(batch_size, dtype))
        return self.zero_state(batch_size, dtype)

    def zero_state(self, batch_size, dtype):
        if False:
            print('Hello World!')
        'Return zero-filled state tensor(s).\n\n    Args:\n      batch_size: int, float, or unit Tensor representing the batch size.\n      dtype: the data type to use for the state.\n\n    Returns:\n      If `state_size` is an int or TensorShape, then the return value is a\n      `N-D` tensor of shape `[batch_size, state_size]` filled with zeros.\n\n      If `state_size` is a nested list or tuple, then the return value is\n      a nested list or tuple (of the same structure) of `2-D` tensors with\n      the shapes `[batch_size, s]` for each s in `state_size`.\n    '
        state_size = self.state_size
        is_eager = context.executing_eagerly()
        if is_eager and _hasattr(self, '_last_zero_state'):
            (last_state_size, last_batch_size, last_dtype, last_output) = getattr(self, '_last_zero_state')
            if last_batch_size == batch_size and last_dtype == dtype and (last_state_size == state_size):
                return last_output
        with backend.name_scope(type(self).__name__ + 'ZeroState'):
            output = _zero_state_tensors(state_size, batch_size, dtype)
        if is_eager:
            self._last_zero_state = (state_size, batch_size, dtype, output)
        return output

    def get_config(self):
        if False:
            i = 10
            return i + 15
        return super(RNNCell, self).get_config()

    @property
    def _use_input_spec_as_call_signature(self):
        if False:
            while True:
                i = 10
        return False

class LayerRNNCell(RNNCell):
    """Subclass of RNNCells that act like proper `tf.Layer` objects.

  For backwards compatibility purposes, most `RNNCell` instances allow their
  `call` methods to instantiate variables via `tf.compat.v1.get_variable`.  The
  underlying
  variable scope thus keeps track of any variables, and returning cached
  versions.  This is atypical of `tf.layer` objects, which separate this
  part of layer building into a `build` method that is only called once.

  Here we provide a subclass for `RNNCell` objects that act exactly as
  `Layer` objects do.  They must provide a `build` method and their
  `call` methods do not access Variables `tf.compat.v1.get_variable`.
  """

    def __call__(self, inputs, state, scope=None, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Run this RNN cell on inputs, starting from the given state.\n\n    Args:\n      inputs: `2-D` tensor with shape `[batch_size, input_size]`.\n      state: if `self.state_size` is an integer, this should be a `2-D Tensor`\n        with shape `[batch_size, self.state_size]`.  Otherwise, if\n        `self.state_size` is a tuple of integers, this should be a tuple with\n        shapes `[batch_size, s] for s in self.state_size`.\n      scope: optional cell scope.\n      *args: Additional positional arguments.\n      **kwargs: Additional keyword arguments.\n\n    Returns:\n      A pair containing:\n\n      - Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.\n      - New state: Either a single `2-D` tensor, or a tuple of tensors matching\n        the arity and shapes of `state`.\n    '
        return base_layer.Layer.__call__(self, inputs, state, *args, scope=scope, **kwargs)

@tf_export(v1=['nn.rnn_cell.BasicRNNCell'])
class BasicRNNCell(LayerRNNCell):
    """The most basic RNN cell.

  Note that this cell is not optimized for performance. Please use
  `tf.contrib.cudnn_rnn.CudnnRNNTanh` for better performance on GPU.

  Args:
    num_units: int, The number of units in the RNN cell.
    activation: Nonlinearity to use.  Default: `tanh`. It could also be string
      that is within Keras activation function names.
    reuse: (optional) Python boolean describing whether to reuse variables in an
      existing scope.  If not `True`, and the existing scope already has the
      given variables, an error is raised.
    name: String, the name of the layer. Layers with the same name will share
      weights, but to avoid mistakes we require reuse=True in such cases.
    dtype: Default dtype of the layer (default of `None` means use the type of
      the first input). Required when `build` is called before `call`.
    **kwargs: Dict, keyword named properties for common layer attributes, like
      `trainable` etc when constructing the cell from configs of get_config().
  """

    def __init__(self, num_units, activation=None, reuse=None, name=None, dtype=None, **kwargs):
        if False:
            i = 10
            return i + 15
        warnings.warn('`tf.nn.rnn_cell.BasicRNNCell` is deprecated and will be removed in a future version. This class is equivalent as `tf.keras.layers.SimpleRNNCell`, and will be replaced by that in Tensorflow 2.0.')
        super(BasicRNNCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        _check_supported_dtypes(self.dtype)
        if context.executing_eagerly() and tf_config.list_logical_devices('GPU'):
            logging.warning('%s: Note that this cell is not optimized for performance. Please use tf.contrib.cudnn_rnn.CudnnRNNTanh for better performance on GPU.', self)
        self.input_spec = input_spec.InputSpec(ndim=2)
        self._num_units = num_units
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh

    @property
    def state_size(self):
        if False:
            print('Hello World!')
        return self._num_units

    @property
    def output_size(self):
        if False:
            for i in range(10):
                print('nop')
        return self._num_units

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if False:
            for i in range(10):
                print('nop')
        if inputs_shape[-1] is None:
            raise ValueError('Expected inputs.shape[-1] to be known, saw shape: %s' % str(inputs_shape))
        _check_supported_dtypes(self.dtype)
        input_depth = inputs_shape[-1]
        self._kernel = self.add_variable(_WEIGHTS_VARIABLE_NAME, shape=[input_depth + self._num_units, self._num_units])
        self._bias = self.add_variable(_BIAS_VARIABLE_NAME, shape=[self._num_units], initializer=init_ops.zeros_initializer(dtype=self.dtype))
        self.built = True

    def call(self, inputs, state):
        if False:
            return 10
        'Most basic RNN: output = new_state = act(W * input + U * state + B).'
        _check_rnn_cell_input_dtypes([inputs, state])
        gate_inputs = math_ops.matmul(array_ops.concat([inputs, state], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
        output = self._activation(gate_inputs)
        return (output, output)

    def get_config(self):
        if False:
            print('Hello World!')
        config = {'num_units': self._num_units, 'activation': activations.serialize(self._activation), 'reuse': self._reuse}
        base_config = super(BasicRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

@tf_export(v1=['nn.rnn_cell.GRUCell'])
class GRUCell(LayerRNNCell):
    """Gated Recurrent Unit cell.

  Note that this cell is not optimized for performance. Please use
  `tf.contrib.cudnn_rnn.CudnnGRU` for better performance on GPU, or
  `tf.contrib.rnn.GRUBlockCellV2` for better performance on CPU.

  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables in an
      existing scope.  If not `True`, and the existing scope already has the
      given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
    name: String, the name of the layer. Layers with the same name will share
      weights, but to avoid mistakes we require reuse=True in such cases.
    dtype: Default dtype of the layer (default of `None` means use the type of
      the first input). Required when `build` is called before `call`.
    **kwargs: Dict, keyword named properties for common layer attributes, like
      `trainable` etc when constructing the cell from configs of get_config().

      References:
    Learning Phrase Representations using RNN Encoder Decoder for Statistical
    Machine Translation:
      [Cho et al., 2014]
      (https://aclanthology.coli.uni-saarland.de/papers/D14-1179/d14-1179)
      ([pdf](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf))
  """

    def __init__(self, num_units, activation=None, reuse=None, kernel_initializer=None, bias_initializer=None, name=None, dtype=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        warnings.warn('`tf.nn.rnn_cell.GRUCell` is deprecated and will be removed in a future version. This class is equivalent as `tf.keras.layers.GRUCell`, and will be replaced by that in Tensorflow 2.0.')
        super(GRUCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        _check_supported_dtypes(self.dtype)
        if context.executing_eagerly() and tf_config.list_logical_devices('GPU'):
            logging.warning('%s: Note that this cell is not optimized for performance. Please use tf.contrib.cudnn_rnn.CudnnGRU for better performance on GPU.', self)
        self.input_spec = input_spec.InputSpec(ndim=2)
        self._num_units = num_units
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)

    @property
    def state_size(self):
        if False:
            for i in range(10):
                print('nop')
        return self._num_units

    @property
    def output_size(self):
        if False:
            return 10
        return self._num_units

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if False:
            print('Hello World!')
        if inputs_shape[-1] is None:
            raise ValueError('Expected inputs.shape[-1] to be known, saw shape: %s' % str(inputs_shape))
        _check_supported_dtypes(self.dtype)
        input_depth = inputs_shape[-1]
        self._gate_kernel = self.add_variable('gates/%s' % _WEIGHTS_VARIABLE_NAME, shape=[input_depth + self._num_units, 2 * self._num_units], initializer=self._kernel_initializer)
        self._gate_bias = self.add_variable('gates/%s' % _BIAS_VARIABLE_NAME, shape=[2 * self._num_units], initializer=self._bias_initializer if self._bias_initializer is not None else init_ops.constant_initializer(1.0, dtype=self.dtype))
        self._candidate_kernel = self.add_variable('candidate/%s' % _WEIGHTS_VARIABLE_NAME, shape=[input_depth + self._num_units, self._num_units], initializer=self._kernel_initializer)
        self._candidate_bias = self.add_variable('candidate/%s' % _BIAS_VARIABLE_NAME, shape=[self._num_units], initializer=self._bias_initializer if self._bias_initializer is not None else init_ops.zeros_initializer(dtype=self.dtype))
        self.built = True

    def call(self, inputs, state):
        if False:
            return 10
        'Gated recurrent unit (GRU) with nunits cells.'
        _check_rnn_cell_input_dtypes([inputs, state])
        gate_inputs = math_ops.matmul(array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)
        value = math_ops.sigmoid(gate_inputs)
        (r, u) = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        r_state = r * state
        candidate = math_ops.matmul(array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)
        c = self._activation(candidate)
        new_h = u * state + (1 - u) * c
        return (new_h, new_h)

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        config = {'num_units': self._num_units, 'kernel_initializer': initializers.serialize(self._kernel_initializer), 'bias_initializer': initializers.serialize(self._bias_initializer), 'activation': activations.serialize(self._activation), 'reuse': self._reuse}
        base_config = super(GRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
_LSTMStateTuple = collections.namedtuple('LSTMStateTuple', ('c', 'h'))

@tf_export(v1=['nn.rnn_cell.LSTMStateTuple'])
class LSTMStateTuple(_LSTMStateTuple):
    """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

  Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
  and `h` is the output.

  Only used when `state_is_tuple=True`.
  """
    __slots__ = ()

    @property
    def dtype(self):
        if False:
            return 10
        (c, h) = self
        if c.dtype != h.dtype:
            raise TypeError('Inconsistent internal state: %s vs %s' % (str(c.dtype), str(h.dtype)))
        return c.dtype

@tf_export(v1=['nn.rnn_cell.BasicLSTMCell'])
class BasicLSTMCell(LayerRNNCell):
    """DEPRECATED: Please use `tf.compat.v1.nn.rnn_cell.LSTMCell` instead.

  Basic LSTM recurrent network cell.

  The implementation is based on

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  For advanced models, please use the full `tf.compat.v1.nn.rnn_cell.LSTMCell`
  that follows.

  Note that this cell is not optimized for performance. Please use
  `tf.contrib.cudnn_rnn.CudnnLSTM` for better performance on GPU, or
  `tf.contrib.rnn.LSTMBlockCell` and `tf.contrib.rnn.LSTMBlockFusedCell` for
  better performance on CPU.
  """

    def __init__(self, num_units, forget_bias=1.0, state_is_tuple=True, activation=None, reuse=None, name=None, dtype=None, **kwargs):
        if False:
            while True:
                i = 10
        'Initialize the basic LSTM cell.\n\n    Args:\n      num_units: int, The number of units in the LSTM cell.\n      forget_bias: float, The bias added to forget gates (see above). Must set\n        to `0.0` manually when restoring from CudnnLSTM-trained checkpoints.\n      state_is_tuple: If True, accepted and returned states are 2-tuples of the\n        `c_state` and `m_state`.  If False, they are concatenated along the\n        column axis.  The latter behavior will soon be deprecated.\n      activation: Activation function of the inner states.  Default: `tanh`. It\n        could also be string that is within Keras activation function names.\n      reuse: (optional) Python boolean describing whether to reuse variables in\n        an existing scope.  If not `True`, and the existing scope already has\n        the given variables, an error is raised.\n      name: String, the name of the layer. Layers with the same name will share\n        weights, but to avoid mistakes we require reuse=True in such cases.\n      dtype: Default dtype of the layer (default of `None` means use the type of\n        the first input). Required when `build` is called before `call`.\n      **kwargs: Dict, keyword named properties for common layer attributes, like\n        `trainable` etc when constructing the cell from configs of get_config().\n        When restoring from CudnnLSTM-trained checkpoints, must use\n        `CudnnCompatibleLSTMCell` instead.\n    '
        warnings.warn('`tf.nn.rnn_cell.BasicLSTMCell` is deprecated and will be removed in a future version. This class is equivalent as `tf.keras.layers.LSTMCell`, and will be replaced by that in Tensorflow 2.0.')
        super(BasicLSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        _check_supported_dtypes(self.dtype)
        if not state_is_tuple:
            logging.warning('%s: Using a concatenated state is slower and will soon be deprecated.  Use state_is_tuple=True.', self)
        if context.executing_eagerly() and tf_config.list_logical_devices('GPU'):
            logging.warning('%s: Note that this cell is not optimized for performance. Please use tf.contrib.cudnn_rnn.CudnnLSTM for better performance on GPU.', self)
        self.input_spec = input_spec.InputSpec(ndim=2)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh

    @property
    def state_size(self):
        if False:
            return 10
        return LSTMStateTuple(self._num_units, self._num_units) if self._state_is_tuple else 2 * self._num_units

    @property
    def output_size(self):
        if False:
            i = 10
            return i + 15
        return self._num_units

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if False:
            return 10
        if inputs_shape[-1] is None:
            raise ValueError('Expected inputs.shape[-1] to be known, saw shape: %s' % str(inputs_shape))
        _check_supported_dtypes(self.dtype)
        input_depth = inputs_shape[-1]
        h_depth = self._num_units
        self._kernel = self.add_variable(_WEIGHTS_VARIABLE_NAME, shape=[input_depth + h_depth, 4 * self._num_units])
        self._bias = self.add_variable(_BIAS_VARIABLE_NAME, shape=[4 * self._num_units], initializer=init_ops.zeros_initializer(dtype=self.dtype))
        self.built = True

    def call(self, inputs, state):
        if False:
            while True:
                i = 10
        'Long short-term memory cell (LSTM).\n\n    Args:\n      inputs: `2-D` tensor with shape `[batch_size, input_size]`.\n      state: An `LSTMStateTuple` of state tensors, each shaped `[batch_size,\n        num_units]`, if `state_is_tuple` has been set to `True`.  Otherwise, a\n        `Tensor` shaped `[batch_size, 2 * num_units]`.\n\n    Returns:\n      A pair containing the new hidden state, and the new state (either a\n        `LSTMStateTuple` or a concatenated state, depending on\n        `state_is_tuple`).\n    '
        _check_rnn_cell_input_dtypes([inputs, state])
        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=dtypes.int32)
        if self._state_is_tuple:
            (c, h) = state
        else:
            (c, h) = array_ops.split(value=state, num_or_size_splits=2, axis=one)
        gate_inputs = math_ops.matmul(array_ops.concat([inputs, h], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
        (i, j, f, o) = array_ops.split(value=gate_inputs, num_or_size_splits=4, axis=one)
        forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
        add = math_ops.add
        multiply = math_ops.multiply
        new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))), multiply(sigmoid(i), self._activation(j)))
        new_h = multiply(self._activation(new_c), sigmoid(o))
        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return (new_h, new_state)

    def get_config(self):
        if False:
            return 10
        config = {'num_units': self._num_units, 'forget_bias': self._forget_bias, 'state_is_tuple': self._state_is_tuple, 'activation': activations.serialize(self._activation), 'reuse': self._reuse}
        base_config = super(BasicLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

@tf_export(v1=['nn.rnn_cell.LSTMCell'])
class LSTMCell(LayerRNNCell):
    """Long short-term memory unit (LSTM) recurrent network cell.

  The default non-peephole implementation is based on (Gers et al., 1999).
  The peephole implementation is based on (Sak et al., 2014).

  The class uses optional peep-hole connections, optional cell clipping, and
  an optional projection layer.

  Note that this cell is not optimized for performance. Please use
  `tf.contrib.cudnn_rnn.CudnnLSTM` for better performance on GPU, or
  `tf.contrib.rnn.LSTMBlockCell` and `tf.contrib.rnn.LSTMBlockFusedCell` for
  better performance on CPU.
  References:
    Long short-term memory recurrent neural network architectures for large
    scale acoustic modeling:
      [Sak et al., 2014]
      (https://www.isca-speech.org/archive/interspeech_2014/i14_0338.html)
      ([pdf]
      (https://www.isca-speech.org/archive/archive_papers/interspeech_2014/i14_0338.pdf))
    Learning to forget:
      [Gers et al., 1999]
      (http://digital-library.theiet.org/content/conferences/10.1049/cp_19991218)
      ([pdf](https://arxiv.org/pdf/1409.2329.pdf))
    Long Short-Term Memory:
      [Hochreiter et al., 1997]
      (https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735)
      ([pdf](http://ml.jku.at/publications/older/3504.pdf))
  """

    def __init__(self, num_units, use_peepholes=False, cell_clip=None, initializer=None, num_proj=None, proj_clip=None, num_unit_shards=None, num_proj_shards=None, forget_bias=1.0, state_is_tuple=True, activation=None, reuse=None, name=None, dtype=None, **kwargs):
        if False:
            print('Hello World!')
        'Initialize the parameters for an LSTM cell.\n\n    Args:\n      num_units: int, The number of units in the LSTM cell.\n      use_peepholes: bool, set True to enable diagonal/peephole connections.\n      cell_clip: (optional) A float value, if provided the cell state is clipped\n        by this value prior to the cell output activation.\n      initializer: (optional) The initializer to use for the weight and\n        projection matrices.\n      num_proj: (optional) int, The output dimensionality for the projection\n        matrices.  If None, no projection is performed.\n      proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is\n        provided, then the projected values are clipped elementwise to within\n        `[-proj_clip, proj_clip]`.\n      num_unit_shards: Deprecated, will be removed by Jan. 2017. Use a\n        variable_scope partitioner instead.\n      num_proj_shards: Deprecated, will be removed by Jan. 2017. Use a\n        variable_scope partitioner instead.\n      forget_bias: Biases of the forget gate are initialized by default to 1 in\n        order to reduce the scale of forgetting at the beginning of the\n        training. Must set it manually to `0.0` when restoring from CudnnLSTM\n        trained checkpoints.\n      state_is_tuple: If True, accepted and returned states are 2-tuples of the\n        `c_state` and `m_state`.  If False, they are concatenated along the\n        column axis.  This latter behavior will soon be deprecated.\n      activation: Activation function of the inner states.  Default: `tanh`. It\n        could also be string that is within Keras activation function names.\n      reuse: (optional) Python boolean describing whether to reuse variables in\n        an existing scope.  If not `True`, and the existing scope already has\n        the given variables, an error is raised.\n      name: String, the name of the layer. Layers with the same name will share\n        weights, but to avoid mistakes we require reuse=True in such cases.\n      dtype: Default dtype of the layer (default of `None` means use the type of\n        the first input). Required when `build` is called before `call`.\n      **kwargs: Dict, keyword named properties for common layer attributes, like\n        `trainable` etc when constructing the cell from configs of get_config().\n        When restoring from CudnnLSTM-trained checkpoints, use\n        `CudnnCompatibleLSTMCell` instead.\n    '
        warnings.warn('`tf.nn.rnn_cell.LSTMCell` is deprecated and will be removed in a future version. This class is equivalent as `tf.keras.layers.LSTMCell`, and will be replaced by that in Tensorflow 2.0.')
        super(LSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        _check_supported_dtypes(self.dtype)
        if not state_is_tuple:
            logging.warning('%s: Using a concatenated state is slower and will soon be deprecated.  Use state_is_tuple=True.', self)
        if num_unit_shards is not None or num_proj_shards is not None:
            logging.warning('%s: The num_unit_shards and proj_unit_shards parameters are deprecated and will be removed in Jan 2017.  Use a variable scope with a partitioner instead.', self)
        if context.executing_eagerly() and tf_config.list_logical_devices('GPU'):
            logging.warning('%s: Note that this cell is not optimized for performance. Please use tf.contrib.cudnn_rnn.CudnnLSTM for better performance on GPU.', self)
        self.input_spec = input_spec.InputSpec(ndim=2)
        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializers.get(initializer)
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh
        if num_proj:
            self._state_size = LSTMStateTuple(num_units, num_proj) if state_is_tuple else num_units + num_proj
            self._output_size = num_proj
        else:
            self._state_size = LSTMStateTuple(num_units, num_units) if state_is_tuple else 2 * num_units
            self._output_size = num_units

    @property
    def state_size(self):
        if False:
            return 10
        return self._state_size

    @property
    def output_size(self):
        if False:
            while True:
                i = 10
        return self._output_size

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if False:
            return 10
        if inputs_shape[-1] is None:
            raise ValueError('Expected inputs.shape[-1] to be known, saw shape: %s' % str(inputs_shape))
        _check_supported_dtypes(self.dtype)
        input_depth = inputs_shape[-1]
        h_depth = self._num_units if self._num_proj is None else self._num_proj
        maybe_partitioner = partitioned_variables.fixed_size_partitioner(self._num_unit_shards) if self._num_unit_shards is not None else None
        self._kernel = self.add_variable(_WEIGHTS_VARIABLE_NAME, shape=[input_depth + h_depth, 4 * self._num_units], initializer=self._initializer, partitioner=maybe_partitioner)
        if self.dtype is None:
            initializer = init_ops.zeros_initializer
        else:
            initializer = init_ops.zeros_initializer(dtype=self.dtype)
        self._bias = self.add_variable(_BIAS_VARIABLE_NAME, shape=[4 * self._num_units], initializer=initializer)
        if self._use_peepholes:
            self._w_f_diag = self.add_variable('w_f_diag', shape=[self._num_units], initializer=self._initializer)
            self._w_i_diag = self.add_variable('w_i_diag', shape=[self._num_units], initializer=self._initializer)
            self._w_o_diag = self.add_variable('w_o_diag', shape=[self._num_units], initializer=self._initializer)
        if self._num_proj is not None:
            maybe_proj_partitioner = partitioned_variables.fixed_size_partitioner(self._num_proj_shards) if self._num_proj_shards is not None else None
            self._proj_kernel = self.add_variable('projection/%s' % _WEIGHTS_VARIABLE_NAME, shape=[self._num_units, self._num_proj], initializer=self._initializer, partitioner=maybe_proj_partitioner)
        self.built = True

    def call(self, inputs, state):
        if False:
            print('Hello World!')
        'Run one step of LSTM.\n\n    Args:\n      inputs: input Tensor, must be 2-D, `[batch, input_size]`.\n      state: if `state_is_tuple` is False, this must be a state Tensor, `2-D,\n        [batch, state_size]`.  If `state_is_tuple` is True, this must be a tuple\n        of state Tensors, both `2-D`, with column sizes `c_state` and `m_state`.\n\n    Returns:\n      A tuple containing:\n\n      - A `2-D, [batch, output_dim]`, Tensor representing the output of the\n        LSTM after reading `inputs` when previous state was `state`.\n        Here output_dim is:\n           num_proj if num_proj was set,\n           num_units otherwise.\n      - Tensor(s) representing the new state of LSTM after reading `inputs` when\n        the previous state was `state`.  Same type and shape(s) as `state`.\n\n    Raises:\n      ValueError: If input size cannot be inferred from inputs via\n        static shape inference.\n    '
        _check_rnn_cell_input_dtypes([inputs, state])
        num_proj = self._num_units if self._num_proj is None else self._num_proj
        sigmoid = math_ops.sigmoid
        if self._state_is_tuple:
            (c_prev, m_prev) = state
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])
        input_size = inputs.get_shape().with_rank(2).dims[1].value
        if input_size is None:
            raise ValueError('Could not infer input size from inputs.get_shape()[-1]')
        lstm_matrix = math_ops.matmul(array_ops.concat([inputs, m_prev], 1), self._kernel)
        lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)
        (i, j, f, o) = array_ops.split(value=lstm_matrix, num_or_size_splits=4, axis=1)
        if self._use_peepholes:
            c = sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev + sigmoid(i + self._w_i_diag * c_prev) * self._activation(j)
        else:
            c = sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) * self._activation(j)
        if self._cell_clip is not None:
            c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
        if self._use_peepholes:
            m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
        else:
            m = sigmoid(o) * self._activation(c)
        if self._num_proj is not None:
            m = math_ops.matmul(m, self._proj_kernel)
            if self._proj_clip is not None:
                m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
        new_state = LSTMStateTuple(c, m) if self._state_is_tuple else array_ops.concat([c, m], 1)
        return (m, new_state)

    def get_config(self):
        if False:
            return 10
        config = {'num_units': self._num_units, 'use_peepholes': self._use_peepholes, 'cell_clip': self._cell_clip, 'initializer': initializers.serialize(self._initializer), 'num_proj': self._num_proj, 'proj_clip': self._proj_clip, 'num_unit_shards': self._num_unit_shards, 'num_proj_shards': self._num_proj_shards, 'forget_bias': self._forget_bias, 'state_is_tuple': self._state_is_tuple, 'activation': activations.serialize(self._activation), 'reuse': self._reuse}
        base_config = super(LSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class _RNNCellWrapperV1(RNNCell):
    """Base class for cells wrappers V1 compatibility.

  This class along with `_RNNCellWrapperV2` allows to define cells wrappers that
  are compatible with V1 and V2, and defines helper methods for this purpose.
  """

    def __init__(self, cell, *args, **kwargs):
        if False:
            print('Hello World!')
        super(_RNNCellWrapperV1, self).__init__(*args, **kwargs)
        assert_like_rnncell('cell', cell)
        self.cell = cell
        if isinstance(cell, trackable.Trackable):
            self._track_trackable(self.cell, name='cell')

    def _call_wrapped_cell(self, inputs, state, cell_call_fn, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Calls the wrapped cell and performs the wrapping logic.\n\n    This method is called from the wrapper's `call` or `__call__` methods.\n\n    Args:\n      inputs: A tensor with wrapped cell's input.\n      state: A tensor or tuple of tensors with wrapped cell's state.\n      cell_call_fn: Wrapped cell's method to use for step computation (cell's\n        `__call__` or 'call' method).\n      **kwargs: Additional arguments.\n\n    Returns:\n      A pair containing:\n      - Output: A tensor with cell's output.\n      - New state: A tensor or tuple of tensors with new wrapped cell's state.\n    "
        raise NotImplementedError

    def __call__(self, inputs, state, scope=None):
        if False:
            print('Hello World!')
        "Runs the RNN cell step computation.\n\n    We assume that the wrapped RNNCell is being built within its `__call__`\n    method. We directly use the wrapped cell's `__call__` in the overridden\n    wrapper `__call__` method.\n\n    This allows to use the wrapped cell and the non-wrapped cell equivalently\n    when using `__call__`.\n\n    Args:\n      inputs: A tensor with wrapped cell's input.\n      state: A tensor or tuple of tensors with wrapped cell's state.\n      scope: VariableScope for the subgraph created in the wrapped cells'\n        `__call__`.\n\n    Returns:\n      A pair containing:\n\n      - Output: A tensor with cell's output.\n      - New state: A tensor or tuple of tensors with new wrapped cell's state.\n    "
        return self._call_wrapped_cell(inputs, state, cell_call_fn=self.cell.__call__, scope=scope)

    def get_config(self):
        if False:
            return 10
        config = {'cell': {'class_name': self.cell.__class__.__name__, 'config': self.cell.get_config()}}
        base_config = super(_RNNCellWrapperV1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if False:
            print('Hello World!')
        config = config.copy()
        cell = config.pop('cell')
        try:
            assert_like_rnncell('cell', cell)
            return cls(cell, **config)
        except TypeError:
            raise ValueError('RNNCellWrapper cannot reconstruct the wrapped cell. Please overwrite the cell in the config with a RNNCell instance.')

@tf_export(v1=['nn.rnn_cell.DropoutWrapper'])
class DropoutWrapper(rnn_cell_wrapper_impl.DropoutWrapperBase, _RNNCellWrapperV1):
    """Operator adding dropout to inputs and outputs of the given cell."""

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(DropoutWrapper, self).__init__(*args, **kwargs)
    __init__.__doc__ = rnn_cell_wrapper_impl.DropoutWrapperBase.__init__.__doc__

@tf_export(v1=['nn.rnn_cell.ResidualWrapper'])
class ResidualWrapper(rnn_cell_wrapper_impl.ResidualWrapperBase, _RNNCellWrapperV1):
    """RNNCell wrapper that ensures cell inputs are added to the outputs."""

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(ResidualWrapper, self).__init__(*args, **kwargs)
    __init__.__doc__ = rnn_cell_wrapper_impl.ResidualWrapperBase.__init__.__doc__

@tf_export(v1=['nn.rnn_cell.DeviceWrapper'])
class DeviceWrapper(rnn_cell_wrapper_impl.DeviceWrapperBase, _RNNCellWrapperV1):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(DeviceWrapper, self).__init__(*args, **kwargs)
    __init__.__doc__ = rnn_cell_wrapper_impl.DeviceWrapperBase.__init__.__doc__

@tf_export(v1=['nn.rnn_cell.MultiRNNCell'])
class MultiRNNCell(RNNCell):
    """RNN cell composed sequentially of multiple simple cells.

  Example:

  ```python
  num_units = [128, 64]
  cells = [BasicLSTMCell(num_units=n) for n in num_units]
  stacked_rnn_cell = MultiRNNCell(cells)
  ```
  """

    def __init__(self, cells, state_is_tuple=True):
        if False:
            return 10
        'Create a RNN cell composed sequentially of a number of RNNCells.\n\n    Args:\n      cells: list of RNNCells that will be composed in this order.\n      state_is_tuple: If True, accepted and returned states are n-tuples, where\n        `n = len(cells)`.  If False, the states are all concatenated along the\n        column axis.  This latter behavior will soon be deprecated.\n\n    Raises:\n      ValueError: if cells is empty (not allowed), or at least one of the cells\n        returns a state tuple but the flag `state_is_tuple` is `False`.\n    '
        logging.warning('`tf.nn.rnn_cell.MultiRNNCell` is deprecated. This class is equivalent as `tf.keras.layers.StackedRNNCells`, and will be replaced by that in Tensorflow 2.0.')
        super(MultiRNNCell, self).__init__()
        if not cells:
            raise ValueError('Must specify at least one cell for MultiRNNCell.')
        if not nest.is_nested(cells):
            raise TypeError('cells must be a list or tuple, but saw: %s.' % cells)
        if len(set((id(cell) for cell in cells))) < len(cells):
            logging.log_first_n(logging.WARN, 'At least two cells provided to MultiRNNCell are the same object and will share weights.', 1)
        self._cells = cells
        for (cell_number, cell) in enumerate(self._cells):
            if isinstance(cell, trackable.Trackable):
                self._track_trackable(cell, name='cell-%d' % (cell_number,))
        self._state_is_tuple = state_is_tuple
        if not state_is_tuple:
            if any((nest.is_nested(c.state_size) for c in self._cells)):
                raise ValueError('Some cells return tuples of states, but the flag state_is_tuple is not set.  State sizes are: %s' % str([c.state_size for c in self._cells]))

    @property
    def state_size(self):
        if False:
            for i in range(10):
                print('nop')
        if self._state_is_tuple:
            return tuple((cell.state_size for cell in self._cells))
        else:
            return sum((cell.state_size for cell in self._cells))

    @property
    def output_size(self):
        if False:
            print('Hello World!')
        return self._cells[-1].output_size

    def zero_state(self, batch_size, dtype):
        if False:
            for i in range(10):
                print('nop')
        with backend.name_scope(type(self).__name__ + 'ZeroState'):
            if self._state_is_tuple:
                return tuple((cell.zero_state(batch_size, dtype) for cell in self._cells))
            else:
                return super(MultiRNNCell, self).zero_state(batch_size, dtype)

    @property
    def trainable_weights(self):
        if False:
            while True:
                i = 10
        if not self.trainable:
            return []
        weights = []
        for cell in self._cells:
            if isinstance(cell, base_layer.Layer):
                weights += cell.trainable_weights
        return weights

    @property
    def non_trainable_weights(self):
        if False:
            i = 10
            return i + 15
        weights = []
        for cell in self._cells:
            if isinstance(cell, base_layer.Layer):
                weights += cell.non_trainable_weights
        if not self.trainable:
            trainable_weights = []
            for cell in self._cells:
                if isinstance(cell, base_layer.Layer):
                    trainable_weights += cell.trainable_weights
            return trainable_weights + weights
        return weights

    def call(self, inputs, state):
        if False:
            return 10
        'Run this multi-layer cell on inputs, starting from state.'
        cur_state_pos = 0
        cur_inp = inputs
        new_states = []
        for (i, cell) in enumerate(self._cells):
            with vs.variable_scope('cell_%d' % i):
                if self._state_is_tuple:
                    if not nest.is_nested(state):
                        raise ValueError('Expected state to be a tuple of length %d, but received: %s' % (len(self.state_size), state))
                    cur_state = state[i]
                else:
                    cur_state = array_ops.slice(state, [0, cur_state_pos], [-1, cell.state_size])
                    cur_state_pos += cell.state_size
                (cur_inp, new_state) = cell(cur_inp, cur_state)
                new_states.append(new_state)
        new_states = tuple(new_states) if self._state_is_tuple else array_ops.concat(new_states, 1)
        return (cur_inp, new_states)

def _check_rnn_cell_input_dtypes(inputs):
    if False:
        i = 10
        return i + 15
    'Check whether the input tensors are with supported dtypes.\n\n  Default RNN cells only support floats and complex as its dtypes since the\n  activation function (tanh and sigmoid) only allow those types. This function\n  will throw a proper error message if the inputs is not in a supported type.\n\n  Args:\n    inputs: tensor or nested structure of tensors that are feed to RNN cell as\n      input or state.\n\n  Raises:\n    ValueError: if any of the input tensor are not having dtypes of float or\n      complex.\n  '
    for t in nest.flatten(inputs):
        _check_supported_dtypes(t.dtype)

def _check_supported_dtypes(dtype):
    if False:
        while True:
            i = 10
    if dtype is None:
        return
    dtype = dtypes.as_dtype(dtype)
    if not (dtype.is_floating or dtype.is_complex):
        raise ValueError('RNN cell only supports floating point inputs, but saw dtype: %s' % dtype)