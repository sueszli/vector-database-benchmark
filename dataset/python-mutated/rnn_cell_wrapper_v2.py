"""Module implementing for RNN wrappers for TF v2."""
from tensorflow.python.keras.layers import recurrent
from tensorflow.python.keras.layers.legacy_rnn import rnn_cell_wrapper_impl
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export

class _RNNCellWrapperV2(recurrent.AbstractRNNCell):
    """Base class for cells wrappers V2 compatibility.

  This class along with `rnn_cell_impl._RNNCellWrapperV1` allows to define
  wrappers that are compatible with V1 and V2, and defines helper methods for
  this purpose.
  """

    def __init__(self, cell, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(_RNNCellWrapperV2, self).__init__(*args, **kwargs)
        self.cell = cell
        cell_call_spec = tf_inspect.getfullargspec(cell.call)
        self._expects_training_arg = 'training' in cell_call_spec.args or cell_call_spec.varkw is not None

    def call(self, inputs, state, **kwargs):
        if False:
            i = 10
            return i + 15
        "Runs the RNN cell step computation.\n\n    When `call` is being used, we assume that the wrapper object has been built,\n    and therefore the wrapped cells has been built via its `build` method and\n    its `call` method can be used directly.\n\n    This allows to use the wrapped cell and the non-wrapped cell equivalently\n    when using `call` and `build`.\n\n    Args:\n      inputs: A tensor with wrapped cell's input.\n      state: A tensor or tuple of tensors with wrapped cell's state.\n      **kwargs: Additional arguments passed to the wrapped cell's `call`.\n\n    Returns:\n      A pair containing:\n\n      - Output: A tensor with cell's output.\n      - New state: A tensor or tuple of tensors with new wrapped cell's state.\n    "
        return self._call_wrapped_cell(inputs, state, cell_call_fn=self.cell.call, **kwargs)

    def build(self, inputs_shape):
        if False:
            for i in range(10):
                print('nop')
        'Builds the wrapped cell.'
        self.cell.build(inputs_shape)
        self.built = True

    def get_config(self):
        if False:
            return 10
        config = {'cell': {'class_name': self.cell.__class__.__name__, 'config': self.cell.get_config()}}
        base_config = super(_RNNCellWrapperV2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if False:
            while True:
                i = 10
        config = config.copy()
        from tensorflow.python.keras.layers.serialization import deserialize as deserialize_layer
        cell = deserialize_layer(config.pop('cell'), custom_objects=custom_objects)
        return cls(cell, **config)

@deprecated(None, 'Please use tf.keras.layers.RNN instead.')
@tf_export('nn.RNNCellDropoutWrapper', v1=[])
class DropoutWrapper(rnn_cell_wrapper_impl.DropoutWrapperBase, _RNNCellWrapperV2):
    """Operator adding dropout to inputs and outputs of the given cell."""

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(DropoutWrapper, self).__init__(*args, **kwargs)
        if isinstance(self.cell, recurrent.LSTMCell):
            raise ValueError('keras LSTM cell does not work with DropoutWrapper. Please use LSTMCell(dropout=x, recurrent_dropout=y) instead.')
    __init__.__doc__ = rnn_cell_wrapper_impl.DropoutWrapperBase.__init__.__doc__

@deprecated(None, 'Please use tf.keras.layers.RNN instead.')
@tf_export('nn.RNNCellResidualWrapper', v1=[])
class ResidualWrapper(rnn_cell_wrapper_impl.ResidualWrapperBase, _RNNCellWrapperV2):
    """RNNCell wrapper that ensures cell inputs are added to the outputs."""

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(ResidualWrapper, self).__init__(*args, **kwargs)
    __init__.__doc__ = rnn_cell_wrapper_impl.ResidualWrapperBase.__init__.__doc__

@deprecated(None, 'Please use tf.keras.layers.RNN instead.')
@tf_export('nn.RNNCellDeviceWrapper', v1=[])
class DeviceWrapper(rnn_cell_wrapper_impl.DeviceWrapperBase, _RNNCellWrapperV2):
    """Operator that ensures an RNNCell runs on a particular device."""

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(DeviceWrapper, self).__init__(*args, **kwargs)
    __init__.__doc__ = rnn_cell_wrapper_impl.DeviceWrapperBase.__init__.__doc__