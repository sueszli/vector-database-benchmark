"""A shim layer for working with functions exported/restored from saved models.

This functionality should ultimately be moved into a first-class core API.
"""
import numpy
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base as trackable

@registration.register_tf_serializable()
class TrackableConstant(trackable.Trackable):
    """Trackable class for captured constants."""
    __slots__ = ('capture', 'function', '_exported_tensor')

    def __init__(self, capture, function):
        if False:
            return 10
        self.capture = capture
        self.function = function
        self._exported_tensor = None

    def _export_to_saved_model_graph(self, tensor_map, **unused_kwargs):
        if False:
            while True:
                i = 10
        capture_constant_value = tensor_util.constant_value(self.capture)
        if capture_constant_value is None:
            raise ValueError(f'Unable to save function {self.function.name} because it captures graph tensor {self.capture} from a parent function which cannot be converted to a constant with `tf.get_static_value`.')
        if numpy.prod(self.capture.shape.as_list()) > 1 and numpy.all(capture_constant_value == capture_constant_value.flat[0]):
            copied_tensor = constant_op.constant(capture_constant_value.flat[0], dtype=self.capture.dtype, shape=self.capture.shape)
        else:
            copied_tensor = constant_op.constant(capture_constant_value)
        tensor_map[self.capture] = copied_tensor
        self._exported_tensor = copied_tensor
        return [self.capture]

    def _serialize_to_proto(self, object_proto=None, **kwargs):
        if False:
            while True:
                i = 10
        object_proto.constant.operation = self._exported_tensor.op.name

    @classmethod
    def _deserialize_from_proto(cls, object_proto, operation_attributes, **kwargs):
        if False:
            print('Hello World!')
        tensor_proto = operation_attributes[object_proto.constant.operation]['value'].tensor
        ndarray = tensor_util.MakeNdarray(tensor_proto)
        if dtypes.as_dtype(tensor_proto.dtype) == dtypes.string:
            with ops.device('CPU'):
                imported_constant = constant_op.constant(ndarray)
        else:
            imported_constant = constant_op.constant(ndarray)
        return imported_constant