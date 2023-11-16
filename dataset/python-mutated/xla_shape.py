"""XLA Shape utilities."""
import numpy as _np
from local_xla.xla import xla_data_pb2
from xla.python_api import types_

class Shape(object):
    """Wraps a xla_data_pb2.ShapeProto message with a convenient Python type.

  Provides direct access to the underlying xla_data_pb2.ShapeProto message in
  the
  message attribute, along with accessor wrappers to the message's fields.
  Avoid direct access to .message unless interacting directly with protobuf APIs
  like CopyFrom. In other words, prefer hauling the shape around in a Shape, and
  only access .message when strictly required by the protobuf API.
  """

    def __init__(self, element_type, dimensions, layout=None):
        if False:
            print('Hello World!')
        'Creates a new XLA Shape.\n\n    Args:\n      element_type: element type from xla_data_pb2.\n      dimensions: sequence of dimensions sizes (integers), or sequence\n        of Shapes in the case of a tuple, i.e. when element_type is\n        TUPLE.\n      layout: optional minor_to_major sequence for layout. If not given, the\n        default major-to-minor layout is used.\n\n    Raises:\n      ValueError: if element_type is TUPLE but dimensions are not Shape objects.\n    '
        self.message = xla_data_pb2.ShapeProto()
        self.message.element_type = element_type
        if element_type == xla_data_pb2.TUPLE:
            if not all((isinstance(subshape, Shape) for subshape in dimensions)):
                raise ValueError('XLA tuple requires sequence of Shape objects as dimensions')
            self._tuple_shapes = tuple(dimensions)
            for component_shape in self._tuple_shapes:
                component_message = self.message.tuple_shapes.add()
                component_message.CopyFrom(component_shape.message)
        else:
            self.message.dimensions.extend(dimensions)
            if layout is None:
                layout = list(reversed(range(len(dimensions))))
            self.message.layout.minor_to_major.extend(layout)

    def element_type(self):
        if False:
            print('Hello World!')
        return self.message.element_type

    def is_tuple(self):
        if False:
            print('Hello World!')
        return self.element_type() == xla_data_pb2.TUPLE

    def dimensions(self):
        if False:
            i = 10
            return i + 15
        if self.is_tuple():
            raise ValueError('Tuple shape has no dimensions. Try tuple_shapes()?')
        return self.message.dimensions

    def tuple_shapes(self):
        if False:
            i = 10
            return i + 15
        'If this is a tuple, returns its sequence of constituent Shape objects.\n\n    Returns:\n      Tuple sub-shapes.\n\n    Raises:\n      ValueError: if this is not a tuple.\n    '
        if not self.is_tuple():
            raise ValueError('tuple_shapes() called on a non-tuple shape')
        return self._tuple_shapes

    def layout(self):
        if False:
            while True:
                i = 10
        return self.message.layout

    @staticmethod
    def from_pyval(pyval):
        if False:
            return 10
        return CreateShapeFromNumpy(pyval)

def _CreateShapeFromNumpy(ndarray):
    if False:
        i = 10
        return i + 15
    'Create a Shape from a given Numpy array.\n\n  Args:\n    ndarray: Numpy array.\n\n  Returns:\n    A Shape object.\n  '
    element_type = types_.MAP_DTYPE_TO_RECORD[str(ndarray.dtype)].primitive_type
    dimensions = ndarray.shape
    if _np.isfortran(ndarray):
        layout = range(ndarray.ndim)
    else:
        layout = list(reversed(range(ndarray.ndim)))
    return Shape(element_type, dimensions, layout)

def CreateShapeFromNumpy(value):
    if False:
        print('Hello World!')
    'Create a Shape from a Numpy array or a nested tuple structure thereof.\n\n  Args:\n    value: Numpy array or (possibly nested) tuple structure that bottoms out in\n      Numpy arrays.\n\n  Returns:\n    A Shape object.\n  '
    if isinstance(value, tuple):
        return Shape(xla_data_pb2.TUPLE, [CreateShapeFromNumpy(component) for component in value])
    else:
        return _CreateShapeFromNumpy(value)

def CreateShapeFromDtypeAndTuple(dtype, shape_tuple):
    if False:
        i = 10
        return i + 15
    "Create a shape from a Numpy dtype and a sequence of nonnegative integers.\n\n  Args:\n    dtype: a numpy dtype, e.g. np.dtype('int32').\n    shape_tuple: a sequence of nonnegative integers.\n\n  Returns:\n    A Shape object.\n  "
    element_type = types_.MAP_DTYPE_TO_RECORD[str(dtype)].primitive_type
    return Shape(element_type, shape_tuple)