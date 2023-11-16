import logging
import numpy as np
import six
from coremltools.converters.mil.mil.types.symbolic import is_symbolic
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.types.type_mapping import numpy_type_to_builtin_type, is_builtin

class ClassifierConfig(object):

    def __init__(self, class_labels, predicted_feature_name='classLabel', predicted_probabilities_output=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Configuration for classifier models.\n\n        Attributes:\n\n        class_labels: str / list of int / list of str\n            If a list if given, the list maps the index of the output of a\n            neural network to labels in a classifier.\n            If a str is given, the str points to a file which maps the index\n            to labels in a classifier.\n\n        predicted_feature_name: str\n            Name of the output feature for the class labels exposed in the\n            Core ML neural network classifier, defaults: 'classLabel'.\n\n        predicted_probabilities_output: str\n            If provided, then this is the name of the neural network blob which\n            generates the probabilities for each class label (typically the output\n            of a softmax layer). If not provided, then the last output layer is\n            assumed.\n        "
        self.class_labels = class_labels
        self.predicted_feature_name = predicted_feature_name
        self.predicted_probabilities_output = predicted_probabilities_output

class InputType(object):

    def __init__(self, name=None, shape=None, dtype=types.fp32):
        if False:
            return 10
        '\n        The Input Type for inputs fed into the model.\n\n        Attributes:\n\n        name: (str)\n            The name of the input.\n        shape: list, tuple, Shape object, EnumeratedShapes object or None\n            The shape(s) that are valid for this input.\n            If set to None, the shape will be infered from the model itself.\n        '
        self.name = name
        if shape is not None:
            self.shape = _get_shaping_class(shape)
        else:
            self.shape = None
        self.dtype = dtype

class ImageType(InputType):

    def __init__(self, name=None, shape=None, scale=1.0, bias=None, color_layout='RGB', channel_first=None):
        if False:
            i = 10
            return i + 15
        "\n        Configuration class used for image inputs in CoreML.\n\n        Attributes:\n\n        scale: (float)\n            The scaling factor for all values in the image channels.\n        bias: float or list of float\n            If `color_layout` is 'G', bias would be a float\n            If `color_layout` is 'RGB' or 'BGR', bias would be a list of float\n        color_layout: string\n            Color layout of the image.\n            Valid values:\n                'G': Grayscale\n                'RGB': [Red, Green, Blue]\n                'BRG': [Blue, Red, Green]\n        channel_first: (bool) or None\n            Set to True if input format is channel first.\n            Default format is for TF is channel last. (channel_first=False)\n                              for PyTorch is channel first. (channel_first=True)\n        "
        super(ImageType, self).__init__(name, shape)
        self.scale = scale
        if color_layout not in ['G', 'RGB', 'BGR']:
            raise ValueError("color_layout should be one of ['G', 'RGB', 'BGR'], got '{}' instead".format(color_layout))
        self.color_layout = color_layout
        if bias is None:
            self.bias = 0.0 if color_layout == 'G' else [0.0, 0.0, 0.0]
        else:
            self.bias = bias
        self.channel_first = channel_first

class TensorType(InputType):

    def __init__(self, name=None, shape=None, dtype=None, is_optional=False, optional_value=None):
        if False:
            for i in range(10):
                print('nop')
        super(TensorType, self).__init__(name, shape)
        if dtype is None:
            self.dtype = types.fp32
        elif is_builtin(dtype):
            self.dtype = dtype
        else:
            try:
                self.dtype = numpy_type_to_builtin_type(dtype)
            except TypeError:
                raise TypeError('dtype={} is unsupported'.format(dtype))
        self.is_optional = is_optional
        self.optional_value = optional_value

class RangeDim(object):

    def __init__(self, lower_bound=1, upper_bound=-1, default=None):
        if False:
            print('Hello World!')
        "\n        A class that can be used to give a range of accepted shapes.\n\n        Attribute:\n\n        lower_bound: (int)\n            The minimum valid value for the shape.\n        upper_bound: (int)\n            The maximum valid value for the shape.\n            Set to -1 if there's no upper limit.\n        default: (int) or None\n            The default value that is used for initiating the model, and set in\n            the metadata of the model file.\n            If set to None, `lower_bound` would be used as default.\n        "
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if default is None:
            self.default = lower_bound
        else:
            if default < lower_bound:
                raise ValueError('Default value {} is less than minimum value ({}) for range'.format(default, lower_bound))
            if upper_bound > 0 and default > upper_bound:
                raise ValueError('Default value {} is greater than maximum value ({}) for range'.format(default, upper_bound))
            self.default = default

class Shape(object):

    def __init__(self, shape, default=None):
        if False:
            while True:
                i = 10
        '\n        The basic shape class to be set in InputType.\n\n        Attribute:\n\n        shape: list of (int), symbolic values, RangeDim object\n            The valid shape of the input\n        default: tuple of int or None\n            The default shape that is used for initiating the model, and set in\n            the metadata of the model file.\n            If None, then `shape` would be used.\n        '
        from coremltools.converters.mil.mil import get_new_symbol
        if not isinstance(shape, (list, tuple)):
            raise ValueError('Shape should be list or tuple, got type {} instead'.format(type(shape)))
        self.symbolic_shape = []
        shape = list(shape)
        for (idx, s) in enumerate(shape):
            if s is None or s == -1 or isinstance(s, RangeDim):
                sym = get_new_symbol()
                self.symbolic_shape.append(sym)
                if s is None or s == -1:
                    shape[idx] = sym
            elif isinstance(s, (np.generic, six.integer_types)) or is_symbolic(s):
                self.symbolic_shape.append(s)
            else:
                raise ValueError('Unknown type {} to build symbolic shape.'.format(type(s)))
        self.shape = tuple(shape)
        if default is not None:
            if not isinstance(default, (list, tuple)):
                raise ValueError('Default shape should be list or tuple, got type {} instead'.format(type(default)))
            for (idx, s) in enumerate(default):
                if not isinstance(s, (np.generic, six.integer_types)) and (not is_symbolic(s)):
                    raise ValueError('Default shape invalid, got error at index {} which is {}'.format(idx, s))
        else:
            default = []
            for (idx, s) in enumerate(self.shape):
                if isinstance(s, RangeDim):
                    default.append(s.default)
                elif s is None or s == -1:
                    default.append(self.symbolic_shape[idx])
                else:
                    default.append(s)
        self.default = tuple(default)

class EnumeratedShapes(object):

    def __init__(self, shapes, default=None):
        if False:
            return 10
        '\n        A shape class that is used for setting multiple valid shape in InputType.\n\n        shapes: list of Shape objects, or Shape-compatible lists.\n            The valid shapes of the inputs.\n            If input provided is not Shape object, but can be converted to Shape,\n            the Shape object would be stored in `shapes` instead.\n        default: tuple of int or None\n            The default shape that is used for initiating the model, and set in\n            the metadata of the model file.\n            If None, then the first element in `shapes` would be used.\n        '
        from coremltools.converters.mil.mil import get_new_symbol
        if not isinstance(shapes, (list, tuple)):
            raise ValueError('EnumeratedShapes should be list or tuple of shape, got type {} instead'.format(type(shapes)))
        if len(shapes) < 2:
            raise ValueError('EnumeratedShapes should be take a list or tuple with len >= 2, got {} instead'.format(len(shapes)))
        self.shapes = []
        for (idx, s) in enumerate(shapes):
            if isinstance(s, Shape):
                self.shapes.append(s)
            else:
                self.shapes.append(Shape(s))
        self.symbolic_shape = self.shapes[0].symbolic_shape
        for shape in self.shapes:
            for (idx, s) in enumerate(shape.symbolic_shape):
                if is_symbolic(self.symbolic_shape[idx]):
                    continue
                elif is_symbolic(s):
                    self.symbolic_shape[idx] = s
                elif s != self.symbolic_shape[idx]:
                    self.symbolic_shape[idx] = get_new_symbol()
        if default is not None:
            if not isinstance(default, (list, tuple)):
                raise ValueError('Default shape should be list or tuple, got type {} instead'.format(type(default)))
            for (idx, s) in enumerate(default):
                if not isinstance(s, (np.generic, six.integer_types)) and (not is_symbolic(s)):
                    raise ValueError('Default shape invalid, got error at index {} which is {}'.format(idx, s))
        else:
            default = self.shapes[0].default
        self.default = default

def _get_shaping_class(shape):
    if False:
        return 10
    '\n        Returns a Shape class or EnumeratedShapes class for `shape`\n        where `shape` could be lists/tuple/Shape/EnumeratedShapes/etc.\n    '
    if isinstance(shape, (Shape, EnumeratedShapes)):
        return shape
    try:
        enum_shape = EnumeratedShapes(shape)
        return enum_shape
    except ValueError:
        pass
    try:
        shape = Shape(shape)
        return shape
    except ValueError:
        pass
    raise ValueError("Can't convert to CoreML shaping class from {}.".format(shape))