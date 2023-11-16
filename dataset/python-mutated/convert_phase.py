"""Utilities for collecting TFLite metrics."""
import collections
import enum
import functools
from typing import Text
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.lite.python.metrics import metrics

class Component(enum.Enum):
    """Enum class defining name of the converter components."""
    PREPARE_TF_MODEL = 'PREPARE_TF_MODEL'
    CONVERT_TF_TO_TFLITE_MODEL = 'CONVERT_TF_TO_TFLITE_MODEL'
    OPTIMIZE_TFLITE_MODEL = 'OPTIMIZE_TFLITE_MODEL'
SubComponentItem = collections.namedtuple('SubComponentItem', ['name', 'component'])

class SubComponent(SubComponentItem, enum.Enum):
    """Enum class defining name of the converter subcomponents.

  This enum only defines the subcomponents in Python, there might be more
  subcomponents defined in C++.
  """

    def __str__(self):
        if False:
            return 10
        return self.value.name

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
        return self.value.name

    @property
    def component(self):
        if False:
            while True:
                i = 10
        return self.value.component
    UNSPECIFIED = SubComponentItem('UNSPECIFIED', None)
    VALIDATE_INPUTS = SubComponentItem('VALIDATE_INPUTS', Component.PREPARE_TF_MODEL)
    LOAD_SAVED_MODEL = SubComponentItem('LOAD_SAVED_MODEL', Component.PREPARE_TF_MODEL)
    FREEZE_SAVED_MODEL = SubComponentItem('FREEZE_SAVED_MODEL', Component.PREPARE_TF_MODEL)
    CONVERT_KERAS_TO_SAVED_MODEL = SubComponentItem('CONVERT_KERAS_TO_SAVED_MODEL', Component.PREPARE_TF_MODEL)
    CONVERT_CONCRETE_FUNCTIONS_TO_SAVED_MODEL = SubComponentItem('CONVERT_CONCRETE_FUNCTIONS_TO_SAVED_MODEL', Component.PREPARE_TF_MODEL)
    FREEZE_KERAS_MODEL = SubComponentItem('FREEZE_KERAS_MODEL', Component.PREPARE_TF_MODEL)
    FREEZE_CONCRETE_FUNCTION = SubComponentItem('FREEZE_CONCRETE_FUNCTION', Component.PREPARE_TF_MODEL)
    OPTIMIZE_TF_MODEL = SubComponentItem('OPTIMIZE_TF_MODEL', Component.PREPARE_TF_MODEL)
    CONVERT_GRAPHDEF_USING_DEPRECATED_CONVERTER = SubComponentItem('CONVERT_GRAPHDEF_USING_DEPRECATED_CONVERTER', Component.CONVERT_TF_TO_TFLITE_MODEL)
    CONVERT_GRAPHDEF = SubComponentItem('CONVERT_GRAPHDEF', Component.CONVERT_TF_TO_TFLITE_MODEL)
    CONVERT_SAVED_MODEL = SubComponentItem('CONVERT_SAVED_MODEL', Component.CONVERT_TF_TO_TFLITE_MODEL)
    CONVERT_JAX_HLO = SubComponentItem('CONVERT_JAX_HLO', Component.CONVERT_TF_TO_TFLITE_MODEL)
    QUANTIZE_USING_DEPRECATED_QUANTIZER = SubComponentItem('QUANTIZE_USING_DEPRECATED_QUANTIZER', Component.OPTIMIZE_TFLITE_MODEL)
    CALIBRATE = SubComponentItem('CALIBRATE', Component.OPTIMIZE_TFLITE_MODEL)
    QUANTIZE = SubComponentItem('QUANTIZE', Component.OPTIMIZE_TFLITE_MODEL)
    SPARSIFY = SubComponentItem('SPARSIFY', Component.OPTIMIZE_TFLITE_MODEL)

class ConverterError(Exception):
    """Raised when an error occurs during model conversion."""

    def __init__(self, message):
        if False:
            print('Hello World!')
        super(ConverterError, self).__init__(message)
        self.errors = []
        self._parse_error_message(message)

    def append_error(self, error_data: converter_error_data_pb2.ConverterErrorData):
        if False:
            i = 10
            return i + 15
        self.errors.append(error_data)

    def _parse_error_message(self, message):
        if False:
            return 10
        'If the message matches a pattern, assigns the associated error code.\n\n    It is difficult to assign an error code to some errrors in MLIR side, Ex:\n    errors thrown by other components than TFLite or not using mlir::emitError.\n    This function try to detect them by the error message and assign the\n    corresponding error code.\n\n    Args:\n      message: The error message of this exception.\n    '
        error_code_mapping = {'Failed to functionalize Control Flow V1 ops. Consider using Control Flow V2 ops instead. See https://www.tensorflow.org/api_docs/python/tf/compat/v1/enable_control_flow_v2.': converter_error_data_pb2.ConverterErrorData.ERROR_UNSUPPORTED_CONTROL_FLOW_V1}
        for (pattern, error_code) in error_code_mapping.items():
            if pattern in message:
                error_data = converter_error_data_pb2.ConverterErrorData()
                error_data.error_message = message
                error_data.error_code = error_code
                self.append_error(error_data)
                return

def convert_phase(component, subcomponent=SubComponent.UNSPECIFIED):
    if False:
        for i in range(10):
            print('nop')
    'The decorator to identify converter component and subcomponent.\n\n  Args:\n    component: Converter component name.\n    subcomponent: Converter subcomponent name.\n\n  Returns:\n    Forward the result from the wrapped function.\n\n  Raises:\n    ValueError: if component and subcomponent name is not valid.\n  '
    if component not in Component:
        raise ValueError('Given component name not found')
    if subcomponent not in SubComponent:
        raise ValueError('Given subcomponent name not found')
    if subcomponent != SubComponent.UNSPECIFIED and subcomponent.component != component:
        raise ValueError("component and subcomponent name don't match")

    def report_error(error_data: converter_error_data_pb2.ConverterErrorData):
        if False:
            i = 10
            return i + 15
        error_data.component = component.value
        if not error_data.subcomponent:
            error_data.subcomponent = subcomponent.name
        tflite_metrics = metrics.TFLiteConverterMetrics()
        tflite_metrics.set_converter_error(error_data)

    def report_error_message(error_message: Text):
        if False:
            i = 10
            return i + 15
        error_data = converter_error_data_pb2.ConverterErrorData()
        error_data.error_message = error_message
        report_error(error_data)

    def actual_decorator(func):
        if False:
            for i in range(10):
                print('nop')

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                return 10
            try:
                return func(*args, **kwargs)
            except ConverterError as converter_error:
                if converter_error.errors:
                    for error_data in converter_error.errors:
                        report_error(error_data)
                else:
                    report_error_message(str(converter_error))
                raise converter_error from None
            except Exception as error:
                report_error_message(str(error))
                raise error from None
        return wrapper
    return actual_decorator