"""Python TFLite metrics helper."""
import os
from typing import Optional, Text
if not os.path.splitext(__file__)[0].endswith(os.path.join('tflite_runtime', 'metrics_portable')):
    from tensorflow.lite.python.metrics import metrics_interface
else:
    from tflite_runtime import metrics_interface

class TFLiteMetrics(metrics_interface.TFLiteMetricsInterface):
    """TFLite metrics helper."""

    def __init__(self, model_hash: Optional[Text]=None, model_path: Optional[Text]=None) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def increase_counter_debugger_creation(self):
        if False:
            print('Hello World!')
        pass

    def increase_counter_interpreter_creation(self):
        if False:
            print('Hello World!')
        pass

    def increase_counter_converter_attempt(self):
        if False:
            return 10
        pass

    def increase_counter_converter_success(self):
        if False:
            i = 10
            return i + 15
        pass

    def set_converter_param(self, name, value):
        if False:
            while True:
                i = 10
        pass

    def set_converter_error(self, error_data):
        if False:
            i = 10
            return i + 15
        pass

    def set_converter_latency(self, value):
        if False:
            for i in range(10):
                print('nop')
        pass

class TFLiteConverterMetrics(TFLiteMetrics):
    """Similar to TFLiteMetrics but specialized for converter."""

    def __del__(self):
        if False:
            return 10
        pass

    def set_export_required(self):
        if False:
            while True:
                i = 10
        pass

    def export_metrics(self):
        if False:
            for i in range(10):
                print('nop')
        pass