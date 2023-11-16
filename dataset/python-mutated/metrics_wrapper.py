"""Stub to make pywrap metrics wrapper accessible."""
from tensorflow.lite.python import wrap_toco
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.lite.python.metrics._pywrap_tensorflow_lite_metrics_wrapper import MetricsWrapper

def retrieve_collected_errors():
    if False:
        for i in range(10):
            print('nop')
    'Returns and clears the list of collected errors in ErrorCollector.\n\n  The RetrieveCollectedErrors function in C++ returns a list of serialized proto\n  messages. This function will convert them to ConverterErrorData instances.\n\n  Returns:\n    A list of ConverterErrorData.\n  '
    serialized_message_list = wrap_toco.wrapped_retrieve_collected_errors()
    return list(map(converter_error_data_pb2.ConverterErrorData.FromString, serialized_message_list))