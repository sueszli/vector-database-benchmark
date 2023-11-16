"""A Python interface for creating TensorFlow servers."""
from tensorflow.core.framework import device_attributes_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import _pywrap_device_lib

def list_local_devices(session_config=None):
    if False:
        print('Hello World!')
    'List the available devices available in the local process.\n\n  Args:\n    session_config: a session config proto or None to use the default config.\n\n  Returns:\n    A list of `DeviceAttribute` protocol buffers.\n  '

    def _convert(pb_str):
        if False:
            print('Hello World!')
        m = device_attributes_pb2.DeviceAttributes()
        m.ParseFromString(pb_str)
        return m
    serialized_config = None
    if session_config is not None:
        serialized_config = session_config.SerializeToString()
    return [_convert(s) for s in _pywrap_device_lib.list_devices(serialized_config)]