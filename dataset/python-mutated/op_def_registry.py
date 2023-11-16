"""Global registry for OpDefs."""
import threading
from tensorflow.core.framework import op_def_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import _op_def_registry
_cache = {}
_cache_lock = threading.Lock()

def get(name):
    if False:
        i = 10
        return i + 15
    'Returns an OpDef for a given `name` or None if the lookup fails.'
    try:
        return _cache[name]
    except KeyError:
        pass
    with _cache_lock:
        try:
            return _cache[name]
        except KeyError:
            pass
        serialized_op_def = _op_def_registry.get(name)
        if serialized_op_def is None:
            return None
        op_def = op_def_pb2.OpDef()
        op_def.ParseFromString(serialized_op_def)
        _cache[name] = op_def
        return op_def

def sync():
    if False:
        for i in range(10):
            print('nop')
    'No-op. Used to synchronize the contents of the Python registry with C++.'