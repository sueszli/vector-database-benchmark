"""Utilities for using the TensorFlow C API."""
import contextlib
from tensorflow.core.framework import api_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib

class AlreadyGarbageCollectedError(Exception):

    def __init__(self, name, obj_type):
        if False:
            print('Hello World!')
        super(AlreadyGarbageCollectedError, self).__init__(f'{name} of type {obj_type} has already been garbage collected and cannot be called.')

class UniquePtr(object):
    """Wrapper around single-ownership C-API objects that handles deletion."""
    __slots__ = ['_obj', 'deleter', 'name', 'type_name']

    def __init__(self, name, obj, deleter):
        if False:
            return 10
        self._obj = obj
        self.name = name
        self.deleter = deleter
        self.type_name = str(type(obj))

    @contextlib.contextmanager
    def get(self):
        if False:
            return 10
        'Yields the managed C-API Object, guaranteeing aliveness.\n\n    This is a context manager. Inside the context the C-API object is\n    guaranteed to be alive.\n\n    Raises:\n      AlreadyGarbageCollectedError: if the object is already deleted.\n    '
        if self._obj is None:
            raise AlreadyGarbageCollectedError(self.name, self.type_name)
        yield self._obj

    def __del__(self):
        if False:
            return 10
        obj = self._obj
        if obj is not None:
            self._obj = None
            self.deleter(obj)

class ScopedTFStatus(object):
    """Wrapper around TF_Status that handles deletion."""
    __slots__ = ['status']

    def __init__(self):
        if False:
            return 10
        self.status = c_api.TF_NewStatus()

    def __del__(self):
        if False:
            print('Hello World!')
        if c_api is not None and c_api.TF_DeleteStatus is not None:
            c_api.TF_DeleteStatus(self.status)

class ScopedTFImportGraphDefOptions(object):
    """Wrapper around TF_ImportGraphDefOptions that handles deletion."""
    __slots__ = ['options']

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.options = c_api.TF_NewImportGraphDefOptions()

    def __del__(self):
        if False:
            print('Hello World!')
        if c_api is not None and c_api.TF_DeleteImportGraphDefOptions is not None:
            c_api.TF_DeleteImportGraphDefOptions(self.options)

class ScopedTFImportGraphDefResults(object):
    """Wrapper around TF_ImportGraphDefOptions that handles deletion."""
    __slots__ = ['results']

    def __init__(self, results):
        if False:
            while True:
                i = 10
        self.results = results

    def __del__(self):
        if False:
            i = 10
            return i + 15
        if c_api is not None and c_api.TF_DeleteImportGraphDefResults is not None:
            c_api.TF_DeleteImportGraphDefResults(self.results)

class ScopedTFFunction(UniquePtr):
    """Wrapper around TF_Function that handles deletion."""

    def __init__(self, func, name):
        if False:
            return 10
        super(ScopedTFFunction, self).__init__(name=name, obj=func, deleter=c_api.TF_DeleteFunction)

class ScopedTFBuffer(object):
    """An internal class to help manage the TF_Buffer lifetime."""
    __slots__ = ['buffer']

    def __init__(self, buf_string):
        if False:
            for i in range(10):
                print('nop')
        self.buffer = c_api.TF_NewBufferFromString(compat.as_bytes(buf_string))

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        c_api.TF_DeleteBuffer(self.buffer)

class ApiDefMap(object):
    """Wrapper around Tf_ApiDefMap that handles querying and deletion.

  The OpDef protos are also stored in this class so that they could
  be queried by op name.
  """
    __slots__ = ['_api_def_map', '_op_per_name']

    def __init__(self):
        if False:
            print('Hello World!')
        op_def_proto = op_def_pb2.OpList()
        buf = c_api.TF_GetAllOpList()
        try:
            op_def_proto.ParseFromString(c_api.TF_GetBuffer(buf))
            self._api_def_map = c_api.TF_NewApiDefMap(buf)
        finally:
            c_api.TF_DeleteBuffer(buf)
        self._op_per_name = {}
        for op in op_def_proto.op:
            self._op_per_name[op.name] = op

    def __del__(self):
        if False:
            while True:
                i = 10
        if c_api is not None and c_api.TF_DeleteApiDefMap is not None:
            c_api.TF_DeleteApiDefMap(self._api_def_map)

    def put_api_def(self, text):
        if False:
            for i in range(10):
                print('nop')
        c_api.TF_ApiDefMapPut(self._api_def_map, text, len(text))

    def get_api_def(self, op_name):
        if False:
            return 10
        api_def_proto = api_def_pb2.ApiDef()
        buf = c_api.TF_ApiDefMapGet(self._api_def_map, op_name, len(op_name))
        try:
            api_def_proto.ParseFromString(c_api.TF_GetBuffer(buf))
        finally:
            c_api.TF_DeleteBuffer(buf)
        return api_def_proto

    def get_op_def(self, op_name):
        if False:
            print('Hello World!')
        if op_name in self._op_per_name:
            return self._op_per_name[op_name]
        raise ValueError(f'No op_def found for op name {op_name}.')

    def op_names(self):
        if False:
            while True:
                i = 10
        return self._op_per_name.keys()

@tf_contextlib.contextmanager
def tf_buffer(data=None):
    if False:
        print('Hello World!')
    'Context manager that creates and deletes TF_Buffer.\n\n  Example usage:\n    with tf_buffer() as buf:\n      # get serialized graph def into buf\n      ...\n      proto_data = c_api.TF_GetBuffer(buf)\n      graph_def.ParseFromString(compat.as_bytes(proto_data))\n    # buf has been deleted\n\n    with tf_buffer(some_string) as buf:\n      c_api.TF_SomeFunction(buf)\n    # buf has been deleted\n\n  Args:\n    data: An optional `bytes`, `str`, or `unicode` object. If not None, the\n      yielded buffer will contain this data.\n\n  Yields:\n    Created TF_Buffer\n  '
    if data:
        buf = c_api.TF_NewBufferFromString(compat.as_bytes(data))
    else:
        buf = c_api.TF_NewBuffer()
    try:
        yield buf
    finally:
        c_api.TF_DeleteBuffer(buf)

def tf_output(c_op, index):
    if False:
        return 10
    'Returns a wrapped TF_Output with specified operation and index.\n\n  Args:\n    c_op: wrapped TF_Operation\n    index: integer\n\n  Returns:\n    Wrapped TF_Output\n  '
    ret = c_api.TF_Output()
    ret.oper = c_op
    ret.index = index
    return ret