"""Determine which implementation of the protobuf API is used in this process.
"""
import os
import warnings
import sys
try:
    from google.protobuf.internal import _api_implementation
    _api_version = _api_implementation.api_version
    _proto_extension_modules_exist_in_build = True
except ImportError:
    _api_version = -1
    _proto_extension_modules_exist_in_build = False
if _api_version == 1:
    raise ValueError('api_version=1 is no longer supported.')
if _api_version < 0:
    try:
        from google.protobuf import _use_fast_cpp_protos
        if not _use_fast_cpp_protos:
            raise ImportError('_use_fast_cpp_protos import succeeded but was None')
        del _use_fast_cpp_protos
        _api_version = 2
    except ImportError:
        if _proto_extension_modules_exist_in_build:
            if sys.version_info[0] >= 3:
                _api_version = 2
_default_implementation_type = 'python' if _api_version <= 0 else 'cpp'
_implementation_type = os.getenv('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', _default_implementation_type)
if _implementation_type != 'python':
    _implementation_type = 'cpp'
if 'PyPy' in sys.version and _implementation_type == 'cpp':
    warnings.warn('PyPy does not work yet with cpp protocol buffers. Falling back to the python implementation.')
    _implementation_type = 'python'
_implementation_version_str = os.getenv('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION', '2')
if _implementation_version_str != '2':
    raise ValueError('unsupported PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION: "' + _implementation_version_str + '" (supported versions: 2)')
_implementation_version = int(_implementation_version_str)

def Type():
    if False:
        i = 10
        return i + 15
    return _implementation_type

def Version():
    if False:
        print('Hello World!')
    return _implementation_version