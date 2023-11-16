"""Python module for Session ops, vars, and functions exported by pybind11."""
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client._pywrap_tf_session import *
from tensorflow.python.client._pywrap_tf_session import _TF_SetTarget
from tensorflow.python.client._pywrap_tf_session import _TF_SetConfig
from tensorflow.python.client._pywrap_tf_session import _TF_NewSessionOptions
from tensorflow.python.util import tf_stack
__version__ = str(get_version())
__git_version__ = str(get_git_version())
__compiler_version__ = str(get_compiler_version())
__cxx11_abi_flag__ = get_cxx11_abi_flag()
__cxx_version__ = get_cxx_version()
__monolithic_build__ = get_monolithic_build()
GRAPH_DEF_VERSION = get_graph_def_version()
GRAPH_DEF_VERSION_MIN_CONSUMER = get_graph_def_version_min_consumer()
GRAPH_DEF_VERSION_MIN_PRODUCER = get_graph_def_version_min_producer()
TENSOR_HANDLE_KEY = get_tensor_handle_key()

def TF_NewSessionOptions(target=None, config=None):
    if False:
        i = 10
        return i + 15
    opts = _TF_NewSessionOptions()
    if target is not None:
        _TF_SetTarget(opts, target)
    if config is not None:
        config_str = config.SerializeToString()
        _TF_SetConfig(opts, config_str)
    return opts

def TF_Reset(target, containers=None, config=None):
    if False:
        i = 10
        return i + 15
    opts = TF_NewSessionOptions(target=target, config=config)
    try:
        TF_Reset_wrapper(opts, containers)
    finally:
        TF_DeleteSessionOptions(opts)