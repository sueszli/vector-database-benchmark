"""
Reliable Allreduce and Broadcast Library.

Author: Tianqi Chen
"""
import cPickle as pickle
import ctypes
import os
import sys
import warnings
import numpy as np
__version__ = '1.0'
if os.name == 'nt':
    WRAPPER_PATH = os.path.dirname(__file__) + '\\..\\windows\\x64\\Release\\rabit_wrapper%s.dll'
else:
    WRAPPER_PATH = os.path.dirname(__file__) + '/librabit_wrapper%s.so'
_LIB = None

def _loadlib(lib='standard'):
    if False:
        return 10
    'Load rabit library.'
    global _LIB
    if _LIB is not None:
        warnings.warn('rabit.int call was ignored because it has already been initialized', level=2)
        return
    if lib == 'standard':
        _LIB = ctypes.cdll.LoadLibrary(WRAPPER_PATH % '')
    elif lib == 'mock':
        _LIB = ctypes.cdll.LoadLibrary(WRAPPER_PATH % '_mock')
    elif lib == 'mpi':
        _LIB = ctypes.cdll.LoadLibrary(WRAPPER_PATH % '_mpi')
    else:
        raise Exception('unknown rabit lib %s, can be standard, mock, mpi' % lib)
    _LIB.RabitGetRank.restype = ctypes.c_int
    _LIB.RabitGetWorldSize.restype = ctypes.c_int
    _LIB.RabitVersionNumber.restype = ctypes.c_int

def _unloadlib():
    if False:
        return 10
    'Unload rabit library.'
    global _LIB
    del _LIB
    _LIB = None
MAX = 0
MIN = 1
SUM = 2
BITOR = 3

def init(args=None, lib='standard'):
    if False:
        return 10
    "Intialize the rabit module, call this once before using anything.\n\n    Parameters\n    ----------\n    args: list of str, optional\n        The list of arguments used to initialized the rabit\n        usually you need to pass in sys.argv.\n        Defaults to sys.argv when it is None.\n    lib: {'standard', 'mock', 'mpi'}\n        Type of library we want to load\n    "
    if args is None:
        args = sys.argv
    _loadlib(lib)
    arr = (ctypes.c_char_p * len(args))()
    arr[:] = args
    _LIB.RabitInit(len(args), arr)

def finalize():
    if False:
        while True:
            i = 10
    'Finalize the rabit engine.\n\n    Call this function after you finished all jobs.\n    '
    _LIB.RabitFinalize()
    _unloadlib()

def get_rank():
    if False:
        print('Hello World!')
    'Get rank of current process.\n\n    Returns\n    -------\n    rank : int\n        Rank of current process.\n    '
    ret = _LIB.RabitGetRank()
    return ret

def get_world_size():
    if False:
        print('Hello World!')
    'Get total number workers.\n\n    Returns\n    -------\n    n : int\n        Total number of process.\n    '
    ret = _LIB.RabitGetWorldSize()
    return ret

def tracker_print(msg):
    if False:
        while True:
            i = 10
    'Print message to the tracker.\n\n    This function can be used to communicate the information of\n    the progress to the tracker\n\n    Parameters\n    ----------\n    msg : str\n        The message to be printed to tracker.\n    '
    if not isinstance(msg, str):
        msg = str(msg)
    _LIB.RabitTrackerPrint(ctypes.c_char_p(msg).encode('utf-8'))

def get_processor_name():
    if False:
        return 10
    'Get the processor name.\n\n    Returns\n    -------\n    name : str\n        the name of processor(host)\n    '
    mxlen = 256
    length = ctypes.c_ulong()
    buf = ctypes.create_string_buffer(mxlen)
    _LIB.RabitGetProcessorName(buf, ctypes.byref(length), mxlen)
    return buf.value

def broadcast(data, root):
    if False:
        return 10
    'Broadcast object from one node to all other nodes.\n\n    Parameters\n    ----------\n    data : any type that can be pickled\n        Input data, if current rank does not equal root, this can be None\n    root : int\n        Rank of the node to broadcast data from.\n\n    Returns\n    -------\n    object : int\n        the result of broadcast.\n    '
    rank = get_rank()
    length = ctypes.c_ulong()
    if root == rank:
        assert data is not None, 'need to pass in data when broadcasting'
        s = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        length.value = len(s)
    _LIB.RabitBroadcast(ctypes.byref(length), ctypes.sizeof(ctypes.c_ulong), root)
    if root != rank:
        dptr = (ctypes.c_char * length.value)()
        _LIB.RabitBroadcast(ctypes.cast(dptr, ctypes.c_void_p), length.value, root)
        data = pickle.loads(dptr.raw)
        del dptr
    else:
        _LIB.RabitBroadcast(ctypes.cast(ctypes.c_char_p(s), ctypes.c_void_p), length.value, root)
        del s
    return data
DTYPE_ENUM__ = {np.dtype('int8'): 0, np.dtype('uint8'): 1, np.dtype('int32'): 2, np.dtype('uint32'): 3, np.dtype('int64'): 4, np.dtype('uint64'): 5, np.dtype('float32'): 6, np.dtype('float64'): 7}

def allreduce(data, op, prepare_fun=None):
    if False:
        return 10
    'Perform allreduce, return the result.\n\n    Parameters\n    ----------\n    data: numpy array\n        Input data.\n    op: int\n        Reduction operators, can be MIN, MAX, SUM, BITOR\n    prepare_fun: function\n        Lazy preprocessing function, if it is not None, prepare_fun(data)\n        will be called by the function before performing allreduce, to intialize the data\n        If the result of Allreduce can be recovered directly,\n        then prepare_fun will NOT be called\n\n    Returns\n    -------\n    result : array_like\n        The result of allreduce, have same shape as data\n\n    Notes\n    -----\n    This function is not thread-safe.\n    '
    if not isinstance(data, np.ndarray):
        raise Exception('allreduce only takes in numpy.ndarray')
    buf = data.ravel()
    if buf.base is data.base:
        buf = buf.copy()
    if buf.dtype not in DTYPE_ENUM__:
        raise Exception('data type %s not supported' % str(buf.dtype))
    if prepare_fun is None:
        _LIB.RabitAllreduce(buf.ctypes.data_as(ctypes.c_void_p), buf.size, DTYPE_ENUM__[buf.dtype], op, None, None)
    else:
        func_ptr = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

        def pfunc(args):
            if False:
                for i in range(10):
                    print('nop')
            'prepare function.'
            prepare_fun(data)
        _LIB.RabitAllreduce(buf.ctypes.data_as(ctypes.c_void_p), buf.size, DTYPE_ENUM__[buf.dtype], op, func_ptr(pfunc), None)
    return buf

def _load_model(ptr, length):
    if False:
        print('Hello World!')
    '\n    Internal function used by the module,\n    unpickle a model from a buffer specified by ptr, length\n    Arguments:\n        ptr: ctypes.POINTER(ctypes._char)\n            pointer to the memory region of buffer\n        length: int\n            the length of buffer\n    '
    data = (ctypes.c_char * length).from_address(ctypes.addressof(ptr.contents))
    return pickle.loads(data.raw)

def load_checkpoint(with_local=False):
    if False:
        while True:
            i = 10
    'Load latest check point.\n\n    Parameters\n    ----------\n    with_local: bool, optional\n        whether the checkpoint contains local model\n\n    Returns\n    -------\n    tuple : tuple\n        if with_local: return (version, gobal_model, local_model)\n        else return (version, gobal_model)\n        if returned version == 0, this means no model has been CheckPointed\n        and global_model, local_model returned will be None\n    '
    gptr = ctypes.POINTER(ctypes.c_char)()
    global_len = ctypes.c_ulong()
    if with_local:
        lptr = ctypes.POINTER(ctypes.c_char)()
        local_len = ctypes.c_ulong()
        version = _LIB.RabitLoadCheckPoint(ctypes.byref(gptr), ctypes.byref(global_len), ctypes.byref(lptr), ctypes.byref(local_len))
        if version == 0:
            return (version, None, None)
        return (version, _load_model(gptr, global_len.value), _load_model(lptr, local_len.value))
    else:
        version = _LIB.RabitLoadCheckPoint(ctypes.byref(gptr), ctypes.byref(global_len), None, None)
        if version == 0:
            return (version, None)
        return (version, _load_model(gptr, global_len.value))

def checkpoint(global_model, local_model=None):
    if False:
        i = 10
        return i + 15
    'Checkpoint the model.\n\n    This means we finished a stage of execution.\n    Every time we call check point, there is a version number which will increase by one.\n\n    Parameters\n    ----------\n    global_model: anytype that can be pickled\n        globally shared model/state when calling this function,\n        the caller need to gauranttees that global_model is the same in all nodes\n\n    local_model: anytype that can be pickled\n       Local model, that is specific to current node/rank.\n       This can be None when no local state is needed.\n\n    Notes\n    -----\n    local_model requires explicit replication of the model for fault-tolerance.\n    This will bring replication cost in checkpoint function.\n    while global_model do not need explicit replication.\n    It is recommended to use global_model if possible.\n    '
    sglobal = pickle.dumps(global_model)
    if local_model is None:
        _LIB.RabitCheckPoint(sglobal, len(sglobal), None, 0)
        del sglobal
    else:
        slocal = pickle.dumps(local_model)
        _LIB.RabitCheckPoint(sglobal, len(sglobal), slocal, len(slocal))
        del slocal
        del sglobal

def version_number():
    if False:
        return 10
    'Returns version number of current stored model.\n\n    This means how many calls to CheckPoint we made so far.\n\n    Returns\n    -------\n    version : int\n        Version number of currently stored model\n    '
    ret = _LIB.RabitVersionNumber()
    return ret