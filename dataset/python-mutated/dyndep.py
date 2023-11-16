import ctypes
import os
from threading import Lock
from caffe2.python import core, extension_loader

def InitOpsLibrary(name, trigger_lazy=True):
    if False:
        i = 10
        return i + 15
    'Loads a dynamic library that contains custom operators into Caffe2.\n\n    Since Caffe2 uses static variable registration, you can optionally load a\n    separate .so file that contains custom operators and registers that into\n    the caffe2 core binary. In C++, this is usually done by either declaring\n    dependency during compilation time, or via dynload. This allows us to do\n    registration similarly on the Python side.\n\n    Args:\n        name: a name that ends in .so, such as "my_custom_op.so". Otherwise,\n            the command will simply be ignored.\n    Returns:\n        None\n    '
    if not os.path.exists(name):
        print('Ignoring {} as it is not a valid file.'.format(name))
        return
    _init_impl(name, trigger_lazy=trigger_lazy)
_IMPORTED_DYNDEPS = set()
dll_lock = Lock()

def GetImportedOpsLibraries():
    if False:
        return 10
    return _IMPORTED_DYNDEPS

def _init_impl(path, trigger_lazy=True):
    if False:
        print('Hello World!')
    with dll_lock:
        _IMPORTED_DYNDEPS.add(path)
        with extension_loader.DlopenGuard():
            ctypes.CDLL(path)
        core.RefreshRegisteredOperators(trigger_lazy)