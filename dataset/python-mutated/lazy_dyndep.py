import os
from caffe2.python import dyndep, lazy

def RegisterOpsLibrary(name):
    if False:
        i = 10
        return i + 15
    'Registers a dynamic library that contains custom operators into Caffe2.\n\n    Since Caffe2 uses static variable registration, you can optionally load a\n    separate .so file that contains custom operators and registers that into\n    the caffe2 core binary. In C++, this is usually done by either declaring\n    dependency during compilation time, or via dynload. This allows us to do\n    registration similarly on the Python side.\n\n    Unlike dyndep.InitOpsLibrary, this does not actually parse the c++ file\n    and refresh operators until caffe2 is called in a fashion which requires\n    operators. In some large codebases this saves a large amount of time\n    during import.\n\n    It is safe to use within a program that also uses dyndep.InitOpsLibrary\n\n    Args:\n        name: a name that ends in .so, such as "my_custom_op.so". Otherwise,\n            the command will simply be ignored.\n    Returns:\n        None\n    '
    if not os.path.exists(name):
        print('Ignoring {} as it is not a valid file.'.format(name))
        return
    global _LAZY_IMPORTED_DYNDEPS
    _LAZY_IMPORTED_DYNDEPS.add(name)
_LAZY_IMPORTED_DYNDEPS = set()
_error_handler = None

def SetErrorHandler(handler):
    if False:
        return 10
    'Registers an error handler for errors from registering operators\n\n    Since the lazy registration may happen at a much later time, having a dedicated\n    error handler allows for custom error handling logic. It is highly\n    recomended to set this to prevent errors from bubbling up in weird parts of the\n    code.\n\n    Args:\n        handler: a function that takes an exception as a single handler.\n    Returns:\n        None\n    '
    global _error_handler
    _error_handler = handler

def GetImportedOpsLibraries():
    if False:
        return 10
    _import_lazy()
    return dyndep.GetImportedOpsLibraries()

def _import_lazy():
    if False:
        for i in range(10):
            print('nop')
    global _LAZY_IMPORTED_DYNDEPS
    if not _LAZY_IMPORTED_DYNDEPS:
        return
    for name in list(_LAZY_IMPORTED_DYNDEPS):
        try:
            dyndep.InitOpLibrary(name, trigger_lazy=False)
        except BaseException as e:
            if _error_handler:
                _error_handler(e)
        finally:
            _LAZY_IMPORTED_DYNDEPS.remove(name)
lazy.RegisterLazyImport(_import_lazy)