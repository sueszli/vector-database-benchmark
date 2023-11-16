from logging import critical
import os

def is_libpsp():
    if False:
        for i in range(10):
            print('nop')
    'Was libpsppy successfully loaded in this module?'
    return __is_libpsp__
__is_libpsp__ = True
try:
    from .table import *
    from .manager import *
    from .viewer import *
    from .table.libpsppy import init_expression_parser

    def set_threadpool_size(nthreads):
        if False:
            print('Hello World!')
        'Sets the size of the global Perspective thread pool, up to the\n        total number of available cores.  Passing an explicit\n        `None` sets this limit to the detected hardware concurrency from the\n        environment, which is also the default if this method is never called.\n        `set_threadpool_size()` must be called before any other\n        `perspective-python` API calls, and cannot be changed after such a call.\n        '
        os.environ['OMP_THREAD_LIMIT'] = '0' if nthreads is None else str(nthreads)
    init_expression_parser()
except ImportError:
    __is_libpsp__ = False
    critical('Failed to import C++ bindings for Perspective probably as it could not be built for your architecture (check install logs for more details).\n', exc_info=True)
    critical('You can still use `PerspectiveWidget` in client mode using JupyterLab.')