import threading
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.libgeos import CONTEXT_PTR, error_h, lgeos, notice_h

class GEOSContextHandle(GEOSBase):
    """Represent a GEOS context handle."""
    ptr_type = CONTEXT_PTR
    destructor = lgeos.finishGEOS_r

    def __init__(self):
        if False:
            print('Hello World!')
        self.ptr = lgeos.initGEOS_r(notice_h, error_h)

class GEOSContext(threading.local):
    handle = None
thread_context = GEOSContext()

class GEOSFunc:
    """
    Serve as a wrapper for GEOS C Functions. Use thread-safe function
    variants when available.
    """

    def __init__(self, func_name):
        if False:
            while True:
                i = 10
        self.cfunc = getattr(lgeos, func_name + '_r')
        self.thread_context = thread_context

    def __call__(self, *args):
        if False:
            while True:
                i = 10
        self.thread_context.handle = self.thread_context.handle or GEOSContextHandle()
        return self.cfunc(self.thread_context.handle.ptr, *args)

    def __str__(self):
        if False:
            print('Hello World!')
        return self.cfunc.__name__

    def _get_argtypes(self):
        if False:
            for i in range(10):
                print('nop')
        return self.cfunc.argtypes

    def _set_argtypes(self, argtypes):
        if False:
            i = 10
            return i + 15
        self.cfunc.argtypes = [CONTEXT_PTR, *argtypes]
    argtypes = property(_get_argtypes, _set_argtypes)

    def _get_restype(self):
        if False:
            print('Hello World!')
        return self.cfunc.restype

    def _set_restype(self, restype):
        if False:
            return 10
        self.cfunc.restype = restype
    restype = property(_get_restype, _set_restype)

    def _get_errcheck(self):
        if False:
            for i in range(10):
                print('nop')
        return self.cfunc.errcheck

    def _set_errcheck(self, errcheck):
        if False:
            print('Hello World!')
        self.cfunc.errcheck = errcheck
    errcheck = property(_get_errcheck, _set_errcheck)