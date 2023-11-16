"""
 This module houses the ctypes initialization procedures, as well
 as the notice and error handler function callbacks (get called
 when an error occurs in GEOS).

 This module also houses GEOS Pointer utilities, including
 get_pointer_arr(), and GEOM_PTR.
"""
import logging
import os
from ctypes import CDLL, CFUNCTYPE, POINTER, Structure, c_char_p
from ctypes.util import find_library
from django.core.exceptions import ImproperlyConfigured
from django.utils.functional import SimpleLazyObject, cached_property
from django.utils.version import get_version_tuple
logger = logging.getLogger('django.contrib.gis')

def load_geos():
    if False:
        return 10
    try:
        from django.conf import settings
        lib_path = settings.GEOS_LIBRARY_PATH
    except (AttributeError, ImportError, ImproperlyConfigured, OSError):
        lib_path = None
    if lib_path:
        lib_names = None
    elif os.name == 'nt':
        lib_names = ['geos_c', 'libgeos_c-1']
    elif os.name == 'posix':
        lib_names = ['geos_c', 'GEOS']
    else:
        raise ImportError('Unsupported OS "%s"' % os.name)
    if lib_names:
        for lib_name in lib_names:
            lib_path = find_library(lib_name)
            if lib_path is not None:
                break
    if lib_path is None:
        raise ImportError('Could not find the GEOS library (tried "%s"). Try setting GEOS_LIBRARY_PATH in your settings.' % '", "'.join(lib_names))
    _lgeos = CDLL(lib_path)
    _lgeos.initGEOS_r.restype = CONTEXT_PTR
    _lgeos.finishGEOS_r.argtypes = [CONTEXT_PTR]
    _lgeos.GEOSversion.restype = c_char_p
    return _lgeos
NOTICEFUNC = CFUNCTYPE(None, c_char_p, c_char_p)

def notice_h(fmt, lst):
    if False:
        print('Hello World!')
    (fmt, lst) = (fmt.decode(), lst.decode())
    try:
        warn_msg = fmt % lst
    except TypeError:
        warn_msg = fmt
    logger.warning('GEOS_NOTICE: %s\n', warn_msg)
notice_h = NOTICEFUNC(notice_h)
ERRORFUNC = CFUNCTYPE(None, c_char_p, c_char_p)

def error_h(fmt, lst):
    if False:
        return 10
    (fmt, lst) = (fmt.decode(), lst.decode())
    try:
        err_msg = fmt % lst
    except TypeError:
        err_msg = fmt
    logger.error('GEOS_ERROR: %s\n', err_msg)
error_h = ERRORFUNC(error_h)

class GEOSGeom_t(Structure):
    pass

class GEOSPrepGeom_t(Structure):
    pass

class GEOSCoordSeq_t(Structure):
    pass

class GEOSContextHandle_t(Structure):
    pass
GEOM_PTR = POINTER(GEOSGeom_t)
PREPGEOM_PTR = POINTER(GEOSPrepGeom_t)
CS_PTR = POINTER(GEOSCoordSeq_t)
CONTEXT_PTR = POINTER(GEOSContextHandle_t)
lgeos = SimpleLazyObject(load_geos)

class GEOSFuncFactory:
    """
    Lazy loading of GEOS functions.
    """
    argtypes = None
    restype = None
    errcheck = None

    def __init__(self, func_name, *, restype=None, errcheck=None, argtypes=None):
        if False:
            return 10
        self.func_name = func_name
        if restype is not None:
            self.restype = restype
        if errcheck is not None:
            self.errcheck = errcheck
        if argtypes is not None:
            self.argtypes = argtypes

    def __call__(self, *args):
        if False:
            print('Hello World!')
        return self.func(*args)

    @cached_property
    def func(self):
        if False:
            return 10
        from django.contrib.gis.geos.prototypes.threadsafe import GEOSFunc
        func = GEOSFunc(self.func_name)
        func.argtypes = self.argtypes or []
        func.restype = self.restype
        if self.errcheck:
            func.errcheck = self.errcheck
        return func

def geos_version():
    if False:
        i = 10
        return i + 15
    'Return the string version of the GEOS library.'
    return lgeos.GEOSversion()

def geos_version_tuple():
    if False:
        return 10
    'Return the GEOS version as a tuple (major, minor, subminor).'
    return get_version_tuple(geos_version().decode())