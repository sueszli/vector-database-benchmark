import ctypes
import itertools
import numpy as np
from . import QT_LIB, QtCore, QtGui, compat
__all__ = ['get_qpainterpath_element_array']
if QT_LIB.startswith('PyQt'):
    from . import sip
elif QT_LIB == 'PySide2':
    from PySide2 import __version_info__ as pyside_version_info
elif QT_LIB == 'PySide6':
    from PySide6 import __version_info__ as pyside_version_info

class QArrayDataQt5(ctypes.Structure):
    _fields_ = [('ref', ctypes.c_int), ('size', ctypes.c_int), ('alloc', ctypes.c_uint, 31), ('offset', ctypes.c_ssize_t)]

class QPainterPathPrivateQt5(ctypes.Structure):
    _fields_ = [('ref', ctypes.c_int), ('adata', ctypes.POINTER(QArrayDataQt5))]

class QArrayDataQt6(ctypes.Structure):
    _fields_ = [('ref', ctypes.c_int), ('flags', ctypes.c_uint), ('alloc', ctypes.c_ssize_t)]

class QPainterPathPrivateQt6(ctypes.Structure):
    _fields_ = [('ref', ctypes.c_int), ('adata', ctypes.POINTER(QArrayDataQt6)), ('data', ctypes.c_void_p), ('size', ctypes.c_ssize_t)]

def get_qpainterpath_element_array(qpath, nelems=None):
    if False:
        i = 10
        return i + 15
    writable = nelems is not None
    if writable:
        qpath.reserve(nelems)
    itemsize = 24
    dtype = dict(names=['x', 'y', 'c'], formats=['f8', 'f8', 'i4'], itemsize=itemsize)
    ptr0 = compat.unwrapinstance(qpath)
    pte_cp = ctypes.c_void_p.from_address(ptr0)
    if not pte_cp:
        return np.zeros(0, dtype=dtype)
    if QT_LIB in ['PyQt5', 'PySide2']:
        PTR = ctypes.POINTER(QPainterPathPrivateQt5)
        pte_ci = ctypes.cast(pte_cp, PTR).contents
        size_ci = pte_ci.adata[0]
        eptr = ctypes.addressof(size_ci) + size_ci.offset
    elif QT_LIB in ['PyQt6', 'PySide6']:
        PTR = ctypes.POINTER(QPainterPathPrivateQt6)
        pte_ci = ctypes.cast(pte_cp, PTR).contents
        size_ci = pte_ci
        eptr = pte_ci.data
    else:
        raise NotImplementedError
    if writable:
        size_ci.size = nelems
    else:
        nelems = size_ci.size
    vp = compat.voidptr(eptr, itemsize * nelems, writable)
    return np.frombuffer(vp, dtype=dtype)

class PrimitiveArray:

    def __init__(self, Klass, nfields, *, use_array=None):
        if False:
            return 10
        self._Klass = Klass
        self._nfields = nfields
        self._capa = -1
        self.use_sip_array = False
        self.use_ptr_to_array = False
        if QT_LIB.startswith('PyQt'):
            if use_array is None:
                use_array = hasattr(sip, 'array') and (393985 <= QtCore.PYQT_VERSION or 331527 <= QtCore.PYQT_VERSION < 393216)
            self.use_sip_array = use_array
        if QT_LIB.startswith('PySide'):
            if use_array is None:
                use_array = Klass is QtGui.QPainter.PixmapFragment or pyside_version_info >= (6, 4, 3)
            self.use_ptr_to_array = use_array
        self.resize(0)

    def resize(self, size):
        if False:
            for i in range(10):
                print('nop')
        if self.use_sip_array:
            if sip.SIP_VERSION >= 395016:
                if size <= self._capa:
                    self._size = size
                    return
            elif size == self._capa:
                return
            self._siparray = sip.array(self._Klass, size)
        else:
            if size <= self._capa:
                self._size = size
                return
            self._ndarray = np.empty((size, self._nfields), dtype=np.float64)
            if self.use_ptr_to_array:
                self._objs = None
            else:
                self._objs = self._wrap_instances(self._ndarray)
        self._capa = size
        self._size = size

    def _wrap_instances(self, array):
        if False:
            while True:
                i = 10
        return list(map(compat.wrapinstance, itertools.count(array.ctypes.data, array.strides[0]), itertools.repeat(self._Klass, array.shape[0])))

    def __len__(self):
        if False:
            print('Hello World!')
        return self._size

    def ndarray(self):
        if False:
            for i in range(10):
                print('nop')
        if self.use_sip_array:
            if sip.SIP_VERSION >= 395016 and np.__version__ != '1.22.4':
                mv = self._siparray
            else:
                mv = sip.voidptr(self._siparray, self._capa * self._nfields * 8)
            nd = np.frombuffer(mv, dtype=np.float64, count=self._size * self._nfields)
            return nd.reshape((-1, self._nfields))
        else:
            return self._ndarray[:self._size]

    def instances(self):
        if False:
            while True:
                i = 10
        if self.use_sip_array:
            if self._size == self._capa:
                return self._siparray
            else:
                return self._siparray[:self._size]
        if self._objs is None:
            self._objs = self._wrap_instances(self._ndarray)
        if self._size == self._capa:
            return self._objs
        else:
            return self._objs[:self._size]

    def drawargs(self):
        if False:
            print('Hello World!')
        if self.use_ptr_to_array:
            if self._capa > 0:
                ptr = compat.wrapinstance(self._ndarray.ctypes.data, self._Klass)
            else:
                ptr = None
            return (ptr, self._size)
        else:
            return (self.instances(),)