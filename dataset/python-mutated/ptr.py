from ctypes import c_void_p

class CPointerBase:
    """
    Base class for objects that have a pointer access property
    that controls access to the underlying C pointer.
    """
    _ptr = None
    ptr_type = c_void_p
    destructor = None
    null_ptr_exception_class = AttributeError

    @property
    def ptr(self):
        if False:
            while True:
                i = 10
        if self._ptr:
            return self._ptr
        raise self.null_ptr_exception_class('NULL %s pointer encountered.' % self.__class__.__name__)

    @ptr.setter
    def ptr(self, ptr):
        if False:
            while True:
                i = 10
        if not (ptr is None or isinstance(ptr, self.ptr_type)):
            raise TypeError('Incompatible pointer type: %s.' % type(ptr))
        self._ptr = ptr

    def __del__(self):
        if False:
            return 10
        '\n        Free the memory used by the C++ object.\n        '
        if self.destructor and self._ptr:
            try:
                self.destructor(self.ptr)
            except (AttributeError, ImportError, TypeError):
                pass