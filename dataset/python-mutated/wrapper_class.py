import cython
from cython.cimports.libc.stdlib import malloc, free
my_c_struct = cython.struct(a=cython.int, b=cython.int)

@cython.cclass
class WrapperClass:
    """A wrapper class for a C/C++ data structure"""
    _ptr: cython.pointer(my_c_struct)
    ptr_owner: cython.bint

    def __cinit__(self):
        if False:
            while True:
                i = 10
        self.ptr_owner = False

    def __dealloc__(self):
        if False:
            return 10
        if self._ptr is not cython.NULL and self.ptr_owner is True:
            free(self._ptr)
            self._ptr = cython.NULL

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        raise TypeError('This class cannot be instantiated directly.')

    @property
    def a(self):
        if False:
            i = 10
            return i + 15
        return self._ptr.a if self._ptr is not cython.NULL else None

    @property
    def b(self):
        if False:
            while True:
                i = 10
        return self._ptr.b if self._ptr is not cython.NULL else None

    @staticmethod
    @cython.cfunc
    def from_ptr(_ptr: cython.pointer(my_c_struct), owner: cython.bint=False) -> WrapperClass:
        if False:
            print('Hello World!')
        'Factory function to create WrapperClass objects from\n        given my_c_struct pointer.\n\n        Setting ``owner`` flag to ``True`` causes\n        the extension type to ``free`` the structure pointed to by ``_ptr``\n        when the wrapper object is deallocated.'
        wrapper: WrapperClass = WrapperClass.__new__(WrapperClass)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    @cython.cfunc
    def new_struct() -> WrapperClass:
        if False:
            while True:
                i = 10
        'Factory function to create WrapperClass objects with\n        newly allocated my_c_struct'
        _ptr: cython.pointer(my_c_struct) = cython.cast(cython.pointer(my_c_struct), malloc(cython.sizeof(my_c_struct)))
        if _ptr is cython.NULL:
            raise MemoryError
        _ptr.a = 0
        _ptr.b = 0
        return WrapperClass.from_ptr(_ptr, owner=True)