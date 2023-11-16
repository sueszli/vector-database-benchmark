import cython
from cython.cimports.libc.stdlib import free

@cython.cclass
class OwnedPointer:
    ptr: cython.pointer(cython.void)

    def __dealloc__(self):
        if False:
            print('Hello World!')
        if self.ptr is not cython.NULL:
            free(self.ptr)

    @staticmethod
    @cython.cfunc
    def create(ptr: cython.pointer(cython.void)):
        if False:
            return 10
        p = OwnedPointer()
        p.ptr = ptr
        return p