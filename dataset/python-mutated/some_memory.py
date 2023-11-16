from cython.cimports.cpython.mem import PyMem_Malloc, PyMem_Realloc, PyMem_Free

@cython.cclass
class SomeMemory:
    data: cython.p_double

    def __cinit__(self, number: cython.size_t):
        if False:
            i = 10
            return i + 15
        self.data = cython.cast(cython.p_double, PyMem_Malloc(number * cython.sizeof(cython.double)))
        if not self.data:
            raise MemoryError()

    def resize(self, new_number: cython.size_t):
        if False:
            i = 10
            return i + 15
        mem = cython.cast(cython.p_double, PyMem_Realloc(self.data, new_number * cython.sizeof(cython.double)))
        if not mem:
            raise MemoryError()
        self.data = mem

    def __dealloc__(self):
        if False:
            while True:
                i = 10
        PyMem_Free(self.data)