from cython.cimports.libc.stdio import FILE, fopen
from cython.cimports.libc.stdlib import malloc, free
from cython.cimports.cpython.exc import PyErr_SetFromErrnoWithFilenameObject

def open_file():
    if False:
        return 10
    p = fopen('spam.txt', 'r')
    if p is cython.NULL:
        PyErr_SetFromErrnoWithFilenameObject(OSError, 'spam.txt')
    ...

def allocating_memory(number=10):
    if False:
        for i in range(10):
            print('nop')
    my_array = cython.cast(p_double, malloc(number * cython.sizeof(double)))
    if not my_array:
        raise MemoryError()
    ...
    free(my_array)