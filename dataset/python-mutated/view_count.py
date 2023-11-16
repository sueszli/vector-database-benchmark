from cython.cimports.cpython import Py_buffer
from cython.cimports.libcpp.vector import vector

@cython.cclass
class Matrix:
    view_count: cython.int
    ncols: cython.Py_ssize_t
    v: vector[cython.float]

    def __cinit__(self, ncols: cython.Py_ssize_t):
        if False:
            print('Hello World!')
        self.ncols = ncols
        self.view_count = 0

    def add_row(self):
        if False:
            print('Hello World!')
        if self.view_count > 0:
            raise ValueError("can't add row while being viewed")
        self.v.resize(self.v.size() + self.ncols)

    def __getbuffer__(self, buffer: cython.pointer(Py_buffer), flags: cython.int):
        if False:
            print('Hello World!')
        self.view_count += 1

    def __releasebuffer__(self, buffer: cython.pointer(Py_buffer)):
        if False:
            while True:
                i = 10
        self.view_count -= 1