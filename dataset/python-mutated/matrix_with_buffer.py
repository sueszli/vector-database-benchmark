from cython.cimports.cpython import Py_buffer
from cython.cimports.libcpp.vector import vector

@cython.cclass
class Matrix:
    ncols: cython.Py_ssize_t
    shape: cython.Py_ssize_t[2]
    strides: cython.Py_ssize_t[2]
    v: vector[cython.float]

    def __cinit__(self, ncols: cython.Py_ssize_t):
        if False:
            for i in range(10):
                print('nop')
        self.ncols = ncols

    def add_row(self):
        if False:
            for i in range(10):
                print('nop')
        'Adds a row, initially zero-filled.'
        self.v.resize(self.v.size() + self.ncols)

    def __getbuffer__(self, buffer: cython.pointer(Py_buffer), flags: cython.int):
        if False:
            while True:
                i = 10
        itemsize: cython.Py_ssize_t = cython.sizeof(self.v[0])
        self.shape[0] = self.v.size() // self.ncols
        self.shape[1] = self.ncols
        self.strides[1] = cython.cast(cython.Py_ssize_t, cython.cast(cython.p_char, cython.address(self.v[1])) - cython.cast(cython.p_char, cython.address(self.v[0])))
        self.strides[0] = self.ncols * self.strides[1]
        buffer.buf = cython.cast(cython.p_char, cython.address(self.v[0]))
        buffer.format = 'f'
        buffer.internal = cython.NULL
        buffer.itemsize = itemsize
        buffer.len = self.v.size() * itemsize
        buffer.ndim = 2
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = cython.NULL

    def __releasebuffer__(self, buffer: cython.pointer(Py_buffer)):
        if False:
            for i in range(10):
                print('nop')
        pass