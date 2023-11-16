from collections import namedtuple
import itertools
import functools
import operator
import ctypes
import numpy as np
from numba import _helperlib
from numba.core import config
Extent = namedtuple('Extent', ['begin', 'end'])
attempt_nocopy_reshape = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_long, np.ctypeslib.ndpointer(np.ctypeslib.c_intp, ndim=1), np.ctypeslib.ndpointer(np.ctypeslib.c_intp, ndim=1), ctypes.c_long, np.ctypeslib.ndpointer(np.ctypeslib.c_intp, ndim=1), np.ctypeslib.ndpointer(np.ctypeslib.c_intp, ndim=1), ctypes.c_long, ctypes.c_int)(_helperlib.c_helpers['attempt_nocopy_reshape'])

class Dim(object):
    """A single dimension of the array

    Attributes
    ----------
    start:
        start offset
    stop:
        stop offset
    size:
        number of items
    stride:
        item stride
    """
    __slots__ = ('start', 'stop', 'size', 'stride', 'single')

    def __init__(self, start, stop, size, stride, single):
        if False:
            while True:
                i = 10
        self.start = start
        self.stop = stop
        self.size = size
        self.stride = stride
        self.single = single
        assert not single or size == 1

    def __getitem__(self, item):
        if False:
            return 10
        if isinstance(item, slice):
            (start, stop, step) = item.indices(self.size)
            stride = step * self.stride
            start = self.start + start * abs(self.stride)
            stop = self.start + stop * abs(self.stride)
            if stride == 0:
                size = 1
            else:
                size = _compute_size(start, stop, stride)
            ret = Dim(start=start, stop=stop, size=size, stride=stride, single=False)
            return ret
        else:
            sliced = self[item:item + 1] if item != -1 else self[-1:]
            if sliced.size != 1:
                raise IndexError
            return Dim(start=sliced.start, stop=sliced.stop, size=sliced.size, stride=sliced.stride, single=True)

    def get_offset(self, idx):
        if False:
            return 10
        return self.start + idx * self.stride

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        strfmt = 'Dim(start=%s, stop=%s, size=%s, stride=%s)'
        return strfmt % (self.start, self.stop, self.size, self.stride)

    def normalize(self, base):
        if False:
            print('Hello World!')
        return Dim(start=self.start - base, stop=self.stop - base, size=self.size, stride=self.stride, single=self.single)

    def copy(self, start=None, stop=None, size=None, stride=None, single=None):
        if False:
            for i in range(10):
                print('nop')
        if start is None:
            start = self.start
        if stop is None:
            stop = self.stop
        if size is None:
            size = self.size
        if stride is None:
            stride = self.stride
        if single is None:
            single = self.single
        return Dim(start, stop, size, stride, single)

    def is_contiguous(self, itemsize):
        if False:
            i = 10
            return i + 15
        return self.stride == itemsize

def compute_index(indices, dims):
    if False:
        i = 10
        return i + 15
    return sum((d.get_offset(i) for (i, d) in zip(indices, dims)))

class Element(object):
    is_array = False

    def __init__(self, extent):
        if False:
            while True:
                i = 10
        self.extent = extent

    def iter_contiguous_extent(self):
        if False:
            while True:
                i = 10
        yield self.extent

class Array(object):
    """A dummy numpy array-like object.  Consider it an array without the
    actual data, but offset from the base data pointer.

    Attributes
    ----------
    dims: tuple of Dim
        describing each dimension of the array

    ndim: int
        number of dimension

    shape: tuple of int
        size of each dimension

    strides: tuple of int
        stride of each dimension

    itemsize: int
        itemsize

    extent: (start, end)
        start and end offset containing the memory region
    """
    is_array = True

    @classmethod
    def from_desc(cls, offset, shape, strides, itemsize):
        if False:
            i = 10
            return i + 15
        dims = []
        for (ashape, astride) in zip(shape, strides):
            dim = Dim(offset, offset + ashape * astride, ashape, astride, single=False)
            dims.append(dim)
            offset = 0
        return cls(dims, itemsize)

    def __init__(self, dims, itemsize):
        if False:
            return 10
        self.dims = tuple(dims)
        self.ndim = len(self.dims)
        self.shape = tuple((dim.size for dim in self.dims))
        self.strides = tuple((dim.stride for dim in self.dims))
        self.itemsize = itemsize
        self.size = functools.reduce(operator.mul, self.shape, 1)
        self.extent = self._compute_extent()
        self.flags = self._compute_layout()

    def _compute_layout(self):
        if False:
            while True:
                i = 10
        if not self.dims:
            return {'C_CONTIGUOUS': True, 'F_CONTIGUOUS': True}
        if any([dim.stride == 0 for dim in self.dims]):
            return {'C_CONTIGUOUS': False, 'F_CONTIGUOUS': False}
        flags = {'C_CONTIGUOUS': True, 'F_CONTIGUOUS': True}
        sd = self.itemsize
        for dim in reversed(self.dims):
            if dim.size == 0:
                return {'C_CONTIGUOUS': True, 'F_CONTIGUOUS': True}
            if dim.size != 1:
                if dim.stride != sd:
                    flags['C_CONTIGUOUS'] = False
                sd *= dim.size
        sd = self.itemsize
        for dim in self.dims:
            if dim.size != 1:
                if dim.stride != sd:
                    flags['F_CONTIGUOUS'] = False
                    return flags
                sd *= dim.size
        return flags

    def _compute_extent(self):
        if False:
            print('Hello World!')
        firstidx = [0] * self.ndim
        lastidx = [s - 1 for s in self.shape]
        start = compute_index(firstidx, self.dims)
        stop = compute_index(lastidx, self.dims) + self.itemsize
        stop = max(stop, start)
        return Extent(start, stop)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<Array dims=%s itemsize=%s>' % (self.dims, self.itemsize)

    def __getitem__(self, item):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(item, tuple):
            item = [item]
        else:
            item = list(item)
        nitem = len(item)
        ndim = len(self.dims)
        if nitem > ndim:
            raise IndexError('%d extra indices given' % (nitem - ndim,))
        while len(item) < ndim:
            item.append(slice(None, None))
        dims = [dim.__getitem__(it) for (dim, it) in zip(self.dims, item)]
        newshape = [d.size for d in dims if not d.single]
        arr = Array(dims, self.itemsize)
        if newshape:
            return arr.reshape(*newshape)[0]
        else:
            return Element(arr.extent)

    @property
    def is_c_contig(self):
        if False:
            return 10
        return self.flags['C_CONTIGUOUS']

    @property
    def is_f_contig(self):
        if False:
            for i in range(10):
                print('nop')
        return self.flags['F_CONTIGUOUS']

    def iter_contiguous_extent(self):
        if False:
            for i in range(10):
                print('nop')
        ' Generates extents\n        '
        if self.is_c_contig or self.is_f_contig:
            yield self.extent
        else:
            if self.dims[0].stride < self.dims[-1].stride:
                innerdim = self.dims[0]
                outerdims = self.dims[1:]
                outershape = self.shape[1:]
            else:
                innerdim = self.dims[-1]
                outerdims = self.dims[:-1]
                outershape = self.shape[:-1]
            if innerdim.is_contiguous(self.itemsize):
                oslen = [range(s) for s in outershape]
                for indices in itertools.product(*oslen):
                    base = compute_index(indices, outerdims)
                    yield (base + innerdim.start, base + innerdim.stop)
            else:
                oslen = [range(s) for s in self.shape]
                for indices in itertools.product(*oslen):
                    offset = compute_index(indices, self.dims)
                    yield (offset, offset + self.itemsize)

    def reshape(self, *newdims, **kws):
        if False:
            print('Hello World!')
        oldnd = self.ndim
        newnd = len(newdims)
        if newdims == self.shape:
            return (self, None)
        order = kws.pop('order', 'C')
        if kws:
            raise TypeError('unknown keyword arguments %s' % kws.keys())
        if order not in 'CFA':
            raise ValueError('order not C|F|A')
        unknownidx = -1
        knownsize = 1
        for (i, dim) in enumerate(newdims):
            if dim < 0:
                if unknownidx == -1:
                    unknownidx = i
                else:
                    raise ValueError('can only specify one unknown dimension')
            else:
                knownsize *= dim
        if unknownidx >= 0:
            if knownsize == 0 or self.size % knownsize != 0:
                raise ValueError('cannot infer valid shape for unknown dimension')
            else:
                newdims = newdims[0:unknownidx] + (self.size // knownsize,) + newdims[unknownidx + 1:]
        newsize = functools.reduce(operator.mul, newdims, 1)
        if order == 'A':
            order = 'F' if self.is_f_contig else 'C'
        if newsize != self.size:
            raise ValueError('reshape changes the size of the array')
        if self.is_c_contig or self.is_f_contig:
            if order == 'C':
                newstrides = list(iter_strides_c_contig(self, newdims))
            elif order == 'F':
                newstrides = list(iter_strides_f_contig(self, newdims))
            else:
                raise AssertionError('unreachable')
        else:
            newstrides = np.empty(newnd, np.ctypeslib.c_intp)
            olddims = np.array(self.shape, dtype=np.ctypeslib.c_intp)
            oldstrides = np.array(self.strides, dtype=np.ctypeslib.c_intp)
            newdims = np.array(newdims, dtype=np.ctypeslib.c_intp)
            if not attempt_nocopy_reshape(oldnd, olddims, oldstrides, newnd, newdims, newstrides, self.itemsize, order == 'F'):
                raise NotImplementedError('reshape would require copy')
        ret = self.from_desc(self.extent.begin, shape=newdims, strides=newstrides, itemsize=self.itemsize)
        return (ret, list(self.iter_contiguous_extent()))

    def squeeze(self, axis=None):
        if False:
            return 10
        (newshape, newstrides) = ([], [])
        if axis is None:
            for (length, stride) in zip(self.shape, self.strides):
                if length != 1:
                    newshape.append(length)
                    newstrides.append(stride)
        else:
            if not isinstance(axis, tuple):
                axis = (axis,)
            for ax in axis:
                if self.shape[ax] != 1:
                    raise ValueError('cannot select an axis to squeeze out which has size not equal to one')
            for (i, (length, stride)) in enumerate(zip(self.shape, self.strides)):
                if i not in axis:
                    newshape.append(length)
                    newstrides.append(stride)
        newarr = self.from_desc(self.extent.begin, shape=newshape, strides=newstrides, itemsize=self.itemsize)
        return (newarr, list(self.iter_contiguous_extent()))

    def ravel(self, order='C'):
        if False:
            return 10
        if order not in 'CFA':
            raise ValueError('order not C|F|A')
        if order in 'CA' and self.is_c_contig or (order in 'FA' and self.is_f_contig):
            newshape = (self.size,)
            newstrides = (self.itemsize,)
            arr = self.from_desc(self.extent.begin, newshape, newstrides, self.itemsize)
            return (arr, list(self.iter_contiguous_extent()))
        else:
            raise NotImplementedError('ravel on non-contiguous array')

def iter_strides_f_contig(arr, shape=None):
    if False:
        i = 10
        return i + 15
    'yields the f-contiguous strides\n    '
    shape = arr.shape if shape is None else shape
    itemsize = arr.itemsize
    yield itemsize
    sum = 1
    for s in shape[:-1]:
        sum *= s
        yield (sum * itemsize)

def iter_strides_c_contig(arr, shape=None):
    if False:
        return 10
    'yields the c-contiguous strides\n    '
    shape = arr.shape if shape is None else shape
    itemsize = arr.itemsize

    def gen():
        if False:
            while True:
                i = 10
        yield itemsize
        sum = 1
        for s in reversed(shape[1:]):
            sum *= s
            yield (sum * itemsize)
    for i in reversed(list(gen())):
        yield i

def is_element_indexing(item, ndim):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(item, slice):
        return False
    elif isinstance(item, tuple):
        if len(item) == ndim:
            if not any((isinstance(it, slice) for it in item)):
                return True
    else:
        return True
    return False

def _compute_size(start, stop, step):
    if False:
        return 10
    'Algorithm adapted from cpython rangeobject.c\n    '
    if step > 0:
        lo = start
        hi = stop
    else:
        lo = stop
        hi = start
        step = -step
    if lo >= hi:
        return 0
    return (hi - lo - 1) // step + 1