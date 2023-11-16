"""
A collection is a container for several (optionally indexed) objects having
the same vertex structure (vtype) and same uniforms type (utype). A collection
allows to manipulate objects individually and each object can have its own set
of uniforms provided they are a combination of floats.
"""
from __future__ import division
import math
import numpy as np
from ...gloo import Texture2D, VertexBuffer, IndexBuffer
from .util import dtype_reduce
from .array_list import ArrayList

def next_power_of_2(n):
    if False:
        return 10
    'Return next power of 2 greater than or equal to n'
    n -= 1
    shift = 1
    while n + 1 & n:
        n |= n >> shift
        shift *= 2
    return max(4, n + 1)

class Item(object):
    """
    An item represent an object within a collection and is created on demand
    when accessing a specific object of the collection.
    """

    def __init__(self, parent, key, vertices, indices, uniforms):
        if False:
            print('Hello World!')
        '\n        Create an item from an existing collection.\n\n        Parameters\n        ----------\n        parent : Collection\n            Collection this item belongs to\n\n        key : int\n            Collection index of this item\n\n        vertices: array-like\n            Vertices of the item\n\n        indices: array-like\n            Indices of the item\n\n        uniforms: array-like\n            Uniform parameters of the item\n        '
        self._parent = parent
        self._key = key
        self._vertices = vertices
        self._indices = indices
        self._uniforms = uniforms

    @property
    def vertices(self):
        if False:
            while True:
                i = 10
        return self._vertices

    @vertices.setter
    def vertices(self, data):
        if False:
            i = 10
            return i + 15
        self._vertices[...] = np.array(data)

    @property
    def indices(self):
        if False:
            while True:
                i = 10
        return self._indices

    @indices.setter
    def indices(self, data):
        if False:
            return 10
        if self._indices is None:
            raise ValueError('Item has no indices')
        start = self._parent.vertices._items[self._key][0]
        self._indices[...] = np.array(data) + start

    @property
    def uniforms(self):
        if False:
            while True:
                i = 10
        return self._uniforms

    @uniforms.setter
    def uniforms(self, data):
        if False:
            while True:
                i = 10
        if self._uniforms is None:
            raise ValueError('Item has no associated uniform')
        self._uniforms[...] = data

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        'Get a specific uniforms value'
        if key in self._vertices.dtype.names:
            return self._vertices[key]
        elif key in self._uniforms.dtype.names:
            return self._uniforms[key]
        else:
            raise IndexError("Unknown key ('%s')" % key)

    def __setitem__(self, key, value):
        if False:
            while True:
                i = 10
        'Set a specific uniforms value'
        if key in self._vertices.dtype.names:
            self._vertices[key] = value
        elif key in self._uniforms.dtype.names:
            self._uniforms[key] = value
        else:
            raise IndexError('Unknown key')

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'Item (%s, %s, %s)' % (self._vertices, self._indices, self._uniforms)

class BaseCollection(object):

    def __init__(self, vtype, utype=None, itype=None):
        if False:
            return 10
        self._vertices_list = None
        self._vertices_buffer = None
        self._indices_list = None
        self._indices_buffer = None
        self._uniforms_list = None
        self._uniforms_texture = None
        vtype = np.dtype(vtype) if vtype is not None else None
        itype = np.dtype(itype) if itype is not None else None
        utype = np.dtype(utype) if utype is not None else None
        if vtype.names is None:
            raise ValueError('vtype must be a structured dtype')
        if itype is not None:
            if itype not in [np.uint8, np.uint16, np.uint32]:
                raise ValueError('itype must be unsigned integer or None')
            self._indices_list = ArrayList(dtype=itype)
        self._programs = []
        self._need_update = True
        if utype is not None:
            if utype.names is None:
                raise ValueError('utype must be a structured dtype')
            vtype = eval(str(np.dtype(vtype)))
            vtype.append(('collection_index', np.float32))
            vtype = np.dtype(vtype)
            utype = eval(str(np.dtype(utype)))
            r_utype = dtype_reduce(utype)
            if not isinstance(r_utype[0], str) or r_utype[2] != 'float32':
                raise RuntimeError('utype cannot be reduced to float32 only')
            count = next_power_of_2(r_utype[1])
            if count - r_utype[1] > 0:
                utype.append(('__unused__', np.float32, count - r_utype[1]))
            self._uniforms_list = ArrayList(dtype=utype)
            self._uniforms_float_count = count
            shape = self._compute_texture_shape(1)
            self._uniforms_list.reserve(shape[1] / (count / 4))
        self._vertices_list = ArrayList(dtype=vtype)
        self._vtype = np.dtype(vtype)
        self._itype = np.dtype(itype) if itype is not None else None
        self._utype = np.dtype(utype) if utype is not None else None

    def __len__(self):
        if False:
            print('Hello World!')
        'x.__len__() <==> len(x)'
        return len(self._vertices_list)

    @property
    def vtype(self):
        if False:
            for i in range(10):
                print('nop')
        'Vertices dtype'
        return self._vtype

    @property
    def itype(self):
        if False:
            return 10
        'Indices dtype'
        return self._itype

    @property
    def utype(self):
        if False:
            while True:
                i = 10
        'Uniforms dtype'
        return self._utype

    def append(self, vertices, uniforms=None, indices=None, itemsize=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        vertices : numpy array\n            An array whose dtype is compatible with self.vdtype\n\n        uniforms: numpy array\n            An array whose dtype is compatible with self.utype\n\n        indices : numpy array\n            An array whose dtype is compatible with self.idtype\n            All index values must be between 0 and len(vertices)\n\n        itemsize: int, tuple or 1-D array\n            If `itemsize` is an integer, N, the array will be divided\n            into elements of size N. If such partition is not possible,\n            an error is raised.\n\n            If `itemsize` is 1-D array, the array will be divided into\n            elements whose successive sizes will be picked from itemsize.\n            If the sum of itemsize values is different from array size,\n            an error is raised.\n        '
        vertices = np.array(vertices).astype(self.vtype).ravel()
        vsize = self._vertices_list.size
        if itemsize is None:
            index = 0
            count = 1
        elif isinstance(itemsize, int):
            count = len(vertices) / itemsize
            index = np.repeat(np.arange(count), itemsize)
        elif isinstance(itemsize, (np.ndarray, list)):
            count = len(itemsize)
            index = np.repeat(np.arange(count), itemsize)
        else:
            raise ValueError('Itemsize not understood')
        if self.utype:
            vertices['collection_index'] = index + len(self)
        self._vertices_list.append(vertices, itemsize)
        if self.itype is not None:
            if indices is None:
                indices = vsize + np.arange(len(vertices))
                self._indices_list.append(indices, itemsize)
            else:
                if itemsize is None:
                    idxs = np.array(indices) + vsize
                elif isinstance(itemsize, int):
                    idxs = vsize + (np.tile(indices, count) + itemsize * np.repeat(np.arange(count), len(indices)))
                else:
                    raise ValueError('Indices not compatible with items')
                self._indices_list.append(idxs, len(indices))
        if self.utype:
            if uniforms is None:
                uniforms = np.zeros(count, dtype=self.utype)
            else:
                uniforms = np.array(uniforms).astype(self.utype).ravel()
            self._uniforms_list.append(uniforms, itemsize=1)
        self._need_update = True

    def __delitem__(self, index):
        if False:
            while True:
                i = 10
        'x.__delitem__(y) <==> del x[y]'
        if isinstance(index, int):
            if index < 0:
                index += len(self)
            if index < 0 or index > len(self):
                raise IndexError('Collection deletion index out of range')
            (istart, istop) = (index, index + 1)
        elif isinstance(index, slice):
            (istart, istop, _) = index.indices(len(self))
            if istart > istop:
                (istart, istop) = (istop, istart)
            if istart == istop:
                return
        elif index is Ellipsis:
            (istart, istop) = (0, len(self))
        else:
            raise TypeError('Collection deletion indices must be integers')
        vsize = len(self._vertices_list[index])
        if self.itype is not None:
            del self._indices_list[index]
            self._indices_list[index:] -= vsize
        if self.utype:
            self._vertices_list[index:]['collection_index'] -= istop - istart
        del self._vertices_list[index]
        if self.utype is not None:
            del self._uniforms_list[index]
        self._need_update = True

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        ' '
        if self._need_update:
            self._update()
        V = self._vertices_buffer
        idxs = None
        U = None
        if self._indices_list is not None:
            idxs = self._indices_buffer
        if self._uniforms_list is not None:
            U = self._uniforms_texture.data.ravel().view(self.utype)
        if isinstance(key, str):
            if key in V.dtype.names:
                return V[key]
            elif U is not None and key in U.dtype.names:
                return U[key][:len(self._uniforms_list)]
            else:
                raise IndexError("Unknown field name ('%s')" % key)
        elif isinstance(key, int):
            (vstart, vend) = self._vertices_list._items[key]
            vertices = V[vstart:vend]
            indices = None
            uniforms = None
            if idxs is not None:
                (istart, iend) = self._indices_list._items[key]
                indices = idxs[istart:iend]
            if U is not None:
                (ustart, uend) = self._uniforms_list._items[key]
                uniforms = U[ustart:uend]
            return Item(self, key, vertices, indices, uniforms)
        else:
            raise IndexError('Cannot get more than one item at once')

    def __setitem__(self, key, data):
        if False:
            return 10
        'x.__setitem__(i, y) <==> x[i]=y'
        if self._need_update:
            self._update()
        V = self._vertices_buffer
        U = None
        if self._uniforms_list is not None:
            U = self._uniforms_texture.data.ravel().view(self.utype)
        if isinstance(key, str):
            if key in self.vtype.names:
                V[key] = data
            elif self.utype and key in self.utype.names:
                U[key][:len(self._uniforms_list)] = data
            else:
                raise IndexError("Unknown field name ('%s')" % key)
        else:
            raise IndexError('Cannot set more than one item')

    def _compute_texture_shape(self, size=1):
        if False:
            for i in range(10):
                print('nop')
        'Compute uniform texture shape'
        linesize = 1024
        count = self._uniforms_float_count
        cols = 4 * linesize // int(count)
        rows = max(1, int(math.ceil(size / float(cols))))
        shape = (rows, cols * (count // 4), count)
        self._ushape = shape
        return shape

    def _update(self):
        if False:
            print('Hello World!')
        'Update vertex buffers & texture'
        if self._vertices_buffer is not None:
            self._vertices_buffer.delete()
        self._vertices_buffer = VertexBuffer(self._vertices_list.data)
        if self.itype is not None:
            if self._indices_buffer is not None:
                self._indices_buffer.delete()
            self._indices_buffer = IndexBuffer(self._indices_list.data)
        if self.utype is not None:
            if self._uniforms_texture is not None:
                self._uniforms_texture.delete()
            texture = self._uniforms_list._data.view(np.float32)
            size = len(texture) / self._uniforms_float_count
            shape = self._compute_texture_shape(size)
            texture = texture.reshape(shape[0], shape[1], 4)
            self._uniforms_texture = Texture2D(texture)
            self._uniforms_texture.data = texture
            self._uniforms_texture.interpolation = 'nearest'
        if len(self._programs):
            for program in self._programs:
                program.bind(self._vertices_buffer)
                if self._uniforms_list is not None:
                    program['uniforms'] = self._uniforms_texture
                    program['uniforms_shape'] = self._ushape