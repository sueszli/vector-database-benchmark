"""
 This module houses the GEOSCoordSeq object, which is used internally
 by GEOSGeometry to house the actual coordinates of the Point,
 LineString, and LinearRing geometries.
"""
from ctypes import byref, c_byte, c_double, c_uint
from django.contrib.gis.geos import prototypes as capi
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.libgeos import CS_PTR
from django.contrib.gis.shortcuts import numpy

class GEOSCoordSeq(GEOSBase):
    """The internal representation of a list of coordinates inside a Geometry."""
    ptr_type = CS_PTR

    def __init__(self, ptr, z=False):
        if False:
            return 10
        'Initialize from a GEOS pointer.'
        if not isinstance(ptr, CS_PTR):
            raise TypeError('Coordinate sequence should initialize with a CS_PTR.')
        self._ptr = ptr
        self._z = z

    def __iter__(self):
        if False:
            while True:
                i = 10
        'Iterate over each point in the coordinate sequence.'
        for i in range(self.size):
            yield self[i]

    def __len__(self):
        if False:
            while True:
                i = 10
        'Return the number of points in the coordinate sequence.'
        return self.size

    def __str__(self):
        if False:
            print('Hello World!')
        'Return the string representation of the coordinate sequence.'
        return str(self.tuple)

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        'Return the coordinate sequence value at the given index.'
        self._checkindex(index)
        return self._point_getter(index)

    def __setitem__(self, index, value):
        if False:
            print('Hello World!')
        'Set the coordinate sequence value at the given index.'
        if isinstance(value, (list, tuple)):
            pass
        elif numpy and isinstance(value, numpy.ndarray):
            pass
        else:
            raise TypeError('Must set coordinate with a sequence (list, tuple, or numpy array).')
        if self.dims == 3 and self._z:
            n_args = 3
            point_setter = self._set_point_3d
        else:
            n_args = 2
            point_setter = self._set_point_2d
        if len(value) != n_args:
            raise TypeError('Dimension of value does not match.')
        self._checkindex(index)
        point_setter(index, value)

    def _checkindex(self, index):
        if False:
            return 10
        'Check the given index.'
        if not 0 <= index < self.size:
            raise IndexError('invalid GEOS Geometry index: %s' % index)

    def _checkdim(self, dim):
        if False:
            i = 10
            return i + 15
        'Check the given dimension.'
        if dim < 0 or dim > 2:
            raise GEOSException('invalid ordinate dimension "%d"' % dim)

    def _get_x(self, index):
        if False:
            print('Hello World!')
        return capi.cs_getx(self.ptr, index, byref(c_double()))

    def _get_y(self, index):
        if False:
            print('Hello World!')
        return capi.cs_gety(self.ptr, index, byref(c_double()))

    def _get_z(self, index):
        if False:
            i = 10
            return i + 15
        return capi.cs_getz(self.ptr, index, byref(c_double()))

    def _set_x(self, index, value):
        if False:
            while True:
                i = 10
        capi.cs_setx(self.ptr, index, value)

    def _set_y(self, index, value):
        if False:
            print('Hello World!')
        capi.cs_sety(self.ptr, index, value)

    def _set_z(self, index, value):
        if False:
            print('Hello World!')
        capi.cs_setz(self.ptr, index, value)

    @property
    def _point_getter(self):
        if False:
            while True:
                i = 10
        return self._get_point_3d if self.dims == 3 and self._z else self._get_point_2d

    def _get_point_2d(self, index):
        if False:
            print('Hello World!')
        return (self._get_x(index), self._get_y(index))

    def _get_point_3d(self, index):
        if False:
            for i in range(10):
                print('nop')
        return (self._get_x(index), self._get_y(index), self._get_z(index))

    def _set_point_2d(self, index, value):
        if False:
            return 10
        (x, y) = value
        self._set_x(index, x)
        self._set_y(index, y)

    def _set_point_3d(self, index, value):
        if False:
            while True:
                i = 10
        (x, y, z) = value
        self._set_x(index, x)
        self._set_y(index, y)
        self._set_z(index, z)

    def getOrdinate(self, dimension, index):
        if False:
            for i in range(10):
                print('nop')
        'Return the value for the given dimension and index.'
        self._checkindex(index)
        self._checkdim(dimension)
        return capi.cs_getordinate(self.ptr, index, dimension, byref(c_double()))

    def setOrdinate(self, dimension, index, value):
        if False:
            while True:
                i = 10
        'Set the value for the given dimension and index.'
        self._checkindex(index)
        self._checkdim(dimension)
        capi.cs_setordinate(self.ptr, index, dimension, value)

    def getX(self, index):
        if False:
            for i in range(10):
                print('nop')
        'Get the X value at the index.'
        return self.getOrdinate(0, index)

    def setX(self, index, value):
        if False:
            return 10
        'Set X with the value at the given index.'
        self.setOrdinate(0, index, value)

    def getY(self, index):
        if False:
            print('Hello World!')
        'Get the Y value at the given index.'
        return self.getOrdinate(1, index)

    def setY(self, index, value):
        if False:
            i = 10
            return i + 15
        'Set Y with the value at the given index.'
        self.setOrdinate(1, index, value)

    def getZ(self, index):
        if False:
            i = 10
            return i + 15
        'Get Z with the value at the given index.'
        return self.getOrdinate(2, index)

    def setZ(self, index, value):
        if False:
            print('Hello World!')
        'Set Z with the value at the given index.'
        self.setOrdinate(2, index, value)

    @property
    def size(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the size of this coordinate sequence.'
        return capi.cs_getsize(self.ptr, byref(c_uint()))

    @property
    def dims(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the dimensions of this coordinate sequence.'
        return capi.cs_getdims(self.ptr, byref(c_uint()))

    @property
    def hasz(self):
        if False:
            print('Hello World!')
        '\n        Return whether this coordinate sequence is 3D. This property value is\n        inherited from the parent Geometry.\n        '
        return self._z

    def clone(self):
        if False:
            while True:
                i = 10
        'Clone this coordinate sequence.'
        return GEOSCoordSeq(capi.cs_clone(self.ptr), self.hasz)

    @property
    def kml(self):
        if False:
            print('Hello World!')
        'Return the KML representation for the coordinates.'
        if self.hasz:
            substr = '%s,%s,%s '
        else:
            substr = '%s,%s,0 '
        return '<coordinates>%s</coordinates>' % ''.join((substr % self[i] for i in range(len(self)))).strip()

    @property
    def tuple(self):
        if False:
            print('Hello World!')
        'Return a tuple version of this coordinate sequence.'
        n = self.size
        get_point = self._point_getter
        if n == 1:
            return get_point(0)
        return tuple((get_point(i) for i in range(n)))

    @property
    def is_counterclockwise(self):
        if False:
            for i in range(10):
                print('nop')
        'Return whether this coordinate sequence is counterclockwise.'
        ret = c_byte()
        if not capi.cs_is_ccw(self.ptr, byref(ret)):
            raise GEOSException('Error encountered in GEOS C function "%s".' % capi.cs_is_ccw.func_name)
        return ret.value == 1