from django.contrib.gis.geos import prototypes as capi
from django.contrib.gis.geos.coordseq import GEOSCoordSeq
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.geometry import GEOSGeometry, LinearGeometryMixin
from django.contrib.gis.geos.point import Point
from django.contrib.gis.shortcuts import numpy

class LineString(LinearGeometryMixin, GEOSGeometry):
    _init_func = capi.create_linestring
    _minlength = 2
    has_cs = True

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        '\n        Initialize on the given sequence -- may take lists, tuples, NumPy arrays\n        of X,Y pairs, or Point objects.  If Point objects are used, ownership is\n        _not_ transferred to the LineString object.\n\n        Examples:\n         ls = LineString((1, 1), (2, 2))\n         ls = LineString([(1, 1), (2, 2)])\n         ls = LineString(array([(1, 1), (2, 2)]))\n         ls = LineString(Point(1, 1), Point(2, 2))\n        '
        if len(args) == 1:
            coords = args[0]
        else:
            coords = args
        if not (isinstance(coords, (tuple, list)) or (numpy and isinstance(coords, numpy.ndarray))):
            raise TypeError('Invalid initialization input for LineStrings.')
        srid = kwargs.get('srid')
        ncoords = len(coords)
        if not ncoords:
            super().__init__(self._init_func(None), srid=srid)
            return
        if ncoords < self._minlength:
            raise ValueError('%s requires at least %d points, got %s.' % (self.__class__.__name__, self._minlength, ncoords))
        numpy_coords = not isinstance(coords, (tuple, list))
        if numpy_coords:
            shape = coords.shape
            if len(shape) != 2:
                raise TypeError('Too many dimensions.')
            self._checkdim(shape[1])
            ndim = shape[1]
        else:
            ndim = None
            for coord in coords:
                if not isinstance(coord, (tuple, list, Point)):
                    raise TypeError('Each coordinate should be a sequence (list or tuple)')
                if ndim is None:
                    ndim = len(coord)
                    self._checkdim(ndim)
                elif len(coord) != ndim:
                    raise TypeError('Dimension mismatch.')
        cs = GEOSCoordSeq(capi.create_cs(ncoords, ndim), z=bool(ndim == 3))
        point_setter = cs._set_point_3d if ndim == 3 else cs._set_point_2d
        for i in range(ncoords):
            if numpy_coords:
                point_coords = coords[i, :]
            elif isinstance(coords[i], Point):
                point_coords = coords[i].tuple
            else:
                point_coords = coords[i]
            point_setter(i, point_coords)
        super().__init__(self._init_func(cs.ptr), srid=srid)

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        'Allow iteration over this LineString.'
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        if False:
            return 10
        'Return the number of points in this LineString.'
        return len(self._cs)

    def _get_single_external(self, index):
        if False:
            return 10
        return self._cs[index]
    _get_single_internal = _get_single_external

    def _set_list(self, length, items):
        if False:
            while True:
                i = 10
        ndim = self._cs.dims
        hasz = self._cs.hasz
        srid = self.srid
        cs = GEOSCoordSeq(capi.create_cs(length, ndim), z=hasz)
        for (i, c) in enumerate(items):
            cs[i] = c
        ptr = self._init_func(cs.ptr)
        if ptr:
            capi.destroy_geom(self.ptr)
            self.ptr = ptr
            if srid is not None:
                self.srid = srid
            self._post_init()
        else:
            raise GEOSException('Geometry resulting from slice deletion was invalid.')

    def _set_single(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        self._cs[index] = value

    def _checkdim(self, dim):
        if False:
            print('Hello World!')
        if dim not in (2, 3):
            raise TypeError('Dimension mismatch.')

    @property
    def tuple(self):
        if False:
            return 10
        'Return a tuple version of the geometry from the coordinate sequence.'
        return self._cs.tuple
    coords = tuple

    def _listarr(self, func):
        if False:
            return 10
        '\n        Return a sequence (list) corresponding with the given function.\n        Return a numpy array if possible.\n        '
        lst = [func(i) for i in range(len(self))]
        if numpy:
            return numpy.array(lst)
        else:
            return lst

    @property
    def array(self):
        if False:
            return 10
        'Return a numpy array for the LineString.'
        return self._listarr(self._cs.__getitem__)

    @property
    def x(self):
        if False:
            return 10
        'Return a list or numpy array of the X variable.'
        return self._listarr(self._cs.getX)

    @property
    def y(self):
        if False:
            return 10
        'Return a list or numpy array of the Y variable.'
        return self._listarr(self._cs.getY)

    @property
    def z(self):
        if False:
            while True:
                i = 10
        'Return a list or numpy array of the Z variable.'
        if not self.hasz:
            return None
        else:
            return self._listarr(self._cs.getZ)

class LinearRing(LineString):
    _minlength = 4
    _init_func = capi.create_linearring

    @property
    def is_counterclockwise(self):
        if False:
            print('Hello World!')
        if self.empty:
            raise ValueError('Orientation of an empty LinearRing cannot be determined.')
        return self._cs.is_counterclockwise