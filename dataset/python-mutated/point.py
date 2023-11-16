from sympy.core.basic import Basic
from sympy.core.symbol import Str
from sympy.vector.vector import Vector
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.functions import _path
from sympy.core.cache import cacheit

class Point(Basic):
    """
    Represents a point in 3-D space.
    """

    def __new__(cls, name, position=Vector.zero, parent_point=None):
        if False:
            while True:
                i = 10
        name = str(name)
        if not isinstance(position, Vector):
            raise TypeError('position should be an instance of Vector, not %s' % type(position))
        if not isinstance(parent_point, Point) and parent_point is not None:
            raise TypeError('parent_point should be an instance of Point, not %s' % type(parent_point))
        if parent_point is None:
            obj = super().__new__(cls, Str(name), position)
        else:
            obj = super().__new__(cls, Str(name), position, parent_point)
        obj._name = name
        obj._pos = position
        if parent_point is None:
            obj._parent = None
            obj._root = obj
        else:
            obj._parent = parent_point
            obj._root = parent_point._root
        return obj

    @cacheit
    def position_wrt(self, other):
        if False:
            print('Hello World!')
        "\n        Returns the position vector of this Point with respect to\n        another Point/CoordSys3D.\n\n        Parameters\n        ==========\n\n        other : Point/CoordSys3D\n            If other is a Point, the position of this Point wrt it is\n            returned. If its an instance of CoordSyRect, the position\n            wrt its origin is returned.\n\n        Examples\n        ========\n\n        >>> from sympy.vector import CoordSys3D\n        >>> N = CoordSys3D('N')\n        >>> p1 = N.origin.locate_new('p1', 10 * N.i)\n        >>> N.origin.position_wrt(p1)\n        (-10)*N.i\n\n        "
        if not isinstance(other, Point) and (not isinstance(other, CoordSys3D)):
            raise TypeError(str(other) + 'is not a Point or CoordSys3D')
        if isinstance(other, CoordSys3D):
            other = other.origin
        if other == self:
            return Vector.zero
        elif other == self._parent:
            return self._pos
        elif other._parent == self:
            return -1 * other._pos
        (rootindex, path) = _path(self, other)
        result = Vector.zero
        i = -1
        for i in range(rootindex):
            result += path[i]._pos
        i += 2
        while i < len(path):
            result -= path[i]._pos
            i += 1
        return result

    def locate_new(self, name, position):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns a new Point located at the given position wrt this\n        Point.\n        Thus, the position vector of the new Point wrt this one will\n        be equal to the given 'position' parameter.\n\n        Parameters\n        ==========\n\n        name : str\n            Name of the new point\n\n        position : Vector\n            The position vector of the new Point wrt this one\n\n        Examples\n        ========\n\n        >>> from sympy.vector import CoordSys3D\n        >>> N = CoordSys3D('N')\n        >>> p1 = N.origin.locate_new('p1', 10 * N.i)\n        >>> p1.position_wrt(N.origin)\n        10*N.i\n\n        "
        return Point(name, position, self)

    def express_coordinates(self, coordinate_system):
        if False:
            while True:
                i = 10
        "\n        Returns the Cartesian/rectangular coordinates of this point\n        wrt the origin of the given CoordSys3D instance.\n\n        Parameters\n        ==========\n\n        coordinate_system : CoordSys3D\n            The coordinate system to express the coordinates of this\n            Point in.\n\n        Examples\n        ========\n\n        >>> from sympy.vector import CoordSys3D\n        >>> N = CoordSys3D('N')\n        >>> p1 = N.origin.locate_new('p1', 10 * N.i)\n        >>> p2 = p1.locate_new('p2', 5 * N.j)\n        >>> p2.express_coordinates(N)\n        (10, 5, 0)\n\n        "
        pos_vect = self.position_wrt(coordinate_system.origin)
        return tuple(pos_vect.to_matrix(coordinate_system))

    def _sympystr(self, printer):
        if False:
            print('Hello World!')
        return self._name