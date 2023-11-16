"""Geometrical Planes.

Contains
========
Plane

"""
from sympy.core import Dummy, Rational, S, Symbol
from sympy.core.symbol import _symbol
from sympy.functions.elementary.trigonometric import cos, sin, acos, asin, sqrt
from .entity import GeometryEntity
from .line import Line, Ray, Segment, Line3D, LinearEntity, LinearEntity3D, Ray3D, Segment3D
from .point import Point, Point3D
from sympy.matrices import Matrix
from sympy.polys.polytools import cancel
from sympy.solvers import solve, linsolve
from sympy.utilities.iterables import uniq, is_sequence
from sympy.utilities.misc import filldedent, func_name, Undecidable
from mpmath.libmp.libmpf import prec_to_dps
import random
(x, y, z, t) = [Dummy('plane_dummy') for i in range(4)]

class Plane(GeometryEntity):
    """
    A plane is a flat, two-dimensional surface. A plane is the two-dimensional
    analogue of a point (zero-dimensions), a line (one-dimension) and a solid
    (three-dimensions). A plane can generally be constructed by two types of
    inputs. They are:
    - three non-collinear points
    - a point and the plane's normal vector

    Attributes
    ==========

    p1
    normal_vector

    Examples
    ========

    >>> from sympy import Plane, Point3D
    >>> Plane(Point3D(1, 1, 1), Point3D(2, 3, 4), Point3D(2, 2, 2))
    Plane(Point3D(1, 1, 1), (-1, 2, -1))
    >>> Plane((1, 1, 1), (2, 3, 4), (2, 2, 2))
    Plane(Point3D(1, 1, 1), (-1, 2, -1))
    >>> Plane(Point3D(1, 1, 1), normal_vector=(1,4,7))
    Plane(Point3D(1, 1, 1), (1, 4, 7))

    """

    def __new__(cls, p1, a=None, b=None, **kwargs):
        if False:
            i = 10
            return i + 15
        p1 = Point3D(p1, dim=3)
        if a and b:
            p2 = Point(a, dim=3)
            p3 = Point(b, dim=3)
            if Point3D.are_collinear(p1, p2, p3):
                raise ValueError('Enter three non-collinear points')
            a = p1.direction_ratio(p2)
            b = p1.direction_ratio(p3)
            normal_vector = tuple(Matrix(a).cross(Matrix(b)))
        else:
            a = kwargs.pop('normal_vector', a)
            evaluate = kwargs.get('evaluate', True)
            if is_sequence(a) and len(a) == 3:
                normal_vector = Point3D(a).args if evaluate else a
            else:
                raise ValueError(filldedent('\n                    Either provide 3 3D points or a point with a\n                    normal vector expressed as a sequence of length 3'))
            if all((coord.is_zero for coord in normal_vector)):
                raise ValueError('Normal vector cannot be zero vector')
        return GeometryEntity.__new__(cls, p1, normal_vector, **kwargs)

    def __contains__(self, o):
        if False:
            i = 10
            return i + 15
        k = self.equation(x, y, z)
        if isinstance(o, (LinearEntity, LinearEntity3D)):
            d = Point3D(o.arbitrary_point(t))
            e = k.subs([(x, d.x), (y, d.y), (z, d.z)])
            return e.equals(0)
        try:
            o = Point(o, dim=3, strict=True)
            d = k.xreplace(dict(zip((x, y, z), o.args)))
            return d.equals(0)
        except TypeError:
            return False

    def _eval_evalf(self, prec=15, **options):
        if False:
            for i in range(10):
                print('nop')
        (pt, tup) = self.args
        dps = prec_to_dps(prec)
        pt = pt.evalf(n=dps, **options)
        tup = tuple([i.evalf(n=dps, **options) for i in tup])
        return self.func(pt, normal_vector=tup, evaluate=False)

    def angle_between(self, o):
        if False:
            return 10
        "Angle between the plane and other geometric entity.\n\n        Parameters\n        ==========\n\n        LinearEntity3D, Plane.\n\n        Returns\n        =======\n\n        angle : angle in radians\n\n        Notes\n        =====\n\n        This method accepts only 3D entities as it's parameter, but if you want\n        to calculate the angle between a 2D entity and a plane you should\n        first convert to a 3D entity by projecting onto a desired plane and\n        then proceed to calculate the angle.\n\n        Examples\n        ========\n\n        >>> from sympy import Point3D, Line3D, Plane\n        >>> a = Plane(Point3D(1, 2, 2), normal_vector=(1, 2, 3))\n        >>> b = Line3D(Point3D(1, 3, 4), Point3D(2, 2, 2))\n        >>> a.angle_between(b)\n        -asin(sqrt(21)/6)\n\n        "
        if isinstance(o, LinearEntity3D):
            a = Matrix(self.normal_vector)
            b = Matrix(o.direction_ratio)
            c = a.dot(b)
            d = sqrt(sum([i ** 2 for i in self.normal_vector]))
            e = sqrt(sum([i ** 2 for i in o.direction_ratio]))
            return asin(c / (d * e))
        if isinstance(o, Plane):
            a = Matrix(self.normal_vector)
            b = Matrix(o.normal_vector)
            c = a.dot(b)
            d = sqrt(sum([i ** 2 for i in self.normal_vector]))
            e = sqrt(sum([i ** 2 for i in o.normal_vector]))
            return acos(c / (d * e))

    def arbitrary_point(self, u=None, v=None):
        if False:
            i = 10
            return i + 15
        ' Returns an arbitrary point on the Plane. If given two\n        parameters, the point ranges over the entire plane. If given 1\n        or no parameters, returns a point with one parameter which,\n        when varying from 0 to 2*pi, moves the point in a circle of\n        radius 1 about p1 of the Plane.\n\n        Examples\n        ========\n\n        >>> from sympy import Plane, Ray\n        >>> from sympy.abc import u, v, t, r\n        >>> p = Plane((1, 1, 1), normal_vector=(1, 0, 0))\n        >>> p.arbitrary_point(u, v)\n        Point3D(1, u + 1, v + 1)\n        >>> p.arbitrary_point(t)\n        Point3D(1, cos(t) + 1, sin(t) + 1)\n\n        While arbitrary values of u and v can move the point anywhere in\n        the plane, the single-parameter point can be used to construct a\n        ray whose arbitrary point can be located at angle t and radius\n        r from p.p1:\n\n        >>> Ray(p.p1, _).arbitrary_point(r)\n        Point3D(1, r*cos(t) + 1, r*sin(t) + 1)\n\n        Returns\n        =======\n\n        Point3D\n\n        '
        circle = v is None
        if circle:
            u = _symbol(u or 't', real=True)
        else:
            u = _symbol(u or 'u', real=True)
            v = _symbol(v or 'v', real=True)
        (x, y, z) = self.normal_vector
        (a, b, c) = self.p1.args
        if x.is_zero and y.is_zero:
            (x1, y1, z1) = (S.One, S.Zero, S.Zero)
        else:
            (x1, y1, z1) = (-y, x, S.Zero)
        (x2, y2, z2) = tuple(Matrix((x, y, z)).cross(Matrix((x1, y1, z1))))
        if circle:
            (x1, y1, z1) = (w / sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2) for w in (x1, y1, z1))
            (x2, y2, z2) = (w / sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2) for w in (x2, y2, z2))
            p = Point3D(a + x1 * cos(u) + x2 * sin(u), b + y1 * cos(u) + y2 * sin(u), c + z1 * cos(u) + z2 * sin(u))
        else:
            p = Point3D(a + x1 * u + x2 * v, b + y1 * u + y2 * v, c + z1 * u + z2 * v)
        return p

    @staticmethod
    def are_concurrent(*planes):
        if False:
            print('Hello World!')
        'Is a sequence of Planes concurrent?\n\n        Two or more Planes are concurrent if their intersections\n        are a common line.\n\n        Parameters\n        ==========\n\n        planes: list\n\n        Returns\n        =======\n\n        Boolean\n\n        Examples\n        ========\n\n        >>> from sympy import Plane, Point3D\n        >>> a = Plane(Point3D(5, 0, 0), normal_vector=(1, -1, 1))\n        >>> b = Plane(Point3D(0, -2, 0), normal_vector=(3, 1, 1))\n        >>> c = Plane(Point3D(0, -1, 0), normal_vector=(5, -1, 9))\n        >>> Plane.are_concurrent(a, b)\n        True\n        >>> Plane.are_concurrent(a, b, c)\n        False\n\n        '
        planes = list(uniq(planes))
        for i in planes:
            if not isinstance(i, Plane):
                raise ValueError('All objects should be Planes but got %s' % i.func)
        if len(planes) < 2:
            return False
        planes = list(planes)
        first = planes.pop(0)
        sol = first.intersection(planes[0])
        if sol == []:
            return False
        else:
            line = sol[0]
            for i in planes[1:]:
                l = first.intersection(i)
                if not l or l[0] not in line:
                    return False
            return True

    def distance(self, o):
        if False:
            return 10
        "Distance between the plane and another geometric entity.\n\n        Parameters\n        ==========\n\n        Point3D, LinearEntity3D, Plane.\n\n        Returns\n        =======\n\n        distance\n\n        Notes\n        =====\n\n        This method accepts only 3D entities as it's parameter, but if you want\n        to calculate the distance between a 2D entity and a plane you should\n        first convert to a 3D entity by projecting onto a desired plane and\n        then proceed to calculate the distance.\n\n        Examples\n        ========\n\n        >>> from sympy import Point3D, Line3D, Plane\n        >>> a = Plane(Point3D(1, 1, 1), normal_vector=(1, 1, 1))\n        >>> b = Point3D(1, 2, 3)\n        >>> a.distance(b)\n        sqrt(3)\n        >>> c = Line3D(Point3D(2, 3, 1), Point3D(1, 2, 2))\n        >>> a.distance(c)\n        0\n\n        "
        if self.intersection(o) != []:
            return S.Zero
        if isinstance(o, (Segment3D, Ray3D)):
            (a, b) = (o.p1, o.p2)
            (pi,) = self.intersection(Line3D(a, b))
            if pi in o:
                return self.distance(pi)
            elif a in Segment3D(pi, b):
                return self.distance(a)
            else:
                assert isinstance(o, Segment3D) is True
                return self.distance(b)
        a = o if isinstance(o, Point3D) else o.p1
        n = Point3D(self.normal_vector).unit
        d = (a - self.p1).dot(n)
        return abs(d)

    def equals(self, o):
        if False:
            while True:
                i = 10
        '\n        Returns True if self and o are the same mathematical entities.\n\n        Examples\n        ========\n\n        >>> from sympy import Plane, Point3D\n        >>> a = Plane(Point3D(1, 2, 3), normal_vector=(1, 1, 1))\n        >>> b = Plane(Point3D(1, 2, 3), normal_vector=(2, 2, 2))\n        >>> c = Plane(Point3D(1, 2, 3), normal_vector=(-1, 4, 6))\n        >>> a.equals(a)\n        True\n        >>> a.equals(b)\n        True\n        >>> a.equals(c)\n        False\n        '
        if isinstance(o, Plane):
            a = self.equation()
            b = o.equation()
            return cancel(a / b).is_constant()
        else:
            return False

    def equation(self, x=None, y=None, z=None):
        if False:
            for i in range(10):
                print('nop')
        'The equation of the Plane.\n\n        Examples\n        ========\n\n        >>> from sympy import Point3D, Plane\n        >>> a = Plane(Point3D(1, 1, 2), Point3D(2, 4, 7), Point3D(3, 5, 1))\n        >>> a.equation()\n        -23*x + 11*y - 2*z + 16\n        >>> a = Plane(Point3D(1, 4, 2), normal_vector=(6, 6, 6))\n        >>> a.equation()\n        6*x + 6*y + 6*z - 42\n\n        '
        (x, y, z) = [i if i else Symbol(j, real=True) for (i, j) in zip((x, y, z), 'xyz')]
        a = Point3D(x, y, z)
        b = self.p1.direction_ratio(a)
        c = self.normal_vector
        return sum((i * j for (i, j) in zip(b, c)))

    def intersection(self, o):
        if False:
            return 10
        ' The intersection with other geometrical entity.\n\n        Parameters\n        ==========\n\n        Point, Point3D, LinearEntity, LinearEntity3D, Plane\n\n        Returns\n        =======\n\n        List\n\n        Examples\n        ========\n\n        >>> from sympy import Point3D, Line3D, Plane\n        >>> a = Plane(Point3D(1, 2, 3), normal_vector=(1, 1, 1))\n        >>> b = Point3D(1, 2, 3)\n        >>> a.intersection(b)\n        [Point3D(1, 2, 3)]\n        >>> c = Line3D(Point3D(1, 4, 7), Point3D(2, 2, 2))\n        >>> a.intersection(c)\n        [Point3D(2, 2, 2)]\n        >>> d = Plane(Point3D(6, 0, 0), normal_vector=(2, -5, 3))\n        >>> e = Plane(Point3D(2, 0, 0), normal_vector=(3, 4, -3))\n        >>> d.intersection(e)\n        [Line3D(Point3D(78/23, -24/23, 0), Point3D(147/23, 321/23, 23))]\n\n        '
        if not isinstance(o, GeometryEntity):
            o = Point(o, dim=3)
        if isinstance(o, Point):
            if o in self:
                return [o]
            else:
                return []
        if isinstance(o, (LinearEntity, LinearEntity3D)):
            (p1, p2) = (o.p1, o.p2)
            if isinstance(o, Segment):
                o = Segment3D(p1, p2)
            elif isinstance(o, Ray):
                o = Ray3D(p1, p2)
            elif isinstance(o, Line):
                o = Line3D(p1, p2)
            else:
                raise ValueError('unhandled linear entity: %s' % o.func)
            if o in self:
                return [o]
            else:
                a = Point3D(o.arbitrary_point(t))
                (p1, n) = (self.p1, Point3D(self.normal_vector))
                c = solve((a - p1).dot(n), t)
                if not c:
                    return []
                else:
                    c = [i for i in c if i.is_real is not False]
                    if len(c) > 1:
                        c = [i for i in c if i.is_real]
                    if len(c) != 1:
                        raise Undecidable('not sure which point is real')
                    p = a.subs(t, c[0])
                    if p not in o:
                        return []
                    return [p]
        if isinstance(o, Plane):
            if self.equals(o):
                return [self]
            if self.is_parallel(o):
                return []
            else:
                (x, y, z) = map(Dummy, 'xyz')
                (a, b) = (Matrix([self.normal_vector]), Matrix([o.normal_vector]))
                c = list(a.cross(b))
                d = self.equation(x, y, z)
                e = o.equation(x, y, z)
                result = list(linsolve([d, e], x, y, z))[0]
                for i in (x, y, z):
                    result = result.subs(i, 0)
                return [Line3D(Point3D(result), direction_ratio=c)]

    def is_coplanar(self, o):
        if False:
            for i in range(10):
                print('nop')
        ' Returns True if `o` is coplanar with self, else False.\n\n        Examples\n        ========\n\n        >>> from sympy import Plane\n        >>> o = (0, 0, 0)\n        >>> p = Plane(o, (1, 1, 1))\n        >>> p2 = Plane(o, (2, 2, 2))\n        >>> p == p2\n        False\n        >>> p.is_coplanar(p2)\n        True\n        '
        if isinstance(o, Plane):
            return not cancel(self.equation(x, y, z) / o.equation(x, y, z)).has(x, y, z)
        if isinstance(o, Point3D):
            return o in self
        elif isinstance(o, LinearEntity3D):
            return all((i in self for i in self))
        elif isinstance(o, GeometryEntity):
            return all((i == 0 for i in self.normal_vector[:2]))

    def is_parallel(self, l):
        if False:
            return 10
        'Is the given geometric entity parallel to the plane?\n\n        Parameters\n        ==========\n\n        LinearEntity3D or Plane\n\n        Returns\n        =======\n\n        Boolean\n\n        Examples\n        ========\n\n        >>> from sympy import Plane, Point3D\n        >>> a = Plane(Point3D(1,4,6), normal_vector=(2, 4, 6))\n        >>> b = Plane(Point3D(3,1,3), normal_vector=(4, 8, 12))\n        >>> a.is_parallel(b)\n        True\n\n        '
        if isinstance(l, LinearEntity3D):
            a = l.direction_ratio
            b = self.normal_vector
            c = sum([i * j for (i, j) in zip(a, b)])
            if c == 0:
                return True
            else:
                return False
        elif isinstance(l, Plane):
            a = Matrix(l.normal_vector)
            b = Matrix(self.normal_vector)
            if a.cross(b).is_zero_matrix:
                return True
            else:
                return False

    def is_perpendicular(self, l):
        if False:
            for i in range(10):
                print('nop')
        'Is the given geometric entity perpendicualar to the given plane?\n\n        Parameters\n        ==========\n\n        LinearEntity3D or Plane\n\n        Returns\n        =======\n\n        Boolean\n\n        Examples\n        ========\n\n        >>> from sympy import Plane, Point3D\n        >>> a = Plane(Point3D(1,4,6), normal_vector=(2, 4, 6))\n        >>> b = Plane(Point3D(2, 2, 2), normal_vector=(-1, 2, -1))\n        >>> a.is_perpendicular(b)\n        True\n\n        '
        if isinstance(l, LinearEntity3D):
            a = Matrix(l.direction_ratio)
            b = Matrix(self.normal_vector)
            if a.cross(b).is_zero_matrix:
                return True
            else:
                return False
        elif isinstance(l, Plane):
            a = Matrix(l.normal_vector)
            b = Matrix(self.normal_vector)
            if a.dot(b) == 0:
                return True
            else:
                return False
        else:
            return False

    @property
    def normal_vector(self):
        if False:
            while True:
                i = 10
        'Normal vector of the given plane.\n\n        Examples\n        ========\n\n        >>> from sympy import Point3D, Plane\n        >>> a = Plane(Point3D(1, 1, 1), Point3D(2, 3, 4), Point3D(2, 2, 2))\n        >>> a.normal_vector\n        (-1, 2, -1)\n        >>> a = Plane(Point3D(1, 1, 1), normal_vector=(1, 4, 7))\n        >>> a.normal_vector\n        (1, 4, 7)\n\n        '
        return self.args[1]

    @property
    def p1(self):
        if False:
            return 10
        'The only defining point of the plane. Others can be obtained from the\n        arbitrary_point method.\n\n        See Also\n        ========\n\n        sympy.geometry.point.Point3D\n\n        Examples\n        ========\n\n        >>> from sympy import Point3D, Plane\n        >>> a = Plane(Point3D(1, 1, 1), Point3D(2, 3, 4), Point3D(2, 2, 2))\n        >>> a.p1\n        Point3D(1, 1, 1)\n\n        '
        return self.args[0]

    def parallel_plane(self, pt):
        if False:
            i = 10
            return i + 15
        '\n        Plane parallel to the given plane and passing through the point pt.\n\n        Parameters\n        ==========\n\n        pt: Point3D\n\n        Returns\n        =======\n\n        Plane\n\n        Examples\n        ========\n\n        >>> from sympy import Plane, Point3D\n        >>> a = Plane(Point3D(1, 4, 6), normal_vector=(2, 4, 6))\n        >>> a.parallel_plane(Point3D(2, 3, 5))\n        Plane(Point3D(2, 3, 5), (2, 4, 6))\n\n        '
        a = self.normal_vector
        return Plane(pt, normal_vector=a)

    def perpendicular_line(self, pt):
        if False:
            return 10
        'A line perpendicular to the given plane.\n\n        Parameters\n        ==========\n\n        pt: Point3D\n\n        Returns\n        =======\n\n        Line3D\n\n        Examples\n        ========\n\n        >>> from sympy import Plane, Point3D\n        >>> a = Plane(Point3D(1,4,6), normal_vector=(2, 4, 6))\n        >>> a.perpendicular_line(Point3D(9, 8, 7))\n        Line3D(Point3D(9, 8, 7), Point3D(11, 12, 13))\n\n        '
        a = self.normal_vector
        return Line3D(pt, direction_ratio=a)

    def perpendicular_plane(self, *pts):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return a perpendicular passing through the given points. If the\n        direction ratio between the points is the same as the Plane's normal\n        vector then, to select from the infinite number of possible planes,\n        a third point will be chosen on the z-axis (or the y-axis\n        if the normal vector is already parallel to the z-axis). If less than\n        two points are given they will be supplied as follows: if no point is\n        given then pt1 will be self.p1; if a second point is not given it will\n        be a point through pt1 on a line parallel to the z-axis (if the normal\n        is not already the z-axis, otherwise on the line parallel to the\n        y-axis).\n\n        Parameters\n        ==========\n\n        pts: 0, 1 or 2 Point3D\n\n        Returns\n        =======\n\n        Plane\n\n        Examples\n        ========\n\n        >>> from sympy import Plane, Point3D\n        >>> a, b = Point3D(0, 0, 0), Point3D(0, 1, 0)\n        >>> Z = (0, 0, 1)\n        >>> p = Plane(a, normal_vector=Z)\n        >>> p.perpendicular_plane(a, b)\n        Plane(Point3D(0, 0, 0), (1, 0, 0))\n        "
        if len(pts) > 2:
            raise ValueError('No more than 2 pts should be provided.')
        pts = list(pts)
        if len(pts) == 0:
            pts.append(self.p1)
        if len(pts) == 1:
            (x, y, z) = self.normal_vector
            if x == y == 0:
                dir = (0, 1, 0)
            else:
                dir = (0, 0, 1)
            pts.append(pts[0] + Point3D(*dir))
        (p1, p2) = [Point(i, dim=3) for i in pts]
        l = Line3D(p1, p2)
        n = Line3D(p1, direction_ratio=self.normal_vector)
        if l in n:
            (x, y, z) = self.normal_vector
            if x == y == 0:
                p3 = Point3D(0, 1, 0)
            else:
                p3 = Point3D(0, 0, 1)
            if p3 in l:
                p3 *= 2
        else:
            p3 = p1 + Point3D(*self.normal_vector)
        return Plane(p1, p2, p3)

    def projection_line(self, line):
        if False:
            for i in range(10):
                print('nop')
        'Project the given line onto the plane through the normal plane\n        containing the line.\n\n        Parameters\n        ==========\n\n        LinearEntity or LinearEntity3D\n\n        Returns\n        =======\n\n        Point3D, Line3D, Ray3D or Segment3D\n\n        Notes\n        =====\n\n        For the interaction between 2D and 3D lines(segments, rays), you should\n        convert the line to 3D by using this method. For example for finding the\n        intersection between a 2D and a 3D line, convert the 2D line to a 3D line\n        by projecting it on a required plane and then proceed to find the\n        intersection between those lines.\n\n        Examples\n        ========\n\n        >>> from sympy import Plane, Line, Line3D, Point3D\n        >>> a = Plane(Point3D(1, 1, 1), normal_vector=(1, 1, 1))\n        >>> b = Line(Point3D(1, 1), Point3D(2, 2))\n        >>> a.projection_line(b)\n        Line3D(Point3D(4/3, 4/3, 1/3), Point3D(5/3, 5/3, -1/3))\n        >>> c = Line3D(Point3D(1, 1, 1), Point3D(2, 2, 2))\n        >>> a.projection_line(c)\n        Point3D(1, 1, 1)\n\n        '
        if not isinstance(line, (LinearEntity, LinearEntity3D)):
            raise NotImplementedError('Enter a linear entity only')
        (a, b) = (self.projection(line.p1), self.projection(line.p2))
        if a == b:
            return a
        if isinstance(line, (Line, Line3D)):
            return Line3D(a, b)
        if isinstance(line, (Ray, Ray3D)):
            return Ray3D(a, b)
        if isinstance(line, (Segment, Segment3D)):
            return Segment3D(a, b)

    def projection(self, pt):
        if False:
            for i in range(10):
                print('nop')
        'Project the given point onto the plane along the plane normal.\n\n        Parameters\n        ==========\n\n        Point or Point3D\n\n        Returns\n        =======\n\n        Point3D\n\n        Examples\n        ========\n\n        >>> from sympy import Plane, Point3D\n        >>> A = Plane(Point3D(1, 1, 2), normal_vector=(1, 1, 1))\n\n        The projection is along the normal vector direction, not the z\n        axis, so (1, 1) does not project to (1, 1, 2) on the plane A:\n\n        >>> b = Point3D(1, 1)\n        >>> A.projection(b)\n        Point3D(5/3, 5/3, 2/3)\n        >>> _ in A\n        True\n\n        But the point (1, 1, 2) projects to (1, 1) on the XY-plane:\n\n        >>> XY = Plane((0, 0, 0), (0, 0, 1))\n        >>> XY.projection((1, 1, 2))\n        Point3D(1, 1, 0)\n        '
        rv = Point(pt, dim=3)
        if rv in self:
            return rv
        return self.intersection(Line3D(rv, rv + Point3D(self.normal_vector)))[0]

    def random_point(self, seed=None):
        if False:
            while True:
                i = 10
        ' Returns a random point on the Plane.\n\n        Returns\n        =======\n\n        Point3D\n\n        Examples\n        ========\n\n        >>> from sympy import Plane\n        >>> p = Plane((1, 0, 0), normal_vector=(0, 1, 0))\n        >>> r = p.random_point(seed=42)  # seed value is optional\n        >>> r.n(3)\n        Point3D(2.29, 0, -1.35)\n\n        The random point can be moved to lie on the circle of radius\n        1 centered on p1:\n\n        >>> c = p.p1 + (r - p.p1).unit\n        >>> c.distance(p.p1).equals(1)\n        True\n        '
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        params = {x: 2 * Rational(rng.gauss(0, 1)) - 1, y: 2 * Rational(rng.gauss(0, 1)) - 1}
        return self.arbitrary_point(x, y).subs(params)

    def parameter_value(self, other, u, v=None):
        if False:
            return 10
        "Return the parameter(s) corresponding to the given point.\n\n        Examples\n        ========\n\n        >>> from sympy import pi, Plane\n        >>> from sympy.abc import t, u, v\n        >>> p = Plane((2, 0, 0), (0, 0, 1), (0, 1, 0))\n\n        By default, the parameter value returned defines a point\n        that is a distance of 1 from the Plane's p1 value and\n        in line with the given point:\n\n        >>> on_circle = p.arbitrary_point(t).subs(t, pi/4)\n        >>> on_circle.distance(p.p1)\n        1\n        >>> p.parameter_value(on_circle, t)\n        {t: pi/4}\n\n        Moving the point twice as far from p1 does not change\n        the parameter value:\n\n        >>> off_circle = p.p1 + (on_circle - p.p1)*2\n        >>> off_circle.distance(p.p1)\n        2\n        >>> p.parameter_value(off_circle, t)\n        {t: pi/4}\n\n        If the 2-value parameter is desired, supply the two\n        parameter symbols and a replacement dictionary will\n        be returned:\n\n        >>> p.parameter_value(on_circle, u, v)\n        {u: sqrt(10)/10, v: sqrt(10)/30}\n        >>> p.parameter_value(off_circle, u, v)\n        {u: sqrt(10)/5, v: sqrt(10)/15}\n        "
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if not isinstance(other, Point):
            raise ValueError('other must be a point')
        if other == self.p1:
            return other
        if isinstance(u, Symbol) and v is None:
            delta = self.arbitrary_point(u) - self.p1
            eq = delta - (other - self.p1).unit
            sol = solve(eq, u, dict=True)
        elif isinstance(u, Symbol) and isinstance(v, Symbol):
            pt = self.arbitrary_point(u, v)
            sol = solve(pt - other, (u, v), dict=True)
        else:
            raise ValueError('expecting 1 or 2 symbols')
        if not sol:
            raise ValueError('Given point is not on %s' % func_name(self))
        return sol[0]

    @property
    def ambient_dimension(self):
        if False:
            print('Hello World!')
        return self.p1.ambient_dimension