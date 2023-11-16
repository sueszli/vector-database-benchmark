"""Line-like geometrical entities.

Contains
========
LinearEntity
Line
Ray
Segment
LinearEntity2D
Line2D
Ray2D
Segment2D
LinearEntity3D
Line3D
Ray3D
Segment3D

"""
from sympy.core.containers import Tuple
from sympy.core.evalf import N
from sympy.core.expr import Expr
from sympy.core.numbers import Rational, oo, Float
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import _symbol, Dummy, uniquely_named_symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import _pi_coeff, acos, tan, atan2
from .entity import GeometryEntity, GeometrySet
from .exceptions import GeometryError
from .point import Point, Point3D
from .util import find, intersection
from sympy.logic.boolalg import And
from sympy.matrices import Matrix
from sympy.sets.sets import Intersection
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.solvers.solveset import linear_coeffs
from sympy.utilities.misc import Undecidable, filldedent
import random
(t, u) = [Dummy('line_dummy') for i in range(2)]

class LinearEntity(GeometrySet):
    """A base class for all linear entities (Line, Ray and Segment)
    in n-dimensional Euclidean space.

    Attributes
    ==========

    ambient_dimension
    direction
    length
    p1
    p2
    points

    Notes
    =====

    This is an abstract class and is not meant to be instantiated.

    See Also
    ========

    sympy.geometry.entity.GeometryEntity

    """

    def __new__(cls, p1, p2=None, **kwargs):
        if False:
            return 10
        (p1, p2) = Point._normalize_dimension(p1, p2)
        if p1 == p2:
            raise ValueError('%s.__new__ requires two unique Points.' % cls.__name__)
        if len(p1) != len(p2):
            raise ValueError('%s.__new__ requires two Points of equal dimension.' % cls.__name__)
        return GeometryEntity.__new__(cls, p1, p2, **kwargs)

    def __contains__(self, other):
        if False:
            while True:
                i = 10
        'Return a definitive answer or else raise an error if it cannot\n        be determined that other is on the boundaries of self.'
        result = self.contains(other)
        if result is not None:
            return result
        else:
            raise Undecidable("Cannot decide whether '%s' contains '%s'" % (self, other))

    def _span_test(self, other):
        if False:
            i = 10
            return i + 15
        "Test whether the point `other` lies in the positive span of `self`.\n        A point x is 'in front' of a point y if x.dot(y) >= 0.  Return\n        -1 if `other` is behind `self.p1`, 0 if `other` is `self.p1` and\n        and 1 if `other` is in front of `self.p1`."
        if self.p1 == other:
            return 0
        rel_pos = other - self.p1
        d = self.direction
        if d.dot(rel_pos) > 0:
            return 1
        return -1

    @property
    def ambient_dimension(self):
        if False:
            return 10
        'A property method that returns the dimension of LinearEntity\n        object.\n\n        Parameters\n        ==========\n\n        p1 : LinearEntity\n\n        Returns\n        =======\n\n        dimension : integer\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Line\n        >>> p1, p2 = Point(0, 0), Point(1, 1)\n        >>> l1 = Line(p1, p2)\n        >>> l1.ambient_dimension\n        2\n\n        >>> from sympy import Point, Line\n        >>> p1, p2 = Point(0, 0, 0), Point(1, 1, 1)\n        >>> l1 = Line(p1, p2)\n        >>> l1.ambient_dimension\n        3\n\n        '
        return len(self.p1)

    def angle_between(l1, l2):
        if False:
            return 10
        'Return the non-reflex angle formed by rays emanating from\n        the origin with directions the same as the direction vectors\n        of the linear entities.\n\n        Parameters\n        ==========\n\n        l1 : LinearEntity\n        l2 : LinearEntity\n\n        Returns\n        =======\n\n        angle : angle in radians\n\n        Notes\n        =====\n\n        From the dot product of vectors v1 and v2 it is known that:\n\n            ``dot(v1, v2) = |v1|*|v2|*cos(A)``\n\n        where A is the angle formed between the two vectors. We can\n        get the directional vectors of the two lines and readily\n        find the angle between the two using the above formula.\n\n        See Also\n        ========\n\n        is_perpendicular, Ray2D.closing_angle\n\n        Examples\n        ========\n\n        >>> from sympy import Line\n        >>> e = Line((0, 0), (1, 0))\n        >>> ne = Line((0, 0), (1, 1))\n        >>> sw = Line((1, 1), (0, 0))\n        >>> ne.angle_between(e)\n        pi/4\n        >>> sw.angle_between(e)\n        3*pi/4\n\n        To obtain the non-obtuse angle at the intersection of lines, use\n        the ``smallest_angle_between`` method:\n\n        >>> sw.smallest_angle_between(e)\n        pi/4\n\n        >>> from sympy import Point3D, Line3D\n        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(-1, 2, 0)\n        >>> l1, l2 = Line3D(p1, p2), Line3D(p2, p3)\n        >>> l1.angle_between(l2)\n        acos(-sqrt(2)/3)\n        >>> l1.smallest_angle_between(l2)\n        acos(sqrt(2)/3)\n        '
        if not isinstance(l1, LinearEntity) and (not isinstance(l2, LinearEntity)):
            raise TypeError('Must pass only LinearEntity objects')
        (v1, v2) = (l1.direction, l2.direction)
        return acos(v1.dot(v2) / (abs(v1) * abs(v2)))

    def smallest_angle_between(l1, l2):
        if False:
            for i in range(10):
                print('nop')
        'Return the smallest angle formed at the intersection of the\n        lines containing the linear entities.\n\n        Parameters\n        ==========\n\n        l1 : LinearEntity\n        l2 : LinearEntity\n\n        Returns\n        =======\n\n        angle : angle in radians\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Line\n        >>> p1, p2, p3 = Point(0, 0), Point(0, 4), Point(2, -2)\n        >>> l1, l2 = Line(p1, p2), Line(p1, p3)\n        >>> l1.smallest_angle_between(l2)\n        pi/4\n\n        See Also\n        ========\n\n        angle_between, is_perpendicular, Ray2D.closing_angle\n        '
        if not isinstance(l1, LinearEntity) and (not isinstance(l2, LinearEntity)):
            raise TypeError('Must pass only LinearEntity objects')
        (v1, v2) = (l1.direction, l2.direction)
        return acos(abs(v1.dot(v2)) / (abs(v1) * abs(v2)))

    def arbitrary_point(self, parameter='t'):
        if False:
            print('Hello World!')
        "A parameterized point on the Line.\n\n        Parameters\n        ==========\n\n        parameter : str, optional\n            The name of the parameter which will be used for the parametric\n            point. The default value is 't'. When this parameter is 0, the\n            first point used to define the line will be returned, and when\n            it is 1 the second point will be returned.\n\n        Returns\n        =======\n\n        point : Point\n\n        Raises\n        ======\n\n        ValueError\n            When ``parameter`` already appears in the Line's definition.\n\n        See Also\n        ========\n\n        sympy.geometry.point.Point\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Line\n        >>> p1, p2 = Point(1, 0), Point(5, 3)\n        >>> l1 = Line(p1, p2)\n        >>> l1.arbitrary_point()\n        Point2D(4*t + 1, 3*t)\n        >>> from sympy import Point3D, Line3D\n        >>> p1, p2 = Point3D(1, 0, 0), Point3D(5, 3, 1)\n        >>> l1 = Line3D(p1, p2)\n        >>> l1.arbitrary_point()\n        Point3D(4*t + 1, 3*t, t)\n\n        "
        t = _symbol(parameter, real=True)
        if t.name in (f.name for f in self.free_symbols):
            raise ValueError(filldedent('\n                Symbol %s already appears in object\n                and cannot be used as a parameter.\n                ' % t.name))
        return self.p1 + (self.p2 - self.p1) * t

    @staticmethod
    def are_concurrent(*lines):
        if False:
            while True:
                i = 10
        'Is a sequence of linear entities concurrent?\n\n        Two or more linear entities are concurrent if they all\n        intersect at a single point.\n\n        Parameters\n        ==========\n\n        lines\n            A sequence of linear entities.\n\n        Returns\n        =======\n\n        True : if the set of linear entities intersect in one point\n        False : otherwise.\n\n        See Also\n        ========\n\n        sympy.geometry.util.intersection\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Line\n        >>> p1, p2 = Point(0, 0), Point(3, 5)\n        >>> p3, p4 = Point(-2, -2), Point(0, 2)\n        >>> l1, l2, l3 = Line(p1, p2), Line(p1, p3), Line(p1, p4)\n        >>> Line.are_concurrent(l1, l2, l3)\n        True\n        >>> l4 = Line(p2, p3)\n        >>> Line.are_concurrent(l2, l3, l4)\n        False\n        >>> from sympy import Point3D, Line3D\n        >>> p1, p2 = Point3D(0, 0, 0), Point3D(3, 5, 2)\n        >>> p3, p4 = Point3D(-2, -2, -2), Point3D(0, 2, 1)\n        >>> l1, l2, l3 = Line3D(p1, p2), Line3D(p1, p3), Line3D(p1, p4)\n        >>> Line3D.are_concurrent(l1, l2, l3)\n        True\n        >>> l4 = Line3D(p2, p3)\n        >>> Line3D.are_concurrent(l2, l3, l4)\n        False\n\n        '
        common_points = Intersection(*lines)
        if common_points.is_FiniteSet and len(common_points) == 1:
            return True
        return False

    def contains(self, other):
        if False:
            return 10
        'Subclasses should implement this method and should return\n            True if other is on the boundaries of self;\n            False if not on the boundaries of self;\n            None if a determination cannot be made.'
        raise NotImplementedError()

    @property
    def direction(self):
        if False:
            i = 10
            return i + 15
        'The direction vector of the LinearEntity.\n\n        Returns\n        =======\n\n        p : a Point; the ray from the origin to this point is the\n            direction of `self`\n\n        Examples\n        ========\n\n        >>> from sympy import Line\n        >>> a, b = (1, 1), (1, 3)\n        >>> Line(a, b).direction\n        Point2D(0, 2)\n        >>> Line(b, a).direction\n        Point2D(0, -2)\n\n        This can be reported so the distance from the origin is 1:\n\n        >>> Line(b, a).direction.unit\n        Point2D(0, -1)\n\n        See Also\n        ========\n\n        sympy.geometry.point.Point.unit\n\n        '
        return self.p2 - self.p1

    def intersection(self, other):
        if False:
            return 10
        'The intersection with another geometrical entity.\n\n        Parameters\n        ==========\n\n        o : Point or LinearEntity\n\n        Returns\n        =======\n\n        intersection : list of geometrical entities\n\n        See Also\n        ========\n\n        sympy.geometry.point.Point\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Line, Segment\n        >>> p1, p2, p3 = Point(0, 0), Point(1, 1), Point(7, 7)\n        >>> l1 = Line(p1, p2)\n        >>> l1.intersection(p3)\n        [Point2D(7, 7)]\n        >>> p4, p5 = Point(5, 0), Point(0, 3)\n        >>> l2 = Line(p4, p5)\n        >>> l1.intersection(l2)\n        [Point2D(15/8, 15/8)]\n        >>> p6, p7 = Point(0, 5), Point(2, 6)\n        >>> s1 = Segment(p6, p7)\n        >>> l1.intersection(s1)\n        []\n        >>> from sympy import Point3D, Line3D, Segment3D\n        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(7, 7, 7)\n        >>> l1 = Line3D(p1, p2)\n        >>> l1.intersection(p3)\n        [Point3D(7, 7, 7)]\n        >>> l1 = Line3D(Point3D(4,19,12), Point3D(5,25,17))\n        >>> l2 = Line3D(Point3D(-3, -15, -19), direction_ratio=[2,8,8])\n        >>> l1.intersection(l2)\n        [Point3D(1, 1, -3)]\n        >>> p6, p7 = Point3D(0, 5, 2), Point3D(2, 6, 3)\n        >>> s1 = Segment3D(p6, p7)\n        >>> l1.intersection(s1)\n        []\n\n        '

        def intersect_parallel_rays(ray1, ray2):
            if False:
                while True:
                    i = 10
            if ray1.direction.dot(ray2.direction) > 0:
                return [ray2] if ray1._span_test(ray2.p1) >= 0 else [ray1]
            else:
                st = ray1._span_test(ray2.p1)
                if st < 0:
                    return []
                elif st == 0:
                    return [ray2.p1]
                return [Segment(ray1.p1, ray2.p1)]

        def intersect_parallel_ray_and_segment(ray, seg):
            if False:
                i = 10
                return i + 15
            (st1, st2) = (ray._span_test(seg.p1), ray._span_test(seg.p2))
            if st1 < 0 and st2 < 0:
                return []
            elif st1 >= 0 and st2 >= 0:
                return [seg]
            elif st1 >= 0:
                return [Segment(ray.p1, seg.p1)]
            else:
                return [Segment(ray.p1, seg.p2)]

        def intersect_parallel_segments(seg1, seg2):
            if False:
                for i in range(10):
                    print('nop')
            if seg1.contains(seg2):
                return [seg2]
            if seg2.contains(seg1):
                return [seg1]
            if seg1.direction.dot(seg2.direction) < 0:
                seg2 = Segment(seg2.p2, seg2.p1)
            if seg1._span_test(seg2.p1) < 0:
                (seg1, seg2) = (seg2, seg1)
            if seg2._span_test(seg1.p2) < 0:
                return []
            return [Segment(seg2.p1, seg1.p2)]
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if other.is_Point:
            if self.contains(other):
                return [other]
            else:
                return []
        elif isinstance(other, LinearEntity):
            pts = Point._normalize_dimension(self.p1, self.p2, other.p1, other.p2)
            rank = Point.affine_rank(*pts)
            if rank == 1:
                if isinstance(self, Line):
                    return [other]
                if isinstance(other, Line):
                    return [self]
                if isinstance(self, Ray) and isinstance(other, Ray):
                    return intersect_parallel_rays(self, other)
                if isinstance(self, Ray) and isinstance(other, Segment):
                    return intersect_parallel_ray_and_segment(self, other)
                if isinstance(self, Segment) and isinstance(other, Ray):
                    return intersect_parallel_ray_and_segment(other, self)
                if isinstance(self, Segment) and isinstance(other, Segment):
                    return intersect_parallel_segments(self, other)
            elif rank == 2:
                l1 = Line(*pts[:2])
                l2 = Line(*pts[2:])
                if l1.direction.is_scalar_multiple(l2.direction):
                    return []
                m = Matrix([l1.direction, -l2.direction]).transpose()
                v = Matrix([l2.p1 - l1.p1]).transpose()
                (m_rref, pivots) = m.col_insert(2, v).rref(simplify=True)
                if len(pivots) != 2:
                    raise GeometryError('Failed when solving Mx=b when M={} and b={}'.format(m, v))
                coeff = m_rref[0, 2]
                line_intersection = l1.direction * coeff + self.p1
                if isinstance(self, Line) and isinstance(other, Line):
                    return [line_intersection]
                if (isinstance(self, Line) or self.contains(line_intersection)) and other.contains(line_intersection):
                    return [line_intersection]
                if not self.atoms(Float) and (not other.atoms(Float)):
                    return []
                tu = solve(self.arbitrary_point(t) - other.arbitrary_point(u), t, u, dict=True)[0]

                def ok(p, l):
                    if False:
                        return 10
                    if isinstance(l, Line):
                        return True
                    if isinstance(l, Ray):
                        return p.is_nonnegative
                    if isinstance(l, Segment):
                        return p.is_nonnegative and (1 - p).is_nonnegative
                    raise ValueError('unexpected line type')
                if ok(tu[t], self) and ok(tu[u], other):
                    return [line_intersection]
                return []
            else:
                return []
        return other.intersection(self)

    def is_parallel(l1, l2):
        if False:
            i = 10
            return i + 15
        'Are two linear entities parallel?\n\n        Parameters\n        ==========\n\n        l1 : LinearEntity\n        l2 : LinearEntity\n\n        Returns\n        =======\n\n        True : if l1 and l2 are parallel,\n        False : otherwise.\n\n        See Also\n        ========\n\n        coefficients\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Line\n        >>> p1, p2 = Point(0, 0), Point(1, 1)\n        >>> p3, p4 = Point(3, 4), Point(6, 7)\n        >>> l1, l2 = Line(p1, p2), Line(p3, p4)\n        >>> Line.is_parallel(l1, l2)\n        True\n        >>> p5 = Point(6, 6)\n        >>> l3 = Line(p3, p5)\n        >>> Line.is_parallel(l1, l3)\n        False\n        >>> from sympy import Point3D, Line3D\n        >>> p1, p2 = Point3D(0, 0, 0), Point3D(3, 4, 5)\n        >>> p3, p4 = Point3D(2, 1, 1), Point3D(8, 9, 11)\n        >>> l1, l2 = Line3D(p1, p2), Line3D(p3, p4)\n        >>> Line3D.is_parallel(l1, l2)\n        True\n        >>> p5 = Point3D(6, 6, 6)\n        >>> l3 = Line3D(p3, p5)\n        >>> Line3D.is_parallel(l1, l3)\n        False\n\n        '
        if not isinstance(l1, LinearEntity) and (not isinstance(l2, LinearEntity)):
            raise TypeError('Must pass only LinearEntity objects')
        return l1.direction.is_scalar_multiple(l2.direction)

    def is_perpendicular(l1, l2):
        if False:
            while True:
                i = 10
        'Are two linear entities perpendicular?\n\n        Parameters\n        ==========\n\n        l1 : LinearEntity\n        l2 : LinearEntity\n\n        Returns\n        =======\n\n        True : if l1 and l2 are perpendicular,\n        False : otherwise.\n\n        See Also\n        ========\n\n        coefficients\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Line\n        >>> p1, p2, p3 = Point(0, 0), Point(1, 1), Point(-1, 1)\n        >>> l1, l2 = Line(p1, p2), Line(p1, p3)\n        >>> l1.is_perpendicular(l2)\n        True\n        >>> p4 = Point(5, 3)\n        >>> l3 = Line(p1, p4)\n        >>> l1.is_perpendicular(l3)\n        False\n        >>> from sympy import Point3D, Line3D\n        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(-1, 2, 0)\n        >>> l1, l2 = Line3D(p1, p2), Line3D(p2, p3)\n        >>> l1.is_perpendicular(l2)\n        False\n        >>> p4 = Point3D(5, 3, 7)\n        >>> l3 = Line3D(p1, p4)\n        >>> l1.is_perpendicular(l3)\n        False\n\n        '
        if not isinstance(l1, LinearEntity) and (not isinstance(l2, LinearEntity)):
            raise TypeError('Must pass only LinearEntity objects')
        return S.Zero.equals(l1.direction.dot(l2.direction))

    def is_similar(self, other):
        if False:
            print('Hello World!')
        '\n        Return True if self and other are contained in the same line.\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Line\n        >>> p1, p2, p3 = Point(0, 1), Point(3, 4), Point(2, 3)\n        >>> l1 = Line(p1, p2)\n        >>> l2 = Line(p1, p3)\n        >>> l1.is_similar(l2)\n        True\n        '
        l = Line(self.p1, self.p2)
        return l.contains(other)

    @property
    def length(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The length of the line.\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Line\n        >>> p1, p2 = Point(0, 0), Point(3, 5)\n        >>> l1 = Line(p1, p2)\n        >>> l1.length\n        oo\n        '
        return S.Infinity

    @property
    def p1(self):
        if False:
            while True:
                i = 10
        'The first defining point of a linear entity.\n\n        See Also\n        ========\n\n        sympy.geometry.point.Point\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Line\n        >>> p1, p2 = Point(0, 0), Point(5, 3)\n        >>> l = Line(p1, p2)\n        >>> l.p1\n        Point2D(0, 0)\n\n        '
        return self.args[0]

    @property
    def p2(self):
        if False:
            return 10
        'The second defining point of a linear entity.\n\n        See Also\n        ========\n\n        sympy.geometry.point.Point\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Line\n        >>> p1, p2 = Point(0, 0), Point(5, 3)\n        >>> l = Line(p1, p2)\n        >>> l.p2\n        Point2D(5, 3)\n\n        '
        return self.args[1]

    def parallel_line(self, p):
        if False:
            return 10
        'Create a new Line parallel to this linear entity which passes\n        through the point `p`.\n\n        Parameters\n        ==========\n\n        p : Point\n\n        Returns\n        =======\n\n        line : Line\n\n        See Also\n        ========\n\n        is_parallel\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Line\n        >>> p1, p2, p3 = Point(0, 0), Point(2, 3), Point(-2, 2)\n        >>> l1 = Line(p1, p2)\n        >>> l2 = l1.parallel_line(p3)\n        >>> p3 in l2\n        True\n        >>> l1.is_parallel(l2)\n        True\n        >>> from sympy import Point3D, Line3D\n        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(2, 3, 4), Point3D(-2, 2, 0)\n        >>> l1 = Line3D(p1, p2)\n        >>> l2 = l1.parallel_line(p3)\n        >>> p3 in l2\n        True\n        >>> l1.is_parallel(l2)\n        True\n\n        '
        p = Point(p, dim=self.ambient_dimension)
        return Line(p, p + self.direction)

    def perpendicular_line(self, p):
        if False:
            return 10
        'Create a new Line perpendicular to this linear entity which passes\n        through the point `p`.\n\n        Parameters\n        ==========\n\n        p : Point\n\n        Returns\n        =======\n\n        line : Line\n\n        See Also\n        ========\n\n        sympy.geometry.line.LinearEntity.is_perpendicular, perpendicular_segment\n\n        Examples\n        ========\n\n        >>> from sympy import Point3D, Line3D\n        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(2, 3, 4), Point3D(-2, 2, 0)\n        >>> L = Line3D(p1, p2)\n        >>> P = L.perpendicular_line(p3); P\n        Line3D(Point3D(-2, 2, 0), Point3D(4/29, 6/29, 8/29))\n        >>> L.is_perpendicular(P)\n        True\n\n        In 3D the, the first point used to define the line is the point\n        through which the perpendicular was required to pass; the\n        second point is (arbitrarily) contained in the given line:\n\n        >>> P.p2 in L\n        True\n        '
        p = Point(p, dim=self.ambient_dimension)
        if p in self:
            p = p + self.direction.orthogonal_direction
        return Line(p, self.projection(p))

    def perpendicular_segment(self, p):
        if False:
            while True:
                i = 10
        'Create a perpendicular line segment from `p` to this line.\n\n        The endpoints of the segment are ``p`` and the closest point in\n        the line containing self. (If self is not a line, the point might\n        not be in self.)\n\n        Parameters\n        ==========\n\n        p : Point\n\n        Returns\n        =======\n\n        segment : Segment\n\n        Notes\n        =====\n\n        Returns `p` itself if `p` is on this linear entity.\n\n        See Also\n        ========\n\n        perpendicular_line\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Line\n        >>> p1, p2, p3 = Point(0, 0), Point(1, 1), Point(0, 2)\n        >>> l1 = Line(p1, p2)\n        >>> s1 = l1.perpendicular_segment(p3)\n        >>> l1.is_perpendicular(s1)\n        True\n        >>> p3 in s1\n        True\n        >>> l1.perpendicular_segment(Point(4, 0))\n        Segment2D(Point2D(4, 0), Point2D(2, 2))\n        >>> from sympy import Point3D, Line3D\n        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(0, 2, 0)\n        >>> l1 = Line3D(p1, p2)\n        >>> s1 = l1.perpendicular_segment(p3)\n        >>> l1.is_perpendicular(s1)\n        True\n        >>> p3 in s1\n        True\n        >>> l1.perpendicular_segment(Point3D(4, 0, 0))\n        Segment3D(Point3D(4, 0, 0), Point3D(4/3, 4/3, 4/3))\n\n        '
        p = Point(p, dim=self.ambient_dimension)
        if p in self:
            return p
        l = self.perpendicular_line(p)
        (p2,) = Intersection(Line(self.p1, self.p2), l)
        return Segment(p, p2)

    @property
    def points(self):
        if False:
            for i in range(10):
                print('nop')
        'The two points used to define this linear entity.\n\n        Returns\n        =======\n\n        points : tuple of Points\n\n        See Also\n        ========\n\n        sympy.geometry.point.Point\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Line\n        >>> p1, p2 = Point(0, 0), Point(5, 11)\n        >>> l1 = Line(p1, p2)\n        >>> l1.points\n        (Point2D(0, 0), Point2D(5, 11))\n\n        '
        return (self.p1, self.p2)

    def projection(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Project a point, line, ray, or segment onto this linear entity.\n\n        Parameters\n        ==========\n\n        other : Point or LinearEntity (Line, Ray, Segment)\n\n        Returns\n        =======\n\n        projection : Point or LinearEntity (Line, Ray, Segment)\n            The return type matches the type of the parameter ``other``.\n\n        Raises\n        ======\n\n        GeometryError\n            When method is unable to perform projection.\n\n        Notes\n        =====\n\n        A projection involves taking the two points that define\n        the linear entity and projecting those points onto a\n        Line and then reforming the linear entity using these\n        projections.\n        A point P is projected onto a line L by finding the point\n        on L that is closest to P. This point is the intersection\n        of L and the line perpendicular to L that passes through P.\n\n        See Also\n        ========\n\n        sympy.geometry.point.Point, perpendicular_line\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Line, Segment, Rational\n        >>> p1, p2, p3 = Point(0, 0), Point(1, 1), Point(Rational(1, 2), 0)\n        >>> l1 = Line(p1, p2)\n        >>> l1.projection(p3)\n        Point2D(1/4, 1/4)\n        >>> p4, p5 = Point(10, 0), Point(12, 1)\n        >>> s1 = Segment(p4, p5)\n        >>> l1.projection(s1)\n        Segment2D(Point2D(5, 5), Point2D(13/2, 13/2))\n        >>> p1, p2, p3 = Point(0, 0, 1), Point(1, 1, 2), Point(2, 0, 1)\n        >>> l1 = Line(p1, p2)\n        >>> l1.projection(p3)\n        Point3D(2/3, 2/3, 5/3)\n        >>> p4, p5 = Point(10, 0, 1), Point(12, 1, 3)\n        >>> s1 = Segment(p4, p5)\n        >>> l1.projection(s1)\n        Segment3D(Point3D(10/3, 10/3, 13/3), Point3D(5, 5, 6))\n\n        '
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)

        def proj_point(p):
            if False:
                for i in range(10):
                    print('nop')
            return Point.project(p - self.p1, self.direction) + self.p1
        if isinstance(other, Point):
            return proj_point(other)
        elif isinstance(other, LinearEntity):
            (p1, p2) = (proj_point(other.p1), proj_point(other.p2))
            if p1 == p2:
                return p1
            projected = other.__class__(p1, p2)
            projected = Intersection(self, projected)
            if projected.is_empty:
                return projected
            if projected.is_FiniteSet and len(projected) == 1:
                (a,) = projected
                return a
            if self.direction.dot(projected.direction) < 0:
                (p1, p2) = projected.args
                projected = projected.func(p2, p1)
            return projected
        raise GeometryError('Do not know how to project %s onto %s' % (other, self))

    def random_point(self, seed=None):
        if False:
            print('Hello World!')
        'A random point on a LinearEntity.\n\n        Returns\n        =======\n\n        point : Point\n\n        See Also\n        ========\n\n        sympy.geometry.point.Point\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Line, Ray, Segment\n        >>> p1, p2 = Point(0, 0), Point(5, 3)\n        >>> line = Line(p1, p2)\n        >>> r = line.random_point(seed=42)  # seed value is optional\n        >>> r.n(3)\n        Point2D(-0.72, -0.432)\n        >>> r in line\n        True\n        >>> Ray(p1, p2).random_point(seed=42).n(3)\n        Point2D(0.72, 0.432)\n        >>> Segment(p1, p2).random_point(seed=42).n(3)\n        Point2D(3.2, 1.92)\n\n        '
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        pt = self.arbitrary_point(t)
        if isinstance(self, Ray):
            v = abs(rng.gauss(0, 1))
        elif isinstance(self, Segment):
            v = rng.random()
        elif isinstance(self, Line):
            v = rng.gauss(0, 1)
        else:
            raise NotImplementedError('unhandled line type')
        return pt.subs(t, Rational(v))

    def bisectors(self, other):
        if False:
            print('Hello World!')
        'Returns the perpendicular lines which pass through the intersections\n        of self and other that are in the same plane.\n\n        Parameters\n        ==========\n\n        line : Line3D\n\n        Returns\n        =======\n\n        list: two Line instances\n\n        Examples\n        ========\n\n        >>> from sympy import Point3D, Line3D\n        >>> r1 = Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0))\n        >>> r2 = Line3D(Point3D(0, 0, 0), Point3D(0, 1, 0))\n        >>> r1.bisectors(r2)\n        [Line3D(Point3D(0, 0, 0), Point3D(1, 1, 0)), Line3D(Point3D(0, 0, 0), Point3D(1, -1, 0))]\n\n        '
        if not isinstance(other, LinearEntity):
            raise GeometryError('Expecting LinearEntity, not %s' % other)
        (l1, l2) = (self, other)
        if l1.p1.ambient_dimension != l2.p1.ambient_dimension:
            if isinstance(l1, Line2D):
                (l1, l2) = (l2, l1)
            (_, p1) = Point._normalize_dimension(l1.p1, l2.p1, on_morph='ignore')
            (_, p2) = Point._normalize_dimension(l1.p2, l2.p2, on_morph='ignore')
            l2 = Line(p1, p2)
        point = intersection(l1, l2)
        if not point:
            raise GeometryError('The lines do not intersect')
        else:
            pt = point[0]
            if isinstance(pt, Line):
                return [self]
        d1 = l1.direction.unit
        d2 = l2.direction.unit
        bis1 = Line(pt, pt + d1 + d2)
        bis2 = Line(pt, pt + d1 - d2)
        return [bis1, bis2]

class Line(LinearEntity):
    """An infinite line in space.

    A 2D line is declared with two distinct points, point and slope, or
    an equation. A 3D line may be defined with a point and a direction ratio.

    Parameters
    ==========

    p1 : Point
    p2 : Point
    slope : SymPy expression
    direction_ratio : list
    equation : equation of a line

    Notes
    =====

    `Line` will automatically subclass to `Line2D` or `Line3D` based
    on the dimension of `p1`.  The `slope` argument is only relevant
    for `Line2D` and the `direction_ratio` argument is only relevant
    for `Line3D`.

    The order of the points will define the direction of the line
    which is used when calculating the angle between lines.

    See Also
    ========

    sympy.geometry.point.Point
    sympy.geometry.line.Line2D
    sympy.geometry.line.Line3D

    Examples
    ========

    >>> from sympy import Line, Segment, Point, Eq
    >>> from sympy.abc import x, y, a, b

    >>> L = Line(Point(2,3), Point(3,5))
    >>> L
    Line2D(Point2D(2, 3), Point2D(3, 5))
    >>> L.points
    (Point2D(2, 3), Point2D(3, 5))
    >>> L.equation()
    -2*x + y + 1
    >>> L.coefficients
    (-2, 1, 1)

    Instantiate with keyword ``slope``:

    >>> Line(Point(0, 0), slope=0)
    Line2D(Point2D(0, 0), Point2D(1, 0))

    Instantiate with another linear object

    >>> s = Segment((0, 0), (0, 1))
    >>> Line(s).equation()
    x

    The line corresponding to an equation in the for `ax + by + c = 0`,
    can be entered:

    >>> Line(3*x + y + 18)
    Line2D(Point2D(0, -18), Point2D(1, -21))

    If `x` or `y` has a different name, then they can be specified, too,
    as a string (to match the name) or symbol:

    >>> Line(Eq(3*a + b, -18), x='a', y=b)
    Line2D(Point2D(0, -18), Point2D(1, -21))
    """

    def __new__(cls, *args, **kwargs):
        if False:
            while True:
                i = 10
        if len(args) == 1 and isinstance(args[0], (Expr, Eq)):
            missing = uniquely_named_symbol('?', args)
            if not kwargs:
                x = 'x'
                y = 'y'
            else:
                x = kwargs.pop('x', missing)
                y = kwargs.pop('y', missing)
            if kwargs:
                raise ValueError('expecting only x and y as keywords')
            equation = args[0]
            if isinstance(equation, Eq):
                equation = equation.lhs - equation.rhs

            def find_or_missing(x):
                if False:
                    print('Hello World!')
                try:
                    return find(x, equation)
                except ValueError:
                    return missing
            x = find_or_missing(x)
            y = find_or_missing(y)
            (a, b, c) = linear_coeffs(equation, x, y)
            if b:
                return Line((0, -c / b), slope=-a / b)
            if a:
                return Line((-c / a, 0), slope=oo)
            raise ValueError('not found in equation: %s' % (set('xy') - {x, y}))
        elif len(args) > 0:
            p1 = args[0]
            if len(args) > 1:
                p2 = args[1]
            else:
                p2 = None
            if isinstance(p1, LinearEntity):
                if p2:
                    raise ValueError('If p1 is a LinearEntity, p2 must be None.')
                dim = len(p1.p1)
            else:
                p1 = Point(p1)
                dim = len(p1)
                if p2 is not None or (isinstance(p2, Point) and p2.ambient_dimension != dim):
                    p2 = Point(p2)
            if dim == 2:
                return Line2D(p1, p2, **kwargs)
            elif dim == 3:
                return Line3D(p1, p2, **kwargs)
            return LinearEntity.__new__(cls, p1, p2, **kwargs)

    def contains(self, other):
        if False:
            while True:
                i = 10
        '\n        Return True if `other` is on this Line, or False otherwise.\n\n        Examples\n        ========\n\n        >>> from sympy import Line,Point\n        >>> p1, p2 = Point(0, 1), Point(3, 4)\n        >>> l = Line(p1, p2)\n        >>> l.contains(p1)\n        True\n        >>> l.contains((0, 1))\n        True\n        >>> l.contains((0, 0))\n        False\n        >>> a = (0, 0, 0)\n        >>> b = (1, 1, 1)\n        >>> c = (2, 2, 2)\n        >>> l1 = Line(a, b)\n        >>> l2 = Line(b, a)\n        >>> l1 == l2\n        False\n        >>> l1 in l2\n        True\n\n        '
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if isinstance(other, Point):
            return Point.is_collinear(other, self.p1, self.p2)
        if isinstance(other, LinearEntity):
            return Point.is_collinear(self.p1, self.p2, other.p1, other.p2)
        return False

    def distance(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Finds the shortest distance between a line and a point.\n\n        Raises\n        ======\n\n        NotImplementedError is raised if `other` is not a Point\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Line\n        >>> p1, p2 = Point(0, 0), Point(1, 1)\n        >>> s = Line(p1, p2)\n        >>> s.distance(Point(-1, 1))\n        sqrt(2)\n        >>> s.distance((-1, 2))\n        3*sqrt(2)/2\n        >>> p1, p2 = Point(0, 0, 0), Point(1, 1, 1)\n        >>> s = Line(p1, p2)\n        >>> s.distance(Point(-1, 1, 1))\n        2*sqrt(6)/3\n        >>> s.distance((-1, 1, 1))\n        2*sqrt(6)/3\n\n        '
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if self.contains(other):
            return S.Zero
        return self.perpendicular_segment(other).length

    def equals(self, other):
        if False:
            return 10
        'Returns True if self and other are the same mathematical entities'
        if not isinstance(other, Line):
            return False
        return Point.is_collinear(self.p1, other.p1, self.p2, other.p2)

    def plot_interval(self, parameter='t'):
        if False:
            i = 10
            return i + 15
        "The plot interval for the default geometric plot of line. Gives\n        values that will produce a line that is +/- 5 units long (where a\n        unit is the distance between the two points that define the line).\n\n        Parameters\n        ==========\n\n        parameter : str, optional\n            Default value is 't'.\n\n        Returns\n        =======\n\n        plot_interval : list (plot interval)\n            [parameter, lower_bound, upper_bound]\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Line\n        >>> p1, p2 = Point(0, 0), Point(5, 3)\n        >>> l1 = Line(p1, p2)\n        >>> l1.plot_interval()\n        [t, -5, 5]\n\n        "
        t = _symbol(parameter, real=True)
        return [t, -5, 5]

class Ray(LinearEntity):
    """A Ray is a semi-line in the space with a source point and a direction.

    Parameters
    ==========

    p1 : Point
        The source of the Ray
    p2 : Point or radian value
        This point determines the direction in which the Ray propagates.
        If given as an angle it is interpreted in radians with the positive
        direction being ccw.

    Attributes
    ==========

    source

    See Also
    ========

    sympy.geometry.line.Ray2D
    sympy.geometry.line.Ray3D
    sympy.geometry.point.Point
    sympy.geometry.line.Line

    Notes
    =====

    `Ray` will automatically subclass to `Ray2D` or `Ray3D` based on the
    dimension of `p1`.

    Examples
    ========

    >>> from sympy import Ray, Point, pi
    >>> r = Ray(Point(2, 3), Point(3, 5))
    >>> r
    Ray2D(Point2D(2, 3), Point2D(3, 5))
    >>> r.points
    (Point2D(2, 3), Point2D(3, 5))
    >>> r.source
    Point2D(2, 3)
    >>> r.xdirection
    oo
    >>> r.ydirection
    oo
    >>> r.slope
    2
    >>> Ray(Point(0, 0), angle=pi/4).slope
    1

    """

    def __new__(cls, p1, p2=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        p1 = Point(p1)
        if p2 is not None:
            (p1, p2) = Point._normalize_dimension(p1, Point(p2))
        dim = len(p1)
        if dim == 2:
            return Ray2D(p1, p2, **kwargs)
        elif dim == 3:
            return Ray3D(p1, p2, **kwargs)
        return LinearEntity.__new__(cls, p1, p2, **kwargs)

    def _svg(self, scale_factor=1.0, fill_color='#66cc99'):
        if False:
            print('Hello World!')
        'Returns SVG path element for the LinearEntity.\n\n        Parameters\n        ==========\n\n        scale_factor : float\n            Multiplication factor for the SVG stroke-width.  Default is 1.\n        fill_color : str, optional\n            Hex string for fill color. Default is "#66cc99".\n        '
        verts = (N(self.p1), N(self.p2))
        coords = ['{},{}'.format(p.x, p.y) for p in verts]
        path = 'M {} L {}'.format(coords[0], ' L '.join(coords[1:]))
        return '<path fill-rule="evenodd" fill="{2}" stroke="#555555" stroke-width="{0}" opacity="0.6" d="{1}" marker-start="url(#markerCircle)" marker-end="url(#markerArrow)"/>'.format(2.0 * scale_factor, path, fill_color)

    def contains(self, other):
        if False:
            return 10
        '\n        Is other GeometryEntity contained in this Ray?\n\n        Examples\n        ========\n\n        >>> from sympy import Ray,Point,Segment\n        >>> p1, p2 = Point(0, 0), Point(4, 4)\n        >>> r = Ray(p1, p2)\n        >>> r.contains(p1)\n        True\n        >>> r.contains((1, 1))\n        True\n        >>> r.contains((1, 3))\n        False\n        >>> s = Segment((1, 1), (2, 2))\n        >>> r.contains(s)\n        True\n        >>> s = Segment((1, 2), (2, 5))\n        >>> r.contains(s)\n        False\n        >>> r1 = Ray((2, 2), (3, 3))\n        >>> r.contains(r1)\n        True\n        >>> r1 = Ray((2, 2), (3, 5))\n        >>> r.contains(r1)\n        False\n        '
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if isinstance(other, Point):
            if Point.is_collinear(self.p1, self.p2, other):
                return bool((self.p2 - self.p1).dot(other - self.p1) >= S.Zero)
            return False
        elif isinstance(other, Ray):
            if Point.is_collinear(self.p1, self.p2, other.p1, other.p2):
                return bool((self.p2 - self.p1).dot(other.p2 - other.p1) > S.Zero)
            return False
        elif isinstance(other, Segment):
            return other.p1 in self and other.p2 in self
        return False

    def distance(self, other):
        if False:
            return 10
        '\n        Finds the shortest distance between the ray and a point.\n\n        Raises\n        ======\n\n        NotImplementedError is raised if `other` is not a Point\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ray\n        >>> p1, p2 = Point(0, 0), Point(1, 1)\n        >>> s = Ray(p1, p2)\n        >>> s.distance(Point(-1, -1))\n        sqrt(2)\n        >>> s.distance((-1, 2))\n        3*sqrt(2)/2\n        >>> p1, p2 = Point(0, 0, 0), Point(1, 1, 2)\n        >>> s = Ray(p1, p2)\n        >>> s\n        Ray3D(Point3D(0, 0, 0), Point3D(1, 1, 2))\n        >>> s.distance(Point(-1, -1, 2))\n        4*sqrt(3)/3\n        >>> s.distance((-1, -1, 2))\n        4*sqrt(3)/3\n\n        '
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if self.contains(other):
            return S.Zero
        proj = Line(self.p1, self.p2).projection(other)
        if self.contains(proj):
            return abs(other - proj)
        else:
            return abs(other - self.source)

    def equals(self, other):
        if False:
            i = 10
            return i + 15
        'Returns True if self and other are the same mathematical entities'
        if not isinstance(other, Ray):
            return False
        return self.source == other.source and other.p2 in self

    def plot_interval(self, parameter='t'):
        if False:
            while True:
                i = 10
        "The plot interval for the default geometric plot of the Ray. Gives\n        values that will produce a ray that is 10 units long (where a unit is\n        the distance between the two points that define the ray).\n\n        Parameters\n        ==========\n\n        parameter : str, optional\n            Default value is 't'.\n\n        Returns\n        =======\n\n        plot_interval : list\n            [parameter, lower_bound, upper_bound]\n\n        Examples\n        ========\n\n        >>> from sympy import Ray, pi\n        >>> r = Ray((0, 0), angle=pi/4)\n        >>> r.plot_interval()\n        [t, 0, 10]\n\n        "
        t = _symbol(parameter, real=True)
        return [t, 0, 10]

    @property
    def source(self):
        if False:
            while True:
                i = 10
        'The point from which the ray emanates.\n\n        See Also\n        ========\n\n        sympy.geometry.point.Point\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ray\n        >>> p1, p2 = Point(0, 0), Point(4, 1)\n        >>> r1 = Ray(p1, p2)\n        >>> r1.source\n        Point2D(0, 0)\n        >>> p1, p2 = Point(0, 0, 0), Point(4, 1, 5)\n        >>> r1 = Ray(p2, p1)\n        >>> r1.source\n        Point3D(4, 1, 5)\n\n        '
        return self.p1

class Segment(LinearEntity):
    """A line segment in space.

    Parameters
    ==========

    p1 : Point
    p2 : Point

    Attributes
    ==========

    length : number or SymPy expression
    midpoint : Point

    See Also
    ========

    sympy.geometry.line.Segment2D
    sympy.geometry.line.Segment3D
    sympy.geometry.point.Point
    sympy.geometry.line.Line

    Notes
    =====

    If 2D or 3D points are used to define `Segment`, it will
    be automatically subclassed to `Segment2D` or `Segment3D`.

    Examples
    ========

    >>> from sympy import Point, Segment
    >>> Segment((1, 0), (1, 1)) # tuples are interpreted as pts
    Segment2D(Point2D(1, 0), Point2D(1, 1))
    >>> s = Segment(Point(4, 3), Point(1, 1))
    >>> s.points
    (Point2D(4, 3), Point2D(1, 1))
    >>> s.slope
    2/3
    >>> s.length
    sqrt(13)
    >>> s.midpoint
    Point2D(5/2, 2)
    >>> Segment((1, 0, 0), (1, 1, 1)) # tuples are interpreted as pts
    Segment3D(Point3D(1, 0, 0), Point3D(1, 1, 1))
    >>> s = Segment(Point(4, 3, 9), Point(1, 1, 7)); s
    Segment3D(Point3D(4, 3, 9), Point3D(1, 1, 7))
    >>> s.points
    (Point3D(4, 3, 9), Point3D(1, 1, 7))
    >>> s.length
    sqrt(17)
    >>> s.midpoint
    Point3D(5/2, 2, 8)

    """

    def __new__(cls, p1, p2, **kwargs):
        if False:
            print('Hello World!')
        (p1, p2) = Point._normalize_dimension(Point(p1), Point(p2))
        dim = len(p1)
        if dim == 2:
            return Segment2D(p1, p2, **kwargs)
        elif dim == 3:
            return Segment3D(p1, p2, **kwargs)
        return LinearEntity.__new__(cls, p1, p2, **kwargs)

    def contains(self, other):
        if False:
            return 10
        '\n        Is the other GeometryEntity contained within this Segment?\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Segment\n        >>> p1, p2 = Point(0, 1), Point(3, 4)\n        >>> s = Segment(p1, p2)\n        >>> s2 = Segment(p2, p1)\n        >>> s.contains(s2)\n        True\n        >>> from sympy import Point3D, Segment3D\n        >>> p1, p2 = Point3D(0, 1, 1), Point3D(3, 4, 5)\n        >>> s = Segment3D(p1, p2)\n        >>> s2 = Segment3D(p2, p1)\n        >>> s.contains(s2)\n        True\n        >>> s.contains((p1 + p2)/2)\n        True\n        '
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if isinstance(other, Point):
            if Point.is_collinear(other, self.p1, self.p2):
                if isinstance(self, Segment2D):
                    vert = (1 / self.slope).equals(0)
                    if vert is False:
                        isin = (self.p1.x - other.x) * (self.p2.x - other.x) <= 0
                        if isin in (True, False):
                            return isin
                    if vert is True:
                        isin = (self.p1.y - other.y) * (self.p2.y - other.y) <= 0
                        if isin in (True, False):
                            return isin
                (d1, d2) = (other - self.p1, other - self.p2)
                d = self.p2 - self.p1
                try:
                    return bool(simplify(Eq(abs(d1) + abs(d2) - abs(d), 0)))
                except TypeError:
                    raise Undecidable('Cannot determine if {} is in {}'.format(other, self))
        if isinstance(other, Segment):
            return other.p1 in self and other.p2 in self
        return False

    def equals(self, other):
        if False:
            return 10
        'Returns True if self and other are the same mathematical entities'
        return isinstance(other, self.func) and list(ordered(self.args)) == list(ordered(other.args))

    def distance(self, other):
        if False:
            return 10
        '\n        Finds the shortest distance between a line segment and a point.\n\n        Raises\n        ======\n\n        NotImplementedError is raised if `other` is not a Point\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Segment\n        >>> p1, p2 = Point(0, 1), Point(3, 4)\n        >>> s = Segment(p1, p2)\n        >>> s.distance(Point(10, 15))\n        sqrt(170)\n        >>> s.distance((0, 12))\n        sqrt(73)\n        >>> from sympy import Point3D, Segment3D\n        >>> p1, p2 = Point3D(0, 0, 3), Point3D(1, 1, 4)\n        >>> s = Segment3D(p1, p2)\n        >>> s.distance(Point3D(10, 15, 12))\n        sqrt(341)\n        >>> s.distance((10, 15, 12))\n        sqrt(341)\n        '
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if isinstance(other, Point):
            vp1 = other - self.p1
            vp2 = other - self.p2
            dot_prod_sign_1 = self.direction.dot(vp1) >= 0
            dot_prod_sign_2 = self.direction.dot(vp2) <= 0
            if dot_prod_sign_1 and dot_prod_sign_2:
                return Line(self.p1, self.p2).distance(other)
            if dot_prod_sign_1 and (not dot_prod_sign_2):
                return abs(vp2)
            if not dot_prod_sign_1 and dot_prod_sign_2:
                return abs(vp1)
        raise NotImplementedError()

    @property
    def length(self):
        if False:
            i = 10
            return i + 15
        'The length of the line segment.\n\n        See Also\n        ========\n\n        sympy.geometry.point.Point.distance\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Segment\n        >>> p1, p2 = Point(0, 0), Point(4, 3)\n        >>> s1 = Segment(p1, p2)\n        >>> s1.length\n        5\n        >>> from sympy import Point3D, Segment3D\n        >>> p1, p2 = Point3D(0, 0, 0), Point3D(4, 3, 3)\n        >>> s1 = Segment3D(p1, p2)\n        >>> s1.length\n        sqrt(34)\n\n        '
        return Point.distance(self.p1, self.p2)

    @property
    def midpoint(self):
        if False:
            for i in range(10):
                print('nop')
        'The midpoint of the line segment.\n\n        See Also\n        ========\n\n        sympy.geometry.point.Point.midpoint\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Segment\n        >>> p1, p2 = Point(0, 0), Point(4, 3)\n        >>> s1 = Segment(p1, p2)\n        >>> s1.midpoint\n        Point2D(2, 3/2)\n        >>> from sympy import Point3D, Segment3D\n        >>> p1, p2 = Point3D(0, 0, 0), Point3D(4, 3, 3)\n        >>> s1 = Segment3D(p1, p2)\n        >>> s1.midpoint\n        Point3D(2, 3/2, 3/2)\n\n        '
        return Point.midpoint(self.p1, self.p2)

    def perpendicular_bisector(self, p=None):
        if False:
            i = 10
            return i + 15
        'The perpendicular bisector of this segment.\n\n        If no point is specified or the point specified is not on the\n        bisector then the bisector is returned as a Line. Otherwise a\n        Segment is returned that joins the point specified and the\n        intersection of the bisector and the segment.\n\n        Parameters\n        ==========\n\n        p : Point\n\n        Returns\n        =======\n\n        bisector : Line or Segment\n\n        See Also\n        ========\n\n        LinearEntity.perpendicular_segment\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Segment\n        >>> p1, p2, p3 = Point(0, 0), Point(6, 6), Point(5, 1)\n        >>> s1 = Segment(p1, p2)\n        >>> s1.perpendicular_bisector()\n        Line2D(Point2D(3, 3), Point2D(-3, 9))\n\n        >>> s1.perpendicular_bisector(p3)\n        Segment2D(Point2D(5, 1), Point2D(3, 3))\n\n        '
        l = self.perpendicular_line(self.midpoint)
        if p is not None:
            p2 = Point(p, dim=self.ambient_dimension)
            if p2 in l:
                return Segment(p2, self.midpoint)
        return l

    def plot_interval(self, parameter='t'):
        if False:
            print('Hello World!')
        "The plot interval for the default geometric plot of the Segment gives\n        values that will produce the full segment in a plot.\n\n        Parameters\n        ==========\n\n        parameter : str, optional\n            Default value is 't'.\n\n        Returns\n        =======\n\n        plot_interval : list\n            [parameter, lower_bound, upper_bound]\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Segment\n        >>> p1, p2 = Point(0, 0), Point(5, 3)\n        >>> s1 = Segment(p1, p2)\n        >>> s1.plot_interval()\n        [t, 0, 1]\n\n        "
        t = _symbol(parameter, real=True)
        return [t, 0, 1]

class LinearEntity2D(LinearEntity):
    """A base class for all linear entities (line, ray and segment)
    in a 2-dimensional Euclidean space.

    Attributes
    ==========

    p1
    p2
    coefficients
    slope
    points

    Notes
    =====

    This is an abstract class and is not meant to be instantiated.

    See Also
    ========

    sympy.geometry.entity.GeometryEntity

    """

    @property
    def bounds(self):
        if False:
            while True:
                i = 10
        'Return a tuple (xmin, ymin, xmax, ymax) representing the bounding\n        rectangle for the geometric figure.\n\n        '
        verts = self.points
        xs = [p.x for p in verts]
        ys = [p.y for p in verts]
        return (min(xs), min(ys), max(xs), max(ys))

    def perpendicular_line(self, p):
        if False:
            while True:
                i = 10
        'Create a new Line perpendicular to this linear entity which passes\n        through the point `p`.\n\n        Parameters\n        ==========\n\n        p : Point\n\n        Returns\n        =======\n\n        line : Line\n\n        See Also\n        ========\n\n        sympy.geometry.line.LinearEntity.is_perpendicular, perpendicular_segment\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Line\n        >>> p1, p2, p3 = Point(0, 0), Point(2, 3), Point(-2, 2)\n        >>> L = Line(p1, p2)\n        >>> P = L.perpendicular_line(p3); P\n        Line2D(Point2D(-2, 2), Point2D(-5, 4))\n        >>> L.is_perpendicular(P)\n        True\n\n        In 2D, the first point of the perpendicular line is the\n        point through which was required to pass; the second\n        point is arbitrarily chosen. To get a line that explicitly\n        uses a point in the line, create a line from the perpendicular\n        segment from the line to the point:\n\n        >>> Line(L.perpendicular_segment(p3))\n        Line2D(Point2D(-2, 2), Point2D(4/13, 6/13))\n        '
        p = Point(p, dim=self.ambient_dimension)
        return Line(p, p + self.direction.orthogonal_direction)

    @property
    def slope(self):
        if False:
            for i in range(10):
                print('nop')
        'The slope of this linear entity, or infinity if vertical.\n\n        Returns\n        =======\n\n        slope : number or SymPy expression\n\n        See Also\n        ========\n\n        coefficients\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Line\n        >>> p1, p2 = Point(0, 0), Point(3, 5)\n        >>> l1 = Line(p1, p2)\n        >>> l1.slope\n        5/3\n\n        >>> p3 = Point(0, 4)\n        >>> l2 = Line(p1, p3)\n        >>> l2.slope\n        oo\n\n        '
        (d1, d2) = (self.p1 - self.p2).args
        if d1 == 0:
            return S.Infinity
        return simplify(d2 / d1)

class Line2D(LinearEntity2D, Line):
    """An infinite line in space 2D.

    A line is declared with two distinct points or a point and slope
    as defined using keyword `slope`.

    Parameters
    ==========

    p1 : Point
    pt : Point
    slope : SymPy expression

    See Also
    ========

    sympy.geometry.point.Point

    Examples
    ========

    >>> from sympy import Line, Segment, Point
    >>> L = Line(Point(2,3), Point(3,5))
    >>> L
    Line2D(Point2D(2, 3), Point2D(3, 5))
    >>> L.points
    (Point2D(2, 3), Point2D(3, 5))
    >>> L.equation()
    -2*x + y + 1
    >>> L.coefficients
    (-2, 1, 1)

    Instantiate with keyword ``slope``:

    >>> Line(Point(0, 0), slope=0)
    Line2D(Point2D(0, 0), Point2D(1, 0))

    Instantiate with another linear object

    >>> s = Segment((0, 0), (0, 1))
    >>> Line(s).equation()
    x
    """

    def __new__(cls, p1, pt=None, slope=None, **kwargs):
        if False:
            while True:
                i = 10
        if isinstance(p1, LinearEntity):
            if pt is not None:
                raise ValueError('When p1 is a LinearEntity, pt should be None')
            (p1, pt) = Point._normalize_dimension(*p1.args, dim=2)
        else:
            p1 = Point(p1, dim=2)
        if pt is not None and slope is None:
            try:
                p2 = Point(pt, dim=2)
            except (NotImplementedError, TypeError, ValueError):
                raise ValueError(filldedent('\n                    The 2nd argument was not a valid Point.\n                    If it was a slope, enter it with keyword "slope".\n                    '))
        elif slope is not None and pt is None:
            slope = sympify(slope)
            if slope.is_finite is False:
                dx = 0
                dy = 1
            else:
                dx = 1
                dy = slope
            p2 = Point(p1.x + dx, p1.y + dy, evaluate=False)
        else:
            raise ValueError('A 2nd Point or keyword "slope" must be used.')
        return LinearEntity2D.__new__(cls, p1, p2, **kwargs)

    def _svg(self, scale_factor=1.0, fill_color='#66cc99'):
        if False:
            for i in range(10):
                print('nop')
        'Returns SVG path element for the LinearEntity.\n\n        Parameters\n        ==========\n\n        scale_factor : float\n            Multiplication factor for the SVG stroke-width.  Default is 1.\n        fill_color : str, optional\n            Hex string for fill color. Default is "#66cc99".\n        '
        verts = (N(self.p1), N(self.p2))
        coords = ['{},{}'.format(p.x, p.y) for p in verts]
        path = 'M {} L {}'.format(coords[0], ' L '.join(coords[1:]))
        return '<path fill-rule="evenodd" fill="{2}" stroke="#555555" stroke-width="{0}" opacity="0.6" d="{1}" marker-start="url(#markerReverseArrow)" marker-end="url(#markerArrow)"/>'.format(2.0 * scale_factor, path, fill_color)

    @property
    def coefficients(self):
        if False:
            return 10
        'The coefficients (`a`, `b`, `c`) for `ax + by + c = 0`.\n\n        See Also\n        ========\n\n        sympy.geometry.line.Line2D.equation\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Line\n        >>> from sympy.abc import x, y\n        >>> p1, p2 = Point(0, 0), Point(5, 3)\n        >>> l = Line(p1, p2)\n        >>> l.coefficients\n        (-3, 5, 0)\n\n        >>> p3 = Point(x, y)\n        >>> l2 = Line(p1, p3)\n        >>> l2.coefficients\n        (-y, x, 0)\n\n        '
        (p1, p2) = self.points
        if p1.x == p2.x:
            return (S.One, S.Zero, -p1.x)
        elif p1.y == p2.y:
            return (S.Zero, S.One, -p1.y)
        return tuple([simplify(i) for i in (self.p1.y - self.p2.y, self.p2.x - self.p1.x, self.p1.x * self.p2.y - self.p1.y * self.p2.x)])

    def equation(self, x='x', y='y'):
        if False:
            print('Hello World!')
        "The equation of the line: ax + by + c.\n\n        Parameters\n        ==========\n\n        x : str, optional\n            The name to use for the x-axis, default value is 'x'.\n        y : str, optional\n            The name to use for the y-axis, default value is 'y'.\n\n        Returns\n        =======\n\n        equation : SymPy expression\n\n        See Also\n        ========\n\n        sympy.geometry.line.Line2D.coefficients\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Line\n        >>> p1, p2 = Point(1, 0), Point(5, 3)\n        >>> l1 = Line(p1, p2)\n        >>> l1.equation()\n        -3*x + 4*y + 3\n\n        "
        x = _symbol(x, real=True)
        y = _symbol(y, real=True)
        (p1, p2) = self.points
        if p1.x == p2.x:
            return x - p1.x
        elif p1.y == p2.y:
            return y - p1.y
        (a, b, c) = self.coefficients
        return a * x + b * y + c

class Ray2D(LinearEntity2D, Ray):
    """
    A Ray is a semi-line in the space with a source point and a direction.

    Parameters
    ==========

    p1 : Point
        The source of the Ray
    p2 : Point or radian value
        This point determines the direction in which the Ray propagates.
        If given as an angle it is interpreted in radians with the positive
        direction being ccw.

    Attributes
    ==========

    source
    xdirection
    ydirection

    See Also
    ========

    sympy.geometry.point.Point, Line

    Examples
    ========

    >>> from sympy import Point, pi, Ray
    >>> r = Ray(Point(2, 3), Point(3, 5))
    >>> r
    Ray2D(Point2D(2, 3), Point2D(3, 5))
    >>> r.points
    (Point2D(2, 3), Point2D(3, 5))
    >>> r.source
    Point2D(2, 3)
    >>> r.xdirection
    oo
    >>> r.ydirection
    oo
    >>> r.slope
    2
    >>> Ray(Point(0, 0), angle=pi/4).slope
    1

    """

    def __new__(cls, p1, pt=None, angle=None, **kwargs):
        if False:
            print('Hello World!')
        p1 = Point(p1, dim=2)
        if pt is not None and angle is None:
            try:
                p2 = Point(pt, dim=2)
            except (NotImplementedError, TypeError, ValueError):
                raise ValueError(filldedent('\n                    The 2nd argument was not a valid Point; if\n                    it was meant to be an angle it should be\n                    given with keyword "angle".'))
            if p1 == p2:
                raise ValueError('A Ray requires two distinct points.')
        elif angle is not None and pt is None:
            angle = sympify(angle)
            c = _pi_coeff(angle)
            p2 = None
            if c is not None:
                if c.is_Rational:
                    if c.q == 2:
                        if c.p == 1:
                            p2 = p1 + Point(0, 1)
                        elif c.p == 3:
                            p2 = p1 + Point(0, -1)
                    elif c.q == 1:
                        if c.p == 0:
                            p2 = p1 + Point(1, 0)
                        elif c.p == 1:
                            p2 = p1 + Point(-1, 0)
                if p2 is None:
                    c *= S.Pi
            else:
                c = angle % (2 * S.Pi)
            if not p2:
                m = 2 * c / S.Pi
                left = And(1 < m, m < 3)
                x = Piecewise((-1, left), (Piecewise((0, Eq(m % 1, 0)), (1, True)), True))
                y = Piecewise((-tan(c), left), (Piecewise((1, Eq(m, 1)), (-1, Eq(m, 3)), (tan(c), True)), True))
                p2 = p1 + Point(x, y)
        else:
            raise ValueError('A 2nd point or keyword "angle" must be used.')
        return LinearEntity2D.__new__(cls, p1, p2, **kwargs)

    @property
    def xdirection(self):
        if False:
            i = 10
            return i + 15
        'The x direction of the ray.\n\n        Positive infinity if the ray points in the positive x direction,\n        negative infinity if the ray points in the negative x direction,\n        or 0 if the ray is vertical.\n\n        See Also\n        ========\n\n        ydirection\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ray\n        >>> p1, p2, p3 = Point(0, 0), Point(1, 1), Point(0, -1)\n        >>> r1, r2 = Ray(p1, p2), Ray(p1, p3)\n        >>> r1.xdirection\n        oo\n        >>> r2.xdirection\n        0\n\n        '
        if self.p1.x < self.p2.x:
            return S.Infinity
        elif self.p1.x == self.p2.x:
            return S.Zero
        else:
            return S.NegativeInfinity

    @property
    def ydirection(self):
        if False:
            print('Hello World!')
        'The y direction of the ray.\n\n        Positive infinity if the ray points in the positive y direction,\n        negative infinity if the ray points in the negative y direction,\n        or 0 if the ray is horizontal.\n\n        See Also\n        ========\n\n        xdirection\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ray\n        >>> p1, p2, p3 = Point(0, 0), Point(-1, -1), Point(-1, 0)\n        >>> r1, r2 = Ray(p1, p2), Ray(p1, p3)\n        >>> r1.ydirection\n        -oo\n        >>> r2.ydirection\n        0\n\n        '
        if self.p1.y < self.p2.y:
            return S.Infinity
        elif self.p1.y == self.p2.y:
            return S.Zero
        else:
            return S.NegativeInfinity

    def closing_angle(r1, r2):
        if False:
            for i in range(10):
                print('nop')
        'Return the angle by which r2 must be rotated so it faces the same\n        direction as r1.\n\n        Parameters\n        ==========\n\n        r1 : Ray2D\n        r2 : Ray2D\n\n        Returns\n        =======\n\n        angle : angle in radians (ccw angle is positive)\n\n        See Also\n        ========\n\n        LinearEntity.angle_between\n\n        Examples\n        ========\n\n        >>> from sympy import Ray, pi\n        >>> r1 = Ray((0, 0), (1, 0))\n        >>> r2 = r1.rotate(-pi/2)\n        >>> angle = r1.closing_angle(r2); angle\n        pi/2\n        >>> r2.rotate(angle).direction.unit == r1.direction.unit\n        True\n        >>> r2.closing_angle(r1)\n        -pi/2\n        '
        if not all((isinstance(r, Ray2D) for r in (r1, r2))):
            raise TypeError('Both arguments must be Ray2D objects.')
        a1 = atan2(*list(reversed(r1.direction.args)))
        a2 = atan2(*list(reversed(r2.direction.args)))
        if a1 * a2 < 0:
            a1 = 2 * S.Pi + a1 if a1 < 0 else a1
            a2 = 2 * S.Pi + a2 if a2 < 0 else a2
        return a1 - a2

class Segment2D(LinearEntity2D, Segment):
    """A line segment in 2D space.

    Parameters
    ==========

    p1 : Point
    p2 : Point

    Attributes
    ==========

    length : number or SymPy expression
    midpoint : Point

    See Also
    ========

    sympy.geometry.point.Point, Line

    Examples
    ========

    >>> from sympy import Point, Segment
    >>> Segment((1, 0), (1, 1)) # tuples are interpreted as pts
    Segment2D(Point2D(1, 0), Point2D(1, 1))
    >>> s = Segment(Point(4, 3), Point(1, 1)); s
    Segment2D(Point2D(4, 3), Point2D(1, 1))
    >>> s.points
    (Point2D(4, 3), Point2D(1, 1))
    >>> s.slope
    2/3
    >>> s.length
    sqrt(13)
    >>> s.midpoint
    Point2D(5/2, 2)

    """

    def __new__(cls, p1, p2, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        p1 = Point(p1, dim=2)
        p2 = Point(p2, dim=2)
        if p1 == p2:
            return p1
        return LinearEntity2D.__new__(cls, p1, p2, **kwargs)

    def _svg(self, scale_factor=1.0, fill_color='#66cc99'):
        if False:
            while True:
                i = 10
        'Returns SVG path element for the LinearEntity.\n\n        Parameters\n        ==========\n\n        scale_factor : float\n            Multiplication factor for the SVG stroke-width.  Default is 1.\n        fill_color : str, optional\n            Hex string for fill color. Default is "#66cc99".\n        '
        verts = (N(self.p1), N(self.p2))
        coords = ['{},{}'.format(p.x, p.y) for p in verts]
        path = 'M {} L {}'.format(coords[0], ' L '.join(coords[1:]))
        return '<path fill-rule="evenodd" fill="{2}" stroke="#555555" stroke-width="{0}" opacity="0.6" d="{1}" />'.format(2.0 * scale_factor, path, fill_color)

class LinearEntity3D(LinearEntity):
    """An base class for all linear entities (line, ray and segment)
    in a 3-dimensional Euclidean space.

    Attributes
    ==========

    p1
    p2
    direction_ratio
    direction_cosine
    points

    Notes
    =====

    This is a base class and is not meant to be instantiated.
    """

    def __new__(cls, p1, p2, **kwargs):
        if False:
            return 10
        p1 = Point3D(p1, dim=3)
        p2 = Point3D(p2, dim=3)
        if p1 == p2:
            raise ValueError('%s.__new__ requires two unique Points.' % cls.__name__)
        return GeometryEntity.__new__(cls, p1, p2, **kwargs)
    ambient_dimension = 3

    @property
    def direction_ratio(self):
        if False:
            while True:
                i = 10
        'The direction ratio of a given line in 3D.\n\n        See Also\n        ========\n\n        sympy.geometry.line.Line3D.equation\n\n        Examples\n        ========\n\n        >>> from sympy import Point3D, Line3D\n        >>> p1, p2 = Point3D(0, 0, 0), Point3D(5, 3, 1)\n        >>> l = Line3D(p1, p2)\n        >>> l.direction_ratio\n        [5, 3, 1]\n        '
        (p1, p2) = self.points
        return p1.direction_ratio(p2)

    @property
    def direction_cosine(self):
        if False:
            i = 10
            return i + 15
        'The normalized direction ratio of a given line in 3D.\n\n        See Also\n        ========\n\n        sympy.geometry.line.Line3D.equation\n\n        Examples\n        ========\n\n        >>> from sympy import Point3D, Line3D\n        >>> p1, p2 = Point3D(0, 0, 0), Point3D(5, 3, 1)\n        >>> l = Line3D(p1, p2)\n        >>> l.direction_cosine\n        [sqrt(35)/7, 3*sqrt(35)/35, sqrt(35)/35]\n        >>> sum(i**2 for i in _)\n        1\n        '
        (p1, p2) = self.points
        return p1.direction_cosine(p2)

class Line3D(LinearEntity3D, Line):
    """An infinite 3D line in space.

    A line is declared with two distinct points or a point and direction_ratio
    as defined using keyword `direction_ratio`.

    Parameters
    ==========

    p1 : Point3D
    pt : Point3D
    direction_ratio : list

    See Also
    ========

    sympy.geometry.point.Point3D
    sympy.geometry.line.Line
    sympy.geometry.line.Line2D

    Examples
    ========

    >>> from sympy import Line3D, Point3D
    >>> L = Line3D(Point3D(2, 3, 4), Point3D(3, 5, 1))
    >>> L
    Line3D(Point3D(2, 3, 4), Point3D(3, 5, 1))
    >>> L.points
    (Point3D(2, 3, 4), Point3D(3, 5, 1))
    """

    def __new__(cls, p1, pt=None, direction_ratio=(), **kwargs):
        if False:
            print('Hello World!')
        if isinstance(p1, LinearEntity3D):
            if pt is not None:
                raise ValueError('if p1 is a LinearEntity, pt must be None.')
            (p1, pt) = p1.args
        else:
            p1 = Point(p1, dim=3)
        if pt is not None and len(direction_ratio) == 0:
            pt = Point(pt, dim=3)
        elif len(direction_ratio) == 3 and pt is None:
            pt = Point3D(p1.x + direction_ratio[0], p1.y + direction_ratio[1], p1.z + direction_ratio[2])
        else:
            raise ValueError('A 2nd Point or keyword "direction_ratio" must be used.')
        return LinearEntity3D.__new__(cls, p1, pt, **kwargs)

    def equation(self, x='x', y='y', z='z'):
        if False:
            print('Hello World!')
        "Return the equations that define the line in 3D.\n\n        Parameters\n        ==========\n\n        x : str, optional\n            The name to use for the x-axis, default value is 'x'.\n        y : str, optional\n            The name to use for the y-axis, default value is 'y'.\n        z : str, optional\n            The name to use for the z-axis, default value is 'z'.\n\n        Returns\n        =======\n\n        equation : Tuple of simultaneous equations\n\n        Examples\n        ========\n\n        >>> from sympy import Point3D, Line3D, solve\n        >>> from sympy.abc import x, y, z\n        >>> p1, p2 = Point3D(1, 0, 0), Point3D(5, 3, 0)\n        >>> l1 = Line3D(p1, p2)\n        >>> eq = l1.equation(x, y, z); eq\n        (-3*x + 4*y + 3, z)\n        >>> solve(eq.subs(z, 0), (x, y, z))\n        {x: 4*y/3 + 1}\n        "
        (x, y, z, k) = [_symbol(i, real=True) for i in (x, y, z, 'k')]
        (p1, p2) = self.points
        (d1, d2, d3) = p1.direction_ratio(p2)
        (x1, y1, z1) = p1
        eqs = [-d1 * k + x - x1, -d2 * k + y - y1, -d3 * k + z - z1]
        for (i, e) in enumerate(eqs):
            if e.has(k):
                kk = solve(eqs[i], k)[0]
                eqs.pop(i)
                break
        return Tuple(*[i.subs(k, kk).as_numer_denom()[0] for i in eqs])

    def distance(self, other):
        if False:
            return 10
        "\n        Finds the shortest distance between a line and another object.\n\n        Parameters\n        ==========\n\n        Point3D, Line3D, Plane, tuple, list\n\n        Returns\n        =======\n\n        distance\n\n        Notes\n        =====\n\n        This method accepts only 3D entities as it's parameter\n\n        Tuples and lists are converted to Point3D and therefore must be of\n        length 3, 2 or 1.\n\n        NotImplementedError is raised if `other` is not an instance of one\n        of the specified classes: Point3D, Line3D, or Plane.\n\n        Examples\n        ========\n\n        >>> from sympy.geometry import Line3D\n        >>> l1 = Line3D((0, 0, 0), (0, 0, 1))\n        >>> l2 = Line3D((0, 1, 0), (1, 1, 1))\n        >>> l1.distance(l2)\n        1\n\n        The computed distance may be symbolic, too:\n\n        >>> from sympy.abc import x, y\n        >>> l1 = Line3D((0, 0, 0), (0, 0, 1))\n        >>> l2 = Line3D((0, x, 0), (y, x, 1))\n        >>> l1.distance(l2)\n        Abs(x*y)/Abs(sqrt(y**2))\n\n        "
        from .plane import Plane
        if isinstance(other, (tuple, list)):
            try:
                other = Point3D(other)
            except ValueError:
                pass
        if isinstance(other, Point3D):
            return super().distance(other)
        if isinstance(other, Line3D):
            if self == other:
                return S.Zero
            if self.is_parallel(other):
                return super().distance(other.p1)
            self_direction = Matrix(self.direction_ratio)
            other_direction = Matrix(other.direction_ratio)
            normal = self_direction.cross(other_direction)
            plane_through_self = Plane(p1=self.p1, normal_vector=normal)
            return other.p1.distance(plane_through_self)
        if isinstance(other, Plane):
            return other.distance(self)
        msg = f'{other} has type {type(other)}, which is unsupported'
        raise NotImplementedError(msg)

class Ray3D(LinearEntity3D, Ray):
    """
    A Ray is a semi-line in the space with a source point and a direction.

    Parameters
    ==========

    p1 : Point3D
        The source of the Ray
    p2 : Point or a direction vector
    direction_ratio: Determines the direction in which the Ray propagates.


    Attributes
    ==========

    source
    xdirection
    ydirection
    zdirection

    See Also
    ========

    sympy.geometry.point.Point3D, Line3D


    Examples
    ========

    >>> from sympy import Point3D, Ray3D
    >>> r = Ray3D(Point3D(2, 3, 4), Point3D(3, 5, 0))
    >>> r
    Ray3D(Point3D(2, 3, 4), Point3D(3, 5, 0))
    >>> r.points
    (Point3D(2, 3, 4), Point3D(3, 5, 0))
    >>> r.source
    Point3D(2, 3, 4)
    >>> r.xdirection
    oo
    >>> r.ydirection
    oo
    >>> r.direction_ratio
    [1, 2, -4]

    """

    def __new__(cls, p1, pt=None, direction_ratio=(), **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(p1, LinearEntity3D):
            if pt is not None:
                raise ValueError('If p1 is a LinearEntity, pt must be None')
            (p1, pt) = p1.args
        else:
            p1 = Point(p1, dim=3)
        if pt is not None and len(direction_ratio) == 0:
            pt = Point(pt, dim=3)
        elif len(direction_ratio) == 3 and pt is None:
            pt = Point3D(p1.x + direction_ratio[0], p1.y + direction_ratio[1], p1.z + direction_ratio[2])
        else:
            raise ValueError(filldedent('\n                A 2nd Point or keyword "direction_ratio" must be used.\n            '))
        return LinearEntity3D.__new__(cls, p1, pt, **kwargs)

    @property
    def xdirection(self):
        if False:
            while True:
                i = 10
        'The x direction of the ray.\n\n        Positive infinity if the ray points in the positive x direction,\n        negative infinity if the ray points in the negative x direction,\n        or 0 if the ray is vertical.\n\n        See Also\n        ========\n\n        ydirection\n\n        Examples\n        ========\n\n        >>> from sympy import Point3D, Ray3D\n        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(0, -1, 0)\n        >>> r1, r2 = Ray3D(p1, p2), Ray3D(p1, p3)\n        >>> r1.xdirection\n        oo\n        >>> r2.xdirection\n        0\n\n        '
        if self.p1.x < self.p2.x:
            return S.Infinity
        elif self.p1.x == self.p2.x:
            return S.Zero
        else:
            return S.NegativeInfinity

    @property
    def ydirection(self):
        if False:
            return 10
        'The y direction of the ray.\n\n        Positive infinity if the ray points in the positive y direction,\n        negative infinity if the ray points in the negative y direction,\n        or 0 if the ray is horizontal.\n\n        See Also\n        ========\n\n        xdirection\n\n        Examples\n        ========\n\n        >>> from sympy import Point3D, Ray3D\n        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(-1, -1, -1), Point3D(-1, 0, 0)\n        >>> r1, r2 = Ray3D(p1, p2), Ray3D(p1, p3)\n        >>> r1.ydirection\n        -oo\n        >>> r2.ydirection\n        0\n\n        '
        if self.p1.y < self.p2.y:
            return S.Infinity
        elif self.p1.y == self.p2.y:
            return S.Zero
        else:
            return S.NegativeInfinity

    @property
    def zdirection(self):
        if False:
            i = 10
            return i + 15
        'The z direction of the ray.\n\n        Positive infinity if the ray points in the positive z direction,\n        negative infinity if the ray points in the negative z direction,\n        or 0 if the ray is horizontal.\n\n        See Also\n        ========\n\n        xdirection\n\n        Examples\n        ========\n\n        >>> from sympy import Point3D, Ray3D\n        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(-1, -1, -1), Point3D(-1, 0, 0)\n        >>> r1, r2 = Ray3D(p1, p2), Ray3D(p1, p3)\n        >>> r1.ydirection\n        -oo\n        >>> r2.ydirection\n        0\n        >>> r2.zdirection\n        0\n\n        '
        if self.p1.z < self.p2.z:
            return S.Infinity
        elif self.p1.z == self.p2.z:
            return S.Zero
        else:
            return S.NegativeInfinity

class Segment3D(LinearEntity3D, Segment):
    """A line segment in a 3D space.

    Parameters
    ==========

    p1 : Point3D
    p2 : Point3D

    Attributes
    ==========

    length : number or SymPy expression
    midpoint : Point3D

    See Also
    ========

    sympy.geometry.point.Point3D, Line3D

    Examples
    ========

    >>> from sympy import Point3D, Segment3D
    >>> Segment3D((1, 0, 0), (1, 1, 1)) # tuples are interpreted as pts
    Segment3D(Point3D(1, 0, 0), Point3D(1, 1, 1))
    >>> s = Segment3D(Point3D(4, 3, 9), Point3D(1, 1, 7)); s
    Segment3D(Point3D(4, 3, 9), Point3D(1, 1, 7))
    >>> s.points
    (Point3D(4, 3, 9), Point3D(1, 1, 7))
    >>> s.length
    sqrt(17)
    >>> s.midpoint
    Point3D(5/2, 2, 8)

    """

    def __new__(cls, p1, p2, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        p1 = Point(p1, dim=3)
        p2 = Point(p2, dim=3)
        if p1 == p2:
            return p1
        return LinearEntity3D.__new__(cls, p1, p2, **kwargs)