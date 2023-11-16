"""The definition of the base geometrical entity with attributes common to
all derived geometrical entities.

Contains
========

GeometryEntity
GeometricSet

Notes
=====

A GeometryEntity is any object that has special geometric properties.
A GeometrySet is a superclass of any GeometryEntity that can also
be viewed as a sympy.sets.Set.  In particular, points are the only
GeometryEntity not considered a Set.

Rn is a GeometrySet representing n-dimensional Euclidean space. R2 and
R3 are currently the only ambient spaces implemented.

"""
from __future__ import annotations
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.evalf import EvalfMixin, N
from sympy.core.numbers import oo
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import cos, sin, atan
from sympy.matrices import eye
from sympy.multipledispatch import dispatch
from sympy.printing import sstr
from sympy.sets import Set, Union, FiniteSet
from sympy.sets.handlers.intersection import intersection_sets
from sympy.sets.handlers.union import union_sets
from sympy.solvers.solvers import solve
from sympy.utilities.misc import func_name
from sympy.utilities.iterables import is_sequence
ordering_of_classes = ['Point2D', 'Point3D', 'Point', 'Segment2D', 'Ray2D', 'Line2D', 'Segment3D', 'Line3D', 'Ray3D', 'Segment', 'Ray', 'Line', 'Plane', 'Triangle', 'RegularPolygon', 'Polygon', 'Circle', 'Ellipse', 'Curve', 'Parabola']
(x, y) = [Dummy('entity_dummy') for i in range(2)]
T = Dummy('entity_dummy', real=True)

class GeometryEntity(Basic, EvalfMixin):
    """The base class for all geometrical entities.

    This class does not represent any particular geometric entity, it only
    provides the implementation of some methods common to all subclasses.

    """
    __slots__: tuple[str, ...] = ()

    def __cmp__(self, other):
        if False:
            i = 10
            return i + 15
        'Comparison of two GeometryEntities.'
        n1 = self.__class__.__name__
        n2 = other.__class__.__name__
        c = (n1 > n2) - (n1 < n2)
        if not c:
            return 0
        i1 = -1
        for cls in self.__class__.__mro__:
            try:
                i1 = ordering_of_classes.index(cls.__name__)
                break
            except ValueError:
                i1 = -1
        if i1 == -1:
            return c
        i2 = -1
        for cls in other.__class__.__mro__:
            try:
                i2 = ordering_of_classes.index(cls.__name__)
                break
            except ValueError:
                i2 = -1
        if i2 == -1:
            return c
        return (i1 > i2) - (i1 < i2)

    def __contains__(self, other):
        if False:
            i = 10
            return i + 15
        'Subclasses should implement this method for anything more complex than equality.'
        if type(self) is type(other):
            return self == other
        raise NotImplementedError()

    def __getnewargs__(self):
        if False:
            i = 10
            return i + 15
        'Returns a tuple that will be passed to __new__ on unpickling.'
        return tuple(self.args)

    def __ne__(self, o):
        if False:
            return 10
        'Test inequality of two geometrical entities.'
        return not self == o

    def __new__(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15

        def is_seq_and_not_point(a):
            if False:
                while True:
                    i = 10
            if hasattr(a, 'is_Point') and a.is_Point:
                return False
            return is_sequence(a)
        args = [Tuple(*a) if is_seq_and_not_point(a) else sympify(a) for a in args]
        return Basic.__new__(cls, *args)

    def __radd__(self, a):
        if False:
            while True:
                i = 10
        'Implementation of reverse add method.'
        return a.__add__(self)

    def __rtruediv__(self, a):
        if False:
            return 10
        'Implementation of reverse division method.'
        return a.__truediv__(self)

    def __repr__(self):
        if False:
            while True:
                i = 10
        'String representation of a GeometryEntity that can be evaluated\n        by sympy.'
        return type(self).__name__ + repr(self.args)

    def __rmul__(self, a):
        if False:
            for i in range(10):
                print('nop')
        'Implementation of reverse multiplication method.'
        return a.__mul__(self)

    def __rsub__(self, a):
        if False:
            print('Hello World!')
        'Implementation of reverse subtraction method.'
        return a.__sub__(self)

    def __str__(self):
        if False:
            return 10
        'String representation of a GeometryEntity.'
        return type(self).__name__ + sstr(self.args)

    def _eval_subs(self, old, new):
        if False:
            for i in range(10):
                print('nop')
        from sympy.geometry.point import Point, Point3D
        if is_sequence(old) or is_sequence(new):
            if isinstance(self, Point3D):
                old = Point3D(old)
                new = Point3D(new)
            else:
                old = Point(old)
                new = Point(new)
            return self._subs(old, new)

    def _repr_svg_(self):
        if False:
            for i in range(10):
                print('nop')
        'SVG representation of a GeometryEntity suitable for IPython'
        try:
            bounds = self.bounds
        except (NotImplementedError, TypeError):
            return None
        if not all((x.is_number and x.is_finite for x in bounds)):
            return None
        svg_top = '<svg xmlns="http://www.w3.org/2000/svg"\n            xmlns:xlink="http://www.w3.org/1999/xlink"\n            width="{1}" height="{2}" viewBox="{0}"\n            preserveAspectRatio="xMinYMin meet">\n            <defs>\n                <marker id="markerCircle" markerWidth="8" markerHeight="8"\n                    refx="5" refy="5" markerUnits="strokeWidth">\n                    <circle cx="5" cy="5" r="1.5" style="stroke: none; fill:#000000;"/>\n                </marker>\n                <marker id="markerArrow" markerWidth="13" markerHeight="13" refx="2" refy="4"\n                       orient="auto" markerUnits="strokeWidth">\n                    <path d="M2,2 L2,6 L6,4" style="fill: #000000;" />\n                </marker>\n                <marker id="markerReverseArrow" markerWidth="13" markerHeight="13" refx="6" refy="4"\n                       orient="auto" markerUnits="strokeWidth">\n                    <path d="M6,2 L6,6 L2,4" style="fill: #000000;" />\n                </marker>\n            </defs>'
        (xmin, ymin, xmax, ymax) = map(N, bounds)
        if xmin == xmax and ymin == ymax:
            (xmin, ymin, xmax, ymax) = (xmin - 0.5, ymin - 0.5, xmax + 0.5, ymax + 0.5)
        else:
            expand = 0.1
            widest_part = max([xmax - xmin, ymax - ymin])
            expand_amount = widest_part * expand
            xmin -= expand_amount
            ymin -= expand_amount
            xmax += expand_amount
            ymax += expand_amount
        dx = xmax - xmin
        dy = ymax - ymin
        width = min([max([100.0, dx]), 300])
        height = min([max([100.0, dy]), 300])
        scale_factor = 1.0 if max(width, height) == 0 else max(dx, dy) / max(width, height)
        try:
            svg = self._svg(scale_factor)
        except (NotImplementedError, TypeError):
            return None
        view_box = '{} {} {} {}'.format(xmin, ymin, dx, dy)
        transform = 'matrix(1,0,0,-1,0,{})'.format(ymax + ymin)
        svg_top = svg_top.format(view_box, width, height)
        return svg_top + '<g transform="{}">{}</g></svg>'.format(transform, svg)

    def _svg(self, scale_factor=1.0, fill_color='#66cc99'):
        if False:
            while True:
                i = 10
        'Returns SVG path element for the GeometryEntity.\n\n        Parameters\n        ==========\n\n        scale_factor : float\n            Multiplication factor for the SVG stroke-width.  Default is 1.\n        fill_color : str, optional\n            Hex string for fill color. Default is "#66cc99".\n        '
        raise NotImplementedError()

    def _sympy_(self):
        if False:
            while True:
                i = 10
        return self

    @property
    def ambient_dimension(self):
        if False:
            while True:
                i = 10
        'What is the dimension of the space that the object is contained in?'
        raise NotImplementedError()

    @property
    def bounds(self):
        if False:
            return 10
        'Return a tuple (xmin, ymin, xmax, ymax) representing the bounding\n        rectangle for the geometric figure.\n\n        '
        raise NotImplementedError()

    def encloses(self, o):
        if False:
            return 10
        '\n        Return True if o is inside (not on or outside) the boundaries of self.\n\n        The object will be decomposed into Points and individual Entities need\n        only define an encloses_point method for their class.\n\n        See Also\n        ========\n\n        sympy.geometry.ellipse.Ellipse.encloses_point\n        sympy.geometry.polygon.Polygon.encloses_point\n\n        Examples\n        ========\n\n        >>> from sympy import RegularPolygon, Point, Polygon\n        >>> t  = Polygon(*RegularPolygon(Point(0, 0), 1, 3).vertices)\n        >>> t2 = Polygon(*RegularPolygon(Point(0, 0), 2, 3).vertices)\n        >>> t2.encloses(t)\n        True\n        >>> t.encloses(t2)\n        False\n\n        '
        from sympy.geometry.point import Point
        from sympy.geometry.line import Segment, Ray, Line
        from sympy.geometry.ellipse import Ellipse
        from sympy.geometry.polygon import Polygon, RegularPolygon
        if isinstance(o, Point):
            return self.encloses_point(o)
        elif isinstance(o, Segment):
            return all((self.encloses_point(x) for x in o.points))
        elif isinstance(o, (Ray, Line)):
            return False
        elif isinstance(o, Ellipse):
            return self.encloses_point(o.center) and self.encloses_point(Point(o.center.x + o.hradius, o.center.y)) and (not self.intersection(o))
        elif isinstance(o, Polygon):
            if isinstance(o, RegularPolygon):
                if not self.encloses_point(o.center):
                    return False
            return all((self.encloses_point(v) for v in o.vertices))
        raise NotImplementedError()

    def equals(self, o):
        if False:
            return 10
        return self == o

    def intersection(self, o):
        if False:
            return 10
        '\n        Returns a list of all of the intersections of self with o.\n\n        Notes\n        =====\n\n        An entity is not required to implement this method.\n\n        If two different types of entities can intersect, the item with\n        higher index in ordering_of_classes should implement\n        intersections with anything having a lower index.\n\n        See Also\n        ========\n\n        sympy.geometry.util.intersection\n\n        '
        raise NotImplementedError()

    def is_similar(self, other):
        if False:
            i = 10
            return i + 15
        'Is this geometrical entity similar to another geometrical entity?\n\n        Two entities are similar if a uniform scaling (enlarging or\n        shrinking) of one of the entities will allow one to obtain the other.\n\n        Notes\n        =====\n\n        This method is not intended to be used directly but rather\n        through the `are_similar` function found in util.py.\n        An entity is not required to implement this method.\n        If two different types of entities can be similar, it is only\n        required that one of them be able to determine this.\n\n        See Also\n        ========\n\n        scale\n\n        '
        raise NotImplementedError()

    def reflect(self, line):
        if False:
            for i in range(10):
                print('nop')
        '\n        Reflects an object across a line.\n\n        Parameters\n        ==========\n\n        line: Line\n\n        Examples\n        ========\n\n        >>> from sympy import pi, sqrt, Line, RegularPolygon\n        >>> l = Line((0, pi), slope=sqrt(2))\n        >>> pent = RegularPolygon((1, 2), 1, 5)\n        >>> rpent = pent.reflect(l)\n        >>> rpent\n        RegularPolygon(Point2D(-2*sqrt(2)*pi/3 - 1/3 + 4*sqrt(2)/3, 2/3 + 2*sqrt(2)/3 + 2*pi/3), -1, 5, -atan(2*sqrt(2)) + 3*pi/5)\n\n        >>> from sympy import pi, Line, Circle, Point\n        >>> l = Line((0, pi), slope=1)\n        >>> circ = Circle(Point(0, 0), 5)\n        >>> rcirc = circ.reflect(l)\n        >>> rcirc\n        Circle(Point2D(-pi, pi), -5)\n\n        '
        from sympy.geometry.point import Point
        g = self
        l = line
        o = Point(0, 0)
        if l.slope.is_zero:
            v = l.args[0].y
            if not v:
                return g.scale(y=-1)
            reps = [(p, p.translate(y=2 * (v - p.y))) for p in g.atoms(Point)]
        elif l.slope is oo:
            v = l.args[0].x
            if not v:
                return g.scale(x=-1)
            reps = [(p, p.translate(x=2 * (v - p.x))) for p in g.atoms(Point)]
        else:
            if not hasattr(g, 'reflect') and (not all((isinstance(arg, Point) for arg in g.args))):
                raise NotImplementedError('reflect undefined or non-Point args in %s' % g)
            a = atan(l.slope)
            c = l.coefficients
            d = -c[-1] / c[1]
            xf = Point(x, y)
            xf = xf.translate(y=-d).rotate(-a, o).scale(y=-1).rotate(a, o).translate(y=d)
            reps = [(p, xf.xreplace({x: p.x, y: p.y})) for p in g.atoms(Point)]
        return g.xreplace(dict(reps))

    def rotate(self, angle, pt=None):
        if False:
            while True:
                i = 10
        'Rotate ``angle`` radians counterclockwise about Point ``pt``.\n\n        The default pt is the origin, Point(0, 0)\n\n        See Also\n        ========\n\n        scale, translate\n\n        Examples\n        ========\n\n        >>> from sympy import Point, RegularPolygon, Polygon, pi\n        >>> t = Polygon(*RegularPolygon(Point(0, 0), 1, 3).vertices)\n        >>> t # vertex on x axis\n        Triangle(Point2D(1, 0), Point2D(-1/2, sqrt(3)/2), Point2D(-1/2, -sqrt(3)/2))\n        >>> t.rotate(pi/2) # vertex on y axis now\n        Triangle(Point2D(0, 1), Point2D(-sqrt(3)/2, -1/2), Point2D(sqrt(3)/2, -1/2))\n\n        '
        newargs = []
        for a in self.args:
            if isinstance(a, GeometryEntity):
                newargs.append(a.rotate(angle, pt))
            else:
                newargs.append(a)
        return type(self)(*newargs)

    def scale(self, x=1, y=1, pt=None):
        if False:
            print('Hello World!')
        'Scale the object by multiplying the x,y-coordinates by x and y.\n\n        If pt is given, the scaling is done relative to that point; the\n        object is shifted by -pt, scaled, and shifted by pt.\n\n        See Also\n        ========\n\n        rotate, translate\n\n        Examples\n        ========\n\n        >>> from sympy import RegularPolygon, Point, Polygon\n        >>> t = Polygon(*RegularPolygon(Point(0, 0), 1, 3).vertices)\n        >>> t\n        Triangle(Point2D(1, 0), Point2D(-1/2, sqrt(3)/2), Point2D(-1/2, -sqrt(3)/2))\n        >>> t.scale(2)\n        Triangle(Point2D(2, 0), Point2D(-1, sqrt(3)/2), Point2D(-1, -sqrt(3)/2))\n        >>> t.scale(2, 2)\n        Triangle(Point2D(2, 0), Point2D(-1, sqrt(3)), Point2D(-1, -sqrt(3)))\n\n        '
        from sympy.geometry.point import Point
        if pt:
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        return type(self)(*[a.scale(x, y) for a in self.args])

    def translate(self, x=0, y=0):
        if False:
            return 10
        'Shift the object by adding to the x,y-coordinates the values x and y.\n\n        See Also\n        ========\n\n        rotate, scale\n\n        Examples\n        ========\n\n        >>> from sympy import RegularPolygon, Point, Polygon\n        >>> t = Polygon(*RegularPolygon(Point(0, 0), 1, 3).vertices)\n        >>> t\n        Triangle(Point2D(1, 0), Point2D(-1/2, sqrt(3)/2), Point2D(-1/2, -sqrt(3)/2))\n        >>> t.translate(2)\n        Triangle(Point2D(3, 0), Point2D(3/2, sqrt(3)/2), Point2D(3/2, -sqrt(3)/2))\n        >>> t.translate(2, 2)\n        Triangle(Point2D(3, 2), Point2D(3/2, sqrt(3)/2 + 2), Point2D(3/2, 2 - sqrt(3)/2))\n\n        '
        newargs = []
        for a in self.args:
            if isinstance(a, GeometryEntity):
                newargs.append(a.translate(x, y))
            else:
                newargs.append(a)
        return self.func(*newargs)

    def parameter_value(self, other, t):
        if False:
            for i in range(10):
                print('nop')
        'Return the parameter corresponding to the given point.\n        Evaluating an arbitrary point of the entity at this parameter\n        value will return the given point.\n\n        Examples\n        ========\n\n        >>> from sympy import Line, Point\n        >>> from sympy.abc import t\n        >>> a = Point(0, 0)\n        >>> b = Point(2, 2)\n        >>> Line(a, b).parameter_value((1, 1), t)\n        {t: 1/2}\n        >>> Line(a, b).arbitrary_point(t).subs(_)\n        Point2D(1, 1)\n        '
        from sympy.geometry.point import Point
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if not isinstance(other, Point):
            raise ValueError('other must be a point')
        sol = solve(self.arbitrary_point(T) - other, T, dict=True)
        if not sol:
            raise ValueError('Given point is not on %s' % func_name(self))
        return {t: sol[0][T]}

class GeometrySet(GeometryEntity, Set):
    """Parent class of all GeometryEntity that are also Sets
    (compatible with sympy.sets)
    """
    __slots__ = ()

    def _contains(self, other):
        if False:
            return 10
        'sympy.sets uses the _contains method, so include it for compatibility.'
        if isinstance(other, Set) and other.is_FiniteSet:
            return all((self.__contains__(i) for i in other))
        return self.__contains__(other)

@dispatch(GeometrySet, Set)
def union_sets(self, o):
    if False:
        for i in range(10):
            print('nop')
    ' Returns the union of self and o\n    for use with sympy.sets.Set, if possible. '
    if o.is_FiniteSet:
        other_points = [p for p in o if not self._contains(p)]
        if len(other_points) == len(o):
            return None
        return Union(self, FiniteSet(*other_points))
    if self._contains(o):
        return self
    return None

@dispatch(GeometrySet, Set)
def intersection_sets(self, o):
    if False:
        while True:
            i = 10
    ' Returns a sympy.sets.Set of intersection objects,\n    if possible. '
    from sympy.geometry.point import Point
    try:
        if o.is_FiniteSet:
            inter = FiniteSet(*(p for p in o if self.contains(p)))
        else:
            inter = self.intersection(o)
    except NotImplementedError:
        return None
    points = FiniteSet(*[p for p in inter if isinstance(p, Point)])
    non_points = [p for p in inter if not isinstance(p, Point)]
    return Union(*non_points + [points])

def translate(x, y):
    if False:
        while True:
            i = 10
    'Return the matrix to translate a 2-D point by x and y.'
    rv = eye(3)
    rv[2, 0] = x
    rv[2, 1] = y
    return rv

def scale(x, y, pt=None):
    if False:
        return 10
    "Return the matrix to multiply a 2-D point's coordinates by x and y.\n\n    If pt is given, the scaling is done relative to that point."
    rv = eye(3)
    rv[0, 0] = x
    rv[1, 1] = y
    if pt:
        from sympy.geometry.point import Point
        pt = Point(pt, dim=2)
        tr1 = translate(*(-pt).args)
        tr2 = translate(*pt.args)
        return tr1 * rv * tr2
    return rv

def rotate(th):
    if False:
        i = 10
        return i + 15
    'Return the matrix to rotate a 2-D point about the origin by ``angle``.\n\n    The angle is measured in radians. To Point a point about a point other\n    then the origin, translate the Point, do the rotation, and\n    translate it back:\n\n    >>> from sympy.geometry.entity import rotate, translate\n    >>> from sympy import Point, pi\n    >>> rot_about_11 = translate(-1, -1)*rotate(pi/2)*translate(1, 1)\n    >>> Point(1, 1).transform(rot_about_11)\n    Point2D(1, 1)\n    >>> Point(0, 0).transform(rot_about_11)\n    Point2D(2, 0)\n    '
    s = sin(th)
    rv = eye(3) * cos(th)
    rv[0, 1] = s
    rv[1, 0] = -s
    rv[2, 2] = 1
    return rv