"""Elliptical geometrical entities.

Contains
* Ellipse
* Circle

"""
from sympy.core.expr import Expr
from sympy.core.relational import Eq
from sympy.core import S, pi, sympify
from sympy.core.evalf import N
from sympy.core.parameters import global_parameters
from sympy.core.logic import fuzzy_bool
from sympy.core.numbers import Rational, oo
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, uniquely_named_symbol, _symbol
from sympy.simplify import simplify, trigsimp
from sympy.functions.elementary.miscellaneous import sqrt, Max
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.elliptic_integrals import elliptic_e
from .entity import GeometryEntity, GeometrySet
from .exceptions import GeometryError
from .line import Line, Segment, Ray2D, Segment2D, Line2D, LinearEntity3D
from .point import Point, Point2D, Point3D
from .util import idiff, find
from sympy.polys import DomainError, Poly, PolynomialError
from sympy.polys.polyutils import _not_a_coeff, _nsort
from sympy.solvers import solve
from sympy.solvers.solveset import linear_coeffs
from sympy.utilities.misc import filldedent, func_name
from mpmath.libmp.libmpf import prec_to_dps
import random
(x, y) = [Dummy('ellipse_dummy', real=True) for i in range(2)]

class Ellipse(GeometrySet):
    """An elliptical GeometryEntity.

    Parameters
    ==========

    center : Point, optional
        Default value is Point(0, 0)
    hradius : number or SymPy expression, optional
    vradius : number or SymPy expression, optional
    eccentricity : number or SymPy expression, optional
        Two of `hradius`, `vradius` and `eccentricity` must be supplied to
        create an Ellipse. The third is derived from the two supplied.

    Attributes
    ==========

    center
    hradius
    vradius
    area
    circumference
    eccentricity
    periapsis
    apoapsis
    focus_distance
    foci

    Raises
    ======

    GeometryError
        When `hradius`, `vradius` and `eccentricity` are incorrectly supplied
        as parameters.
    TypeError
        When `center` is not a Point.

    See Also
    ========

    Circle

    Notes
    -----
    Constructed from a center and two radii, the first being the horizontal
    radius (along the x-axis) and the second being the vertical radius (along
    the y-axis).

    When symbolic value for hradius and vradius are used, any calculation that
    refers to the foci or the major or minor axis will assume that the ellipse
    has its major radius on the x-axis. If this is not true then a manual
    rotation is necessary.

    Examples
    ========

    >>> from sympy import Ellipse, Point, Rational
    >>> e1 = Ellipse(Point(0, 0), 5, 1)
    >>> e1.hradius, e1.vradius
    (5, 1)
    >>> e2 = Ellipse(Point(3, 1), hradius=3, eccentricity=Rational(4, 5))
    >>> e2
    Ellipse(Point2D(3, 1), 3, 9/5)

    """

    def __contains__(self, o):
        if False:
            while True:
                i = 10
        if isinstance(o, Point):
            res = self.equation(x, y).subs({x: o.x, y: o.y})
            return trigsimp(simplify(res)) is S.Zero
        elif isinstance(o, Ellipse):
            return self == o
        return False

    def __eq__(self, o):
        if False:
            i = 10
            return i + 15
        'Is the other GeometryEntity the same as this ellipse?'
        return isinstance(o, Ellipse) and (self.center == o.center and self.hradius == o.hradius and (self.vradius == o.vradius))

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return super().__hash__()

    def __new__(cls, center=None, hradius=None, vradius=None, eccentricity=None, **kwargs):
        if False:
            i = 10
            return i + 15
        hradius = sympify(hradius)
        vradius = sympify(vradius)
        if center is None:
            center = Point(0, 0)
        else:
            if len(center) != 2:
                raise ValueError('The center of "{}" must be a two dimensional point'.format(cls))
            center = Point(center, dim=2)
        if len(list(filter(lambda x: x is not None, (hradius, vradius, eccentricity)))) != 2:
            raise ValueError(filldedent('\n                Exactly two arguments of "hradius", "vradius", and\n                "eccentricity" must not be None.'))
        if eccentricity is not None:
            eccentricity = sympify(eccentricity)
            if eccentricity.is_negative:
                raise GeometryError('Eccentricity of ellipse/circle should lie between [0, 1)')
            elif hradius is None:
                hradius = vradius / sqrt(1 - eccentricity ** 2)
            elif vradius is None:
                vradius = hradius * sqrt(1 - eccentricity ** 2)
        if hradius == vradius:
            return Circle(center, hradius, **kwargs)
        if S.Zero in (hradius, vradius):
            return Segment(Point(center[0] - hradius, center[1] - vradius), Point(center[0] + hradius, center[1] + vradius))
        if hradius.is_real is False or vradius.is_real is False:
            raise GeometryError('Invalid value encountered when computing hradius / vradius.')
        return GeometryEntity.__new__(cls, center, hradius, vradius, **kwargs)

    def _svg(self, scale_factor=1.0, fill_color='#66cc99'):
        if False:
            i = 10
            return i + 15
        'Returns SVG ellipse element for the Ellipse.\n\n        Parameters\n        ==========\n\n        scale_factor : float\n            Multiplication factor for the SVG stroke-width.  Default is 1.\n        fill_color : str, optional\n            Hex string for fill color. Default is "#66cc99".\n        '
        c = N(self.center)
        (h, v) = (N(self.hradius), N(self.vradius))
        return '<ellipse fill="{1}" stroke="#555555" stroke-width="{0}" opacity="0.6" cx="{2}" cy="{3}" rx="{4}" ry="{5}"/>'.format(2.0 * scale_factor, fill_color, c.x, c.y, h, v)

    @property
    def ambient_dimension(self):
        if False:
            return 10
        return 2

    @property
    def apoapsis(self):
        if False:
            return 10
        'The apoapsis of the ellipse.\n\n        The greatest distance between the focus and the contour.\n\n        Returns\n        =======\n\n        apoapsis : number\n\n        See Also\n        ========\n\n        periapsis : Returns shortest distance between foci and contour\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ellipse\n        >>> p1 = Point(0, 0)\n        >>> e1 = Ellipse(p1, 3, 1)\n        >>> e1.apoapsis\n        2*sqrt(2) + 3\n\n        '
        return self.major * (1 + self.eccentricity)

    def arbitrary_point(self, parameter='t'):
        if False:
            while True:
                i = 10
        "A parameterized point on the ellipse.\n\n        Parameters\n        ==========\n\n        parameter : str, optional\n            Default value is 't'.\n\n        Returns\n        =======\n\n        arbitrary_point : Point\n\n        Raises\n        ======\n\n        ValueError\n            When `parameter` already appears in the functions.\n\n        See Also\n        ========\n\n        sympy.geometry.point.Point\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ellipse\n        >>> e1 = Ellipse(Point(0, 0), 3, 2)\n        >>> e1.arbitrary_point()\n        Point2D(3*cos(t), 2*sin(t))\n\n        "
        t = _symbol(parameter, real=True)
        if t.name in (f.name for f in self.free_symbols):
            raise ValueError(filldedent('Symbol %s already appears in object and cannot be used as a parameter.' % t.name))
        return Point(self.center.x + self.hradius * cos(t), self.center.y + self.vradius * sin(t))

    @property
    def area(self):
        if False:
            print('Hello World!')
        'The area of the ellipse.\n\n        Returns\n        =======\n\n        area : number\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ellipse\n        >>> p1 = Point(0, 0)\n        >>> e1 = Ellipse(p1, 3, 1)\n        >>> e1.area\n        3*pi\n\n        '
        return simplify(S.Pi * self.hradius * self.vradius)

    @property
    def bounds(self):
        if False:
            while True:
                i = 10
        'Return a tuple (xmin, ymin, xmax, ymax) representing the bounding\n        rectangle for the geometric figure.\n\n        '
        (h, v) = (self.hradius, self.vradius)
        return (self.center.x - h, self.center.y - v, self.center.x + h, self.center.y + v)

    @property
    def center(self):
        if False:
            print('Hello World!')
        'The center of the ellipse.\n\n        Returns\n        =======\n\n        center : number\n\n        See Also\n        ========\n\n        sympy.geometry.point.Point\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ellipse\n        >>> p1 = Point(0, 0)\n        >>> e1 = Ellipse(p1, 3, 1)\n        >>> e1.center\n        Point2D(0, 0)\n\n        '
        return self.args[0]

    @property
    def circumference(self):
        if False:
            i = 10
            return i + 15
        'The circumference of the ellipse.\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ellipse\n        >>> p1 = Point(0, 0)\n        >>> e1 = Ellipse(p1, 3, 1)\n        >>> e1.circumference\n        12*elliptic_e(8/9)\n\n        '
        if self.eccentricity == 1:
            return 4 * self.major
        elif self.eccentricity == 0:
            return 2 * pi * self.hradius
        else:
            return 4 * self.major * elliptic_e(self.eccentricity ** 2)

    @property
    def eccentricity(self):
        if False:
            print('Hello World!')
        'The eccentricity of the ellipse.\n\n        Returns\n        =======\n\n        eccentricity : number\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ellipse, sqrt\n        >>> p1 = Point(0, 0)\n        >>> e1 = Ellipse(p1, 3, sqrt(2))\n        >>> e1.eccentricity\n        sqrt(7)/3\n\n        '
        return self.focus_distance / self.major

    def encloses_point(self, p):
        if False:
            i = 10
            return i + 15
        '\n        Return True if p is enclosed by (is inside of) self.\n\n        Notes\n        -----\n        Being on the border of self is considered False.\n\n        Parameters\n        ==========\n\n        p : Point\n\n        Returns\n        =======\n\n        encloses_point : True, False or None\n\n        See Also\n        ========\n\n        sympy.geometry.point.Point\n\n        Examples\n        ========\n\n        >>> from sympy import Ellipse, S\n        >>> from sympy.abc import t\n        >>> e = Ellipse((0, 0), 3, 2)\n        >>> e.encloses_point((0, 0))\n        True\n        >>> e.encloses_point(e.arbitrary_point(t).subs(t, S.Half))\n        False\n        >>> e.encloses_point((4, 0))\n        False\n\n        '
        p = Point(p, dim=2)
        if p in self:
            return False
        if len(self.foci) == 2:
            (h1, h2) = [f.distance(p) for f in self.foci]
            test = 2 * self.major - (h1 + h2)
        else:
            test = self.radius - self.center.distance(p)
        return fuzzy_bool(test.is_positive)

    def equation(self, x='x', y='y', _slope=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns the equation of an ellipse aligned with the x and y axes;\n        when slope is given, the equation returned corresponds to an ellipse\n        with a major axis having that slope.\n\n        Parameters\n        ==========\n\n        x : str, optional\n            Label for the x-axis. Default value is 'x'.\n        y : str, optional\n            Label for the y-axis. Default value is 'y'.\n        _slope : Expr, optional\n                The slope of the major axis. Ignored when 'None'.\n\n        Returns\n        =======\n\n        equation : SymPy expression\n\n        See Also\n        ========\n\n        arbitrary_point : Returns parameterized point on ellipse\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ellipse, pi\n        >>> from sympy.abc import x, y\n        >>> e1 = Ellipse(Point(1, 0), 3, 2)\n        >>> eq1 = e1.equation(x, y); eq1\n        y**2/4 + (x/3 - 1/3)**2 - 1\n        >>> eq2 = e1.equation(x, y, _slope=1); eq2\n        (-x + y + 1)**2/8 + (x + y - 1)**2/18 - 1\n\n        A point on e1 satisfies eq1. Let's use one on the x-axis:\n\n        >>> p1 = e1.center + Point(e1.major, 0)\n        >>> assert eq1.subs(x, p1.x).subs(y, p1.y) == 0\n\n        When rotated the same as the rotated ellipse, about the center\n        point of the ellipse, it will satisfy the rotated ellipse's\n        equation, too:\n\n        >>> r1 = p1.rotate(pi/4, e1.center)\n        >>> assert eq2.subs(x, r1.x).subs(y, r1.y) == 0\n\n        References\n        ==========\n\n        .. [1] https://math.stackexchange.com/questions/108270/what-is-the-equation-of-an-ellipse-that-is-not-aligned-with-the-axis\n        .. [2] https://en.wikipedia.org/wiki/Ellipse#Shifted_ellipse\n\n        "
        x = _symbol(x, real=True)
        y = _symbol(y, real=True)
        dx = x - self.center.x
        dy = y - self.center.y
        if _slope is not None:
            L = (dy - _slope * dx) ** 2
            l = (_slope * dy + dx) ** 2
            h = 1 + _slope ** 2
            b = h * self.major ** 2
            a = h * self.minor ** 2
            return l / b + L / a - 1
        else:
            t1 = (dx / self.hradius) ** 2
            t2 = (dy / self.vradius) ** 2
            return t1 + t2 - 1

    def evolute(self, x='x', y='y'):
        if False:
            for i in range(10):
                print('nop')
        "The equation of evolute of the ellipse.\n\n        Parameters\n        ==========\n\n        x : str, optional\n            Label for the x-axis. Default value is 'x'.\n        y : str, optional\n            Label for the y-axis. Default value is 'y'.\n\n        Returns\n        =======\n\n        equation : SymPy expression\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ellipse\n        >>> e1 = Ellipse(Point(1, 0), 3, 2)\n        >>> e1.evolute()\n        2**(2/3)*y**(2/3) + (3*x - 3)**(2/3) - 5**(2/3)\n        "
        if len(self.args) != 3:
            raise NotImplementedError('Evolute of arbitrary Ellipse is not supported.')
        x = _symbol(x, real=True)
        y = _symbol(y, real=True)
        t1 = (self.hradius * (x - self.center.x)) ** Rational(2, 3)
        t2 = (self.vradius * (y - self.center.y)) ** Rational(2, 3)
        return t1 + t2 - (self.hradius ** 2 - self.vradius ** 2) ** Rational(2, 3)

    @property
    def foci(self):
        if False:
            return 10
        'The foci of the ellipse.\n\n        Notes\n        -----\n        The foci can only be calculated if the major/minor axes are known.\n\n        Raises\n        ======\n\n        ValueError\n            When the major and minor axis cannot be determined.\n\n        See Also\n        ========\n\n        sympy.geometry.point.Point\n        focus_distance : Returns the distance between focus and center\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ellipse\n        >>> p1 = Point(0, 0)\n        >>> e1 = Ellipse(p1, 3, 1)\n        >>> e1.foci\n        (Point2D(-2*sqrt(2), 0), Point2D(2*sqrt(2), 0))\n\n        '
        c = self.center
        (hr, vr) = (self.hradius, self.vradius)
        if hr == vr:
            return (c, c)
        fd = sqrt(self.major ** 2 - self.minor ** 2)
        if hr == self.minor:
            return (c + Point(0, -fd), c + Point(0, fd))
        elif hr == self.major:
            return (c + Point(-fd, 0), c + Point(fd, 0))

    @property
    def focus_distance(self):
        if False:
            print('Hello World!')
        'The focal distance of the ellipse.\n\n        The distance between the center and one focus.\n\n        Returns\n        =======\n\n        focus_distance : number\n\n        See Also\n        ========\n\n        foci\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ellipse\n        >>> p1 = Point(0, 0)\n        >>> e1 = Ellipse(p1, 3, 1)\n        >>> e1.focus_distance\n        2*sqrt(2)\n\n        '
        return Point.distance(self.center, self.foci[0])

    @property
    def hradius(self):
        if False:
            return 10
        'The horizontal radius of the ellipse.\n\n        Returns\n        =======\n\n        hradius : number\n\n        See Also\n        ========\n\n        vradius, major, minor\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ellipse\n        >>> p1 = Point(0, 0)\n        >>> e1 = Ellipse(p1, 3, 1)\n        >>> e1.hradius\n        3\n\n        '
        return self.args[1]

    def intersection(self, o):
        if False:
            print('Hello World!')
        'The intersection of this ellipse and another geometrical entity\n        `o`.\n\n        Parameters\n        ==========\n\n        o : GeometryEntity\n\n        Returns\n        =======\n\n        intersection : list of GeometryEntity objects\n\n        Notes\n        -----\n        Currently supports intersections with Point, Line, Segment, Ray,\n        Circle and Ellipse types.\n\n        See Also\n        ========\n\n        sympy.geometry.entity.GeometryEntity\n\n        Examples\n        ========\n\n        >>> from sympy import Ellipse, Point, Line\n        >>> e = Ellipse(Point(0, 0), 5, 7)\n        >>> e.intersection(Point(0, 0))\n        []\n        >>> e.intersection(Point(5, 0))\n        [Point2D(5, 0)]\n        >>> e.intersection(Line(Point(0,0), Point(0, 1)))\n        [Point2D(0, -7), Point2D(0, 7)]\n        >>> e.intersection(Line(Point(5,0), Point(5, 1)))\n        [Point2D(5, 0)]\n        >>> e.intersection(Line(Point(6,0), Point(6, 1)))\n        []\n        >>> e = Ellipse(Point(-1, 0), 4, 3)\n        >>> e.intersection(Ellipse(Point(1, 0), 4, 3))\n        [Point2D(0, -3*sqrt(15)/4), Point2D(0, 3*sqrt(15)/4)]\n        >>> e.intersection(Ellipse(Point(5, 0), 4, 3))\n        [Point2D(2, -3*sqrt(7)/4), Point2D(2, 3*sqrt(7)/4)]\n        >>> e.intersection(Ellipse(Point(100500, 0), 4, 3))\n        []\n        >>> e.intersection(Ellipse(Point(0, 0), 3, 4))\n        [Point2D(3, 0), Point2D(-363/175, -48*sqrt(111)/175), Point2D(-363/175, 48*sqrt(111)/175)]\n        >>> e.intersection(Ellipse(Point(-1, 0), 3, 4))\n        [Point2D(-17/5, -12/5), Point2D(-17/5, 12/5), Point2D(7/5, -12/5), Point2D(7/5, 12/5)]\n        '
        if isinstance(o, Point):
            if o in self:
                return [o]
            else:
                return []
        elif isinstance(o, (Segment2D, Ray2D)):
            ellipse_equation = self.equation(x, y)
            result = solve([ellipse_equation, Line(o.points[0], o.points[1]).equation(x, y)], [x, y], set=True)[1]
            return list(ordered([Point(i) for i in result if i in o]))
        elif isinstance(o, Polygon):
            return o.intersection(self)
        elif isinstance(o, (Ellipse, Line2D)):
            if o == self:
                return self
            else:
                ellipse_equation = self.equation(x, y)
                return list(ordered([Point(i) for i in solve([ellipse_equation, o.equation(x, y)], [x, y], set=True)[1]]))
        elif isinstance(o, LinearEntity3D):
            raise TypeError('Entity must be two dimensional, not three dimensional')
        else:
            raise TypeError('Intersection not handled for %s' % func_name(o))

    def is_tangent(self, o):
        if False:
            while True:
                i = 10
        'Is `o` tangent to the ellipse?\n\n        Parameters\n        ==========\n\n        o : GeometryEntity\n            An Ellipse, LinearEntity or Polygon\n\n        Raises\n        ======\n\n        NotImplementedError\n            When the wrong type of argument is supplied.\n\n        Returns\n        =======\n\n        is_tangent: boolean\n            True if o is tangent to the ellipse, False otherwise.\n\n        See Also\n        ========\n\n        tangent_lines\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ellipse, Line\n        >>> p0, p1, p2 = Point(0, 0), Point(3, 0), Point(3, 3)\n        >>> e1 = Ellipse(p0, 3, 2)\n        >>> l1 = Line(p1, p2)\n        >>> e1.is_tangent(l1)\n        True\n\n        '
        if isinstance(o, Point2D):
            return False
        elif isinstance(o, Ellipse):
            intersect = self.intersection(o)
            if isinstance(intersect, Ellipse):
                return True
            elif intersect:
                return all((self.tangent_lines(i)[0].equals(o.tangent_lines(i)[0]) for i in intersect))
            else:
                return False
        elif isinstance(o, Line2D):
            hit = self.intersection(o)
            if not hit:
                return False
            if len(hit) == 1:
                return True
            return hit[0].equals(hit[1])
        elif isinstance(o, Ray2D):
            intersect = self.intersection(o)
            if len(intersect) == 1:
                return intersect[0] != o.source and (not self.encloses_point(o.source))
            else:
                return False
        elif isinstance(o, (Segment2D, Polygon)):
            all_tangents = False
            segments = o.sides if isinstance(o, Polygon) else [o]
            for segment in segments:
                intersect = self.intersection(segment)
                if len(intersect) == 1:
                    if not any((intersect[0] in i for i in segment.points)) and (not any((self.encloses_point(i) for i in segment.points))):
                        all_tangents = True
                        continue
                    else:
                        return False
                else:
                    return False
            return all_tangents
        elif isinstance(o, (LinearEntity3D, Point3D)):
            raise TypeError('Entity must be two dimensional, not three dimensional')
        else:
            raise TypeError('Is_tangent not handled for %s' % func_name(o))

    @property
    def major(self):
        if False:
            i = 10
            return i + 15
        "Longer axis of the ellipse (if it can be determined) else hradius.\n\n        Returns\n        =======\n\n        major : number or expression\n\n        See Also\n        ========\n\n        hradius, vradius, minor\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ellipse, Symbol\n        >>> p1 = Point(0, 0)\n        >>> e1 = Ellipse(p1, 3, 1)\n        >>> e1.major\n        3\n\n        >>> a = Symbol('a')\n        >>> b = Symbol('b')\n        >>> Ellipse(p1, a, b).major\n        a\n        >>> Ellipse(p1, b, a).major\n        b\n\n        >>> m = Symbol('m')\n        >>> M = m + 1\n        >>> Ellipse(p1, m, M).major\n        m + 1\n\n        "
        ab = self.args[1:3]
        if len(ab) == 1:
            return ab[0]
        (a, b) = ab
        o = b - a < 0
        if o == True:
            return a
        elif o == False:
            return b
        return self.hradius

    @property
    def minor(self):
        if False:
            print('Hello World!')
        "Shorter axis of the ellipse (if it can be determined) else vradius.\n\n        Returns\n        =======\n\n        minor : number or expression\n\n        See Also\n        ========\n\n        hradius, vradius, major\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ellipse, Symbol\n        >>> p1 = Point(0, 0)\n        >>> e1 = Ellipse(p1, 3, 1)\n        >>> e1.minor\n        1\n\n        >>> a = Symbol('a')\n        >>> b = Symbol('b')\n        >>> Ellipse(p1, a, b).minor\n        b\n        >>> Ellipse(p1, b, a).minor\n        a\n\n        >>> m = Symbol('m')\n        >>> M = m + 1\n        >>> Ellipse(p1, m, M).minor\n        m\n\n        "
        ab = self.args[1:3]
        if len(ab) == 1:
            return ab[0]
        (a, b) = ab
        o = a - b < 0
        if o == True:
            return a
        elif o == False:
            return b
        return self.vradius

    def normal_lines(self, p, prec=None):
        if False:
            print('Hello World!')
        'Normal lines between `p` and the ellipse.\n\n        Parameters\n        ==========\n\n        p : Point\n\n        Returns\n        =======\n\n        normal_lines : list with 1, 2 or 4 Lines\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ellipse\n        >>> e = Ellipse((0, 0), 2, 3)\n        >>> c = e.center\n        >>> e.normal_lines(c + Point(1, 0))\n        [Line2D(Point2D(0, 0), Point2D(1, 0))]\n        >>> e.normal_lines(c)\n        [Line2D(Point2D(0, 0), Point2D(0, 1)), Line2D(Point2D(0, 0), Point2D(1, 0))]\n\n        Off-axis points require the solution of a quartic equation. This\n        often leads to very large expressions that may be of little practical\n        use. An approximate solution of `prec` digits can be obtained by\n        passing in the desired value:\n\n        >>> e.normal_lines((3, 3), prec=2)\n        [Line2D(Point2D(-0.81, -2.7), Point2D(0.19, -1.2)),\n        Line2D(Point2D(1.5, -2.0), Point2D(2.5, -2.7))]\n\n        Whereas the above solution has an operation count of 12, the exact\n        solution has an operation count of 2020.\n        '
        p = Point(p, dim=2)
        if True:
            rv = []
            if p.x == self.center.x:
                rv.append(Line(self.center, slope=oo))
            if p.y == self.center.y:
                rv.append(Line(self.center, slope=0))
            if rv:
                return rv
        eq = self.equation(x, y)
        dydx = idiff(eq, y, x)
        norm = -1 / dydx
        slope = Line(p, (x, y)).slope
        seq = slope - norm
        yis = solve(seq, y)[0]
        xeq = eq.subs(y, yis).as_numer_denom()[0].expand()
        if len(xeq.free_symbols) == 1:
            try:
                xsol = Poly(xeq, x).real_roots()
            except (DomainError, PolynomialError, NotImplementedError):
                xsol = _nsort(solve(xeq, x), separated=True)[0]
            points = [Point(i, solve(eq.subs(x, i), y)[0]) for i in xsol]
        else:
            raise NotImplementedError('intersections for the general ellipse are not supported')
        slopes = [norm.subs(zip((x, y), pt.args)) for pt in points]
        if prec is not None:
            points = [pt.n(prec) for pt in points]
            slopes = [i if _not_a_coeff(i) else i.n(prec) for i in slopes]
        return [Line(pt, slope=s) for (pt, s) in zip(points, slopes)]

    @property
    def periapsis(self):
        if False:
            i = 10
            return i + 15
        'The periapsis of the ellipse.\n\n        The shortest distance between the focus and the contour.\n\n        Returns\n        =======\n\n        periapsis : number\n\n        See Also\n        ========\n\n        apoapsis : Returns greatest distance between focus and contour\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ellipse\n        >>> p1 = Point(0, 0)\n        >>> e1 = Ellipse(p1, 3, 1)\n        >>> e1.periapsis\n        3 - 2*sqrt(2)\n\n        '
        return self.major * (1 - self.eccentricity)

    @property
    def semilatus_rectum(self):
        if False:
            i = 10
            return i + 15
        '\n        Calculates the semi-latus rectum of the Ellipse.\n\n        Semi-latus rectum is defined as one half of the chord through a\n        focus parallel to the conic section directrix of a conic section.\n\n        Returns\n        =======\n\n        semilatus_rectum : number\n\n        See Also\n        ========\n\n        apoapsis : Returns greatest distance between focus and contour\n\n        periapsis : The shortest distance between the focus and the contour\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ellipse\n        >>> p1 = Point(0, 0)\n        >>> e1 = Ellipse(p1, 3, 1)\n        >>> e1.semilatus_rectum\n        1/3\n\n        References\n        ==========\n\n        .. [1] https://mathworld.wolfram.com/SemilatusRectum.html\n        .. [2] https://en.wikipedia.org/wiki/Ellipse#Semi-latus_rectum\n\n        '
        return self.major * (1 - self.eccentricity ** 2)

    def auxiliary_circle(self):
        if False:
            return 10
        "Returns a Circle whose diameter is the major axis of the ellipse.\n\n        Examples\n        ========\n\n        >>> from sympy import Ellipse, Point, symbols\n        >>> c = Point(1, 2)\n        >>> Ellipse(c, 8, 7).auxiliary_circle()\n        Circle(Point2D(1, 2), 8)\n        >>> a, b = symbols('a b')\n        >>> Ellipse(c, a, b).auxiliary_circle()\n        Circle(Point2D(1, 2), Max(a, b))\n        "
        return Circle(self.center, Max(self.hradius, self.vradius))

    def director_circle(self):
        if False:
            while True:
                i = 10
        "\n        Returns a Circle consisting of all points where two perpendicular\n        tangent lines to the ellipse cross each other.\n\n        Returns\n        =======\n\n        Circle\n            A director circle returned as a geometric object.\n\n        Examples\n        ========\n\n        >>> from sympy import Ellipse, Point, symbols\n        >>> c = Point(3,8)\n        >>> Ellipse(c, 7, 9).director_circle()\n        Circle(Point2D(3, 8), sqrt(130))\n        >>> a, b = symbols('a b')\n        >>> Ellipse(c, a, b).director_circle()\n        Circle(Point2D(3, 8), sqrt(a**2 + b**2))\n\n        References\n        ==========\n\n        .. [1] https://en.wikipedia.org/wiki/Director_circle\n\n        "
        return Circle(self.center, sqrt(self.hradius ** 2 + self.vradius ** 2))

    def plot_interval(self, parameter='t'):
        if False:
            for i in range(10):
                print('nop')
        "The plot interval for the default geometric plot of the Ellipse.\n\n        Parameters\n        ==========\n\n        parameter : str, optional\n            Default value is 't'.\n\n        Returns\n        =======\n\n        plot_interval : list\n            [parameter, lower_bound, upper_bound]\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ellipse\n        >>> e1 = Ellipse(Point(0, 0), 3, 2)\n        >>> e1.plot_interval()\n        [t, -pi, pi]\n\n        "
        t = _symbol(parameter, real=True)
        return [t, -S.Pi, S.Pi]

    def random_point(self, seed=None):
        if False:
            return 10
        "A random point on the ellipse.\n\n        Returns\n        =======\n\n        point : Point\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ellipse\n        >>> e1 = Ellipse(Point(0, 0), 3, 2)\n        >>> e1.random_point() # gives some random point\n        Point2D(...)\n        >>> p1 = e1.random_point(seed=0); p1.n(2)\n        Point2D(2.1, 1.4)\n\n        Notes\n        =====\n\n        When creating a random point, one may simply replace the\n        parameter with a random number. When doing so, however, the\n        random number should be made a Rational or else the point\n        may not test as being in the ellipse:\n\n        >>> from sympy.abc import t\n        >>> from sympy import Rational\n        >>> arb = e1.arbitrary_point(t); arb\n        Point2D(3*cos(t), 2*sin(t))\n        >>> arb.subs(t, .1) in e1\n        False\n        >>> arb.subs(t, Rational(.1)) in e1\n        True\n        >>> arb.subs(t, Rational('.1')) in e1\n        True\n\n        See Also\n        ========\n        sympy.geometry.point.Point\n        arbitrary_point : Returns parameterized point on ellipse\n        "
        t = _symbol('t', real=True)
        (x, y) = self.arbitrary_point(t).args
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        r = Rational(rng.random())
        c = 2 * r - 1
        s = sqrt(1 - c ** 2)
        return Point(x.subs(cos(t), c), y.subs(sin(t), s))

    def reflect(self, line):
        if False:
            return 10
        'Override GeometryEntity.reflect since the radius\n        is not a GeometryEntity.\n\n        Examples\n        ========\n\n        >>> from sympy import Circle, Line\n        >>> Circle((0, 1), 1).reflect(Line((0, 0), (1, 1)))\n        Circle(Point2D(1, 0), -1)\n        >>> from sympy import Ellipse, Line, Point\n        >>> Ellipse(Point(3, 4), 1, 3).reflect(Line(Point(0, -4), Point(5, 0)))\n        Traceback (most recent call last):\n        ...\n        NotImplementedError:\n        General Ellipse is not supported but the equation of the reflected\n        Ellipse is given by the zeros of: f(x, y) = (9*x/41 + 40*y/41 +\n        37/41)**2 + (40*x/123 - 3*y/41 - 364/123)**2 - 1\n\n        Notes\n        =====\n\n        Until the general ellipse (with no axis parallel to the x-axis) is\n        supported a NotImplemented error is raised and the equation whose\n        zeros define the rotated ellipse is given.\n\n        '
        if line.slope in (0, oo):
            c = self.center
            c = c.reflect(line)
            return self.func(c, -self.hradius, self.vradius)
        else:
            (x, y) = [uniquely_named_symbol(name, (self, line), modify=lambda s: '_' + s, real=True) for name in 'xy']
            expr = self.equation(x, y)
            p = Point(x, y).reflect(line)
            result = expr.subs(zip((x, y), p.args), simultaneous=True)
            raise NotImplementedError(filldedent('General Ellipse is not supported but the equation of the reflected Ellipse is given by the zeros of: ' + 'f(%s, %s) = %s' % (str(x), str(y), str(result))))

    def rotate(self, angle=0, pt=None):
        if False:
            while True:
                i = 10
        'Rotate ``angle`` radians counterclockwise about Point ``pt``.\n\n        Note: since the general ellipse is not supported, only rotations that\n        are integer multiples of pi/2 are allowed.\n\n        Examples\n        ========\n\n        >>> from sympy import Ellipse, pi\n        >>> Ellipse((1, 0), 2, 1).rotate(pi/2)\n        Ellipse(Point2D(0, 1), 1, 2)\n        >>> Ellipse((1, 0), 2, 1).rotate(pi)\n        Ellipse(Point2D(-1, 0), 2, 1)\n        '
        if self.hradius == self.vradius:
            return self.func(self.center.rotate(angle, pt), self.hradius)
        if (angle / S.Pi).is_integer:
            return super().rotate(angle, pt)
        if (2 * angle / S.Pi).is_integer:
            return self.func(self.center.rotate(angle, pt), self.vradius, self.hradius)
        raise NotImplementedError('Only rotations of pi/2 are currently supported for Ellipse.')

    def scale(self, x=1, y=1, pt=None):
        if False:
            while True:
                i = 10
        'Override GeometryEntity.scale since it is the major and minor\n        axes which must be scaled and they are not GeometryEntities.\n\n        Examples\n        ========\n\n        >>> from sympy import Ellipse\n        >>> Ellipse((0, 0), 2, 1).scale(2, 4)\n        Circle(Point2D(0, 0), 4)\n        >>> Ellipse((0, 0), 2, 1).scale(2)\n        Ellipse(Point2D(0, 0), 4, 1)\n        '
        c = self.center
        if pt:
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        h = self.hradius
        v = self.vradius
        return self.func(c.scale(x, y), hradius=h * x, vradius=v * y)

    def tangent_lines(self, p):
        if False:
            return 10
        'Tangent lines between `p` and the ellipse.\n\n        If `p` is on the ellipse, returns the tangent line through point `p`.\n        Otherwise, returns the tangent line(s) from `p` to the ellipse, or\n        None if no tangent line is possible (e.g., `p` inside ellipse).\n\n        Parameters\n        ==========\n\n        p : Point\n\n        Returns\n        =======\n\n        tangent_lines : list with 1 or 2 Lines\n\n        Raises\n        ======\n\n        NotImplementedError\n            Can only find tangent lines for a point, `p`, on the ellipse.\n\n        See Also\n        ========\n\n        sympy.geometry.point.Point, sympy.geometry.line.Line\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ellipse\n        >>> e1 = Ellipse(Point(0, 0), 3, 2)\n        >>> e1.tangent_lines(Point(3, 0))\n        [Line2D(Point2D(3, 0), Point2D(3, -12))]\n\n        '
        p = Point(p, dim=2)
        if self.encloses_point(p):
            return []
        if p in self:
            delta = self.center - p
            rise = self.vradius ** 2 * delta.x
            run = -self.hradius ** 2 * delta.y
            p2 = Point(simplify(p.x + run), simplify(p.y + rise))
            return [Line(p, p2)]
        else:
            if len(self.foci) == 2:
                (f1, f2) = self.foci
                maj = self.hradius
                test = 2 * maj - Point.distance(f1, p) - Point.distance(f2, p)
            else:
                test = self.radius - Point.distance(self.center, p)
            if test.is_number and test.is_positive:
                return []
            eq = self.equation(x, y)
            dydx = idiff(eq, y, x)
            slope = Line(p, Point(x, y)).slope
            tangent_points = solve([slope - dydx, eq], [x, y])
            if len(tangent_points) == 1:
                if tangent_points[0][0] == p.x or tangent_points[0][1] == p.y:
                    return [Line(p, p + Point(1, 0)), Line(p, p + Point(0, 1))]
                else:
                    return [Line(p, p + Point(0, 1)), Line(p, tangent_points[0])]
            return [Line(p, tangent_points[0]), Line(p, tangent_points[1])]

    @property
    def vradius(self):
        if False:
            return 10
        'The vertical radius of the ellipse.\n\n        Returns\n        =======\n\n        vradius : number\n\n        See Also\n        ========\n\n        hradius, major, minor\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ellipse\n        >>> p1 = Point(0, 0)\n        >>> e1 = Ellipse(p1, 3, 1)\n        >>> e1.vradius\n        1\n\n        '
        return self.args[2]

    def second_moment_of_area(self, point=None):
        if False:
            return 10
        'Returns the second moment and product moment area of an ellipse.\n\n        Parameters\n        ==========\n\n        point : Point, two-tuple of sympifiable objects, or None(default=None)\n            point is the point about which second moment of area is to be found.\n            If "point=None" it will be calculated about the axis passing through the\n            centroid of the ellipse.\n\n        Returns\n        =======\n\n        I_xx, I_yy, I_xy : number or SymPy expression\n            I_xx, I_yy are second moment of area of an ellise.\n            I_xy is product moment of area of an ellipse.\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Ellipse\n        >>> p1 = Point(0, 0)\n        >>> e1 = Ellipse(p1, 3, 1)\n        >>> e1.second_moment_of_area()\n        (3*pi/4, 27*pi/4, 0)\n\n        References\n        ==========\n\n        .. [1] https://en.wikipedia.org/wiki/List_of_second_moments_of_area\n\n        '
        I_xx = S.Pi * self.hradius * self.vradius ** 3 / 4
        I_yy = S.Pi * self.hradius ** 3 * self.vradius / 4
        I_xy = 0
        if point is None:
            return (I_xx, I_yy, I_xy)
        I_xx = I_xx + self.area * (point[1] - self.center.y) ** 2
        I_yy = I_yy + self.area * (point[0] - self.center.x) ** 2
        I_xy = I_xy + self.area * (point[0] - self.center.x) * (point[1] - self.center.y)
        return (I_xx, I_yy, I_xy)

    def polar_second_moment_of_area(self):
        if False:
            return 10
        "Returns the polar second moment of area of an Ellipse\n\n        It is a constituent of the second moment of area, linked through\n        the perpendicular axis theorem. While the planar second moment of\n        area describes an object's resistance to deflection (bending) when\n        subjected to a force applied to a plane parallel to the central\n        axis, the polar second moment of area describes an object's\n        resistance to deflection when subjected to a moment applied in a\n        plane perpendicular to the object's central axis (i.e. parallel to\n        the cross-section)\n\n        Examples\n        ========\n\n        >>> from sympy import symbols, Circle, Ellipse\n        >>> c = Circle((5, 5), 4)\n        >>> c.polar_second_moment_of_area()\n        128*pi\n        >>> a, b = symbols('a, b')\n        >>> e = Ellipse((0, 0), a, b)\n        >>> e.polar_second_moment_of_area()\n        pi*a**3*b/4 + pi*a*b**3/4\n\n        References\n        ==========\n\n        .. [1] https://en.wikipedia.org/wiki/Polar_moment_of_inertia\n\n        "
        second_moment = self.second_moment_of_area()
        return second_moment[0] + second_moment[1]

    def section_modulus(self, point=None):
        if False:
            print('Hello World!')
        'Returns a tuple with the section modulus of an ellipse\n\n        Section modulus is a geometric property of an ellipse defined as the\n        ratio of second moment of area to the distance of the extreme end of\n        the ellipse from the centroidal axis.\n\n        Parameters\n        ==========\n\n        point : Point, two-tuple of sympifyable objects, or None(default=None)\n            point is the point at which section modulus is to be found.\n            If "point=None" section modulus will be calculated for the\n            point farthest from the centroidal axis of the ellipse.\n\n        Returns\n        =======\n\n        S_x, S_y: numbers or SymPy expressions\n                  S_x is the section modulus with respect to the x-axis\n                  S_y is the section modulus with respect to the y-axis\n                  A negative sign indicates that the section modulus is\n                  determined for a point below the centroidal axis.\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol, Ellipse, Circle, Point2D\n        >>> d = Symbol(\'d\', positive=True)\n        >>> c = Circle((0, 0), d/2)\n        >>> c.section_modulus()\n        (pi*d**3/32, pi*d**3/32)\n        >>> e = Ellipse(Point2D(0, 0), 2, 4)\n        >>> e.section_modulus()\n        (8*pi, 4*pi)\n        >>> e.section_modulus((2, 2))\n        (16*pi, 4*pi)\n\n        References\n        ==========\n\n        .. [1] https://en.wikipedia.org/wiki/Section_modulus\n\n        '
        (x_c, y_c) = self.center
        if point is None:
            (x_min, y_min, x_max, y_max) = self.bounds
            y = max(y_c - y_min, y_max - y_c)
            x = max(x_c - x_min, x_max - x_c)
        else:
            point = Point2D(point)
            y = point.y - y_c
            x = point.x - x_c
        second_moment = self.second_moment_of_area()
        S_x = second_moment[0] / y
        S_y = second_moment[1] / x
        return (S_x, S_y)

class Circle(Ellipse):
    """A circle in space.

    Constructed simply from a center and a radius, from three
    non-collinear points, or the equation of a circle.

    Parameters
    ==========

    center : Point
    radius : number or SymPy expression
    points : sequence of three Points
    equation : equation of a circle

    Attributes
    ==========

    radius (synonymous with hradius, vradius, major and minor)
    circumference
    equation

    Raises
    ======

    GeometryError
        When the given equation is not that of a circle.
        When trying to construct circle from incorrect parameters.

    See Also
    ========

    Ellipse, sympy.geometry.point.Point

    Examples
    ========

    >>> from sympy import Point, Circle, Eq
    >>> from sympy.abc import x, y, a, b

    A circle constructed from a center and radius:

    >>> c1 = Circle(Point(0, 0), 5)
    >>> c1.hradius, c1.vradius, c1.radius
    (5, 5, 5)

    A circle constructed from three points:

    >>> c2 = Circle(Point(0, 0), Point(1, 1), Point(1, 0))
    >>> c2.hradius, c2.vradius, c2.radius, c2.center
    (sqrt(2)/2, sqrt(2)/2, sqrt(2)/2, Point2D(1/2, 1/2))

    A circle can be constructed from an equation in the form
    `a*x**2 + by**2 + gx + hy + c = 0`, too:

    >>> Circle(x**2 + y**2 - 25)
    Circle(Point2D(0, 0), 5)

    If the variables corresponding to x and y are named something
    else, their name or symbol can be supplied:

    >>> Circle(Eq(a**2 + b**2, 25), x='a', y=b)
    Circle(Point2D(0, 0), 5)
    """

    def __new__(cls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
        if len(args) == 1 and isinstance(args[0], (Expr, Eq)):
            x = kwargs.get('x', 'x')
            y = kwargs.get('y', 'y')
            equation = args[0].expand()
            if isinstance(equation, Eq):
                equation = equation.lhs - equation.rhs
            x = find(x, equation)
            y = find(y, equation)
            try:
                (a, b, c, d, e) = linear_coeffs(equation, x ** 2, y ** 2, x, y)
            except ValueError:
                raise GeometryError('The given equation is not that of a circle.')
            if S.Zero in (a, b) or a != b:
                raise GeometryError('The given equation is not that of a circle.')
            center_x = -c / a / 2
            center_y = -d / b / 2
            r2 = center_x ** 2 + center_y ** 2 - e / a
            return Circle((center_x, center_y), sqrt(r2), evaluate=evaluate)
        else:
            (c, r) = (None, None)
            if len(args) == 3:
                args = [Point(a, dim=2, evaluate=evaluate) for a in args]
                t = Triangle(*args)
                if not isinstance(t, Triangle):
                    return t
                c = t.circumcenter
                r = t.circumradius
            elif len(args) == 2:
                c = Point(args[0], dim=2, evaluate=evaluate)
                r = args[1]
                try:
                    r = Point(r, 0, evaluate=evaluate).x
                except ValueError:
                    raise GeometryError('Circle with imaginary radius is not permitted')
            if not (c is None or r is None):
                if r == 0:
                    return c
                return GeometryEntity.__new__(cls, c, r, **kwargs)
            raise GeometryError('Circle.__new__ received unknown arguments')

    def _eval_evalf(self, prec=15, **options):
        if False:
            return 10
        (pt, r) = self.args
        dps = prec_to_dps(prec)
        pt = pt.evalf(n=dps, **options)
        r = r.evalf(n=dps, **options)
        return self.func(pt, r, evaluate=False)

    @property
    def circumference(self):
        if False:
            i = 10
            return i + 15
        'The circumference of the circle.\n\n        Returns\n        =======\n\n        circumference : number or SymPy expression\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Circle\n        >>> c1 = Circle(Point(3, 4), 6)\n        >>> c1.circumference\n        12*pi\n\n        '
        return 2 * S.Pi * self.radius

    def equation(self, x='x', y='y'):
        if False:
            print('Hello World!')
        "The equation of the circle.\n\n        Parameters\n        ==========\n\n        x : str or Symbol, optional\n            Default value is 'x'.\n        y : str or Symbol, optional\n            Default value is 'y'.\n\n        Returns\n        =======\n\n        equation : SymPy expression\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Circle\n        >>> c1 = Circle(Point(0, 0), 5)\n        >>> c1.equation()\n        x**2 + y**2 - 25\n\n        "
        x = _symbol(x, real=True)
        y = _symbol(y, real=True)
        t1 = (x - self.center.x) ** 2
        t2 = (y - self.center.y) ** 2
        return t1 + t2 - self.major ** 2

    def intersection(self, o):
        if False:
            return 10
        'The intersection of this circle with another geometrical entity.\n\n        Parameters\n        ==========\n\n        o : GeometryEntity\n\n        Returns\n        =======\n\n        intersection : list of GeometryEntities\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Circle, Line, Ray\n        >>> p1, p2, p3 = Point(0, 0), Point(5, 5), Point(6, 0)\n        >>> p4 = Point(5, 0)\n        >>> c1 = Circle(p1, 5)\n        >>> c1.intersection(p2)\n        []\n        >>> c1.intersection(p4)\n        [Point2D(5, 0)]\n        >>> c1.intersection(Ray(p1, p2))\n        [Point2D(5*sqrt(2)/2, 5*sqrt(2)/2)]\n        >>> c1.intersection(Line(p2, p3))\n        []\n\n        '
        return Ellipse.intersection(self, o)

    @property
    def radius(self):
        if False:
            for i in range(10):
                print('nop')
        'The radius of the circle.\n\n        Returns\n        =======\n\n        radius : number or SymPy expression\n\n        See Also\n        ========\n\n        Ellipse.major, Ellipse.minor, Ellipse.hradius, Ellipse.vradius\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Circle\n        >>> c1 = Circle(Point(3, 4), 6)\n        >>> c1.radius\n        6\n\n        '
        return self.args[1]

    def reflect(self, line):
        if False:
            for i in range(10):
                print('nop')
        'Override GeometryEntity.reflect since the radius\n        is not a GeometryEntity.\n\n        Examples\n        ========\n\n        >>> from sympy import Circle, Line\n        >>> Circle((0, 1), 1).reflect(Line((0, 0), (1, 1)))\n        Circle(Point2D(1, 0), -1)\n        '
        c = self.center
        c = c.reflect(line)
        return self.func(c, -self.radius)

    def scale(self, x=1, y=1, pt=None):
        if False:
            while True:
                i = 10
        'Override GeometryEntity.scale since the radius\n        is not a GeometryEntity.\n\n        Examples\n        ========\n\n        >>> from sympy import Circle\n        >>> Circle((0, 0), 1).scale(2, 2)\n        Circle(Point2D(0, 0), 2)\n        >>> Circle((0, 0), 1).scale(2, 4)\n        Ellipse(Point2D(0, 0), 2, 4)\n        '
        c = self.center
        if pt:
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        c = c.scale(x, y)
        (x, y) = [abs(i) for i in (x, y)]
        if x == y:
            return self.func(c, x * self.radius)
        h = v = self.radius
        return Ellipse(c, hradius=h * x, vradius=v * y)

    @property
    def vradius(self):
        if False:
            while True:
                i = 10
        "\n        This Ellipse property is an alias for the Circle's radius.\n\n        Whereas hradius, major and minor can use Ellipse's conventions,\n        the vradius does not exist for a circle. It is always a positive\n        value in order that the Circle, like Polygons, will have an\n        area that can be positive or negative as determined by the sign\n        of the hradius.\n\n        Examples\n        ========\n\n        >>> from sympy import Point, Circle\n        >>> c1 = Circle(Point(3, 4), 6)\n        >>> c1.vradius\n        6\n        "
        return abs(self.radius)
from .polygon import Polygon, Triangle