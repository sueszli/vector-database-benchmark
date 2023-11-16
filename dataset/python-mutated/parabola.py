"""Parabolic geometrical entity.

Contains
* Parabola

"""
from sympy.core import S
from sympy.core.sorting import ordered
from sympy.core.symbol import _symbol, symbols
from sympy.geometry.entity import GeometryEntity, GeometrySet
from sympy.geometry.point import Point, Point2D
from sympy.geometry.line import Line, Line2D, Ray2D, Segment2D, LinearEntity3D
from sympy.geometry.ellipse import Ellipse
from sympy.functions import sign
from sympy.simplify import simplify
from sympy.solvers.solvers import solve

class Parabola(GeometrySet):
    """A parabolic GeometryEntity.

    A parabola is declared with a point, that is called 'focus', and
    a line, that is called 'directrix'.
    Only vertical or horizontal parabolas are currently supported.

    Parameters
    ==========

    focus : Point
        Default value is Point(0, 0)
    directrix : Line

    Attributes
    ==========

    focus
    directrix
    axis of symmetry
    focal length
    p parameter
    vertex
    eccentricity

    Raises
    ======
    ValueError
        When `focus` is not a two dimensional point.
        When `focus` is a point of directrix.
    NotImplementedError
        When `directrix` is neither horizontal nor vertical.

    Examples
    ========

    >>> from sympy import Parabola, Point, Line
    >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7,8)))
    >>> p1.focus
    Point2D(0, 0)
    >>> p1.directrix
    Line2D(Point2D(5, 8), Point2D(7, 8))

    """

    def __new__(cls, focus=None, directrix=None, **kwargs):
        if False:
            print('Hello World!')
        if focus:
            focus = Point(focus, dim=2)
        else:
            focus = Point(0, 0)
        directrix = Line(directrix)
        if directrix.contains(focus):
            raise ValueError('The focus must not be a point of directrix')
        return GeometryEntity.__new__(cls, focus, directrix, **kwargs)

    @property
    def ambient_dimension(self):
        if False:
            i = 10
            return i + 15
        'Returns the ambient dimension of parabola.\n\n        Returns\n        =======\n\n        ambient_dimension : integer\n\n        Examples\n        ========\n\n        >>> from sympy import Parabola, Point, Line\n        >>> f1 = Point(0, 0)\n        >>> p1 = Parabola(f1, Line(Point(5, 8), Point(7, 8)))\n        >>> p1.ambient_dimension\n        2\n\n        '
        return 2

    @property
    def axis_of_symmetry(self):
        if False:
            print('Hello World!')
        'Return the axis of symmetry of the parabola: a line\n        perpendicular to the directrix passing through the focus.\n\n        Returns\n        =======\n\n        axis_of_symmetry : Line\n\n        See Also\n        ========\n\n        sympy.geometry.line.Line\n\n        Examples\n        ========\n\n        >>> from sympy import Parabola, Point, Line\n        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))\n        >>> p1.axis_of_symmetry\n        Line2D(Point2D(0, 0), Point2D(0, 1))\n\n        '
        return self.directrix.perpendicular_line(self.focus)

    @property
    def directrix(self):
        if False:
            print('Hello World!')
        'The directrix of the parabola.\n\n        Returns\n        =======\n\n        directrix : Line\n\n        See Also\n        ========\n\n        sympy.geometry.line.Line\n\n        Examples\n        ========\n\n        >>> from sympy import Parabola, Point, Line\n        >>> l1 = Line(Point(5, 8), Point(7, 8))\n        >>> p1 = Parabola(Point(0, 0), l1)\n        >>> p1.directrix\n        Line2D(Point2D(5, 8), Point2D(7, 8))\n\n        '
        return self.args[1]

    @property
    def eccentricity(self):
        if False:
            print('Hello World!')
        'The eccentricity of the parabola.\n\n        Returns\n        =======\n\n        eccentricity : number\n\n        A parabola may also be characterized as a conic section with an\n        eccentricity of 1. As a consequence of this, all parabolas are\n        similar, meaning that while they can be different sizes,\n        they are all the same shape.\n\n        See Also\n        ========\n\n        https://en.wikipedia.org/wiki/Parabola\n\n\n        Examples\n        ========\n\n        >>> from sympy import Parabola, Point, Line\n        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))\n        >>> p1.eccentricity\n        1\n\n        Notes\n        -----\n        The eccentricity for every Parabola is 1 by definition.\n\n        '
        return S.One

    def equation(self, x='x', y='y'):
        if False:
            while True:
                i = 10
        "The equation of the parabola.\n\n        Parameters\n        ==========\n        x : str, optional\n            Label for the x-axis. Default value is 'x'.\n        y : str, optional\n            Label for the y-axis. Default value is 'y'.\n\n        Returns\n        =======\n        equation : SymPy expression\n\n        Examples\n        ========\n\n        >>> from sympy import Parabola, Point, Line\n        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))\n        >>> p1.equation()\n        -x**2 - 16*y + 64\n        >>> p1.equation('f')\n        -f**2 - 16*y + 64\n        >>> p1.equation(y='z')\n        -x**2 - 16*z + 64\n\n        "
        x = _symbol(x, real=True)
        y = _symbol(y, real=True)
        m = self.directrix.slope
        if m is S.Infinity:
            t1 = 4 * self.p_parameter * (x - self.vertex.x)
            t2 = (y - self.vertex.y) ** 2
        elif m == 0:
            t1 = 4 * self.p_parameter * (y - self.vertex.y)
            t2 = (x - self.vertex.x) ** 2
        else:
            (a, b) = self.focus
            (c, d) = self.directrix.coefficients[:2]
            t1 = (x - a) ** 2 + (y - b) ** 2
            t2 = self.directrix.equation(x, y) ** 2 / (c ** 2 + d ** 2)
        return t1 - t2

    @property
    def focal_length(self):
        if False:
            print('Hello World!')
        'The focal length of the parabola.\n\n        Returns\n        =======\n\n        focal_lenght : number or symbolic expression\n\n        Notes\n        =====\n\n        The distance between the vertex and the focus\n        (or the vertex and directrix), measured along the axis\n        of symmetry, is the "focal length".\n\n        See Also\n        ========\n\n        https://en.wikipedia.org/wiki/Parabola\n\n        Examples\n        ========\n\n        >>> from sympy import Parabola, Point, Line\n        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))\n        >>> p1.focal_length\n        4\n\n        '
        distance = self.directrix.distance(self.focus)
        focal_length = distance / 2
        return focal_length

    @property
    def focus(self):
        if False:
            while True:
                i = 10
        'The focus of the parabola.\n\n        Returns\n        =======\n\n        focus : Point\n\n        See Also\n        ========\n\n        sympy.geometry.point.Point\n\n        Examples\n        ========\n\n        >>> from sympy import Parabola, Point, Line\n        >>> f1 = Point(0, 0)\n        >>> p1 = Parabola(f1, Line(Point(5, 8), Point(7, 8)))\n        >>> p1.focus\n        Point2D(0, 0)\n\n        '
        return self.args[0]

    def intersection(self, o):
        if False:
            for i in range(10):
                print('nop')
        'The intersection of the parabola and another geometrical entity `o`.\n\n        Parameters\n        ==========\n\n        o : GeometryEntity, LinearEntity\n\n        Returns\n        =======\n\n        intersection : list of GeometryEntity objects\n\n        Examples\n        ========\n\n        >>> from sympy import Parabola, Point, Ellipse, Line, Segment\n        >>> p1 = Point(0,0)\n        >>> l1 = Line(Point(1, -2), Point(-1,-2))\n        >>> parabola1 = Parabola(p1, l1)\n        >>> parabola1.intersection(Ellipse(Point(0, 0), 2, 5))\n        [Point2D(-2, 0), Point2D(2, 0)]\n        >>> parabola1.intersection(Line(Point(-7, 3), Point(12, 3)))\n        [Point2D(-4, 3), Point2D(4, 3)]\n        >>> parabola1.intersection(Segment((-12, -65), (14, -68)))\n        []\n\n        '
        (x, y) = symbols('x y', real=True)
        parabola_eq = self.equation()
        if isinstance(o, Parabola):
            if o in self:
                return [o]
            else:
                return list(ordered([Point(i) for i in solve([parabola_eq, o.equation()], [x, y], set=True)[1]]))
        elif isinstance(o, Point2D):
            if simplify(parabola_eq.subs([(x, o._args[0]), (y, o._args[1])])) == 0:
                return [o]
            else:
                return []
        elif isinstance(o, (Segment2D, Ray2D)):
            result = solve([parabola_eq, Line2D(o.points[0], o.points[1]).equation()], [x, y], set=True)[1]
            return list(ordered([Point2D(i) for i in result if i in o]))
        elif isinstance(o, (Line2D, Ellipse)):
            return list(ordered([Point2D(i) for i in solve([parabola_eq, o.equation()], [x, y], set=True)[1]]))
        elif isinstance(o, LinearEntity3D):
            raise TypeError('Entity must be two dimensional, not three dimensional')
        else:
            raise TypeError('Wrong type of argument were put')

    @property
    def p_parameter(self):
        if False:
            for i in range(10):
                print('nop')
        'P is a parameter of parabola.\n\n        Returns\n        =======\n\n        p : number or symbolic expression\n\n        Notes\n        =====\n\n        The absolute value of p is the focal length. The sign on p tells\n        which way the parabola faces. Vertical parabolas that open up\n        and horizontal that open right, give a positive value for p.\n        Vertical parabolas that open down and horizontal that open left,\n        give a negative value for p.\n\n\n        See Also\n        ========\n\n        https://www.sparknotes.com/math/precalc/conicsections/section2/\n\n        Examples\n        ========\n\n        >>> from sympy import Parabola, Point, Line\n        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))\n        >>> p1.p_parameter\n        -4\n\n        '
        m = self.directrix.slope
        if m is S.Infinity:
            x = self.directrix.coefficients[2]
            p = sign(self.focus.args[0] + x)
        elif m == 0:
            y = self.directrix.coefficients[2]
            p = sign(self.focus.args[1] + y)
        else:
            d = self.directrix.projection(self.focus)
            p = sign(self.focus.x - d.x)
        return p * self.focal_length

    @property
    def vertex(self):
        if False:
            for i in range(10):
                print('nop')
        'The vertex of the parabola.\n\n        Returns\n        =======\n\n        vertex : Point\n\n        See Also\n        ========\n\n        sympy.geometry.point.Point\n\n        Examples\n        ========\n\n        >>> from sympy import Parabola, Point, Line\n        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))\n        >>> p1.vertex\n        Point2D(0, 4)\n\n        '
        focus = self.focus
        m = self.directrix.slope
        if m is S.Infinity:
            vertex = Point(focus.args[0] - self.p_parameter, focus.args[1])
        elif m == 0:
            vertex = Point(focus.args[0], focus.args[1] - self.p_parameter)
        else:
            vertex = self.axis_of_symmetry.intersection(self)[0]
        return vertex