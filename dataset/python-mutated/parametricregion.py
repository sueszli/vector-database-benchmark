from functools import singledispatch
from sympy.core.numbers import pi
from sympy.functions.elementary.trigonometric import tan
from sympy.simplify import trigsimp
from sympy.core import Basic, Tuple
from sympy.core.symbol import _symbol
from sympy.solvers import solve
from sympy.geometry import Point, Segment, Curve, Ellipse, Polygon
from sympy.vector import ImplicitRegion

class ParametricRegion(Basic):
    """
    Represents a parametric region in space.

    Examples
    ========

    >>> from sympy import cos, sin, pi
    >>> from sympy.abc import r, theta, t, a, b, x, y
    >>> from sympy.vector import ParametricRegion

    >>> ParametricRegion((t, t**2), (t, -1, 2))
    ParametricRegion((t, t**2), (t, -1, 2))
    >>> ParametricRegion((x, y), (x, 3, 4), (y, 5, 6))
    ParametricRegion((x, y), (x, 3, 4), (y, 5, 6))
    >>> ParametricRegion((r*cos(theta), r*sin(theta)), (r, -2, 2), (theta, 0, pi))
    ParametricRegion((r*cos(theta), r*sin(theta)), (r, -2, 2), (theta, 0, pi))
    >>> ParametricRegion((a*cos(t), b*sin(t)), t)
    ParametricRegion((a*cos(t), b*sin(t)), t)

    >>> circle = ParametricRegion((r*cos(theta), r*sin(theta)), r, (theta, 0, pi))
    >>> circle.parameters
    (r, theta)
    >>> circle.definition
    (r*cos(theta), r*sin(theta))
    >>> circle.limits
    {theta: (0, pi)}

    Dimension of a parametric region determines whether a region is a curve, surface
    or volume region. It does not represent its dimensions in space.

    >>> circle.dimensions
    1

    Parameters
    ==========

    definition : tuple to define base scalars in terms of parameters.

    bounds : Parameter or a tuple of length 3 to define parameter and corresponding lower and upper bound.

    """

    def __new__(cls, definition, *bounds):
        if False:
            for i in range(10):
                print('nop')
        parameters = ()
        limits = {}
        if not isinstance(bounds, Tuple):
            bounds = Tuple(*bounds)
        for bound in bounds:
            if isinstance(bound, (tuple, Tuple)):
                if len(bound) != 3:
                    raise ValueError('Tuple should be in the form (parameter, lowerbound, upperbound)')
                parameters += (bound[0],)
                limits[bound[0]] = (bound[1], bound[2])
            else:
                parameters += (bound,)
        if not isinstance(definition, (tuple, Tuple)):
            definition = (definition,)
        obj = super().__new__(cls, Tuple(*definition), *bounds)
        obj._parameters = parameters
        obj._limits = limits
        return obj

    @property
    def definition(self):
        if False:
            i = 10
            return i + 15
        return self.args[0]

    @property
    def limits(self):
        if False:
            return 10
        return self._limits

    @property
    def parameters(self):
        if False:
            print('Hello World!')
        return self._parameters

    @property
    def dimensions(self):
        if False:
            return 10
        return len(self.limits)

@singledispatch
def parametric_region_list(reg):
    if False:
        return 10
    '\n    Returns a list of ParametricRegion objects representing the geometric region.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import t\n    >>> from sympy.vector import parametric_region_list\n    >>> from sympy.geometry import Point, Curve, Ellipse, Segment, Polygon\n\n    >>> p = Point(2, 5)\n    >>> parametric_region_list(p)\n    [ParametricRegion((2, 5))]\n\n    >>> c = Curve((t**3, 4*t), (t, -3, 4))\n    >>> parametric_region_list(c)\n    [ParametricRegion((t**3, 4*t), (t, -3, 4))]\n\n    >>> e = Ellipse(Point(1, 3), 2, 3)\n    >>> parametric_region_list(e)\n    [ParametricRegion((2*cos(t) + 1, 3*sin(t) + 3), (t, 0, 2*pi))]\n\n    >>> s = Segment(Point(1, 3), Point(2, 6))\n    >>> parametric_region_list(s)\n    [ParametricRegion((t + 1, 3*t + 3), (t, 0, 1))]\n\n    >>> p1, p2, p3, p4 = [(0, 1), (2, -3), (5, 3), (-2, 3)]\n    >>> poly = Polygon(p1, p2, p3, p4)\n    >>> parametric_region_list(poly)\n    [ParametricRegion((2*t, 1 - 4*t), (t, 0, 1)), ParametricRegion((3*t + 2, 6*t - 3), (t, 0, 1)),     ParametricRegion((5 - 7*t, 3), (t, 0, 1)), ParametricRegion((2*t - 2, 3 - 2*t),  (t, 0, 1))]\n\n    '
    raise ValueError('SymPy cannot determine parametric representation of the region.')

@parametric_region_list.register(Point)
def _(obj):
    if False:
        while True:
            i = 10
    return [ParametricRegion(obj.args)]

@parametric_region_list.register(Curve)
def _(obj):
    if False:
        print('Hello World!')
    definition = obj.arbitrary_point(obj.parameter).args
    bounds = obj.limits
    return [ParametricRegion(definition, bounds)]

@parametric_region_list.register(Ellipse)
def _(obj, parameter='t'):
    if False:
        return 10
    definition = obj.arbitrary_point(parameter).args
    t = _symbol(parameter, real=True)
    bounds = (t, 0, 2 * pi)
    return [ParametricRegion(definition, bounds)]

@parametric_region_list.register(Segment)
def _(obj, parameter='t'):
    if False:
        print('Hello World!')
    t = _symbol(parameter, real=True)
    definition = obj.arbitrary_point(t).args
    for i in range(0, 3):
        lower_bound = solve(definition[i] - obj.points[0].args[i], t)
        upper_bound = solve(definition[i] - obj.points[1].args[i], t)
        if len(lower_bound) == 1 and len(upper_bound) == 1:
            bounds = (t, lower_bound[0], upper_bound[0])
            break
    definition_tuple = obj.arbitrary_point(parameter).args
    return [ParametricRegion(definition_tuple, bounds)]

@parametric_region_list.register(Polygon)
def _(obj, parameter='t'):
    if False:
        while True:
            i = 10
    l = [parametric_region_list(side, parameter)[0] for side in obj.sides]
    return l

@parametric_region_list.register(ImplicitRegion)
def _(obj, parameters=('t', 's')):
    if False:
        while True:
            i = 10
    definition = obj.rational_parametrization(parameters)
    bounds = []
    for i in range(len(obj.variables) - 1):
        parameter = _symbol(parameters[i], real=True)
        definition = [trigsimp(elem.subs(parameter, tan(parameter / 2))) for elem in definition]
        bounds.append((parameter, 0, 2 * pi))
    definition = Tuple(*definition)
    return [ParametricRegion(definition, *bounds)]