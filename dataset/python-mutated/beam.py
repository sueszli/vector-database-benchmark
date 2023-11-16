"""
This module can be used to solve 2D beam bending problems with
singularity functions in mechanics.
"""
from sympy.core import S, Symbol, diff, symbols
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Derivative, Function
from sympy.core.mul import Mul
from sympy.core.relational import Eq
from sympy.core.sympify import sympify
from sympy.solvers import linsolve
from sympy.solvers.ode.ode import dsolve
from sympy.solvers.solvers import solve
from sympy.printing import sstr
from sympy.functions import SingularityFunction, Piecewise, factorial
from sympy.integrals import integrate
from sympy.series import limit
from sympy.plotting import plot, PlotGrid
from sympy.geometry.entity import GeometryEntity
from sympy.external import import_module
from sympy.sets.sets import Interval
from sympy.utilities.lambdify import lambdify
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import iterable
import warnings
numpy = import_module('numpy', import_kwargs={'fromlist': ['arange']})

class Beam:
    """
    A Beam is a structural element that is capable of withstanding load
    primarily by resisting against bending. Beams are characterized by
    their cross sectional profile(Second moment of area), their length
    and their material.

    .. note::
       A consistent sign convention must be used while solving a beam
       bending problem; the results will
       automatically follow the chosen sign convention. However, the
       chosen sign convention must respect the rule that, on the positive
       side of beam's axis (in respect to current section), a loading force
       giving positive shear yields a negative moment, as below (the
       curved arrow shows the positive moment and rotation):

    .. image:: allowed-sign-conventions.png

    Examples
    ========
    There is a beam of length 4 meters. A constant distributed load of 6 N/m
    is applied from half of the beam till the end. There are two simple supports
    below the beam, one at the starting point and another at the ending point
    of the beam. The deflection of the beam at the end is restricted.

    Using the sign convention of downwards forces being positive.

    >>> from sympy.physics.continuum_mechanics.beam import Beam
    >>> from sympy import symbols, Piecewise
    >>> E, I = symbols('E, I')
    >>> R1, R2 = symbols('R1, R2')
    >>> b = Beam(4, E, I)
    >>> b.apply_load(R1, 0, -1)
    >>> b.apply_load(6, 2, 0)
    >>> b.apply_load(R2, 4, -1)
    >>> b.bc_deflection = [(0, 0), (4, 0)]
    >>> b.boundary_conditions
    {'deflection': [(0, 0), (4, 0)], 'slope': []}
    >>> b.load
    R1*SingularityFunction(x, 0, -1) + R2*SingularityFunction(x, 4, -1) + 6*SingularityFunction(x, 2, 0)
    >>> b.solve_for_reaction_loads(R1, R2)
    >>> b.load
    -3*SingularityFunction(x, 0, -1) + 6*SingularityFunction(x, 2, 0) - 9*SingularityFunction(x, 4, -1)
    >>> b.shear_force()
    3*SingularityFunction(x, 0, 0) - 6*SingularityFunction(x, 2, 1) + 9*SingularityFunction(x, 4, 0)
    >>> b.bending_moment()
    3*SingularityFunction(x, 0, 1) - 3*SingularityFunction(x, 2, 2) + 9*SingularityFunction(x, 4, 1)
    >>> b.slope()
    (-3*SingularityFunction(x, 0, 2)/2 + SingularityFunction(x, 2, 3) - 9*SingularityFunction(x, 4, 2)/2 + 7)/(E*I)
    >>> b.deflection()
    (7*x - SingularityFunction(x, 0, 3)/2 + SingularityFunction(x, 2, 4)/4 - 3*SingularityFunction(x, 4, 3)/2)/(E*I)
    >>> b.deflection().rewrite(Piecewise)
    (7*x - Piecewise((x**3, x >= 0), (0, True))/2
         - 3*Piecewise(((x - 4)**3, x >= 4), (0, True))/2
         + Piecewise(((x - 2)**4, x >= 2), (0, True))/4)/(E*I)

    Calculate the support reactions for a fully symbolic beam of length L.
    There are two simple supports below the beam, one at the starting point
    and another at the ending point of the beam. The deflection of the beam
    at the end is restricted. The beam is loaded with:

    * a downward point load P1 applied at L/4
    * an upward point load P2 applied at L/8
    * a counterclockwise moment M1 applied at L/2
    * a clockwise moment M2 applied at 3*L/4
    * a distributed constant load q1, applied downward, starting from L/2
      up to 3*L/4
    * a distributed constant load q2, applied upward, starting from 3*L/4
      up to L

    No assumptions are needed for symbolic loads. However, defining a positive
    length will help the algorithm to compute the solution.

    >>> E, I = symbols('E, I')
    >>> L = symbols("L", positive=True)
    >>> P1, P2, M1, M2, q1, q2 = symbols("P1, P2, M1, M2, q1, q2")
    >>> R1, R2 = symbols('R1, R2')
    >>> b = Beam(L, E, I)
    >>> b.apply_load(R1, 0, -1)
    >>> b.apply_load(R2, L, -1)
    >>> b.apply_load(P1, L/4, -1)
    >>> b.apply_load(-P2, L/8, -1)
    >>> b.apply_load(M1, L/2, -2)
    >>> b.apply_load(-M2, 3*L/4, -2)
    >>> b.apply_load(q1, L/2, 0, 3*L/4)
    >>> b.apply_load(-q2, 3*L/4, 0, L)
    >>> b.bc_deflection = [(0, 0), (L, 0)]
    >>> b.solve_for_reaction_loads(R1, R2)
    >>> print(b.reaction_loads[R1])
    (-3*L**2*q1 + L**2*q2 - 24*L*P1 + 28*L*P2 - 32*M1 + 32*M2)/(32*L)
    >>> print(b.reaction_loads[R2])
    (-5*L**2*q1 + 7*L**2*q2 - 8*L*P1 + 4*L*P2 + 32*M1 - 32*M2)/(32*L)
    """

    def __init__(self, length, elastic_modulus, second_moment, area=Symbol('A'), variable=Symbol('x'), base_char='C'):
        if False:
            print('Hello World!')
        "Initializes the class.\n\n        Parameters\n        ==========\n\n        length : Sympifyable\n            A Symbol or value representing the Beam's length.\n\n        elastic_modulus : Sympifyable\n            A SymPy expression representing the Beam's Modulus of Elasticity.\n            It is a measure of the stiffness of the Beam material. It can\n            also be a continuous function of position along the beam.\n\n        second_moment : Sympifyable or Geometry object\n            Describes the cross-section of the beam via a SymPy expression\n            representing the Beam's second moment of area. It is a geometrical\n            property of an area which reflects how its points are distributed\n            with respect to its neutral axis. It can also be a continuous\n            function of position along the beam. Alternatively ``second_moment``\n            can be a shape object such as a ``Polygon`` from the geometry module\n            representing the shape of the cross-section of the beam. In such cases,\n            it is assumed that the x-axis of the shape object is aligned with the\n            bending axis of the beam. The second moment of area will be computed\n            from the shape object internally.\n\n        area : Symbol/float\n            Represents the cross-section area of beam\n\n        variable : Symbol, optional\n            A Symbol object that will be used as the variable along the beam\n            while representing the load, shear, moment, slope and deflection\n            curve. By default, it is set to ``Symbol('x')``.\n\n        base_char : String, optional\n            A String that will be used as base character to generate sequential\n            symbols for integration constants in cases where boundary conditions\n            are not sufficient to solve them.\n        "
        self.length = length
        self.elastic_modulus = elastic_modulus
        if isinstance(second_moment, GeometryEntity):
            self.cross_section = second_moment
        else:
            self.cross_section = None
            self.second_moment = second_moment
        self.variable = variable
        self._base_char = base_char
        self._boundary_conditions = {'deflection': [], 'slope': []}
        self._load = 0
        self.area = area
        self._applied_supports = []
        self._support_as_loads = []
        self._applied_loads = []
        self._reaction_loads = {}
        self._ild_reactions = {}
        self._ild_shear = 0
        self._ild_moment = 0
        self._original_load = 0
        self._composite_type = None
        self._hinge_position = None

    def __str__(self):
        if False:
            print('Hello World!')
        shape_description = self._cross_section if self._cross_section else self._second_moment
        str_sol = 'Beam({}, {}, {})'.format(sstr(self._length), sstr(self._elastic_modulus), sstr(shape_description))
        return str_sol

    @property
    def reaction_loads(self):
        if False:
            while True:
                i = 10
        ' Returns the reaction forces in a dictionary.'
        return self._reaction_loads

    @property
    def ild_shear(self):
        if False:
            print('Hello World!')
        ' Returns the I.L.D. shear equation.'
        return self._ild_shear

    @property
    def ild_reactions(self):
        if False:
            return 10
        ' Returns the I.L.D. reaction forces in a dictionary.'
        return self._ild_reactions

    @property
    def ild_moment(self):
        if False:
            i = 10
            return i + 15
        ' Returns the I.L.D. moment equation.'
        return self._ild_moment

    @property
    def length(self):
        if False:
            while True:
                i = 10
        'Length of the Beam.'
        return self._length

    @length.setter
    def length(self, l):
        if False:
            return 10
        self._length = sympify(l)

    @property
    def area(self):
        if False:
            return 10
        'Cross-sectional area of the Beam. '
        return self._area

    @area.setter
    def area(self, a):
        if False:
            for i in range(10):
                print('nop')
        self._area = sympify(a)

    @property
    def variable(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        A symbol that can be used as a variable along the length of the beam\n        while representing load distribution, shear force curve, bending\n        moment, slope curve and the deflection curve. By default, it is set\n        to ``Symbol('x')``, but this property is mutable.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.continuum_mechanics.beam import Beam\n        >>> from sympy import symbols\n        >>> E, I, A = symbols('E, I, A')\n        >>> x, y, z = symbols('x, y, z')\n        >>> b = Beam(4, E, I)\n        >>> b.variable\n        x\n        >>> b.variable = y\n        >>> b.variable\n        y\n        >>> b = Beam(4, E, I, A, z)\n        >>> b.variable\n        z\n        "
        return self._variable

    @variable.setter
    def variable(self, v):
        if False:
            print('Hello World!')
        if isinstance(v, Symbol):
            self._variable = v
        else:
            raise TypeError('The variable should be a Symbol object.')

    @property
    def elastic_modulus(self):
        if False:
            i = 10
            return i + 15
        "Young's Modulus of the Beam. "
        return self._elastic_modulus

    @elastic_modulus.setter
    def elastic_modulus(self, e):
        if False:
            i = 10
            return i + 15
        self._elastic_modulus = sympify(e)

    @property
    def second_moment(self):
        if False:
            for i in range(10):
                print('nop')
        'Second moment of area of the Beam. '
        return self._second_moment

    @second_moment.setter
    def second_moment(self, i):
        if False:
            i = 10
            return i + 15
        self._cross_section = None
        if isinstance(i, GeometryEntity):
            raise ValueError('To update cross-section geometry use `cross_section` attribute')
        else:
            self._second_moment = sympify(i)

    @property
    def cross_section(self):
        if False:
            while True:
                i = 10
        'Cross-section of the beam'
        return self._cross_section

    @cross_section.setter
    def cross_section(self, s):
        if False:
            for i in range(10):
                print('nop')
        if s:
            self._second_moment = s.second_moment_of_area()[0]
        self._cross_section = s

    @property
    def boundary_conditions(self):
        if False:
            while True:
                i = 10
        "\n        Returns a dictionary of boundary conditions applied on the beam.\n        The dictionary has three keywords namely moment, slope and deflection.\n        The value of each keyword is a list of tuple, where each tuple\n        contains location and value of a boundary condition in the format\n        (location, value).\n\n        Examples\n        ========\n        There is a beam of length 4 meters. The bending moment at 0 should be 4\n        and at 4 it should be 0. The slope of the beam should be 1 at 0. The\n        deflection should be 2 at 0.\n\n        >>> from sympy.physics.continuum_mechanics.beam import Beam\n        >>> from sympy import symbols\n        >>> E, I = symbols('E, I')\n        >>> b = Beam(4, E, I)\n        >>> b.bc_deflection = [(0, 2)]\n        >>> b.bc_slope = [(0, 1)]\n        >>> b.boundary_conditions\n        {'deflection': [(0, 2)], 'slope': [(0, 1)]}\n\n        Here the deflection of the beam should be ``2`` at ``0``.\n        Similarly, the slope of the beam should be ``1`` at ``0``.\n        "
        return self._boundary_conditions

    @property
    def bc_slope(self):
        if False:
            print('Hello World!')
        return self._boundary_conditions['slope']

    @bc_slope.setter
    def bc_slope(self, s_bcs):
        if False:
            while True:
                i = 10
        self._boundary_conditions['slope'] = s_bcs

    @property
    def bc_deflection(self):
        if False:
            for i in range(10):
                print('nop')
        return self._boundary_conditions['deflection']

    @bc_deflection.setter
    def bc_deflection(self, d_bcs):
        if False:
            while True:
                i = 10
        self._boundary_conditions['deflection'] = d_bcs

    def join(self, beam, via='fixed'):
        if False:
            return 10
        '\n        This method joins two beams to make a new composite beam system.\n        Passed Beam class instance is attached to the right end of calling\n        object. This method can be used to form beams having Discontinuous\n        values of Elastic modulus or Second moment.\n\n        Parameters\n        ==========\n        beam : Beam class object\n            The Beam object which would be connected to the right of calling\n            object.\n        via : String\n            States the way two Beam object would get connected\n            - For axially fixed Beams, via="fixed"\n            - For Beams connected via hinge, via="hinge"\n\n        Examples\n        ========\n        There is a cantilever beam of length 4 meters. For first 2 meters\n        its moment of inertia is `1.5*I` and `I` for the other end.\n        A pointload of magnitude 4 N is applied from the top at its free end.\n\n        >>> from sympy.physics.continuum_mechanics.beam import Beam\n        >>> from sympy import symbols\n        >>> E, I = symbols(\'E, I\')\n        >>> R1, R2 = symbols(\'R1, R2\')\n        >>> b1 = Beam(2, E, 1.5*I)\n        >>> b2 = Beam(2, E, I)\n        >>> b = b1.join(b2, "fixed")\n        >>> b.apply_load(20, 4, -1)\n        >>> b.apply_load(R1, 0, -1)\n        >>> b.apply_load(R2, 0, -2)\n        >>> b.bc_slope = [(0, 0)]\n        >>> b.bc_deflection = [(0, 0)]\n        >>> b.solve_for_reaction_loads(R1, R2)\n        >>> b.load\n        80*SingularityFunction(x, 0, -2) - 20*SingularityFunction(x, 0, -1) + 20*SingularityFunction(x, 4, -1)\n        >>> b.slope()\n        (-((-80*SingularityFunction(x, 0, 1) + 10*SingularityFunction(x, 0, 2) - 10*SingularityFunction(x, 4, 2))/I + 120/I)/E + 80.0/(E*I))*SingularityFunction(x, 2, 0)\n        - 0.666666666666667*(-80*SingularityFunction(x, 0, 1) + 10*SingularityFunction(x, 0, 2) - 10*SingularityFunction(x, 4, 2))*SingularityFunction(x, 0, 0)/(E*I)\n        + 0.666666666666667*(-80*SingularityFunction(x, 0, 1) + 10*SingularityFunction(x, 0, 2) - 10*SingularityFunction(x, 4, 2))*SingularityFunction(x, 2, 0)/(E*I)\n        '
        x = self.variable
        E = self.elastic_modulus
        new_length = self.length + beam.length
        if self.second_moment != beam.second_moment:
            new_second_moment = Piecewise((self.second_moment, x <= self.length), (beam.second_moment, x <= new_length))
        else:
            new_second_moment = self.second_moment
        if via == 'fixed':
            new_beam = Beam(new_length, E, new_second_moment, x)
            new_beam._composite_type = 'fixed'
            return new_beam
        if via == 'hinge':
            new_beam = Beam(new_length, E, new_second_moment, x)
            new_beam._composite_type = 'hinge'
            new_beam._hinge_position = self.length
            return new_beam

    def apply_support(self, loc, type='fixed'):
        if False:
            while True:
                i = 10
        '\n        This method applies support to a particular beam object.\n\n        Parameters\n        ==========\n        loc : Sympifyable\n            Location of point at which support is applied.\n        type : String\n            Determines type of Beam support applied. To apply support structure\n            with\n            - zero degree of freedom, type = "fixed"\n            - one degree of freedom, type = "pin"\n            - two degrees of freedom, type = "roller"\n\n        Examples\n        ========\n        There is a beam of length 30 meters. A moment of magnitude 120 Nm is\n        applied in the clockwise direction at the end of the beam. A pointload\n        of magnitude 8 N is applied from the top of the beam at the starting\n        point. There are two simple supports below the beam. One at the end\n        and another one at a distance of 10 meters from the start. The\n        deflection is restricted at both the supports.\n\n        Using the sign convention of upward forces and clockwise moment\n        being positive.\n\n        >>> from sympy.physics.continuum_mechanics.beam import Beam\n        >>> from sympy import symbols\n        >>> E, I = symbols(\'E, I\')\n        >>> b = Beam(30, E, I)\n        >>> b.apply_support(10, \'roller\')\n        >>> b.apply_support(30, \'roller\')\n        >>> b.apply_load(-8, 0, -1)\n        >>> b.apply_load(120, 30, -2)\n        >>> R_10, R_30 = symbols(\'R_10, R_30\')\n        >>> b.solve_for_reaction_loads(R_10, R_30)\n        >>> b.load\n        -8*SingularityFunction(x, 0, -1) + 6*SingularityFunction(x, 10, -1)\n        + 120*SingularityFunction(x, 30, -2) + 2*SingularityFunction(x, 30, -1)\n        >>> b.slope()\n        (-4*SingularityFunction(x, 0, 2) + 3*SingularityFunction(x, 10, 2)\n            + 120*SingularityFunction(x, 30, 1) + SingularityFunction(x, 30, 2) + 4000/3)/(E*I)\n        '
        loc = sympify(loc)
        self._applied_supports.append((loc, type))
        if type in ('pin', 'roller'):
            reaction_load = Symbol('R_' + str(loc))
            self.apply_load(reaction_load, loc, -1)
            self.bc_deflection.append((loc, 0))
        else:
            reaction_load = Symbol('R_' + str(loc))
            reaction_moment = Symbol('M_' + str(loc))
            self.apply_load(reaction_load, loc, -1)
            self.apply_load(reaction_moment, loc, -2)
            self.bc_deflection.append((loc, 0))
            self.bc_slope.append((loc, 0))
            self._support_as_loads.append((reaction_moment, loc, -2, None))
        self._support_as_loads.append((reaction_load, loc, -1, None))

    def apply_load(self, value, start, order, end=None):
        if False:
            while True:
                i = 10
        "\n        This method adds up the loads given to a particular beam object.\n\n        Parameters\n        ==========\n        value : Sympifyable\n            The value inserted should have the units [Force/(Distance**(n+1)]\n            where n is the order of applied load.\n            Units for applied loads:\n\n               - For moments, unit = kN*m\n               - For point loads, unit = kN\n               - For constant distributed load, unit = kN/m\n               - For ramp loads, unit = kN/m/m\n               - For parabolic ramp loads, unit = kN/m/m/m\n               - ... so on.\n\n        start : Sympifyable\n            The starting point of the applied load. For point moments and\n            point forces this is the location of application.\n        order : Integer\n            The order of the applied load.\n\n               - For moments, order = -2\n               - For point loads, order =-1\n               - For constant distributed load, order = 0\n               - For ramp loads, order = 1\n               - For parabolic ramp loads, order = 2\n               - ... so on.\n\n        end : Sympifyable, optional\n            An optional argument that can be used if the load has an end point\n            within the length of the beam.\n\n        Examples\n        ========\n        There is a beam of length 4 meters. A moment of magnitude 3 Nm is\n        applied in the clockwise direction at the starting point of the beam.\n        A point load of magnitude 4 N is applied from the top of the beam at\n        2 meters from the starting point and a parabolic ramp load of magnitude\n        2 N/m is applied below the beam starting from 2 meters to 3 meters\n        away from the starting point of the beam.\n\n        >>> from sympy.physics.continuum_mechanics.beam import Beam\n        >>> from sympy import symbols\n        >>> E, I = symbols('E, I')\n        >>> b = Beam(4, E, I)\n        >>> b.apply_load(-3, 0, -2)\n        >>> b.apply_load(4, 2, -1)\n        >>> b.apply_load(-2, 2, 2, end=3)\n        >>> b.load\n        -3*SingularityFunction(x, 0, -2) + 4*SingularityFunction(x, 2, -1) - 2*SingularityFunction(x, 2, 2) + 2*SingularityFunction(x, 3, 0) + 4*SingularityFunction(x, 3, 1) + 2*SingularityFunction(x, 3, 2)\n\n        "
        x = self.variable
        value = sympify(value)
        start = sympify(start)
        order = sympify(order)
        self._applied_loads.append((value, start, order, end))
        self._load += value * SingularityFunction(x, start, order)
        self._original_load += value * SingularityFunction(x, start, order)
        if end:
            self._handle_end(x, value, start, order, end, type='apply')

    def remove_load(self, value, start, order, end=None):
        if False:
            i = 10
            return i + 15
        "\n        This method removes a particular load present on the beam object.\n        Returns a ValueError if the load passed as an argument is not\n        present on the beam.\n\n        Parameters\n        ==========\n        value : Sympifyable\n            The magnitude of an applied load.\n        start : Sympifyable\n            The starting point of the applied load. For point moments and\n            point forces this is the location of application.\n        order : Integer\n            The order of the applied load.\n            - For moments, order= -2\n            - For point loads, order=-1\n            - For constant distributed load, order=0\n            - For ramp loads, order=1\n            - For parabolic ramp loads, order=2\n            - ... so on.\n        end : Sympifyable, optional\n            An optional argument that can be used if the load has an end point\n            within the length of the beam.\n\n        Examples\n        ========\n        There is a beam of length 4 meters. A moment of magnitude 3 Nm is\n        applied in the clockwise direction at the starting point of the beam.\n        A pointload of magnitude 4 N is applied from the top of the beam at\n        2 meters from the starting point and a parabolic ramp load of magnitude\n        2 N/m is applied below the beam starting from 2 meters to 3 meters\n        away from the starting point of the beam.\n\n        >>> from sympy.physics.continuum_mechanics.beam import Beam\n        >>> from sympy import symbols\n        >>> E, I = symbols('E, I')\n        >>> b = Beam(4, E, I)\n        >>> b.apply_load(-3, 0, -2)\n        >>> b.apply_load(4, 2, -1)\n        >>> b.apply_load(-2, 2, 2, end=3)\n        >>> b.load\n        -3*SingularityFunction(x, 0, -2) + 4*SingularityFunction(x, 2, -1) - 2*SingularityFunction(x, 2, 2) + 2*SingularityFunction(x, 3, 0) + 4*SingularityFunction(x, 3, 1) + 2*SingularityFunction(x, 3, 2)\n        >>> b.remove_load(-2, 2, 2, end = 3)\n        >>> b.load\n        -3*SingularityFunction(x, 0, -2) + 4*SingularityFunction(x, 2, -1)\n        "
        x = self.variable
        value = sympify(value)
        start = sympify(start)
        order = sympify(order)
        if (value, start, order, end) in self._applied_loads:
            self._load -= value * SingularityFunction(x, start, order)
            self._original_load -= value * SingularityFunction(x, start, order)
            self._applied_loads.remove((value, start, order, end))
        else:
            msg = 'No such load distribution exists on the beam object.'
            raise ValueError(msg)
        if end:
            self._handle_end(x, value, start, order, end, type='remove')

    def _handle_end(self, x, value, start, order, end, type):
        if False:
            return 10
        '\n        This functions handles the optional `end` value in the\n        `apply_load` and `remove_load` functions. When the value\n        of end is not NULL, this function will be executed.\n        '
        if order.is_negative:
            msg = "If 'end' is provided the 'order' of the load cannot be negative, i.e. 'end' is only valid for distributed loads."
            raise ValueError(msg)
        f = value * x ** order
        if type == 'apply':
            for i in range(0, order + 1):
                self._load -= f.diff(x, i).subs(x, end - start) * SingularityFunction(x, end, i) / factorial(i)
                self._original_load -= f.diff(x, i).subs(x, end - start) * SingularityFunction(x, end, i) / factorial(i)
        elif type == 'remove':
            for i in range(0, order + 1):
                self._load += f.diff(x, i).subs(x, end - start) * SingularityFunction(x, end, i) / factorial(i)
                self._original_load += f.diff(x, i).subs(x, end - start) * SingularityFunction(x, end, i) / factorial(i)

    @property
    def load(self):
        if False:
            print('Hello World!')
        "\n        Returns a Singularity Function expression which represents\n        the load distribution curve of the Beam object.\n\n        Examples\n        ========\n        There is a beam of length 4 meters. A moment of magnitude 3 Nm is\n        applied in the clockwise direction at the starting point of the beam.\n        A point load of magnitude 4 N is applied from the top of the beam at\n        2 meters from the starting point and a parabolic ramp load of magnitude\n        2 N/m is applied below the beam starting from 3 meters away from the\n        starting point of the beam.\n\n        >>> from sympy.physics.continuum_mechanics.beam import Beam\n        >>> from sympy import symbols\n        >>> E, I = symbols('E, I')\n        >>> b = Beam(4, E, I)\n        >>> b.apply_load(-3, 0, -2)\n        >>> b.apply_load(4, 2, -1)\n        >>> b.apply_load(-2, 3, 2)\n        >>> b.load\n        -3*SingularityFunction(x, 0, -2) + 4*SingularityFunction(x, 2, -1) - 2*SingularityFunction(x, 3, 2)\n        "
        return self._load

    @property
    def applied_loads(self):
        if False:
            while True:
                i = 10
        "\n        Returns a list of all loads applied on the beam object.\n        Each load in the list is a tuple of form (value, start, order, end).\n\n        Examples\n        ========\n        There is a beam of length 4 meters. A moment of magnitude 3 Nm is\n        applied in the clockwise direction at the starting point of the beam.\n        A pointload of magnitude 4 N is applied from the top of the beam at\n        2 meters from the starting point. Another pointload of magnitude 5 N\n        is applied at same position.\n\n        >>> from sympy.physics.continuum_mechanics.beam import Beam\n        >>> from sympy import symbols\n        >>> E, I = symbols('E, I')\n        >>> b = Beam(4, E, I)\n        >>> b.apply_load(-3, 0, -2)\n        >>> b.apply_load(4, 2, -1)\n        >>> b.apply_load(5, 2, -1)\n        >>> b.load\n        -3*SingularityFunction(x, 0, -2) + 9*SingularityFunction(x, 2, -1)\n        >>> b.applied_loads\n        [(-3, 0, -2, None), (4, 2, -1, None), (5, 2, -1, None)]\n        "
        return self._applied_loads

    def _solve_hinge_beams(self, *reactions):
        if False:
            for i in range(10):
                print('nop')
        'Method to find integration constants and reactional variables in a\n        composite beam connected via hinge.\n        This method resolves the composite Beam into its sub-beams and then\n        equations of shear force, bending moment, slope and deflection are\n        evaluated for both of them separately. These equations are then solved\n        for unknown reactions and integration constants using the boundary\n        conditions applied on the Beam. Equal deflection of both sub-beams\n        at the hinge joint gives us another equation to solve the system.\n\n        Examples\n        ========\n        A combined beam, with constant fkexural rigidity E*I, is formed by joining\n        a Beam of length 2*l to the right of another Beam of length l. The whole beam\n        is fixed at both of its both end. A point load of magnitude P is also applied\n        from the top at a distance of 2*l from starting point.\n\n        >>> from sympy.physics.continuum_mechanics.beam import Beam\n        >>> from sympy import symbols\n        >>> E, I = symbols(\'E, I\')\n        >>> l=symbols(\'l\', positive=True)\n        >>> b1=Beam(l, E, I)\n        >>> b2=Beam(2*l, E, I)\n        >>> b=b1.join(b2,"hinge")\n        >>> M1, A1, M2, A2, P = symbols(\'M1 A1 M2 A2 P\')\n        >>> b.apply_load(A1,0,-1)\n        >>> b.apply_load(M1,0,-2)\n        >>> b.apply_load(P,2*l,-1)\n        >>> b.apply_load(A2,3*l,-1)\n        >>> b.apply_load(M2,3*l,-2)\n        >>> b.bc_slope=[(0,0), (3*l, 0)]\n        >>> b.bc_deflection=[(0,0), (3*l, 0)]\n        >>> b.solve_for_reaction_loads(M1, A1, M2, A2)\n        >>> b.reaction_loads\n        {A1: -5*P/18, A2: -13*P/18, M1: 5*P*l/18, M2: -4*P*l/9}\n        >>> b.slope()\n        (5*P*l*SingularityFunction(x, 0, 1)/18 - 5*P*SingularityFunction(x, 0, 2)/36 + 5*P*SingularityFunction(x, l, 2)/36)*SingularityFunction(x, 0, 0)/(E*I)\n        - (5*P*l*SingularityFunction(x, 0, 1)/18 - 5*P*SingularityFunction(x, 0, 2)/36 + 5*P*SingularityFunction(x, l, 2)/36)*SingularityFunction(x, l, 0)/(E*I)\n        + (P*l**2/18 - 4*P*l*SingularityFunction(-l + x, 2*l, 1)/9 - 5*P*SingularityFunction(-l + x, 0, 2)/36 + P*SingularityFunction(-l + x, l, 2)/2\n        - 13*P*SingularityFunction(-l + x, 2*l, 2)/36)*SingularityFunction(x, l, 0)/(E*I)\n        >>> b.deflection()\n        (5*P*l*SingularityFunction(x, 0, 2)/36 - 5*P*SingularityFunction(x, 0, 3)/108 + 5*P*SingularityFunction(x, l, 3)/108)*SingularityFunction(x, 0, 0)/(E*I)\n        - (5*P*l*SingularityFunction(x, 0, 2)/36 - 5*P*SingularityFunction(x, 0, 3)/108 + 5*P*SingularityFunction(x, l, 3)/108)*SingularityFunction(x, l, 0)/(E*I)\n        + (5*P*l**3/54 + P*l**2*(-l + x)/18 - 2*P*l*SingularityFunction(-l + x, 2*l, 2)/9 - 5*P*SingularityFunction(-l + x, 0, 3)/108 + P*SingularityFunction(-l + x, l, 3)/6\n        - 13*P*SingularityFunction(-l + x, 2*l, 3)/108)*SingularityFunction(x, l, 0)/(E*I)\n        '
        x = self.variable
        l = self._hinge_position
        E = self._elastic_modulus
        I = self._second_moment
        if isinstance(I, Piecewise):
            I1 = I.args[0][0]
            I2 = I.args[1][0]
        else:
            I1 = I2 = I
        load_1 = 0
        load_2 = 0
        for load in self.applied_loads:
            if load[1] < l:
                load_1 += load[0] * SingularityFunction(x, load[1], load[2])
                if load[2] == 0:
                    load_1 -= load[0] * SingularityFunction(x, load[3], load[2])
                elif load[2] > 0:
                    load_1 -= load[0] * SingularityFunction(x, load[3], load[2]) + load[0] * SingularityFunction(x, load[3], 0)
            elif load[1] == l:
                load_1 += load[0] * SingularityFunction(x, load[1], load[2])
                load_2 += load[0] * SingularityFunction(x, load[1] - l, load[2])
            elif load[1] > l:
                load_2 += load[0] * SingularityFunction(x, load[1] - l, load[2])
                if load[2] == 0:
                    load_2 -= load[0] * SingularityFunction(x, load[3] - l, load[2])
                elif load[2] > 0:
                    load_2 -= load[0] * SingularityFunction(x, load[3] - l, load[2]) + load[0] * SingularityFunction(x, load[3] - l, 0)
        h = Symbol('h')
        load_1 += h * SingularityFunction(x, l, -1)
        load_2 -= h * SingularityFunction(x, 0, -1)
        eq = []
        shear_1 = integrate(load_1, x)
        shear_curve_1 = limit(shear_1, x, l)
        eq.append(shear_curve_1)
        bending_1 = integrate(shear_1, x)
        moment_curve_1 = limit(bending_1, x, l)
        eq.append(moment_curve_1)
        shear_2 = integrate(load_2, x)
        shear_curve_2 = limit(shear_2, x, self.length - l)
        eq.append(shear_curve_2)
        bending_2 = integrate(shear_2, x)
        moment_curve_2 = limit(bending_2, x, self.length - l)
        eq.append(moment_curve_2)
        C1 = Symbol('C1')
        C2 = Symbol('C2')
        C3 = Symbol('C3')
        C4 = Symbol('C4')
        slope_1 = S.One / (E * I1) * (integrate(bending_1, x) + C1)
        def_1 = S.One / (E * I1) * (integrate(E * I * slope_1, x) + C1 * x + C2)
        slope_2 = S.One / (E * I2) * (integrate(integrate(integrate(load_2, x), x), x) + C3)
        def_2 = S.One / (E * I2) * (integrate(E * I * slope_2, x) + C4)
        for (position, value) in self.bc_slope:
            if position < l:
                eq.append(slope_1.subs(x, position) - value)
            else:
                eq.append(slope_2.subs(x, position - l) - value)
        for (position, value) in self.bc_deflection:
            if position < l:
                eq.append(def_1.subs(x, position) - value)
            else:
                eq.append(def_2.subs(x, position - l) - value)
        eq.append(def_1.subs(x, l) - def_2.subs(x, 0))
        constants = list(linsolve(eq, C1, C2, C3, C4, h, *reactions))
        reaction_values = list(constants[0])[5:]
        self._reaction_loads = dict(zip(reactions, reaction_values))
        self._load = self._load.subs(self._reaction_loads)
        slope_1 = slope_1.subs({C1: constants[0][0], h: constants[0][4]}).subs(self._reaction_loads)
        def_1 = def_1.subs({C1: constants[0][0], C2: constants[0][1], h: constants[0][4]}).subs(self._reaction_loads)
        slope_2 = slope_2.subs({x: x - l, C3: constants[0][2], h: constants[0][4]}).subs(self._reaction_loads)
        def_2 = def_2.subs({x: x - l, C3: constants[0][2], C4: constants[0][3], h: constants[0][4]}).subs(self._reaction_loads)
        self._hinge_beam_slope = slope_1 * SingularityFunction(x, 0, 0) - slope_1 * SingularityFunction(x, l, 0) + slope_2 * SingularityFunction(x, l, 0)
        self._hinge_beam_deflection = def_1 * SingularityFunction(x, 0, 0) - def_1 * SingularityFunction(x, l, 0) + def_2 * SingularityFunction(x, l, 0)

    def solve_for_reaction_loads(self, *reactions):
        if False:
            print('Hello World!')
        "\n        Solves for the reaction forces.\n\n        Examples\n        ========\n        There is a beam of length 30 meters. A moment of magnitude 120 Nm is\n        applied in the clockwise direction at the end of the beam. A pointload\n        of magnitude 8 N is applied from the top of the beam at the starting\n        point. There are two simple supports below the beam. One at the end\n        and another one at a distance of 10 meters from the start. The\n        deflection is restricted at both the supports.\n\n        Using the sign convention of upward forces and clockwise moment\n        being positive.\n\n        >>> from sympy.physics.continuum_mechanics.beam import Beam\n        >>> from sympy import symbols\n        >>> E, I = symbols('E, I')\n        >>> R1, R2 = symbols('R1, R2')\n        >>> b = Beam(30, E, I)\n        >>> b.apply_load(-8, 0, -1)\n        >>> b.apply_load(R1, 10, -1)  # Reaction force at x = 10\n        >>> b.apply_load(R2, 30, -1)  # Reaction force at x = 30\n        >>> b.apply_load(120, 30, -2)\n        >>> b.bc_deflection = [(10, 0), (30, 0)]\n        >>> b.load\n        R1*SingularityFunction(x, 10, -1) + R2*SingularityFunction(x, 30, -1)\n            - 8*SingularityFunction(x, 0, -1) + 120*SingularityFunction(x, 30, -2)\n        >>> b.solve_for_reaction_loads(R1, R2)\n        >>> b.reaction_loads\n        {R1: 6, R2: 2}\n        >>> b.load\n        -8*SingularityFunction(x, 0, -1) + 6*SingularityFunction(x, 10, -1)\n            + 120*SingularityFunction(x, 30, -2) + 2*SingularityFunction(x, 30, -1)\n        "
        if self._composite_type == 'hinge':
            return self._solve_hinge_beams(*reactions)
        x = self.variable
        l = self.length
        C3 = Symbol('C3')
        C4 = Symbol('C4')
        shear_curve = limit(self.shear_force(), x, l)
        moment_curve = limit(self.bending_moment(), x, l)
        slope_eqs = []
        deflection_eqs = []
        slope_curve = integrate(self.bending_moment(), x) + C3
        for (position, value) in self._boundary_conditions['slope']:
            eqs = slope_curve.subs(x, position) - value
            slope_eqs.append(eqs)
        deflection_curve = integrate(slope_curve, x) + C4
        for (position, value) in self._boundary_conditions['deflection']:
            eqs = deflection_curve.subs(x, position) - value
            deflection_eqs.append(eqs)
        solution = list(linsolve([shear_curve, moment_curve] + slope_eqs + deflection_eqs, (C3, C4) + reactions).args[0])
        solution = solution[2:]
        self._reaction_loads = dict(zip(reactions, solution))
        self._load = self._load.subs(self._reaction_loads)

    def shear_force(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns a Singularity Function expression which represents\n        the shear force curve of the Beam object.\n\n        Examples\n        ========\n        There is a beam of length 30 meters. A moment of magnitude 120 Nm is\n        applied in the clockwise direction at the end of the beam. A pointload\n        of magnitude 8 N is applied from the top of the beam at the starting\n        point. There are two simple supports below the beam. One at the end\n        and another one at a distance of 10 meters from the start. The\n        deflection is restricted at both the supports.\n\n        Using the sign convention of upward forces and clockwise moment\n        being positive.\n\n        >>> from sympy.physics.continuum_mechanics.beam import Beam\n        >>> from sympy import symbols\n        >>> E, I = symbols('E, I')\n        >>> R1, R2 = symbols('R1, R2')\n        >>> b = Beam(30, E, I)\n        >>> b.apply_load(-8, 0, -1)\n        >>> b.apply_load(R1, 10, -1)\n        >>> b.apply_load(R2, 30, -1)\n        >>> b.apply_load(120, 30, -2)\n        >>> b.bc_deflection = [(10, 0), (30, 0)]\n        >>> b.solve_for_reaction_loads(R1, R2)\n        >>> b.shear_force()\n        8*SingularityFunction(x, 0, 0) - 6*SingularityFunction(x, 10, 0) - 120*SingularityFunction(x, 30, -1) - 2*SingularityFunction(x, 30, 0)\n        "
        x = self.variable
        return -integrate(self.load, x)

    def max_shear_force(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns maximum Shear force and its coordinate\n        in the Beam object.'
        shear_curve = self.shear_force()
        x = self.variable
        terms = shear_curve.args
        singularity = []
        for term in terms:
            if isinstance(term, Mul):
                term = term.args[-1]
            singularity.append(term.args[1])
        singularity.sort()
        singularity = list(set(singularity))
        intervals = []
        shear_values = []
        for (i, s) in enumerate(singularity):
            if s == 0:
                continue
            try:
                shear_slope = Piecewise((float('nan'), x <= singularity[i - 1]), (self._load.rewrite(Piecewise), x < s), (float('nan'), True))
                points = solve(shear_slope, x)
                val = []
                for point in points:
                    val.append(abs(shear_curve.subs(x, point)))
                points.extend([singularity[i - 1], s])
                val += [abs(limit(shear_curve, x, singularity[i - 1], '+')), abs(limit(shear_curve, x, s, '-'))]
                max_shear = max(val)
                shear_values.append(max_shear)
                intervals.append(points[val.index(max_shear)])
            except NotImplementedError:
                initial_shear = limit(shear_curve, x, singularity[i - 1], '+')
                final_shear = limit(shear_curve, x, s, '-')
                if shear_curve.subs(x, (singularity[i - 1] + s) / 2) == (initial_shear + final_shear) / 2 and initial_shear != final_shear:
                    shear_values.extend([initial_shear, final_shear])
                    intervals.extend([singularity[i - 1], s])
                else:
                    shear_values.append(final_shear)
                    intervals.append(Interval(singularity[i - 1], s))
        shear_values = list(map(abs, shear_values))
        maximum_shear = max(shear_values)
        point = intervals[shear_values.index(maximum_shear)]
        return (point, maximum_shear)

    def bending_moment(self):
        if False:
            while True:
                i = 10
        "\n        Returns a Singularity Function expression which represents\n        the bending moment curve of the Beam object.\n\n        Examples\n        ========\n        There is a beam of length 30 meters. A moment of magnitude 120 Nm is\n        applied in the clockwise direction at the end of the beam. A pointload\n        of magnitude 8 N is applied from the top of the beam at the starting\n        point. There are two simple supports below the beam. One at the end\n        and another one at a distance of 10 meters from the start. The\n        deflection is restricted at both the supports.\n\n        Using the sign convention of upward forces and clockwise moment\n        being positive.\n\n        >>> from sympy.physics.continuum_mechanics.beam import Beam\n        >>> from sympy import symbols\n        >>> E, I = symbols('E, I')\n        >>> R1, R2 = symbols('R1, R2')\n        >>> b = Beam(30, E, I)\n        >>> b.apply_load(-8, 0, -1)\n        >>> b.apply_load(R1, 10, -1)\n        >>> b.apply_load(R2, 30, -1)\n        >>> b.apply_load(120, 30, -2)\n        >>> b.bc_deflection = [(10, 0), (30, 0)]\n        >>> b.solve_for_reaction_loads(R1, R2)\n        >>> b.bending_moment()\n        8*SingularityFunction(x, 0, 1) - 6*SingularityFunction(x, 10, 1) - 120*SingularityFunction(x, 30, 0) - 2*SingularityFunction(x, 30, 1)\n        "
        x = self.variable
        return integrate(self.shear_force(), x)

    def max_bmoment(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns maximum Shear force and its coordinate\n        in the Beam object.'
        bending_curve = self.bending_moment()
        x = self.variable
        terms = bending_curve.args
        singularity = []
        for term in terms:
            if isinstance(term, Mul):
                term = term.args[-1]
            singularity.append(term.args[1])
        singularity.sort()
        singularity = list(set(singularity))
        intervals = []
        moment_values = []
        for (i, s) in enumerate(singularity):
            if s == 0:
                continue
            try:
                moment_slope = Piecewise((float('nan'), x <= singularity[i - 1]), (self.shear_force().rewrite(Piecewise), x < s), (float('nan'), True))
                points = solve(moment_slope, x)
                val = []
                for point in points:
                    val.append(abs(bending_curve.subs(x, point)))
                points.extend([singularity[i - 1], s])
                val += [abs(limit(bending_curve, x, singularity[i - 1], '+')), abs(limit(bending_curve, x, s, '-'))]
                max_moment = max(val)
                moment_values.append(max_moment)
                intervals.append(points[val.index(max_moment)])
            except NotImplementedError:
                initial_moment = limit(bending_curve, x, singularity[i - 1], '+')
                final_moment = limit(bending_curve, x, s, '-')
                if bending_curve.subs(x, (singularity[i - 1] + s) / 2) == (initial_moment + final_moment) / 2 and initial_moment != final_moment:
                    moment_values.extend([initial_moment, final_moment])
                    intervals.extend([singularity[i - 1], s])
                else:
                    moment_values.append(final_moment)
                    intervals.append(Interval(singularity[i - 1], s))
        moment_values = list(map(abs, moment_values))
        maximum_moment = max(moment_values)
        point = intervals[moment_values.index(maximum_moment)]
        return (point, maximum_moment)

    def point_cflexure(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns a Set of point(s) with zero bending moment and\n        where bending moment curve of the beam object changes\n        its sign from negative to positive or vice versa.\n\n        Examples\n        ========\n        There is is 10 meter long overhanging beam. There are\n        two simple supports below the beam. One at the start\n        and another one at a distance of 6 meters from the start.\n        Point loads of magnitude 10KN and 20KN are applied at\n        2 meters and 4 meters from start respectively. A Uniformly\n        distribute load of magnitude of magnitude 3KN/m is also\n        applied on top starting from 6 meters away from starting\n        point till end.\n        Using the sign convention of upward forces and clockwise moment\n        being positive.\n\n        >>> from sympy.physics.continuum_mechanics.beam import Beam\n        >>> from sympy import symbols\n        >>> E, I = symbols('E, I')\n        >>> b = Beam(10, E, I)\n        >>> b.apply_load(-4, 0, -1)\n        >>> b.apply_load(-46, 6, -1)\n        >>> b.apply_load(10, 2, -1)\n        >>> b.apply_load(20, 4, -1)\n        >>> b.apply_load(3, 6, 0)\n        >>> b.point_cflexure()\n        [10/3]\n        "
        moment_curve = Piecewise((float('nan'), self.variable <= 0), (self.bending_moment(), self.variable < self.length), (float('nan'), True))
        points = solve(moment_curve.rewrite(Piecewise), self.variable, domain=S.Reals)
        return points

    def slope(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns a Singularity Function expression which represents\n        the slope the elastic curve of the Beam object.\n\n        Examples\n        ========\n        There is a beam of length 30 meters. A moment of magnitude 120 Nm is\n        applied in the clockwise direction at the end of the beam. A pointload\n        of magnitude 8 N is applied from the top of the beam at the starting\n        point. There are two simple supports below the beam. One at the end\n        and another one at a distance of 10 meters from the start. The\n        deflection is restricted at both the supports.\n\n        Using the sign convention of upward forces and clockwise moment\n        being positive.\n\n        >>> from sympy.physics.continuum_mechanics.beam import Beam\n        >>> from sympy import symbols\n        >>> E, I = symbols('E, I')\n        >>> R1, R2 = symbols('R1, R2')\n        >>> b = Beam(30, E, I)\n        >>> b.apply_load(-8, 0, -1)\n        >>> b.apply_load(R1, 10, -1)\n        >>> b.apply_load(R2, 30, -1)\n        >>> b.apply_load(120, 30, -2)\n        >>> b.bc_deflection = [(10, 0), (30, 0)]\n        >>> b.solve_for_reaction_loads(R1, R2)\n        >>> b.slope()\n        (-4*SingularityFunction(x, 0, 2) + 3*SingularityFunction(x, 10, 2)\n            + 120*SingularityFunction(x, 30, 1) + SingularityFunction(x, 30, 2) + 4000/3)/(E*I)\n        "
        x = self.variable
        E = self.elastic_modulus
        I = self.second_moment
        if self._composite_type == 'hinge':
            return self._hinge_beam_slope
        if not self._boundary_conditions['slope']:
            return diff(self.deflection(), x)
        if isinstance(I, Piecewise) and self._composite_type == 'fixed':
            args = I.args
            slope = 0
            prev_slope = 0
            prev_end = 0
            for i in range(len(args)):
                if i != 0:
                    prev_end = args[i - 1][1].args[1]
                slope_value = -S.One / E * integrate(self.bending_moment() / args[i][0], (x, prev_end, x))
                if i != len(args) - 1:
                    slope += (prev_slope + slope_value) * SingularityFunction(x, prev_end, 0) - (prev_slope + slope_value) * SingularityFunction(x, args[i][1].args[1], 0)
                else:
                    slope += (prev_slope + slope_value) * SingularityFunction(x, prev_end, 0)
                prev_slope = slope_value.subs(x, args[i][1].args[1])
            return slope
        C3 = Symbol('C3')
        slope_curve = -integrate(S.One / (E * I) * self.bending_moment(), x) + C3
        bc_eqs = []
        for (position, value) in self._boundary_conditions['slope']:
            eqs = slope_curve.subs(x, position) - value
            bc_eqs.append(eqs)
        constants = list(linsolve(bc_eqs, C3))
        slope_curve = slope_curve.subs({C3: constants[0][0]})
        return slope_curve

    def deflection(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns a Singularity Function expression which represents\n        the elastic curve or deflection of the Beam object.\n\n        Examples\n        ========\n        There is a beam of length 30 meters. A moment of magnitude 120 Nm is\n        applied in the clockwise direction at the end of the beam. A pointload\n        of magnitude 8 N is applied from the top of the beam at the starting\n        point. There are two simple supports below the beam. One at the end\n        and another one at a distance of 10 meters from the start. The\n        deflection is restricted at both the supports.\n\n        Using the sign convention of upward forces and clockwise moment\n        being positive.\n\n        >>> from sympy.physics.continuum_mechanics.beam import Beam\n        >>> from sympy import symbols\n        >>> E, I = symbols('E, I')\n        >>> R1, R2 = symbols('R1, R2')\n        >>> b = Beam(30, E, I)\n        >>> b.apply_load(-8, 0, -1)\n        >>> b.apply_load(R1, 10, -1)\n        >>> b.apply_load(R2, 30, -1)\n        >>> b.apply_load(120, 30, -2)\n        >>> b.bc_deflection = [(10, 0), (30, 0)]\n        >>> b.solve_for_reaction_loads(R1, R2)\n        >>> b.deflection()\n        (4000*x/3 - 4*SingularityFunction(x, 0, 3)/3 + SingularityFunction(x, 10, 3)\n            + 60*SingularityFunction(x, 30, 2) + SingularityFunction(x, 30, 3)/3 - 12000)/(E*I)\n        "
        x = self.variable
        E = self.elastic_modulus
        I = self.second_moment
        if self._composite_type == 'hinge':
            return self._hinge_beam_deflection
        if not self._boundary_conditions['deflection'] and (not self._boundary_conditions['slope']):
            if isinstance(I, Piecewise) and self._composite_type == 'fixed':
                args = I.args
                prev_slope = 0
                prev_def = 0
                prev_end = 0
                deflection = 0
                for i in range(len(args)):
                    if i != 0:
                        prev_end = args[i - 1][1].args[1]
                    slope_value = -S.One / E * integrate(self.bending_moment() / args[i][0], (x, prev_end, x))
                    recent_segment_slope = prev_slope + slope_value
                    deflection_value = integrate(recent_segment_slope, (x, prev_end, x))
                    if i != len(args) - 1:
                        deflection += (prev_def + deflection_value) * SingularityFunction(x, prev_end, 0) - (prev_def + deflection_value) * SingularityFunction(x, args[i][1].args[1], 0)
                    else:
                        deflection += (prev_def + deflection_value) * SingularityFunction(x, prev_end, 0)
                    prev_slope = slope_value.subs(x, args[i][1].args[1])
                    prev_def = deflection_value.subs(x, args[i][1].args[1])
                return deflection
            base_char = self._base_char
            constants = symbols(base_char + '3:5')
            return S.One / (E * I) * integrate(-integrate(self.bending_moment(), x), x) + constants[0] * x + constants[1]
        elif not self._boundary_conditions['deflection']:
            base_char = self._base_char
            constant = symbols(base_char + '4')
            return integrate(self.slope(), x) + constant
        elif not self._boundary_conditions['slope'] and self._boundary_conditions['deflection']:
            if isinstance(I, Piecewise) and self._composite_type == 'fixed':
                args = I.args
                prev_slope = 0
                prev_def = 0
                prev_end = 0
                deflection = 0
                for i in range(len(args)):
                    if i != 0:
                        prev_end = args[i - 1][1].args[1]
                    slope_value = -S.One / E * integrate(self.bending_moment() / args[i][0], (x, prev_end, x))
                    recent_segment_slope = prev_slope + slope_value
                    deflection_value = integrate(recent_segment_slope, (x, prev_end, x))
                    if i != len(args) - 1:
                        deflection += (prev_def + deflection_value) * SingularityFunction(x, prev_end, 0) - (prev_def + deflection_value) * SingularityFunction(x, args[i][1].args[1], 0)
                    else:
                        deflection += (prev_def + deflection_value) * SingularityFunction(x, prev_end, 0)
                    prev_slope = slope_value.subs(x, args[i][1].args[1])
                    prev_def = deflection_value.subs(x, args[i][1].args[1])
                return deflection
            base_char = self._base_char
            (C3, C4) = symbols(base_char + '3:5')
            slope_curve = -integrate(self.bending_moment(), x) + C3
            deflection_curve = integrate(slope_curve, x) + C4
            bc_eqs = []
            for (position, value) in self._boundary_conditions['deflection']:
                eqs = deflection_curve.subs(x, position) - value
                bc_eqs.append(eqs)
            constants = list(linsolve(bc_eqs, (C3, C4)))
            deflection_curve = deflection_curve.subs({C3: constants[0][0], C4: constants[0][1]})
            return S.One / (E * I) * deflection_curve
        if isinstance(I, Piecewise) and self._composite_type == 'fixed':
            args = I.args
            prev_slope = 0
            prev_def = 0
            prev_end = 0
            deflection = 0
            for i in range(len(args)):
                if i != 0:
                    prev_end = args[i - 1][1].args[1]
                slope_value = S.One / E * integrate(self.bending_moment() / args[i][0], (x, prev_end, x))
                recent_segment_slope = prev_slope + slope_value
                deflection_value = integrate(recent_segment_slope, (x, prev_end, x))
                if i != len(args) - 1:
                    deflection += (prev_def + deflection_value) * SingularityFunction(x, prev_end, 0) - (prev_def + deflection_value) * SingularityFunction(x, args[i][1].args[1], 0)
                else:
                    deflection += (prev_def + deflection_value) * SingularityFunction(x, prev_end, 0)
                prev_slope = slope_value.subs(x, args[i][1].args[1])
                prev_def = deflection_value.subs(x, args[i][1].args[1])
            return deflection
        C4 = Symbol('C4')
        deflection_curve = integrate(self.slope(), x) + C4
        bc_eqs = []
        for (position, value) in self._boundary_conditions['deflection']:
            eqs = deflection_curve.subs(x, position) - value
            bc_eqs.append(eqs)
        constants = list(linsolve(bc_eqs, C4))
        deflection_curve = deflection_curve.subs({C4: constants[0][0]})
        return deflection_curve

    def max_deflection(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns point of max deflection and its corresponding deflection value\n        in a Beam object.\n        '
        slope_curve = Piecewise((float('nan'), self.variable <= 0), (self.slope(), self.variable < self.length), (float('nan'), True))
        points = solve(slope_curve.rewrite(Piecewise), self.variable, domain=S.Reals)
        deflection_curve = self.deflection()
        deflections = [deflection_curve.subs(self.variable, x) for x in points]
        deflections = list(map(abs, deflections))
        if len(deflections) != 0:
            max_def = max(deflections)
            return (points[deflections.index(max_def)], max_def)
        else:
            return None

    def shear_stress(self):
        if False:
            while True:
                i = 10
        '\n        Returns an expression representing the Shear Stress\n        curve of the Beam object.\n        '
        return self.shear_force() / self._area

    def plot_shear_stress(self, subs=None):
        if False:
            i = 10
            return i + 15
        "\n\n        Returns a plot of shear stress present in the beam object.\n\n        Parameters\n        ==========\n        subs : dictionary\n            Python dictionary containing Symbols as key and their\n            corresponding values.\n\n        Examples\n        ========\n        There is a beam of length 8 meters and area of cross section 2 square\n        meters. A constant distributed load of 10 KN/m is applied from half of\n        the beam till the end. There are two simple supports below the beam,\n        one at the starting point and another at the ending point of the beam.\n        A pointload of magnitude 5 KN is also applied from top of the\n        beam, at a distance of 4 meters from the starting point.\n        Take E = 200 GPa and I = 400*(10**-6) meter**4.\n\n        Using the sign convention of downwards forces being positive.\n\n        .. plot::\n            :context: close-figs\n            :format: doctest\n            :include-source: True\n\n            >>> from sympy.physics.continuum_mechanics.beam import Beam\n            >>> from sympy import symbols\n            >>> R1, R2 = symbols('R1, R2')\n            >>> b = Beam(8, 200*(10**9), 400*(10**-6), 2)\n            >>> b.apply_load(5000, 2, -1)\n            >>> b.apply_load(R1, 0, -1)\n            >>> b.apply_load(R2, 8, -1)\n            >>> b.apply_load(10000, 4, 0, end=8)\n            >>> b.bc_deflection = [(0, 0), (8, 0)]\n            >>> b.solve_for_reaction_loads(R1, R2)\n            >>> b.plot_shear_stress()\n            Plot object containing:\n            [0]: cartesian line: 6875*SingularityFunction(x, 0, 0) - 2500*SingularityFunction(x, 2, 0)\n            - 5000*SingularityFunction(x, 4, 1) + 15625*SingularityFunction(x, 8, 0)\n            + 5000*SingularityFunction(x, 8, 1) for x over (0.0, 8.0)\n        "
        shear_stress = self.shear_stress()
        x = self.variable
        length = self.length
        if subs is None:
            subs = {}
        for sym in shear_stress.atoms(Symbol):
            if sym != x and sym not in subs:
                raise ValueError('value of %s was not passed.' % sym)
        if length in subs:
            length = subs[length]
        return plot(shear_stress.subs(subs), (x, 0, length), title='Shear Stress', xlabel='$\\mathrm{x}$', ylabel='$\\tau$', line_color='r')

    def plot_shear_force(self, subs=None):
        if False:
            print('Hello World!')
        "\n\n        Returns a plot for Shear force present in the Beam object.\n\n        Parameters\n        ==========\n        subs : dictionary\n            Python dictionary containing Symbols as key and their\n            corresponding values.\n\n        Examples\n        ========\n        There is a beam of length 8 meters. A constant distributed load of 10 KN/m\n        is applied from half of the beam till the end. There are two simple supports\n        below the beam, one at the starting point and another at the ending point\n        of the beam. A pointload of magnitude 5 KN is also applied from top of the\n        beam, at a distance of 4 meters from the starting point.\n        Take E = 200 GPa and I = 400*(10**-6) meter**4.\n\n        Using the sign convention of downwards forces being positive.\n\n        .. plot::\n            :context: close-figs\n            :format: doctest\n            :include-source: True\n\n            >>> from sympy.physics.continuum_mechanics.beam import Beam\n            >>> from sympy import symbols\n            >>> R1, R2 = symbols('R1, R2')\n            >>> b = Beam(8, 200*(10**9), 400*(10**-6))\n            >>> b.apply_load(5000, 2, -1)\n            >>> b.apply_load(R1, 0, -1)\n            >>> b.apply_load(R2, 8, -1)\n            >>> b.apply_load(10000, 4, 0, end=8)\n            >>> b.bc_deflection = [(0, 0), (8, 0)]\n            >>> b.solve_for_reaction_loads(R1, R2)\n            >>> b.plot_shear_force()\n            Plot object containing:\n            [0]: cartesian line: 13750*SingularityFunction(x, 0, 0) - 5000*SingularityFunction(x, 2, 0)\n            - 10000*SingularityFunction(x, 4, 1) + 31250*SingularityFunction(x, 8, 0)\n            + 10000*SingularityFunction(x, 8, 1) for x over (0.0, 8.0)\n        "
        shear_force = self.shear_force()
        if subs is None:
            subs = {}
        for sym in shear_force.atoms(Symbol):
            if sym == self.variable:
                continue
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(shear_force.subs(subs), (self.variable, 0, length), title='Shear Force', xlabel='$\\mathrm{x}$', ylabel='$\\mathrm{V}$', line_color='g')

    def plot_bending_moment(self, subs=None):
        if False:
            i = 10
            return i + 15
        "\n\n        Returns a plot for Bending moment present in the Beam object.\n\n        Parameters\n        ==========\n        subs : dictionary\n            Python dictionary containing Symbols as key and their\n            corresponding values.\n\n        Examples\n        ========\n        There is a beam of length 8 meters. A constant distributed load of 10 KN/m\n        is applied from half of the beam till the end. There are two simple supports\n        below the beam, one at the starting point and another at the ending point\n        of the beam. A pointload of magnitude 5 KN is also applied from top of the\n        beam, at a distance of 4 meters from the starting point.\n        Take E = 200 GPa and I = 400*(10**-6) meter**4.\n\n        Using the sign convention of downwards forces being positive.\n\n        .. plot::\n            :context: close-figs\n            :format: doctest\n            :include-source: True\n\n            >>> from sympy.physics.continuum_mechanics.beam import Beam\n            >>> from sympy import symbols\n            >>> R1, R2 = symbols('R1, R2')\n            >>> b = Beam(8, 200*(10**9), 400*(10**-6))\n            >>> b.apply_load(5000, 2, -1)\n            >>> b.apply_load(R1, 0, -1)\n            >>> b.apply_load(R2, 8, -1)\n            >>> b.apply_load(10000, 4, 0, end=8)\n            >>> b.bc_deflection = [(0, 0), (8, 0)]\n            >>> b.solve_for_reaction_loads(R1, R2)\n            >>> b.plot_bending_moment()\n            Plot object containing:\n            [0]: cartesian line: 13750*SingularityFunction(x, 0, 1) - 5000*SingularityFunction(x, 2, 1)\n            - 5000*SingularityFunction(x, 4, 2) + 31250*SingularityFunction(x, 8, 1)\n            + 5000*SingularityFunction(x, 8, 2) for x over (0.0, 8.0)\n        "
        bending_moment = self.bending_moment()
        if subs is None:
            subs = {}
        for sym in bending_moment.atoms(Symbol):
            if sym == self.variable:
                continue
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(bending_moment.subs(subs), (self.variable, 0, length), title='Bending Moment', xlabel='$\\mathrm{x}$', ylabel='$\\mathrm{M}$', line_color='b')

    def plot_slope(self, subs=None):
        if False:
            print('Hello World!')
        "\n\n        Returns a plot for slope of deflection curve of the Beam object.\n\n        Parameters\n        ==========\n        subs : dictionary\n            Python dictionary containing Symbols as key and their\n            corresponding values.\n\n        Examples\n        ========\n        There is a beam of length 8 meters. A constant distributed load of 10 KN/m\n        is applied from half of the beam till the end. There are two simple supports\n        below the beam, one at the starting point and another at the ending point\n        of the beam. A pointload of magnitude 5 KN is also applied from top of the\n        beam, at a distance of 4 meters from the starting point.\n        Take E = 200 GPa and I = 400*(10**-6) meter**4.\n\n        Using the sign convention of downwards forces being positive.\n\n        .. plot::\n            :context: close-figs\n            :format: doctest\n            :include-source: True\n\n            >>> from sympy.physics.continuum_mechanics.beam import Beam\n            >>> from sympy import symbols\n            >>> R1, R2 = symbols('R1, R2')\n            >>> b = Beam(8, 200*(10**9), 400*(10**-6))\n            >>> b.apply_load(5000, 2, -1)\n            >>> b.apply_load(R1, 0, -1)\n            >>> b.apply_load(R2, 8, -1)\n            >>> b.apply_load(10000, 4, 0, end=8)\n            >>> b.bc_deflection = [(0, 0), (8, 0)]\n            >>> b.solve_for_reaction_loads(R1, R2)\n            >>> b.plot_slope()\n            Plot object containing:\n            [0]: cartesian line: -8.59375e-5*SingularityFunction(x, 0, 2) + 3.125e-5*SingularityFunction(x, 2, 2)\n            + 2.08333333333333e-5*SingularityFunction(x, 4, 3) - 0.0001953125*SingularityFunction(x, 8, 2)\n            - 2.08333333333333e-5*SingularityFunction(x, 8, 3) + 0.00138541666666667 for x over (0.0, 8.0)\n        "
        slope = self.slope()
        if subs is None:
            subs = {}
        for sym in slope.atoms(Symbol):
            if sym == self.variable:
                continue
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(slope.subs(subs), (self.variable, 0, length), title='Slope', xlabel='$\\mathrm{x}$', ylabel='$\\theta$', line_color='m')

    def plot_deflection(self, subs=None):
        if False:
            print('Hello World!')
        "\n\n        Returns a plot for deflection curve of the Beam object.\n\n        Parameters\n        ==========\n        subs : dictionary\n            Python dictionary containing Symbols as key and their\n            corresponding values.\n\n        Examples\n        ========\n        There is a beam of length 8 meters. A constant distributed load of 10 KN/m\n        is applied from half of the beam till the end. There are two simple supports\n        below the beam, one at the starting point and another at the ending point\n        of the beam. A pointload of magnitude 5 KN is also applied from top of the\n        beam, at a distance of 4 meters from the starting point.\n        Take E = 200 GPa and I = 400*(10**-6) meter**4.\n\n        Using the sign convention of downwards forces being positive.\n\n        .. plot::\n            :context: close-figs\n            :format: doctest\n            :include-source: True\n\n            >>> from sympy.physics.continuum_mechanics.beam import Beam\n            >>> from sympy import symbols\n            >>> R1, R2 = symbols('R1, R2')\n            >>> b = Beam(8, 200*(10**9), 400*(10**-6))\n            >>> b.apply_load(5000, 2, -1)\n            >>> b.apply_load(R1, 0, -1)\n            >>> b.apply_load(R2, 8, -1)\n            >>> b.apply_load(10000, 4, 0, end=8)\n            >>> b.bc_deflection = [(0, 0), (8, 0)]\n            >>> b.solve_for_reaction_loads(R1, R2)\n            >>> b.plot_deflection()\n            Plot object containing:\n            [0]: cartesian line: 0.00138541666666667*x - 2.86458333333333e-5*SingularityFunction(x, 0, 3)\n            + 1.04166666666667e-5*SingularityFunction(x, 2, 3) + 5.20833333333333e-6*SingularityFunction(x, 4, 4)\n            - 6.51041666666667e-5*SingularityFunction(x, 8, 3) - 5.20833333333333e-6*SingularityFunction(x, 8, 4)\n            for x over (0.0, 8.0)\n        "
        deflection = self.deflection()
        if subs is None:
            subs = {}
        for sym in deflection.atoms(Symbol):
            if sym == self.variable:
                continue
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(deflection.subs(subs), (self.variable, 0, length), title='Deflection', xlabel='$\\mathrm{x}$', ylabel='$\\delta$', line_color='r')

    def plot_loading_results(self, subs=None):
        if False:
            return 10
        "\n        Returns a subplot of Shear Force, Bending Moment,\n        Slope and Deflection of the Beam object.\n\n        Parameters\n        ==========\n\n        subs : dictionary\n               Python dictionary containing Symbols as key and their\n               corresponding values.\n\n        Examples\n        ========\n\n        There is a beam of length 8 meters. A constant distributed load of 10 KN/m\n        is applied from half of the beam till the end. There are two simple supports\n        below the beam, one at the starting point and another at the ending point\n        of the beam. A pointload of magnitude 5 KN is also applied from top of the\n        beam, at a distance of 4 meters from the starting point.\n        Take E = 200 GPa and I = 400*(10**-6) meter**4.\n\n        Using the sign convention of downwards forces being positive.\n\n        .. plot::\n            :context: close-figs\n            :format: doctest\n            :include-source: True\n\n            >>> from sympy.physics.continuum_mechanics.beam import Beam\n            >>> from sympy import symbols\n            >>> R1, R2 = symbols('R1, R2')\n            >>> b = Beam(8, 200*(10**9), 400*(10**-6))\n            >>> b.apply_load(5000, 2, -1)\n            >>> b.apply_load(R1, 0, -1)\n            >>> b.apply_load(R2, 8, -1)\n            >>> b.apply_load(10000, 4, 0, end=8)\n            >>> b.bc_deflection = [(0, 0), (8, 0)]\n            >>> b.solve_for_reaction_loads(R1, R2)\n            >>> axes = b.plot_loading_results()\n        "
        length = self.length
        variable = self.variable
        if subs is None:
            subs = {}
        for sym in self.deflection().atoms(Symbol):
            if sym == self.variable:
                continue
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        if length in subs:
            length = subs[length]
        ax1 = plot(self.shear_force().subs(subs), (variable, 0, length), title='Shear Force', xlabel='$\\mathrm{x}$', ylabel='$\\mathrm{V}$', line_color='g', show=False)
        ax2 = plot(self.bending_moment().subs(subs), (variable, 0, length), title='Bending Moment', xlabel='$\\mathrm{x}$', ylabel='$\\mathrm{M}$', line_color='b', show=False)
        ax3 = plot(self.slope().subs(subs), (variable, 0, length), title='Slope', xlabel='$\\mathrm{x}$', ylabel='$\\theta$', line_color='m', show=False)
        ax4 = plot(self.deflection().subs(subs), (variable, 0, length), title='Deflection', xlabel='$\\mathrm{x}$', ylabel='$\\delta$', line_color='r', show=False)
        return PlotGrid(4, 1, ax1, ax2, ax3, ax4)

    def _solve_for_ild_equations(self):
        if False:
            while True:
                i = 10
        '\n\n        Helper function for I.L.D. It takes the unsubstituted\n        copy of the load equation and uses it to calculate shear force and bending\n        moment equations.\n        '
        x = self.variable
        shear_force = -integrate(self._original_load, x)
        bending_moment = integrate(shear_force, x)
        return (shear_force, bending_moment)

    def solve_for_ild_reactions(self, value, *reactions):
        if False:
            return 10
        "\n\n        Determines the Influence Line Diagram equations for reaction\n        forces under the effect of a moving load.\n\n        Parameters\n        ==========\n        value : Integer\n            Magnitude of moving load\n        reactions :\n            The reaction forces applied on the beam.\n\n        Examples\n        ========\n\n        There is a beam of length 10 meters. There are two simple supports\n        below the beam, one at the starting point and another at the ending\n        point of the beam. Calculate the I.L.D. equations for reaction forces\n        under the effect of a moving load of magnitude 1kN.\n\n        Using the sign convention of downwards forces being positive.\n\n        .. plot::\n            :context: close-figs\n            :format: doctest\n            :include-source: True\n\n            >>> from sympy import symbols\n            >>> from sympy.physics.continuum_mechanics.beam import Beam\n            >>> E, I = symbols('E, I')\n            >>> R_0, R_10 = symbols('R_0, R_10')\n            >>> b = Beam(10, E, I)\n            >>> b.apply_support(0, 'roller')\n            >>> b.apply_support(10, 'roller')\n            >>> b.solve_for_ild_reactions(1,R_0,R_10)\n            >>> b.ild_reactions\n            {R_0: x/10 - 1, R_10: -x/10}\n\n        "
        (shear_force, bending_moment) = self._solve_for_ild_equations()
        x = self.variable
        l = self.length
        C3 = Symbol('C3')
        C4 = Symbol('C4')
        shear_curve = limit(shear_force, x, l) - value
        moment_curve = limit(bending_moment, x, l) - value * (l - x)
        slope_eqs = []
        deflection_eqs = []
        slope_curve = integrate(bending_moment, x) + C3
        for (position, value) in self._boundary_conditions['slope']:
            eqs = slope_curve.subs(x, position) - value
            slope_eqs.append(eqs)
        deflection_curve = integrate(slope_curve, x) + C4
        for (position, value) in self._boundary_conditions['deflection']:
            eqs = deflection_curve.subs(x, position) - value
            deflection_eqs.append(eqs)
        solution = list(linsolve([shear_curve, moment_curve] + slope_eqs + deflection_eqs, (C3, C4) + reactions).args[0])
        solution = solution[2:]
        self._ild_reactions = dict(zip(reactions, solution))

    def plot_ild_reactions(self, subs=None):
        if False:
            i = 10
            return i + 15
        "\n\n        Plots the Influence Line Diagram of Reaction Forces\n        under the effect of a moving load. This function\n        should be called after calling solve_for_ild_reactions().\n\n        Parameters\n        ==========\n\n        subs : dictionary\n               Python dictionary containing Symbols as key and their\n               corresponding values.\n\n        Examples\n        ========\n\n        There is a beam of length 10 meters. A point load of magnitude 5KN\n        is also applied from top of the beam, at a distance of 4 meters\n        from the starting point. There are two simple supports below the\n        beam, located at the starting point and at a distance of 7 meters\n        from the starting point. Plot the I.L.D. equations for reactions\n        at both support points under the effect of a moving load\n        of magnitude 1kN.\n\n        Using the sign convention of downwards forces being positive.\n\n        .. plot::\n            :context: close-figs\n            :format: doctest\n            :include-source: True\n\n            >>> from sympy import symbols\n            >>> from sympy.physics.continuum_mechanics.beam import Beam\n            >>> E, I = symbols('E, I')\n            >>> R_0, R_7 = symbols('R_0, R_7')\n            >>> b = Beam(10, E, I)\n            >>> b.apply_support(0, 'roller')\n            >>> b.apply_support(7, 'roller')\n            >>> b.apply_load(5,4,-1)\n            >>> b.solve_for_ild_reactions(1,R_0,R_7)\n            >>> b.ild_reactions\n            {R_0: x/7 - 22/7, R_7: -x/7 - 20/7}\n            >>> b.plot_ild_reactions()\n            PlotGrid object containing:\n            Plot[0]:Plot object containing:\n            [0]: cartesian line: x/7 - 22/7 for x over (0.0, 10.0)\n            Plot[1]:Plot object containing:\n            [0]: cartesian line: -x/7 - 20/7 for x over (0.0, 10.0)\n\n        "
        if not self._ild_reactions:
            raise ValueError('I.L.D. reaction equations not found. Please use solve_for_ild_reactions() to generate the I.L.D. reaction equations.')
        x = self.variable
        ildplots = []
        if subs is None:
            subs = {}
        for reaction in self._ild_reactions:
            for sym in self._ild_reactions[reaction].atoms(Symbol):
                if sym != x and sym not in subs:
                    raise ValueError('Value of %s was not passed.' % sym)
        for sym in self._length.atoms(Symbol):
            if sym != x and sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        for reaction in self._ild_reactions:
            ildplots.append(plot(self._ild_reactions[reaction].subs(subs), (x, 0, self._length.subs(subs)), title='I.L.D. for Reactions', xlabel=x, ylabel=reaction, line_color='blue', show=False))
        return PlotGrid(len(ildplots), 1, *ildplots)

    def solve_for_ild_shear(self, distance, value, *reactions):
        if False:
            while True:
                i = 10
        "\n\n        Determines the Influence Line Diagram equations for shear at a\n        specified point under the effect of a moving load.\n\n        Parameters\n        ==========\n        distance : Integer\n            Distance of the point from the start of the beam\n            for which equations are to be determined\n        value : Integer\n            Magnitude of moving load\n        reactions :\n            The reaction forces applied on the beam.\n\n        Examples\n        ========\n\n        There is a beam of length 12 meters. There are two simple supports\n        below the beam, one at the starting point and another at a distance\n        of 8 meters. Calculate the I.L.D. equations for Shear at a distance\n        of 4 meters under the effect of a moving load of magnitude 1kN.\n\n        Using the sign convention of downwards forces being positive.\n\n        .. plot::\n            :context: close-figs\n            :format: doctest\n            :include-source: True\n\n            >>> from sympy import symbols\n            >>> from sympy.physics.continuum_mechanics.beam import Beam\n            >>> E, I = symbols('E, I')\n            >>> R_0, R_8 = symbols('R_0, R_8')\n            >>> b = Beam(12, E, I)\n            >>> b.apply_support(0, 'roller')\n            >>> b.apply_support(8, 'roller')\n            >>> b.solve_for_ild_reactions(1, R_0, R_8)\n            >>> b.solve_for_ild_shear(4, 1, R_0, R_8)\n            >>> b.ild_shear\n            Piecewise((x/8, x < 4), (x/8 - 1, x > 4))\n\n        "
        x = self.variable
        l = self.length
        (shear_force, _) = self._solve_for_ild_equations()
        shear_curve1 = value - limit(shear_force, x, distance)
        shear_curve2 = limit(shear_force, x, l) - limit(shear_force, x, distance) - value
        for reaction in reactions:
            shear_curve1 = shear_curve1.subs(reaction, self._ild_reactions[reaction])
            shear_curve2 = shear_curve2.subs(reaction, self._ild_reactions[reaction])
        shear_eq = Piecewise((shear_curve1, x < distance), (shear_curve2, x > distance))
        self._ild_shear = shear_eq

    def plot_ild_shear(self, subs=None):
        if False:
            i = 10
            return i + 15
        "\n\n        Plots the Influence Line Diagram for Shear under the effect\n        of a moving load. This function should be called after\n        calling solve_for_ild_shear().\n\n        Parameters\n        ==========\n\n        subs : dictionary\n               Python dictionary containing Symbols as key and their\n               corresponding values.\n\n        Examples\n        ========\n\n        There is a beam of length 12 meters. There are two simple supports\n        below the beam, one at the starting point and another at a distance\n        of 8 meters. Plot the I.L.D. for Shear at a distance\n        of 4 meters under the effect of a moving load of magnitude 1kN.\n\n        Using the sign convention of downwards forces being positive.\n\n        .. plot::\n            :context: close-figs\n            :format: doctest\n            :include-source: True\n\n            >>> from sympy import symbols\n            >>> from sympy.physics.continuum_mechanics.beam import Beam\n            >>> E, I = symbols('E, I')\n            >>> R_0, R_8 = symbols('R_0, R_8')\n            >>> b = Beam(12, E, I)\n            >>> b.apply_support(0, 'roller')\n            >>> b.apply_support(8, 'roller')\n            >>> b.solve_for_ild_reactions(1, R_0, R_8)\n            >>> b.solve_for_ild_shear(4, 1, R_0, R_8)\n            >>> b.ild_shear\n            Piecewise((x/8, x < 4), (x/8 - 1, x > 4))\n            >>> b.plot_ild_shear()\n            Plot object containing:\n            [0]: cartesian line: Piecewise((x/8, x < 4), (x/8 - 1, x > 4)) for x over (0.0, 12.0)\n\n        "
        if not self._ild_shear:
            raise ValueError('I.L.D. shear equation not found. Please use solve_for_ild_shear() to generate the I.L.D. shear equations.')
        x = self.variable
        l = self._length
        if subs is None:
            subs = {}
        for sym in self._ild_shear.atoms(Symbol):
            if sym != x and sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        for sym in self._length.atoms(Symbol):
            if sym != x and sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        return plot(self._ild_shear.subs(subs), (x, 0, l), title='I.L.D. for Shear', xlabel='$\\mathrm{X}$', ylabel='$\\mathrm{V}$', line_color='blue', show=True)

    def solve_for_ild_moment(self, distance, value, *reactions):
        if False:
            print('Hello World!')
        "\n\n        Determines the Influence Line Diagram equations for moment at a\n        specified point under the effect of a moving load.\n\n        Parameters\n        ==========\n        distance : Integer\n            Distance of the point from the start of the beam\n            for which equations are to be determined\n        value : Integer\n            Magnitude of moving load\n        reactions :\n            The reaction forces applied on the beam.\n\n        Examples\n        ========\n\n        There is a beam of length 12 meters. There are two simple supports\n        below the beam, one at the starting point and another at a distance\n        of 8 meters. Calculate the I.L.D. equations for Moment at a distance\n        of 4 meters under the effect of a moving load of magnitude 1kN.\n\n        Using the sign convention of downwards forces being positive.\n\n        .. plot::\n            :context: close-figs\n            :format: doctest\n            :include-source: True\n\n            >>> from sympy import symbols\n            >>> from sympy.physics.continuum_mechanics.beam import Beam\n            >>> E, I = symbols('E, I')\n            >>> R_0, R_8 = symbols('R_0, R_8')\n            >>> b = Beam(12, E, I)\n            >>> b.apply_support(0, 'roller')\n            >>> b.apply_support(8, 'roller')\n            >>> b.solve_for_ild_reactions(1, R_0, R_8)\n            >>> b.solve_for_ild_moment(4, 1, R_0, R_8)\n            >>> b.ild_moment\n            Piecewise((-x/2, x < 4), (x/2 - 4, x > 4))\n\n        "
        x = self.variable
        l = self.length
        (_, moment) = self._solve_for_ild_equations()
        moment_curve1 = value * (distance - x) - limit(moment, x, distance)
        moment_curve2 = limit(moment, x, l) - limit(moment, x, distance) - value * (l - x)
        for reaction in reactions:
            moment_curve1 = moment_curve1.subs(reaction, self._ild_reactions[reaction])
            moment_curve2 = moment_curve2.subs(reaction, self._ild_reactions[reaction])
        moment_eq = Piecewise((moment_curve1, x < distance), (moment_curve2, x > distance))
        self._ild_moment = moment_eq

    def plot_ild_moment(self, subs=None):
        if False:
            while True:
                i = 10
        "\n\n        Plots the Influence Line Diagram for Moment under the effect\n        of a moving load. This function should be called after\n        calling solve_for_ild_moment().\n\n        Parameters\n        ==========\n\n        subs : dictionary\n               Python dictionary containing Symbols as key and their\n               corresponding values.\n\n        Examples\n        ========\n\n        There is a beam of length 12 meters. There are two simple supports\n        below the beam, one at the starting point and another at a distance\n        of 8 meters. Plot the I.L.D. for Moment at a distance\n        of 4 meters under the effect of a moving load of magnitude 1kN.\n\n        Using the sign convention of downwards forces being positive.\n\n        .. plot::\n            :context: close-figs\n            :format: doctest\n            :include-source: True\n\n            >>> from sympy import symbols\n            >>> from sympy.physics.continuum_mechanics.beam import Beam\n            >>> E, I = symbols('E, I')\n            >>> R_0, R_8 = symbols('R_0, R_8')\n            >>> b = Beam(12, E, I)\n            >>> b.apply_support(0, 'roller')\n            >>> b.apply_support(8, 'roller')\n            >>> b.solve_for_ild_reactions(1, R_0, R_8)\n            >>> b.solve_for_ild_moment(4, 1, R_0, R_8)\n            >>> b.ild_moment\n            Piecewise((-x/2, x < 4), (x/2 - 4, x > 4))\n            >>> b.plot_ild_moment()\n            Plot object containing:\n            [0]: cartesian line: Piecewise((-x/2, x < 4), (x/2 - 4, x > 4)) for x over (0.0, 12.0)\n\n        "
        if not self._ild_moment:
            raise ValueError('I.L.D. moment equation not found. Please use solve_for_ild_moment() to generate the I.L.D. moment equations.')
        x = self.variable
        if subs is None:
            subs = {}
        for sym in self._ild_moment.atoms(Symbol):
            if sym != x and sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        for sym in self._length.atoms(Symbol):
            if sym != x and sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        return plot(self._ild_moment.subs(subs), (x, 0, self._length), title='I.L.D. for Moment', xlabel='$\\mathrm{X}$', ylabel='$\\mathrm{M}$', line_color='blue', show=True)

    @doctest_depends_on(modules=('numpy',))
    def draw(self, pictorial=True):
        if False:
            i = 10
            return i + 15
        '\n        Returns a plot object representing the beam diagram of the beam.\n        In particular, the diagram might include:\n\n        * the beam.\n        * vertical black arrows represent point loads and support reaction\n          forces (the latter if they have been added with the ``apply_load``\n          method).\n        * circular arrows represent moments.\n        * shaded areas represent distributed loads.\n        * the support, if ``apply_support`` has been executed.\n        * if a composite beam has been created with the ``join`` method and\n          a hinge has been specified, it will be shown with a white disc.\n\n        The diagram shows positive loads on the upper side of the beam,\n        and negative loads on the lower side. If two or more distributed\n        loads acts along the same direction over the same region, the\n        function will add them up together.\n\n        .. note::\n            The user must be careful while entering load values.\n            The draw function assumes a sign convention which is used\n            for plotting loads.\n            Given a right handed coordinate system with XYZ coordinates,\n            the beam\'s length is assumed to be along the positive X axis.\n            The draw function recognizes positive loads(with n>-2) as loads\n            acting along negative Y direction and positive moments acting\n            along positive Z direction.\n\n        Parameters\n        ==========\n\n        pictorial: Boolean (default=True)\n            Setting ``pictorial=True`` would simply create a pictorial (scaled)\n            view of the beam diagram. On the other hand, ``pictorial=False``\n            would create a beam diagram with the exact dimensions on the plot.\n\n        Examples\n        ========\n\n        .. plot::\n            :context: close-figs\n            :format: doctest\n            :include-source: True\n\n            >>> from sympy.physics.continuum_mechanics.beam import Beam\n            >>> from sympy import symbols\n            >>> P1, P2, M = symbols(\'P1, P2, M\')\n            >>> E, I = symbols(\'E, I\')\n            >>> b = Beam(50, 20, 30)\n            >>> b.apply_load(-10, 2, -1)\n            >>> b.apply_load(15, 26, -1)\n            >>> b.apply_load(P1, 10, -1)\n            >>> b.apply_load(-P2, 40, -1)\n            >>> b.apply_load(90, 5, 0, 23)\n            >>> b.apply_load(10, 30, 1, 50)\n            >>> b.apply_load(M, 15, -2)\n            >>> b.apply_load(-M, 30, -2)\n            >>> b.apply_support(50, "pin")\n            >>> b.apply_support(0, "fixed")\n            >>> b.apply_support(20, "roller")\n            >>> p = b.draw()\n            >>> p  # doctest: +ELLIPSIS\n            Plot object containing:\n            [0]: cartesian line: 25*SingularityFunction(x, 5, 0) - 25*SingularityFunction(x, 23, 0)\n            + SingularityFunction(x, 30, 1) - 20*SingularityFunction(x, 50, 0)\n            - SingularityFunction(x, 50, 1) + 5 for x over (0.0, 50.0)\n            [1]: cartesian line: 5 for x over (0.0, 50.0)\n            ...\n            >>> p.show()\n\n        '
        if not numpy:
            raise ImportError('To use this function numpy module is required')
        loads = list(set(self.applied_loads) - set(self._support_as_loads))
        if not pictorial and any((len(l[0].free_symbols) > 0 and l[2] >= 0 for l in loads)):
            raise ValueError('`pictorial=False` requires numerical distributed loads. Instead, symbolic loads were found. Cannot continue.')
        x = self.variable
        if isinstance(self.length, Expr):
            l = list(self.length.atoms(Symbol))
            l = {i: 10 for i in l}
            length = self.length.subs(l)
        else:
            l = {}
            length = self.length
        height = length / 10
        rectangles = []
        rectangles.append({'xy': (0, 0), 'width': length, 'height': height, 'facecolor': 'brown'})
        (annotations, markers, load_eq, load_eq1, fill) = self._draw_load(pictorial, length, l)
        (support_markers, support_rectangles) = self._draw_supports(length, l)
        rectangles += support_rectangles
        markers += support_markers
        if self._composite_type == 'hinge':
            ratio = self._hinge_position / self.length
            x_pos = float(ratio) * length
            markers += [{'args': [[x_pos], [height / 2]], 'marker': 'o', 'markersize': 6, 'color': 'white'}]
        ylim = (-length, 1.25 * length)
        if fill:
            _min = min(min(fill['y2']), min((r['xy'][1] for r in rectangles)))
            _max = max(max(fill['y1']), max((r['xy'][1] for r in rectangles)))
            if _min < ylim[0] or _max > ylim[1]:
                offset = abs(_max - _min) * 0.1
                ylim = (_min - offset, _max + offset)
        sing_plot = plot(height + load_eq, height + load_eq1, (x, 0, length), xlim=(-height, length + height), ylim=ylim, annotations=annotations, markers=markers, rectangles=rectangles, line_color='brown', fill=fill, axis=False, show=False)
        return sing_plot

    def _is_load_negative(self, load):
        if False:
            i = 10
            return i + 15
        'Try to determine if a load is negative or positive, using\n        expansion and doit if necessary.\n\n        Returns\n        =======\n        True: if the load is negative\n        False: if the load is positive\n        None: if it is indeterminate\n\n        '
        rv = load.is_negative
        if load.is_Atom or rv is not None:
            return rv
        return load.doit().expand().is_negative

    def _draw_load(self, pictorial, length, l):
        if False:
            while True:
                i = 10
        loads = list(set(self.applied_loads) - set(self._support_as_loads))
        height = length / 10
        x = self.variable
        annotations = []
        markers = []
        load_args = []
        scaled_load = 0
        load_args1 = []
        scaled_load1 = 0
        load_eq = S.Zero
        load_eq1 = S.Zero
        fill = None
        warning_head = 'Please, note that this schematic view might not be in agreement with the sign convention used by the Beam class for load-related computations, because it was not possible to determine the sign (hence, the direction) of the following loads:\n'
        warning_body = ''
        for load in loads:
            if l:
                pos = load[1].subs(l)
            else:
                pos = load[1]
            if load[2] == -1:
                iln = self._is_load_negative(load[0])
                if iln is None:
                    warning_body += '* Point load %s located at %s\n' % (load[0], load[1])
                if iln:
                    annotations.append({'text': '', 'xy': (pos, 0), 'xytext': (pos, height - 4 * height), 'arrowprops': {'width': 1.5, 'headlength': 5, 'headwidth': 5, 'facecolor': 'black'}})
                else:
                    annotations.append({'text': '', 'xy': (pos, height), 'xytext': (pos, height * 4), 'arrowprops': {'width': 1.5, 'headlength': 4, 'headwidth': 4, 'facecolor': 'black'}})
            elif load[2] == -2:
                iln = self._is_load_negative(load[0])
                if iln is None:
                    warning_body += '* Moment %s located at %s\n' % (load[0], load[1])
                if self._is_load_negative(load[0]):
                    markers.append({'args': [[pos], [height / 2]], 'marker': '$\\circlearrowright$', 'markersize': 15})
                else:
                    markers.append({'args': [[pos], [height / 2]], 'marker': '$\\circlearrowleft$', 'markersize': 15})
            elif load[2] >= 0:
                (value, start, order, end) = load
                iln = self._is_load_negative(value)
                if iln is None:
                    warning_body += '* Distributed load %s from %s to %s\n' % (value, start, end)
                if not iln:
                    if pictorial:
                        value = 10 ** (1 - order) if order > 0 else length / 2
                    scaled_load += value * SingularityFunction(x, start, order)
                    if end:
                        f2 = value * x ** order if order >= 0 else length / 2 * x ** order
                        for i in range(0, order + 1):
                            scaled_load -= f2.diff(x, i).subs(x, end - start) * SingularityFunction(x, end, i) / factorial(i)
                    if isinstance(scaled_load, Add):
                        load_args = scaled_load.args
                    else:
                        load_args = (scaled_load,)
                    load_eq = Add(*[i.subs(l) for i in load_args])
                else:
                    if pictorial:
                        value = 10 ** (1 - order) if order > 0 else length / 2
                    scaled_load1 += abs(value) * SingularityFunction(x, start, order)
                    if end:
                        f2 = abs(value) * x ** order if order >= 0 else length / 2 * x ** order
                        for i in range(0, order + 1):
                            scaled_load1 -= f2.diff(x, i).subs(x, end - start) * SingularityFunction(x, end, i) / factorial(i)
                    if isinstance(scaled_load1, Add):
                        load_args1 = scaled_load1.args
                    else:
                        load_args1 = (scaled_load1,)
                    load_eq1 = [i.subs(l) for i in load_args1]
                    load_eq1 = -Add(*load_eq1) - height
        if len(warning_body) > 0:
            warnings.warn(warning_head + warning_body)
        xx = numpy.arange(0, float(length), 0.001)
        yy1 = lambdify([x], height + load_eq.rewrite(Piecewise))(xx)
        yy2 = lambdify([x], height + load_eq1.rewrite(Piecewise))(xx)
        if not isinstance(yy1, numpy.ndarray):
            yy1 *= numpy.ones_like(xx)
        if not isinstance(yy2, numpy.ndarray):
            yy2 *= numpy.ones_like(xx)
        fill = {'x': xx, 'y1': yy1, 'y2': yy2, 'color': 'darkkhaki', 'zorder': -1}
        return (annotations, markers, load_eq, load_eq1, fill)

    def _draw_supports(self, length, l):
        if False:
            while True:
                i = 10
        height = float(length / 10)
        support_markers = []
        support_rectangles = []
        for support in self._applied_supports:
            if l:
                pos = support[0].subs(l)
            else:
                pos = support[0]
            if support[1] == 'pin':
                support_markers.append({'args': [pos, [0]], 'marker': 6, 'markersize': 13, 'color': 'black'})
            elif support[1] == 'roller':
                support_markers.append({'args': [pos, [-height / 2.5]], 'marker': 'o', 'markersize': 11, 'color': 'black'})
            elif support[1] == 'fixed':
                if pos == 0:
                    support_rectangles.append({'xy': (0, -3 * height), 'width': -length / 20, 'height': 6 * height + height, 'fill': False, 'hatch': '/////'})
                else:
                    support_rectangles.append({'xy': (length, -3 * height), 'width': length / 20, 'height': 6 * height + height, 'fill': False, 'hatch': '/////'})
        return (support_markers, support_rectangles)

class Beam3D(Beam):
    """
    This class handles loads applied in any direction of a 3D space along
    with unequal values of Second moment along different axes.

    .. note::
       A consistent sign convention must be used while solving a beam
       bending problem; the results will
       automatically follow the chosen sign convention.
       This class assumes that any kind of distributed load/moment is
       applied through out the span of a beam.

    Examples
    ========
    There is a beam of l meters long. A constant distributed load of magnitude q
    is applied along y-axis from start till the end of beam. A constant distributed
    moment of magnitude m is also applied along z-axis from start till the end of beam.
    Beam is fixed at both of its end. So, deflection of the beam at the both ends
    is restricted.

    >>> from sympy.physics.continuum_mechanics.beam import Beam3D
    >>> from sympy import symbols, simplify, collect, factor
    >>> l, E, G, I, A = symbols('l, E, G, I, A')
    >>> b = Beam3D(l, E, G, I, A)
    >>> x, q, m = symbols('x, q, m')
    >>> b.apply_load(q, 0, 0, dir="y")
    >>> b.apply_moment_load(m, 0, -1, dir="z")
    >>> b.shear_force()
    [0, -q*x, 0]
    >>> b.bending_moment()
    [0, 0, -m*x + q*x**2/2]
    >>> b.bc_slope = [(0, [0, 0, 0]), (l, [0, 0, 0])]
    >>> b.bc_deflection = [(0, [0, 0, 0]), (l, [0, 0, 0])]
    >>> b.solve_slope_deflection()
    >>> factor(b.slope())
    [0, 0, x*(-l + x)*(-A*G*l**3*q + 2*A*G*l**2*q*x - 12*E*I*l*q
        - 72*E*I*m + 24*E*I*q*x)/(12*E*I*(A*G*l**2 + 12*E*I))]
    >>> dx, dy, dz = b.deflection()
    >>> dy = collect(simplify(dy), x)
    >>> dx == dz == 0
    True
    >>> dy == (x*(12*E*I*l*(A*G*l**2*q - 2*A*G*l*m + 12*E*I*q)
    ... + x*(A*G*l*(3*l*(A*G*l**2*q - 2*A*G*l*m + 12*E*I*q) + x*(-2*A*G*l**2*q + 4*A*G*l*m - 24*E*I*q))
    ... + A*G*(A*G*l**2 + 12*E*I)*(-2*l**2*q + 6*l*m - 4*m*x + q*x**2)
    ... - 12*E*I*q*(A*G*l**2 + 12*E*I)))/(24*A*E*G*I*(A*G*l**2 + 12*E*I)))
    True

    References
    ==========

    .. [1] https://homes.civil.aau.dk/jc/FemteSemester/Beams3D.pdf

    """

    def __init__(self, length, elastic_modulus, shear_modulus, second_moment, area, variable=Symbol('x')):
        if False:
            while True:
                i = 10
        "Initializes the class.\n\n        Parameters\n        ==========\n        length : Sympifyable\n            A Symbol or value representing the Beam's length.\n        elastic_modulus : Sympifyable\n            A SymPy expression representing the Beam's Modulus of Elasticity.\n            It is a measure of the stiffness of the Beam material.\n        shear_modulus : Sympifyable\n            A SymPy expression representing the Beam's Modulus of rigidity.\n            It is a measure of rigidity of the Beam material.\n        second_moment : Sympifyable or list\n            A list of two elements having SymPy expression representing the\n            Beam's Second moment of area. First value represent Second moment\n            across y-axis and second across z-axis.\n            Single SymPy expression can be passed if both values are same\n        area : Sympifyable\n            A SymPy expression representing the Beam's cross-sectional area\n            in a plane perpendicular to length of the Beam.\n        variable : Symbol, optional\n            A Symbol object that will be used as the variable along the beam\n            while representing the load, shear, moment, slope and deflection\n            curve. By default, it is set to ``Symbol('x')``.\n        "
        super().__init__(length, elastic_modulus, second_moment, variable)
        self.shear_modulus = shear_modulus
        self.area = area
        self._load_vector = [0, 0, 0]
        self._moment_load_vector = [0, 0, 0]
        self._torsion_moment = {}
        self._load_Singularity = [0, 0, 0]
        self._slope = [0, 0, 0]
        self._deflection = [0, 0, 0]
        self._angular_deflection = 0

    @property
    def shear_modulus(self):
        if False:
            return 10
        "Young's Modulus of the Beam. "
        return self._shear_modulus

    @shear_modulus.setter
    def shear_modulus(self, e):
        if False:
            for i in range(10):
                print('nop')
        self._shear_modulus = sympify(e)

    @property
    def second_moment(self):
        if False:
            print('Hello World!')
        'Second moment of area of the Beam. '
        return self._second_moment

    @second_moment.setter
    def second_moment(self, i):
        if False:
            while True:
                i = 10
        if isinstance(i, list):
            i = [sympify(x) for x in i]
            self._second_moment = i
        else:
            self._second_moment = sympify(i)

    @property
    def area(self):
        if False:
            print('Hello World!')
        'Cross-sectional area of the Beam. '
        return self._area

    @area.setter
    def area(self, a):
        if False:
            while True:
                i = 10
        self._area = sympify(a)

    @property
    def load_vector(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a three element list representing the load vector.\n        '
        return self._load_vector

    @property
    def moment_load_vector(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a three element list representing moment loads on Beam.\n        '
        return self._moment_load_vector

    @property
    def boundary_conditions(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns a dictionary of boundary conditions applied on the beam.\n        The dictionary has two keywords namely slope and deflection.\n        The value of each keyword is a list of tuple, where each tuple\n        contains location and value of a boundary condition in the format\n        (location, value). Further each value is a list corresponding to\n        slope or deflection(s) values along three axes at that location.\n\n        Examples\n        ========\n        There is a beam of length 4 meters. The slope at 0 should be 4 along\n        the x-axis and 0 along others. At the other end of beam, deflection\n        along all the three axes should be zero.\n\n        >>> from sympy.physics.continuum_mechanics.beam import Beam3D\n        >>> from sympy import symbols\n        >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')\n        >>> b = Beam3D(30, E, G, I, A, x)\n        >>> b.bc_slope = [(0, (4, 0, 0))]\n        >>> b.bc_deflection = [(4, [0, 0, 0])]\n        >>> b.boundary_conditions\n        {'deflection': [(4, [0, 0, 0])], 'slope': [(0, (4, 0, 0))]}\n\n        Here the deflection of the beam should be ``0`` along all the three axes at ``4``.\n        Similarly, the slope of the beam should be ``4`` along x-axis and ``0``\n        along y and z axis at ``0``.\n        "
        return self._boundary_conditions

    def polar_moment(self):
        if False:
            while True:
                i = 10
        "\n        Returns the polar moment of area of the beam\n        about the X axis with respect to the centroid.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.continuum_mechanics.beam import Beam3D\n        >>> from sympy import symbols\n        >>> l, E, G, I, A = symbols('l, E, G, I, A')\n        >>> b = Beam3D(l, E, G, I, A)\n        >>> b.polar_moment()\n        2*I\n        >>> I1 = [9, 15]\n        >>> b = Beam3D(l, E, G, I1, A)\n        >>> b.polar_moment()\n        24\n        "
        if not iterable(self.second_moment):
            return 2 * self.second_moment
        return sum(self.second_moment)

    def apply_load(self, value, start, order, dir='y'):
        if False:
            while True:
                i = 10
        '\n        This method adds up the force load to a particular beam object.\n\n        Parameters\n        ==========\n        value : Sympifyable\n            The magnitude of an applied load.\n        dir : String\n            Axis along which load is applied.\n        order : Integer\n            The order of the applied load.\n            - For point loads, order=-1\n            - For constant distributed load, order=0\n            - For ramp loads, order=1\n            - For parabolic ramp loads, order=2\n            - ... so on.\n        '
        x = self.variable
        value = sympify(value)
        start = sympify(start)
        order = sympify(order)
        if dir == 'x':
            if not order == -1:
                self._load_vector[0] += value
            self._load_Singularity[0] += value * SingularityFunction(x, start, order)
        elif dir == 'y':
            if not order == -1:
                self._load_vector[1] += value
            self._load_Singularity[1] += value * SingularityFunction(x, start, order)
        else:
            if not order == -1:
                self._load_vector[2] += value
            self._load_Singularity[2] += value * SingularityFunction(x, start, order)

    def apply_moment_load(self, value, start, order, dir='y'):
        if False:
            for i in range(10):
                print('nop')
        '\n        This method adds up the moment loads to a particular beam object.\n\n        Parameters\n        ==========\n        value : Sympifyable\n            The magnitude of an applied moment.\n        dir : String\n            Axis along which moment is applied.\n        order : Integer\n            The order of the applied load.\n            - For point moments, order=-2\n            - For constant distributed moment, order=-1\n            - For ramp moments, order=0\n            - For parabolic ramp moments, order=1\n            - ... so on.\n        '
        x = self.variable
        value = sympify(value)
        start = sympify(start)
        order = sympify(order)
        if dir == 'x':
            if not order == -2:
                self._moment_load_vector[0] += value
            elif start in list(self._torsion_moment):
                self._torsion_moment[start] += value
            else:
                self._torsion_moment[start] = value
            self._load_Singularity[0] += value * SingularityFunction(x, start, order)
        elif dir == 'y':
            if not order == -2:
                self._moment_load_vector[1] += value
            self._load_Singularity[0] += value * SingularityFunction(x, start, order)
        else:
            if not order == -2:
                self._moment_load_vector[2] += value
            self._load_Singularity[0] += value * SingularityFunction(x, start, order)

    def apply_support(self, loc, type='fixed'):
        if False:
            return 10
        if type in ('pin', 'roller'):
            reaction_load = Symbol('R_' + str(loc))
            self._reaction_loads[reaction_load] = reaction_load
            self.bc_deflection.append((loc, [0, 0, 0]))
        else:
            reaction_load = Symbol('R_' + str(loc))
            reaction_moment = Symbol('M_' + str(loc))
            self._reaction_loads[reaction_load] = [reaction_load, reaction_moment]
            self.bc_deflection.append((loc, [0, 0, 0]))
            self.bc_slope.append((loc, [0, 0, 0]))

    def solve_for_reaction_loads(self, *reaction):
        if False:
            return 10
        '\n        Solves for the reaction forces.\n\n        Examples\n        ========\n        There is a beam of length 30 meters. It it supported by rollers at\n        of its end. A constant distributed load of magnitude 8 N is applied\n        from start till its end along y-axis. Another linear load having\n        slope equal to 9 is applied along z-axis.\n\n        >>> from sympy.physics.continuum_mechanics.beam import Beam3D\n        >>> from sympy import symbols\n        >>> l, E, G, I, A, x = symbols(\'l, E, G, I, A, x\')\n        >>> b = Beam3D(30, E, G, I, A, x)\n        >>> b.apply_load(8, start=0, order=0, dir="y")\n        >>> b.apply_load(9*x, start=0, order=0, dir="z")\n        >>> b.bc_deflection = [(0, [0, 0, 0]), (30, [0, 0, 0])]\n        >>> R1, R2, R3, R4 = symbols(\'R1, R2, R3, R4\')\n        >>> b.apply_load(R1, start=0, order=-1, dir="y")\n        >>> b.apply_load(R2, start=30, order=-1, dir="y")\n        >>> b.apply_load(R3, start=0, order=-1, dir="z")\n        >>> b.apply_load(R4, start=30, order=-1, dir="z")\n        >>> b.solve_for_reaction_loads(R1, R2, R3, R4)\n        >>> b.reaction_loads\n        {R1: -120, R2: -120, R3: -1350, R4: -2700}\n        '
        x = self.variable
        l = self.length
        q = self._load_Singularity
        shear_curves = [integrate(load, x) for load in q]
        moment_curves = [integrate(shear, x) for shear in shear_curves]
        for i in range(3):
            react = [r for r in reaction if shear_curves[i].has(r) or moment_curves[i].has(r)]
            if len(react) == 0:
                continue
            shear_curve = limit(shear_curves[i], x, l)
            moment_curve = limit(moment_curves[i], x, l)
            sol = list(linsolve([shear_curve, moment_curve], react).args[0])
            sol_dict = dict(zip(react, sol))
            reaction_loads = self._reaction_loads
            for key in sol_dict:
                if key in reaction_loads and sol_dict[key] != reaction_loads[key]:
                    raise ValueError('Ambiguous solution for %s in different directions.' % key)
            self._reaction_loads.update(sol_dict)

    def shear_force(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a list of three expressions which represents the shear force\n        curve of the Beam object along all three axes.\n        '
        x = self.variable
        q = self._load_vector
        return [integrate(-q[0], x), integrate(-q[1], x), integrate(-q[2], x)]

    def axial_force(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns expression of Axial shear force present inside the Beam object.\n        '
        return self.shear_force()[0]

    def shear_stress(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a list of three expressions which represents the shear stress\n        curve of the Beam object along all three axes.\n        '
        return [self.shear_force()[0] / self._area, self.shear_force()[1] / self._area, self.shear_force()[2] / self._area]

    def axial_stress(self):
        if False:
            while True:
                i = 10
        '\n        Returns expression of Axial stress present inside the Beam object.\n        '
        return self.axial_force() / self._area

    def bending_moment(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a list of three expressions which represents the bending moment\n        curve of the Beam object along all three axes.\n        '
        x = self.variable
        m = self._moment_load_vector
        shear = self.shear_force()
        return [integrate(-m[0], x), integrate(-m[1] + shear[2], x), integrate(-m[2] - shear[1], x)]

    def torsional_moment(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns expression of Torsional moment present inside the Beam object.\n        '
        return self.bending_moment()[0]

    def solve_for_torsion(self):
        if False:
            print('Hello World!')
        "\n        Solves for the angular deflection due to the torsional effects of\n        moments being applied in the x-direction i.e. out of or into the beam.\n\n        Here, a positive torque means the direction of the torque is positive\n        i.e. out of the beam along the beam-axis. Likewise, a negative torque\n        signifies a torque into the beam cross-section.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.continuum_mechanics.beam import Beam3D\n        >>> from sympy import symbols\n        >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')\n        >>> b = Beam3D(20, E, G, I, A, x)\n        >>> b.apply_moment_load(4, 4, -2, dir='x')\n        >>> b.apply_moment_load(4, 8, -2, dir='x')\n        >>> b.apply_moment_load(4, 8, -2, dir='x')\n        >>> b.solve_for_torsion()\n        >>> b.angular_deflection().subs(x, 3)\n        18/(G*I)\n        "
        x = self.variable
        sum_moments = 0
        for point in list(self._torsion_moment):
            sum_moments += self._torsion_moment[point]
        list(self._torsion_moment).sort()
        pointsList = list(self._torsion_moment)
        torque_diagram = Piecewise((sum_moments, x <= pointsList[0]), (0, x >= pointsList[0]))
        for i in range(len(pointsList))[1:]:
            sum_moments -= self._torsion_moment[pointsList[i - 1]]
            torque_diagram += Piecewise((0, x <= pointsList[i - 1]), (sum_moments, x <= pointsList[i]), (0, x >= pointsList[i]))
        integrated_torque_diagram = integrate(torque_diagram)
        self._angular_deflection = integrated_torque_diagram / (self.shear_modulus * self.polar_moment())

    def solve_slope_deflection(self):
        if False:
            i = 10
            return i + 15
        x = self.variable
        l = self.length
        E = self.elastic_modulus
        G = self.shear_modulus
        I = self.second_moment
        if isinstance(I, list):
            (I_y, I_z) = (I[0], I[1])
        else:
            I_y = I_z = I
        A = self._area
        load = self._load_vector
        moment = self._moment_load_vector
        defl = Function('defl')
        theta = Function('theta')
        eq = Derivative(E * A * Derivative(defl(x), x), x) + load[0]
        def_x = dsolve(Eq(eq, 0), defl(x)).args[1]
        C1 = Symbol('C1')
        C2 = Symbol('C2')
        constants = list(linsolve([def_x.subs(x, 0), def_x.subs(x, l)], C1, C2).args[0])
        def_x = def_x.subs({C1: constants[0], C2: constants[1]})
        slope_x = def_x.diff(x)
        self._deflection[0] = def_x
        self._slope[0] = slope_x
        C_i = Symbol('C_i')
        eq1 = Derivative(E * I_z * Derivative(theta(x), x), x) + (integrate(-load[1], x) + C_i) + moment[2]
        slope_z = dsolve(Eq(eq1, 0)).args[1]
        constants = list(linsolve([slope_z.subs(x, 0), slope_z.subs(x, l)], C1, C2).args[0])
        slope_z = slope_z.subs({C1: constants[0], C2: constants[1]})
        eq2 = G * A * Derivative(defl(x), x) + load[1] * x - C_i - G * A * slope_z
        def_y = dsolve(Eq(eq2, 0), defl(x)).args[1]
        constants = list(linsolve([def_y.subs(x, 0), def_y.subs(x, l)], C1, C_i).args[0])
        self._deflection[1] = def_y.subs({C1: constants[0], C_i: constants[1]})
        self._slope[2] = slope_z.subs(C_i, constants[1])
        eq1 = Derivative(E * I_y * Derivative(theta(x), x), x) + (integrate(load[2], x) - C_i) + moment[1]
        slope_y = dsolve(Eq(eq1, 0)).args[1]
        constants = list(linsolve([slope_y.subs(x, 0), slope_y.subs(x, l)], C1, C2).args[0])
        slope_y = slope_y.subs({C1: constants[0], C2: constants[1]})
        eq2 = G * A * Derivative(defl(x), x) + load[2] * x - C_i + G * A * slope_y
        def_z = dsolve(Eq(eq2, 0)).args[1]
        constants = list(linsolve([def_z.subs(x, 0), def_z.subs(x, l)], C1, C_i).args[0])
        self._deflection[2] = def_z.subs({C1: constants[0], C_i: constants[1]})
        self._slope[1] = slope_y.subs(C_i, constants[1])

    def slope(self):
        if False:
            while True:
                i = 10
        '\n        Returns a three element list representing slope of deflection curve\n        along all the three axes.\n        '
        return self._slope

    def deflection(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a three element list representing deflection curve along all\n        the three axes.\n        '
        return self._deflection

    def angular_deflection(self):
        if False:
            while True:
                i = 10
        '\n        Returns a function in x depicting how the angular deflection, due to moments\n        in the x-axis on the beam, varies with x.\n        '
        return self._angular_deflection

    def _plot_shear_force(self, dir, subs=None):
        if False:
            while True:
                i = 10
        shear_force = self.shear_force()
        if dir == 'x':
            dir_num = 0
            color = 'r'
        elif dir == 'y':
            dir_num = 1
            color = 'g'
        elif dir == 'z':
            dir_num = 2
            color = 'b'
        if subs is None:
            subs = {}
        for sym in shear_force[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(shear_force[dir_num].subs(subs), (self.variable, 0, length), show=False, title='Shear Force along %c direction' % dir, xlabel='$\\mathrm{X}$', ylabel='$\\mathrm{V(%c)}$' % dir, line_color=color)

    def plot_shear_force(self, dir='all', subs=None):
        if False:
            return 10
        '\n\n        Returns a plot for Shear force along all three directions\n        present in the Beam object.\n\n        Parameters\n        ==========\n        dir : string (default : "all")\n            Direction along which shear force plot is required.\n            If no direction is specified, all plots are displayed.\n        subs : dictionary\n            Python dictionary containing Symbols as key and their\n            corresponding values.\n\n        Examples\n        ========\n        There is a beam of length 20 meters. It is supported by rollers\n        at both of its ends. A linear load having slope equal to 12 is applied\n        along y-axis. A constant distributed load of magnitude 15 N is\n        applied from start till its end along z-axis.\n\n        .. plot::\n            :context: close-figs\n            :format: doctest\n            :include-source: True\n\n            >>> from sympy.physics.continuum_mechanics.beam import Beam3D\n            >>> from sympy import symbols\n            >>> l, E, G, I, A, x = symbols(\'l, E, G, I, A, x\')\n            >>> b = Beam3D(20, E, G, I, A, x)\n            >>> b.apply_load(15, start=0, order=0, dir="z")\n            >>> b.apply_load(12*x, start=0, order=0, dir="y")\n            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]\n            >>> R1, R2, R3, R4 = symbols(\'R1, R2, R3, R4\')\n            >>> b.apply_load(R1, start=0, order=-1, dir="z")\n            >>> b.apply_load(R2, start=20, order=-1, dir="z")\n            >>> b.apply_load(R3, start=0, order=-1, dir="y")\n            >>> b.apply_load(R4, start=20, order=-1, dir="y")\n            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)\n            >>> b.plot_shear_force()\n            PlotGrid object containing:\n            Plot[0]:Plot object containing:\n            [0]: cartesian line: 0 for x over (0.0, 20.0)\n            Plot[1]:Plot object containing:\n            [0]: cartesian line: -6*x**2 for x over (0.0, 20.0)\n            Plot[2]:Plot object containing:\n            [0]: cartesian line: -15*x for x over (0.0, 20.0)\n\n        '
        dir = dir.lower()
        if dir == 'x':
            Px = self._plot_shear_force('x', subs)
            return Px.show()
        elif dir == 'y':
            Py = self._plot_shear_force('y', subs)
            return Py.show()
        elif dir == 'z':
            Pz = self._plot_shear_force('z', subs)
            return Pz.show()
        else:
            Px = self._plot_shear_force('x', subs)
            Py = self._plot_shear_force('y', subs)
            Pz = self._plot_shear_force('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def _plot_bending_moment(self, dir, subs=None):
        if False:
            for i in range(10):
                print('nop')
        bending_moment = self.bending_moment()
        if dir == 'x':
            dir_num = 0
            color = 'g'
        elif dir == 'y':
            dir_num = 1
            color = 'c'
        elif dir == 'z':
            dir_num = 2
            color = 'm'
        if subs is None:
            subs = {}
        for sym in bending_moment[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(bending_moment[dir_num].subs(subs), (self.variable, 0, length), show=False, title='Bending Moment along %c direction' % dir, xlabel='$\\mathrm{X}$', ylabel='$\\mathrm{M(%c)}$' % dir, line_color=color)

    def plot_bending_moment(self, dir='all', subs=None):
        if False:
            while True:
                i = 10
        '\n\n        Returns a plot for bending moment along all three directions\n        present in the Beam object.\n\n        Parameters\n        ==========\n        dir : string (default : "all")\n            Direction along which bending moment plot is required.\n            If no direction is specified, all plots are displayed.\n        subs : dictionary\n            Python dictionary containing Symbols as key and their\n            corresponding values.\n\n        Examples\n        ========\n        There is a beam of length 20 meters. It is supported by rollers\n        at both of its ends. A linear load having slope equal to 12 is applied\n        along y-axis. A constant distributed load of magnitude 15 N is\n        applied from start till its end along z-axis.\n\n        .. plot::\n            :context: close-figs\n            :format: doctest\n            :include-source: True\n\n            >>> from sympy.physics.continuum_mechanics.beam import Beam3D\n            >>> from sympy import symbols\n            >>> l, E, G, I, A, x = symbols(\'l, E, G, I, A, x\')\n            >>> b = Beam3D(20, E, G, I, A, x)\n            >>> b.apply_load(15, start=0, order=0, dir="z")\n            >>> b.apply_load(12*x, start=0, order=0, dir="y")\n            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]\n            >>> R1, R2, R3, R4 = symbols(\'R1, R2, R3, R4\')\n            >>> b.apply_load(R1, start=0, order=-1, dir="z")\n            >>> b.apply_load(R2, start=20, order=-1, dir="z")\n            >>> b.apply_load(R3, start=0, order=-1, dir="y")\n            >>> b.apply_load(R4, start=20, order=-1, dir="y")\n            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)\n            >>> b.plot_bending_moment()\n            PlotGrid object containing:\n            Plot[0]:Plot object containing:\n            [0]: cartesian line: 0 for x over (0.0, 20.0)\n            Plot[1]:Plot object containing:\n            [0]: cartesian line: -15*x**2/2 for x over (0.0, 20.0)\n            Plot[2]:Plot object containing:\n            [0]: cartesian line: 2*x**3 for x over (0.0, 20.0)\n\n        '
        dir = dir.lower()
        if dir == 'x':
            Px = self._plot_bending_moment('x', subs)
            return Px.show()
        elif dir == 'y':
            Py = self._plot_bending_moment('y', subs)
            return Py.show()
        elif dir == 'z':
            Pz = self._plot_bending_moment('z', subs)
            return Pz.show()
        else:
            Px = self._plot_bending_moment('x', subs)
            Py = self._plot_bending_moment('y', subs)
            Pz = self._plot_bending_moment('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def _plot_slope(self, dir, subs=None):
        if False:
            print('Hello World!')
        slope = self.slope()
        if dir == 'x':
            dir_num = 0
            color = 'b'
        elif dir == 'y':
            dir_num = 1
            color = 'm'
        elif dir == 'z':
            dir_num = 2
            color = 'g'
        if subs is None:
            subs = {}
        for sym in slope[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(slope[dir_num].subs(subs), (self.variable, 0, length), show=False, title='Slope along %c direction' % dir, xlabel='$\\mathrm{X}$', ylabel='$\\mathrm{\\theta(%c)}$' % dir, line_color=color)

    def plot_slope(self, dir='all', subs=None):
        if False:
            i = 10
            return i + 15
        '\n\n        Returns a plot for Slope along all three directions\n        present in the Beam object.\n\n        Parameters\n        ==========\n        dir : string (default : "all")\n            Direction along which Slope plot is required.\n            If no direction is specified, all plots are displayed.\n        subs : dictionary\n            Python dictionary containing Symbols as keys and their\n            corresponding values.\n\n        Examples\n        ========\n        There is a beam of length 20 meters. It is supported by rollers\n        at both of its ends. A linear load having slope equal to 12 is applied\n        along y-axis. A constant distributed load of magnitude 15 N is\n        applied from start till its end along z-axis.\n\n        .. plot::\n            :context: close-figs\n            :format: doctest\n            :include-source: True\n\n            >>> from sympy.physics.continuum_mechanics.beam import Beam3D\n            >>> from sympy import symbols\n            >>> l, E, G, I, A, x = symbols(\'l, E, G, I, A, x\')\n            >>> b = Beam3D(20, 40, 21, 100, 25, x)\n            >>> b.apply_load(15, start=0, order=0, dir="z")\n            >>> b.apply_load(12*x, start=0, order=0, dir="y")\n            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]\n            >>> R1, R2, R3, R4 = symbols(\'R1, R2, R3, R4\')\n            >>> b.apply_load(R1, start=0, order=-1, dir="z")\n            >>> b.apply_load(R2, start=20, order=-1, dir="z")\n            >>> b.apply_load(R3, start=0, order=-1, dir="y")\n            >>> b.apply_load(R4, start=20, order=-1, dir="y")\n            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)\n            >>> b.solve_slope_deflection()\n            >>> b.plot_slope()\n            PlotGrid object containing:\n            Plot[0]:Plot object containing:\n            [0]: cartesian line: 0 for x over (0.0, 20.0)\n            Plot[1]:Plot object containing:\n            [0]: cartesian line: -x**3/1600 + 3*x**2/160 - x/8 for x over (0.0, 20.0)\n            Plot[2]:Plot object containing:\n            [0]: cartesian line: x**4/8000 - 19*x**2/172 + 52*x/43 for x over (0.0, 20.0)\n\n        '
        dir = dir.lower()
        if dir == 'x':
            Px = self._plot_slope('x', subs)
            return Px.show()
        elif dir == 'y':
            Py = self._plot_slope('y', subs)
            return Py.show()
        elif dir == 'z':
            Pz = self._plot_slope('z', subs)
            return Pz.show()
        else:
            Px = self._plot_slope('x', subs)
            Py = self._plot_slope('y', subs)
            Pz = self._plot_slope('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def _plot_deflection(self, dir, subs=None):
        if False:
            while True:
                i = 10
        deflection = self.deflection()
        if dir == 'x':
            dir_num = 0
            color = 'm'
        elif dir == 'y':
            dir_num = 1
            color = 'r'
        elif dir == 'z':
            dir_num = 2
            color = 'c'
        if subs is None:
            subs = {}
        for sym in deflection[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(deflection[dir_num].subs(subs), (self.variable, 0, length), show=False, title='Deflection along %c direction' % dir, xlabel='$\\mathrm{X}$', ylabel='$\\mathrm{\\delta(%c)}$' % dir, line_color=color)

    def plot_deflection(self, dir='all', subs=None):
        if False:
            print('Hello World!')
        '\n\n        Returns a plot for Deflection along all three directions\n        present in the Beam object.\n\n        Parameters\n        ==========\n        dir : string (default : "all")\n            Direction along which deflection plot is required.\n            If no direction is specified, all plots are displayed.\n        subs : dictionary\n            Python dictionary containing Symbols as keys and their\n            corresponding values.\n\n        Examples\n        ========\n        There is a beam of length 20 meters. It is supported by rollers\n        at both of its ends. A linear load having slope equal to 12 is applied\n        along y-axis. A constant distributed load of magnitude 15 N is\n        applied from start till its end along z-axis.\n\n        .. plot::\n            :context: close-figs\n            :format: doctest\n            :include-source: True\n\n            >>> from sympy.physics.continuum_mechanics.beam import Beam3D\n            >>> from sympy import symbols\n            >>> l, E, G, I, A, x = symbols(\'l, E, G, I, A, x\')\n            >>> b = Beam3D(20, 40, 21, 100, 25, x)\n            >>> b.apply_load(15, start=0, order=0, dir="z")\n            >>> b.apply_load(12*x, start=0, order=0, dir="y")\n            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]\n            >>> R1, R2, R3, R4 = symbols(\'R1, R2, R3, R4\')\n            >>> b.apply_load(R1, start=0, order=-1, dir="z")\n            >>> b.apply_load(R2, start=20, order=-1, dir="z")\n            >>> b.apply_load(R3, start=0, order=-1, dir="y")\n            >>> b.apply_load(R4, start=20, order=-1, dir="y")\n            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)\n            >>> b.solve_slope_deflection()\n            >>> b.plot_deflection()\n            PlotGrid object containing:\n            Plot[0]:Plot object containing:\n            [0]: cartesian line: 0 for x over (0.0, 20.0)\n            Plot[1]:Plot object containing:\n            [0]: cartesian line: x**5/40000 - 4013*x**3/90300 + 26*x**2/43 + 1520*x/903 for x over (0.0, 20.0)\n            Plot[2]:Plot object containing:\n            [0]: cartesian line: x**4/6400 - x**3/160 + 27*x**2/560 + 2*x/7 for x over (0.0, 20.0)\n\n\n        '
        dir = dir.lower()
        if dir == 'x':
            Px = self._plot_deflection('x', subs)
            return Px.show()
        elif dir == 'y':
            Py = self._plot_deflection('y', subs)
            return Py.show()
        elif dir == 'z':
            Pz = self._plot_deflection('z', subs)
            return Pz.show()
        else:
            Px = self._plot_deflection('x', subs)
            Py = self._plot_deflection('y', subs)
            Pz = self._plot_deflection('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def plot_loading_results(self, dir='x', subs=None):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        Returns a subplot of Shear Force, Bending Moment,\n        Slope and Deflection of the Beam object along the direction specified.\n\n        Parameters\n        ==========\n\n        dir : string (default : "x")\n               Direction along which plots are required.\n               If no direction is specified, plots along x-axis are displayed.\n        subs : dictionary\n               Python dictionary containing Symbols as key and their\n               corresponding values.\n\n        Examples\n        ========\n        There is a beam of length 20 meters. It is supported by rollers\n        at both of its ends. A linear load having slope equal to 12 is applied\n        along y-axis. A constant distributed load of magnitude 15 N is\n        applied from start till its end along z-axis.\n\n        .. plot::\n            :context: close-figs\n            :format: doctest\n            :include-source: True\n\n            >>> from sympy.physics.continuum_mechanics.beam import Beam3D\n            >>> from sympy import symbols\n            >>> l, E, G, I, A, x = symbols(\'l, E, G, I, A, x\')\n            >>> b = Beam3D(20, E, G, I, A, x)\n            >>> subs = {E:40, G:21, I:100, A:25}\n            >>> b.apply_load(15, start=0, order=0, dir="z")\n            >>> b.apply_load(12*x, start=0, order=0, dir="y")\n            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]\n            >>> R1, R2, R3, R4 = symbols(\'R1, R2, R3, R4\')\n            >>> b.apply_load(R1, start=0, order=-1, dir="z")\n            >>> b.apply_load(R2, start=20, order=-1, dir="z")\n            >>> b.apply_load(R3, start=0, order=-1, dir="y")\n            >>> b.apply_load(R4, start=20, order=-1, dir="y")\n            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)\n            >>> b.solve_slope_deflection()\n            >>> b.plot_loading_results(\'y\',subs)\n            PlotGrid object containing:\n            Plot[0]:Plot object containing:\n            [0]: cartesian line: -6*x**2 for x over (0.0, 20.0)\n            Plot[1]:Plot object containing:\n            [0]: cartesian line: -15*x**2/2 for x over (0.0, 20.0)\n            Plot[2]:Plot object containing:\n            [0]: cartesian line: -x**3/1600 + 3*x**2/160 - x/8 for x over (0.0, 20.0)\n            Plot[3]:Plot object containing:\n            [0]: cartesian line: x**5/40000 - 4013*x**3/90300 + 26*x**2/43 + 1520*x/903 for x over (0.0, 20.0)\n\n        '
        dir = dir.lower()
        if subs is None:
            subs = {}
        ax1 = self._plot_shear_force(dir, subs)
        ax2 = self._plot_bending_moment(dir, subs)
        ax3 = self._plot_slope(dir, subs)
        ax4 = self._plot_deflection(dir, subs)
        return PlotGrid(4, 1, ax1, ax2, ax3, ax4)

    def _plot_shear_stress(self, dir, subs=None):
        if False:
            for i in range(10):
                print('nop')
        shear_stress = self.shear_stress()
        if dir == 'x':
            dir_num = 0
            color = 'r'
        elif dir == 'y':
            dir_num = 1
            color = 'g'
        elif dir == 'z':
            dir_num = 2
            color = 'b'
        if subs is None:
            subs = {}
        for sym in shear_stress[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(shear_stress[dir_num].subs(subs), (self.variable, 0, length), show=False, title='Shear stress along %c direction' % dir, xlabel='$\\mathrm{X}$', ylabel='$\\tau(%c)$' % dir, line_color=color)

    def plot_shear_stress(self, dir='all', subs=None):
        if False:
            print('Hello World!')
        '\n\n        Returns a plot for Shear Stress along all three directions\n        present in the Beam object.\n\n        Parameters\n        ==========\n        dir : string (default : "all")\n            Direction along which shear stress plot is required.\n            If no direction is specified, all plots are displayed.\n        subs : dictionary\n            Python dictionary containing Symbols as key and their\n            corresponding values.\n\n        Examples\n        ========\n        There is a beam of length 20 meters and area of cross section 2 square\n        meters. It is supported by rollers at both of its ends. A linear load having\n        slope equal to 12 is applied along y-axis. A constant distributed load\n        of magnitude 15 N is applied from start till its end along z-axis.\n\n        .. plot::\n            :context: close-figs\n            :format: doctest\n            :include-source: True\n\n            >>> from sympy.physics.continuum_mechanics.beam import Beam3D\n            >>> from sympy import symbols\n            >>> l, E, G, I, A, x = symbols(\'l, E, G, I, A, x\')\n            >>> b = Beam3D(20, E, G, I, 2, x)\n            >>> b.apply_load(15, start=0, order=0, dir="z")\n            >>> b.apply_load(12*x, start=0, order=0, dir="y")\n            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]\n            >>> R1, R2, R3, R4 = symbols(\'R1, R2, R3, R4\')\n            >>> b.apply_load(R1, start=0, order=-1, dir="z")\n            >>> b.apply_load(R2, start=20, order=-1, dir="z")\n            >>> b.apply_load(R3, start=0, order=-1, dir="y")\n            >>> b.apply_load(R4, start=20, order=-1, dir="y")\n            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)\n            >>> b.plot_shear_stress()\n            PlotGrid object containing:\n            Plot[0]:Plot object containing:\n            [0]: cartesian line: 0 for x over (0.0, 20.0)\n            Plot[1]:Plot object containing:\n            [0]: cartesian line: -3*x**2 for x over (0.0, 20.0)\n            Plot[2]:Plot object containing:\n            [0]: cartesian line: -15*x/2 for x over (0.0, 20.0)\n\n        '
        dir = dir.lower()
        if dir == 'x':
            Px = self._plot_shear_stress('x', subs)
            return Px.show()
        elif dir == 'y':
            Py = self._plot_shear_stress('y', subs)
            return Py.show()
        elif dir == 'z':
            Pz = self._plot_shear_stress('z', subs)
            return Pz.show()
        else:
            Px = self._plot_shear_stress('x', subs)
            Py = self._plot_shear_stress('y', subs)
            Pz = self._plot_shear_stress('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def _max_shear_force(self, dir):
        if False:
            while True:
                i = 10
        '\n        Helper function for max_shear_force().\n        '
        dir = dir.lower()
        if dir == 'x':
            dir_num = 0
        elif dir == 'y':
            dir_num = 1
        elif dir == 'z':
            dir_num = 2
        if not self.shear_force()[dir_num]:
            return (0, 0)
        load_curve = Piecewise((float('nan'), self.variable <= 0), (self._load_vector[dir_num], self.variable < self.length), (float('nan'), True))
        points = solve(load_curve.rewrite(Piecewise), self.variable, domain=S.Reals)
        points.append(0)
        points.append(self.length)
        shear_curve = self.shear_force()[dir_num]
        shear_values = [shear_curve.subs(self.variable, x) for x in points]
        shear_values = list(map(abs, shear_values))
        max_shear = max(shear_values)
        return (points[shear_values.index(max_shear)], max_shear)

    def max_shear_force(self):
        if False:
            while True:
                i = 10
        '\n        Returns point of max shear force and its corresponding shear value\n        along all directions in a Beam object as a list.\n        solve_for_reaction_loads() must be called before using this function.\n\n        Examples\n        ========\n        There is a beam of length 20 meters. It is supported by rollers\n        at both of its ends. A linear load having slope equal to 12 is applied\n        along y-axis. A constant distributed load of magnitude 15 N is\n        applied from start till its end along z-axis.\n\n        .. plot::\n            :context: close-figs\n            :format: doctest\n            :include-source: True\n\n            >>> from sympy.physics.continuum_mechanics.beam import Beam3D\n            >>> from sympy import symbols\n            >>> l, E, G, I, A, x = symbols(\'l, E, G, I, A, x\')\n            >>> b = Beam3D(20, 40, 21, 100, 25, x)\n            >>> b.apply_load(15, start=0, order=0, dir="z")\n            >>> b.apply_load(12*x, start=0, order=0, dir="y")\n            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]\n            >>> R1, R2, R3, R4 = symbols(\'R1, R2, R3, R4\')\n            >>> b.apply_load(R1, start=0, order=-1, dir="z")\n            >>> b.apply_load(R2, start=20, order=-1, dir="z")\n            >>> b.apply_load(R3, start=0, order=-1, dir="y")\n            >>> b.apply_load(R4, start=20, order=-1, dir="y")\n            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)\n            >>> b.max_shear_force()\n            [(0, 0), (20, 2400), (20, 300)]\n        '
        max_shear = []
        max_shear.append(self._max_shear_force('x'))
        max_shear.append(self._max_shear_force('y'))
        max_shear.append(self._max_shear_force('z'))
        return max_shear

    def _max_bending_moment(self, dir):
        if False:
            return 10
        '\n        Helper function for max_bending_moment().\n        '
        dir = dir.lower()
        if dir == 'x':
            dir_num = 0
        elif dir == 'y':
            dir_num = 1
        elif dir == 'z':
            dir_num = 2
        if not self.bending_moment()[dir_num]:
            return (0, 0)
        shear_curve = Piecewise((float('nan'), self.variable <= 0), (self.shear_force()[dir_num], self.variable < self.length), (float('nan'), True))
        points = solve(shear_curve.rewrite(Piecewise), self.variable, domain=S.Reals)
        points.append(0)
        points.append(self.length)
        bending_moment_curve = self.bending_moment()[dir_num]
        bending_moments = [bending_moment_curve.subs(self.variable, x) for x in points]
        bending_moments = list(map(abs, bending_moments))
        max_bending_moment = max(bending_moments)
        return (points[bending_moments.index(max_bending_moment)], max_bending_moment)

    def max_bending_moment(self):
        if False:
            return 10
        '\n        Returns point of max bending moment and its corresponding bending moment value\n        along all directions in a Beam object as a list.\n        solve_for_reaction_loads() must be called before using this function.\n\n        Examples\n        ========\n        There is a beam of length 20 meters. It is supported by rollers\n        at both of its ends. A linear load having slope equal to 12 is applied\n        along y-axis. A constant distributed load of magnitude 15 N is\n        applied from start till its end along z-axis.\n\n        .. plot::\n            :context: close-figs\n            :format: doctest\n            :include-source: True\n\n            >>> from sympy.physics.continuum_mechanics.beam import Beam3D\n            >>> from sympy import symbols\n            >>> l, E, G, I, A, x = symbols(\'l, E, G, I, A, x\')\n            >>> b = Beam3D(20, 40, 21, 100, 25, x)\n            >>> b.apply_load(15, start=0, order=0, dir="z")\n            >>> b.apply_load(12*x, start=0, order=0, dir="y")\n            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]\n            >>> R1, R2, R3, R4 = symbols(\'R1, R2, R3, R4\')\n            >>> b.apply_load(R1, start=0, order=-1, dir="z")\n            >>> b.apply_load(R2, start=20, order=-1, dir="z")\n            >>> b.apply_load(R3, start=0, order=-1, dir="y")\n            >>> b.apply_load(R4, start=20, order=-1, dir="y")\n            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)\n            >>> b.max_bending_moment()\n            [(0, 0), (20, 3000), (20, 16000)]\n        '
        max_bmoment = []
        max_bmoment.append(self._max_bending_moment('x'))
        max_bmoment.append(self._max_bending_moment('y'))
        max_bmoment.append(self._max_bending_moment('z'))
        return max_bmoment
    max_bmoment = max_bending_moment

    def _max_deflection(self, dir):
        if False:
            return 10
        '\n        Helper function for max_Deflection()\n        '
        dir = dir.lower()
        if dir == 'x':
            dir_num = 0
        elif dir == 'y':
            dir_num = 1
        elif dir == 'z':
            dir_num = 2
        if not self.deflection()[dir_num]:
            return (0, 0)
        slope_curve = Piecewise((float('nan'), self.variable <= 0), (self.slope()[dir_num], self.variable < self.length), (float('nan'), True))
        points = solve(slope_curve.rewrite(Piecewise), self.variable, domain=S.Reals)
        points.append(0)
        points.append(self._length)
        deflection_curve = self.deflection()[dir_num]
        deflections = [deflection_curve.subs(self.variable, x) for x in points]
        deflections = list(map(abs, deflections))
        max_def = max(deflections)
        return (points[deflections.index(max_def)], max_def)

    def max_deflection(self):
        if False:
            print('Hello World!')
        '\n        Returns point of max deflection and its corresponding deflection value\n        along all directions in a Beam object as a list.\n        solve_for_reaction_loads() and solve_slope_deflection() must be called\n        before using this function.\n\n        Examples\n        ========\n        There is a beam of length 20 meters. It is supported by rollers\n        at both of its ends. A linear load having slope equal to 12 is applied\n        along y-axis. A constant distributed load of magnitude 15 N is\n        applied from start till its end along z-axis.\n\n        .. plot::\n            :context: close-figs\n            :format: doctest\n            :include-source: True\n\n            >>> from sympy.physics.continuum_mechanics.beam import Beam3D\n            >>> from sympy import symbols\n            >>> l, E, G, I, A, x = symbols(\'l, E, G, I, A, x\')\n            >>> b = Beam3D(20, 40, 21, 100, 25, x)\n            >>> b.apply_load(15, start=0, order=0, dir="z")\n            >>> b.apply_load(12*x, start=0, order=0, dir="y")\n            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]\n            >>> R1, R2, R3, R4 = symbols(\'R1, R2, R3, R4\')\n            >>> b.apply_load(R1, start=0, order=-1, dir="z")\n            >>> b.apply_load(R2, start=20, order=-1, dir="z")\n            >>> b.apply_load(R3, start=0, order=-1, dir="y")\n            >>> b.apply_load(R4, start=20, order=-1, dir="y")\n            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)\n            >>> b.solve_slope_deflection()\n            >>> b.max_deflection()\n            [(0, 0), (10, 495/14), (-10 + 10*sqrt(10793)/43, (10 - 10*sqrt(10793)/43)**3/160 - 20/7 + (10 - 10*sqrt(10793)/43)**4/6400 + 20*sqrt(10793)/301 + 27*(10 - 10*sqrt(10793)/43)**2/560)]\n        '
        max_def = []
        max_def.append(self._max_deflection('x'))
        max_def.append(self._max_deflection('y'))
        max_def.append(self._max_deflection('z'))
        return max_def