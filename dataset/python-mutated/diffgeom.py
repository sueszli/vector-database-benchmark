from __future__ import annotations
from typing import Any
from functools import reduce
from itertools import permutations
from sympy.combinatorics import Permutation
from sympy.core import Basic, Expr, Function, diff, Pow, Mul, Add, Lambda, S, Tuple, Dict
from sympy.core.cache import cacheit
from sympy.core.symbol import Symbol, Dummy
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.functions import factorial
from sympy.matrices import ImmutableDenseMatrix as Matrix
from sympy.solvers import solve
from sympy.utilities.exceptions import sympy_deprecation_warning, SymPyDeprecationWarning, ignore_warnings
from sympy.tensor.array import ImmutableDenseNDimArray

class Manifold(Basic):
    """
    A mathematical manifold.

    Explanation
    ===========

    A manifold is a topological space that locally resembles
    Euclidean space near each point [1].
    This class does not provide any means to study the topological
    characteristics of the manifold that it represents, though.

    Parameters
    ==========

    name : str
        The name of the manifold.

    dim : int
        The dimension of the manifold.

    Examples
    ========

    >>> from sympy.diffgeom import Manifold
    >>> m = Manifold('M', 2)
    >>> m
    M
    >>> m.dim
    2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Manifold
    """

    def __new__(cls, name, dim, **kwargs):
        if False:
            return 10
        if not isinstance(name, Str):
            name = Str(name)
        dim = _sympify(dim)
        obj = super().__new__(cls, name, dim)
        obj.patches = _deprecated_list('\n            Manifold.patches is deprecated. The Manifold object is now\n            immutable. Instead use a separate list to keep track of the\n            patches.\n            ', [])
        return obj

    @property
    def name(self):
        if False:
            while True:
                i = 10
        return self.args[0]

    @property
    def dim(self):
        if False:
            for i in range(10):
                print('nop')
        return self.args[1]

class Patch(Basic):
    """
    A patch on a manifold.

    Explanation
    ===========

    Coordinate patch, or patch in short, is a simply-connected open set around
    a point in the manifold [1]. On a manifold one can have many patches that
    do not always include the whole manifold. On these patches coordinate
    charts can be defined that permit the parameterization of any point on the
    patch in terms of a tuple of real numbers (the coordinates).

    This class does not provide any means to study the topological
    characteristics of the patch that it represents.

    Parameters
    ==========

    name : str
        The name of the patch.

    manifold : Manifold
        The manifold on which the patch is defined.

    Examples
    ========

    >>> from sympy.diffgeom import Manifold, Patch
    >>> m = Manifold('M', 2)
    >>> p = Patch('P', m)
    >>> p
    P
    >>> p.dim
    2

    References
    ==========

    .. [1] G. Sussman, J. Wisdom, W. Farr, Functional Differential Geometry
           (2013)

    """

    def __new__(cls, name, manifold, **kwargs):
        if False:
            print('Hello World!')
        if not isinstance(name, Str):
            name = Str(name)
        obj = super().__new__(cls, name, manifold)
        obj.manifold.patches.append(obj)
        obj.coord_systems = _deprecated_list('\n            Patch.coord_systms is deprecated. The Patch class is now\n            immutable. Instead use a separate list to keep track of coordinate\n            systems.\n            ', [])
        return obj

    @property
    def name(self):
        if False:
            print('Hello World!')
        return self.args[0]

    @property
    def manifold(self):
        if False:
            while True:
                i = 10
        return self.args[1]

    @property
    def dim(self):
        if False:
            while True:
                i = 10
        return self.manifold.dim

class CoordSystem(Basic):
    """
    A coordinate system defined on the patch.

    Explanation
    ===========

    Coordinate system is a system that uses one or more coordinates to uniquely
    determine the position of the points or other geometric elements on a
    manifold [1].

    By passing ``Symbols`` to *symbols* parameter, user can define the name and
    assumptions of coordinate symbols of the coordinate system. If not passed,
    these symbols are generated automatically and are assumed to be real valued.

    By passing *relations* parameter, user can define the transform relations of
    coordinate systems. Inverse transformation and indirect transformation can
    be found automatically. If this parameter is not passed, coordinate
    transformation cannot be done.

    Parameters
    ==========

    name : str
        The name of the coordinate system.

    patch : Patch
        The patch where the coordinate system is defined.

    symbols : list of Symbols, optional
        Defines the names and assumptions of coordinate symbols.

    relations : dict, optional
        Key is a tuple of two strings, who are the names of the systems where
        the coordinates transform from and transform to.
        Value is a tuple of the symbols before transformation and a tuple of
        the expressions after transformation.

    Examples
    ========

    We define two-dimensional Cartesian coordinate system and polar coordinate
    system.

    >>> from sympy import symbols, pi, sqrt, atan2, cos, sin
    >>> from sympy.diffgeom import Manifold, Patch, CoordSystem
    >>> m = Manifold('M', 2)
    >>> p = Patch('P', m)
    >>> x, y = symbols('x y', real=True)
    >>> r, theta = symbols('r theta', nonnegative=True)
    >>> relation_dict = {
    ... ('Car2D', 'Pol'): [(x, y), (sqrt(x**2 + y**2), atan2(y, x))],
    ... ('Pol', 'Car2D'): [(r, theta), (r*cos(theta), r*sin(theta))]
    ... }
    >>> Car2D = CoordSystem('Car2D', p, (x, y), relation_dict)
    >>> Pol = CoordSystem('Pol', p, (r, theta), relation_dict)

    ``symbols`` property returns ``CoordinateSymbol`` instances. These symbols
    are not same with the symbols used to construct the coordinate system.

    >>> Car2D
    Car2D
    >>> Car2D.dim
    2
    >>> Car2D.symbols
    (x, y)
    >>> _[0].func
    <class 'sympy.diffgeom.diffgeom.CoordinateSymbol'>

    ``transformation()`` method returns the transformation function from
    one coordinate system to another. ``transform()`` method returns the
    transformed coordinates.

    >>> Car2D.transformation(Pol)
    Lambda((x, y), Matrix([
    [sqrt(x**2 + y**2)],
    [      atan2(y, x)]]))
    >>> Car2D.transform(Pol)
    Matrix([
    [sqrt(x**2 + y**2)],
    [      atan2(y, x)]])
    >>> Car2D.transform(Pol, [1, 2])
    Matrix([
    [sqrt(5)],
    [atan(2)]])

    ``jacobian()`` method returns the Jacobian matrix of coordinate
    transformation between two systems. ``jacobian_determinant()`` method
    returns the Jacobian determinant of coordinate transformation between two
    systems.

    >>> Pol.jacobian(Car2D)
    Matrix([
    [cos(theta), -r*sin(theta)],
    [sin(theta),  r*cos(theta)]])
    >>> Pol.jacobian(Car2D, [1, pi/2])
    Matrix([
    [0, -1],
    [1,  0]])
    >>> Car2D.jacobian_determinant(Pol)
    1/sqrt(x**2 + y**2)
    >>> Car2D.jacobian_determinant(Pol, [1,0])
    1

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Coordinate_system

    """

    def __new__(cls, name, patch, symbols=None, relations={}, **kwargs):
        if False:
            i = 10
            return i + 15
        if not isinstance(name, Str):
            name = Str(name)
        if symbols is None:
            names = kwargs.get('names', None)
            if names is None:
                symbols = Tuple(*[Symbol('%s_%s' % (name.name, i), real=True) for i in range(patch.dim)])
            else:
                sympy_deprecation_warning(f"\nThe 'names' argument to CoordSystem is deprecated. Use 'symbols' instead. That\nis, replace\n\n    CoordSystem(..., names={names})\n\nwith\n\n    CoordSystem(..., symbols=[{', '.join(['Symbol(' + repr(n) + ', real=True)' for n in names])}])\n                    ", deprecated_since_version='1.7', active_deprecations_target='deprecated-diffgeom-mutable')
                symbols = Tuple(*[Symbol(n, real=True) for n in names])
        else:
            syms = []
            for s in symbols:
                if isinstance(s, Symbol):
                    syms.append(Symbol(s.name, **s._assumptions.generator))
                elif isinstance(s, str):
                    sympy_deprecation_warning(f'\n\nPassing a string as the coordinate symbol name to CoordSystem is deprecated.\nPass a Symbol with the appropriate name and assumptions instead.\n\nThat is, replace {s} with Symbol({s!r}, real=True).\n                        ', deprecated_since_version='1.7', active_deprecations_target='deprecated-diffgeom-mutable')
                    syms.append(Symbol(s, real=True))
            symbols = Tuple(*syms)
        rel_temp = {}
        for (k, v) in relations.items():
            (s1, s2) = k
            if not isinstance(s1, Str):
                s1 = Str(s1)
            if not isinstance(s2, Str):
                s2 = Str(s2)
            key = Tuple(s1, s2)
            if isinstance(v, Lambda):
                v = (tuple(v.signature), tuple(v.expr))
            else:
                v = (tuple(v[0]), tuple(v[1]))
            rel_temp[key] = v
        relations = Dict(rel_temp)
        obj = super().__new__(cls, name, patch, symbols, relations)
        obj.transforms = _deprecated_dict("\n            CoordSystem.transforms is deprecated. The CoordSystem class is now\n            immutable. Use the 'relations' keyword argument to the\n            CoordSystems() constructor to specify relations.\n            ", {})
        obj._names = [str(n) for n in symbols]
        obj.patch.coord_systems.append(obj)
        obj._dummies = [Dummy(str(n)) for n in symbols]
        obj._dummy = Dummy()
        return obj

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
        return self.args[0]

    @property
    def patch(self):
        if False:
            i = 10
            return i + 15
        return self.args[1]

    @property
    def manifold(self):
        if False:
            while True:
                i = 10
        return self.patch.manifold

    @property
    def symbols(self):
        if False:
            while True:
                i = 10
        return tuple((CoordinateSymbol(self, i, **s._assumptions.generator) for (i, s) in enumerate(self.args[2])))

    @property
    def relations(self):
        if False:
            i = 10
            return i + 15
        return self.args[3]

    @property
    def dim(self):
        if False:
            return 10
        return self.patch.dim

    def transformation(self, sys):
        if False:
            while True:
                i = 10
        '\n        Return coordinate transformation function from *self* to *sys*.\n\n        Parameters\n        ==========\n\n        sys : CoordSystem\n\n        Returns\n        =======\n\n        sympy.Lambda\n\n        Examples\n        ========\n\n        >>> from sympy.diffgeom.rn import R2_r, R2_p\n        >>> R2_r.transformation(R2_p)\n        Lambda((x, y), Matrix([\n        [sqrt(x**2 + y**2)],\n        [      atan2(y, x)]]))\n\n        '
        signature = self.args[2]
        key = Tuple(self.name, sys.name)
        if self == sys:
            expr = Matrix(self.symbols)
        elif key in self.relations:
            expr = Matrix(self.relations[key][1])
        elif key[::-1] in self.relations:
            expr = Matrix(self._inverse_transformation(sys, self))
        else:
            expr = Matrix(self._indirect_transformation(self, sys))
        return Lambda(signature, expr)

    @staticmethod
    def _solve_inverse(sym1, sym2, exprs, sys1_name, sys2_name):
        if False:
            while True:
                i = 10
        ret = solve([t[0] - t[1] for t in zip(sym2, exprs)], list(sym1), dict=True)
        if len(ret) == 0:
            temp = 'Cannot solve inverse relation from {} to {}.'
            raise NotImplementedError(temp.format(sys1_name, sys2_name))
        elif len(ret) > 1:
            temp = 'Obtained multiple inverse relation from {} to {}.'
            raise ValueError(temp.format(sys1_name, sys2_name))
        return ret[0]

    @classmethod
    def _inverse_transformation(cls, sys1, sys2):
        if False:
            for i in range(10):
                print('nop')
        forward = sys1.transform(sys2)
        inv_results = cls._solve_inverse(sys1.symbols, sys2.symbols, forward, sys1.name, sys2.name)
        signature = tuple(sys1.symbols)
        return [inv_results[s] for s in signature]

    @classmethod
    @cacheit
    def _indirect_transformation(cls, sys1, sys2):
        if False:
            i = 10
            return i + 15
        rel = sys1.relations
        path = cls._dijkstra(sys1, sys2)
        transforms = []
        for (s1, s2) in zip(path, path[1:]):
            if (s1, s2) in rel:
                transforms.append(rel[s1, s2])
            else:
                (sym2, inv_exprs) = rel[s2, s1]
                sym1 = tuple((Dummy() for i in sym2))
                ret = cls._solve_inverse(sym2, sym1, inv_exprs, s2, s1)
                ret = tuple((ret[s] for s in sym2))
                transforms.append((sym1, ret))
        syms = sys1.args[2]
        exprs = syms
        for (newsyms, newexprs) in transforms:
            exprs = tuple((e.subs(zip(newsyms, exprs)) for e in newexprs))
        return exprs

    @staticmethod
    def _dijkstra(sys1, sys2):
        if False:
            i = 10
            return i + 15
        relations = sys1.relations
        graph = {}
        for (s1, s2) in relations.keys():
            if s1 not in graph:
                graph[s1] = {s2}
            else:
                graph[s1].add(s2)
            if s2 not in graph:
                graph[s2] = {s1}
            else:
                graph[s2].add(s1)
        path_dict = {sys: [0, [], 0] for sys in graph}

        def visit(sys):
            if False:
                while True:
                    i = 10
            path_dict[sys][2] = 1
            for newsys in graph[sys]:
                distance = path_dict[sys][0] + 1
                if path_dict[newsys][0] >= distance or not path_dict[newsys][1]:
                    path_dict[newsys][0] = distance
                    path_dict[newsys][1] = list(path_dict[sys][1])
                    path_dict[newsys][1].append(sys)
        visit(sys1.name)
        while True:
            min_distance = max(path_dict.values(), key=lambda x: x[0])[0]
            newsys = None
            for (sys, lst) in path_dict.items():
                if 0 < lst[0] <= min_distance and (not lst[2]):
                    min_distance = lst[0]
                    newsys = sys
            if newsys is None:
                break
            visit(newsys)
        result = path_dict[sys2.name][1]
        result.append(sys2.name)
        if result == [sys2.name]:
            raise KeyError('Two coordinate systems are not connected.')
        return result

    def connect_to(self, to_sys, from_coords, to_exprs, inverse=True, fill_in_gaps=False):
        if False:
            while True:
                i = 10
        sympy_deprecation_warning("\n            The CoordSystem.connect_to() method is deprecated. Instead,\n            generate a new instance of CoordSystem with the 'relations'\n            keyword argument (CoordSystem classes are now immutable).\n            ", deprecated_since_version='1.7', active_deprecations_target='deprecated-diffgeom-mutable')
        (from_coords, to_exprs) = dummyfy(from_coords, to_exprs)
        self.transforms[to_sys] = (Matrix(from_coords), Matrix(to_exprs))
        if inverse:
            to_sys.transforms[self] = self._inv_transf(from_coords, to_exprs)
        if fill_in_gaps:
            self._fill_gaps_in_transformations()

    @staticmethod
    def _inv_transf(from_coords, to_exprs):
        if False:
            print('Hello World!')
        inv_from = [i.as_dummy() for i in from_coords]
        inv_to = solve([t[0] - t[1] for t in zip(inv_from, to_exprs)], list(from_coords), dict=True)[0]
        inv_to = [inv_to[fc] for fc in from_coords]
        return (Matrix(inv_from), Matrix(inv_to))

    @staticmethod
    def _fill_gaps_in_transformations():
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def transform(self, sys, coordinates=None):
        if False:
            i = 10
            return i + 15
        '\n        Return the result of coordinate transformation from *self* to *sys*.\n        If coordinates are not given, coordinate symbols of *self* are used.\n\n        Parameters\n        ==========\n\n        sys : CoordSystem\n\n        coordinates : Any iterable, optional.\n\n        Returns\n        =======\n\n        sympy.ImmutableDenseMatrix containing CoordinateSymbol\n\n        Examples\n        ========\n\n        >>> from sympy.diffgeom.rn import R2_r, R2_p\n        >>> R2_r.transform(R2_p)\n        Matrix([\n        [sqrt(x**2 + y**2)],\n        [      atan2(y, x)]])\n        >>> R2_r.transform(R2_p, [0, 1])\n        Matrix([\n        [   1],\n        [pi/2]])\n\n        '
        if coordinates is None:
            coordinates = self.symbols
        if self != sys:
            transf = self.transformation(sys)
            coordinates = transf(*coordinates)
        else:
            coordinates = Matrix(coordinates)
        return coordinates

    def coord_tuple_transform_to(self, to_sys, coords):
        if False:
            print('Hello World!')
        'Transform ``coords`` to coord system ``to_sys``.'
        sympy_deprecation_warning('\n            The CoordSystem.coord_tuple_transform_to() method is deprecated.\n            Use the CoordSystem.transform() method instead.\n            ', deprecated_since_version='1.7', active_deprecations_target='deprecated-diffgeom-mutable')
        coords = Matrix(coords)
        if self != to_sys:
            with ignore_warnings(SymPyDeprecationWarning):
                transf = self.transforms[to_sys]
            coords = transf[1].subs(list(zip(transf[0], coords)))
        return coords

    def jacobian(self, sys, coordinates=None):
        if False:
            i = 10
            return i + 15
        '\n        Return the jacobian matrix of a transformation on given coordinates.\n        If coordinates are not given, coordinate symbols of *self* are used.\n\n        Parameters\n        ==========\n\n        sys : CoordSystem\n\n        coordinates : Any iterable, optional.\n\n        Returns\n        =======\n\n        sympy.ImmutableDenseMatrix\n\n        Examples\n        ========\n\n        >>> from sympy.diffgeom.rn import R2_r, R2_p\n        >>> R2_p.jacobian(R2_r)\n        Matrix([\n        [cos(theta), -rho*sin(theta)],\n        [sin(theta),  rho*cos(theta)]])\n        >>> R2_p.jacobian(R2_r, [1, 0])\n        Matrix([\n        [1, 0],\n        [0, 1]])\n\n        '
        result = self.transform(sys).jacobian(self.symbols)
        if coordinates is not None:
            result = result.subs(list(zip(self.symbols, coordinates)))
        return result
    jacobian_matrix = jacobian

    def jacobian_determinant(self, sys, coordinates=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the jacobian determinant of a transformation on given\n        coordinates. If coordinates are not given, coordinate symbols of *self*\n        are used.\n\n        Parameters\n        ==========\n\n        sys : CoordSystem\n\n        coordinates : Any iterable, optional.\n\n        Returns\n        =======\n\n        sympy.Expr\n\n        Examples\n        ========\n\n        >>> from sympy.diffgeom.rn import R2_r, R2_p\n        >>> R2_r.jacobian_determinant(R2_p)\n        1/sqrt(x**2 + y**2)\n        >>> R2_r.jacobian_determinant(R2_p, [1, 0])\n        1\n\n        '
        return self.jacobian(sys, coordinates).det()

    def point(self, coords):
        if False:
            while True:
                i = 10
        'Create a ``Point`` with coordinates given in this coord system.'
        return Point(self, coords)

    def point_to_coords(self, point):
        if False:
            print('Hello World!')
        'Calculate the coordinates of a point in this coord system.'
        return point.coords(self)

    def base_scalar(self, coord_index):
        if False:
            print('Hello World!')
        'Return ``BaseScalarField`` that takes a point and returns one of the coordinates.'
        return BaseScalarField(self, coord_index)
    coord_function = base_scalar

    def base_scalars(self):
        if False:
            print('Hello World!')
        'Returns a list of all coordinate functions.\n        For more details see the ``base_scalar`` method of this class.'
        return [self.base_scalar(i) for i in range(self.dim)]
    coord_functions = base_scalars

    def base_vector(self, coord_index):
        if False:
            print('Hello World!')
        'Return a basis vector field.\n        The basis vector field for this coordinate system. It is also an\n        operator on scalar fields.'
        return BaseVectorField(self, coord_index)

    def base_vectors(self):
        if False:
            i = 10
            return i + 15
        'Returns a list of all base vectors.\n        For more details see the ``base_vector`` method of this class.'
        return [self.base_vector(i) for i in range(self.dim)]

    def base_oneform(self, coord_index):
        if False:
            while True:
                i = 10
        'Return a basis 1-form field.\n        The basis one-form field for this coordinate system. It is also an\n        operator on vector fields.'
        return Differential(self.coord_function(coord_index))

    def base_oneforms(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a list of all base oneforms.\n        For more details see the ``base_oneform`` method of this class.'
        return [self.base_oneform(i) for i in range(self.dim)]

class CoordinateSymbol(Symbol):
    """A symbol which denotes an abstract value of i-th coordinate of
    the coordinate system with given context.

    Explanation
    ===========

    Each coordinates in coordinate system are represented by unique symbol,
    such as x, y, z in Cartesian coordinate system.

    You may not construct this class directly. Instead, use `symbols` method
    of CoordSystem.

    Parameters
    ==========

    coord_sys : CoordSystem

    index : integer

    Examples
    ========

    >>> from sympy import symbols, Lambda, Matrix, sqrt, atan2, cos, sin
    >>> from sympy.diffgeom import Manifold, Patch, CoordSystem
    >>> m = Manifold('M', 2)
    >>> p = Patch('P', m)
    >>> x, y = symbols('x y', real=True)
    >>> r, theta = symbols('r theta', nonnegative=True)
    >>> relation_dict = {
    ... ('Car2D', 'Pol'): Lambda((x, y), Matrix([sqrt(x**2 + y**2), atan2(y, x)])),
    ... ('Pol', 'Car2D'): Lambda((r, theta), Matrix([r*cos(theta), r*sin(theta)]))
    ... }
    >>> Car2D = CoordSystem('Car2D', p, [x, y], relation_dict)
    >>> Pol = CoordSystem('Pol', p, [r, theta], relation_dict)
    >>> x, y = Car2D.symbols

    ``CoordinateSymbol`` contains its coordinate symbol and index.

    >>> x.name
    'x'
    >>> x.coord_sys == Car2D
    True
    >>> x.index
    0
    >>> x.is_real
    True

    You can transform ``CoordinateSymbol`` into other coordinate system using
    ``rewrite()`` method.

    >>> x.rewrite(Pol)
    r*cos(theta)
    >>> sqrt(x**2 + y**2).rewrite(Pol).simplify()
    r

    """

    def __new__(cls, coord_sys, index, **assumptions):
        if False:
            print('Hello World!')
        name = coord_sys.args[2][index].name
        obj = super().__new__(cls, name, **assumptions)
        obj.coord_sys = coord_sys
        obj.index = index
        return obj

    def __getnewargs__(self):
        if False:
            print('Hello World!')
        return (self.coord_sys, self.index)

    def _hashable_content(self):
        if False:
            return 10
        return (self.coord_sys, self.index) + tuple(sorted(self.assumptions0.items()))

    def _eval_rewrite(self, rule, args, **hints):
        if False:
            print('Hello World!')
        if isinstance(rule, CoordSystem):
            return rule.transform(self.coord_sys)[self.index]
        return super()._eval_rewrite(rule, args, **hints)

class Point(Basic):
    """Point defined in a coordinate system.

    Explanation
    ===========

    Mathematically, point is defined in the manifold and does not have any coordinates
    by itself. Coordinate system is what imbues the coordinates to the point by coordinate
    chart. However, due to the difficulty of realizing such logic, you must supply
    a coordinate system and coordinates to define a Point here.

    The usage of this object after its definition is independent of the
    coordinate system that was used in order to define it, however due to
    limitations in the simplification routines you can arrive at complicated
    expressions if you use inappropriate coordinate systems.

    Parameters
    ==========

    coord_sys : CoordSystem

    coords : list
        The coordinates of the point.

    Examples
    ========

    >>> from sympy import pi
    >>> from sympy.diffgeom import Point
    >>> from sympy.diffgeom.rn import R2, R2_r, R2_p
    >>> rho, theta = R2_p.symbols

    >>> p = Point(R2_p, [rho, 3*pi/4])

    >>> p.manifold == R2
    True

    >>> p.coords()
    Matrix([
    [   rho],
    [3*pi/4]])
    >>> p.coords(R2_r)
    Matrix([
    [-sqrt(2)*rho/2],
    [ sqrt(2)*rho/2]])

    """

    def __new__(cls, coord_sys, coords, **kwargs):
        if False:
            i = 10
            return i + 15
        coords = Matrix(coords)
        obj = super().__new__(cls, coord_sys, coords)
        obj._coord_sys = coord_sys
        obj._coords = coords
        return obj

    @property
    def patch(self):
        if False:
            print('Hello World!')
        return self._coord_sys.patch

    @property
    def manifold(self):
        if False:
            return 10
        return self._coord_sys.manifold

    @property
    def dim(self):
        if False:
            print('Hello World!')
        return self.manifold.dim

    def coords(self, sys=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Coordinates of the point in given coordinate system. If coordinate system\n        is not passed, it returns the coordinates in the coordinate system in which\n        the poin was defined.\n        '
        if sys is None:
            return self._coords
        else:
            return self._coord_sys.transform(sys, self._coords)

    @property
    def free_symbols(self):
        if False:
            i = 10
            return i + 15
        return self._coords.free_symbols

class BaseScalarField(Expr):
    """Base scalar field over a manifold for a given coordinate system.

    Explanation
    ===========

    A scalar field takes a point as an argument and returns a scalar.
    A base scalar field of a coordinate system takes a point and returns one of
    the coordinates of that point in the coordinate system in question.

    To define a scalar field you need to choose the coordinate system and the
    index of the coordinate.

    The use of the scalar field after its definition is independent of the
    coordinate system in which it was defined, however due to limitations in
    the simplification routines you may arrive at more complicated
    expression if you use unappropriate coordinate systems.
    You can build complicated scalar fields by just building up SymPy
    expressions containing ``BaseScalarField`` instances.

    Parameters
    ==========

    coord_sys : CoordSystem

    index : integer

    Examples
    ========

    >>> from sympy import Function, pi
    >>> from sympy.diffgeom import BaseScalarField
    >>> from sympy.diffgeom.rn import R2_r, R2_p
    >>> rho, _ = R2_p.symbols
    >>> point = R2_p.point([rho, 0])
    >>> fx, fy = R2_r.base_scalars()
    >>> ftheta = BaseScalarField(R2_r, 1)

    >>> fx(point)
    rho
    >>> fy(point)
    0

    >>> (fx**2+fy**2).rcall(point)
    rho**2

    >>> g = Function('g')
    >>> fg = g(ftheta-pi)
    >>> fg.rcall(point)
    g(-pi)

    """
    is_commutative = True

    def __new__(cls, coord_sys, index, **kwargs):
        if False:
            i = 10
            return i + 15
        index = _sympify(index)
        obj = super().__new__(cls, coord_sys, index)
        obj._coord_sys = coord_sys
        obj._index = index
        return obj

    @property
    def coord_sys(self):
        if False:
            return 10
        return self.args[0]

    @property
    def index(self):
        if False:
            print('Hello World!')
        return self.args[1]

    @property
    def patch(self):
        if False:
            while True:
                i = 10
        return self.coord_sys.patch

    @property
    def manifold(self):
        if False:
            while True:
                i = 10
        return self.coord_sys.manifold

    @property
    def dim(self):
        if False:
            return 10
        return self.manifold.dim

    def __call__(self, *args):
        if False:
            while True:
                i = 10
        'Evaluating the field at a point or doing nothing.\n        If the argument is a ``Point`` instance, the field is evaluated at that\n        point. The field is returned itself if the argument is any other\n        object. It is so in order to have working recursive calling mechanics\n        for all fields (check the ``__call__`` method of ``Expr``).\n        '
        point = args[0]
        if len(args) != 1 or not isinstance(point, Point):
            return self
        coords = point.coords(self._coord_sys)
        return simplify(coords[self._index]).doit()
    free_symbols: set[Any] = set()

class BaseVectorField(Expr):
    """Base vector field over a manifold for a given coordinate system.

    Explanation
    ===========

    A vector field is an operator taking a scalar field and returning a
    directional derivative (which is also a scalar field).
    A base vector field is the same type of operator, however the derivation is
    specifically done with respect to a chosen coordinate.

    To define a base vector field you need to choose the coordinate system and
    the index of the coordinate.

    The use of the vector field after its definition is independent of the
    coordinate system in which it was defined, however due to limitations in the
    simplification routines you may arrive at more complicated expression if you
    use unappropriate coordinate systems.

    Parameters
    ==========
    coord_sys : CoordSystem

    index : integer

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.diffgeom.rn import R2_p, R2_r
    >>> from sympy.diffgeom import BaseVectorField
    >>> from sympy import pprint

    >>> x, y = R2_r.symbols
    >>> rho, theta = R2_p.symbols
    >>> fx, fy = R2_r.base_scalars()
    >>> point_p = R2_p.point([rho, theta])
    >>> point_r = R2_r.point([x, y])

    >>> g = Function('g')
    >>> s_field = g(fx, fy)

    >>> v = BaseVectorField(R2_r, 1)
    >>> pprint(v(s_field))
    / d           \\|
    |---(g(x, xi))||
    \\dxi          /|xi=y
    >>> pprint(v(s_field).rcall(point_r).doit())
    d
    --(g(x, y))
    dy
    >>> pprint(v(s_field).rcall(point_p))
    / d                        \\|
    |---(g(rho*cos(theta), xi))||
    \\dxi                       /|xi=rho*sin(theta)

    """
    is_commutative = False

    def __new__(cls, coord_sys, index, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        index = _sympify(index)
        obj = super().__new__(cls, coord_sys, index)
        obj._coord_sys = coord_sys
        obj._index = index
        return obj

    @property
    def coord_sys(self):
        if False:
            return 10
        return self.args[0]

    @property
    def index(self):
        if False:
            return 10
        return self.args[1]

    @property
    def patch(self):
        if False:
            i = 10
            return i + 15
        return self.coord_sys.patch

    @property
    def manifold(self):
        if False:
            i = 10
            return i + 15
        return self.coord_sys.manifold

    @property
    def dim(self):
        if False:
            return 10
        return self.manifold.dim

    def __call__(self, scalar_field):
        if False:
            while True:
                i = 10
        'Apply on a scalar field.\n        The action of a vector field on a scalar field is a directional\n        differentiation.\n        If the argument is not a scalar field an error is raised.\n        '
        if covariant_order(scalar_field) or contravariant_order(scalar_field):
            raise ValueError('Only scalar fields can be supplied as arguments to vector fields.')
        if scalar_field is None:
            return self
        base_scalars = list(scalar_field.atoms(BaseScalarField))
        d_var = self._coord_sys._dummy
        d_funcs = [Function('_#_%s' % i)(d_var) for (i, b) in enumerate(base_scalars)]
        d_result = scalar_field.subs(list(zip(base_scalars, d_funcs)))
        d_result = d_result.diff(d_var)
        coords = self._coord_sys.symbols
        d_funcs_deriv = [f.diff(d_var) for f in d_funcs]
        d_funcs_deriv_sub = []
        for b in base_scalars:
            jac = self._coord_sys.jacobian(b._coord_sys, coords)
            d_funcs_deriv_sub.append(jac[b._index, self._index])
        d_result = d_result.subs(list(zip(d_funcs_deriv, d_funcs_deriv_sub)))
        result = d_result.subs(list(zip(d_funcs, base_scalars)))
        result = result.subs(list(zip(coords, self._coord_sys.coord_functions())))
        return result.doit()

def _find_coords(expr):
    if False:
        for i in range(10):
            print('nop')
    fields = expr.atoms(BaseScalarField, BaseVectorField)
    result = set()
    for f in fields:
        result.add(f._coord_sys)
    return result

class Commutator(Expr):
    """Commutator of two vector fields.

    Explanation
    ===========

    The commutator of two vector fields `v_1` and `v_2` is defined as the
    vector field `[v_1, v_2]` that evaluated on each scalar field `f` is equal
    to `v_1(v_2(f)) - v_2(v_1(f))`.

    Examples
    ========


    >>> from sympy.diffgeom.rn import R2_p, R2_r
    >>> from sympy.diffgeom import Commutator
    >>> from sympy import simplify

    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> e_r = R2_p.base_vector(0)

    >>> c_xy = Commutator(e_x, e_y)
    >>> c_xr = Commutator(e_x, e_r)
    >>> c_xy
    0

    Unfortunately, the current code is not able to compute everything:

    >>> c_xr
    Commutator(e_x, e_rho)
    >>> simplify(c_xr(fy**2))
    -2*cos(theta)*y**2/(x**2 + y**2)

    """

    def __new__(cls, v1, v2):
        if False:
            print('Hello World!')
        if covariant_order(v1) or contravariant_order(v1) != 1 or covariant_order(v2) or (contravariant_order(v2) != 1):
            raise ValueError('Only commutators of vector fields are supported.')
        if v1 == v2:
            return S.Zero
        coord_sys = set().union(*[_find_coords(v) for v in (v1, v2)])
        if len(coord_sys) == 1:
            if all((isinstance(v, BaseVectorField) for v in (v1, v2))):
                return S.Zero
            (bases_1, bases_2) = [list(v.atoms(BaseVectorField)) for v in (v1, v2)]
            coeffs_1 = [v1.expand().coeff(b) for b in bases_1]
            coeffs_2 = [v2.expand().coeff(b) for b in bases_2]
            res = 0
            for (c1, b1) in zip(coeffs_1, bases_1):
                for (c2, b2) in zip(coeffs_2, bases_2):
                    res += c1 * b1(c2) * b2 - c2 * b2(c1) * b1
            return res
        else:
            obj = super().__new__(cls, v1, v2)
            obj._v1 = v1
            obj._v2 = v2
            return obj

    @property
    def v1(self):
        if False:
            return 10
        return self.args[0]

    @property
    def v2(self):
        if False:
            print('Hello World!')
        return self.args[1]

    def __call__(self, scalar_field):
        if False:
            return 10
        'Apply on a scalar field.\n        If the argument is not a scalar field an error is raised.\n        '
        return self.v1(self.v2(scalar_field)) - self.v2(self.v1(scalar_field))

class Differential(Expr):
    """Return the differential (exterior derivative) of a form field.

    Explanation
    ===========

    The differential of a form (i.e. the exterior derivative) has a complicated
    definition in the general case.
    The differential `df` of the 0-form `f` is defined for any vector field `v`
    as `df(v) = v(f)`.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.diffgeom.rn import R2_r
    >>> from sympy.diffgeom import Differential
    >>> from sympy import pprint

    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> g = Function('g')
    >>> s_field = g(fx, fy)
    >>> dg = Differential(s_field)

    >>> dg
    d(g(x, y))
    >>> pprint(dg(e_x))
    / d           \\|
    |---(g(xi, y))||
    \\dxi          /|xi=x
    >>> pprint(dg(e_y))
    / d           \\|
    |---(g(x, xi))||
    \\dxi          /|xi=y

    Applying the exterior derivative operator twice always results in:

    >>> Differential(dg)
    0
    """
    is_commutative = False

    def __new__(cls, form_field):
        if False:
            print('Hello World!')
        if contravariant_order(form_field):
            raise ValueError('A vector field was supplied as an argument to Differential.')
        if isinstance(form_field, Differential):
            return S.Zero
        else:
            obj = super().__new__(cls, form_field)
            obj._form_field = form_field
            return obj

    @property
    def form_field(self):
        if False:
            i = 10
            return i + 15
        return self.args[0]

    def __call__(self, *vector_fields):
        if False:
            for i in range(10):
                print('nop')
        'Apply on a list of vector_fields.\n\n        Explanation\n        ===========\n\n        If the number of vector fields supplied is not equal to 1 + the order of\n        the form field inside the differential the result is undefined.\n\n        For 1-forms (i.e. differentials of scalar fields) the evaluation is\n        done as `df(v)=v(f)`. However if `v` is ``None`` instead of a vector\n        field, the differential is returned unchanged. This is done in order to\n        permit partial contractions for higher forms.\n\n        In the general case the evaluation is done by applying the form field\n        inside the differential on a list with one less elements than the number\n        of elements in the original list. Lowering the number of vector fields\n        is achieved through replacing each pair of fields by their\n        commutator.\n\n        If the arguments are not vectors or ``None``s an error is raised.\n        '
        if any(((contravariant_order(a) != 1 or covariant_order(a)) and a is not None for a in vector_fields)):
            raise ValueError('The arguments supplied to Differential should be vector fields or Nones.')
        k = len(vector_fields)
        if k == 1:
            if vector_fields[0]:
                return vector_fields[0].rcall(self._form_field)
            return self
        else:
            f = self._form_field
            v = vector_fields
            ret = 0
            for i in range(k):
                t = v[i].rcall(f.rcall(*v[:i] + v[i + 1:]))
                ret += (-1) ** i * t
                for j in range(i + 1, k):
                    c = Commutator(v[i], v[j])
                    if c:
                        t = f.rcall(*(c,) + v[:i] + v[i + 1:j] + v[j + 1:])
                        ret += (-1) ** (i + j) * t
            return ret

class TensorProduct(Expr):
    """Tensor product of forms.

    Explanation
    ===========

    The tensor product permits the creation of multilinear functionals (i.e.
    higher order tensors) out of lower order fields (e.g. 1-forms and vector
    fields). However, the higher tensors thus created lack the interesting
    features provided by the other type of product, the wedge product, namely
    they are not antisymmetric and hence are not form fields.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r
    >>> from sympy.diffgeom import TensorProduct

    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()

    >>> TensorProduct(dx, dy)(e_x, e_y)
    1
    >>> TensorProduct(dx, dy)(e_y, e_x)
    0
    >>> TensorProduct(dx, fx*dy)(fx*e_x, e_y)
    x**2
    >>> TensorProduct(e_x, e_y)(fx**2, fy**2)
    4*x*y
    >>> TensorProduct(e_y, dx)(fy)
    dx

    You can nest tensor products.

    >>> tp1 = TensorProduct(dx, dy)
    >>> TensorProduct(tp1, dx)(e_x, e_y, e_x)
    1

    You can make partial contraction for instance when 'raising an index'.
    Putting ``None`` in the second argument of ``rcall`` means that the
    respective position in the tensor product is left as it is.

    >>> TP = TensorProduct
    >>> metric = TP(dx, dx) + 3*TP(dy, dy)
    >>> metric.rcall(e_y, None)
    3*dy

    Or automatically pad the args with ``None`` without specifying them.

    >>> metric.rcall(e_y)
    3*dy

    """

    def __new__(cls, *args):
        if False:
            while True:
                i = 10
        scalar = Mul(*[m for m in args if covariant_order(m) + contravariant_order(m) == 0])
        multifields = [m for m in args if covariant_order(m) + contravariant_order(m)]
        if multifields:
            if len(multifields) == 1:
                return scalar * multifields[0]
            return scalar * super().__new__(cls, *multifields)
        else:
            return scalar

    def __call__(self, *fields):
        if False:
            i = 10
            return i + 15
        "Apply on a list of fields.\n\n        If the number of input fields supplied is not equal to the order of\n        the tensor product field, the list of arguments is padded with ``None``'s.\n\n        The list of arguments is divided in sublists depending on the order of\n        the forms inside the tensor product. The sublists are provided as\n        arguments to these forms and the resulting expressions are given to the\n        constructor of ``TensorProduct``.\n\n        "
        tot_order = covariant_order(self) + contravariant_order(self)
        tot_args = len(fields)
        if tot_args != tot_order:
            fields = list(fields) + [None] * (tot_order - tot_args)
        orders = [covariant_order(f) + contravariant_order(f) for f in self._args]
        indices = [sum(orders[:i + 1]) for i in range(len(orders) - 1)]
        fields = [fields[i:j] for (i, j) in zip([0] + indices, indices + [None])]
        multipliers = [t[0].rcall(*t[1]) for t in zip(self._args, fields)]
        return TensorProduct(*multipliers)

class WedgeProduct(TensorProduct):
    """Wedge product of forms.

    Explanation
    ===========

    In the context of integration only completely antisymmetric forms make
    sense. The wedge product permits the creation of such forms.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r
    >>> from sympy.diffgeom import WedgeProduct

    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()

    >>> WedgeProduct(dx, dy)(e_x, e_y)
    1
    >>> WedgeProduct(dx, dy)(e_y, e_x)
    -1
    >>> WedgeProduct(dx, fx*dy)(fx*e_x, e_y)
    x**2
    >>> WedgeProduct(e_x, e_y)(fy, None)
    -e_x

    You can nest wedge products.

    >>> wp1 = WedgeProduct(dx, dy)
    >>> WedgeProduct(wp1, dx)(e_x, e_y, e_x)
    0

    """

    def __call__(self, *fields):
        if False:
            for i in range(10):
                print('nop')
        'Apply on a list of vector_fields.\n        The expression is rewritten internally in terms of tensor products and evaluated.'
        orders = (covariant_order(e) + contravariant_order(e) for e in self.args)
        mul = 1 / Mul(*(factorial(o) for o in orders))
        perms = permutations(fields)
        perms_par = (Permutation(p).signature() for p in permutations(range(len(fields))))
        tensor_prod = TensorProduct(*self.args)
        return mul * Add(*[tensor_prod(*p[0]) * p[1] for p in zip(perms, perms_par)])

class LieDerivative(Expr):
    """Lie derivative with respect to a vector field.

    Explanation
    ===========

    The transport operator that defines the Lie derivative is the pushforward of
    the field to be derived along the integral curve of the field with respect
    to which one derives.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r, R2_p
    >>> from sympy.diffgeom import (LieDerivative, TensorProduct)

    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> e_rho, e_theta = R2_p.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()

    >>> LieDerivative(e_x, fy)
    0
    >>> LieDerivative(e_x, fx)
    1
    >>> LieDerivative(e_x, e_x)
    0

    The Lie derivative of a tensor field by another tensor field is equal to
    their commutator:

    >>> LieDerivative(e_x, e_rho)
    Commutator(e_x, e_rho)
    >>> LieDerivative(e_x + e_y, fx)
    1

    >>> tp = TensorProduct(dx, dy)
    >>> LieDerivative(e_x, tp)
    LieDerivative(e_x, TensorProduct(dx, dy))
    >>> LieDerivative(e_x, tp)
    LieDerivative(e_x, TensorProduct(dx, dy))

    """

    def __new__(cls, v_field, expr):
        if False:
            for i in range(10):
                print('nop')
        expr_form_ord = covariant_order(expr)
        if contravariant_order(v_field) != 1 or covariant_order(v_field):
            raise ValueError('Lie derivatives are defined only with respect to vector fields. The supplied argument was not a vector field.')
        if expr_form_ord > 0:
            obj = super().__new__(cls, v_field, expr)
            obj._v_field = v_field
            obj._expr = expr
            return obj
        if expr.atoms(BaseVectorField):
            return Commutator(v_field, expr)
        else:
            return v_field.rcall(expr)

    @property
    def v_field(self):
        if False:
            for i in range(10):
                print('nop')
        return self.args[0]

    @property
    def expr(self):
        if False:
            while True:
                i = 10
        return self.args[1]

    def __call__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        v = self.v_field
        expr = self.expr
        lead_term = v(expr(*args))
        rest = Add(*[Mul(*args[:i] + (Commutator(v, args[i]),) + args[i + 1:]) for i in range(len(args))])
        return lead_term - rest

class BaseCovarDerivativeOp(Expr):
    """Covariant derivative operator with respect to a base vector.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r
    >>> from sympy.diffgeom import BaseCovarDerivativeOp
    >>> from sympy.diffgeom import metric_to_Christoffel_2nd, TensorProduct

    >>> TP = TensorProduct
    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()

    >>> ch = metric_to_Christoffel_2nd(TP(dx, dx) + TP(dy, dy))
    >>> ch
    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    >>> cvd = BaseCovarDerivativeOp(R2_r, 0, ch)
    >>> cvd(fx)
    1
    >>> cvd(fx*e_x)
    e_x
    """

    def __new__(cls, coord_sys, index, christoffel):
        if False:
            for i in range(10):
                print('nop')
        index = _sympify(index)
        christoffel = ImmutableDenseNDimArray(christoffel)
        obj = super().__new__(cls, coord_sys, index, christoffel)
        obj._coord_sys = coord_sys
        obj._index = index
        obj._christoffel = christoffel
        return obj

    @property
    def coord_sys(self):
        if False:
            print('Hello World!')
        return self.args[0]

    @property
    def index(self):
        if False:
            i = 10
            return i + 15
        return self.args[1]

    @property
    def christoffel(self):
        if False:
            print('Hello World!')
        return self.args[2]

    def __call__(self, field):
        if False:
            while True:
                i = 10
        'Apply on a scalar field.\n\n        The action of a vector field on a scalar field is a directional\n        differentiation.\n        If the argument is not a scalar field the behaviour is undefined.\n        '
        if covariant_order(field) != 0:
            raise NotImplementedError()
        field = vectors_in_basis(field, self._coord_sys)
        wrt_vector = self._coord_sys.base_vector(self._index)
        wrt_scalar = self._coord_sys.coord_function(self._index)
        vectors = list(field.atoms(BaseVectorField))
        d_funcs = [Function('_#_%s' % i)(wrt_scalar) for (i, b) in enumerate(vectors)]
        d_result = field.subs(list(zip(vectors, d_funcs)))
        d_result = wrt_vector(d_result)
        d_result = d_result.subs(list(zip(d_funcs, vectors)))
        derivs = []
        for v in vectors:
            d = Add(*[self._christoffel[k, wrt_vector._index, v._index] * v._coord_sys.base_vector(k) for k in range(v._coord_sys.dim)])
            derivs.append(d)
        to_subs = [wrt_vector(d) for d in d_funcs]
        result = d_result.subs(list(zip(to_subs, derivs)))
        result = result.subs(list(zip(d_funcs, vectors)))
        return result.doit()

class CovarDerivativeOp(Expr):
    """Covariant derivative operator.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r
    >>> from sympy.diffgeom import CovarDerivativeOp
    >>> from sympy.diffgeom import metric_to_Christoffel_2nd, TensorProduct
    >>> TP = TensorProduct
    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()
    >>> ch = metric_to_Christoffel_2nd(TP(dx, dx) + TP(dy, dy))

    >>> ch
    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    >>> cvd = CovarDerivativeOp(fx*e_x, ch)
    >>> cvd(fx)
    x
    >>> cvd(fx*e_x)
    x*e_x

    """

    def __new__(cls, wrt, christoffel):
        if False:
            print('Hello World!')
        if len({v._coord_sys for v in wrt.atoms(BaseVectorField)}) > 1:
            raise NotImplementedError()
        if contravariant_order(wrt) != 1 or covariant_order(wrt):
            raise ValueError('Covariant derivatives are defined only with respect to vector fields. The supplied argument was not a vector field.')
        christoffel = ImmutableDenseNDimArray(christoffel)
        obj = super().__new__(cls, wrt, christoffel)
        obj._wrt = wrt
        obj._christoffel = christoffel
        return obj

    @property
    def wrt(self):
        if False:
            return 10
        return self.args[0]

    @property
    def christoffel(self):
        if False:
            while True:
                i = 10
        return self.args[1]

    def __call__(self, field):
        if False:
            for i in range(10):
                print('nop')
        vectors = list(self._wrt.atoms(BaseVectorField))
        base_ops = [BaseCovarDerivativeOp(v._coord_sys, v._index, self._christoffel) for v in vectors]
        return self._wrt.subs(list(zip(vectors, base_ops))).rcall(field)

def intcurve_series(vector_field, param, start_point, n=6, coord_sys=None, coeffs=False):
    if False:
        while True:
            i = 10
    'Return the series expansion for an integral curve of the field.\n\n    Explanation\n    ===========\n\n    Integral curve is a function `\\gamma` taking a parameter in `R` to a point\n    in the manifold. It verifies the equation:\n\n    `V(f)\\big(\\gamma(t)\\big) = \\frac{d}{dt}f\\big(\\gamma(t)\\big)`\n\n    where the given ``vector_field`` is denoted as `V`. This holds for any\n    value `t` for the parameter and any scalar field `f`.\n\n    This equation can also be decomposed of a basis of coordinate functions\n    `V(f_i)\\big(\\gamma(t)\\big) = \\frac{d}{dt}f_i\\big(\\gamma(t)\\big) \\quad \\forall i`\n\n    This function returns a series expansion of `\\gamma(t)` in terms of the\n    coordinate system ``coord_sys``. The equations and expansions are necessarily\n    done in coordinate-system-dependent way as there is no other way to\n    represent movement between points on the manifold (i.e. there is no such\n    thing as a difference of points for a general manifold).\n\n    Parameters\n    ==========\n    vector_field\n        the vector field for which an integral curve will be given\n\n    param\n        the argument of the function `\\gamma` from R to the curve\n\n    start_point\n        the point which corresponds to `\\gamma(0)`\n\n    n\n        the order to which to expand\n\n    coord_sys\n        the coordinate system in which to expand\n        coeffs (default False) - if True return a list of elements of the expansion\n\n    Examples\n    ========\n\n    Use the predefined R2 manifold:\n\n    >>> from sympy.abc import t, x, y\n    >>> from sympy.diffgeom.rn import R2_p, R2_r\n    >>> from sympy.diffgeom import intcurve_series\n\n    Specify a starting point and a vector field:\n\n    >>> start_point = R2_r.point([x, y])\n    >>> vector_field = R2_r.e_x\n\n    Calculate the series:\n\n    >>> intcurve_series(vector_field, t, start_point, n=3)\n    Matrix([\n    [t + x],\n    [    y]])\n\n    Or get the elements of the expansion in a list:\n\n    >>> series = intcurve_series(vector_field, t, start_point, n=3, coeffs=True)\n    >>> series[0]\n    Matrix([\n    [x],\n    [y]])\n    >>> series[1]\n    Matrix([\n    [t],\n    [0]])\n    >>> series[2]\n    Matrix([\n    [0],\n    [0]])\n\n    The series in the polar coordinate system:\n\n    >>> series = intcurve_series(vector_field, t, start_point,\n    ...             n=3, coord_sys=R2_p, coeffs=True)\n    >>> series[0]\n    Matrix([\n    [sqrt(x**2 + y**2)],\n    [      atan2(y, x)]])\n    >>> series[1]\n    Matrix([\n    [t*x/sqrt(x**2 + y**2)],\n    [   -t*y/(x**2 + y**2)]])\n    >>> series[2]\n    Matrix([\n    [t**2*(-x**2/(x**2 + y**2)**(3/2) + 1/sqrt(x**2 + y**2))/2],\n    [                                t**2*x*y/(x**2 + y**2)**2]])\n\n    See Also\n    ========\n\n    intcurve_diffequ\n\n    '
    if contravariant_order(vector_field) != 1 or covariant_order(vector_field):
        raise ValueError('The supplied field was not a vector field.')

    def iter_vfield(scalar_field, i):
        if False:
            print('Hello World!')
        'Return ``vector_field`` called `i` times on ``scalar_field``.'
        return reduce(lambda s, v: v.rcall(s), [vector_field] * i, scalar_field)

    def taylor_terms_per_coord(coord_function):
        if False:
            while True:
                i = 10
        'Return the series for one of the coordinates.'
        return [param ** i * iter_vfield(coord_function, i).rcall(start_point) / factorial(i) for i in range(n)]
    coord_sys = coord_sys if coord_sys else start_point._coord_sys
    coord_functions = coord_sys.coord_functions()
    taylor_terms = [taylor_terms_per_coord(f) for f in coord_functions]
    if coeffs:
        return [Matrix(t) for t in zip(*taylor_terms)]
    else:
        return Matrix([sum(c) for c in taylor_terms])

def intcurve_diffequ(vector_field, param, start_point, coord_sys=None):
    if False:
        i = 10
        return i + 15
    'Return the differential equation for an integral curve of the field.\n\n    Explanation\n    ===========\n\n    Integral curve is a function `\\gamma` taking a parameter in `R` to a point\n    in the manifold. It verifies the equation:\n\n    `V(f)\\big(\\gamma(t)\\big) = \\frac{d}{dt}f\\big(\\gamma(t)\\big)`\n\n    where the given ``vector_field`` is denoted as `V`. This holds for any\n    value `t` for the parameter and any scalar field `f`.\n\n    This function returns the differential equation of `\\gamma(t)` in terms of the\n    coordinate system ``coord_sys``. The equations and expansions are necessarily\n    done in coordinate-system-dependent way as there is no other way to\n    represent movement between points on the manifold (i.e. there is no such\n    thing as a difference of points for a general manifold).\n\n    Parameters\n    ==========\n\n    vector_field\n        the vector field for which an integral curve will be given\n\n    param\n        the argument of the function `\\gamma` from R to the curve\n\n    start_point\n        the point which corresponds to `\\gamma(0)`\n\n    coord_sys\n        the coordinate system in which to give the equations\n\n    Returns\n    =======\n\n    a tuple of (equations, initial conditions)\n\n    Examples\n    ========\n\n    Use the predefined R2 manifold:\n\n    >>> from sympy.abc import t\n    >>> from sympy.diffgeom.rn import R2, R2_p, R2_r\n    >>> from sympy.diffgeom import intcurve_diffequ\n\n    Specify a starting point and a vector field:\n\n    >>> start_point = R2_r.point([0, 1])\n    >>> vector_field = -R2.y*R2.e_x + R2.x*R2.e_y\n\n    Get the equation:\n\n    >>> equations, init_cond = intcurve_diffequ(vector_field, t, start_point)\n    >>> equations\n    [f_1(t) + Derivative(f_0(t), t), -f_0(t) + Derivative(f_1(t), t)]\n    >>> init_cond\n    [f_0(0), f_1(0) - 1]\n\n    The series in the polar coordinate system:\n\n    >>> equations, init_cond = intcurve_diffequ(vector_field, t, start_point, R2_p)\n    >>> equations\n    [Derivative(f_0(t), t), Derivative(f_1(t), t) - 1]\n    >>> init_cond\n    [f_0(0) - 1, f_1(0) - pi/2]\n\n    See Also\n    ========\n\n    intcurve_series\n\n    '
    if contravariant_order(vector_field) != 1 or covariant_order(vector_field):
        raise ValueError('The supplied field was not a vector field.')
    coord_sys = coord_sys if coord_sys else start_point._coord_sys
    gammas = [Function('f_%d' % i)(param) for i in range(start_point._coord_sys.dim)]
    arbitrary_p = Point(coord_sys, gammas)
    coord_functions = coord_sys.coord_functions()
    equations = [simplify(diff(cf.rcall(arbitrary_p), param) - vector_field.rcall(cf).rcall(arbitrary_p)) for cf in coord_functions]
    init_cond = [simplify(cf.rcall(arbitrary_p).subs(param, 0) - cf.rcall(start_point)) for cf in coord_functions]
    return (equations, init_cond)

def dummyfy(args, exprs):
    if False:
        i = 10
        return i + 15
    d_args = Matrix([s.as_dummy() for s in args])
    reps = dict(zip(args, d_args))
    d_exprs = Matrix([_sympify(expr).subs(reps) for expr in exprs])
    return (d_args, d_exprs)

def contravariant_order(expr, _strict=False):
    if False:
        print('Hello World!')
    'Return the contravariant order of an expression.\n\n    Examples\n    ========\n\n    >>> from sympy.diffgeom import contravariant_order\n    >>> from sympy.diffgeom.rn import R2\n    >>> from sympy.abc import a\n\n    >>> contravariant_order(a)\n    0\n    >>> contravariant_order(a*R2.x + 2)\n    0\n    >>> contravariant_order(a*R2.x*R2.e_y + R2.e_x)\n    1\n\n    '
    if isinstance(expr, Add):
        orders = [contravariant_order(e) for e in expr.args]
        if len(set(orders)) != 1:
            raise ValueError('Misformed expression containing contravariant fields of varying order.')
        return orders[0]
    elif isinstance(expr, Mul):
        orders = [contravariant_order(e) for e in expr.args]
        not_zero = [o for o in orders if o != 0]
        if len(not_zero) > 1:
            raise ValueError('Misformed expression containing multiplication between vectors.')
        return 0 if not not_zero else not_zero[0]
    elif isinstance(expr, Pow):
        if covariant_order(expr.base) or covariant_order(expr.exp):
            raise ValueError('Misformed expression containing a power of a vector.')
        return 0
    elif isinstance(expr, BaseVectorField):
        return 1
    elif isinstance(expr, TensorProduct):
        return sum((contravariant_order(a) for a in expr.args))
    elif not _strict or expr.atoms(BaseScalarField):
        return 0
    else:
        return -1

def covariant_order(expr, _strict=False):
    if False:
        return 10
    'Return the covariant order of an expression.\n\n    Examples\n    ========\n\n    >>> from sympy.diffgeom import covariant_order\n    >>> from sympy.diffgeom.rn import R2\n    >>> from sympy.abc import a\n\n    >>> covariant_order(a)\n    0\n    >>> covariant_order(a*R2.x + 2)\n    0\n    >>> covariant_order(a*R2.x*R2.dy + R2.dx)\n    1\n\n    '
    if isinstance(expr, Add):
        orders = [covariant_order(e) for e in expr.args]
        if len(set(orders)) != 1:
            raise ValueError('Misformed expression containing form fields of varying order.')
        return orders[0]
    elif isinstance(expr, Mul):
        orders = [covariant_order(e) for e in expr.args]
        not_zero = [o for o in orders if o != 0]
        if len(not_zero) > 1:
            raise ValueError('Misformed expression containing multiplication between forms.')
        return 0 if not not_zero else not_zero[0]
    elif isinstance(expr, Pow):
        if covariant_order(expr.base) or covariant_order(expr.exp):
            raise ValueError('Misformed expression containing a power of a form.')
        return 0
    elif isinstance(expr, Differential):
        return covariant_order(*expr.args) + 1
    elif isinstance(expr, TensorProduct):
        return sum((covariant_order(a) for a in expr.args))
    elif not _strict or expr.atoms(BaseScalarField):
        return 0
    else:
        return -1

def vectors_in_basis(expr, to_sys):
    if False:
        for i in range(10):
            print('nop')
    'Transform all base vectors in base vectors of a specified coord basis.\n    While the new base vectors are in the new coordinate system basis, any\n    coefficients are kept in the old system.\n\n    Examples\n    ========\n\n    >>> from sympy.diffgeom import vectors_in_basis\n    >>> from sympy.diffgeom.rn import R2_r, R2_p\n\n    >>> vectors_in_basis(R2_r.e_x, R2_p)\n    -y*e_theta/(x**2 + y**2) + x*e_rho/sqrt(x**2 + y**2)\n    >>> vectors_in_basis(R2_p.e_r, R2_r)\n    sin(theta)*e_y + cos(theta)*e_x\n\n    '
    vectors = list(expr.atoms(BaseVectorField))
    new_vectors = []
    for v in vectors:
        cs = v._coord_sys
        jac = cs.jacobian(to_sys, cs.coord_functions())
        new = (jac.T * Matrix(to_sys.base_vectors()))[v._index]
        new_vectors.append(new)
    return expr.subs(list(zip(vectors, new_vectors)))

def twoform_to_matrix(expr):
    if False:
        i = 10
        return i + 15
    'Return the matrix representing the twoform.\n\n    For the twoform `w` return the matrix `M` such that `M[i,j]=w(e_i, e_j)`,\n    where `e_i` is the i-th base vector field for the coordinate system in\n    which the expression of `w` is given.\n\n    Examples\n    ========\n\n    >>> from sympy.diffgeom.rn import R2\n    >>> from sympy.diffgeom import twoform_to_matrix, TensorProduct\n    >>> TP = TensorProduct\n\n    >>> twoform_to_matrix(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))\n    Matrix([\n    [1, 0],\n    [0, 1]])\n    >>> twoform_to_matrix(R2.x*TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))\n    Matrix([\n    [x, 0],\n    [0, 1]])\n    >>> twoform_to_matrix(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy) - TP(R2.dx, R2.dy)/2)\n    Matrix([\n    [   1, 0],\n    [-1/2, 1]])\n\n    '
    if covariant_order(expr) != 2 or contravariant_order(expr):
        raise ValueError('The input expression is not a two-form.')
    coord_sys = _find_coords(expr)
    if len(coord_sys) != 1:
        raise ValueError('The input expression concerns more than one coordinate systems, hence there is no unambiguous way to choose a coordinate system for the matrix.')
    coord_sys = coord_sys.pop()
    vectors = coord_sys.base_vectors()
    expr = expr.expand()
    matrix_content = [[expr.rcall(v1, v2) for v1 in vectors] for v2 in vectors]
    return Matrix(matrix_content)

def metric_to_Christoffel_1st(expr):
    if False:
        print('Hello World!')
    'Return the nested list of Christoffel symbols for the given metric.\n    This returns the Christoffel symbol of first kind that represents the\n    Levi-Civita connection for the given metric.\n\n    Examples\n    ========\n\n    >>> from sympy.diffgeom.rn import R2\n    >>> from sympy.diffgeom import metric_to_Christoffel_1st, TensorProduct\n    >>> TP = TensorProduct\n\n    >>> metric_to_Christoffel_1st(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))\n    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]\n    >>> metric_to_Christoffel_1st(R2.x*TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))\n    [[[1/2, 0], [0, 0]], [[0, 0], [0, 0]]]\n\n    '
    matrix = twoform_to_matrix(expr)
    if not matrix.is_symmetric():
        raise ValueError('The two-form representing the metric is not symmetric.')
    coord_sys = _find_coords(expr).pop()
    deriv_matrices = [matrix.applyfunc(d) for d in coord_sys.base_vectors()]
    indices = list(range(coord_sys.dim))
    christoffel = [[[(deriv_matrices[k][i, j] + deriv_matrices[j][i, k] - deriv_matrices[i][j, k]) / 2 for k in indices] for j in indices] for i in indices]
    return ImmutableDenseNDimArray(christoffel)

def metric_to_Christoffel_2nd(expr):
    if False:
        while True:
            i = 10
    'Return the nested list of Christoffel symbols for the given metric.\n    This returns the Christoffel symbol of second kind that represents the\n    Levi-Civita connection for the given metric.\n\n    Examples\n    ========\n\n    >>> from sympy.diffgeom.rn import R2\n    >>> from sympy.diffgeom import metric_to_Christoffel_2nd, TensorProduct\n    >>> TP = TensorProduct\n\n    >>> metric_to_Christoffel_2nd(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))\n    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]\n    >>> metric_to_Christoffel_2nd(R2.x*TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))\n    [[[1/(2*x), 0], [0, 0]], [[0, 0], [0, 0]]]\n\n    '
    ch_1st = metric_to_Christoffel_1st(expr)
    coord_sys = _find_coords(expr).pop()
    indices = list(range(coord_sys.dim))
    matrix = twoform_to_matrix(expr)
    s_fields = set()
    for e in matrix:
        s_fields.update(e.atoms(BaseScalarField))
    s_fields = list(s_fields)
    dums = coord_sys.symbols
    matrix = matrix.subs(list(zip(s_fields, dums))).inv().subs(list(zip(dums, s_fields)))
    christoffel = [[[Add(*[matrix[i, l] * ch_1st[l, j, k] for l in indices]) for k in indices] for j in indices] for i in indices]
    return ImmutableDenseNDimArray(christoffel)

def metric_to_Riemann_components(expr):
    if False:
        i = 10
        return i + 15
    'Return the components of the Riemann tensor expressed in a given basis.\n\n    Given a metric it calculates the components of the Riemann tensor in the\n    canonical basis of the coordinate system in which the metric expression is\n    given.\n\n    Examples\n    ========\n\n    >>> from sympy import exp\n    >>> from sympy.diffgeom.rn import R2\n    >>> from sympy.diffgeom import metric_to_Riemann_components, TensorProduct\n    >>> TP = TensorProduct\n\n    >>> metric_to_Riemann_components(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))\n    [[[[0, 0], [0, 0]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]]\n    >>> non_trivial_metric = exp(2*R2.r)*TP(R2.dr, R2.dr) +         R2.r**2*TP(R2.dtheta, R2.dtheta)\n    >>> non_trivial_metric\n    exp(2*rho)*TensorProduct(drho, drho) + rho**2*TensorProduct(dtheta, dtheta)\n    >>> riemann = metric_to_Riemann_components(non_trivial_metric)\n    >>> riemann[0, :, :, :]\n    [[[0, 0], [0, 0]], [[0, exp(-2*rho)*rho], [-exp(-2*rho)*rho, 0]]]\n    >>> riemann[1, :, :, :]\n    [[[0, -1/rho], [1/rho, 0]], [[0, 0], [0, 0]]]\n\n    '
    ch_2nd = metric_to_Christoffel_2nd(expr)
    coord_sys = _find_coords(expr).pop()
    indices = list(range(coord_sys.dim))
    deriv_ch = [[[[d(ch_2nd[i, j, k]) for d in coord_sys.base_vectors()] for k in indices] for j in indices] for i in indices]
    riemann_a = [[[[deriv_ch[rho][sig][nu][mu] - deriv_ch[rho][sig][mu][nu] for nu in indices] for mu in indices] for sig in indices] for rho in indices]
    riemann_b = [[[[Add(*[ch_2nd[rho, l, mu] * ch_2nd[l, sig, nu] - ch_2nd[rho, l, nu] * ch_2nd[l, sig, mu] for l in indices]) for nu in indices] for mu in indices] for sig in indices] for rho in indices]
    riemann = [[[[riemann_a[rho][sig][mu][nu] + riemann_b[rho][sig][mu][nu] for nu in indices] for mu in indices] for sig in indices] for rho in indices]
    return ImmutableDenseNDimArray(riemann)

def metric_to_Ricci_components(expr):
    if False:
        i = 10
        return i + 15
    'Return the components of the Ricci tensor expressed in a given basis.\n\n    Given a metric it calculates the components of the Ricci tensor in the\n    canonical basis of the coordinate system in which the metric expression is\n    given.\n\n    Examples\n    ========\n\n    >>> from sympy import exp\n    >>> from sympy.diffgeom.rn import R2\n    >>> from sympy.diffgeom import metric_to_Ricci_components, TensorProduct\n    >>> TP = TensorProduct\n\n    >>> metric_to_Ricci_components(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))\n    [[0, 0], [0, 0]]\n    >>> non_trivial_metric = exp(2*R2.r)*TP(R2.dr, R2.dr) +                              R2.r**2*TP(R2.dtheta, R2.dtheta)\n    >>> non_trivial_metric\n    exp(2*rho)*TensorProduct(drho, drho) + rho**2*TensorProduct(dtheta, dtheta)\n    >>> metric_to_Ricci_components(non_trivial_metric)\n    [[1/rho, 0], [0, exp(-2*rho)*rho]]\n\n    '
    riemann = metric_to_Riemann_components(expr)
    coord_sys = _find_coords(expr).pop()
    indices = list(range(coord_sys.dim))
    ricci = [[Add(*[riemann[k, i, k, j] for k in indices]) for j in indices] for i in indices]
    return ImmutableDenseNDimArray(ricci)

class _deprecated_container:

    def __init__(self, message, data):
        if False:
            print('Hello World!')
        super().__init__(data)
        self.message = message

    def warn(self):
        if False:
            for i in range(10):
                print('nop')
        sympy_deprecation_warning(self.message, deprecated_since_version='1.7', active_deprecations_target='deprecated-diffgeom-mutable', stacklevel=4)

    def __iter__(self):
        if False:
            return 10
        self.warn()
        return super().__iter__()

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        self.warn()
        return super().__getitem__(key)

    def __contains__(self, key):
        if False:
            while True:
                i = 10
        self.warn()
        return super().__contains__(key)

class _deprecated_list(_deprecated_container, list):
    pass

class _deprecated_dict(_deprecated_container, dict):
    pass
from sympy.simplify.simplify import simplify