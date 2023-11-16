from __future__ import annotations
from itertools import product
from sympy.core.add import Add
from sympy.core.assumptions import StdFactKB
from sympy.core.expr import AtomicExpr, Expr
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.vector.basisdependent import BasisDependentZero, BasisDependent, BasisDependentMul, BasisDependentAdd
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.dyadic import Dyadic, BaseDyadic, DyadicAdd

class Vector(BasisDependent):
    """
    Super class for all Vector classes.
    Ideally, neither this class nor any of its subclasses should be
    instantiated by the user.
    """
    is_scalar = False
    is_Vector = True
    _op_priority = 12.0
    _expr_type: type[Vector]
    _mul_func: type[Vector]
    _add_func: type[Vector]
    _zero_func: type[Vector]
    _base_func: type[Vector]
    zero: VectorZero

    @property
    def components(self):
        if False:
            while True:
                i = 10
        "\n        Returns the components of this vector in the form of a\n        Python dictionary mapping BaseVector instances to the\n        corresponding measure numbers.\n\n        Examples\n        ========\n\n        >>> from sympy.vector import CoordSys3D\n        >>> C = CoordSys3D('C')\n        >>> v = 3*C.i + 4*C.j + 5*C.k\n        >>> v.components\n        {C.i: 3, C.j: 4, C.k: 5}\n\n        "
        return self._components

    def magnitude(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the magnitude of this vector.\n        '
        return sqrt(self & self)

    def normalize(self):
        if False:
            return 10
        '\n        Returns the normalized version of this vector.\n        '
        return self / self.magnitude()

    def dot(self, other):
        if False:
            return 10
        "\n        Returns the dot product of this Vector, either with another\n        Vector, or a Dyadic, or a Del operator.\n        If 'other' is a Vector, returns the dot product scalar (SymPy\n        expression).\n        If 'other' is a Dyadic, the dot product is returned as a Vector.\n        If 'other' is an instance of Del, returns the directional\n        derivative operator as a Python function. If this function is\n        applied to a scalar expression, it returns the directional\n        derivative of the scalar field wrt this Vector.\n\n        Parameters\n        ==========\n\n        other: Vector/Dyadic/Del\n            The Vector or Dyadic we are dotting with, or a Del operator .\n\n        Examples\n        ========\n\n        >>> from sympy.vector import CoordSys3D, Del\n        >>> C = CoordSys3D('C')\n        >>> delop = Del()\n        >>> C.i.dot(C.j)\n        0\n        >>> C.i & C.i\n        1\n        >>> v = 3*C.i + 4*C.j + 5*C.k\n        >>> v.dot(C.k)\n        5\n        >>> (C.i & delop)(C.x*C.y*C.z)\n        C.y*C.z\n        >>> d = C.i.outer(C.i)\n        >>> C.i.dot(d)\n        C.i\n\n        "
        if isinstance(other, Dyadic):
            if isinstance(self, VectorZero):
                return Vector.zero
            outvec = Vector.zero
            for (k, v) in other.components.items():
                vect_dot = k.args[0].dot(self)
                outvec += vect_dot * v * k.args[1]
            return outvec
        from sympy.vector.deloperator import Del
        if not isinstance(other, (Del, Vector)):
            raise TypeError(str(other) + ' is not a vector, dyadic or ' + 'del operator')
        if isinstance(other, Del):

            def directional_derivative(field):
                if False:
                    for i in range(10):
                        print('nop')
                from sympy.vector.functions import directional_derivative
                return directional_derivative(field, self)
            return directional_derivative
        return dot(self, other)

    def __and__(self, other):
        if False:
            print('Hello World!')
        return self.dot(other)
    __and__.__doc__ = dot.__doc__

    def cross(self, other):
        if False:
            while True:
                i = 10
        "\n        Returns the cross product of this Vector with another Vector or\n        Dyadic instance.\n        The cross product is a Vector, if 'other' is a Vector. If 'other'\n        is a Dyadic, this returns a Dyadic instance.\n\n        Parameters\n        ==========\n\n        other: Vector/Dyadic\n            The Vector or Dyadic we are crossing with.\n\n        Examples\n        ========\n\n        >>> from sympy.vector import CoordSys3D\n        >>> C = CoordSys3D('C')\n        >>> C.i.cross(C.j)\n        C.k\n        >>> C.i ^ C.i\n        0\n        >>> v = 3*C.i + 4*C.j + 5*C.k\n        >>> v ^ C.i\n        5*C.j + (-4)*C.k\n        >>> d = C.i.outer(C.i)\n        >>> C.j.cross(d)\n        (-1)*(C.k|C.i)\n\n        "
        if isinstance(other, Dyadic):
            if isinstance(self, VectorZero):
                return Dyadic.zero
            outdyad = Dyadic.zero
            for (k, v) in other.components.items():
                cross_product = self.cross(k.args[0])
                outer = cross_product.outer(k.args[1])
                outdyad += v * outer
            return outdyad
        return cross(self, other)

    def __xor__(self, other):
        if False:
            i = 10
            return i + 15
        return self.cross(other)
    __xor__.__doc__ = cross.__doc__

    def outer(self, other):
        if False:
            return 10
        "\n        Returns the outer product of this vector with another, in the\n        form of a Dyadic instance.\n\n        Parameters\n        ==========\n\n        other : Vector\n            The Vector with respect to which the outer product is to\n            be computed.\n\n        Examples\n        ========\n\n        >>> from sympy.vector import CoordSys3D\n        >>> N = CoordSys3D('N')\n        >>> N.i.outer(N.j)\n        (N.i|N.j)\n\n        "
        if not isinstance(other, Vector):
            raise TypeError('Invalid operand for outer product')
        elif isinstance(self, VectorZero) or isinstance(other, VectorZero):
            return Dyadic.zero
        args = [v1 * v2 * BaseDyadic(k1, k2) for ((k1, v1), (k2, v2)) in product(self.components.items(), other.components.items())]
        return DyadicAdd(*args)

    def projection(self, other, scalar=False):
        if False:
            i = 10
            return i + 15
        "\n        Returns the vector or scalar projection of the 'other' on 'self'.\n\n        Examples\n        ========\n\n        >>> from sympy.vector.coordsysrect import CoordSys3D\n        >>> C = CoordSys3D('C')\n        >>> i, j, k = C.base_vectors()\n        >>> v1 = i + j + k\n        >>> v2 = 3*i + 4*j\n        >>> v1.projection(v2)\n        7/3*C.i + 7/3*C.j + 7/3*C.k\n        >>> v1.projection(v2, scalar=True)\n        7/3\n\n        "
        if self.equals(Vector.zero):
            return S.Zero if scalar else Vector.zero
        if scalar:
            return self.dot(other) / self.dot(self)
        else:
            return self.dot(other) / self.dot(self) * self

    @property
    def _projections(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns the components of this vector but the output includes\n        also zero values components.\n\n        Examples\n        ========\n\n        >>> from sympy.vector import CoordSys3D, Vector\n        >>> C = CoordSys3D('C')\n        >>> v1 = 3*C.i + 4*C.j + 5*C.k\n        >>> v1._projections\n        (3, 4, 5)\n        >>> v2 = C.x*C.y*C.z*C.i\n        >>> v2._projections\n        (C.x*C.y*C.z, 0, 0)\n        >>> v3 = Vector.zero\n        >>> v3._projections\n        (0, 0, 0)\n        "
        from sympy.vector.operators import _get_coord_systems
        if isinstance(self, VectorZero):
            return (S.Zero, S.Zero, S.Zero)
        base_vec = next(iter(_get_coord_systems(self))).base_vectors()
        return tuple([self.dot(i) for i in base_vec])

    def __or__(self, other):
        if False:
            return 10
        return self.outer(other)
    __or__.__doc__ = outer.__doc__

    def to_matrix(self, system):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns the matrix form of this vector with respect to the\n        specified coordinate system.\n\n        Parameters\n        ==========\n\n        system : CoordSys3D\n            The system wrt which the matrix form is to be computed\n\n        Examples\n        ========\n\n        >>> from sympy.vector import CoordSys3D\n        >>> C = CoordSys3D('C')\n        >>> from sympy.abc import a, b, c\n        >>> v = a*C.i + b*C.j + c*C.k\n        >>> v.to_matrix(C)\n        Matrix([\n        [a],\n        [b],\n        [c]])\n\n        "
        return Matrix([self.dot(unit_vec) for unit_vec in system.base_vectors()])

    def separate(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The constituents of this vector in different coordinate systems,\n        as per its definition.\n\n        Returns a dict mapping each CoordSys3D to the corresponding\n        constituent Vector.\n\n        Examples\n        ========\n\n        >>> from sympy.vector import CoordSys3D\n        >>> R1 = CoordSys3D('R1')\n        >>> R2 = CoordSys3D('R2')\n        >>> v = R1.i + R2.i\n        >>> v.separate() == {R1: R1.i, R2: R2.i}\n        True\n\n        "
        parts = {}
        for (vect, measure) in self.components.items():
            parts[vect.system] = parts.get(vect.system, Vector.zero) + vect * measure
        return parts

    def _div_helper(one, other):
        if False:
            print('Hello World!')
        ' Helper for division involving vectors. '
        if isinstance(one, Vector) and isinstance(other, Vector):
            raise TypeError('Cannot divide two vectors')
        elif isinstance(one, Vector):
            if other == S.Zero:
                raise ValueError('Cannot divide a vector by zero')
            return VectorMul(one, Pow(other, S.NegativeOne))
        else:
            raise TypeError('Invalid division involving a vector')

class BaseVector(Vector, AtomicExpr):
    """
    Class to denote a base vector.

    """

    def __new__(cls, index, system, pretty_str=None, latex_str=None):
        if False:
            i = 10
            return i + 15
        if pretty_str is None:
            pretty_str = 'x{}'.format(index)
        if latex_str is None:
            latex_str = 'x_{}'.format(index)
        pretty_str = str(pretty_str)
        latex_str = str(latex_str)
        if index not in range(0, 3):
            raise ValueError('index must be 0, 1 or 2')
        if not isinstance(system, CoordSys3D):
            raise TypeError('system should be a CoordSys3D')
        name = system._vector_names[index]
        obj = super().__new__(cls, S(index), system)
        obj._base_instance = obj
        obj._components = {obj: S.One}
        obj._measure_number = S.One
        obj._name = system._name + '.' + name
        obj._pretty_form = '' + pretty_str
        obj._latex_form = latex_str
        obj._system = system
        obj._id = (index, system)
        assumptions = {'commutative': True}
        obj._assumptions = StdFactKB(assumptions)
        obj._sys = system
        return obj

    @property
    def system(self):
        if False:
            while True:
                i = 10
        return self._system

    def _sympystr(self, printer):
        if False:
            print('Hello World!')
        return self._name

    def _sympyrepr(self, printer):
        if False:
            print('Hello World!')
        (index, system) = self._id
        return printer._print(system) + '.' + system._vector_names[index]

    @property
    def free_symbols(self):
        if False:
            for i in range(10):
                print('nop')
        return {self}

class VectorAdd(BasisDependentAdd, Vector):
    """
    Class to denote sum of Vector instances.
    """

    def __new__(cls, *args, **options):
        if False:
            return 10
        obj = BasisDependentAdd.__new__(cls, *args, **options)
        return obj

    def _sympystr(self, printer):
        if False:
            i = 10
            return i + 15
        ret_str = ''
        items = list(self.separate().items())
        items.sort(key=lambda x: x[0].__str__())
        for (system, vect) in items:
            base_vects = system.base_vectors()
            for x in base_vects:
                if x in vect.components:
                    temp_vect = self.components[x] * x
                    ret_str += printer._print(temp_vect) + ' + '
        return ret_str[:-3]

class VectorMul(BasisDependentMul, Vector):
    """
    Class to denote products of scalars and BaseVectors.
    """

    def __new__(cls, *args, **options):
        if False:
            print('Hello World!')
        obj = BasisDependentMul.__new__(cls, *args, **options)
        return obj

    @property
    def base_vector(self):
        if False:
            return 10
        ' The BaseVector involved in the product. '
        return self._base_instance

    @property
    def measure_number(self):
        if False:
            for i in range(10):
                print('nop')
        ' The scalar expression involved in the definition of\n        this VectorMul.\n        '
        return self._measure_number

class VectorZero(BasisDependentZero, Vector):
    """
    Class to denote a zero vector
    """
    _op_priority = 12.1
    _pretty_form = '0'
    _latex_form = '\\mathbf{\\hat{0}}'

    def __new__(cls):
        if False:
            while True:
                i = 10
        obj = BasisDependentZero.__new__(cls)
        return obj

class Cross(Vector):
    """
    Represents unevaluated Cross product.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, Cross
    >>> R = CoordSys3D('R')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = R.x * R.i + R.y * R.j + R.z * R.k
    >>> Cross(v1, v2)
    Cross(R.i + R.j + R.k, R.x*R.i + R.y*R.j + R.z*R.k)
    >>> Cross(v1, v2).doit()
    (-R.y + R.z)*R.i + (R.x - R.z)*R.j + (-R.x + R.y)*R.k

    """

    def __new__(cls, expr1, expr2):
        if False:
            while True:
                i = 10
        expr1 = sympify(expr1)
        expr2 = sympify(expr2)
        if default_sort_key(expr1) > default_sort_key(expr2):
            return -Cross(expr2, expr1)
        obj = Expr.__new__(cls, expr1, expr2)
        obj._expr1 = expr1
        obj._expr2 = expr2
        return obj

    def doit(self, **hints):
        if False:
            print('Hello World!')
        return cross(self._expr1, self._expr2)

class Dot(Expr):
    """
    Represents unevaluated Dot product.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, Dot
    >>> from sympy import symbols
    >>> R = CoordSys3D('R')
    >>> a, b, c = symbols('a b c')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = a * R.i + b * R.j + c * R.k
    >>> Dot(v1, v2)
    Dot(R.i + R.j + R.k, a*R.i + b*R.j + c*R.k)
    >>> Dot(v1, v2).doit()
    a + b + c

    """

    def __new__(cls, expr1, expr2):
        if False:
            while True:
                i = 10
        expr1 = sympify(expr1)
        expr2 = sympify(expr2)
        (expr1, expr2) = sorted([expr1, expr2], key=default_sort_key)
        obj = Expr.__new__(cls, expr1, expr2)
        obj._expr1 = expr1
        obj._expr2 = expr2
        return obj

    def doit(self, **hints):
        if False:
            return 10
        return dot(self._expr1, self._expr2)

def cross(vect1, vect2):
    if False:
        i = 10
        return i + 15
    "\n    Returns cross product of two vectors.\n\n    Examples\n    ========\n\n    >>> from sympy.vector import CoordSys3D\n    >>> from sympy.vector.vector import cross\n    >>> R = CoordSys3D('R')\n    >>> v1 = R.i + R.j + R.k\n    >>> v2 = R.x * R.i + R.y * R.j + R.z * R.k\n    >>> cross(v1, v2)\n    (-R.y + R.z)*R.i + (R.x - R.z)*R.j + (-R.x + R.y)*R.k\n\n    "
    if isinstance(vect1, Add):
        return VectorAdd.fromiter((cross(i, vect2) for i in vect1.args))
    if isinstance(vect2, Add):
        return VectorAdd.fromiter((cross(vect1, i) for i in vect2.args))
    if isinstance(vect1, BaseVector) and isinstance(vect2, BaseVector):
        if vect1._sys == vect2._sys:
            n1 = vect1.args[0]
            n2 = vect2.args[0]
            if n1 == n2:
                return Vector.zero
            n3 = {0, 1, 2}.difference({n1, n2}).pop()
            sign = 1 if (n1 + 1) % 3 == n2 else -1
            return sign * vect1._sys.base_vectors()[n3]
        from .functions import express
        try:
            v = express(vect1, vect2._sys)
        except ValueError:
            return Cross(vect1, vect2)
        else:
            return cross(v, vect2)
    if isinstance(vect1, VectorZero) or isinstance(vect2, VectorZero):
        return Vector.zero
    if isinstance(vect1, VectorMul):
        (v1, m1) = next(iter(vect1.components.items()))
        return m1 * cross(v1, vect2)
    if isinstance(vect2, VectorMul):
        (v2, m2) = next(iter(vect2.components.items()))
        return m2 * cross(vect1, v2)
    return Cross(vect1, vect2)

def dot(vect1, vect2):
    if False:
        while True:
            i = 10
    "\n    Returns dot product of two vectors.\n\n    Examples\n    ========\n\n    >>> from sympy.vector import CoordSys3D\n    >>> from sympy.vector.vector import dot\n    >>> R = CoordSys3D('R')\n    >>> v1 = R.i + R.j + R.k\n    >>> v2 = R.x * R.i + R.y * R.j + R.z * R.k\n    >>> dot(v1, v2)\n    R.x + R.y + R.z\n\n    "
    if isinstance(vect1, Add):
        return Add.fromiter((dot(i, vect2) for i in vect1.args))
    if isinstance(vect2, Add):
        return Add.fromiter((dot(vect1, i) for i in vect2.args))
    if isinstance(vect1, BaseVector) and isinstance(vect2, BaseVector):
        if vect1._sys == vect2._sys:
            return S.One if vect1 == vect2 else S.Zero
        from .functions import express
        try:
            v = express(vect2, vect1._sys)
        except ValueError:
            return Dot(vect1, vect2)
        else:
            return dot(vect1, v)
    if isinstance(vect1, VectorZero) or isinstance(vect2, VectorZero):
        return S.Zero
    if isinstance(vect1, VectorMul):
        (v1, m1) = next(iter(vect1.components.items()))
        return m1 * dot(v1, vect2)
    if isinstance(vect2, VectorMul):
        (v2, m2) = next(iter(vect2.components.items()))
        return m2 * dot(vect1, v2)
    return Dot(vect1, vect2)
Vector._expr_type = Vector
Vector._mul_func = VectorMul
Vector._add_func = VectorAdd
Vector._zero_func = VectorZero
Vector._base_func = BaseVector
Vector.zero = VectorZero()