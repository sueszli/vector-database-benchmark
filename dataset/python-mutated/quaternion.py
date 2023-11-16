from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.relational import is_eq
from sympy.functions.elementary.complexes import conjugate, im, re, sign
from sympy.functions.elementary.exponential import exp, log as ln
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import acos, asin, atan2
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.simplify.trigsimp import trigsimp
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.core.sympify import sympify, _sympify
from sympy.core.expr import Expr
from sympy.core.logic import fuzzy_not, fuzzy_or
from sympy.utilities.misc import as_int
from mpmath.libmp.libmpf import prec_to_dps

def _check_norm(elements, norm):
    if False:
        print('Hello World!')
    'validate if input norm is consistent'
    if norm is not None and norm.is_number:
        if norm.is_positive is False:
            raise ValueError('Input norm must be positive.')
        numerical = all((i.is_number and i.is_real is True for i in elements))
        if numerical and is_eq(norm ** 2, sum((i ** 2 for i in elements))) is False:
            raise ValueError('Incompatible value for norm.')

def _is_extrinsic(seq):
    if False:
        for i in range(10):
            print('nop')
    'validate seq and return True if seq is lowercase and False if uppercase'
    if type(seq) != str:
        raise ValueError('Expected seq to be a string.')
    if len(seq) != 3:
        raise ValueError('Expected 3 axes, got `{}`.'.format(seq))
    intrinsic = seq.isupper()
    extrinsic = seq.islower()
    if not (intrinsic or extrinsic):
        raise ValueError('seq must either be fully uppercase (for extrinsic rotations), or fully lowercase, for intrinsic rotations).')
    (i, j, k) = seq.lower()
    if i == j or j == k:
        raise ValueError('Consecutive axes must be different')
    bad = set(seq) - set('xyzXYZ')
    if bad:
        raise ValueError("Expected axes from `seq` to be from ['x', 'y', 'z'] or ['X', 'Y', 'Z'], got {}".format(''.join(bad)))
    return extrinsic

class Quaternion(Expr):
    """Provides basic quaternion operations.
    Quaternion objects can be instantiated as ``Quaternion(a, b, c, d)``
    as in $q = a + bi + cj + dk$.

    Parameters
    ==========

    norm : None or number
        Pre-defined quaternion norm. If a value is given, Quaternion.norm
        returns this pre-defined value instead of calculating the norm

    Examples
    ========

    >>> from sympy import Quaternion
    >>> q = Quaternion(1, 2, 3, 4)
    >>> q
    1 + 2*i + 3*j + 4*k

    Quaternions over complex fields can be defined as:

    >>> from sympy import Quaternion
    >>> from sympy import symbols, I
    >>> x = symbols('x')
    >>> q1 = Quaternion(x, x**3, x, x**2, real_field = False)
    >>> q2 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
    >>> q1
    x + x**3*i + x*j + x**2*k
    >>> q2
    (3 + 4*I) + (2 + 5*I)*i + 0*j + (7 + 8*I)*k

    Defining symbolic unit quaternions:

    >>> from sympy import Quaternion
    >>> from sympy.abc import w, x, y, z
    >>> q = Quaternion(w, x, y, z, norm=1)
    >>> q
    w + x*i + y*j + z*k
    >>> q.norm()
    1

    References
    ==========

    .. [1] https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/
    .. [2] https://en.wikipedia.org/wiki/Quaternion

    """
    _op_priority = 11.0
    is_commutative = False

    def __new__(cls, a=0, b=0, c=0, d=0, real_field=True, norm=None):
        if False:
            return 10
        (a, b, c, d) = map(sympify, (a, b, c, d))
        if any((i.is_commutative is False for i in [a, b, c, d])):
            raise ValueError('arguments have to be commutative')
        obj = super().__new__(cls, a, b, c, d)
        obj._real_field = real_field
        obj.set_norm(norm)
        return obj

    def set_norm(self, norm):
        if False:
            while True:
                i = 10
        'Sets norm of an already instantiated quaternion.\n\n        Parameters\n        ==========\n\n        norm : None or number\n            Pre-defined quaternion norm. If a value is given, Quaternion.norm\n            returns this pre-defined value instead of calculating the norm\n\n        Examples\n        ========\n\n        >>> from sympy import Quaternion\n        >>> from sympy.abc import a, b, c, d\n        >>> q = Quaternion(a, b, c, d)\n        >>> q.norm()\n        sqrt(a**2 + b**2 + c**2 + d**2)\n\n        Setting the norm:\n\n        >>> q.set_norm(1)\n        >>> q.norm()\n        1\n\n        Removing set norm:\n\n        >>> q.set_norm(None)\n        >>> q.norm()\n        sqrt(a**2 + b**2 + c**2 + d**2)\n\n        '
        norm = sympify(norm)
        _check_norm(self.args, norm)
        self._norm = norm

    @property
    def a(self):
        if False:
            print('Hello World!')
        return self.args[0]

    @property
    def b(self):
        if False:
            while True:
                i = 10
        return self.args[1]

    @property
    def c(self):
        if False:
            print('Hello World!')
        return self.args[2]

    @property
    def d(self):
        if False:
            i = 10
            return i + 15
        return self.args[3]

    @property
    def real_field(self):
        if False:
            while True:
                i = 10
        return self._real_field

    @property
    def product_matrix_left(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns 4 x 4 Matrix equivalent to a Hamilton product from the\n        left. This can be useful when treating quaternion elements as column\n        vectors. Given a quaternion $q = a + bi + cj + dk$ where a, b, c and d\n        are real numbers, the product matrix from the left is:\n\n        .. math::\n\n            M  =  \\begin{bmatrix} a  &-b  &-c  &-d \\\\\n                                  b  & a  &-d  & c \\\\\n                                  c  & d  & a  &-b \\\\\n                                  d  &-c  & b  & a \\end{bmatrix}\n\n        Examples\n        ========\n\n        >>> from sympy import Quaternion\n        >>> from sympy.abc import a, b, c, d\n        >>> q1 = Quaternion(1, 0, 0, 1)\n        >>> q2 = Quaternion(a, b, c, d)\n        >>> q1.product_matrix_left\n        Matrix([\n        [1, 0,  0, -1],\n        [0, 1, -1,  0],\n        [0, 1,  1,  0],\n        [1, 0,  0,  1]])\n\n        >>> q1.product_matrix_left * q2.to_Matrix()\n        Matrix([\n        [a - d],\n        [b - c],\n        [b + c],\n        [a + d]])\n\n        This is equivalent to:\n\n        >>> (q1 * q2).to_Matrix()\n        Matrix([\n        [a - d],\n        [b - c],\n        [b + c],\n        [a + d]])\n        '
        return Matrix([[self.a, -self.b, -self.c, -self.d], [self.b, self.a, -self.d, self.c], [self.c, self.d, self.a, -self.b], [self.d, -self.c, self.b, self.a]])

    @property
    def product_matrix_right(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns 4 x 4 Matrix equivalent to a Hamilton product from the\n        right. This can be useful when treating quaternion elements as column\n        vectors. Given a quaternion $q = a + bi + cj + dk$ where a, b, c and d\n        are real numbers, the product matrix from the left is:\n\n        .. math::\n\n            M  =  \\begin{bmatrix} a  &-b  &-c  &-d \\\\\n                                  b  & a  & d  &-c \\\\\n                                  c  &-d  & a  & b \\\\\n                                  d  & c  &-b  & a \\end{bmatrix}\n\n\n        Examples\n        ========\n\n        >>> from sympy import Quaternion\n        >>> from sympy.abc import a, b, c, d\n        >>> q1 = Quaternion(a, b, c, d)\n        >>> q2 = Quaternion(1, 0, 0, 1)\n        >>> q2.product_matrix_right\n        Matrix([\n        [1, 0, 0, -1],\n        [0, 1, 1, 0],\n        [0, -1, 1, 0],\n        [1, 0, 0, 1]])\n\n        Note the switched arguments: the matrix represents the quaternion on\n        the right, but is still considered as a matrix multiplication from the\n        left.\n\n        >>> q2.product_matrix_right * q1.to_Matrix()\n        Matrix([\n        [ a - d],\n        [ b + c],\n        [-b + c],\n        [ a + d]])\n\n        This is equivalent to:\n\n        >>> (q1 * q2).to_Matrix()\n        Matrix([\n        [ a - d],\n        [ b + c],\n        [-b + c],\n        [ a + d]])\n        '
        return Matrix([[self.a, -self.b, -self.c, -self.d], [self.b, self.a, self.d, -self.c], [self.c, -self.d, self.a, self.b], [self.d, self.c, -self.b, self.a]])

    def to_Matrix(self, vector_only=False):
        if False:
            return 10
        'Returns elements of quaternion as a column vector.\n        By default, a ``Matrix`` of length 4 is returned, with the real part as the\n        first element.\n        If ``vector_only`` is ``True``, returns only imaginary part as a Matrix of\n        length 3.\n\n        Parameters\n        ==========\n\n        vector_only : bool\n            If True, only imaginary part is returned.\n            Default value: False\n\n        Returns\n        =======\n\n        Matrix\n            A column vector constructed by the elements of the quaternion.\n\n        Examples\n        ========\n\n        >>> from sympy import Quaternion\n        >>> from sympy.abc import a, b, c, d\n        >>> q = Quaternion(a, b, c, d)\n        >>> q\n        a + b*i + c*j + d*k\n\n        >>> q.to_Matrix()\n        Matrix([\n        [a],\n        [b],\n        [c],\n        [d]])\n\n\n        >>> q.to_Matrix(vector_only=True)\n        Matrix([\n        [b],\n        [c],\n        [d]])\n\n        '
        if vector_only:
            return Matrix(self.args[1:])
        else:
            return Matrix(self.args)

    @classmethod
    def from_Matrix(cls, elements):
        if False:
            return 10
        'Returns quaternion from elements of a column vector`.\n        If vector_only is True, returns only imaginary part as a Matrix of\n        length 3.\n\n        Parameters\n        ==========\n\n        elements : Matrix, list or tuple of length 3 or 4. If length is 3,\n            assume real part is zero.\n            Default value: False\n\n        Returns\n        =======\n\n        Quaternion\n            A quaternion created from the input elements.\n\n        Examples\n        ========\n\n        >>> from sympy import Quaternion\n        >>> from sympy.abc import a, b, c, d\n        >>> q = Quaternion.from_Matrix([a, b, c, d])\n        >>> q\n        a + b*i + c*j + d*k\n\n        >>> q = Quaternion.from_Matrix([b, c, d])\n        >>> q\n        0 + b*i + c*j + d*k\n\n        '
        length = len(elements)
        if length != 3 and length != 4:
            raise ValueError('Input elements must have length 3 or 4, got {} elements'.format(length))
        if length == 3:
            return Quaternion(0, *elements)
        else:
            return Quaternion(*elements)

    @classmethod
    def from_euler(cls, angles, seq):
        if False:
            while True:
                i = 10
        "Returns quaternion equivalent to rotation represented by the Euler\n        angles, in the sequence defined by ``seq``.\n\n        Parameters\n        ==========\n\n        angles : list, tuple or Matrix of 3 numbers\n            The Euler angles (in radians).\n        seq : string of length 3\n            Represents the sequence of rotations.\n            For extrinsic rotations, seq must be all lowercase and its elements\n            must be from the set ``{'x', 'y', 'z'}``\n            For intrinsic rotations, seq must be all uppercase and its elements\n            must be from the set ``{'X', 'Y', 'Z'}``\n\n        Returns\n        =======\n\n        Quaternion\n            The normalized rotation quaternion calculated from the Euler angles\n            in the given sequence.\n\n        Examples\n        ========\n\n        >>> from sympy import Quaternion\n        >>> from sympy import pi\n        >>> q = Quaternion.from_euler([pi/2, 0, 0], 'xyz')\n        >>> q\n        sqrt(2)/2 + sqrt(2)/2*i + 0*j + 0*k\n\n        >>> q = Quaternion.from_euler([0, pi/2, pi] , 'zyz')\n        >>> q\n        0 + (-sqrt(2)/2)*i + 0*j + sqrt(2)/2*k\n\n        >>> q = Quaternion.from_euler([0, pi/2, pi] , 'ZYZ')\n        >>> q\n        0 + sqrt(2)/2*i + 0*j + sqrt(2)/2*k\n\n        "
        if len(angles) != 3:
            raise ValueError('3 angles must be given.')
        extrinsic = _is_extrinsic(seq)
        (i, j, k) = seq.lower()
        ei = [1 if n == i else 0 for n in 'xyz']
        ej = [1 if n == j else 0 for n in 'xyz']
        ek = [1 if n == k else 0 for n in 'xyz']
        qi = cls.from_axis_angle(ei, angles[0])
        qj = cls.from_axis_angle(ej, angles[1])
        qk = cls.from_axis_angle(ek, angles[2])
        if extrinsic:
            return trigsimp(qk * qj * qi)
        else:
            return trigsimp(qi * qj * qk)

    def to_euler(self, seq, angle_addition=True, avoid_square_root=False):
        if False:
            for i in range(10):
                print('nop')
        "Returns Euler angles representing same rotation as the quaternion,\n        in the sequence given by ``seq``. This implements the method described\n        in [1]_.\n\n        For degenerate cases (gymbal lock cases), the third angle is\n        set to zero.\n\n        Parameters\n        ==========\n\n        seq : string of length 3\n            Represents the sequence of rotations.\n            For extrinsic rotations, seq must be all lowercase and its elements\n            must be from the set ``{'x', 'y', 'z'}``\n            For intrinsic rotations, seq must be all uppercase and its elements\n            must be from the set ``{'X', 'Y', 'Z'}``\n\n        angle_addition : bool\n            When True, first and third angles are given as an addition and\n            subtraction of two simpler ``atan2`` expressions. When False, the\n            first and third angles are each given by a single more complicated\n            ``atan2`` expression. This equivalent expression is given by:\n\n            .. math::\n\n                \\operatorname{atan_2} (b,a) \\pm \\operatorname{atan_2} (d,c) =\n                \\operatorname{atan_2} (bc\\pm ad, ac\\mp bd)\n\n            Default value: True\n\n        avoid_square_root : bool\n            When True, the second angle is calculated with an expression based\n            on ``acos``, which is slightly more complicated but avoids a square\n            root. When False, second angle is calculated with ``atan2``, which\n            is simpler and can be better for numerical reasons (some\n            numerical implementations of ``acos`` have problems near zero).\n            Default value: False\n\n\n        Returns\n        =======\n\n        Tuple\n            The Euler angles calculated from the quaternion\n\n        Examples\n        ========\n\n        >>> from sympy import Quaternion\n        >>> from sympy.abc import a, b, c, d\n        >>> euler = Quaternion(a, b, c, d).to_euler('zyz')\n        >>> euler\n        (-atan2(-b, c) + atan2(d, a),\n         2*atan2(sqrt(b**2 + c**2), sqrt(a**2 + d**2)),\n         atan2(-b, c) + atan2(d, a))\n\n\n        References\n        ==========\n\n        .. [1] https://doi.org/10.1371/journal.pone.0276302\n\n        "
        if self.is_zero_quaternion():
            raise ValueError('Cannot convert a quaternion with norm 0.')
        angles = [0, 0, 0]
        extrinsic = _is_extrinsic(seq)
        (i, j, k) = seq.lower()
        i = 'xyz'.index(i) + 1
        j = 'xyz'.index(j) + 1
        k = 'xyz'.index(k) + 1
        if not extrinsic:
            (i, k) = (k, i)
        symmetric = i == k
        if symmetric:
            k = 6 - i - j
        sign = (i - j) * (j - k) * (k - i) // 2
        elements = [self.a, self.b, self.c, self.d]
        a = elements[0]
        b = elements[i]
        c = elements[j]
        d = elements[k] * sign
        if not symmetric:
            (a, b, c, d) = (a - c, b + d, c + a, d - b)
        if avoid_square_root:
            if symmetric:
                n2 = self.norm() ** 2
                angles[1] = acos((a * a + b * b - c * c - d * d) / n2)
            else:
                n2 = 2 * self.norm() ** 2
                angles[1] = asin((c * c + d * d - a * a - b * b) / n2)
        else:
            angles[1] = 2 * atan2(sqrt(c * c + d * d), sqrt(a * a + b * b))
            if not symmetric:
                angles[1] -= S.Pi / 2
        case = 0
        if is_eq(c, S.Zero) and is_eq(d, S.Zero):
            case = 1
        if is_eq(a, S.Zero) and is_eq(b, S.Zero):
            case = 2
        if case == 0:
            if angle_addition:
                angles[0] = atan2(b, a) + atan2(d, c)
                angles[2] = atan2(b, a) - atan2(d, c)
            else:
                angles[0] = atan2(b * c + a * d, a * c - b * d)
                angles[2] = atan2(b * c - a * d, a * c + b * d)
        else:
            angles[2 * (not extrinsic)] = S.Zero
            if case == 1:
                angles[2 * extrinsic] = 2 * atan2(b, a)
            else:
                angles[2 * extrinsic] = 2 * atan2(d, c)
                angles[2 * extrinsic] *= -1 if extrinsic else 1
        if not symmetric:
            angles[0] *= sign
        if extrinsic:
            return tuple(angles[::-1])
        else:
            return tuple(angles)

    @classmethod
    def from_axis_angle(cls, vector, angle):
        if False:
            i = 10
            return i + 15
        'Returns a rotation quaternion given the axis and the angle of rotation.\n\n        Parameters\n        ==========\n\n        vector : tuple of three numbers\n            The vector representation of the given axis.\n        angle : number\n            The angle by which axis is rotated (in radians).\n\n        Returns\n        =======\n\n        Quaternion\n            The normalized rotation quaternion calculated from the given axis and the angle of rotation.\n\n        Examples\n        ========\n\n        >>> from sympy import Quaternion\n        >>> from sympy import pi, sqrt\n        >>> q = Quaternion.from_axis_angle((sqrt(3)/3, sqrt(3)/3, sqrt(3)/3), 2*pi/3)\n        >>> q\n        1/2 + 1/2*i + 1/2*j + 1/2*k\n\n        '
        (x, y, z) = vector
        norm = sqrt(x ** 2 + y ** 2 + z ** 2)
        (x, y, z) = (x / norm, y / norm, z / norm)
        s = sin(angle * S.Half)
        a = cos(angle * S.Half)
        b = x * s
        c = y * s
        d = z * s
        return cls(a, b, c, d)

    @classmethod
    def from_rotation_matrix(cls, M):
        if False:
            for i in range(10):
                print('nop')
        "Returns the equivalent quaternion of a matrix. The quaternion will be normalized\n        only if the matrix is special orthogonal (orthogonal and det(M) = 1).\n\n        Parameters\n        ==========\n\n        M : Matrix\n            Input matrix to be converted to equivalent quaternion. M must be special\n            orthogonal (orthogonal and det(M) = 1) for the quaternion to be normalized.\n\n        Returns\n        =======\n\n        Quaternion\n            The quaternion equivalent to given matrix.\n\n        Examples\n        ========\n\n        >>> from sympy import Quaternion\n        >>> from sympy import Matrix, symbols, cos, sin, trigsimp\n        >>> x = symbols('x')\n        >>> M = Matrix([[cos(x), -sin(x), 0], [sin(x), cos(x), 0], [0, 0, 1]])\n        >>> q = trigsimp(Quaternion.from_rotation_matrix(M))\n        >>> q\n        sqrt(2)*sqrt(cos(x) + 1)/2 + 0*i + 0*j + sqrt(2 - 2*cos(x))*sign(sin(x))/2*k\n\n        "
        absQ = M.det() ** Rational(1, 3)
        a = sqrt(absQ + M[0, 0] + M[1, 1] + M[2, 2]) / 2
        b = sqrt(absQ + M[0, 0] - M[1, 1] - M[2, 2]) / 2
        c = sqrt(absQ - M[0, 0] + M[1, 1] - M[2, 2]) / 2
        d = sqrt(absQ - M[0, 0] - M[1, 1] + M[2, 2]) / 2
        b = b * sign(M[2, 1] - M[1, 2])
        c = c * sign(M[0, 2] - M[2, 0])
        d = d * sign(M[1, 0] - M[0, 1])
        return Quaternion(a, b, c, d)

    def __add__(self, other):
        if False:
            return 10
        return self.add(other)

    def __radd__(self, other):
        if False:
            print('Hello World!')
        return self.add(other)

    def __sub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.add(other * -1)

    def __mul__(self, other):
        if False:
            while True:
                i = 10
        return self._generic_mul(self, _sympify(other))

    def __rmul__(self, other):
        if False:
            print('Hello World!')
        return self._generic_mul(_sympify(other), self)

    def __pow__(self, p):
        if False:
            for i in range(10):
                print('nop')
        return self.pow(p)

    def __neg__(self):
        if False:
            i = 10
            return i + 15
        return Quaternion(-self.a, -self.b, -self.c, -self.d)

    def __truediv__(self, other):
        if False:
            i = 10
            return i + 15
        return self * sympify(other) ** (-1)

    def __rtruediv__(self, other):
        if False:
            print('Hello World!')
        return sympify(other) * self ** (-1)

    def _eval_Integral(self, *args):
        if False:
            print('Hello World!')
        return self.integrate(*args)

    def diff(self, *symbols, **kwargs):
        if False:
            print('Hello World!')
        kwargs.setdefault('evaluate', True)
        return self.func(*[a.diff(*symbols, **kwargs) for a in self.args])

    def add(self, other):
        if False:
            print('Hello World!')
        "Adds quaternions.\n\n        Parameters\n        ==========\n\n        other : Quaternion\n            The quaternion to add to current (self) quaternion.\n\n        Returns\n        =======\n\n        Quaternion\n            The resultant quaternion after adding self to other\n\n        Examples\n        ========\n\n        >>> from sympy import Quaternion\n        >>> from sympy import symbols\n        >>> q1 = Quaternion(1, 2, 3, 4)\n        >>> q2 = Quaternion(5, 6, 7, 8)\n        >>> q1.add(q2)\n        6 + 8*i + 10*j + 12*k\n        >>> q1 + 5\n        6 + 2*i + 3*j + 4*k\n        >>> x = symbols('x', real = True)\n        >>> q1.add(x)\n        (x + 1) + 2*i + 3*j + 4*k\n\n        Quaternions over complex fields :\n\n        >>> from sympy import Quaternion\n        >>> from sympy import I\n        >>> q3 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)\n        >>> q3.add(2 + 3*I)\n        (5 + 7*I) + (2 + 5*I)*i + 0*j + (7 + 8*I)*k\n\n        "
        q1 = self
        q2 = sympify(other)
        if not isinstance(q2, Quaternion):
            if q1.real_field and q2.is_complex:
                return Quaternion(re(q2) + q1.a, im(q2) + q1.b, q1.c, q1.d)
            elif q2.is_commutative:
                return Quaternion(q1.a + q2, q1.b, q1.c, q1.d)
            else:
                raise ValueError('Only commutative expressions can be added with a Quaternion.')
        return Quaternion(q1.a + q2.a, q1.b + q2.b, q1.c + q2.c, q1.d + q2.d)

    def mul(self, other):
        if False:
            i = 10
            return i + 15
        "Multiplies quaternions.\n\n        Parameters\n        ==========\n\n        other : Quaternion or symbol\n            The quaternion to multiply to current (self) quaternion.\n\n        Returns\n        =======\n\n        Quaternion\n            The resultant quaternion after multiplying self with other\n\n        Examples\n        ========\n\n        >>> from sympy import Quaternion\n        >>> from sympy import symbols\n        >>> q1 = Quaternion(1, 2, 3, 4)\n        >>> q2 = Quaternion(5, 6, 7, 8)\n        >>> q1.mul(q2)\n        (-60) + 12*i + 30*j + 24*k\n        >>> q1.mul(2)\n        2 + 4*i + 6*j + 8*k\n        >>> x = symbols('x', real = True)\n        >>> q1.mul(x)\n        x + 2*x*i + 3*x*j + 4*x*k\n\n        Quaternions over complex fields :\n\n        >>> from sympy import Quaternion\n        >>> from sympy import I\n        >>> q3 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)\n        >>> q3.mul(2 + 3*I)\n        (2 + 3*I)*(3 + 4*I) + (2 + 3*I)*(2 + 5*I)*i + 0*j + (2 + 3*I)*(7 + 8*I)*k\n\n        "
        return self._generic_mul(self, _sympify(other))

    @staticmethod
    def _generic_mul(q1, q2):
        if False:
            return 10
        "Generic multiplication.\n\n        Parameters\n        ==========\n\n        q1 : Quaternion or symbol\n        q2 : Quaternion or symbol\n\n        It is important to note that if neither q1 nor q2 is a Quaternion,\n        this function simply returns q1 * q2.\n\n        Returns\n        =======\n\n        Quaternion\n            The resultant quaternion after multiplying q1 and q2\n\n        Examples\n        ========\n\n        >>> from sympy import Quaternion\n        >>> from sympy import Symbol, S\n        >>> q1 = Quaternion(1, 2, 3, 4)\n        >>> q2 = Quaternion(5, 6, 7, 8)\n        >>> Quaternion._generic_mul(q1, q2)\n        (-60) + 12*i + 30*j + 24*k\n        >>> Quaternion._generic_mul(q1, S(2))\n        2 + 4*i + 6*j + 8*k\n        >>> x = Symbol('x', real = True)\n        >>> Quaternion._generic_mul(q1, x)\n        x + 2*x*i + 3*x*j + 4*x*k\n\n        Quaternions over complex fields :\n\n        >>> from sympy import I\n        >>> q3 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)\n        >>> Quaternion._generic_mul(q3, 2 + 3*I)\n        (2 + 3*I)*(3 + 4*I) + (2 + 3*I)*(2 + 5*I)*i + 0*j + (2 + 3*I)*(7 + 8*I)*k\n\n        "
        if not isinstance(q1, Quaternion) and (not isinstance(q2, Quaternion)):
            return q1 * q2
        if not isinstance(q1, Quaternion):
            if q2.real_field and q1.is_complex:
                return Quaternion(re(q1), im(q1), 0, 0) * q2
            elif q1.is_commutative:
                return Quaternion(q1 * q2.a, q1 * q2.b, q1 * q2.c, q1 * q2.d)
            else:
                raise ValueError('Only commutative expressions can be multiplied with a Quaternion.')
        if not isinstance(q2, Quaternion):
            if q1.real_field and q2.is_complex:
                return q1 * Quaternion(re(q2), im(q2), 0, 0)
            elif q2.is_commutative:
                return Quaternion(q2 * q1.a, q2 * q1.b, q2 * q1.c, q2 * q1.d)
            else:
                raise ValueError('Only commutative expressions can be multiplied with a Quaternion.')
        if q1._norm is None and q2._norm is None:
            norm = None
        else:
            norm = q1.norm() * q2.norm()
        return Quaternion(-q1.b * q2.b - q1.c * q2.c - q1.d * q2.d + q1.a * q2.a, q1.b * q2.a + q1.c * q2.d - q1.d * q2.c + q1.a * q2.b, -q1.b * q2.d + q1.c * q2.a + q1.d * q2.b + q1.a * q2.c, q1.b * q2.c - q1.c * q2.b + q1.d * q2.a + q1.a * q2.d, norm=norm)

    def _eval_conjugate(self):
        if False:
            i = 10
            return i + 15
        'Returns the conjugate of the quaternion.'
        q = self
        return Quaternion(q.a, -q.b, -q.c, -q.d, norm=q._norm)

    def norm(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the norm of the quaternion.'
        if self._norm is None:
            q = self
            return sqrt(trigsimp(q.a ** 2 + q.b ** 2 + q.c ** 2 + q.d ** 2))
        return self._norm

    def normalize(self):
        if False:
            i = 10
            return i + 15
        'Returns the normalized form of the quaternion.'
        q = self
        return q * (1 / q.norm())

    def inverse(self):
        if False:
            i = 10
            return i + 15
        'Returns the inverse of the quaternion.'
        q = self
        if not q.norm():
            raise ValueError('Cannot compute inverse for a quaternion with zero norm')
        return conjugate(q) * (1 / q.norm() ** 2)

    def pow(self, p):
        if False:
            while True:
                i = 10
        'Finds the pth power of the quaternion.\n\n        Parameters\n        ==========\n\n        p : int\n            Power to be applied on quaternion.\n\n        Returns\n        =======\n\n        Quaternion\n            Returns the p-th power of the current quaternion.\n            Returns the inverse if p = -1.\n\n        Examples\n        ========\n\n        >>> from sympy import Quaternion\n        >>> q = Quaternion(1, 2, 3, 4)\n        >>> q.pow(4)\n        668 + (-224)*i + (-336)*j + (-448)*k\n\n        '
        try:
            (q, p) = (self, as_int(p))
        except ValueError:
            return NotImplemented
        if p < 0:
            (q, p) = (q.inverse(), -p)
        if p == 1:
            return q
        res = Quaternion(1, 0, 0, 0)
        while p > 0:
            if p & 1:
                res *= q
            q *= q
            p >>= 1
        return res

    def exp(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the exponential of $q$, given by $e^q$.\n\n        Returns\n        =======\n\n        Quaternion\n            The exponential of the quaternion.\n\n        Examples\n        ========\n\n        >>> from sympy import Quaternion\n        >>> q = Quaternion(1, 2, 3, 4)\n        >>> q.exp()\n        E*cos(sqrt(29))\n        + 2*sqrt(29)*E*sin(sqrt(29))/29*i\n        + 3*sqrt(29)*E*sin(sqrt(29))/29*j\n        + 4*sqrt(29)*E*sin(sqrt(29))/29*k\n\n        '
        q = self
        vector_norm = sqrt(q.b ** 2 + q.c ** 2 + q.d ** 2)
        a = exp(q.a) * cos(vector_norm)
        b = exp(q.a) * sin(vector_norm) * q.b / vector_norm
        c = exp(q.a) * sin(vector_norm) * q.c / vector_norm
        d = exp(q.a) * sin(vector_norm) * q.d / vector_norm
        return Quaternion(a, b, c, d)

    def _ln(self):
        if False:
            print('Hello World!')
        'Returns the natural logarithm of the quaternion (_ln(q)).\n\n        Examples\n        ========\n\n        >>> from sympy import Quaternion\n        >>> q = Quaternion(1, 2, 3, 4)\n        >>> q._ln()\n        log(sqrt(30))\n        + 2*sqrt(29)*acos(sqrt(30)/30)/29*i\n        + 3*sqrt(29)*acos(sqrt(30)/30)/29*j\n        + 4*sqrt(29)*acos(sqrt(30)/30)/29*k\n\n        '
        q = self
        vector_norm = sqrt(q.b ** 2 + q.c ** 2 + q.d ** 2)
        q_norm = q.norm()
        a = ln(q_norm)
        b = q.b * acos(q.a / q_norm) / vector_norm
        c = q.c * acos(q.a / q_norm) / vector_norm
        d = q.d * acos(q.a / q_norm) / vector_norm
        return Quaternion(a, b, c, d)

    def _eval_subs(self, *args):
        if False:
            i = 10
            return i + 15
        elements = [i.subs(*args) for i in self.args]
        norm = self._norm
        norm = norm.subs(*args)
        _check_norm(elements, norm)
        return Quaternion(*elements, norm=norm)

    def _eval_evalf(self, prec):
        if False:
            while True:
                i = 10
        'Returns the floating point approximations (decimal numbers) of the quaternion.\n\n        Returns\n        =======\n\n        Quaternion\n            Floating point approximations of quaternion(self)\n\n        Examples\n        ========\n\n        >>> from sympy import Quaternion\n        >>> from sympy import sqrt\n        >>> q = Quaternion(1/sqrt(1), 1/sqrt(2), 1/sqrt(3), 1/sqrt(4))\n        >>> q.evalf()\n        1.00000000000000\n        + 0.707106781186547*i\n        + 0.577350269189626*j\n        + 0.500000000000000*k\n\n        '
        nprec = prec_to_dps(prec)
        return Quaternion(*[arg.evalf(n=nprec) for arg in self.args])

    def pow_cos_sin(self, p):
        if False:
            i = 10
            return i + 15
        'Computes the pth power in the cos-sin form.\n\n        Parameters\n        ==========\n\n        p : int\n            Power to be applied on quaternion.\n\n        Returns\n        =======\n\n        Quaternion\n            The p-th power in the cos-sin form.\n\n        Examples\n        ========\n\n        >>> from sympy import Quaternion\n        >>> q = Quaternion(1, 2, 3, 4)\n        >>> q.pow_cos_sin(4)\n        900*cos(4*acos(sqrt(30)/30))\n        + 1800*sqrt(29)*sin(4*acos(sqrt(30)/30))/29*i\n        + 2700*sqrt(29)*sin(4*acos(sqrt(30)/30))/29*j\n        + 3600*sqrt(29)*sin(4*acos(sqrt(30)/30))/29*k\n\n        '
        q = self
        (v, angle) = q.to_axis_angle()
        q2 = Quaternion.from_axis_angle(v, p * angle)
        return q2 * q.norm() ** p

    def integrate(self, *args):
        if False:
            while True:
                i = 10
        'Computes integration of quaternion.\n\n        Returns\n        =======\n\n        Quaternion\n            Integration of the quaternion(self) with the given variable.\n\n        Examples\n        ========\n\n        Indefinite Integral of quaternion :\n\n        >>> from sympy import Quaternion\n        >>> from sympy.abc import x\n        >>> q = Quaternion(1, 2, 3, 4)\n        >>> q.integrate(x)\n        x + 2*x*i + 3*x*j + 4*x*k\n\n        Definite integral of quaternion :\n\n        >>> from sympy import Quaternion\n        >>> from sympy.abc import x\n        >>> q = Quaternion(1, 2, 3, 4)\n        >>> q.integrate((x, 1, 5))\n        4 + 8*i + 12*j + 16*k\n\n        '
        return Quaternion(integrate(self.a, *args), integrate(self.b, *args), integrate(self.c, *args), integrate(self.d, *args))

    @staticmethod
    def rotate_point(pin, r):
        if False:
            print('Hello World!')
        "Returns the coordinates of the point pin (a 3 tuple) after rotation.\n\n        Parameters\n        ==========\n\n        pin : tuple\n            A 3-element tuple of coordinates of a point which needs to be\n            rotated.\n        r : Quaternion or tuple\n            Axis and angle of rotation.\n\n            It's important to note that when r is a tuple, it must be of the form\n            (axis, angle)\n\n        Returns\n        =======\n\n        tuple\n            The coordinates of the point after rotation.\n\n        Examples\n        ========\n\n        >>> from sympy import Quaternion\n        >>> from sympy import symbols, trigsimp, cos, sin\n        >>> x = symbols('x')\n        >>> q = Quaternion(cos(x/2), 0, 0, sin(x/2))\n        >>> trigsimp(Quaternion.rotate_point((1, 1, 1), q))\n        (sqrt(2)*cos(x + pi/4), sqrt(2)*sin(x + pi/4), 1)\n        >>> (axis, angle) = q.to_axis_angle()\n        >>> trigsimp(Quaternion.rotate_point((1, 1, 1), (axis, angle)))\n        (sqrt(2)*cos(x + pi/4), sqrt(2)*sin(x + pi/4), 1)\n\n        "
        if isinstance(r, tuple):
            q = Quaternion.from_axis_angle(r[0], r[1])
        else:
            q = r.normalize()
        pout = q * Quaternion(0, pin[0], pin[1], pin[2]) * conjugate(q)
        return (pout.b, pout.c, pout.d)

    def to_axis_angle(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the axis and angle of rotation of a quaternion.\n\n        Returns\n        =======\n\n        tuple\n            Tuple of (axis, angle)\n\n        Examples\n        ========\n\n        >>> from sympy import Quaternion\n        >>> q = Quaternion(1, 1, 1, 1)\n        >>> (axis, angle) = q.to_axis_angle()\n        >>> axis\n        (sqrt(3)/3, sqrt(3)/3, sqrt(3)/3)\n        >>> angle\n        2*pi/3\n\n        '
        q = self
        if q.a.is_negative:
            q = q * -1
        q = q.normalize()
        angle = trigsimp(2 * acos(q.a))
        s = sqrt(1 - q.a * q.a)
        x = trigsimp(q.b / s)
        y = trigsimp(q.c / s)
        z = trigsimp(q.d / s)
        v = (x, y, z)
        t = (v, angle)
        return t

    def to_rotation_matrix(self, v=None, homogeneous=True):
        if False:
            i = 10
            return i + 15
        "Returns the equivalent rotation transformation matrix of the quaternion\n        which represents rotation about the origin if ``v`` is not passed.\n\n        Parameters\n        ==========\n\n        v : tuple or None\n            Default value: None\n        homogeneous : bool\n            When True, gives an expression that may be more efficient for\n            symbolic calculations but less so for direct evaluation. Both\n            formulas are mathematically equivalent.\n            Default value: True\n\n        Returns\n        =======\n\n        tuple\n            Returns the equivalent rotation transformation matrix of the quaternion\n            which represents rotation about the origin if v is not passed.\n\n        Examples\n        ========\n\n        >>> from sympy import Quaternion\n        >>> from sympy import symbols, trigsimp, cos, sin\n        >>> x = symbols('x')\n        >>> q = Quaternion(cos(x/2), 0, 0, sin(x/2))\n        >>> trigsimp(q.to_rotation_matrix())\n        Matrix([\n        [cos(x), -sin(x), 0],\n        [sin(x),  cos(x), 0],\n        [     0,       0, 1]])\n\n        Generates a 4x4 transformation matrix (used for rotation about a point\n        other than the origin) if the point(v) is passed as an argument.\n        "
        q = self
        s = q.norm() ** (-2)
        if homogeneous:
            m00 = s * (q.a ** 2 + q.b ** 2 - q.c ** 2 - q.d ** 2)
            m11 = s * (q.a ** 2 - q.b ** 2 + q.c ** 2 - q.d ** 2)
            m22 = s * (q.a ** 2 - q.b ** 2 - q.c ** 2 + q.d ** 2)
        else:
            m00 = 1 - 2 * s * (q.c ** 2 + q.d ** 2)
            m11 = 1 - 2 * s * (q.b ** 2 + q.d ** 2)
            m22 = 1 - 2 * s * (q.b ** 2 + q.c ** 2)
        m01 = 2 * s * (q.b * q.c - q.d * q.a)
        m02 = 2 * s * (q.b * q.d + q.c * q.a)
        m10 = 2 * s * (q.b * q.c + q.d * q.a)
        m12 = 2 * s * (q.c * q.d - q.b * q.a)
        m20 = 2 * s * (q.b * q.d - q.c * q.a)
        m21 = 2 * s * (q.c * q.d + q.b * q.a)
        if not v:
            return Matrix([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])
        else:
            (x, y, z) = v
            m03 = x - x * m00 - y * m01 - z * m02
            m13 = y - x * m10 - y * m11 - z * m12
            m23 = z - x * m20 - y * m21 - z * m22
            m30 = m31 = m32 = 0
            m33 = 1
            return Matrix([[m00, m01, m02, m03], [m10, m11, m12, m13], [m20, m21, m22, m23], [m30, m31, m32, m33]])

    def scalar_part(self):
        if False:
            i = 10
            return i + 15
        'Returns scalar part($\\mathbf{S}(q)$) of the quaternion q.\n\n        Explanation\n        ===========\n\n        Given a quaternion $q = a + bi + cj + dk$, returns $\\mathbf{S}(q) = a$.\n\n        Examples\n        ========\n\n        >>> from sympy.algebras.quaternion import Quaternion\n        >>> q = Quaternion(4, 8, 13, 12)\n        >>> q.scalar_part()\n        4\n\n        '
        return self.a

    def vector_part(self):
        if False:
            return 10
        '\n        Returns $\\mathbf{V}(q)$, the vector part of the quaternion $q$.\n\n        Explanation\n        ===========\n\n        Given a quaternion $q = a + bi + cj + dk$, returns $\\mathbf{V}(q) = bi + cj + dk$.\n\n        Examples\n        ========\n\n        >>> from sympy.algebras.quaternion import Quaternion\n        >>> q = Quaternion(1, 1, 1, 1)\n        >>> q.vector_part()\n        0 + 1*i + 1*j + 1*k\n\n        >>> q = Quaternion(4, 8, 13, 12)\n        >>> q.vector_part()\n        0 + 8*i + 13*j + 12*k\n\n        '
        return Quaternion(0, self.b, self.c, self.d)

    def axis(self):
        if False:
            return 10
        '\n        Returns $\\mathbf{Ax}(q)$, the axis of the quaternion $q$.\n\n        Explanation\n        ===========\n\n        Given a quaternion $q = a + bi + cj + dk$, returns $\\mathbf{Ax}(q)$  i.e., the versor of the vector part of that quaternion\n        equal to $\\mathbf{U}[\\mathbf{V}(q)]$.\n        The axis is always an imaginary unit with square equal to $-1 + 0i + 0j + 0k$.\n\n        Examples\n        ========\n\n        >>> from sympy.algebras.quaternion import Quaternion\n        >>> q = Quaternion(1, 1, 1, 1)\n        >>> q.axis()\n        0 + sqrt(3)/3*i + sqrt(3)/3*j + sqrt(3)/3*k\n\n        See Also\n        ========\n\n        vector_part\n\n        '
        axis = self.vector_part().normalize()
        return Quaternion(0, axis.b, axis.c, axis.d)

    def is_pure(self):
        if False:
            while True:
                i = 10
        '\n        Returns true if the quaternion is pure, false if the quaternion is not pure\n        or returns none if it is unknown.\n\n        Explanation\n        ===========\n\n        A pure quaternion (also a vector quaternion) is a quaternion with scalar\n        part equal to 0.\n\n        Examples\n        ========\n\n        >>> from sympy.algebras.quaternion import Quaternion\n        >>> q = Quaternion(0, 8, 13, 12)\n        >>> q.is_pure()\n        True\n\n        See Also\n        ========\n        scalar_part\n\n        '
        return self.a.is_zero

    def is_zero_quaternion(self):
        if False:
            return 10
        '\n        Returns true if the quaternion is a zero quaternion or false if it is not a zero quaternion\n        and None if the value is unknown.\n\n        Explanation\n        ===========\n\n        A zero quaternion is a quaternion with both scalar part and\n        vector part equal to 0.\n\n        Examples\n        ========\n\n        >>> from sympy.algebras.quaternion import Quaternion\n        >>> q = Quaternion(1, 0, 0, 0)\n        >>> q.is_zero_quaternion()\n        False\n\n        >>> q = Quaternion(0, 0, 0, 0)\n        >>> q.is_zero_quaternion()\n        True\n\n        See Also\n        ========\n        scalar_part\n        vector_part\n\n        '
        return self.norm().is_zero

    def angle(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the angle of the quaternion measured in the real-axis plane.\n\n        Explanation\n        ===========\n\n        Given a quaternion $q = a + bi + cj + dk$ where $a$, $b$, $c$ and $d$\n        are real numbers, returns the angle of the quaternion given by\n\n        .. math::\n            \\theta := 2 \\operatorname{atan_2}\\left(\\sqrt{b^2 + c^2 + d^2}, {a}\\right)\n\n        Examples\n        ========\n\n        >>> from sympy.algebras.quaternion import Quaternion\n        >>> q = Quaternion(1, 4, 4, 4)\n        >>> q.angle()\n        2*atan(4*sqrt(3))\n\n        '
        return 2 * atan2(self.vector_part().norm(), self.scalar_part())

    def arc_coplanar(self, other):
        if False:
            return 10
        '\n        Returns True if the transformation arcs represented by the input quaternions happen in the same plane.\n\n        Explanation\n        ===========\n\n        Two quaternions are said to be coplanar (in this arc sense) when their axes are parallel.\n        The plane of a quaternion is the one normal to its axis.\n\n        Parameters\n        ==========\n\n        other : a Quaternion\n\n        Returns\n        =======\n\n        True : if the planes of the two quaternions are the same, apart from its orientation/sign.\n        False : if the planes of the two quaternions are not the same, apart from its orientation/sign.\n        None : if plane of either of the quaternion is unknown.\n\n        Examples\n        ========\n\n        >>> from sympy.algebras.quaternion import Quaternion\n        >>> q1 = Quaternion(1, 4, 4, 4)\n        >>> q2 = Quaternion(3, 8, 8, 8)\n        >>> Quaternion.arc_coplanar(q1, q2)\n        True\n\n        >>> q1 = Quaternion(2, 8, 13, 12)\n        >>> Quaternion.arc_coplanar(q1, q2)\n        False\n\n        See Also\n        ========\n\n        vector_coplanar\n        is_pure\n\n        '
        if self.is_zero_quaternion() or other.is_zero_quaternion():
            raise ValueError('Neither of the given quaternions can be 0')
        return fuzzy_or([(self.axis() - other.axis()).is_zero_quaternion(), (self.axis() + other.axis()).is_zero_quaternion()])

    @classmethod
    def vector_coplanar(cls, q1, q2, q3):
        if False:
            while True:
                i = 10
        '\n        Returns True if the axis of the pure quaternions seen as 3D vectors\n        ``q1``, ``q2``, and ``q3`` are coplanar.\n\n        Explanation\n        ===========\n\n        Three pure quaternions are vector coplanar if the quaternions seen as 3D vectors are coplanar.\n\n        Parameters\n        ==========\n\n        q1\n            A pure Quaternion.\n        q2\n            A pure Quaternion.\n        q3\n            A pure Quaternion.\n\n        Returns\n        =======\n\n        True : if the axis of the pure quaternions seen as 3D vectors\n        q1, q2, and q3 are coplanar.\n        False : if the axis of the pure quaternions seen as 3D vectors\n        q1, q2, and q3 are not coplanar.\n        None : if the axis of the pure quaternions seen as 3D vectors\n        q1, q2, and q3 are coplanar is unknown.\n\n        Examples\n        ========\n\n        >>> from sympy.algebras.quaternion import Quaternion\n        >>> q1 = Quaternion(0, 4, 4, 4)\n        >>> q2 = Quaternion(0, 8, 8, 8)\n        >>> q3 = Quaternion(0, 24, 24, 24)\n        >>> Quaternion.vector_coplanar(q1, q2, q3)\n        True\n\n        >>> q1 = Quaternion(0, 8, 16, 8)\n        >>> q2 = Quaternion(0, 8, 3, 12)\n        >>> Quaternion.vector_coplanar(q1, q2, q3)\n        False\n\n        See Also\n        ========\n\n        axis\n        is_pure\n\n        '
        if fuzzy_not(q1.is_pure()) or fuzzy_not(q2.is_pure()) or fuzzy_not(q3.is_pure()):
            raise ValueError('The given quaternions must be pure')
        M = Matrix([[q1.b, q1.c, q1.d], [q2.b, q2.c, q2.d], [q3.b, q3.c, q3.d]]).det()
        return M.is_zero

    def parallel(self, other):
        if False:
            i = 10
            return i + 15
        '\n        Returns True if the two pure quaternions seen as 3D vectors are parallel.\n\n        Explanation\n        ===========\n\n        Two pure quaternions are called parallel when their vector product is commutative which\n        implies that the quaternions seen as 3D vectors have same direction.\n\n        Parameters\n        ==========\n\n        other : a Quaternion\n\n        Returns\n        =======\n\n        True : if the two pure quaternions seen as 3D vectors are parallel.\n        False : if the two pure quaternions seen as 3D vectors are not parallel.\n        None : if the two pure quaternions seen as 3D vectors are parallel is unknown.\n\n        Examples\n        ========\n\n        >>> from sympy.algebras.quaternion import Quaternion\n        >>> q = Quaternion(0, 4, 4, 4)\n        >>> q1 = Quaternion(0, 8, 8, 8)\n        >>> q.parallel(q1)\n        True\n\n        >>> q1 = Quaternion(0, 8, 13, 12)\n        >>> q.parallel(q1)\n        False\n\n        '
        if fuzzy_not(self.is_pure()) or fuzzy_not(other.is_pure()):
            raise ValueError('The provided quaternions must be pure')
        return (self * other - other * self).is_zero_quaternion()

    def orthogonal(self, other):
        if False:
            i = 10
            return i + 15
        '\n        Returns the orthogonality of two quaternions.\n\n        Explanation\n        ===========\n\n        Two pure quaternions are called orthogonal when their product is anti-commutative.\n\n        Parameters\n        ==========\n\n        other : a Quaternion\n\n        Returns\n        =======\n\n        True : if the two pure quaternions seen as 3D vectors are orthogonal.\n        False : if the two pure quaternions seen as 3D vectors are not orthogonal.\n        None : if the two pure quaternions seen as 3D vectors are orthogonal is unknown.\n\n        Examples\n        ========\n\n        >>> from sympy.algebras.quaternion import Quaternion\n        >>> q = Quaternion(0, 4, 4, 4)\n        >>> q1 = Quaternion(0, 8, 8, 8)\n        >>> q.orthogonal(q1)\n        False\n\n        >>> q1 = Quaternion(0, 2, 2, 0)\n        >>> q = Quaternion(0, 2, -2, 0)\n        >>> q.orthogonal(q1)\n        True\n\n        '
        if fuzzy_not(self.is_pure()) or fuzzy_not(other.is_pure()):
            raise ValueError('The given quaternions must be pure')
        return (self * other + other * self).is_zero_quaternion()

    def index_vector(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the index vector of the quaternion.\n\n        Explanation\n        ===========\n\n        The index vector is given by $\\mathbf{T}(q)$, the norm (or magnitude) of\n        the quaternion $q$, multiplied by $\\mathbf{Ax}(q)$, the axis of $q$.\n\n        Returns\n        =======\n\n        Quaternion: representing index vector of the provided quaternion.\n\n        Examples\n        ========\n\n        >>> from sympy.algebras.quaternion import Quaternion\n        >>> q = Quaternion(2, 4, 2, 4)\n        >>> q.index_vector()\n        0 + 4*sqrt(10)/3*i + 2*sqrt(10)/3*j + 4*sqrt(10)/3*k\n\n        See Also\n        ========\n\n        axis\n        norm\n\n        '
        return self.norm() * self.axis()

    def mensor(self):
        if False:
            return 10
        '\n        Returns the natural logarithm of the norm(magnitude) of the quaternion.\n\n        Examples\n        ========\n\n        >>> from sympy.algebras.quaternion import Quaternion\n        >>> q = Quaternion(2, 4, 2, 4)\n        >>> q.mensor()\n        log(2*sqrt(10))\n        >>> q.norm()\n        2*sqrt(10)\n\n        See Also\n        ========\n\n        norm\n\n        '
        return ln(self.norm())