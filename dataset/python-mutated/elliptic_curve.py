from sympy.core.numbers import oo
from sympy.core.symbol import symbols
from sympy.polys.domains import FiniteField, QQ, RationalField, FF
from sympy.polys.polytools import Poly
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int
from .factor_ import divisors
from .residue_ntheory import polynomial_congruence

class EllipticCurve:
    """
    Create the following Elliptic Curve over domain.

    `y^{2} + a_{1} x y + a_{3} y = x^{3} + a_{2} x^{2} + a_{4} x + a_{6}`

    The default domain is ``QQ``. If no coefficient ``a1``, ``a2``, ``a3``,
    is given then it creates a curve with the following form:

    `y^{2} = x^{3} + a_{4} x + a_{6}`

    Examples
    ========

    References
    ==========

    .. [1] J. Silverman "A Friendly Introduction to Number Theory" Third Edition
    .. [2] https://mathworld.wolfram.com/EllipticDiscriminant.html
    .. [3] G. Hardy, E. Wright "An Introduction to the Theory of Numbers" Sixth Edition

    """

    def __init__(self, a4, a6, a1=0, a2=0, a3=0, modulus=0):
        if False:
            print('Hello World!')
        if modulus == 0:
            domain = QQ
        else:
            domain = FF(modulus)
        (a1, a2, a3, a4, a6) = map(domain.convert, (a1, a2, a3, a4, a6))
        self._domain = domain
        self.modulus = modulus
        b2 = a1 ** 2 + 4 * a2
        b4 = 2 * a4 + a1 * a3
        b6 = a3 ** 2 + 4 * a6
        b8 = a1 ** 2 * a6 + 4 * a2 * a6 - a1 * a3 * a4 + a2 * a3 ** 2 - a4 ** 2
        (self._b2, self._b4, self._b6, self._b8) = (b2, b4, b6, b8)
        self._discrim = -b2 ** 2 * b8 - 8 * b4 ** 3 - 27 * b6 ** 2 + 9 * b2 * b4 * b6
        self._a1 = a1
        self._a2 = a2
        self._a3 = a3
        self._a4 = a4
        self._a6 = a6
        (x, y, z) = symbols('x y z')
        (self.x, self.y, self.z) = (x, y, z)
        self._poly = Poly(y ** 2 * z + a1 * x * y * z + a3 * y * z ** 2 - x ** 3 - a2 * x ** 2 * z - a4 * x * z ** 2 - a6 * z ** 3, domain=domain)
        if isinstance(self._domain, FiniteField):
            self._rank = 0
        elif isinstance(self._domain, RationalField):
            self._rank = None

    def __call__(self, x, y, z=1):
        if False:
            i = 10
            return i + 15
        return EllipticCurvePoint(x, y, z, self)

    def __contains__(self, point):
        if False:
            i = 10
            return i + 15
        if is_sequence(point):
            if len(point) == 2:
                z1 = 1
            else:
                z1 = point[2]
            (x1, y1) = point[:2]
        elif isinstance(point, EllipticCurvePoint):
            (x1, y1, z1) = (point.x, point.y, point.z)
        else:
            raise ValueError('Invalid point.')
        if self.characteristic == 0 and z1 == 0:
            return True
        return self._poly.subs({self.x: x1, self.y: y1, self.z: z1}) == 0

    def __repr__(self):
        if False:
            return 10
        return self._poly.__repr__()

    def minimal(self):
        if False:
            i = 10
            return i + 15
        "\n        Return minimal Weierstrass equation.\n\n        Examples\n        ========\n\n        >>> from sympy.ntheory.elliptic_curve import EllipticCurve\n\n        >>> e1 = EllipticCurve(-10, -20, 0, -1, 1)\n        >>> e1.minimal()\n        Poly(-x**3 + 13392*x*z**2 + y**2*z + 1080432*z**3, x, y, z, domain='QQ')\n\n        "
        char = self.characteristic
        if char == 2:
            return self
        if char == 3:
            return EllipticCurve(self._b4 / 2, self._b6 / 4, a2=self._b2 / 4, modulus=self.modulus)
        c4 = self._b2 ** 2 - 24 * self._b4
        c6 = -self._b2 ** 3 + 36 * self._b2 * self._b4 - 216 * self._b6
        return EllipticCurve(-27 * c4, -54 * c6, modulus=self.modulus)

    def points(self):
        if False:
            i = 10
            return i + 15
        '\n        Return points of curve over Finite Field.\n\n        Examples\n        ========\n\n        >>> from sympy.ntheory.elliptic_curve import EllipticCurve\n        >>> e2 = EllipticCurve(1, 1, 1, 1, 1, modulus=5)\n        >>> e2.points()\n        {(0, 2), (1, 4), (2, 0), (2, 2), (3, 0), (3, 1), (4, 0)}\n\n        '
        char = self.characteristic
        all_pt = set()
        if char >= 1:
            for i in range(char):
                congruence_eq = self._poly.subs({self.x: i, self.z: 1}).expr
                sol = polynomial_congruence(congruence_eq, char)
                for num in sol:
                    all_pt.add((i, num))
            return all_pt
        else:
            raise ValueError('Infinitely many points')

    def points_x(self, x):
        if False:
            print('Hello World!')
        'Returns points on the curve for the given x-coordinate.'
        pt = []
        if self._domain == QQ:
            for y in solve(self._poly.subs(self.x, x)):
                pt.append((x, y))
        else:
            congruence_eq = self._poly.subs({self.x: x, self.z: 1}).expr
            for y in polynomial_congruence(congruence_eq, self.characteristic):
                pt.append((x, y))
        return pt

    def torsion_points(self):
        if False:
            return 10
        "\n        Return torsion points of curve over Rational number.\n\n        Return point objects those are finite order.\n        According to Nagell-Lutz theorem, torsion point p(x, y)\n        x and y are integers, either y = 0 or y**2 is divisor\n        of discriminent. According to Mazur's theorem, there are\n        at most 15 points in torsion collection.\n\n        Examples\n        ========\n\n        >>> from sympy.ntheory.elliptic_curve import EllipticCurve\n        >>> e2 = EllipticCurve(-43, 166)\n        >>> sorted(e2.torsion_points())\n        [(-5, -16), (-5, 16), O, (3, -8), (3, 8), (11, -32), (11, 32)]\n\n        "
        if self.characteristic > 0:
            raise ValueError('No torsion point for Finite Field.')
        l = [EllipticCurvePoint.point_at_infinity(self)]
        for xx in solve(self._poly.subs({self.y: 0, self.z: 1})):
            if xx.is_rational:
                l.append(self(xx, 0))
        for i in divisors(self.discriminant, generator=True):
            j = int(i ** 0.5)
            if j ** 2 == i:
                for xx in solve(self._poly.subs({self.y: j, self.z: 1})):
                    if not xx.is_rational:
                        continue
                    p = self(xx, j)
                    if p.order() != oo:
                        l.extend([p, -p])
        return l

    @property
    def characteristic(self):
        if False:
            return 10
        '\n        Return domain characteristic.\n\n        Examples\n        ========\n\n        >>> from sympy.ntheory.elliptic_curve import EllipticCurve\n        >>> e2 = EllipticCurve(-43, 166)\n        >>> e2.characteristic\n        0\n\n        '
        return self._domain.characteristic()

    @property
    def discriminant(self):
        if False:
            return 10
        '\n        Return curve discriminant.\n\n        Examples\n        ========\n\n        >>> from sympy.ntheory.elliptic_curve import EllipticCurve\n        >>> e2 = EllipticCurve(0, 17)\n        >>> e2.discriminant\n        -124848\n\n        '
        return int(self._discrim)

    @property
    def is_singular(self):
        if False:
            print('Hello World!')
        '\n        Return True if curve discriminant is equal to zero.\n        '
        return self.discriminant == 0

    @property
    def j_invariant(self):
        if False:
            return 10
        '\n        Return curve j-invariant.\n\n        Examples\n        ========\n\n        >>> from sympy.ntheory.elliptic_curve import EllipticCurve\n        >>> e1 = EllipticCurve(-2, 0, 0, 1, 1)\n        >>> e1.j_invariant\n        1404928/389\n\n        '
        c4 = self._b2 ** 2 - 24 * self._b4
        return self._domain.to_sympy(c4 ** 3 / self._discrim)

    @property
    def order(self):
        if False:
            return 10
        '\n        Number of points in Finite field.\n\n        Examples\n        ========\n\n        >>> from sympy.ntheory.elliptic_curve import EllipticCurve\n        >>> e2 = EllipticCurve(1, 0, modulus=19)\n        >>> e2.order\n        19\n\n        '
        if self.characteristic == 0:
            raise NotImplementedError('Still not implemented')
        return len(self.points())

    @property
    def rank(self):
        if False:
            return 10
        '\n        Number of independent points of infinite order.\n\n        For Finite field, it must be 0.\n        '
        if self._rank is not None:
            return self._rank
        raise NotImplementedError('Still not implemented')

class EllipticCurvePoint:
    """
    Point of Elliptic Curve

    Examples
    ========

    >>> from sympy.ntheory.elliptic_curve import EllipticCurve
    >>> e1 = EllipticCurve(-17, 16)
    >>> p1 = e1(0, -4, 1)
    >>> p2 = e1(1, 0)
    >>> p1 + p2
    (15, -56)
    >>> e3 = EllipticCurve(-1, 9)
    >>> e3(1, -3) * 3
    (664/169, 17811/2197)
    >>> (e3(1, -3) * 3).order()
    oo
    >>> e2 = EllipticCurve(-2, 0, 0, 1, 1)
    >>> p = e2(-1,1)
    >>> q = e2(0, -1)
    >>> p+q
    (4, 8)
    >>> p-q
    (1, 0)
    >>> 3*p-5*q
    (328/361, -2800/6859)
    """

    @staticmethod
    def point_at_infinity(curve):
        if False:
            while True:
                i = 10
        return EllipticCurvePoint(0, 1, 0, curve)

    def __init__(self, x, y, z, curve):
        if False:
            for i in range(10):
                print('nop')
        dom = curve._domain.convert
        self.x = dom(x)
        self.y = dom(y)
        self.z = dom(z)
        self._curve = curve
        self._domain = self._curve._domain
        if not self._curve.__contains__(self):
            raise ValueError('The curve does not contain this point')

    def __add__(self, p):
        if False:
            return 10
        if self.z == 0:
            return p
        if p.z == 0:
            return self
        (x1, y1) = (self.x / self.z, self.y / self.z)
        (x2, y2) = (p.x / p.z, p.y / p.z)
        a1 = self._curve._a1
        a2 = self._curve._a2
        a3 = self._curve._a3
        a4 = self._curve._a4
        a6 = self._curve._a6
        if x1 != x2:
            slope = (y1 - y2) / (x1 - x2)
            yint = (y1 * x2 - y2 * x1) / (x2 - x1)
        else:
            if y1 + y2 == 0:
                return self.point_at_infinity(self._curve)
            slope = (3 * x1 ** 2 + 2 * a2 * x1 + a4 - a1 * y1) / (a1 * x1 + a3 + 2 * y1)
            yint = (-x1 ** 3 + a4 * x1 + 2 * a6 - a3 * y1) / (a1 * x1 + a3 + 2 * y1)
        x3 = slope ** 2 + a1 * slope - a2 - x1 - x2
        y3 = -(slope + a1) * x3 - yint - a3
        return self._curve(x3, y3, 1)

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        return (self.x, self.y, self.z) < (other.x, other.y, other.z)

    def __mul__(self, n):
        if False:
            i = 10
            return i + 15
        n = as_int(n)
        r = self.point_at_infinity(self._curve)
        if n == 0:
            return r
        if n < 0:
            return -self * -n
        p = self
        while n:
            if n & 1:
                r = r + p
            n >>= 1
            p = p + p
        return r

    def __rmul__(self, n):
        if False:
            i = 10
            return i + 15
        return self * n

    def __neg__(self):
        if False:
            return 10
        return EllipticCurvePoint(self.x, -self.y - self._curve._a1 * self.x - self._curve._a3, self.z, self._curve)

    def __repr__(self):
        if False:
            print('Hello World!')
        if self.z == 0:
            return 'O'
        dom = self._curve._domain
        try:
            return '({}, {})'.format(dom.to_sympy(self.x), dom.to_sympy(self.y))
        except TypeError:
            pass
        return '({}, {})'.format(self.x, self.y)

    def __sub__(self, other):
        if False:
            i = 10
            return i + 15
        return self.__add__(-other)

    def order(self):
        if False:
            print('Hello World!')
        '\n        Return point order n where nP = 0.\n\n        '
        if self.z == 0:
            return 1
        if self.y == 0:
            return 2
        p = self * 2
        if p.y == -self.y:
            return 3
        i = 2
        if self._domain != QQ:
            while int(p.x) == p.x and int(p.y) == p.y:
                p = self + p
                i += 1
                if p.z == 0:
                    return i
            return oo
        while p.x.numerator == p.x and p.y.numerator == p.y:
            p = self + p
            i += 1
            if i > 12:
                return oo
            if p.z == 0:
                return i
        return oo