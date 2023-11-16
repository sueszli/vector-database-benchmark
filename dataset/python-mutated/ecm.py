from math import log
from sympy.core.random import _randint
from sympy.external.gmpy import gcd, invert, sqrt
from sympy.utilities.misc import as_int
from .generate import sieve, primerange
from .primetest import isprime

class Point:
    """Montgomery form of Points in an elliptic curve.
    In this form, the addition and doubling of points
    does not need any y-coordinate information thus
    decreasing the number of operations.
    Using Montgomery form we try to perform point addition
    and doubling in least amount of multiplications.

    The elliptic curve used here is of the form
    (E : b*y**2*z = x**3 + a*x**2*z + x*z**2).
    The a_24 parameter is equal to (a + 2)/4.

    References
    ==========

    .. [1] Kris Gaj, Soonhak Kwon, Patrick Baier, Paul Kohlbrenner, Hoang Le, Mohammed Khaleeluddin, Ramakrishna Bachimanchi,
           Implementing the Elliptic Curve Method of Factoring in Reconfigurable Hardware,
           Cryptographic Hardware and Embedded Systems - CHES 2006 (2006), pp. 119-133,
           https://doi.org/10.1007/11894063_10
           https://www.hyperelliptic.org/tanja/SHARCS/talks06/Gaj.pdf

    """

    def __init__(self, x_cord, z_cord, a_24, mod):
        if False:
            i = 10
            return i + 15
        '\n        Initial parameters for the Point class.\n\n        Parameters\n        ==========\n\n        x_cord : X coordinate of the Point\n        z_cord : Z coordinate of the Point\n        a_24 : Parameter of the elliptic curve in Montgomery form\n        mod : modulus\n        '
        self.x_cord = x_cord
        self.z_cord = z_cord
        self.a_24 = a_24
        self.mod = mod

    def __eq__(self, other):
        if False:
            return 10
        'Two points are equal if X/Z of both points are equal\n        '
        if self.a_24 != other.a_24 or self.mod != other.mod:
            return False
        return self.x_cord * other.z_cord % self.mod == other.x_cord * self.z_cord % self.mod

    def add(self, Q, diff):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add two points self and Q where diff = self - Q. Moreover the assumption\n        is self.x_cord*Q.x_cord*(self.x_cord - Q.x_cord) != 0. This algorithm\n        requires 6 multiplications. Here the difference between the points\n        is already known and using this algorithm speeds up the addition\n        by reducing the number of multiplication required. Also in the\n        mont_ladder algorithm is constructed in a way so that the difference\n        between intermediate points is always equal to the initial point.\n        So, we always know what the difference between the point is.\n\n\n        Parameters\n        ==========\n\n        Q : point on the curve in Montgomery form\n        diff : self - Q\n\n        Examples\n        ========\n\n        >>> from sympy.ntheory.ecm import Point\n        >>> p1 = Point(11, 16, 7, 29)\n        >>> p2 = Point(13, 10, 7, 29)\n        >>> p3 = p2.add(p1, p1)\n        >>> p3.x_cord\n        23\n        >>> p3.z_cord\n        17\n        '
        u = (self.x_cord - self.z_cord) * (Q.x_cord + Q.z_cord)
        v = (self.x_cord + self.z_cord) * (Q.x_cord - Q.z_cord)
        (add, subt) = (u + v, u - v)
        x_cord = diff.z_cord * add * add % self.mod
        z_cord = diff.x_cord * subt * subt % self.mod
        return Point(x_cord, z_cord, self.a_24, self.mod)

    def double(self):
        if False:
            return 10
        '\n        Doubles a point in an elliptic curve in Montgomery form.\n        This algorithm requires 5 multiplications.\n\n        Examples\n        ========\n\n        >>> from sympy.ntheory.ecm import Point\n        >>> p1 = Point(11, 16, 7, 29)\n        >>> p2 = p1.double()\n        >>> p2.x_cord\n        13\n        >>> p2.z_cord\n        10\n        '
        u = pow(self.x_cord + self.z_cord, 2, self.mod)
        v = pow(self.x_cord - self.z_cord, 2, self.mod)
        diff = u - v
        x_cord = u * v % self.mod
        z_cord = diff * (v + self.a_24 * diff) % self.mod
        return Point(x_cord, z_cord, self.a_24, self.mod)

    def mont_ladder(self, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        Scalar multiplication of a point in Montgomery form\n        using Montgomery Ladder Algorithm.\n        A total of 11 multiplications are required in each step of this\n        algorithm.\n\n        Parameters\n        ==========\n\n        k : The positive integer multiplier\n\n        Examples\n        ========\n\n        >>> from sympy.ntheory.ecm import Point\n        >>> p1 = Point(11, 16, 7, 29)\n        >>> p3 = p1.mont_ladder(3)\n        >>> p3.x_cord\n        23\n        >>> p3.z_cord\n        17\n        '
        Q = self
        R = self.double()
        for i in bin(k)[3:]:
            if i == '1':
                Q = R.add(Q, self)
                R = R.double()
            else:
                R = Q.add(R, self)
                Q = Q.double()
        return Q

def _ecm_one_factor(n, B1=10000, B2=100000, max_curve=200, seed=None):
    if False:
        print('Hello World!')
    "Returns one factor of n using\n    Lenstra's 2 Stage Elliptic curve Factorization\n    with Suyama's Parameterization. Here Montgomery\n    arithmetic is used for fast computation of addition\n    and doubling of points in elliptic curve.\n\n    Explanation\n    ===========\n\n    This ECM method considers elliptic curves in Montgomery\n    form (E : b*y**2*z = x**3 + a*x**2*z + x*z**2) and involves\n    elliptic curve operations (mod N), where the elements in\n    Z are reduced (mod N). Since N is not a prime, E over FF(N)\n    is not really an elliptic curve but we can still do point additions\n    and doubling as if FF(N) was a field.\n\n    Stage 1 : The basic algorithm involves taking a random point (P) on an\n    elliptic curve in FF(N). The compute k*P using Montgomery ladder algorithm.\n    Let q be an unknown factor of N. Then the order of the curve E, |E(FF(q))|,\n    might be a smooth number that divides k. Then we have k = l * |E(FF(q))|\n    for some l. For any point belonging to the curve E, |E(FF(q))|*P = O,\n    hence k*P = l*|E(FF(q))|*P. Thus kP.z_cord = 0 (mod q), and the unknownn\n    factor of N (q) can be recovered by taking gcd(kP.z_cord, N).\n\n    Stage 2 : This is a continuation of Stage 1 if k*P != O. The idea utilize\n    the fact that even if kP != 0, the value of k might miss just one large\n    prime divisor of |E(FF(q))|. In this case we only need to compute the\n    scalar multiplication by p to get p*k*P = O. Here a second bound B2\n    restrict the size of possible values of p.\n\n    Parameters\n    ==========\n\n    n : Number to be Factored\n    B1 : Stage 1 Bound. Must be an even number.\n    B2 : Stage 2 Bound. Must be an even number.\n    max_curve : Maximum number of curves generated\n\n    Returns\n    =======\n\n    integer | None : ``n`` (if it is prime) else a non-trivial divisor of ``n``. ``None`` if not found\n\n    References\n    ==========\n\n    .. [1] Carl Pomerance, Richard Crandall, Prime Numbers: A Computational Perspective,\n           2nd Edition (2005), page 344, ISBN:978-0387252827\n    "
    randint = _randint(seed)
    if isprime(n):
        return n
    D = min(sqrt(B2), B1 // 2 - 1)
    sieve.extend(D)
    beta = [0] * D
    S = [0] * D
    k = 1
    for p in primerange(2, B1 + 1):
        k *= pow(p, int(log(B1, p)))
    deltas_list = []
    for r in range(B1 + 2 * D, B2 + 2 * D, 4 * D):
        deltas = set()
        for q in primerange(r - 2 * D, r + 2 * D):
            deltas.add((abs(q - r) - 1) // 2)
        deltas_list.append(list(deltas))
    for _ in range(max_curve):
        sigma = randint(6, n - 1)
        u = (sigma ** 2 - 5) % n
        v = 4 * sigma % n
        u_3 = pow(u, 3, n)
        try:
            a24 = pow(v - u, 3, n) * (3 * u + v) * invert(16 * u_3 * v, n) % n
        except ZeroDivisionError:
            g = gcd(2 * u_3 * v, n)
            if g == n:
                continue
            return g
        Q = Point(u_3, pow(v, 3, n), a24, n)
        Q = Q.mont_ladder(k)
        g = gcd(Q.z_cord, n)
        if g != 1 and g != n:
            return g
        elif g == n:
            continue
        S[0] = Q
        Q2 = Q.double()
        S[1] = Q2.add(Q, Q)
        beta[0] = S[0].x_cord * S[0].z_cord % n
        beta[1] = S[1].x_cord * S[1].z_cord % n
        for d in range(2, D):
            S[d] = S[d - 1].add(Q2, S[d - 2])
            beta[d] = S[d].x_cord * S[d].z_cord % n
        g = 1
        W = Q.mont_ladder(4 * D)
        T = Q.mont_ladder(B1 - 2 * D)
        R = Q.mont_ladder(B1 + 2 * D)
        for deltas in deltas_list:
            alpha = R.x_cord * R.z_cord % n
            for delta in deltas:
                f = (R.x_cord - S[delta].x_cord) * (R.z_cord + S[delta].z_cord) - alpha + beta[delta]
                g = g * f % n
            (T, R) = (R, R.add(W, T))
        g = gcd(n, g)
        if g != 1 and g != n:
            return g

def ecm(n, B1=10000, B2=100000, max_curve=200, seed=1234):
    if False:
        print('Hello World!')
    "Performs factorization using Lenstra's Elliptic curve method.\n\n    This function repeatedly calls ``_ecm_one_factor`` to compute the factors\n    of n. First all the small factors are taken out using trial division.\n    Then ``_ecm_one_factor`` is used to compute one factor at a time.\n\n    Parameters\n    ==========\n\n    n : Number to be Factored\n    B1 : Stage 1 Bound. Must be an even number.\n    B2 : Stage 2 Bound. Must be an even number.\n    max_curve : Maximum number of curves generated\n    seed : Initialize pseudorandom generator\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory import ecm\n    >>> ecm(25645121643901801)\n    {5394769, 4753701529}\n    >>> ecm(9804659461513846513)\n    {4641991, 2112166839943}\n    "
    n = as_int(n)
    if B1 % 2 != 0 or B2 % 2 != 0:
        raise ValueError('both bounds must be even')
    _factors = set()
    for prime in sieve.primerange(1, 100000):
        if n % prime == 0:
            _factors.add(prime)
            while n % prime == 0:
                n //= prime
    while n > 1:
        factor = _ecm_one_factor(n, B1, B2, max_curve, seed)
        if factor is None:
            raise ValueError('Increase the bounds')
        _factors.add(factor)
        n //= factor
    factors = set()
    for factor in _factors:
        if isprime(factor):
            factors.add(factor)
            continue
        factors |= ecm(factor)
    return factors