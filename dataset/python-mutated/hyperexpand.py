"""
Expand Hypergeometric (and Meijer G) functions into named
special functions.

The algorithm for doing this uses a collection of lookup tables of
hypergeometric functions, and various of their properties, to expand
many hypergeometric functions in terms of special functions.

It is based on the following paper:
      Kelly B. Roach.  Meijer G Function Representations.
      In: Proceedings of the 1997 International Symposium on Symbolic and
      Algebraic Computation, pages 205-211, New York, 1997. ACM.

It is described in great(er) detail in the Sphinx documentation.
"""
from collections import defaultdict
from itertools import product
from functools import reduce
from math import prod
from sympy import SYMPY_DEBUG
from sympy.core import S, Dummy, symbols, sympify, Tuple, expand, I, pi, Mul, EulerGamma, oo, zoo, expand_func, Add, nan, Expr, Rational
from sympy.core.mod import Mod
from sympy.core.sorting import default_sort_key
from sympy.functions import exp, sqrt, root, log, lowergamma, cos, besseli, gamma, uppergamma, expint, erf, sin, besselj, Ei, Ci, Si, Shi, sinh, cosh, Chi, fresnels, fresnelc, polar_lift, exp_polar, floor, ceiling, rf, factorial, lerchphi, Piecewise, re, elliptic_k, elliptic_e
from sympy.functions.elementary.complexes import polarify, unpolarify
from sympy.functions.special.hyper import hyper, HyperRep_atanh, HyperRep_power1, HyperRep_power2, HyperRep_log1, HyperRep_asin1, HyperRep_asin2, HyperRep_sqrts1, HyperRep_sqrts2, HyperRep_log2, HyperRep_cosasin, HyperRep_sinasin, meijerg
from sympy.matrices import Matrix, eye, zeros
from sympy.polys import apart, poly, Poly
from sympy.series import residue
from sympy.simplify.powsimp import powdenest
from sympy.utilities.iterables import sift

def _mod1(x):
    if False:
        i = 10
        return i + 15
    if x.is_Number:
        return Mod(x, 1)
    (c, x) = x.as_coeff_Add()
    return Mod(c, 1) + x

def add_formulae(formulae):
    if False:
        i = 10
        return i + 15
    ' Create our knowledge base. '
    (a, b, c, z) = symbols('a b c, z', cls=Dummy)

    def add(ap, bq, res):
        if False:
            return 10
        func = Hyper_Function(ap, bq)
        formulae.append(Formula(func, z, res, (a, b, c)))

    def addb(ap, bq, B, C, M):
        if False:
            return 10
        func = Hyper_Function(ap, bq)
        formulae.append(Formula(func, z, None, (a, b, c), B, C, M))
    add((), (), exp(z))
    add((a,), (), HyperRep_power1(-a, z))
    addb((a, a - S.Half), (2 * a,), Matrix([HyperRep_power2(a, z), HyperRep_power2(a + S.Half, z) / 2]), Matrix([[1, 0]]), Matrix([[(a - S.Half) * z / (1 - z), (S.Half - a) * z / (1 - z)], [a / (1 - z), a * (z - 2) / (1 - z)]]))
    addb((1, 1), (2,), Matrix([HyperRep_log1(z), 1]), Matrix([[-1 / z, 0]]), Matrix([[0, z / (z - 1)], [0, 0]]))
    addb((S.Half, 1), (S('3/2'),), Matrix([HyperRep_atanh(z), 1]), Matrix([[1, 0]]), Matrix([[Rational(-1, 2), 1 / (1 - z) / 2], [0, 0]]))
    addb((S.Half, S.Half), (S('3/2'),), Matrix([HyperRep_asin1(z), HyperRep_power1(Rational(-1, 2), z)]), Matrix([[1, 0]]), Matrix([[Rational(-1, 2), S.Half], [0, z / (1 - z) / 2]]))
    addb((a, S.Half + a), (S.Half,), Matrix([HyperRep_sqrts1(-a, z), -HyperRep_sqrts2(-a - S.Half, z)]), Matrix([[1, 0]]), Matrix([[0, -a], [z * (-2 * a - 1) / 2 / (1 - z), S.Half - z * (-2 * a - 1) / (1 - z)]]))
    addb([a, -a], [S.Half], Matrix([HyperRep_cosasin(a, z), HyperRep_sinasin(a, z)]), Matrix([[1, 0]]), Matrix([[0, -a], [a * z / (1 - z), 1 / (1 - z) / 2]]))
    addb([1, 1], [3 * S.Half], Matrix([HyperRep_asin2(z), 1]), Matrix([[1, 0]]), Matrix([[(z - S.Half) / (1 - z), 1 / (1 - z) / 2], [0, 0]]))
    addb([S.Half, S.Half], [S.One], Matrix([elliptic_k(z), elliptic_e(z)]), Matrix([[2 / pi, 0]]), Matrix([[Rational(-1, 2), -1 / (2 * z - 2)], [Rational(-1, 2), S.Half]]))
    addb([Rational(-1, 2), S.Half], [S.One], Matrix([elliptic_k(z), elliptic_e(z)]), Matrix([[0, 2 / pi]]), Matrix([[Rational(-1, 2), -1 / (2 * z - 2)], [Rational(-1, 2), S.Half]]))
    addb([Rational(-1, 2), 1, 1], [S.Half, 2], Matrix([z * HyperRep_atanh(z), HyperRep_log1(z), 1]), Matrix([[Rational(-2, 3), -S.One / (3 * z), Rational(2, 3)]]), Matrix([[S.Half, 0, z / (1 - z) / 2], [0, 0, z / (z - 1)], [0, 0, 0]]))
    addb([Rational(-1, 2), 1, 1], [2, 2], Matrix([HyperRep_power1(S.Half, z), HyperRep_log2(z), 1]), Matrix([[Rational(4, 9) - 16 / (9 * z), 4 / (3 * z), 16 / (9 * z)]]), Matrix([[z / 2 / (z - 1), 0, 0], [1 / (2 * (z - 1)), 0, S.Half], [0, 0, 0]]))
    addb([1], [b], Matrix([z ** (1 - b) * exp(z) * lowergamma(b - 1, z), 1]), Matrix([[b - 1, 0]]), Matrix([[1 - b + z, 1], [0, 0]]))
    addb([a], [2 * a], Matrix([z ** (S.Half - a) * exp(z / 2) * besseli(a - S.Half, z / 2) * gamma(a + S.Half) / 4 ** (S.Half - a), z ** (S.Half - a) * exp(z / 2) * besseli(a + S.Half, z / 2) * gamma(a + S.Half) / 4 ** (S.Half - a)]), Matrix([[1, 0]]), Matrix([[z / 2, z / 2], [z / 2, z / 2 - 2 * a]]))
    mz = polar_lift(-1) * z
    addb([a], [a + 1], Matrix([mz ** (-a) * a * lowergamma(a, mz), a * exp(z)]), Matrix([[1, 0]]), Matrix([[-a, 1], [0, z]]))
    add([Rational(-1, 2)], [S.Half], exp(z) - sqrt(pi * z) * -I * erf(I * sqrt(z)))
    addb([1], [Rational(3, 4), Rational(5, 4)], Matrix([sqrt(pi) * (I * sinh(2 * sqrt(z)) * fresnels(2 * root(z, 4) * exp(I * pi / 4) / sqrt(pi)) + cosh(2 * sqrt(z)) * fresnelc(2 * root(z, 4) * exp(I * pi / 4) / sqrt(pi))) * exp(-I * pi / 4) / (2 * root(z, 4)), sqrt(pi) * root(z, 4) * (sinh(2 * sqrt(z)) * fresnelc(2 * root(z, 4) * exp(I * pi / 4) / sqrt(pi)) + I * cosh(2 * sqrt(z)) * fresnels(2 * root(z, 4) * exp(I * pi / 4) / sqrt(pi))) * exp(-I * pi / 4) / 2, 1]), Matrix([[1, 0, 0]]), Matrix([[Rational(-1, 4), 1, Rational(1, 4)], [z, Rational(1, 4), 0], [0, 0, 0]]))
    addb([S.Half, a], [Rational(3, 2), a + 1], Matrix([a / (2 * a - 1) * -I * sqrt(pi / z) * erf(I * sqrt(z)), a / (2 * a - 1) * (polar_lift(-1) * z) ** (-a) * lowergamma(a, polar_lift(-1) * z), a / (2 * a - 1) * exp(z)]), Matrix([[1, -1, 0]]), Matrix([[Rational(-1, 2), 0, 1], [0, -a, 1], [0, 0, z]]))
    addb([1, 1], [2, 2], Matrix([Ei(z) - log(z), exp(z), 1, EulerGamma]), Matrix([[1 / z, 0, 0, -1 / z]]), Matrix([[0, 1, -1, 0], [0, z, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    add((), (S.Half,), cosh(2 * sqrt(z)))
    addb([], [b], Matrix([gamma(b) * z ** ((1 - b) / 2) * besseli(b - 1, 2 * sqrt(z)), gamma(b) * z ** (1 - b / 2) * besseli(b, 2 * sqrt(z))]), Matrix([[1, 0]]), Matrix([[0, 1], [z, 1 - b]]))
    x = 4 * z ** Rational(1, 4)

    def fp(a, z):
        if False:
            return 10
        return besseli(a, x) + besselj(a, x)

    def fm(a, z):
        if False:
            i = 10
            return i + 15
        return besseli(a, x) - besselj(a, x)
    addb([], [S.Half, a, a + S.Half], Matrix([fp(2 * a - 1, z), fm(2 * a, z) * z ** Rational(1, 4), fm(2 * a - 1, z) * sqrt(z), fp(2 * a, z) * z ** Rational(3, 4)]) * 2 ** (-2 * a) * gamma(2 * a) * z ** ((1 - 2 * a) / 4), Matrix([[1, 0, 0, 0]]), Matrix([[0, 1, 0, 0], [0, S.Half - a, 1, 0], [0, 0, S.Half, 1], [z, 0, 0, 1 - a]]))
    x = 2 * (4 * z) ** Rational(1, 4) * exp_polar(I * pi / 4)
    addb([], [a, a + S.Half, 2 * a], (2 * sqrt(polar_lift(-1) * z)) ** (1 - 2 * a) * gamma(2 * a) ** 2 * Matrix([besselj(2 * a - 1, x) * besseli(2 * a - 1, x), x * (besseli(2 * a, x) * besselj(2 * a - 1, x) - besseli(2 * a - 1, x) * besselj(2 * a, x)), x ** 2 * besseli(2 * a, x) * besselj(2 * a, x), x ** 3 * (besseli(2 * a, x) * besselj(2 * a - 1, x) + besseli(2 * a - 1, x) * besselj(2 * a, x))]), Matrix([[1, 0, 0, 0]]), Matrix([[0, Rational(1, 4), 0, 0], [0, (1 - 2 * a) / 2, Rational(-1, 2), 0], [0, 0, 1 - 2 * a, Rational(1, 4)], [-32 * z, 0, 0, 1 - a]]))
    addb([a], [a - S.Half, 2 * a], Matrix([z ** (S.Half - a) * besseli(a - S.Half, sqrt(z)) ** 2, z ** (1 - a) * besseli(a - S.Half, sqrt(z)) * besseli(a - Rational(3, 2), sqrt(z)), z ** (Rational(3, 2) - a) * besseli(a - Rational(3, 2), sqrt(z)) ** 2]), Matrix([[-gamma(a + S.Half) ** 2 / 4 ** (S.Half - a), 2 * gamma(a - S.Half) * gamma(a + S.Half) / 4 ** (1 - a), 0]]), Matrix([[1 - 2 * a, 1, 0], [z / 2, S.Half - a, S.Half], [0, z, 0]]))
    addb([S.Half], [b, 2 - b], pi * (1 - b) / sin(pi * b) * Matrix([besseli(1 - b, sqrt(z)) * besseli(b - 1, sqrt(z)), sqrt(z) * (besseli(-b, sqrt(z)) * besseli(b - 1, sqrt(z)) + besseli(1 - b, sqrt(z)) * besseli(b, sqrt(z))), besseli(-b, sqrt(z)) * besseli(b, sqrt(z))]), Matrix([[1, 0, 0]]), Matrix([[b - 1, S.Half, 0], [z, 0, z], [0, S.Half, -b]]))
    addb([S.Half], [Rational(3, 2), Rational(3, 2)], Matrix([Shi(2 * sqrt(z)) / 2 / sqrt(z), sinh(2 * sqrt(z)) / 2 / sqrt(z), cosh(2 * sqrt(z))]), Matrix([[1, 0, 0]]), Matrix([[Rational(-1, 2), S.Half, 0], [0, Rational(-1, 2), S.Half], [0, 2 * z, 0]]))
    addb([Rational(3, 4)], [Rational(3, 2), Rational(7, 4)], Matrix([fresnels(exp(pi * I / 4) * root(z, 4) * 2 / sqrt(pi)) / (pi * (exp(pi * I / 4) * root(z, 4) * 2 / sqrt(pi)) ** 3), sinh(2 * sqrt(z)) / sqrt(z), cosh(2 * sqrt(z))]), Matrix([[6, 0, 0]]), Matrix([[Rational(-3, 4), Rational(1, 16), 0], [0, Rational(-1, 2), 1], [0, z, 0]]))
    addb([Rational(1, 4)], [S.Half, Rational(5, 4)], Matrix([sqrt(pi) * exp(-I * pi / 4) * fresnelc(2 * root(z, 4) * exp(I * pi / 4) / sqrt(pi)) / (2 * root(z, 4)), cosh(2 * sqrt(z)), sinh(2 * sqrt(z)) * sqrt(z)]), Matrix([[1, 0, 0]]), Matrix([[Rational(-1, 4), Rational(1, 4), 0], [0, 0, 1], [0, z, S.Half]]))
    addb([a, a + S.Half], [2 * a, b, 2 * a - b + 1], gamma(b) * gamma(2 * a - b + 1) * (sqrt(z) / 2) ** (1 - 2 * a) * Matrix([besseli(b - 1, sqrt(z)) * besseli(2 * a - b, sqrt(z)), sqrt(z) * besseli(b, sqrt(z)) * besseli(2 * a - b, sqrt(z)), sqrt(z) * besseli(b - 1, sqrt(z)) * besseli(2 * a - b + 1, sqrt(z)), besseli(b, sqrt(z)) * besseli(2 * a - b + 1, sqrt(z))]), Matrix([[1, 0, 0, 0]]), Matrix([[0, S.Half, S.Half, 0], [z / 2, 1 - b, 0, z / 2], [z / 2, 0, b - 2 * a, z / 2], [0, S.Half, S.Half, -2 * a]]))
    addb([1, 1], [2, 2, Rational(3, 2)], Matrix([Chi(2 * sqrt(z)) - log(2 * sqrt(z)), cosh(2 * sqrt(z)), sqrt(z) * sinh(2 * sqrt(z)), 1, EulerGamma]), Matrix([[1 / z, 0, 0, 0, -1 / z]]), Matrix([[0, S.Half, 0, Rational(-1, 2), 0], [0, 0, 1, 0, 0], [0, z, S.Half, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))
    addb([1, 1, a], [2, 2, a + 1], Matrix([a * (log(-z) + expint(1, -z) + EulerGamma) / (z * (a ** 2 - 2 * a + 1)), a * (-z) ** (-a) * (gamma(a) - uppergamma(a, -z)) / (a - 1) ** 2, a * exp(z) / (a ** 2 - 2 * a + 1), a / (z * (a ** 2 - 2 * a + 1))]), Matrix([[1 - a, 1, -1 / z, 1]]), Matrix([[-1, 0, -1 / z, 1], [0, -a, 1, 0], [0, 0, z, 0], [0, 0, 0, -1]]))

def add_meijerg_formulae(formulae):
    if False:
        for i in range(10):
            print('nop')
    (a, b, c, z) = list(map(Dummy, 'abcz'))
    rho = Dummy('rho')

    def add(an, ap, bm, bq, B, C, M, matcher):
        if False:
            return 10
        formulae.append(MeijerFormula(an, ap, bm, bq, z, [a, b, c, rho], B, C, M, matcher))

    def detect_uppergamma(func):
        if False:
            return 10
        x = func.an[0]
        (y, z) = func.bm
        swapped = False
        if not _mod1((x - y).simplify()):
            swapped = True
            (y, z) = (z, y)
        if _mod1((x - z).simplify()) or x - z > 0:
            return None
        l = [y, x]
        if swapped:
            l = [x, y]
        return ({rho: y, a: x - y}, G_Function([x], [], l, []))
    add([a + rho], [], [rho, a + rho], [], Matrix([gamma(1 - a) * z ** rho * exp(z) * uppergamma(a, z), gamma(1 - a) * z ** (a + rho)]), Matrix([[1, 0]]), Matrix([[rho + z, -1], [0, a + rho]]), detect_uppergamma)

    def detect_3113(func):
        if False:
            i = 10
            return i + 15
        'https://functions.wolfram.com/07.34.03.0984.01'
        x = func.an[0]
        (u, v, w) = func.bm
        if _mod1((u - v).simplify()) == 0:
            if _mod1((v - w).simplify()) == 0:
                return
            sig = (S.Half, S.Half, S.Zero)
            (x1, x2, y) = (u, v, w)
        elif _mod1((x - u).simplify()) == 0:
            sig = (S.Half, S.Zero, S.Half)
            (x1, y, x2) = (u, v, w)
        else:
            sig = (S.Zero, S.Half, S.Half)
            (y, x1, x2) = (u, v, w)
        if _mod1((x - x1).simplify()) != 0 or _mod1((x - x2).simplify()) != 0 or _mod1((x - y).simplify()) != S.Half or (x - x1 > 0) or (x - x2 > 0):
            return
        return ({a: x}, G_Function([x], [], [x - S.Half + t for t in sig], []))
    s = sin(2 * sqrt(z))
    c_ = cos(2 * sqrt(z))
    S_ = Si(2 * sqrt(z)) - pi / 2
    C = Ci(2 * sqrt(z))
    add([a], [], [a, a, a - S.Half], [], Matrix([sqrt(pi) * z ** (a - S.Half) * (c_ * S_ - s * C), sqrt(pi) * z ** a * (s * S_ + c_ * C), sqrt(pi) * z ** a]), Matrix([[-2, 0, 0]]), Matrix([[a - S.Half, -1, 0], [z, a, S.Half], [0, 0, a]]), detect_3113)

def make_simp(z):
    if False:
        print('Hello World!')
    ' Create a function that simplifies rational functions in ``z``. '

    def simp(expr):
        if False:
            return 10
        ' Efficiently simplify the rational function ``expr``. '
        (numer, denom) = expr.as_numer_denom()
        numer = numer.expand()
        (c, numer, denom) = poly(numer, z).cancel(poly(denom, z))
        return c * numer.as_expr() / denom.as_expr()
    return simp

def debug(*args):
    if False:
        while True:
            i = 10
    if SYMPY_DEBUG:
        for a in args:
            print(a, end='')
        print()

class Hyper_Function(Expr):
    """ A generalized hypergeometric function. """

    def __new__(cls, ap, bq):
        if False:
            while True:
                i = 10
        obj = super().__new__(cls)
        obj.ap = Tuple(*list(map(expand, ap)))
        obj.bq = Tuple(*list(map(expand, bq)))
        return obj

    @property
    def args(self):
        if False:
            while True:
                i = 10
        return (self.ap, self.bq)

    @property
    def sizes(self):
        if False:
            for i in range(10):
                print('nop')
        return (len(self.ap), len(self.bq))

    @property
    def gamma(self):
        if False:
            i = 10
            return i + 15
        '\n        Number of upper parameters that are negative integers\n\n        This is a transformation invariant.\n        '
        return sum((bool(x.is_integer and x.is_negative) for x in self.ap))

    def _hashable_content(self):
        if False:
            i = 10
            return i + 15
        return super()._hashable_content() + (self.ap, self.bq)

    def __call__(self, arg):
        if False:
            i = 10
            return i + 15
        return hyper(self.ap, self.bq, arg)

    def build_invariants(self):
        if False:
            i = 10
            return i + 15
        '\n        Compute the invariant vector.\n\n        Explanation\n        ===========\n\n        The invariant vector is:\n            (gamma, ((s1, n1), ..., (sk, nk)), ((t1, m1), ..., (tr, mr)))\n        where gamma is the number of integer a < 0,\n              s1 < ... < sk\n              nl is the number of parameters a_i congruent to sl mod 1\n              t1 < ... < tr\n              ml is the number of parameters b_i congruent to tl mod 1\n\n        If the index pair contains parameters, then this is not truly an\n        invariant, since the parameters cannot be sorted uniquely mod1.\n\n        Examples\n        ========\n\n        >>> from sympy.simplify.hyperexpand import Hyper_Function\n        >>> from sympy import S\n        >>> ap = (S.Half, S.One/3, S(-1)/2, -2)\n        >>> bq = (1, 2)\n\n        Here gamma = 1,\n             k = 3, s1 = 0, s2 = 1/3, s3 = 1/2\n                    n1 = 1, n2 = 1,   n2 = 2\n             r = 1, t1 = 0\n                    m1 = 2:\n\n        >>> Hyper_Function(ap, bq).build_invariants()\n        (1, ((0, 1), (1/3, 1), (1/2, 2)), ((0, 2),))\n        '
        (abuckets, bbuckets) = (sift(self.ap, _mod1), sift(self.bq, _mod1))

        def tr(bucket):
            if False:
                print('Hello World!')
            bucket = list(bucket.items())
            if not any((isinstance(x[0], Mod) for x in bucket)):
                bucket.sort(key=lambda x: default_sort_key(x[0]))
            bucket = tuple([(mod, len(values)) for (mod, values) in bucket if values])
            return bucket
        return (self.gamma, tr(abuckets), tr(bbuckets))

    def difficulty(self, func):
        if False:
            print('Hello World!')
        ' Estimate how many steps it takes to reach ``func`` from self.\n            Return -1 if impossible. '
        if self.gamma != func.gamma:
            return -1
        (oabuckets, obbuckets, abuckets, bbuckets) = [sift(params, _mod1) for params in (self.ap, self.bq, func.ap, func.bq)]
        diff = 0
        for (bucket, obucket) in [(abuckets, oabuckets), (bbuckets, obbuckets)]:
            for mod in set(list(bucket.keys()) + list(obucket.keys())):
                if mod not in bucket or mod not in obucket or len(bucket[mod]) != len(obucket[mod]):
                    return -1
                l1 = list(bucket[mod])
                l2 = list(obucket[mod])
                l1.sort()
                l2.sort()
                for (i, j) in zip(l1, l2):
                    diff += abs(i - j)
        return diff

    def _is_suitable_origin(self):
        if False:
            i = 10
            return i + 15
        '\n        Decide if ``self`` is a suitable origin.\n\n        Explanation\n        ===========\n\n        A function is a suitable origin iff:\n        * none of the ai equals bj + n, with n a non-negative integer\n        * none of the ai is zero\n        * none of the bj is a non-positive integer\n\n        Note that this gives meaningful results only when none of the indices\n        are symbolic.\n\n        '
        for a in self.ap:
            for b in self.bq:
                if (a - b).is_integer and (a - b).is_negative is False:
                    return False
        for a in self.ap:
            if a == 0:
                return False
        for b in self.bq:
            if b.is_integer and b.is_nonpositive:
                return False
        return True

class G_Function(Expr):
    """ A Meijer G-function. """

    def __new__(cls, an, ap, bm, bq):
        if False:
            return 10
        obj = super().__new__(cls)
        obj.an = Tuple(*list(map(expand, an)))
        obj.ap = Tuple(*list(map(expand, ap)))
        obj.bm = Tuple(*list(map(expand, bm)))
        obj.bq = Tuple(*list(map(expand, bq)))
        return obj

    @property
    def args(self):
        if False:
            return 10
        return (self.an, self.ap, self.bm, self.bq)

    def _hashable_content(self):
        if False:
            i = 10
            return i + 15
        return super()._hashable_content() + self.args

    def __call__(self, z):
        if False:
            for i in range(10):
                print('nop')
        return meijerg(self.an, self.ap, self.bm, self.bq, z)

    def compute_buckets(self):
        if False:
            while True:
                i = 10
        '\n        Compute buckets for the fours sets of parameters.\n\n        Explanation\n        ===========\n\n        We guarantee that any two equal Mod objects returned are actually the\n        same, and that the buckets are sorted by real part (an and bq\n        descendending, bm and ap ascending).\n\n        Examples\n        ========\n\n        >>> from sympy.simplify.hyperexpand import G_Function\n        >>> from sympy.abc import y\n        >>> from sympy import S\n\n        >>> a, b = [1, 3, 2, S(3)/2], [1 + y, y, 2, y + 3]\n        >>> G_Function(a, b, [2], [y]).compute_buckets()\n        ({0: [3, 2, 1], 1/2: [3/2]},\n        {0: [2], y: [y, y + 1, y + 3]}, {0: [2]}, {y: [y]})\n\n        '
        dicts = (pan, pap, pbm, pbq) = [defaultdict(list) for i in range(4)]
        for (dic, lis) in zip(dicts, (self.an, self.ap, self.bm, self.bq)):
            for x in lis:
                dic[_mod1(x)].append(x)
        for (dic, flip) in zip(dicts, (True, False, False, True)):
            for (m, items) in dic.items():
                x0 = items[0]
                items.sort(key=lambda x: x - x0, reverse=flip)
                dic[m] = items
        return tuple([dict(w) for w in dicts])

    @property
    def signature(self):
        if False:
            i = 10
            return i + 15
        return (len(self.an), len(self.ap), len(self.bm), len(self.bq))
_x = Dummy('x')

class Formula:
    """
    This class represents hypergeometric formulae.

    Explanation
    ===========

    Its data members are:
    - z, the argument
    - closed_form, the closed form expression
    - symbols, the free symbols (parameters) in the formula
    - func, the function
    - B, C, M (see _compute_basis)

    Examples
    ========

    >>> from sympy.abc import a, b, z
    >>> from sympy.simplify.hyperexpand import Formula, Hyper_Function
    >>> func = Hyper_Function((a/2, a/3 + b, (1+a)/2), (a, b, (a+b)/7))
    >>> f = Formula(func, z, None, [a, b])

    """

    def _compute_basis(self, closed_form):
        if False:
            print('Hello World!')
        '\n        Compute a set of functions B=(f1, ..., fn), a nxn matrix M\n        and a 1xn matrix C such that:\n           closed_form = C B\n           z d/dz B = M B.\n        '
        afactors = [_x + a for a in self.func.ap]
        bfactors = [_x + b - 1 for b in self.func.bq]
        expr = _x * Mul(*bfactors) - self.z * Mul(*afactors)
        poly = Poly(expr, _x)
        n = poly.degree() - 1
        b = [closed_form]
        for _ in range(n):
            b.append(self.z * b[-1].diff(self.z))
        self.B = Matrix(b)
        self.C = Matrix([[1] + [0] * n])
        m = eye(n)
        m = m.col_insert(0, zeros(n, 1))
        l = poly.all_coeffs()[1:]
        l.reverse()
        self.M = m.row_insert(n, -Matrix([l]) / poly.all_coeffs()[0])

    def __init__(self, func, z, res, symbols, B=None, C=None, M=None):
        if False:
            while True:
                i = 10
        z = sympify(z)
        res = sympify(res)
        symbols = [x for x in sympify(symbols) if func.has(x)]
        self.z = z
        self.symbols = symbols
        self.B = B
        self.C = C
        self.M = M
        self.func = func
        if res is not None:
            self._compute_basis(res)

    @property
    def closed_form(self):
        if False:
            while True:
                i = 10
        return reduce(lambda s, m: s + m[0] * m[1], zip(self.C, self.B), S.Zero)

    def find_instantiations(self, func):
        if False:
            print('Hello World!')
        '\n        Find substitutions of the free symbols that match ``func``.\n\n        Return the substitution dictionaries as a list. Note that the returned\n        instantiations need not actually match, or be valid!\n\n        '
        from sympy.solvers import solve
        ap = func.ap
        bq = func.bq
        if len(ap) != len(self.func.ap) or len(bq) != len(self.func.bq):
            raise TypeError('Cannot instantiate other number of parameters')
        symbol_values = []
        for a in self.symbols:
            if a in self.func.ap.args:
                symbol_values.append(ap)
            elif a in self.func.bq.args:
                symbol_values.append(bq)
            else:
                raise ValueError('At least one of the parameters of the formula must be equal to %s' % (a,))
        base_repl = [dict(list(zip(self.symbols, values))) for values in product(*symbol_values)]
        (abuckets, bbuckets) = [sift(params, _mod1) for params in [ap, bq]]
        (a_inv, b_inv) = [{a: len(vals) for (a, vals) in bucket.items()} for bucket in [abuckets, bbuckets]]
        critical_values = [[0] for _ in self.symbols]
        result = []
        _n = Dummy()
        for repl in base_repl:
            (symb_a, symb_b) = [sift(params, lambda x: _mod1(x.xreplace(repl))) for params in [self.func.ap, self.func.bq]]
            for (bucket, obucket) in [(abuckets, symb_a), (bbuckets, symb_b)]:
                for mod in set(list(bucket.keys()) + list(obucket.keys())):
                    if mod not in bucket or mod not in obucket or len(bucket[mod]) != len(obucket[mod]):
                        break
                    for (a, vals) in zip(self.symbols, critical_values):
                        if repl[a].free_symbols:
                            continue
                        exprs = [expr for expr in obucket[mod] if expr.has(a)]
                        repl0 = repl.copy()
                        repl0[a] += _n
                        for expr in exprs:
                            for target in bucket[mod]:
                                (n0,) = solve(expr.xreplace(repl0) - target, _n)
                                if n0.free_symbols:
                                    raise ValueError('Value should not be true')
                                vals.append(n0)
            else:
                values = []
                for (a, vals) in zip(self.symbols, critical_values):
                    a0 = repl[a]
                    min_ = floor(min(vals))
                    max_ = ceiling(max(vals))
                    values.append([a0 + n for n in range(min_, max_ + 1)])
                result.extend((dict(list(zip(self.symbols, l))) for l in product(*values)))
        return result

class FormulaCollection:
    """ A collection of formulae to use as origins. """

    def __init__(self):
        if False:
            print('Hello World!')
        ' Doing this globally at module init time is a pain ... '
        self.symbolic_formulae = {}
        self.concrete_formulae = {}
        self.formulae = []
        add_formulae(self.formulae)
        for f in self.formulae:
            sizes = f.func.sizes
            if len(f.symbols) > 0:
                self.symbolic_formulae.setdefault(sizes, []).append(f)
            else:
                inv = f.func.build_invariants()
                self.concrete_formulae.setdefault(sizes, {})[inv] = f

    def lookup_origin(self, func):
        if False:
            return 10
        "\n        Given the suitable target ``func``, try to find an origin in our\n        knowledge base.\n\n        Examples\n        ========\n\n        >>> from sympy.simplify.hyperexpand import (FormulaCollection,\n        ...     Hyper_Function)\n        >>> f = FormulaCollection()\n        >>> f.lookup_origin(Hyper_Function((), ())).closed_form\n        exp(_z)\n        >>> f.lookup_origin(Hyper_Function([1], ())).closed_form\n        HyperRep_power1(-1, _z)\n\n        >>> from sympy import S\n        >>> i = Hyper_Function([S('1/4'), S('3/4 + 4')], [S.Half])\n        >>> f.lookup_origin(i).closed_form\n        HyperRep_sqrts1(-1/4, _z)\n        "
        inv = func.build_invariants()
        sizes = func.sizes
        if sizes in self.concrete_formulae and inv in self.concrete_formulae[sizes]:
            return self.concrete_formulae[sizes][inv]
        if sizes not in self.symbolic_formulae:
            return None
        possible = []
        for f in self.symbolic_formulae[sizes]:
            repls = f.find_instantiations(func)
            for repl in repls:
                func2 = f.func.xreplace(repl)
                if not func2._is_suitable_origin():
                    continue
                diff = func2.difficulty(func)
                if diff == -1:
                    continue
                possible.append((diff, repl, f, func2))
        possible.sort(key=lambda x: x[0])
        for (_, repl, f, func2) in possible:
            f2 = Formula(func2, f.z, None, [], f.B.subs(repl), f.C.subs(repl), f.M.subs(repl))
            if not any((e.has(S.NaN, oo, -oo, zoo) for e in [f2.B, f2.M, f2.C])):
                return f2
        return None

class MeijerFormula:
    """
    This class represents a Meijer G-function formula.

    Its data members are:
    - z, the argument
    - symbols, the free symbols (parameters) in the formula
    - func, the function
    - B, C, M (c/f ordinary Formula)
    """

    def __init__(self, an, ap, bm, bq, z, symbols, B, C, M, matcher):
        if False:
            while True:
                i = 10
        (an, ap, bm, bq) = [Tuple(*list(map(expand, w))) for w in [an, ap, bm, bq]]
        self.func = G_Function(an, ap, bm, bq)
        self.z = z
        self.symbols = symbols
        self._matcher = matcher
        self.B = B
        self.C = C
        self.M = M

    @property
    def closed_form(self):
        if False:
            return 10
        return reduce(lambda s, m: s + m[0] * m[1], zip(self.C, self.B), S.Zero)

    def try_instantiate(self, func):
        if False:
            return 10
        '\n        Try to instantiate the current formula to (almost) match func.\n        This uses the _matcher passed on init.\n        '
        if func.signature != self.func.signature:
            return None
        res = self._matcher(func)
        if res is not None:
            (subs, newfunc) = res
            return MeijerFormula(newfunc.an, newfunc.ap, newfunc.bm, newfunc.bq, self.z, [], self.B.subs(subs), self.C.subs(subs), self.M.subs(subs), None)

class MeijerFormulaCollection:
    """
    This class holds a collection of meijer g formulae.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        formulae = []
        add_meijerg_formulae(formulae)
        self.formulae = defaultdict(list)
        for formula in formulae:
            self.formulae[formula.func.signature].append(formula)
        self.formulae = dict(self.formulae)

    def lookup_origin(self, func):
        if False:
            for i in range(10):
                print('nop')
        ' Try to find a formula that matches func. '
        if func.signature not in self.formulae:
            return None
        for formula in self.formulae[func.signature]:
            res = formula.try_instantiate(func)
            if res is not None:
                return res

class Operator:
    """
    Base class for operators to be applied to our functions.

    Explanation
    ===========

    These operators are differential operators. They are by convention
    expressed in the variable D = z*d/dz (although this base class does
    not actually care).
    Note that when the operator is applied to an object, we typically do
    *not* blindly differentiate but instead use a different representation
    of the z*d/dz operator (see make_derivative_operator).

    To subclass from this, define a __init__ method that initializes a
    self._poly variable. This variable stores a polynomial. By convention
    the generator is z*d/dz, and acts to the right of all coefficients.

    Thus this poly
        x**2 + 2*z*x + 1
    represents the differential operator
        (z*d/dz)**2 + 2*z**2*d/dz.

    This class is used only in the implementation of the hypergeometric
    function expansion algorithm.
    """

    def apply(self, obj, op):
        if False:
            return 10
        '\n        Apply ``self`` to the object ``obj``, where the generator is ``op``.\n\n        Examples\n        ========\n\n        >>> from sympy.simplify.hyperexpand import Operator\n        >>> from sympy.polys.polytools import Poly\n        >>> from sympy.abc import x, y, z\n        >>> op = Operator()\n        >>> op._poly = Poly(x**2 + z*x + y, x)\n        >>> op.apply(z**7, lambda f: f.diff(z))\n        y*z**7 + 7*z**7 + 42*z**5\n        '
        coeffs = self._poly.all_coeffs()
        coeffs.reverse()
        diffs = [obj]
        for c in coeffs[1:]:
            diffs.append(op(diffs[-1]))
        r = coeffs[0] * diffs[0]
        for (c, d) in zip(coeffs[1:], diffs[1:]):
            r += c * d
        return r

class MultOperator(Operator):
    """ Simply multiply by a "constant" """

    def __init__(self, p):
        if False:
            for i in range(10):
                print('nop')
        self._poly = Poly(p, _x)

class ShiftA(Operator):
    """ Increment an upper index. """

    def __init__(self, ai):
        if False:
            while True:
                i = 10
        ai = sympify(ai)
        if ai == 0:
            raise ValueError('Cannot increment zero upper index.')
        self._poly = Poly(_x / ai + 1, _x)

    def __str__(self):
        if False:
            while True:
                i = 10
        return '<Increment upper %s.>' % (1 / self._poly.all_coeffs()[0])

class ShiftB(Operator):
    """ Decrement a lower index. """

    def __init__(self, bi):
        if False:
            i = 10
            return i + 15
        bi = sympify(bi)
        if bi == 1:
            raise ValueError('Cannot decrement unit lower index.')
        self._poly = Poly(_x / (bi - 1) + 1, _x)

    def __str__(self):
        if False:
            while True:
                i = 10
        return '<Decrement lower %s.>' % (1 / self._poly.all_coeffs()[0] + 1)

class UnShiftA(Operator):
    """ Decrement an upper index. """

    def __init__(self, ap, bq, i, z):
        if False:
            print('Hello World!')
        ' Note: i counts from zero! '
        (ap, bq, i) = list(map(sympify, [ap, bq, i]))
        self._ap = ap
        self._bq = bq
        self._i = i
        ap = list(ap)
        bq = list(bq)
        ai = ap.pop(i) - 1
        if ai == 0:
            raise ValueError('Cannot decrement unit upper index.')
        m = Poly(z * ai, _x)
        for a in ap:
            m *= Poly(_x + a, _x)
        A = Dummy('A')
        n = D = Poly(ai * A - ai, A)
        for b in bq:
            n *= D + (b - 1).as_poly(A)
        b0 = -n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot decrement upper index: cancels with lower')
        n = Poly(Poly(n.all_coeffs()[:-1], A).as_expr().subs(A, _x / ai + 1), _x)
        self._poly = Poly((n - m) / b0, _x)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '<Decrement upper index #%s of %s, %s.>' % (self._i, self._ap, self._bq)

class UnShiftB(Operator):
    """ Increment a lower index. """

    def __init__(self, ap, bq, i, z):
        if False:
            i = 10
            return i + 15
        ' Note: i counts from zero! '
        (ap, bq, i) = list(map(sympify, [ap, bq, i]))
        self._ap = ap
        self._bq = bq
        self._i = i
        ap = list(ap)
        bq = list(bq)
        bi = bq.pop(i) + 1
        if bi == 0:
            raise ValueError('Cannot increment -1 lower index.')
        m = Poly(_x * (bi - 1), _x)
        for b in bq:
            m *= Poly(_x + b - 1, _x)
        B = Dummy('B')
        D = Poly((bi - 1) * B - bi + 1, B)
        n = Poly(z, B)
        for a in ap:
            n *= D + a.as_poly(B)
        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot increment index: cancels with upper')
        n = Poly(Poly(n.all_coeffs()[:-1], B).as_expr().subs(B, _x / (bi - 1) + 1), _x)
        self._poly = Poly((m - n) / b0, _x)

    def __str__(self):
        if False:
            while True:
                i = 10
        return '<Increment lower index #%s of %s, %s.>' % (self._i, self._ap, self._bq)

class MeijerShiftA(Operator):
    """ Increment an upper b index. """

    def __init__(self, bi):
        if False:
            for i in range(10):
                print('nop')
        bi = sympify(bi)
        self._poly = Poly(bi - _x, _x)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<Increment upper b=%s.>' % self._poly.all_coeffs()[1]

class MeijerShiftB(Operator):
    """ Decrement an upper a index. """

    def __init__(self, bi):
        if False:
            i = 10
            return i + 15
        bi = sympify(bi)
        self._poly = Poly(1 - bi + _x, _x)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '<Decrement upper a=%s.>' % (1 - self._poly.all_coeffs()[1])

class MeijerShiftC(Operator):
    """ Increment a lower b index. """

    def __init__(self, bi):
        if False:
            print('Hello World!')
        bi = sympify(bi)
        self._poly = Poly(-bi + _x, _x)

    def __str__(self):
        if False:
            return 10
        return '<Increment lower b=%s.>' % -self._poly.all_coeffs()[1]

class MeijerShiftD(Operator):
    """ Decrement a lower a index. """

    def __init__(self, bi):
        if False:
            return 10
        bi = sympify(bi)
        self._poly = Poly(bi - 1 - _x, _x)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<Decrement lower a=%s.>' % (self._poly.all_coeffs()[1] + 1)

class MeijerUnShiftA(Operator):
    """ Decrement an upper b index. """

    def __init__(self, an, ap, bm, bq, i, z):
        if False:
            for i in range(10):
                print('nop')
        ' Note: i counts from zero! '
        (an, ap, bm, bq, i) = list(map(sympify, [an, ap, bm, bq, i]))
        self._an = an
        self._ap = ap
        self._bm = bm
        self._bq = bq
        self._i = i
        an = list(an)
        ap = list(ap)
        bm = list(bm)
        bq = list(bq)
        bi = bm.pop(i) - 1
        m = Poly(1, _x) * prod((Poly(b - _x, _x) for b in bm)) * prod((Poly(_x - b, _x) for b in bq))
        A = Dummy('A')
        D = Poly(bi - A, A)
        n = Poly(z, A) * prod((D + 1 - a for a in an)) * prod((-D + a - 1 for a in ap))
        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot decrement upper b index (cancels)')
        n = Poly(Poly(n.all_coeffs()[:-1], A).as_expr().subs(A, bi - _x), _x)
        self._poly = Poly((m - n) / b0, _x)

    def __str__(self):
        if False:
            print('Hello World!')
        return '<Decrement upper b index #%s of %s, %s, %s, %s.>' % (self._i, self._an, self._ap, self._bm, self._bq)

class MeijerUnShiftB(Operator):
    """ Increment an upper a index. """

    def __init__(self, an, ap, bm, bq, i, z):
        if False:
            i = 10
            return i + 15
        ' Note: i counts from zero! '
        (an, ap, bm, bq, i) = list(map(sympify, [an, ap, bm, bq, i]))
        self._an = an
        self._ap = ap
        self._bm = bm
        self._bq = bq
        self._i = i
        an = list(an)
        ap = list(ap)
        bm = list(bm)
        bq = list(bq)
        ai = an.pop(i) + 1
        m = Poly(z, _x)
        for a in an:
            m *= Poly(1 - a + _x, _x)
        for a in ap:
            m *= Poly(a - 1 - _x, _x)
        B = Dummy('B')
        D = Poly(B + ai - 1, B)
        n = Poly(1, B)
        for b in bm:
            n *= -D + b
        for b in bq:
            n *= D - b
        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot increment upper a index (cancels)')
        n = Poly(Poly(n.all_coeffs()[:-1], B).as_expr().subs(B, 1 - ai + _x), _x)
        self._poly = Poly((m - n) / b0, _x)

    def __str__(self):
        if False:
            while True:
                i = 10
        return '<Increment upper a index #%s of %s, %s, %s, %s.>' % (self._i, self._an, self._ap, self._bm, self._bq)

class MeijerUnShiftC(Operator):
    """ Decrement a lower b index. """

    def __init__(self, an, ap, bm, bq, i, z):
        if False:
            while True:
                i = 10
        ' Note: i counts from zero! '
        (an, ap, bm, bq, i) = list(map(sympify, [an, ap, bm, bq, i]))
        self._an = an
        self._ap = ap
        self._bm = bm
        self._bq = bq
        self._i = i
        an = list(an)
        ap = list(ap)
        bm = list(bm)
        bq = list(bq)
        bi = bq.pop(i) - 1
        m = Poly(1, _x)
        for b in bm:
            m *= Poly(b - _x, _x)
        for b in bq:
            m *= Poly(_x - b, _x)
        C = Dummy('C')
        D = Poly(bi + C, C)
        n = Poly(z, C)
        for a in an:
            n *= D + 1 - a
        for a in ap:
            n *= -D + a - 1
        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot decrement lower b index (cancels)')
        n = Poly(Poly(n.all_coeffs()[:-1], C).as_expr().subs(C, _x - bi), _x)
        self._poly = Poly((m - n) / b0, _x)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '<Decrement lower b index #%s of %s, %s, %s, %s.>' % (self._i, self._an, self._ap, self._bm, self._bq)

class MeijerUnShiftD(Operator):
    """ Increment a lower a index. """

    def __init__(self, an, ap, bm, bq, i, z):
        if False:
            print('Hello World!')
        ' Note: i counts from zero! '
        (an, ap, bm, bq, i) = list(map(sympify, [an, ap, bm, bq, i]))
        self._an = an
        self._ap = ap
        self._bm = bm
        self._bq = bq
        self._i = i
        an = list(an)
        ap = list(ap)
        bm = list(bm)
        bq = list(bq)
        ai = ap.pop(i) + 1
        m = Poly(z, _x)
        for a in an:
            m *= Poly(1 - a + _x, _x)
        for a in ap:
            m *= Poly(a - 1 - _x, _x)
        B = Dummy('B')
        D = Poly(ai - 1 - B, B)
        n = Poly(1, B)
        for b in bm:
            n *= -D + b
        for b in bq:
            n *= D - b
        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot increment lower a index (cancels)')
        n = Poly(Poly(n.all_coeffs()[:-1], B).as_expr().subs(B, ai - 1 - _x), _x)
        self._poly = Poly((m - n) / b0, _x)

    def __str__(self):
        if False:
            while True:
                i = 10
        return '<Increment lower a index #%s of %s, %s, %s, %s.>' % (self._i, self._an, self._ap, self._bm, self._bq)

class ReduceOrder(Operator):
    """ Reduce Order by cancelling an upper and a lower index. """

    def __new__(cls, ai, bj):
        if False:
            while True:
                i = 10
        ' For convenience if reduction is not possible, return None. '
        ai = sympify(ai)
        bj = sympify(bj)
        n = ai - bj
        if not n.is_Integer or n < 0:
            return None
        if bj.is_integer and bj.is_nonpositive:
            return None
        expr = Operator.__new__(cls)
        p = S.One
        for k in range(n):
            p *= (_x + bj + k) / (bj + k)
        expr._poly = Poly(p, _x)
        expr._a = ai
        expr._b = bj
        return expr

    @classmethod
    def _meijer(cls, b, a, sign):
        if False:
            for i in range(10):
                print('nop')
        ' Cancel b + sign*s and a + sign*s\n            This is for meijer G functions. '
        b = sympify(b)
        a = sympify(a)
        n = b - a
        if n.is_negative or not n.is_Integer:
            return None
        expr = Operator.__new__(cls)
        p = S.One
        for k in range(n):
            p *= sign * _x + a + k
        expr._poly = Poly(p, _x)
        if sign == -1:
            expr._a = b
            expr._b = a
        else:
            expr._b = Add(1, a - 1, evaluate=False)
            expr._a = Add(1, b - 1, evaluate=False)
        return expr

    @classmethod
    def meijer_minus(cls, b, a):
        if False:
            i = 10
            return i + 15
        return cls._meijer(b, a, -1)

    @classmethod
    def meijer_plus(cls, a, b):
        if False:
            print('Hello World!')
        return cls._meijer(1 - a, 1 - b, 1)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '<Reduce order by cancelling upper %s with lower %s.>' % (self._a, self._b)

def _reduce_order(ap, bq, gen, key):
    if False:
        return 10
    ' Order reduction algorithm used in Hypergeometric and Meijer G '
    ap = list(ap)
    bq = list(bq)
    ap.sort(key=key)
    bq.sort(key=key)
    nap = []
    operators = []
    for a in ap:
        op = None
        for i in range(len(bq)):
            op = gen(a, bq[i])
            if op is not None:
                bq.pop(i)
                break
        if op is None:
            nap.append(a)
        else:
            operators.append(op)
    return (nap, bq, operators)

def reduce_order(func):
    if False:
        i = 10
        return i + 15
    '\n    Given the hypergeometric function ``func``, find a sequence of operators to\n    reduces order as much as possible.\n\n    Explanation\n    ===========\n\n    Return (newfunc, [operators]), where applying the operators to the\n    hypergeometric function newfunc yields func.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.hyperexpand import reduce_order, Hyper_Function\n    >>> reduce_order(Hyper_Function((1, 2), (3, 4)))\n    (Hyper_Function((1, 2), (3, 4)), [])\n    >>> reduce_order(Hyper_Function((1,), (1,)))\n    (Hyper_Function((), ()), [<Reduce order by cancelling upper 1 with lower 1.>])\n    >>> reduce_order(Hyper_Function((2, 4), (3, 3)))\n    (Hyper_Function((2,), (3,)), [<Reduce order by cancelling\n    upper 4 with lower 3.>])\n    '
    (nap, nbq, operators) = _reduce_order(func.ap, func.bq, ReduceOrder, default_sort_key)
    return (Hyper_Function(Tuple(*nap), Tuple(*nbq)), operators)

def reduce_order_meijer(func):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given the Meijer G function parameters, ``func``, find a sequence of\n    operators that reduces order as much as possible.\n\n    Return newfunc, [operators].\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.hyperexpand import (reduce_order_meijer,\n    ...                                         G_Function)\n    >>> reduce_order_meijer(G_Function([3, 4], [5, 6], [3, 4], [1, 2]))[0]\n    G_Function((4, 3), (5, 6), (3, 4), (2, 1))\n    >>> reduce_order_meijer(G_Function([3, 4], [5, 6], [3, 4], [1, 8]))[0]\n    G_Function((3,), (5, 6), (3, 4), (1,))\n    >>> reduce_order_meijer(G_Function([3, 4], [5, 6], [7, 5], [1, 5]))[0]\n    G_Function((3,), (), (), (1,))\n    >>> reduce_order_meijer(G_Function([3, 4], [5, 6], [7, 5], [5, 3]))[0]\n    G_Function((), (), (), ())\n    '
    (nan, nbq, ops1) = _reduce_order(func.an, func.bq, ReduceOrder.meijer_plus, lambda x: default_sort_key(-x))
    (nbm, nap, ops2) = _reduce_order(func.bm, func.ap, ReduceOrder.meijer_minus, default_sort_key)
    return (G_Function(nan, nap, nbm, nbq), ops1 + ops2)

def make_derivative_operator(M, z):
    if False:
        i = 10
        return i + 15
    ' Create a derivative operator, to be passed to Operator.apply. '

    def doit(C):
        if False:
            print('Hello World!')
        r = z * C.diff(z) + C * M
        r = r.applyfunc(make_simp(z))
        return r
    return doit

def apply_operators(obj, ops, op):
    if False:
        i = 10
        return i + 15
    '\n    Apply the list of operators ``ops`` to object ``obj``, substituting\n    ``op`` for the generator.\n    '
    res = obj
    for o in reversed(ops):
        res = o.apply(res, op)
    return res

def devise_plan(target, origin, z):
    if False:
        return 10
    "\n    Devise a plan (consisting of shift and un-shift operators) to be applied\n    to the hypergeometric function ``target`` to yield ``origin``.\n    Returns a list of operators.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.hyperexpand import devise_plan, Hyper_Function\n    >>> from sympy.abc import z\n\n    Nothing to do:\n\n    >>> devise_plan(Hyper_Function((1, 2), ()), Hyper_Function((1, 2), ()), z)\n    []\n    >>> devise_plan(Hyper_Function((), (1, 2)), Hyper_Function((), (1, 2)), z)\n    []\n\n    Very simple plans:\n\n    >>> devise_plan(Hyper_Function((2,), ()), Hyper_Function((1,), ()), z)\n    [<Increment upper 1.>]\n    >>> devise_plan(Hyper_Function((), (2,)), Hyper_Function((), (1,)), z)\n    [<Increment lower index #0 of [], [1].>]\n\n    Several buckets:\n\n    >>> from sympy import S\n    >>> devise_plan(Hyper_Function((1, S.Half), ()),\n    ...             Hyper_Function((2, S('3/2')), ()), z) #doctest: +NORMALIZE_WHITESPACE\n    [<Decrement upper index #0 of [3/2, 1], [].>,\n    <Decrement upper index #0 of [2, 3/2], [].>]\n\n    A slightly more complicated plan:\n\n    >>> devise_plan(Hyper_Function((1, 3), ()), Hyper_Function((2, 2), ()), z)\n    [<Increment upper 2.>, <Decrement upper index #0 of [2, 2], [].>]\n\n    Another more complicated plan: (note that the ap have to be shifted first!)\n\n    >>> devise_plan(Hyper_Function((1, -1), (2,)), Hyper_Function((3, -2), (4,)), z)\n    [<Decrement lower 3.>, <Decrement lower 4.>,\n    <Decrement upper index #1 of [-1, 2], [4].>,\n    <Decrement upper index #1 of [-1, 3], [4].>, <Increment upper -2.>]\n    "
    (abuckets, bbuckets, nabuckets, nbbuckets) = [sift(params, _mod1) for params in (target.ap, target.bq, origin.ap, origin.bq)]
    if len(list(abuckets.keys())) != len(list(nabuckets.keys())) or len(list(bbuckets.keys())) != len(list(nbbuckets.keys())):
        raise ValueError('%s not reachable from %s' % (target, origin))
    ops = []

    def do_shifts(fro, to, inc, dec):
        if False:
            for i in range(10):
                print('nop')
        ops = []
        for i in range(len(fro)):
            if to[i] - fro[i] > 0:
                sh = inc
                ch = 1
            else:
                sh = dec
                ch = -1
            while to[i] != fro[i]:
                ops += [sh(fro, i)]
                fro[i] += ch
        return ops

    def do_shifts_a(nal, nbk, al, aother, bother):
        if False:
            i = 10
            return i + 15
        ' Shift us from (nal, nbk) to (al, nbk). '
        return do_shifts(nal, al, lambda p, i: ShiftA(p[i]), lambda p, i: UnShiftA(p + aother, nbk + bother, i, z))

    def do_shifts_b(nal, nbk, bk, aother, bother):
        if False:
            return 10
        ' Shift us from (nal, nbk) to (nal, bk). '
        return do_shifts(nbk, bk, lambda p, i: UnShiftB(nal + aother, p + bother, i, z), lambda p, i: ShiftB(p[i]))
    for r in sorted(list(abuckets.keys()) + list(bbuckets.keys()), key=default_sort_key):
        al = ()
        nal = ()
        bk = ()
        nbk = ()
        if r in abuckets:
            al = abuckets[r]
            nal = nabuckets[r]
        if r in bbuckets:
            bk = bbuckets[r]
            nbk = nbbuckets[r]
        if len(al) != len(nal) or len(bk) != len(nbk):
            raise ValueError('%s not reachable from %s' % (target, origin))
        (al, nal, bk, nbk) = [sorted(w, key=default_sort_key) for w in [al, nal, bk, nbk]]

        def others(dic, key):
            if False:
                return 10
            l = []
            for (k, value) in dic.items():
                if k != key:
                    l += list(dic[k])
            return l
        aother = others(nabuckets, r)
        bother = others(nbbuckets, r)
        if len(al) == 0:
            ops += do_shifts_b([], nbk, bk, aother, bother)
        elif len(bk) == 0:
            ops += do_shifts_a(nal, [], al, aother, bother)
        else:
            namax = nal[-1]
            amax = al[-1]
            if nbk[0] - namax <= 0 or bk[0] - amax <= 0:
                raise ValueError('Non-suitable parameters.')
            if namax - amax > 0:
                ops += do_shifts_a(nal, nbk, al, aother, bother)
                ops += do_shifts_b(al, nbk, bk, aother, bother)
            else:
                ops += do_shifts_b(nal, nbk, bk, aother, bother)
                ops += do_shifts_a(nal, bk, al, aother, bother)
        nabuckets[r] = al
        nbbuckets[r] = bk
    ops.reverse()
    return ops

def try_shifted_sum(func, z):
    if False:
        i = 10
        return i + 15
    ' Try to recognise a hypergeometric sum that starts from k > 0. '
    (abuckets, bbuckets) = (sift(func.ap, _mod1), sift(func.bq, _mod1))
    if len(abuckets[S.Zero]) != 1:
        return None
    r = abuckets[S.Zero][0]
    if r <= 0:
        return None
    if S.Zero not in bbuckets:
        return None
    l = list(bbuckets[S.Zero])
    l.sort()
    k = l[0]
    if k <= 0:
        return None
    nap = list(func.ap)
    nap.remove(r)
    nbq = list(func.bq)
    nbq.remove(k)
    k -= 1
    nap = [x - k for x in nap]
    nbq = [x - k for x in nbq]
    ops = []
    for n in range(r - 1):
        ops.append(ShiftA(n + 1))
    ops.reverse()
    fac = factorial(k) / z ** k
    fac *= Mul(*[rf(b, k) for b in nbq])
    fac /= Mul(*[rf(a, k) for a in nap])
    ops += [MultOperator(fac)]
    p = 0
    for n in range(k):
        m = z ** n / factorial(n)
        m *= Mul(*[rf(a, n) for a in nap])
        m /= Mul(*[rf(b, n) for b in nbq])
        p += m
    return (Hyper_Function(nap, nbq), ops, -p)

def try_polynomial(func, z):
    if False:
        i = 10
        return i + 15
    ' Recognise polynomial cases. Returns None if not such a case.\n        Requires order to be fully reduced. '
    (abuckets, bbuckets) = (sift(func.ap, _mod1), sift(func.bq, _mod1))
    a0 = abuckets[S.Zero]
    b0 = bbuckets[S.Zero]
    a0.sort()
    b0.sort()
    al0 = [x for x in a0 if x <= 0]
    bl0 = [x for x in b0 if x <= 0]
    if bl0 and all((a < bl0[-1] for a in al0)):
        return oo
    if not al0:
        return None
    a = al0[-1]
    fac = 1
    res = S.One
    for n in Tuple(*list(range(-a))):
        fac *= z
        fac /= n + 1
        fac *= Mul(*[a + n for a in func.ap])
        fac /= Mul(*[b + n for b in func.bq])
        res += fac
    return res

def try_lerchphi(func):
    if False:
        for i in range(10):
            print('nop')
    '\n    Try to find an expression for Hyper_Function ``func`` in terms of Lerch\n    Transcendents.\n\n    Return None if no such expression can be found.\n    '
    (abuckets, bbuckets) = (sift(func.ap, _mod1), sift(func.bq, _mod1))
    paired = {}
    for (key, value) in abuckets.items():
        if key != 0 and key not in bbuckets:
            return None
        bvalue = bbuckets[key]
        paired[key] = (list(value), list(bvalue))
        bbuckets.pop(key, None)
    if bbuckets != {}:
        return None
    if S.Zero not in abuckets:
        return None
    (aints, bints) = paired[S.Zero]
    paired[S.Zero] = (aints, bints + [1])
    t = Dummy('t')
    numer = S.One
    denom = S.One
    for (key, (avalue, bvalue)) in paired.items():
        if len(avalue) != len(bvalue):
            return None
        for (a, b) in zip(avalue, bvalue):
            if (a - b).is_positive:
                k = a - b
                numer *= rf(b + t, k)
                denom *= rf(b, k)
            else:
                k = b - a
                numer *= rf(a, k)
                denom *= rf(a + t, k)
    part = apart(numer / denom, t)
    args = Add.make_args(part)
    monomials = []
    terms = {}
    for arg in args:
        (numer, denom) = arg.as_numer_denom()
        if not denom.has(t):
            p = Poly(numer, t)
            if not p.is_monomial:
                raise TypeError('p should be monomial')
            ((b,), a) = p.LT()
            monomials += [(a / denom, b)]
            continue
        if numer.has(t):
            raise NotImplementedError('Need partial fraction decomposition with linear denominators')
        (indep, [dep]) = denom.as_coeff_mul(t)
        n = 1
        if dep.is_Pow:
            n = dep.exp
            dep = dep.base
        if dep == t:
            a == 0
        elif dep.is_Add:
            (a, tmp) = dep.as_independent(t)
            b = 1
            if tmp != t:
                (b, _) = tmp.as_independent(t)
            if dep != b * t + a:
                raise NotImplementedError('unrecognised form %s' % dep)
            a /= b
            indep *= b ** n
        else:
            raise NotImplementedError('unrecognised form of partial fraction')
        terms.setdefault(a, []).append((numer / indep, n))
    deriv = {}
    coeffs = {}
    z = Dummy('z')
    monomials.sort(key=lambda x: x[1])
    mon = {0: 1 / (1 - z)}
    if monomials:
        for k in range(monomials[-1][1]):
            mon[k + 1] = z * mon[k].diff(z)
    for (a, n) in monomials:
        coeffs.setdefault(S.One, []).append(a * mon[n])
    for (a, l) in terms.items():
        for (c, k) in l:
            coeffs.setdefault(lerchphi(z, k, a), []).append(c)
        l.sort(key=lambda x: x[1])
        for k in range(2, l[-1][1] + 1):
            deriv[lerchphi(z, k, a)] = [(-a, lerchphi(z, k, a)), (1, lerchphi(z, k - 1, a))]
        deriv[lerchphi(z, 1, a)] = [(-a, lerchphi(z, 1, a)), (1 / (1 - z), S.One)]
    trans = {}
    for (n, b) in enumerate([S.One] + list(deriv.keys())):
        trans[b] = n
    basis = [expand_func(b) for (b, _) in sorted(trans.items(), key=lambda x: x[1])]
    B = Matrix(basis)
    C = Matrix([[0] * len(B)])
    for (b, c) in coeffs.items():
        C[trans[b]] = Add(*c)
    M = zeros(len(B))
    for (b, l) in deriv.items():
        for (c, b2) in l:
            M[trans[b], trans[b2]] = c
    return Formula(func, z, None, [], B, C, M)

def build_hypergeometric_formula(func):
    if False:
        print('Hello World!')
    '\n    Create a formula object representing the hypergeometric function ``func``.\n\n    '
    z = Dummy('z')
    if func.ap:
        afactors = [_x + a for a in func.ap]
        bfactors = [_x + b - 1 for b in func.bq]
        expr = _x * Mul(*bfactors) - z * Mul(*afactors)
        poly = Poly(expr, _x)
        n = poly.degree()
        basis = []
        M = zeros(n)
        for k in range(n):
            a = func.ap[0] + k
            basis += [hyper([a] + list(func.ap[1:]), func.bq, z)]
            if k < n - 1:
                M[k, k] = -a
                M[k, k + 1] = a
        B = Matrix(basis)
        C = Matrix([[1] + [0] * (n - 1)])
        derivs = [eye(n)]
        for k in range(n):
            derivs.append(M * derivs[k])
        l = poly.all_coeffs()
        l.reverse()
        res = [0] * n
        for (k, c) in enumerate(l):
            for (r, d) in enumerate(C * derivs[k]):
                res[r] += c * d
        for (k, c) in enumerate(res):
            M[n - 1, k] = -c / derivs[n - 1][0, n - 1] / poly.all_coeffs()[0]
        return Formula(func, z, None, [], B, C, M)
    else:
        basis = []
        bq = list(func.bq[:])
        for i in range(len(bq)):
            basis += [hyper([], bq, z)]
            bq[i] += 1
        basis += [hyper([], bq, z)]
        B = Matrix(basis)
        n = len(B)
        C = Matrix([[1] + [0] * (n - 1)])
        M = zeros(n)
        M[0, n - 1] = z / Mul(*func.bq)
        for k in range(1, n):
            M[k, k - 1] = func.bq[k - 1]
            M[k, k] = -func.bq[k - 1]
        return Formula(func, z, None, [], B, C, M)

def hyperexpand_special(ap, bq, z):
    if False:
        while True:
            i = 10
    '\n    Try to find a closed-form expression for hyper(ap, bq, z), where ``z``\n    is supposed to be a "special" value, e.g. 1.\n\n    This function tries various of the classical summation formulae\n    (Gauss, Saalschuetz, etc).\n    '
    (p, q) = (len(ap), len(bq))
    z_ = z
    z = unpolarify(z)
    if z == 0:
        return S.One
    from sympy.simplify.simplify import simplify
    if p == 2 and q == 1:
        (a, b, c) = ap + bq
        if z == 1:
            return gamma(c - a - b) * gamma(c) / gamma(c - a) / gamma(c - b)
        if z == -1 and simplify(b - a + c) == 1:
            (b, a) = (a, b)
        if z == -1 and simplify(a - b + c) == 1:
            if b.is_integer and b.is_negative:
                return 2 * cos(pi * b / 2) * gamma(-b) * gamma(b - a + 1) / gamma(-b / 2) / gamma(b / 2 - a + 1)
            else:
                return gamma(b / 2 + 1) * gamma(b - a + 1) / gamma(b + 1) / gamma(b / 2 - a + 1)
    return hyper(ap, bq, z_)
_collection = None

def _hyperexpand(func, z, ops0=[], z0=Dummy('z0'), premult=1, prem=0, rewrite='default'):
    if False:
        i = 10
        return i + 15
    '\n    Try to find an expression for the hypergeometric function ``func``.\n\n    Explanation\n    ===========\n\n    The result is expressed in terms of a dummy variable ``z0``. Then it\n    is multiplied by ``premult``. Then ``ops0`` is applied.\n    ``premult`` must be a*z**prem for some a independent of ``z``.\n    '
    if z.is_zero:
        return S.One
    from sympy.simplify.simplify import simplify
    z = polarify(z, subs=False)
    if rewrite == 'default':
        rewrite = 'nonrepsmall'

    def carryout_plan(f, ops):
        if False:
            for i in range(10):
                print('nop')
        C = apply_operators(f.C.subs(f.z, z0), ops, make_derivative_operator(f.M.subs(f.z, z0), z0))
        C = apply_operators(C, ops0, make_derivative_operator(f.M.subs(f.z, z0) + prem * eye(f.M.shape[0]), z0))
        if premult == 1:
            C = C.applyfunc(make_simp(z0))
        r = reduce(lambda s, m: s + m[0] * m[1], zip(C, f.B.subs(f.z, z0)), S.Zero) * premult
        res = r.subs(z0, z)
        if rewrite:
            res = res.rewrite(rewrite)
        return res
    global _collection
    if _collection is None:
        _collection = FormulaCollection()
    debug('Trying to expand hypergeometric function ', func)
    (func, ops) = reduce_order(func)
    if ops:
        debug('  Reduced order to ', func)
    else:
        debug('  Could not reduce order.')
    res = try_polynomial(func, z0)
    if res is not None:
        debug('  Recognised polynomial.')
        p = apply_operators(res, ops, lambda f: z0 * f.diff(z0))
        p = apply_operators(p * premult, ops0, lambda f: z0 * f.diff(z0))
        return unpolarify(simplify(p).subs(z0, z))
    p = S.Zero
    res = try_shifted_sum(func, z0)
    if res is not None:
        (func, nops, p) = res
        debug('  Recognised shifted sum, reduced order to ', func)
        ops += nops
    p = apply_operators(p, ops, lambda f: z0 * f.diff(z0))
    p = apply_operators(p * premult, ops0, lambda f: z0 * f.diff(z0))
    p = simplify(p).subs(z0, z)
    if unpolarify(z) in [1, -1] and (len(func.ap), len(func.bq)) == (2, 1):
        f = build_hypergeometric_formula(func)
        r = carryout_plan(f, ops).replace(hyper, hyperexpand_special)
        if not r.has(hyper):
            return r + p
    formula = _collection.lookup_origin(func)
    if formula is None:
        formula = try_lerchphi(func)
    if formula is None:
        debug('  Could not find an origin. ', 'Will return answer in terms of simpler hypergeometric functions.')
        formula = build_hypergeometric_formula(func)
    debug('  Found an origin: ', formula.closed_form, ' ', formula.func)
    ops += devise_plan(func, formula.func, z0)
    r = carryout_plan(formula, ops) + p
    return powdenest(r, polar=True).replace(hyper, hyperexpand_special)

def devise_plan_meijer(fro, to, z):
    if False:
        i = 10
        return i + 15
    '\n    Find operators to convert G-function ``fro`` into G-function ``to``.\n\n    Explanation\n    ===========\n\n    It is assumed that ``fro`` and ``to`` have the same signatures, and that in fact\n    any corresponding pair of parameters differs by integers, and a direct path\n    is possible. I.e. if there are parameters a1 b1 c1  and a2 b2 c2 it is\n    assumed that a1 can be shifted to a2, etc. The only thing this routine\n    determines is the order of shifts to apply, nothing clever will be tried.\n    It is also assumed that ``fro`` is suitable.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.hyperexpand import (devise_plan_meijer,\n    ...                                         G_Function)\n    >>> from sympy.abc import z\n\n    Empty plan:\n\n    >>> devise_plan_meijer(G_Function([1], [2], [3], [4]),\n    ...                    G_Function([1], [2], [3], [4]), z)\n    []\n\n    Very simple plans:\n\n    >>> devise_plan_meijer(G_Function([0], [], [], []),\n    ...                    G_Function([1], [], [], []), z)\n    [<Increment upper a index #0 of [0], [], [], [].>]\n    >>> devise_plan_meijer(G_Function([0], [], [], []),\n    ...                    G_Function([-1], [], [], []), z)\n    [<Decrement upper a=0.>]\n    >>> devise_plan_meijer(G_Function([], [1], [], []),\n    ...                    G_Function([], [2], [], []), z)\n    [<Increment lower a index #0 of [], [1], [], [].>]\n\n    Slightly more complicated plans:\n\n    >>> devise_plan_meijer(G_Function([0], [], [], []),\n    ...                    G_Function([2], [], [], []), z)\n    [<Increment upper a index #0 of [1], [], [], [].>,\n    <Increment upper a index #0 of [0], [], [], [].>]\n    >>> devise_plan_meijer(G_Function([0], [], [0], []),\n    ...                    G_Function([-1], [], [1], []), z)\n    [<Increment upper b=0.>, <Decrement upper a=0.>]\n\n    Order matters:\n\n    >>> devise_plan_meijer(G_Function([0], [], [0], []),\n    ...                    G_Function([1], [], [1], []), z)\n    [<Increment upper a index #0 of [0], [], [1], [].>, <Increment upper b=0.>]\n    '

    def try_shift(f, t, shifter, diff, counter):
        if False:
            while True:
                i = 10
        ' Try to apply ``shifter`` in order to bring some element in ``f``\n            nearer to its counterpart in ``to``. ``diff`` is +/- 1 and\n            determines the effect of ``shifter``. Counter is a list of elements\n            blocking the shift.\n\n            Return an operator if change was possible, else None.\n        '
        for (idx, (a, b)) in enumerate(zip(f, t)):
            if (a - b).is_integer and (b - a) / diff > 0 and all((a != x for x in counter)):
                sh = shifter(idx)
                f[idx] += diff
                return sh
    fan = list(fro.an)
    fap = list(fro.ap)
    fbm = list(fro.bm)
    fbq = list(fro.bq)
    ops = []
    change = True
    while change:
        change = False
        op = try_shift(fan, to.an, lambda i: MeijerUnShiftB(fan, fap, fbm, fbq, i, z), 1, fbm + fbq)
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fap, to.ap, lambda i: MeijerUnShiftD(fan, fap, fbm, fbq, i, z), 1, fbm + fbq)
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fbm, to.bm, lambda i: MeijerUnShiftA(fan, fap, fbm, fbq, i, z), -1, fan + fap)
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fbq, to.bq, lambda i: MeijerUnShiftC(fan, fap, fbm, fbq, i, z), -1, fan + fap)
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fan, to.an, lambda i: MeijerShiftB(fan[i]), -1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fap, to.ap, lambda i: MeijerShiftD(fap[i]), -1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fbm, to.bm, lambda i: MeijerShiftA(fbm[i]), 1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fbq, to.bq, lambda i: MeijerShiftC(fbq[i]), 1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
    if fan != list(to.an) or fap != list(to.ap) or fbm != list(to.bm) or (fbq != list(to.bq)):
        raise NotImplementedError('Could not devise plan.')
    ops.reverse()
    return ops
_meijercollection = None

def _meijergexpand(func, z0, allow_hyper=False, rewrite='default', place=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Try to find an expression for the Meijer G function specified\n    by the G_Function ``func``. If ``allow_hyper`` is True, then returning\n    an expression in terms of hypergeometric functions is allowed.\n\n    Currently this just does Slater's theorem.\n    If expansions exist both at zero and at infinity, ``place``\n    can be set to ``0`` or ``zoo`` for the preferred choice.\n    "
    global _meijercollection
    if _meijercollection is None:
        _meijercollection = MeijerFormulaCollection()
    if rewrite == 'default':
        rewrite = None
    func0 = func
    debug('Try to expand Meijer G function corresponding to ', func)
    z = Dummy('z')
    (func, ops) = reduce_order_meijer(func)
    if ops:
        debug('  Reduced order to ', func)
    else:
        debug('  Could not reduce order.')
    f = _meijercollection.lookup_origin(func)
    if f is not None:
        debug('  Found a Meijer G formula: ', f.func)
        ops += devise_plan_meijer(f.func, func, z)
        C = apply_operators(f.C.subs(f.z, z), ops, make_derivative_operator(f.M.subs(f.z, z), z))
        C = C.applyfunc(make_simp(z))
        r = C * f.B.subs(f.z, z)
        r = r[0].subs(z, z0)
        return powdenest(r, polar=True)
    debug("  Could not find a direct formula. Trying Slater's theorem.")

    def can_do(pbm, pap):
        if False:
            print('Hello World!')
        ' Test if slater applies. '
        for i in pbm:
            if len(pbm[i]) > 1:
                l = 0
                if i in pap:
                    l = len(pap[i])
                if l + 1 < len(pbm[i]):
                    return False
        return True

    def do_slater(an, bm, ap, bq, z, zfinal):
        if False:
            return 10
        func = G_Function(an, bm, ap, bq)
        (_, pbm, pap, _) = func.compute_buckets()
        if not can_do(pbm, pap):
            return (S.Zero, False)
        cond = len(an) + len(ap) < len(bm) + len(bq)
        if len(an) + len(ap) == len(bm) + len(bq):
            cond = abs(z) < 1
        if cond is False:
            return (S.Zero, False)
        res = S.Zero
        for m in pbm:
            if len(pbm[m]) == 1:
                bh = pbm[m][0]
                fac = 1
                bo = list(bm)
                bo.remove(bh)
                for bj in bo:
                    fac *= gamma(bj - bh)
                for aj in an:
                    fac *= gamma(1 + bh - aj)
                for bj in bq:
                    fac /= gamma(1 + bh - bj)
                for aj in ap:
                    fac /= gamma(aj - bh)
                nap = [1 + bh - a for a in list(an) + list(ap)]
                nbq = [1 + bh - b for b in list(bo) + list(bq)]
                k = polar_lift(S.NegativeOne ** (len(ap) - len(bm)))
                harg = k * zfinal
                premult = (t / k) ** bh
                hyp = _hyperexpand(Hyper_Function(nap, nbq), harg, ops, t, premult, bh, rewrite=None)
                res += fac * hyp
            else:
                b_ = pbm[m][0]
                ki = [bi - b_ for bi in pbm[m][1:]]
                u = len(ki)
                li = [ai - b_ for ai in pap[m][:u + 1]]
                bo = list(bm)
                for b in pbm[m]:
                    bo.remove(b)
                ao = list(ap)
                for a in pap[m][:u]:
                    ao.remove(a)
                lu = li[-1]
                di = [l - k for (l, k) in zip(li, ki)]
                s = Dummy('s')
                integrand = z ** s
                for b in bm:
                    if not Mod(b, 1) and b.is_Number:
                        b = int(round(b))
                    integrand *= gamma(b - s)
                for a in an:
                    integrand *= gamma(1 - a + s)
                for b in bq:
                    integrand /= gamma(1 - b + s)
                for a in ap:
                    integrand /= gamma(a - s)
                integrand = expand_func(integrand)
                for r in range(int(round(lu))):
                    resid = residue(integrand, s, b_ + r)
                    resid = apply_operators(resid, ops, lambda f: z * f.diff(z))
                    res -= resid
                au = b_ + lu
                k = polar_lift(S.NegativeOne ** (len(ao) + len(bo) + 1))
                harg = k * zfinal
                premult = (t / k) ** au
                nap = [1 + au - a for a in list(an) + list(ap)] + [1]
                nbq = [1 + au - b for b in list(bm) + list(bq)]
                hyp = _hyperexpand(Hyper_Function(nap, nbq), harg, ops, t, premult, au, rewrite=None)
                C = S.NegativeOne ** lu / factorial(lu)
                for i in range(u):
                    C *= S.NegativeOne ** di[i] / rf(lu - li[i] + 1, di[i])
                for a in an:
                    C *= gamma(1 - a + au)
                for b in bo:
                    C *= gamma(b - au)
                for a in ao:
                    C /= gamma(a - au)
                for b in bq:
                    C /= gamma(1 - b + au)
                res += C * hyp
        return (res, cond)
    t = Dummy('t')
    (slater1, cond1) = do_slater(func.an, func.bm, func.ap, func.bq, z, z0)

    def tr(l):
        if False:
            i = 10
            return i + 15
        return [1 - x for x in l]
    for op in ops:
        op._poly = Poly(op._poly.subs({z: 1 / t, _x: -_x}), _x)
    (slater2, cond2) = do_slater(tr(func.bm), tr(func.an), tr(func.bq), tr(func.ap), t, 1 / z0)
    slater1 = powdenest(slater1.subs(z, z0), polar=True)
    slater2 = powdenest(slater2.subs(t, 1 / z0), polar=True)
    if not isinstance(cond2, bool):
        cond2 = cond2.subs(t, 1 / z)
    m = func(z)
    if m.delta > 0 or (m.delta == 0 and len(m.ap) == len(m.bq) and ((re(m.nu) < -1) is not False) and (polar_lift(z0) == polar_lift(1))):
        if cond1 is not False:
            cond1 = True
        if cond2 is not False:
            cond2 = True
    if cond1 is True:
        slater1 = slater1.rewrite(rewrite or 'nonrep')
    else:
        slater1 = slater1.rewrite(rewrite or 'nonrepsmall')
    if cond2 is True:
        slater2 = slater2.rewrite(rewrite or 'nonrep')
    else:
        slater2 = slater2.rewrite(rewrite or 'nonrepsmall')
    if cond1 is not False and cond2 is not False:
        if place == 0:
            cond2 = False
        if place == zoo:
            cond1 = False
    if not isinstance(cond1, bool):
        cond1 = cond1.subs(z, z0)
    if not isinstance(cond2, bool):
        cond2 = cond2.subs(z, z0)

    def weight(expr, cond):
        if False:
            while True:
                i = 10
        if cond is True:
            c0 = 0
        elif cond is False:
            c0 = 1
        else:
            c0 = 2
        if expr.has(oo, zoo, -oo, nan):
            c0 = 3
        return (c0, expr.count(hyper), expr.count_ops())
    w1 = weight(slater1, cond1)
    w2 = weight(slater2, cond2)
    if min(w1, w2) <= (0, 1, oo):
        if w1 < w2:
            return slater1
        else:
            return slater2
    if max(w1[0], w2[0]) <= 1 and max(w1[1], w2[1]) <= 1:
        return Piecewise((slater1, cond1), (slater2, cond2), (func0(z0), True))
    r = Piecewise((slater1, cond1), (slater2, cond2), (func0(z0), True))
    if r.has(hyper) and (not allow_hyper):
        debug('  Could express using hypergeometric functions, but not allowed.')
    if not r.has(hyper) or allow_hyper:
        return r
    return func0(z0)

def hyperexpand(f, allow_hyper=False, rewrite='default', place=None):
    if False:
        i = 10
        return i + 15
    '\n    Expand hypergeometric functions. If allow_hyper is True, allow partial\n    simplification (that is a result different from input,\n    but still containing hypergeometric functions).\n\n    If a G-function has expansions both at zero and at infinity,\n    ``place`` can be set to ``0`` or ``zoo`` to indicate the\n    preferred choice.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.hyperexpand import hyperexpand\n    >>> from sympy.functions import hyper\n    >>> from sympy.abc import z\n    >>> hyperexpand(hyper([], [], z))\n    exp(z)\n\n    Non-hyperegeometric parts of the expression and hypergeometric expressions\n    that are not recognised are left unchanged:\n\n    >>> hyperexpand(1 + hyper([1, 1, 1], [], z))\n    hyper((1, 1, 1), (), z) + 1\n    '
    f = sympify(f)

    def do_replace(ap, bq, z):
        if False:
            while True:
                i = 10
        r = _hyperexpand(Hyper_Function(ap, bq), z, rewrite=rewrite)
        if r is None:
            return hyper(ap, bq, z)
        else:
            return r

    def do_meijer(ap, bq, z):
        if False:
            for i in range(10):
                print('nop')
        r = _meijergexpand(G_Function(ap[0], ap[1], bq[0], bq[1]), z, allow_hyper, rewrite=rewrite, place=place)
        if not r.has(nan, zoo, oo, -oo):
            return r
    return f.replace(hyper, do_replace).replace(meijerg, do_meijer)