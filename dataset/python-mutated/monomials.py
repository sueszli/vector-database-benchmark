"""Tools and arithmetics for monomials of distributed polynomials. """
from itertools import combinations_with_replacement, product
from textwrap import dedent
from sympy.core import Mul, S, Tuple, sympify
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import PicklableWithSlots, dict_from_expr
from sympy.utilities import public
from sympy.utilities.iterables import is_sequence, iterable

@public
def itermonomials(variables, max_degrees, min_degrees=None):
    if False:
        while True:
            i = 10
    "\n    ``max_degrees`` and ``min_degrees`` are either both integers or both lists.\n    Unless otherwise specified, ``min_degrees`` is either ``0`` or\n    ``[0, ..., 0]``.\n\n    A generator of all monomials ``monom`` is returned, such that\n    either\n    ``min_degree <= total_degree(monom) <= max_degree``,\n    or\n    ``min_degrees[i] <= degree_list(monom)[i] <= max_degrees[i]``,\n    for all ``i``.\n\n    Case I. ``max_degrees`` and ``min_degrees`` are both integers\n    =============================================================\n\n    Given a set of variables $V$ and a min_degree $N$ and a max_degree $M$\n    generate a set of monomials of degree less than or equal to $N$ and greater\n    than or equal to $M$. The total number of monomials in commutative\n    variables is huge and is given by the following formula if $M = 0$:\n\n        .. math::\n            \\frac{(\\#V + N)!}{\\#V! N!}\n\n    For example if we would like to generate a dense polynomial of\n    a total degree $N = 50$ and $M = 0$, which is the worst case, in 5\n    variables, assuming that exponents and all of coefficients are 32-bit long\n    and stored in an array we would need almost 80 GiB of memory! Fortunately\n    most polynomials, that we will encounter, are sparse.\n\n    Consider monomials in commutative variables $x$ and $y$\n    and non-commutative variables $a$ and $b$::\n\n        >>> from sympy import symbols\n        >>> from sympy.polys.monomials import itermonomials\n        >>> from sympy.polys.orderings import monomial_key\n        >>> from sympy.abc import x, y\n\n        >>> sorted(itermonomials([x, y], 2), key=monomial_key('grlex', [y, x]))\n        [1, x, y, x**2, x*y, y**2]\n\n        >>> sorted(itermonomials([x, y], 3), key=monomial_key('grlex', [y, x]))\n        [1, x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3]\n\n        >>> a, b = symbols('a, b', commutative=False)\n        >>> set(itermonomials([a, b, x], 2))\n        {1, a, a**2, b, b**2, x, x**2, a*b, b*a, x*a, x*b}\n\n        >>> sorted(itermonomials([x, y], 2, 1), key=monomial_key('grlex', [y, x]))\n        [x, y, x**2, x*y, y**2]\n\n    Case II. ``max_degrees`` and ``min_degrees`` are both lists\n    ===========================================================\n\n    If ``max_degrees = [d_1, ..., d_n]`` and\n    ``min_degrees = [e_1, ..., e_n]``, the number of monomials generated\n    is:\n\n    .. math::\n        (d_1 - e_1 + 1) (d_2 - e_2 + 1) \\cdots (d_n - e_n + 1)\n\n    Let us generate all monomials ``monom`` in variables $x$ and $y$\n    such that ``[1, 2][i] <= degree_list(monom)[i] <= [2, 4][i]``,\n    ``i = 0, 1`` ::\n\n        >>> from sympy import symbols\n        >>> from sympy.polys.monomials import itermonomials\n        >>> from sympy.polys.orderings import monomial_key\n        >>> from sympy.abc import x, y\n\n        >>> sorted(itermonomials([x, y], [2, 4], [1, 2]), reverse=True, key=monomial_key('lex', [x, y]))\n        [x**2*y**4, x**2*y**3, x**2*y**2, x*y**4, x*y**3, x*y**2]\n    "
    n = len(variables)
    if is_sequence(max_degrees):
        if len(max_degrees) != n:
            raise ValueError('Argument sizes do not match')
        if min_degrees is None:
            min_degrees = [0] * n
        elif not is_sequence(min_degrees):
            raise ValueError('min_degrees is not a list')
        else:
            if len(min_degrees) != n:
                raise ValueError('Argument sizes do not match')
            if any((i < 0 for i in min_degrees)):
                raise ValueError('min_degrees cannot contain negative numbers')
        total_degree = False
    else:
        max_degree = max_degrees
        if max_degree < 0:
            raise ValueError('max_degrees cannot be negative')
        if min_degrees is None:
            min_degree = 0
        else:
            if min_degrees < 0:
                raise ValueError('min_degrees cannot be negative')
            min_degree = min_degrees
        total_degree = True
    if total_degree:
        if min_degree > max_degree:
            return
        if not variables or max_degree == 0:
            yield S.One
            return
        variables = list(variables) + [S.One]
        if all((variable.is_commutative for variable in variables)):
            monomials_list_comm = []
            for item in combinations_with_replacement(variables, max_degree):
                powers = {variable: 0 for variable in variables}
                for variable in item:
                    if variable != 1:
                        powers[variable] += 1
                if sum(powers.values()) >= min_degree:
                    monomials_list_comm.append(Mul(*item))
            yield from set(monomials_list_comm)
        else:
            monomials_list_non_comm = []
            for item in product(variables, repeat=max_degree):
                powers = {variable: 0 for variable in variables}
                for variable in item:
                    if variable != 1:
                        powers[variable] += 1
                if sum(powers.values()) >= min_degree:
                    monomials_list_non_comm.append(Mul(*item))
            yield from set(monomials_list_non_comm)
    else:
        if any((min_degrees[i] > max_degrees[i] for i in range(n))):
            raise ValueError('min_degrees[i] must be <= max_degrees[i] for all i')
        power_lists = []
        for (var, min_d, max_d) in zip(variables, min_degrees, max_degrees):
            power_lists.append([var ** i for i in range(min_d, max_d + 1)])
        for powers in product(*power_lists):
            yield Mul(*powers)

def monomial_count(V, N):
    if False:
        print('Hello World!')
    "\n    Computes the number of monomials.\n\n    The number of monomials is given by the following formula:\n\n    .. math::\n\n        \\frac{(\\#V + N)!}{\\#V! N!}\n\n    where `N` is a total degree and `V` is a set of variables.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.monomials import itermonomials, monomial_count\n    >>> from sympy.polys.orderings import monomial_key\n    >>> from sympy.abc import x, y\n\n    >>> monomial_count(2, 2)\n    6\n\n    >>> M = list(itermonomials([x, y], 2))\n\n    >>> sorted(M, key=monomial_key('grlex', [y, x]))\n    [1, x, y, x**2, x*y, y**2]\n    >>> len(M)\n    6\n\n    "
    from sympy.functions.combinatorial.factorials import factorial
    return factorial(V + N) / factorial(V) / factorial(N)

def monomial_mul(A, B):
    if False:
        print('Hello World!')
    '\n    Multiplication of tuples representing monomials.\n\n    Examples\n    ========\n\n    Lets multiply `x**3*y**4*z` with `x*y**2`::\n\n        >>> from sympy.polys.monomials import monomial_mul\n\n        >>> monomial_mul((3, 4, 1), (1, 2, 0))\n        (4, 6, 1)\n\n    which gives `x**4*y**5*z`.\n\n    '
    return tuple([a + b for (a, b) in zip(A, B)])

def monomial_div(A, B):
    if False:
        while True:
            i = 10
    '\n    Division of tuples representing monomials.\n\n    Examples\n    ========\n\n    Lets divide `x**3*y**4*z` by `x*y**2`::\n\n        >>> from sympy.polys.monomials import monomial_div\n\n        >>> monomial_div((3, 4, 1), (1, 2, 0))\n        (2, 2, 1)\n\n    which gives `x**2*y**2*z`. However::\n\n        >>> monomial_div((3, 4, 1), (1, 2, 2)) is None\n        True\n\n    `x*y**2*z**2` does not divide `x**3*y**4*z`.\n\n    '
    C = monomial_ldiv(A, B)
    if all((c >= 0 for c in C)):
        return tuple(C)
    else:
        return None

def monomial_ldiv(A, B):
    if False:
        return 10
    '\n    Division of tuples representing monomials.\n\n    Examples\n    ========\n\n    Lets divide `x**3*y**4*z` by `x*y**2`::\n\n        >>> from sympy.polys.monomials import monomial_ldiv\n\n        >>> monomial_ldiv((3, 4, 1), (1, 2, 0))\n        (2, 2, 1)\n\n    which gives `x**2*y**2*z`.\n\n        >>> monomial_ldiv((3, 4, 1), (1, 2, 2))\n        (2, 2, -1)\n\n    which gives `x**2*y**2*z**-1`.\n\n    '
    return tuple([a - b for (a, b) in zip(A, B)])

def monomial_pow(A, n):
    if False:
        for i in range(10):
            print('nop')
    'Return the n-th pow of the monomial. '
    return tuple([a * n for a in A])

def monomial_gcd(A, B):
    if False:
        while True:
            i = 10
    '\n    Greatest common divisor of tuples representing monomials.\n\n    Examples\n    ========\n\n    Lets compute GCD of `x*y**4*z` and `x**3*y**2`::\n\n        >>> from sympy.polys.monomials import monomial_gcd\n\n        >>> monomial_gcd((1, 4, 1), (3, 2, 0))\n        (1, 2, 0)\n\n    which gives `x*y**2`.\n\n    '
    return tuple([min(a, b) for (a, b) in zip(A, B)])

def monomial_lcm(A, B):
    if False:
        i = 10
        return i + 15
    '\n    Least common multiple of tuples representing monomials.\n\n    Examples\n    ========\n\n    Lets compute LCM of `x*y**4*z` and `x**3*y**2`::\n\n        >>> from sympy.polys.monomials import monomial_lcm\n\n        >>> monomial_lcm((1, 4, 1), (3, 2, 0))\n        (3, 4, 1)\n\n    which gives `x**3*y**4*z`.\n\n    '
    return tuple([max(a, b) for (a, b) in zip(A, B)])

def monomial_divides(A, B):
    if False:
        while True:
            i = 10
    '\n    Does there exist a monomial X such that XA == B?\n\n    Examples\n    ========\n\n    >>> from sympy.polys.monomials import monomial_divides\n    >>> monomial_divides((1, 2), (3, 4))\n    True\n    >>> monomial_divides((1, 2), (0, 2))\n    False\n    '
    return all((a <= b for (a, b) in zip(A, B)))

def monomial_max(*monoms):
    if False:
        return 10
    '\n    Returns maximal degree for each variable in a set of monomials.\n\n    Examples\n    ========\n\n    Consider monomials `x**3*y**4*z**5`, `y**5*z` and `x**6*y**3*z**9`.\n    We wish to find out what is the maximal degree for each of `x`, `y`\n    and `z` variables::\n\n        >>> from sympy.polys.monomials import monomial_max\n\n        >>> monomial_max((3,4,5), (0,5,1), (6,3,9))\n        (6, 5, 9)\n\n    '
    M = list(monoms[0])
    for N in monoms[1:]:
        for (i, n) in enumerate(N):
            M[i] = max(M[i], n)
    return tuple(M)

def monomial_min(*monoms):
    if False:
        print('Hello World!')
    '\n    Returns minimal degree for each variable in a set of monomials.\n\n    Examples\n    ========\n\n    Consider monomials `x**3*y**4*z**5`, `y**5*z` and `x**6*y**3*z**9`.\n    We wish to find out what is the minimal degree for each of `x`, `y`\n    and `z` variables::\n\n        >>> from sympy.polys.monomials import monomial_min\n\n        >>> monomial_min((3,4,5), (0,5,1), (6,3,9))\n        (0, 3, 1)\n\n    '
    M = list(monoms[0])
    for N in monoms[1:]:
        for (i, n) in enumerate(N):
            M[i] = min(M[i], n)
    return tuple(M)

def monomial_deg(M):
    if False:
        i = 10
        return i + 15
    '\n    Returns the total degree of a monomial.\n\n    Examples\n    ========\n\n    The total degree of `xy^2` is 3:\n\n    >>> from sympy.polys.monomials import monomial_deg\n    >>> monomial_deg((1, 2))\n    3\n    '
    return sum(M)

def term_div(a, b, domain):
    if False:
        print('Hello World!')
    'Division of two terms in over a ring/field. '
    (a_lm, a_lc) = a
    (b_lm, b_lc) = b
    monom = monomial_div(a_lm, b_lm)
    if domain.is_Field:
        if monom is not None:
            return (monom, domain.quo(a_lc, b_lc))
        else:
            return None
    elif not (monom is None or a_lc % b_lc):
        return (monom, domain.quo(a_lc, b_lc))
    else:
        return None

class MonomialOps:
    """Code generator of fast monomial arithmetic functions. """

    def __init__(self, ngens):
        if False:
            for i in range(10):
                print('nop')
        self.ngens = ngens

    def _build(self, code, name):
        if False:
            return 10
        ns = {}
        exec(code, ns)
        return ns[name]

    def _vars(self, name):
        if False:
            print('Hello World!')
        return ['%s%s' % (name, i) for i in range(self.ngens)]

    def mul(self):
        if False:
            while True:
                i = 10
        name = 'monomial_mul'
        template = dedent('        def %(name)s(A, B):\n            (%(A)s,) = A\n            (%(B)s,) = B\n            return (%(AB)s,)\n        ')
        A = self._vars('a')
        B = self._vars('b')
        AB = ['%s + %s' % (a, b) for (a, b) in zip(A, B)]
        code = template % {'name': name, 'A': ', '.join(A), 'B': ', '.join(B), 'AB': ', '.join(AB)}
        return self._build(code, name)

    def pow(self):
        if False:
            return 10
        name = 'monomial_pow'
        template = dedent('        def %(name)s(A, k):\n            (%(A)s,) = A\n            return (%(Ak)s,)\n        ')
        A = self._vars('a')
        Ak = ['%s*k' % a for a in A]
        code = template % {'name': name, 'A': ', '.join(A), 'Ak': ', '.join(Ak)}
        return self._build(code, name)

    def mulpow(self):
        if False:
            print('Hello World!')
        name = 'monomial_mulpow'
        template = dedent('        def %(name)s(A, B, k):\n            (%(A)s,) = A\n            (%(B)s,) = B\n            return (%(ABk)s,)\n        ')
        A = self._vars('a')
        B = self._vars('b')
        ABk = ['%s + %s*k' % (a, b) for (a, b) in zip(A, B)]
        code = template % {'name': name, 'A': ', '.join(A), 'B': ', '.join(B), 'ABk': ', '.join(ABk)}
        return self._build(code, name)

    def ldiv(self):
        if False:
            return 10
        name = 'monomial_ldiv'
        template = dedent('        def %(name)s(A, B):\n            (%(A)s,) = A\n            (%(B)s,) = B\n            return (%(AB)s,)\n        ')
        A = self._vars('a')
        B = self._vars('b')
        AB = ['%s - %s' % (a, b) for (a, b) in zip(A, B)]
        code = template % {'name': name, 'A': ', '.join(A), 'B': ', '.join(B), 'AB': ', '.join(AB)}
        return self._build(code, name)

    def div(self):
        if False:
            i = 10
            return i + 15
        name = 'monomial_div'
        template = dedent('        def %(name)s(A, B):\n            (%(A)s,) = A\n            (%(B)s,) = B\n            %(RAB)s\n            return (%(R)s,)\n        ')
        A = self._vars('a')
        B = self._vars('b')
        RAB = ['r%(i)s = a%(i)s - b%(i)s\n    if r%(i)s < 0: return None' % {'i': i} for i in range(self.ngens)]
        R = self._vars('r')
        code = template % {'name': name, 'A': ', '.join(A), 'B': ', '.join(B), 'RAB': '\n    '.join(RAB), 'R': ', '.join(R)}
        return self._build(code, name)

    def lcm(self):
        if False:
            print('Hello World!')
        name = 'monomial_lcm'
        template = dedent('        def %(name)s(A, B):\n            (%(A)s,) = A\n            (%(B)s,) = B\n            return (%(AB)s,)\n        ')
        A = self._vars('a')
        B = self._vars('b')
        AB = ['%s if %s >= %s else %s' % (a, a, b, b) for (a, b) in zip(A, B)]
        code = template % {'name': name, 'A': ', '.join(A), 'B': ', '.join(B), 'AB': ', '.join(AB)}
        return self._build(code, name)

    def gcd(self):
        if False:
            for i in range(10):
                print('nop')
        name = 'monomial_gcd'
        template = dedent('        def %(name)s(A, B):\n            (%(A)s,) = A\n            (%(B)s,) = B\n            return (%(AB)s,)\n        ')
        A = self._vars('a')
        B = self._vars('b')
        AB = ['%s if %s <= %s else %s' % (a, a, b, b) for (a, b) in zip(A, B)]
        code = template % {'name': name, 'A': ', '.join(A), 'B': ', '.join(B), 'AB': ', '.join(AB)}
        return self._build(code, name)

@public
class Monomial(PicklableWithSlots):
    """Class representing a monomial, i.e. a product of powers. """
    __slots__ = ('exponents', 'gens')

    def __init__(self, monom, gens=None):
        if False:
            while True:
                i = 10
        if not iterable(monom):
            (rep, gens) = dict_from_expr(sympify(monom), gens=gens)
            if len(rep) == 1 and list(rep.values())[0] == 1:
                monom = list(rep.keys())[0]
            else:
                raise ValueError('Expected a monomial got {}'.format(monom))
        self.exponents = tuple(map(int, monom))
        self.gens = gens

    def rebuild(self, exponents, gens=None):
        if False:
            return 10
        return self.__class__(exponents, gens or self.gens)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.exponents)

    def __iter__(self):
        if False:
            return 10
        return iter(self.exponents)

    def __getitem__(self, item):
        if False:
            while True:
                i = 10
        return self.exponents[item]

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash((self.__class__.__name__, self.exponents, self.gens))

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.gens:
            return '*'.join(['%s**%s' % (gen, exp) for (gen, exp) in zip(self.gens, self.exponents)])
        else:
            return '%s(%s)' % (self.__class__.__name__, self.exponents)

    def as_expr(self, *gens):
        if False:
            i = 10
            return i + 15
        'Convert a monomial instance to a SymPy expression. '
        gens = gens or self.gens
        if not gens:
            raise ValueError('Cannot convert %s to an expression without generators' % self)
        return Mul(*[gen ** exp for (gen, exp) in zip(gens, self.exponents)])

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            return False
        return self.exponents == exponents

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        return not self == other

    def __mul__(self, other):
        if False:
            return 10
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            raise NotImplementedError
        return self.rebuild(monomial_mul(self.exponents, exponents))

    def __truediv__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            raise NotImplementedError
        result = monomial_div(self.exponents, exponents)
        if result is not None:
            return self.rebuild(result)
        else:
            raise ExactQuotientFailed(self, Monomial(other))
    __floordiv__ = __truediv__

    def __pow__(self, other):
        if False:
            while True:
                i = 10
        n = int(other)
        if not n:
            return self.rebuild([0] * len(self))
        elif n > 0:
            exponents = self.exponents
            for i in range(1, n):
                exponents = monomial_mul(exponents, self.exponents)
            return self.rebuild(exponents)
        else:
            raise ValueError('a non-negative integer expected, got %s' % other)

    def gcd(self, other):
        if False:
            while True:
                i = 10
        'Greatest common divisor of monomials. '
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            raise TypeError('an instance of Monomial class expected, got %s' % other)
        return self.rebuild(monomial_gcd(self.exponents, exponents))

    def lcm(self, other):
        if False:
            i = 10
            return i + 15
        'Least common multiple of monomials. '
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            raise TypeError('an instance of Monomial class expected, got %s' % other)
        return self.rebuild(monomial_lcm(self.exponents, exponents))