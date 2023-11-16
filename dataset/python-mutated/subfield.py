"""
Functions in ``polys.numberfields.subfield`` solve the "Subfield Problem" and
allied problems, for algebraic number fields.

Following Cohen (see [Cohen93]_ Section 4.5), we can define the main problem as
follows:

* **Subfield Problem:**

  Given two number fields $\\mathbb{Q}(\\alpha)$, $\\mathbb{Q}(\\beta)$
  via the minimal polynomials for their generators $\\alpha$ and $\\beta$, decide
  whether one field is isomorphic to a subfield of the other.

From a solution to this problem flow solutions to the following problems as
well:

* **Primitive Element Problem:**

  Given several algebraic numbers
  $\\alpha_1, \\ldots, \\alpha_m$, compute a single algebraic number $\\theta$
  such that $\\mathbb{Q}(\\alpha_1, \\ldots, \\alpha_m) = \\mathbb{Q}(\\theta)$.

* **Field Isomorphism Problem:**

  Decide whether two number fields
  $\\mathbb{Q}(\\alpha)$, $\\mathbb{Q}(\\beta)$ are isomorphic.

* **Field Membership Problem:**

  Given two algebraic numbers $\\alpha$,
  $\\beta$, decide whether $\\alpha \\in \\mathbb{Q}(\\beta)$, and if so write
  $\\alpha = f(\\beta)$ for some $f(x) \\in \\mathbb{Q}[x]$.
"""
from sympy.core.add import Add
from sympy.core.numbers import AlgebraicNumber
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify, _sympify
from sympy.ntheory import sieve
from sympy.polys.densetools import dup_eval
from sympy.polys.domains import QQ
from sympy.polys.numberfields.minpoly import _choose_factor, minimal_polynomial
from sympy.polys.polyerrors import IsomorphismFailed
from sympy.polys.polytools import Poly, PurePoly, factor_list
from sympy.utilities import public
from mpmath import MPContext

def is_isomorphism_possible(a, b):
    if False:
        while True:
            i = 10
    'Necessary but not sufficient test for isomorphism. '
    n = a.minpoly.degree()
    m = b.minpoly.degree()
    if m % n != 0:
        return False
    if n == m:
        return True
    da = a.minpoly.discriminant()
    db = b.minpoly.discriminant()
    (i, k, half) = (1, m // n, db // 2)
    while True:
        p = sieve[i]
        P = p ** k
        if P > half:
            break
        if da % p % 2 and (not db % P):
            return False
        i += 1
    return True

def field_isomorphism_pslq(a, b):
    if False:
        while True:
            i = 10
    'Construct field isomorphism using PSLQ algorithm. '
    if not a.root.is_real or not b.root.is_real:
        raise NotImplementedError("PSLQ doesn't support complex coefficients")
    f = a.minpoly
    g = b.minpoly.replace(f.gen)
    (n, m, prev) = (100, b.minpoly.degree(), None)
    ctx = MPContext()
    for i in range(1, 5):
        A = a.root.evalf(n)
        B = b.root.evalf(n)
        basis = [1, B] + [B ** i for i in range(2, m)] + [-A]
        ctx.dps = n
        coeffs = ctx.pslq(basis, maxcoeff=10 ** 10, maxsteps=1000)
        if coeffs is None:
            break
        if coeffs != prev:
            prev = coeffs
        else:
            break
        coeffs = [S(c) / coeffs[-1] for c in coeffs[:-1]]
        while not coeffs[-1]:
            coeffs.pop()
        coeffs = list(reversed(coeffs))
        h = Poly(coeffs, f.gen, domain='QQ')
        if f.compose(h).rem(g).is_zero:
            return coeffs
        else:
            n *= 2
    return None

def field_isomorphism_factor(a, b):
    if False:
        return 10
    'Construct field isomorphism via factorization. '
    (_, factors) = factor_list(a.minpoly, extension=b)
    for (f, _) in factors:
        if f.degree() == 1:
            c = -f.rep.TC()
            coeffs = c.to_sympy_list()
            (d, terms) = (len(coeffs) - 1, [])
            for (i, coeff) in enumerate(coeffs):
                terms.append(coeff * b.root ** (d - i))
            r = Add(*terms)
            if a.minpoly.same_root(r, a):
                return coeffs
    return None

@public
def field_isomorphism(a, b, *, fast=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Find an embedding of one number field into another.\n\n    Explanation\n    ===========\n\n    This function looks for an isomorphism from $\\mathbb{Q}(a)$ onto some\n    subfield of $\\mathbb{Q}(b)$. Thus, it solves the Subfield Problem.\n\n    Examples\n    ========\n\n    >>> from sympy import sqrt, field_isomorphism, I\n    >>> print(field_isomorphism(3, sqrt(2)))  # doctest: +SKIP\n    [3]\n    >>> print(field_isomorphism( I*sqrt(3), I*sqrt(3)/2))  # doctest: +SKIP\n    [2, 0]\n\n    Parameters\n    ==========\n\n    a : :py:class:`~.Expr`\n        Any expression representing an algebraic number.\n    b : :py:class:`~.Expr`\n        Any expression representing an algebraic number.\n    fast : boolean, optional (default=True)\n        If ``True``, we first attempt a potentially faster way of computing the\n        isomorphism, falling back on a slower method if this fails. If\n        ``False``, we go directly to the slower method, which is guaranteed to\n        return a result.\n\n    Returns\n    =======\n\n    List of rational numbers, or None\n        If $\\mathbb{Q}(a)$ is not isomorphic to some subfield of\n        $\\mathbb{Q}(b)$, then return ``None``. Otherwise, return a list of\n        rational numbers representing an element of $\\mathbb{Q}(b)$ to which\n        $a$ may be mapped, in order to define a monomorphism, i.e. an\n        isomorphism from $\\mathbb{Q}(a)$ to some subfield of $\\mathbb{Q}(b)$.\n        The elements of the list are the coefficients of falling powers of $b$.\n\n    '
    (a, b) = (sympify(a), sympify(b))
    if not a.is_AlgebraicNumber:
        a = AlgebraicNumber(a)
    if not b.is_AlgebraicNumber:
        b = AlgebraicNumber(b)
    a = a.to_primitive_element()
    b = b.to_primitive_element()
    if a == b:
        return a.coeffs()
    n = a.minpoly.degree()
    m = b.minpoly.degree()
    if n == 1:
        return [a.root]
    if m % n != 0:
        return None
    if fast:
        try:
            result = field_isomorphism_pslq(a, b)
            if result is not None:
                return result
        except NotImplementedError:
            pass
    return field_isomorphism_factor(a, b)

def _switch_domain(g, K):
    if False:
        while True:
            i = 10
    frep = g.rep.inject()
    hrep = frep.eject(K, front=True)
    return g.new(hrep, g.gens[0])

def _linsolve(p):
    if False:
        return 10
    (c, d) = p.rep.to_list()
    return -d / c

@public
def primitive_element(extension, x=None, *, ex=False, polys=False):
    if False:
        i = 10
        return i + 15
    '\n    Find a single generator for a number field given by several generators.\n\n    Explanation\n    ===========\n\n    The basic problem is this: Given several algebraic numbers\n    $\\alpha_1, \\alpha_2, \\ldots, \\alpha_n$, find a single algebraic number\n    $\\theta$ such that\n    $\\mathbb{Q}(\\alpha_1, \\alpha_2, \\ldots, \\alpha_n) = \\mathbb{Q}(\\theta)$.\n\n    This function actually guarantees that $\\theta$ will be a linear\n    combination of the $\\alpha_i$, with non-negative integer coefficients.\n\n    Furthermore, if desired, this function will tell you how to express each\n    $\\alpha_i$ as a $\\mathbb{Q}$-linear combination of the powers of $\\theta$.\n\n    Examples\n    ========\n\n    >>> from sympy import primitive_element, sqrt, S, minpoly, simplify\n    >>> from sympy.abc import x\n    >>> f, lincomb, reps = primitive_element([sqrt(2), sqrt(3)], x, ex=True)\n\n    Then ``lincomb`` tells us the primitive element as a linear combination of\n    the given generators ``sqrt(2)`` and ``sqrt(3)``.\n\n    >>> print(lincomb)\n    [1, 1]\n\n    This means the primtiive element is $\\sqrt{2} + \\sqrt{3}$.\n    Meanwhile ``f`` is the minimal polynomial for this primitive element.\n\n    >>> print(f)\n    x**4 - 10*x**2 + 1\n    >>> print(minpoly(sqrt(2) + sqrt(3), x))\n    x**4 - 10*x**2 + 1\n\n    Finally, ``reps`` (which was returned only because we set keyword arg\n    ``ex=True``) tells us how to recover each of the generators $\\sqrt{2}$ and\n    $\\sqrt{3}$ as $\\mathbb{Q}$-linear combinations of the powers of the\n    primitive element $\\sqrt{2} + \\sqrt{3}$.\n\n    >>> print([S(r) for r in reps[0]])\n    [1/2, 0, -9/2, 0]\n    >>> theta = sqrt(2) + sqrt(3)\n    >>> print(simplify(theta**3/2 - 9*theta/2))\n    sqrt(2)\n    >>> print([S(r) for r in reps[1]])\n    [-1/2, 0, 11/2, 0]\n    >>> print(simplify(-theta**3/2 + 11*theta/2))\n    sqrt(3)\n\n    Parameters\n    ==========\n\n    extension : list of :py:class:`~.Expr`\n        Each expression must represent an algebraic number $\\alpha_i$.\n    x : :py:class:`~.Symbol`, optional (default=None)\n        The desired symbol to appear in the computed minimal polynomial for the\n        primitive element $\\theta$. If ``None``, we use a dummy symbol.\n    ex : boolean, optional (default=False)\n        If and only if ``True``, compute the representation of each $\\alpha_i$\n        as a $\\mathbb{Q}$-linear combination over the powers of $\\theta$.\n    polys : boolean, optional (default=False)\n        If ``True``, return the minimal polynomial as a :py:class:`~.Poly`.\n        Otherwise return it as an :py:class:`~.Expr`.\n\n    Returns\n    =======\n\n    Pair (f, coeffs) or triple (f, coeffs, reps), where:\n        ``f`` is the minimal polynomial for the primitive element.\n        ``coeffs`` gives the primitive element as a linear combination of the\n        given generators.\n        ``reps`` is present if and only if argument ``ex=True`` was passed,\n        and is a list of lists of rational numbers. Each list gives the\n        coefficients of falling powers of the primitive element, to recover\n        one of the original, given generators.\n\n    '
    if not extension:
        raise ValueError('Cannot compute primitive element for empty extension')
    extension = [_sympify(ext) for ext in extension]
    if x is not None:
        (x, cls) = (sympify(x), Poly)
    else:
        (x, cls) = (Dummy('x'), PurePoly)
    if not ex:
        (gen, coeffs) = (extension[0], [1])
        g = minimal_polynomial(gen, x, polys=True)
        for ext in extension[1:]:
            if ext.is_Rational:
                coeffs.append(0)
                continue
            (_, factors) = factor_list(g, extension=ext)
            g = _choose_factor(factors, x, gen)
            (s, _, g) = g.sqf_norm()
            gen += s * ext
            coeffs.append(s)
        if not polys:
            return (g.as_expr(), coeffs)
        else:
            return (cls(g), coeffs)
    (gen, coeffs) = (extension[0], [1])
    f = minimal_polynomial(gen, x, polys=True)
    K = QQ.algebraic_field((f, gen))
    reps = [K.unit]
    for ext in extension[1:]:
        if ext.is_Rational:
            coeffs.append(0)
            reps.append(K.convert(ext))
            continue
        p = minimal_polynomial(ext, x, polys=True)
        L = QQ.algebraic_field((p, ext))
        (_, factors) = factor_list(f, domain=L)
        f = _choose_factor(factors, x, gen)
        (s, g, f) = f.sqf_norm()
        gen += s * ext
        coeffs.append(s)
        K = QQ.algebraic_field((f, gen))
        h = _switch_domain(g, K)
        erep = _linsolve(h.gcd(p))
        ogen = K.unit - s * erep
        reps = [dup_eval(_.to_list(), ogen, K) for _ in reps] + [erep]
    if K.ext.root.is_Rational:
        H = [K.convert(_).rep for _ in extension]
        coeffs = [0] * len(extension)
        f = cls(x, domain=QQ)
    else:
        H = [_.to_list() for _ in reps]
    if not polys:
        return (f.as_expr(), coeffs, H)
    else:
        return (f, coeffs, H)

@public
def to_number_field(extension, theta=None, *, gen=None, alias=None):
    if False:
        while True:
            i = 10
    "\n    Express one algebraic number in the field generated by another.\n\n    Explanation\n    ===========\n\n    Given two algebraic numbers $\\eta, \\theta$, this function either expresses\n    $\\eta$ as an element of $\\mathbb{Q}(\\theta)$, or else raises an exception\n    if $\\eta \\not\\in \\mathbb{Q}(\\theta)$.\n\n    This function is essentially just a convenience, utilizing\n    :py:func:`~.field_isomorphism` (our solution of the Subfield Problem) to\n    solve this, the Field Membership Problem.\n\n    As an additional convenience, this function allows you to pass a list of\n    algebraic numbers $\\alpha_1, \\alpha_2, \\ldots, \\alpha_n$ instead of $\\eta$.\n    It then computes $\\eta$ for you, as a solution of the Primitive Element\n    Problem, using :py:func:`~.primitive_element` on the list of $\\alpha_i$.\n\n    Examples\n    ========\n\n    >>> from sympy import sqrt, to_number_field\n    >>> eta = sqrt(2)\n    >>> theta = sqrt(2) + sqrt(3)\n    >>> a = to_number_field(eta, theta)\n    >>> print(type(a))\n    <class 'sympy.core.numbers.AlgebraicNumber'>\n    >>> a.root\n    sqrt(2) + sqrt(3)\n    >>> print(a)\n    sqrt(2)\n    >>> a.coeffs()\n    [1/2, 0, -9/2, 0]\n\n    We get an :py:class:`~.AlgebraicNumber`, whose ``.root`` is $\\theta$, whose\n    value is $\\eta$, and whose ``.coeffs()`` show how to write $\\eta$ as a\n    $\\mathbb{Q}$-linear combination in falling powers of $\\theta$.\n\n    Parameters\n    ==========\n\n    extension : :py:class:`~.Expr` or list of :py:class:`~.Expr`\n        Either the algebraic number that is to be expressed in the other field,\n        or else a list of algebraic numbers, a primitive element for which is\n        to be expressed in the other field.\n    theta : :py:class:`~.Expr`, None, optional (default=None)\n        If an :py:class:`~.Expr` representing an algebraic number, behavior is\n        as described under **Explanation**. If ``None``, then this function\n        reduces to a shorthand for calling :py:func:`~.primitive_element` on\n        ``extension`` and turning the computed primitive element into an\n        :py:class:`~.AlgebraicNumber`.\n    gen : :py:class:`~.Symbol`, None, optional (default=None)\n        If provided, this will be used as the generator symbol for the minimal\n        polynomial in the returned :py:class:`~.AlgebraicNumber`.\n    alias : str, :py:class:`~.Symbol`, None, optional (default=None)\n        If provided, this will be used as the alias symbol for the returned\n        :py:class:`~.AlgebraicNumber`.\n\n    Returns\n    =======\n\n    AlgebraicNumber\n        Belonging to $\\mathbb{Q}(\\theta)$ and equaling $\\eta$.\n\n    Raises\n    ======\n\n    IsomorphismFailed\n        If $\\eta \\not\\in \\mathbb{Q}(\\theta)$.\n\n    See Also\n    ========\n\n    field_isomorphism\n    primitive_element\n\n    "
    if hasattr(extension, '__iter__'):
        extension = list(extension)
    else:
        extension = [extension]
    if len(extension) == 1 and isinstance(extension[0], tuple):
        return AlgebraicNumber(extension[0], alias=alias)
    (minpoly, coeffs) = primitive_element(extension, gen, polys=True)
    root = sum([coeff * ext for (coeff, ext) in zip(coeffs, extension)])
    if theta is None:
        return AlgebraicNumber((minpoly, root), alias=alias)
    else:
        theta = sympify(theta)
        if not theta.is_AlgebraicNumber:
            theta = AlgebraicNumber(theta, gen=gen, alias=alias)
        coeffs = field_isomorphism(root, theta)
        if coeffs is not None:
            return AlgebraicNumber(theta, coeffs, alias=alias)
        else:
            raise IsomorphismFailed('%s is not in a subfield of %s' % (root, theta.root))