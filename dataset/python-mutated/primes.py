"""Prime ideals in number fields. """
from sympy.polys.polytools import Poly
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable
from sympy.utilities.decorator import public
from .basis import round_two, nilradical_mod_p
from .exceptions import StructureError
from .modules import ModuleEndomorphism, find_min_poly
from .utilities import coeff_search, supplement_a_subspace

def _check_formal_conditions_for_maximal_order(submodule):
    if False:
        for i in range(10):
            print('nop')
    '\n    Several functions in this module accept an argument which is to be a\n    :py:class:`~.Submodule` representing the maximal order in a number field,\n    such as returned by the :py:func:`~sympy.polys.numberfields.basis.round_two`\n    algorithm.\n\n    We do not attempt to check that the given ``Submodule`` actually represents\n    a maximal order, but we do check a basic set of formal conditions that the\n    ``Submodule`` must satisfy, at a minimum. The purpose is to catch an\n    obviously ill-formed argument.\n    '
    prefix = 'The submodule representing the maximal order should '
    cond = None
    if not submodule.is_power_basis_submodule():
        cond = 'be a direct submodule of a power basis.'
    elif not submodule.starts_with_unity():
        cond = 'have 1 as its first generator.'
    elif not submodule.is_sq_maxrank_HNF():
        cond = 'have square matrix, of maximal rank, in Hermite Normal Form.'
    if cond is not None:
        raise StructureError(prefix + cond)

class PrimeIdeal(IntegerPowerable):
    """
    A prime ideal in a ring of algebraic integers.
    """

    def __init__(self, ZK, p, alpha, f, e=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ==========\n\n        ZK : :py:class:`~.Submodule`\n            The maximal order where this ideal lives.\n        p : int\n            The rational prime this ideal divides.\n        alpha : :py:class:`~.PowerBasisElement`\n            Such that the ideal is equal to ``p*ZK + alpha*ZK``.\n        f : int\n            The inertia degree.\n        e : int, ``None``, optional\n            The ramification index, if already known. If ``None``, we will\n            compute it here.\n\n        '
        _check_formal_conditions_for_maximal_order(ZK)
        self.ZK = ZK
        self.p = p
        self.alpha = alpha
        self.f = f
        self._test_factor = None
        self.e = e if e is not None else self.valuation(p * ZK)

    def __str__(self):
        if False:
            while True:
                i = 10
        if self.is_inert:
            return f'({self.p})'
        return f'({self.p}, {self.alpha.as_expr()})'

    @property
    def is_inert(self):
        if False:
            return 10
        '\n        Say whether the rational prime we divide is inert, i.e. stays prime in\n        our ring of integers.\n        '
        return self.f == self.ZK.n

    def repr(self, field_gen=None, just_gens=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Print a representation of this prime ideal.\n\n        Examples\n        ========\n\n        >>> from sympy import cyclotomic_poly, QQ\n        >>> from sympy.abc import x, zeta\n        >>> T = cyclotomic_poly(7, x)\n        >>> K = QQ.algebraic_field((T, zeta))\n        >>> P = K.primes_above(11)\n        >>> print(P[0].repr())\n        [ (11, x**3 + 5*x**2 + 4*x - 1) e=1, f=3 ]\n        >>> print(P[0].repr(field_gen=zeta))\n        [ (11, zeta**3 + 5*zeta**2 + 4*zeta - 1) e=1, f=3 ]\n        >>> print(P[0].repr(field_gen=zeta, just_gens=True))\n        (11, zeta**3 + 5*zeta**2 + 4*zeta - 1)\n\n        Parameters\n        ==========\n\n        field_gen : :py:class:`~.Symbol`, ``None``, optional (default=None)\n            The symbol to use for the generator of the field. This will appear\n            in our representation of ``self.alpha``. If ``None``, we use the\n            variable of the defining polynomial of ``self.ZK``.\n        just_gens : bool, optional (default=False)\n            If ``True``, just print the "(p, alpha)" part, showing "just the\n            generators" of the prime ideal. Otherwise, print a string of the\n            form "[ (p, alpha) e=..., f=... ]", giving the ramification index\n            and inertia degree, along with the generators.\n\n        '
        field_gen = field_gen or self.ZK.parent.T.gen
        (p, alpha, e, f) = (self.p, self.alpha, self.e, self.f)
        alpha_rep = str(alpha.numerator(x=field_gen).as_expr())
        if alpha.denom > 1:
            alpha_rep = f'({alpha_rep})/{alpha.denom}'
        gens = f'({p}, {alpha_rep})'
        if just_gens:
            return gens
        return f'[ {gens} e={e}, f={f} ]'

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.repr()

    def as_submodule(self):
        if False:
            print('Hello World!')
        '\n        Represent this prime ideal as a :py:class:`~.Submodule`.\n\n        Explanation\n        ===========\n\n        The :py:class:`~.PrimeIdeal` class serves to bundle information about\n        a prime ideal, such as its inertia degree, ramification index, and\n        two-generator representation, as well as to offer helpful methods like\n        :py:meth:`~.PrimeIdeal.valuation` and\n        :py:meth:`~.PrimeIdeal.test_factor`.\n\n        However, in order to be added and multiplied by other ideals or\n        rational numbers, it must first be converted into a\n        :py:class:`~.Submodule`, which is a class that supports these\n        operations.\n\n        In many cases, the user need not perform this conversion deliberately,\n        since it is automatically performed by the arithmetic operator methods\n        :py:meth:`~.PrimeIdeal.__add__` and :py:meth:`~.PrimeIdeal.__mul__`.\n\n        Raising a :py:class:`~.PrimeIdeal` to a non-negative integer power is\n        also supported.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly, cyclotomic_poly, prime_decomp\n        >>> T = Poly(cyclotomic_poly(7))\n        >>> P0 = prime_decomp(7, T)[0]\n        >>> print(P0**6 == 7*P0.ZK)\n        True\n\n        Note that, on both sides of the equation above, we had a\n        :py:class:`~.Submodule`. In the next equation we recall that adding\n        ideals yields their GCD. This time, we need a deliberate conversion\n        to :py:class:`~.Submodule` on the right:\n\n        >>> print(P0 + 7*P0.ZK == P0.as_submodule())\n        True\n\n        Returns\n        =======\n\n        :py:class:`~.Submodule`\n            Will be equal to ``self.p * self.ZK + self.alpha * self.ZK``.\n\n        See Also\n        ========\n\n        __add__\n        __mul__\n\n        '
        M = self.p * self.ZK + self.alpha * self.ZK
        M._starts_with_unity = False
        M._is_sq_maxrank_HNF = True
        return M

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, PrimeIdeal):
            return self.as_submodule() == other.as_submodule()
        return NotImplemented

    def __add__(self, other):
        if False:
            i = 10
            return i + 15
        '\n        Convert to a :py:class:`~.Submodule` and add to another\n        :py:class:`~.Submodule`.\n\n        See Also\n        ========\n\n        as_submodule\n\n        '
        return self.as_submodule() + other
    __radd__ = __add__

    def __mul__(self, other):
        if False:
            i = 10
            return i + 15
        '\n        Convert to a :py:class:`~.Submodule` and multiply by another\n        :py:class:`~.Submodule` or a rational number.\n\n        See Also\n        ========\n\n        as_submodule\n\n        '
        return self.as_submodule() * other
    __rmul__ = __mul__

    def _zeroth_power(self):
        if False:
            return 10
        return self.ZK

    def _first_power(self):
        if False:
            print('Hello World!')
        return self

    def test_factor(self):
        if False:
            return 10
        '\n        Compute a test factor for this prime ideal.\n\n        Explanation\n        ===========\n\n        Write $\\mathfrak{p}$ for this prime ideal, $p$ for the rational prime\n        it divides. Then, for computing $\\mathfrak{p}$-adic valuations it is\n        useful to have a number $\\beta \\in \\mathbb{Z}_K$ such that\n        $p/\\mathfrak{p} = p \\mathbb{Z}_K + \\beta \\mathbb{Z}_K$.\n\n        Essentially, this is the same as the number $\\Psi$ (or the "reagent")\n        from Kummer\'s 1847 paper (*Ueber die Zerlegung...*, Crelle vol. 35) in\n        which ideal divisors were invented.\n        '
        if self._test_factor is None:
            self._test_factor = _compute_test_factor(self.p, [self.alpha], self.ZK)
        return self._test_factor

    def valuation(self, I):
        if False:
            i = 10
            return i + 15
        '\n        Compute the $\\mathfrak{p}$-adic valuation of integral ideal I at this\n        prime ideal.\n\n        Parameters\n        ==========\n\n        I : :py:class:`~.Submodule`\n\n        See Also\n        ========\n\n        prime_valuation\n\n        '
        return prime_valuation(I, self)

    def reduce_element(self, elt):
        if False:
            return 10
        '\n        Reduce a :py:class:`~.PowerBasisElement` to a "small representative"\n        modulo this prime ideal.\n\n        Parameters\n        ==========\n\n        elt : :py:class:`~.PowerBasisElement`\n            The element to be reduced.\n\n        Returns\n        =======\n\n        :py:class:`~.PowerBasisElement`\n            The reduced element.\n\n        See Also\n        ========\n\n        reduce_ANP\n        reduce_alg_num\n        .Submodule.reduce_element\n\n        '
        return self.as_submodule().reduce_element(elt)

    def reduce_ANP(self, a):
        if False:
            i = 10
            return i + 15
        '\n        Reduce an :py:class:`~.ANP` to a "small representative" modulo this\n        prime ideal.\n\n        Parameters\n        ==========\n\n        elt : :py:class:`~.ANP`\n            The element to be reduced.\n\n        Returns\n        =======\n\n        :py:class:`~.ANP`\n            The reduced element.\n\n        See Also\n        ========\n\n        reduce_element\n        reduce_alg_num\n        .Submodule.reduce_element\n\n        '
        elt = self.ZK.parent.element_from_ANP(a)
        red = self.reduce_element(elt)
        return red.to_ANP()

    def reduce_alg_num(self, a):
        if False:
            for i in range(10):
                print('nop')
        '\n        Reduce an :py:class:`~.AlgebraicNumber` to a "small representative"\n        modulo this prime ideal.\n\n        Parameters\n        ==========\n\n        elt : :py:class:`~.AlgebraicNumber`\n            The element to be reduced.\n\n        Returns\n        =======\n\n        :py:class:`~.AlgebraicNumber`\n            The reduced element.\n\n        See Also\n        ========\n\n        reduce_element\n        reduce_ANP\n        .Submodule.reduce_element\n\n        '
        elt = self.ZK.parent.element_from_alg_num(a)
        red = self.reduce_element(elt)
        return a.field_element(list(reversed(red.QQ_col.flat())))

def _compute_test_factor(p, gens, ZK):
    if False:
        while True:
            i = 10
    '\n    Compute the test factor for a :py:class:`~.PrimeIdeal` $\\mathfrak{p}$.\n\n    Parameters\n    ==========\n\n    p : int\n        The rational prime $\\mathfrak{p}$ divides\n\n    gens : list of :py:class:`PowerBasisElement`\n        A complete set of generators for $\\mathfrak{p}$ over *ZK*, EXCEPT that\n        an element equivalent to rational *p* can and should be omitted (since\n        it has no effect except to waste time).\n\n    ZK : :py:class:`~.Submodule`\n        The maximal order where the prime ideal $\\mathfrak{p}$ lives.\n\n    Returns\n    =======\n\n    :py:class:`~.PowerBasisElement`\n\n    References\n    ==========\n\n    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*\n    (See Proposition 4.8.15.)\n\n    '
    _check_formal_conditions_for_maximal_order(ZK)
    E = ZK.endomorphism_ring()
    matrices = [E.inner_endomorphism(g).matrix(modulus=p) for g in gens]
    B = DomainMatrix.zeros((0, ZK.n), FF(p)).vstack(*matrices)
    x = B.nullspace()[0, :].transpose()
    beta = ZK.parent(ZK.matrix * x.convert_to(ZZ), denom=ZK.denom)
    return beta

@public
def prime_valuation(I, P):
    if False:
        return 10
    '\n    Compute the *P*-adic valuation for an integral ideal *I*.\n\n    Examples\n    ========\n\n    >>> from sympy import QQ\n    >>> from sympy.polys.numberfields import prime_valuation\n    >>> K = QQ.cyclotomic_field(5)\n    >>> P = K.primes_above(5)\n    >>> ZK = K.maximal_order()\n    >>> print(prime_valuation(25*ZK, P[0]))\n    8\n\n    Parameters\n    ==========\n\n    I : :py:class:`~.Submodule`\n        An integral ideal whose valuation is desired.\n\n    P : :py:class:`~.PrimeIdeal`\n        The prime at which to compute the valuation.\n\n    Returns\n    =======\n\n    int\n\n    See Also\n    ========\n\n    .PrimeIdeal.valuation\n\n    References\n    ==========\n\n    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*\n       (See Algorithm 4.8.17.)\n\n    '
    (p, ZK) = (P.p, P.ZK)
    (n, W, d) = (ZK.n, ZK.matrix, ZK.denom)
    A = W.convert_to(QQ).inv() * I.matrix * d / I.denom
    A = A.convert_to(ZZ)
    D = A.det()
    if D % p != 0:
        return 0
    beta = P.test_factor()
    f = d ** n // W.det()
    need_complete_test = f % p == 0
    v = 0
    while True:
        A = W * A
        for j in range(n):
            c = ZK.parent(A[:, j], denom=d)
            c *= beta
            c = ZK.represent(c).flat()
            for i in range(n):
                A[i, j] = c[i]
        if A[n - 1, n - 1].element % p != 0:
            break
        A = A / p
        if need_complete_test:
            try:
                A = A.convert_to(ZZ)
            except CoercionFailed:
                break
        else:
            A = A.convert_to(ZZ)
        v += 1
    return v

def _two_elt_rep(gens, ZK, p, f=None, Np=None):
    if False:
        return 10
    '\n    Given a set of *ZK*-generators of a prime ideal, compute a set of just two\n    *ZK*-generators for the same ideal, one of which is *p* itself.\n\n    Parameters\n    ==========\n\n    gens : list of :py:class:`PowerBasisElement`\n        Generators for the prime ideal over *ZK*, the ring of integers of the\n        field $K$.\n\n    ZK : :py:class:`~.Submodule`\n        The maximal order in $K$.\n\n    p : int\n        The rational prime divided by the prime ideal.\n\n    f : int, optional\n        The inertia degree of the prime ideal, if known.\n\n    Np : int, optional\n        The norm $p^f$ of the prime ideal, if known.\n        NOTE: There is no reason to supply both *f* and *Np*. Either one will\n        save us from having to compute the norm *Np* ourselves. If both are known,\n        *Np* is preferred since it saves one exponentiation.\n\n    Returns\n    =======\n\n    :py:class:`~.PowerBasisElement` representing a single algebraic integer\n    alpha such that the prime ideal is equal to ``p*ZK + alpha*ZK``.\n\n    References\n    ==========\n\n    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*\n    (See Algorithm 4.7.10.)\n\n    '
    _check_formal_conditions_for_maximal_order(ZK)
    pb = ZK.parent
    T = pb.T
    if all(((g % p).equiv(0) for g in gens)):
        return pb.zero()
    if Np is None:
        if f is not None:
            Np = p ** f
        else:
            Np = abs(pb.submodule_from_gens(gens).matrix.det())
    omega = ZK.basis_element_pullbacks()
    beta = [p * om for om in omega[1:]]
    beta += gens
    search = coeff_search(len(beta), 1)
    for c in search:
        alpha = sum((ci * betai for (ci, betai) in zip(c, beta)))
        n = alpha.norm(T) // Np
        if n % p != 0:
            return alpha % p

def _prime_decomp_easy_case(p, ZK):
    if False:
        i = 10
        return i + 15
    '\n    Compute the decomposition of rational prime *p* in the ring of integers\n    *ZK* (given as a :py:class:`~.Submodule`), in the "easy case", i.e. the\n    case where *p* does not divide the index of $\\theta$ in *ZK*, where\n    $\\theta$ is the generator of the ``PowerBasis`` of which *ZK* is a\n    ``Submodule``.\n    '
    T = ZK.parent.T
    T_bar = Poly(T, modulus=p)
    (lc, fl) = T_bar.factor_list()
    if len(fl) == 1 and fl[0][1] == 1:
        return [PrimeIdeal(ZK, p, ZK.parent.zero(), ZK.n, 1)]
    return [PrimeIdeal(ZK, p, ZK.parent.element_from_poly(Poly(t, domain=ZZ)), t.degree(), e) for (t, e) in fl]

def _prime_decomp_compute_kernel(I, p, ZK):
    if False:
        print('Hello World!')
    '\n    Parameters\n    ==========\n\n    I : :py:class:`~.Module`\n        An ideal of ``ZK/pZK``.\n    p : int\n        The rational prime being factored.\n    ZK : :py:class:`~.Submodule`\n        The maximal order.\n\n    Returns\n    =======\n\n    Pair ``(N, G)``, where:\n\n        ``N`` is a :py:class:`~.Module` representing the kernel of the map\n        ``a |--> a**p - a`` on ``(O/pO)/I``, guaranteed to be a module with\n        unity.\n\n        ``G`` is a :py:class:`~.Module` representing a basis for the separable\n        algebra ``A = O/I`` (see Cohen).\n\n    '
    W = I.matrix
    (n, r) = W.shape
    if r == 0:
        B = W.eye(n, ZZ)
    else:
        B = W.hstack(W.eye(n, ZZ)[:, 0])
    if B.shape[1] < n:
        B = supplement_a_subspace(B.convert_to(FF(p))).convert_to(ZZ)
    G = ZK.submodule_from_matrix(B)
    G.compute_mult_tab()
    G = G.discard_before(r)
    phi = ModuleEndomorphism(G, lambda x: x ** p - x)
    N = phi.kernel(modulus=p)
    assert N.starts_with_unity()
    return (N, G)

def _prime_decomp_maximal_ideal(I, p, ZK):
    if False:
        print('Hello World!')
    '\n    We have reached the case where we have a maximal (hence prime) ideal *I*,\n    which we know because the quotient ``O/I`` is a field.\n\n    Parameters\n    ==========\n\n    I : :py:class:`~.Module`\n        An ideal of ``O/pO``.\n    p : int\n        The rational prime being factored.\n    ZK : :py:class:`~.Submodule`\n        The maximal order.\n\n    Returns\n    =======\n\n    :py:class:`~.PrimeIdeal` instance representing this prime\n\n    '
    (m, n) = I.matrix.shape
    f = m - n
    G = ZK.matrix * I.matrix
    gens = [ZK.parent(G[:, j], denom=ZK.denom) for j in range(G.shape[1])]
    alpha = _two_elt_rep(gens, ZK, p, f=f)
    return PrimeIdeal(ZK, p, alpha, f)

def _prime_decomp_split_ideal(I, p, N, G, ZK):
    if False:
        return 10
    '\n    Perform the step in the prime decomposition algorithm where we have determined\n    the the quotient ``ZK/I`` is _not_ a field, and we want to perform a non-trivial\n    factorization of *I* by locating an idempotent element of ``ZK/I``.\n    '
    assert I.parent == ZK and G.parent is ZK and (N.parent is G)
    alpha = N(1).to_parent()
    assert alpha.module is G
    alpha_powers = []
    m = find_min_poly(alpha, FF(p), powers=alpha_powers)
    (lc, fl) = m.factor_list()
    m1 = fl[0][0]
    m2 = m.quo(m1)
    (U, V, g) = m1.gcdex(m2)
    assert g == 1
    E = list(reversed(Poly(U * m1, domain=ZZ).rep.to_list()))
    eps1 = sum((E[i] * alpha_powers[i] for i in range(len(E))))
    eps2 = 1 - eps1
    idemps = [eps1, eps2]
    factors = []
    for eps in idemps:
        e = eps.to_parent()
        assert e.module is ZK
        D = I.matrix.convert_to(FF(p)).hstack(*[(e * om).column(domain=FF(p)) for om in ZK.basis_elements()])
        W = D.columnspace().convert_to(ZZ)
        H = ZK.submodule_from_matrix(W)
        factors.append(H)
    return factors

@public
def prime_decomp(p, T=None, ZK=None, dK=None, radical=None):
    if False:
        i = 10
        return i + 15
    '\n    Compute the decomposition of rational prime *p* in a number field.\n\n    Explanation\n    ===========\n\n    Ordinarily this should be accessed through the\n    :py:meth:`~.AlgebraicField.primes_above` method of an\n    :py:class:`~.AlgebraicField`.\n\n    Examples\n    ========\n\n    >>> from sympy import Poly, QQ\n    >>> from sympy.abc import x, theta\n    >>> T = Poly(x ** 3 + x ** 2 - 2 * x + 8)\n    >>> K = QQ.algebraic_field((T, theta))\n    >>> print(K.primes_above(2))\n    [[ (2, x**2 + 1) e=1, f=1 ], [ (2, (x**2 + 3*x + 2)/2) e=1, f=1 ],\n     [ (2, (3*x**2 + 3*x)/2) e=1, f=1 ]]\n\n    Parameters\n    ==========\n\n    p : int\n        The rational prime whose decomposition is desired.\n\n    T : :py:class:`~.Poly`, optional\n        Monic irreducible polynomial defining the number field $K$ in which to\n        factor. NOTE: at least one of *T* or *ZK* must be provided.\n\n    ZK : :py:class:`~.Submodule`, optional\n        The maximal order for $K$, if already known.\n        NOTE: at least one of *T* or *ZK* must be provided.\n\n    dK : int, optional\n        The discriminant of the field $K$, if already known.\n\n    radical : :py:class:`~.Submodule`, optional\n        The nilradical mod *p* in the integers of $K$, if already known.\n\n    Returns\n    =======\n\n    List of :py:class:`~.PrimeIdeal` instances.\n\n    References\n    ==========\n\n    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*\n       (See Algorithm 6.2.9.)\n\n    '
    if T is None and ZK is None:
        raise ValueError('At least one of T or ZK must be provided.')
    if ZK is not None:
        _check_formal_conditions_for_maximal_order(ZK)
    if T is None:
        T = ZK.parent.T
    radicals = {}
    if dK is None or ZK is None:
        (ZK, dK) = round_two(T, radicals=radicals)
    dT = T.discriminant()
    f_squared = dT // dK
    if f_squared % p != 0:
        return _prime_decomp_easy_case(p, ZK)
    radical = radical or radicals.get(p) or nilradical_mod_p(ZK, p)
    stack = [radical]
    primes = []
    while stack:
        I = stack.pop()
        (N, G) = _prime_decomp_compute_kernel(I, p, ZK)
        if N.n == 1:
            P = _prime_decomp_maximal_ideal(I, p, ZK)
            primes.append(P)
        else:
            (I1, I2) = _prime_decomp_split_ideal(I, p, N, G, ZK)
            stack.extend([I1, I2])
    return primes