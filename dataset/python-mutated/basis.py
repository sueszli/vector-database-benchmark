"""Computing integral bases for number fields. """
from sympy.polys.polytools import Poly
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.utilities.decorator import public
from .modules import ModuleEndomorphism, ModuleHomomorphism, PowerBasis
from .utilities import extract_fundamental_discriminant

def _apply_Dedekind_criterion(T, p):
    if False:
        while True:
            i = 10
    '\n    Apply the "Dedekind criterion" to test whether the order needs to be\n    enlarged relative to a given prime *p*.\n    '
    x = T.gen
    T_bar = Poly(T, modulus=p)
    (lc, fl) = T_bar.factor_list()
    assert lc == 1
    g_bar = Poly(1, x, modulus=p)
    for (ti_bar, _) in fl:
        g_bar *= ti_bar
    h_bar = T_bar // g_bar
    g = Poly(g_bar, domain=ZZ)
    h = Poly(h_bar, domain=ZZ)
    f = (g * h - T) // p
    f_bar = Poly(f, modulus=p)
    Z_bar = f_bar
    for b in [g_bar, h_bar]:
        Z_bar = Z_bar.gcd(b)
    U_bar = T_bar // Z_bar
    m = Z_bar.degree()
    return (U_bar, m)

def nilradical_mod_p(H, p, q=None):
    if False:
        while True:
            i = 10
    '\n    Compute the nilradical mod *p* for a given order *H*, and prime *p*.\n\n    Explanation\n    ===========\n\n    This is the ideal $I$ in $H/pH$ consisting of all elements some positive\n    power of which is zero in this quotient ring, i.e. is a multiple of *p*.\n\n    Parameters\n    ==========\n\n    H : :py:class:`~.Submodule`\n        The given order.\n    p : int\n        The rational prime.\n    q : int, optional\n        If known, the smallest power of *p* that is $>=$ the dimension of *H*.\n        If not provided, we compute it here.\n\n    Returns\n    =======\n\n    :py:class:`~.Module` representing the nilradical mod *p* in *H*.\n\n    References\n    ==========\n\n    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory*.\n    (See Lemma 6.1.6.)\n\n    '
    n = H.n
    if q is None:
        q = p
        while q < n:
            q *= p
    phi = ModuleEndomorphism(H, lambda x: x ** q)
    return phi.kernel(modulus=p)

def _second_enlargement(H, p, q):
    if False:
        return 10
    '\n    Perform the second enlargement in the Round Two algorithm.\n    '
    Ip = nilradical_mod_p(H, p, q=q)
    B = H.parent.submodule_from_matrix(H.matrix * Ip.matrix, denom=H.denom)
    C = B + p * H
    E = C.endomorphism_ring()
    phi = ModuleHomomorphism(H, E, lambda x: E.inner_endomorphism(x))
    gamma = phi.kernel(modulus=p)
    G = H.parent.submodule_from_matrix(H.matrix * gamma.matrix, denom=H.denom * p)
    H1 = G + H
    return (H1, Ip)

@public
def round_two(T, radicals=None):
    if False:
        i = 10
        return i + 15
    '\n    Zassenhaus\'s "Round 2" algorithm.\n\n    Explanation\n    ===========\n\n    Carry out Zassenhaus\'s "Round 2" algorithm on an irreducible polynomial\n    *T* over :ref:`ZZ` or :ref:`QQ`. This computes an integral basis and the\n    discriminant for the field $K = \\mathbb{Q}[x]/(T(x))$.\n\n    Alternatively, you may pass an :py:class:`~.AlgebraicField` instance, in\n    place of the polynomial *T*, in which case the algorithm is applied to the\n    minimal polynomial for the field\'s primitive element.\n\n    Ordinarily this function need not be called directly, as one can instead\n    access the :py:meth:`~.AlgebraicField.maximal_order`,\n    :py:meth:`~.AlgebraicField.integral_basis`, and\n    :py:meth:`~.AlgebraicField.discriminant` methods of an\n    :py:class:`~.AlgebraicField`.\n\n    Examples\n    ========\n\n    Working through an AlgebraicField:\n\n    >>> from sympy import Poly, QQ\n    >>> from sympy.abc import x\n    >>> T = Poly(x ** 3 + x ** 2 - 2 * x + 8)\n    >>> K = QQ.alg_field_from_poly(T, "theta")\n    >>> print(K.maximal_order())\n    Submodule[[2, 0, 0], [0, 2, 0], [0, 1, 1]]/2\n    >>> print(K.discriminant())\n    -503\n    >>> print(K.integral_basis(fmt=\'sympy\'))\n    [1, theta, theta/2 + theta**2/2]\n\n    Calling directly:\n\n    >>> from sympy import Poly\n    >>> from sympy.abc import x\n    >>> from sympy.polys.numberfields.basis import round_two\n    >>> T = Poly(x ** 3 + x ** 2 - 2 * x + 8)\n    >>> print(round_two(T))\n    (Submodule[[2, 0, 0], [0, 2, 0], [0, 1, 1]]/2, -503)\n\n    The nilradicals mod $p$ that are sometimes computed during the Round Two\n    algorithm may be useful in further calculations. Pass a dictionary under\n    `radicals` to receive these:\n\n    >>> T = Poly(x**3 + 3*x**2 + 5)\n    >>> rad = {}\n    >>> ZK, dK = round_two(T, radicals=rad)\n    >>> print(rad)\n    {3: Submodule[[-1, 1, 0], [-1, 0, 1]]}\n\n    Parameters\n    ==========\n\n    T : :py:class:`~.Poly`, :py:class:`~.AlgebraicField`\n        Either (1) the irreducible polynomial over :ref:`ZZ` or :ref:`QQ`\n        defining the number field, or (2) an :py:class:`~.AlgebraicField`\n        representing the number field itself.\n\n    radicals : dict, optional\n        This is a way for any $p$-radicals (if computed) to be returned by\n        reference. If desired, pass an empty dictionary. If the algorithm\n        reaches the point where it computes the nilradical mod $p$ of the ring\n        of integers $Z_K$, then an $\\mathbb{F}_p$-basis for this ideal will be\n        stored in this dictionary under the key ``p``. This can be useful for\n        other algorithms, such as prime decomposition.\n\n    Returns\n    =======\n\n    Pair ``(ZK, dK)``, where:\n\n        ``ZK`` is a :py:class:`~sympy.polys.numberfields.modules.Submodule`\n        representing the maximal order.\n\n        ``dK`` is the discriminant of the field $K = \\mathbb{Q}[x]/(T(x))$.\n\n    See Also\n    ========\n\n    .AlgebraicField.maximal_order\n    .AlgebraicField.integral_basis\n    .AlgebraicField.discriminant\n\n    References\n    ==========\n\n    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*\n\n    '
    K = None
    if isinstance(T, AlgebraicField):
        (K, T) = (T, T.ext.minpoly_of_element())
    if not T.is_univariate or not T.is_irreducible or T.domain not in [ZZ, QQ]:
        raise ValueError('Round 2 requires an irreducible univariate polynomial over ZZ or QQ.')
    (T, _) = T.make_monic_over_integers_by_scaling_roots()
    n = T.degree()
    D = T.discriminant()
    D_modulus = ZZ.from_sympy(abs(D))
    (_, F) = extract_fundamental_discriminant(D)
    Ztheta = PowerBasis(K or T)
    H = Ztheta.whole_submodule()
    nilrad = None
    while F:
        (p, e) = F.popitem()
        (U_bar, m) = _apply_Dedekind_criterion(T, p)
        if m == 0:
            continue
        U = Ztheta.element_from_poly(Poly(U_bar, domain=ZZ))
        H = H.add(U // p * H, hnf_modulus=D_modulus)
        if e <= m:
            continue
        q = p
        while q < n:
            q *= p
        (H1, nilrad) = _second_enlargement(H, p, q)
        while H1 != H:
            H = H1
            (H1, nilrad) = _second_enlargement(H, p, q)
    if nilrad is not None and isinstance(radicals, dict):
        radicals[p] = nilrad
    ZK = H
    ZK._starts_with_unity = True
    ZK._is_sq_maxrank_HNF = True
    dK = D * ZK.matrix.det() ** 2 // ZK.denom ** (2 * n)
    return (ZK, dK)