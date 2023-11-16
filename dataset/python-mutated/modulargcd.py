from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import gf_gcd, gf_from_dict, gf_gcdex, gf_div, gf_lcm
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random

def _trivial_gcd(f, g):
    if False:
        i = 10
        return i + 15
    '\n    Compute the GCD of two polynomials in trivial cases, i.e. when one\n    or both polynomials are zero.\n    '
    ring = f.ring
    if not (f or g):
        return (ring.zero, ring.zero, ring.zero)
    elif not f:
        if g.LC < ring.domain.zero:
            return (-g, ring.zero, -ring.one)
        else:
            return (g, ring.zero, ring.one)
    elif not g:
        if f.LC < ring.domain.zero:
            return (-f, -ring.one, ring.zero)
        else:
            return (f, ring.one, ring.zero)
    return None

def _gf_gcd(fp, gp, p):
    if False:
        print('Hello World!')
    '\n    Compute the GCD of two univariate polynomials in `\\mathbb{Z}_p[x]`.\n    '
    dom = fp.ring.domain
    while gp:
        rem = fp
        deg = gp.degree()
        lcinv = dom.invert(gp.LC, p)
        while True:
            degrem = rem.degree()
            if degrem < deg:
                break
            rem = (rem - gp.mul_monom((degrem - deg,)).mul_ground(lcinv * rem.LC)).trunc_ground(p)
        fp = gp
        gp = rem
    return fp.mul_ground(dom.invert(fp.LC, p)).trunc_ground(p)

def _degree_bound_univariate(f, g):
    if False:
        return 10
    '\n    Compute an upper bound for the degree of the GCD of two univariate\n    integer polynomials `f` and `g`.\n\n    The function chooses a suitable prime `p` and computes the GCD of\n    `f` and `g` in `\\mathbb{Z}_p[x]`. The choice of `p` guarantees that\n    the degree in `\\mathbb{Z}_p[x]` is greater than or equal to the degree\n    in `\\mathbb{Z}[x]`.\n\n    Parameters\n    ==========\n\n    f : PolyElement\n        univariate integer polynomial\n    g : PolyElement\n        univariate integer polynomial\n\n    '
    gamma = f.ring.domain.gcd(f.LC, g.LC)
    p = 1
    p = nextprime(p)
    while gamma % p == 0:
        p = nextprime(p)
    fp = f.trunc_ground(p)
    gp = g.trunc_ground(p)
    hp = _gf_gcd(fp, gp, p)
    deghp = hp.degree()
    return deghp

def _chinese_remainder_reconstruction_univariate(hp, hq, p, q):
    if False:
        return 10
    '\n    Construct a polynomial `h_{pq}` in `\\mathbb{Z}_{p q}[x]` such that\n\n    .. math ::\n\n        h_{pq} = h_p \\; \\mathrm{mod} \\, p\n\n        h_{pq} = h_q \\; \\mathrm{mod} \\, q\n\n    for relatively prime integers `p` and `q` and polynomials\n    `h_p` and `h_q` in `\\mathbb{Z}_p[x]` and `\\mathbb{Z}_q[x]`\n    respectively.\n\n    The coefficients of the polynomial `h_{pq}` are computed with the\n    Chinese Remainder Theorem. The symmetric representation in\n    `\\mathbb{Z}_p[x]`, `\\mathbb{Z}_q[x]` and `\\mathbb{Z}_{p q}[x]` is used.\n    It is assumed that `h_p` and `h_q` have the same degree.\n\n    Parameters\n    ==========\n\n    hp : PolyElement\n        univariate integer polynomial with coefficients in `\\mathbb{Z}_p`\n    hq : PolyElement\n        univariate integer polynomial with coefficients in `\\mathbb{Z}_q`\n    p : Integer\n        modulus of `h_p`, relatively prime to `q`\n    q : Integer\n        modulus of `h_q`, relatively prime to `p`\n\n    Examples\n    ========\n\n    >>> from sympy.polys.modulargcd import _chinese_remainder_reconstruction_univariate\n    >>> from sympy.polys import ring, ZZ\n\n    >>> R, x = ring("x", ZZ)\n    >>> p = 3\n    >>> q = 5\n\n    >>> hp = -x**3 - 1\n    >>> hq = 2*x**3 - 2*x**2 + x\n\n    >>> hpq = _chinese_remainder_reconstruction_univariate(hp, hq, p, q)\n    >>> hpq\n    2*x**3 + 3*x**2 + 6*x + 5\n\n    >>> hpq.trunc_ground(p) == hp\n    True\n    >>> hpq.trunc_ground(q) == hq\n    True\n\n    '
    n = hp.degree()
    x = hp.ring.gens[0]
    hpq = hp.ring.zero
    for i in range(n + 1):
        hpq[i,] = crt([p, q], [hp.coeff(x ** i), hq.coeff(x ** i)], symmetric=True)[0]
    hpq.strip_zero()
    return hpq

def modgcd_univariate(f, g):
    if False:
        for i in range(10):
            print('nop')
    '\n    Computes the GCD of two polynomials in `\\mathbb{Z}[x]` using a modular\n    algorithm.\n\n    The algorithm computes the GCD of two univariate integer polynomials\n    `f` and `g` by computing the GCD in `\\mathbb{Z}_p[x]` for suitable\n    primes `p` and then reconstructing the coefficients with the Chinese\n    Remainder Theorem. Trial division is only made for candidates which\n    are very likely the desired GCD.\n\n    Parameters\n    ==========\n\n    f : PolyElement\n        univariate integer polynomial\n    g : PolyElement\n        univariate integer polynomial\n\n    Returns\n    =======\n\n    h : PolyElement\n        GCD of the polynomials `f` and `g`\n    cff : PolyElement\n        cofactor of `f`, i.e. `\\frac{f}{h}`\n    cfg : PolyElement\n        cofactor of `g`, i.e. `\\frac{g}{h}`\n\n    Examples\n    ========\n\n    >>> from sympy.polys.modulargcd import modgcd_univariate\n    >>> from sympy.polys import ring, ZZ\n\n    >>> R, x = ring("x", ZZ)\n\n    >>> f = x**5 - 1\n    >>> g = x - 1\n\n    >>> h, cff, cfg = modgcd_univariate(f, g)\n    >>> h, cff, cfg\n    (x - 1, x**4 + x**3 + x**2 + x + 1, 1)\n\n    >>> cff * h == f\n    True\n    >>> cfg * h == g\n    True\n\n    >>> f = 6*x**2 - 6\n    >>> g = 2*x**2 + 4*x + 2\n\n    >>> h, cff, cfg = modgcd_univariate(f, g)\n    >>> h, cff, cfg\n    (2*x + 2, 3*x - 3, x + 1)\n\n    >>> cff * h == f\n    True\n    >>> cfg * h == g\n    True\n\n    References\n    ==========\n\n    1. [Monagan00]_\n\n    '
    assert f.ring == g.ring and f.ring.domain.is_ZZ
    result = _trivial_gcd(f, g)
    if result is not None:
        return result
    ring = f.ring
    (cf, f) = f.primitive()
    (cg, g) = g.primitive()
    ch = ring.domain.gcd(cf, cg)
    bound = _degree_bound_univariate(f, g)
    if bound == 0:
        return (ring(ch), f.mul_ground(cf // ch), g.mul_ground(cg // ch))
    gamma = ring.domain.gcd(f.LC, g.LC)
    m = 1
    p = 1
    while True:
        p = nextprime(p)
        while gamma % p == 0:
            p = nextprime(p)
        fp = f.trunc_ground(p)
        gp = g.trunc_ground(p)
        hp = _gf_gcd(fp, gp, p)
        deghp = hp.degree()
        if deghp > bound:
            continue
        elif deghp < bound:
            m = 1
            bound = deghp
            continue
        hp = hp.mul_ground(gamma).trunc_ground(p)
        if m == 1:
            m = p
            hlastm = hp
            continue
        hm = _chinese_remainder_reconstruction_univariate(hp, hlastm, p, m)
        m *= p
        if not hm == hlastm:
            hlastm = hm
            continue
        h = hm.quo_ground(hm.content())
        (fquo, frem) = f.div(h)
        (gquo, grem) = g.div(h)
        if not frem and (not grem):
            if h.LC < 0:
                ch = -ch
            h = h.mul_ground(ch)
            cff = fquo.mul_ground(cf // ch)
            cfg = gquo.mul_ground(cg // ch)
            return (h, cff, cfg)

def _primitive(f, p):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the content and the primitive part of a polynomial in\n    `\\mathbb{Z}_p[x_0, \\ldots, x_{k-2}, y] \\cong \\mathbb{Z}_p[y][x_0, \\ldots, x_{k-2}]`.\n\n    Parameters\n    ==========\n\n    f : PolyElement\n        integer polynomial in `\\mathbb{Z}_p[x0, \\ldots, x{k-2}, y]`\n    p : Integer\n        modulus of `f`\n\n    Returns\n    =======\n\n    contf : PolyElement\n        integer polynomial in `\\mathbb{Z}_p[y]`, content of `f`\n    ppf : PolyElement\n        primitive part of `f`, i.e. `\\frac{f}{contf}`\n\n    Examples\n    ========\n\n    >>> from sympy.polys.modulargcd import _primitive\n    >>> from sympy.polys import ring, ZZ\n\n    >>> R, x, y = ring("x, y", ZZ)\n    >>> p = 3\n\n    >>> f = x**2*y**2 + x**2*y - y**2 - y\n    >>> _primitive(f, p)\n    (y**2 + y, x**2 - 1)\n\n    >>> R, x, y, z = ring("x, y, z", ZZ)\n\n    >>> f = x*y*z - y**2*z**2\n    >>> _primitive(f, p)\n    (z, x*y - y**2*z)\n\n    '
    ring = f.ring
    dom = ring.domain
    k = ring.ngens
    coeffs = {}
    for (monom, coeff) in f.iterterms():
        if monom[:-1] not in coeffs:
            coeffs[monom[:-1]] = {}
        coeffs[monom[:-1]][monom[-1]] = coeff
    cont = []
    for coeff in iter(coeffs.values()):
        cont = gf_gcd(cont, gf_from_dict(coeff, p, dom), p, dom)
    yring = ring.clone(symbols=ring.symbols[k - 1])
    contf = yring.from_dense(cont).trunc_ground(p)
    return (contf, f.quo(contf.set_ring(ring)))

def _deg(f):
    if False:
        i = 10
        return i + 15
    '\n    Compute the degree of a multivariate polynomial\n    `f \\in K[x_0, \\ldots, x_{k-2}, y] \\cong K[y][x_0, \\ldots, x_{k-2}]`.\n\n    Parameters\n    ==========\n\n    f : PolyElement\n        polynomial in `K[x_0, \\ldots, x_{k-2}, y]`\n\n    Returns\n    =======\n\n    degf : Integer tuple\n        degree of `f` in `x_0, \\ldots, x_{k-2}`\n\n    Examples\n    ========\n\n    >>> from sympy.polys.modulargcd import _deg\n    >>> from sympy.polys import ring, ZZ\n\n    >>> R, x, y = ring("x, y", ZZ)\n\n    >>> f = x**2*y**2 + x**2*y - 1\n    >>> _deg(f)\n    (2,)\n\n    >>> R, x, y, z = ring("x, y, z", ZZ)\n\n    >>> f = x**2*y**2 + x**2*y - 1\n    >>> _deg(f)\n    (2, 2)\n\n    >>> f = x*y*z - y**2*z**2\n    >>> _deg(f)\n    (1, 1)\n\n    '
    k = f.ring.ngens
    degf = (0,) * (k - 1)
    for monom in f.itermonoms():
        if monom[:-1] > degf:
            degf = monom[:-1]
    return degf

def _LC(f):
    if False:
        print('Hello World!')
    '\n    Compute the leading coefficient of a multivariate polynomial\n    `f \\in K[x_0, \\ldots, x_{k-2}, y] \\cong K[y][x_0, \\ldots, x_{k-2}]`.\n\n    Parameters\n    ==========\n\n    f : PolyElement\n        polynomial in `K[x_0, \\ldots, x_{k-2}, y]`\n\n    Returns\n    =======\n\n    lcf : PolyElement\n        polynomial in `K[y]`, leading coefficient of `f`\n\n    Examples\n    ========\n\n    >>> from sympy.polys.modulargcd import _LC\n    >>> from sympy.polys import ring, ZZ\n\n    >>> R, x, y = ring("x, y", ZZ)\n\n    >>> f = x**2*y**2 + x**2*y - 1\n    >>> _LC(f)\n    y**2 + y\n\n    >>> R, x, y, z = ring("x, y, z", ZZ)\n\n    >>> f = x**2*y**2 + x**2*y - 1\n    >>> _LC(f)\n    1\n\n    >>> f = x*y*z - y**2*z**2\n    >>> _LC(f)\n    z\n\n    '
    ring = f.ring
    k = ring.ngens
    yring = ring.clone(symbols=ring.symbols[k - 1])
    y = yring.gens[0]
    degf = _deg(f)
    lcf = yring.zero
    for (monom, coeff) in f.iterterms():
        if monom[:-1] == degf:
            lcf += coeff * y ** monom[-1]
    return lcf

def _swap(f, i):
    if False:
        i = 10
        return i + 15
    '\n    Make the variable `x_i` the leading one in a multivariate polynomial `f`.\n    '
    ring = f.ring
    fswap = ring.zero
    for (monom, coeff) in f.iterterms():
        monomswap = (monom[i],) + monom[:i] + monom[i + 1:]
        fswap[monomswap] = coeff
    return fswap

def _degree_bound_bivariate(f, g):
    if False:
        while True:
            i = 10
    '\n    Compute upper degree bounds for the GCD of two bivariate\n    integer polynomials `f` and `g`.\n\n    The GCD is viewed as a polynomial in `\\mathbb{Z}[y][x]` and the\n    function returns an upper bound for its degree and one for the degree\n    of its content. This is done by choosing a suitable prime `p` and\n    computing the GCD of the contents of `f \\; \\mathrm{mod} \\, p` and\n    `g \\; \\mathrm{mod} \\, p`. The choice of `p` guarantees that the degree\n    of the content in `\\mathbb{Z}_p[y]` is greater than or equal to the\n    degree in `\\mathbb{Z}[y]`. To obtain the degree bound in the variable\n    `x`, the polynomials are evaluated at `y = a` for a suitable\n    `a \\in \\mathbb{Z}_p` and then their GCD in `\\mathbb{Z}_p[x]` is\n    computed. If no such `a` exists, i.e. the degree in `\\mathbb{Z}_p[x]`\n    is always smaller than the one in `\\mathbb{Z}[y][x]`, then the bound is\n    set to the minimum of the degrees of `f` and `g` in `x`.\n\n    Parameters\n    ==========\n\n    f : PolyElement\n        bivariate integer polynomial\n    g : PolyElement\n        bivariate integer polynomial\n\n    Returns\n    =======\n\n    xbound : Integer\n        upper bound for the degree of the GCD of the polynomials `f` and\n        `g` in the variable `x`\n    ycontbound : Integer\n        upper bound for the degree of the content of the GCD of the\n        polynomials `f` and `g` in the variable `y`\n\n    References\n    ==========\n\n    1. [Monagan00]_\n\n    '
    ring = f.ring
    gamma1 = ring.domain.gcd(f.LC, g.LC)
    gamma2 = ring.domain.gcd(_swap(f, 1).LC, _swap(g, 1).LC)
    badprimes = gamma1 * gamma2
    p = 1
    p = nextprime(p)
    while badprimes % p == 0:
        p = nextprime(p)
    fp = f.trunc_ground(p)
    gp = g.trunc_ground(p)
    (contfp, fp) = _primitive(fp, p)
    (contgp, gp) = _primitive(gp, p)
    conthp = _gf_gcd(contfp, contgp, p)
    ycontbound = conthp.degree()
    delta = _gf_gcd(_LC(fp), _LC(gp), p)
    for a in range(p):
        if not delta.evaluate(0, a) % p:
            continue
        fpa = fp.evaluate(1, a).trunc_ground(p)
        gpa = gp.evaluate(1, a).trunc_ground(p)
        hpa = _gf_gcd(fpa, gpa, p)
        xbound = hpa.degree()
        return (xbound, ycontbound)
    return (min(fp.degree(), gp.degree()), ycontbound)

def _chinese_remainder_reconstruction_multivariate(hp, hq, p, q):
    if False:
        print('Hello World!')
    '\n    Construct a polynomial `h_{pq}` in\n    `\\mathbb{Z}_{p q}[x_0, \\ldots, x_{k-1}]` such that\n\n    .. math ::\n\n        h_{pq} = h_p \\; \\mathrm{mod} \\, p\n\n        h_{pq} = h_q \\; \\mathrm{mod} \\, q\n\n    for relatively prime integers `p` and `q` and polynomials\n    `h_p` and `h_q` in `\\mathbb{Z}_p[x_0, \\ldots, x_{k-1}]` and\n    `\\mathbb{Z}_q[x_0, \\ldots, x_{k-1}]` respectively.\n\n    The coefficients of the polynomial `h_{pq}` are computed with the\n    Chinese Remainder Theorem. The symmetric representation in\n    `\\mathbb{Z}_p[x_0, \\ldots, x_{k-1}]`,\n    `\\mathbb{Z}_q[x_0, \\ldots, x_{k-1}]` and\n    `\\mathbb{Z}_{p q}[x_0, \\ldots, x_{k-1}]` is used.\n\n    Parameters\n    ==========\n\n    hp : PolyElement\n        multivariate integer polynomial with coefficients in `\\mathbb{Z}_p`\n    hq : PolyElement\n        multivariate integer polynomial with coefficients in `\\mathbb{Z}_q`\n    p : Integer\n        modulus of `h_p`, relatively prime to `q`\n    q : Integer\n        modulus of `h_q`, relatively prime to `p`\n\n    Examples\n    ========\n\n    >>> from sympy.polys.modulargcd import _chinese_remainder_reconstruction_multivariate\n    >>> from sympy.polys import ring, ZZ\n\n    >>> R, x, y = ring("x, y", ZZ)\n    >>> p = 3\n    >>> q = 5\n\n    >>> hp = x**3*y - x**2 - 1\n    >>> hq = -x**3*y - 2*x*y**2 + 2\n\n    >>> hpq = _chinese_remainder_reconstruction_multivariate(hp, hq, p, q)\n    >>> hpq\n    4*x**3*y + 5*x**2 + 3*x*y**2 + 2\n\n    >>> hpq.trunc_ground(p) == hp\n    True\n    >>> hpq.trunc_ground(q) == hq\n    True\n\n    >>> R, x, y, z = ring("x, y, z", ZZ)\n    >>> p = 6\n    >>> q = 5\n\n    >>> hp = 3*x**4 - y**3*z + z\n    >>> hq = -2*x**4 + z\n\n    >>> hpq = _chinese_remainder_reconstruction_multivariate(hp, hq, p, q)\n    >>> hpq\n    3*x**4 + 5*y**3*z + z\n\n    >>> hpq.trunc_ground(p) == hp\n    True\n    >>> hpq.trunc_ground(q) == hq\n    True\n\n    '
    hpmonoms = set(hp.monoms())
    hqmonoms = set(hq.monoms())
    monoms = hpmonoms.intersection(hqmonoms)
    hpmonoms.difference_update(monoms)
    hqmonoms.difference_update(monoms)
    zero = hp.ring.domain.zero
    hpq = hp.ring.zero
    if isinstance(hp.ring.domain, PolynomialRing):
        crt_ = _chinese_remainder_reconstruction_multivariate
    else:

        def crt_(cp, cq, p, q):
            if False:
                i = 10
                return i + 15
            return crt([p, q], [cp, cq], symmetric=True)[0]
    for monom in monoms:
        hpq[monom] = crt_(hp[monom], hq[monom], p, q)
    for monom in hpmonoms:
        hpq[monom] = crt_(hp[monom], zero, p, q)
    for monom in hqmonoms:
        hpq[monom] = crt_(zero, hq[monom], p, q)
    return hpq

def _interpolate_multivariate(evalpoints, hpeval, ring, i, p, ground=False):
    if False:
        i = 10
        return i + 15
    '\n    Reconstruct a polynomial `h_p` in `\\mathbb{Z}_p[x_0, \\ldots, x_{k-1}]`\n    from a list of evaluation points in `\\mathbb{Z}_p` and a list of\n    polynomials in\n    `\\mathbb{Z}_p[x_0, \\ldots, x_{i-1}, x_{i+1}, \\ldots, x_{k-1}]`, which\n    are the images of `h_p` evaluated in the variable `x_i`.\n\n    It is also possible to reconstruct a parameter of the ground domain,\n    i.e. if `h_p` is a polynomial over `\\mathbb{Z}_p[x_0, \\ldots, x_{k-1}]`.\n    In this case, one has to set ``ground=True``.\n\n    Parameters\n    ==========\n\n    evalpoints : list of Integer objects\n        list of evaluation points in `\\mathbb{Z}_p`\n    hpeval : list of PolyElement objects\n        list of polynomials in (resp. over)\n        `\\mathbb{Z}_p[x_0, \\ldots, x_{i-1}, x_{i+1}, \\ldots, x_{k-1}]`,\n        images of `h_p` evaluated in the variable `x_i`\n    ring : PolyRing\n        `h_p` will be an element of this ring\n    i : Integer\n        index of the variable which has to be reconstructed\n    p : Integer\n        prime number, modulus of `h_p`\n    ground : Boolean\n        indicates whether `x_i` is in the ground domain, default is\n        ``False``\n\n    Returns\n    =======\n\n    hp : PolyElement\n        interpolated polynomial in (resp. over)\n        `\\mathbb{Z}_p[x_0, \\ldots, x_{k-1}]`\n\n    '
    hp = ring.zero
    if ground:
        domain = ring.domain.domain
        y = ring.domain.gens[i]
    else:
        domain = ring.domain
        y = ring.gens[i]
    for (a, hpa) in zip(evalpoints, hpeval):
        numer = ring.one
        denom = domain.one
        for b in evalpoints:
            if b == a:
                continue
            numer *= y - b
            denom *= a - b
        denom = domain.invert(denom, p)
        coeff = numer.mul_ground(denom)
        hp += hpa.set_ring(ring) * coeff
    return hp.trunc_ground(p)

def modgcd_bivariate(f, g):
    if False:
        return 10
    '\n    Computes the GCD of two polynomials in `\\mathbb{Z}[x, y]` using a\n    modular algorithm.\n\n    The algorithm computes the GCD of two bivariate integer polynomials\n    `f` and `g` by calculating the GCD in `\\mathbb{Z}_p[x, y]` for\n    suitable primes `p` and then reconstructing the coefficients with the\n    Chinese Remainder Theorem. To compute the bivariate GCD over\n    `\\mathbb{Z}_p`, the polynomials `f \\; \\mathrm{mod} \\, p` and\n    `g \\; \\mathrm{mod} \\, p` are evaluated at `y = a` for certain\n    `a \\in \\mathbb{Z}_p` and then their univariate GCD in `\\mathbb{Z}_p[x]`\n    is computed. Interpolating those yields the bivariate GCD in\n    `\\mathbb{Z}_p[x, y]`. To verify the result in `\\mathbb{Z}[x, y]`, trial\n    division is done, but only for candidates which are very likely the\n    desired GCD.\n\n    Parameters\n    ==========\n\n    f : PolyElement\n        bivariate integer polynomial\n    g : PolyElement\n        bivariate integer polynomial\n\n    Returns\n    =======\n\n    h : PolyElement\n        GCD of the polynomials `f` and `g`\n    cff : PolyElement\n        cofactor of `f`, i.e. `\\frac{f}{h}`\n    cfg : PolyElement\n        cofactor of `g`, i.e. `\\frac{g}{h}`\n\n    Examples\n    ========\n\n    >>> from sympy.polys.modulargcd import modgcd_bivariate\n    >>> from sympy.polys import ring, ZZ\n\n    >>> R, x, y = ring("x, y", ZZ)\n\n    >>> f = x**2 - y**2\n    >>> g = x**2 + 2*x*y + y**2\n\n    >>> h, cff, cfg = modgcd_bivariate(f, g)\n    >>> h, cff, cfg\n    (x + y, x - y, x + y)\n\n    >>> cff * h == f\n    True\n    >>> cfg * h == g\n    True\n\n    >>> f = x**2*y - x**2 - 4*y + 4\n    >>> g = x + 2\n\n    >>> h, cff, cfg = modgcd_bivariate(f, g)\n    >>> h, cff, cfg\n    (x + 2, x*y - x - 2*y + 2, 1)\n\n    >>> cff * h == f\n    True\n    >>> cfg * h == g\n    True\n\n    References\n    ==========\n\n    1. [Monagan00]_\n\n    '
    assert f.ring == g.ring and f.ring.domain.is_ZZ
    result = _trivial_gcd(f, g)
    if result is not None:
        return result
    ring = f.ring
    (cf, f) = f.primitive()
    (cg, g) = g.primitive()
    ch = ring.domain.gcd(cf, cg)
    (xbound, ycontbound) = _degree_bound_bivariate(f, g)
    if xbound == ycontbound == 0:
        return (ring(ch), f.mul_ground(cf // ch), g.mul_ground(cg // ch))
    fswap = _swap(f, 1)
    gswap = _swap(g, 1)
    degyf = fswap.degree()
    degyg = gswap.degree()
    (ybound, xcontbound) = _degree_bound_bivariate(fswap, gswap)
    if ybound == xcontbound == 0:
        return (ring(ch), f.mul_ground(cf // ch), g.mul_ground(cg // ch))
    gamma1 = ring.domain.gcd(f.LC, g.LC)
    gamma2 = ring.domain.gcd(fswap.LC, gswap.LC)
    badprimes = gamma1 * gamma2
    m = 1
    p = 1
    while True:
        p = nextprime(p)
        while badprimes % p == 0:
            p = nextprime(p)
        fp = f.trunc_ground(p)
        gp = g.trunc_ground(p)
        (contfp, fp) = _primitive(fp, p)
        (contgp, gp) = _primitive(gp, p)
        conthp = _gf_gcd(contfp, contgp, p)
        degconthp = conthp.degree()
        if degconthp > ycontbound:
            continue
        elif degconthp < ycontbound:
            m = 1
            ycontbound = degconthp
            continue
        delta = _gf_gcd(_LC(fp), _LC(gp), p)
        degcontfp = contfp.degree()
        degcontgp = contgp.degree()
        degdelta = delta.degree()
        N = min(degyf - degcontfp, degyg - degcontgp, ybound - ycontbound + degdelta) + 1
        if p < N:
            continue
        n = 0
        evalpoints = []
        hpeval = []
        unlucky = False
        for a in range(p):
            deltaa = delta.evaluate(0, a)
            if not deltaa % p:
                continue
            fpa = fp.evaluate(1, a).trunc_ground(p)
            gpa = gp.evaluate(1, a).trunc_ground(p)
            hpa = _gf_gcd(fpa, gpa, p)
            deghpa = hpa.degree()
            if deghpa > xbound:
                continue
            elif deghpa < xbound:
                m = 1
                xbound = deghpa
                unlucky = True
                break
            hpa = hpa.mul_ground(deltaa).trunc_ground(p)
            evalpoints.append(a)
            hpeval.append(hpa)
            n += 1
            if n == N:
                break
        if unlucky:
            continue
        if n < N:
            continue
        hp = _interpolate_multivariate(evalpoints, hpeval, ring, 1, p)
        hp = _primitive(hp, p)[1]
        hp = hp * conthp.set_ring(ring)
        degyhp = hp.degree(1)
        if degyhp > ybound:
            continue
        if degyhp < ybound:
            m = 1
            ybound = degyhp
            continue
        hp = hp.mul_ground(gamma1).trunc_ground(p)
        if m == 1:
            m = p
            hlastm = hp
            continue
        hm = _chinese_remainder_reconstruction_multivariate(hp, hlastm, p, m)
        m *= p
        if not hm == hlastm:
            hlastm = hm
            continue
        h = hm.quo_ground(hm.content())
        (fquo, frem) = f.div(h)
        (gquo, grem) = g.div(h)
        if not frem and (not grem):
            if h.LC < 0:
                ch = -ch
            h = h.mul_ground(ch)
            cff = fquo.mul_ground(cf // ch)
            cfg = gquo.mul_ground(cg // ch)
            return (h, cff, cfg)

def _modgcd_multivariate_p(f, g, p, degbound, contbound):
    if False:
        return 10
    '\n    Compute the GCD of two polynomials in\n    `\\mathbb{Z}_p[x_0, \\ldots, x_{k-1}]`.\n\n    The algorithm reduces the problem step by step by evaluating the\n    polynomials `f` and `g` at `x_{k-1} = a` for suitable\n    `a \\in \\mathbb{Z}_p` and then calls itself recursively to compute the GCD\n    in `\\mathbb{Z}_p[x_0, \\ldots, x_{k-2}]`. If these recursive calls are\n    successful for enough evaluation points, the GCD in `k` variables is\n    interpolated, otherwise the algorithm returns ``None``. Every time a GCD\n    or a content is computed, their degrees are compared with the bounds. If\n    a degree greater then the bound is encountered, then the current call\n    returns ``None`` and a new evaluation point has to be chosen. If at some\n    point the degree is smaller, the correspondent bound is updated and the\n    algorithm fails.\n\n    Parameters\n    ==========\n\n    f : PolyElement\n        multivariate integer polynomial with coefficients in `\\mathbb{Z}_p`\n    g : PolyElement\n        multivariate integer polynomial with coefficients in `\\mathbb{Z}_p`\n    p : Integer\n        prime number, modulus of `f` and `g`\n    degbound : list of Integer objects\n        ``degbound[i]`` is an upper bound for the degree of the GCD of `f`\n        and `g` in the variable `x_i`\n    contbound : list of Integer objects\n        ``contbound[i]`` is an upper bound for the degree of the content of\n        the GCD in `\\mathbb{Z}_p[x_i][x_0, \\ldots, x_{i-1}]`,\n        ``contbound[0]`` is not used can therefore be chosen\n        arbitrarily.\n\n    Returns\n    =======\n\n    h : PolyElement\n        GCD of the polynomials `f` and `g` or ``None``\n\n    References\n    ==========\n\n    1. [Monagan00]_\n    2. [Brown71]_\n\n    '
    ring = f.ring
    k = ring.ngens
    if k == 1:
        h = _gf_gcd(f, g, p).trunc_ground(p)
        degh = h.degree()
        if degh > degbound[0]:
            return None
        if degh < degbound[0]:
            degbound[0] = degh
            raise ModularGCDFailed
        return h
    degyf = f.degree(k - 1)
    degyg = g.degree(k - 1)
    (contf, f) = _primitive(f, p)
    (contg, g) = _primitive(g, p)
    conth = _gf_gcd(contf, contg, p)
    degcontf = contf.degree()
    degcontg = contg.degree()
    degconth = conth.degree()
    if degconth > contbound[k - 1]:
        return None
    if degconth < contbound[k - 1]:
        contbound[k - 1] = degconth
        raise ModularGCDFailed
    lcf = _LC(f)
    lcg = _LC(g)
    delta = _gf_gcd(lcf, lcg, p)
    evaltest = delta
    for i in range(k - 1):
        evaltest *= _gf_gcd(_LC(_swap(f, i)), _LC(_swap(g, i)), p)
    degdelta = delta.degree()
    N = min(degyf - degcontf, degyg - degcontg, degbound[k - 1] - contbound[k - 1] + degdelta) + 1
    if p < N:
        return None
    n = 0
    d = 0
    evalpoints = []
    heval = []
    points = list(range(p))
    while points:
        a = random.sample(points, 1)[0]
        points.remove(a)
        if not evaltest.evaluate(0, a) % p:
            continue
        deltaa = delta.evaluate(0, a) % p
        fa = f.evaluate(k - 1, a).trunc_ground(p)
        ga = g.evaluate(k - 1, a).trunc_ground(p)
        ha = _modgcd_multivariate_p(fa, ga, p, degbound, contbound)
        if ha is None:
            d += 1
            if d > n:
                return None
            continue
        if ha.is_ground:
            h = conth.set_ring(ring).trunc_ground(p)
            return h
        ha = ha.mul_ground(deltaa).trunc_ground(p)
        evalpoints.append(a)
        heval.append(ha)
        n += 1
        if n == N:
            h = _interpolate_multivariate(evalpoints, heval, ring, k - 1, p)
            h = _primitive(h, p)[1] * conth.set_ring(ring)
            degyh = h.degree(k - 1)
            if degyh > degbound[k - 1]:
                return None
            if degyh < degbound[k - 1]:
                degbound[k - 1] = degyh
                raise ModularGCDFailed
            return h
    return None

def modgcd_multivariate(f, g):
    if False:
        while True:
            i = 10
    '\n    Compute the GCD of two polynomials in `\\mathbb{Z}[x_0, \\ldots, x_{k-1}]`\n    using a modular algorithm.\n\n    The algorithm computes the GCD of two multivariate integer polynomials\n    `f` and `g` by calculating the GCD in\n    `\\mathbb{Z}_p[x_0, \\ldots, x_{k-1}]` for suitable primes `p` and then\n    reconstructing the coefficients with the Chinese Remainder Theorem. To\n    compute the multivariate GCD over `\\mathbb{Z}_p` the recursive\n    subroutine :func:`_modgcd_multivariate_p` is used. To verify the result in\n    `\\mathbb{Z}[x_0, \\ldots, x_{k-1}]`, trial division is done, but only for\n    candidates which are very likely the desired GCD.\n\n    Parameters\n    ==========\n\n    f : PolyElement\n        multivariate integer polynomial\n    g : PolyElement\n        multivariate integer polynomial\n\n    Returns\n    =======\n\n    h : PolyElement\n        GCD of the polynomials `f` and `g`\n    cff : PolyElement\n        cofactor of `f`, i.e. `\\frac{f}{h}`\n    cfg : PolyElement\n        cofactor of `g`, i.e. `\\frac{g}{h}`\n\n    Examples\n    ========\n\n    >>> from sympy.polys.modulargcd import modgcd_multivariate\n    >>> from sympy.polys import ring, ZZ\n\n    >>> R, x, y = ring("x, y", ZZ)\n\n    >>> f = x**2 - y**2\n    >>> g = x**2 + 2*x*y + y**2\n\n    >>> h, cff, cfg = modgcd_multivariate(f, g)\n    >>> h, cff, cfg\n    (x + y, x - y, x + y)\n\n    >>> cff * h == f\n    True\n    >>> cfg * h == g\n    True\n\n    >>> R, x, y, z = ring("x, y, z", ZZ)\n\n    >>> f = x*z**2 - y*z**2\n    >>> g = x**2*z + z\n\n    >>> h, cff, cfg = modgcd_multivariate(f, g)\n    >>> h, cff, cfg\n    (z, x*z - y*z, x**2 + 1)\n\n    >>> cff * h == f\n    True\n    >>> cfg * h == g\n    True\n\n    References\n    ==========\n\n    1. [Monagan00]_\n    2. [Brown71]_\n\n    See also\n    ========\n\n    _modgcd_multivariate_p\n\n    '
    assert f.ring == g.ring and f.ring.domain.is_ZZ
    result = _trivial_gcd(f, g)
    if result is not None:
        return result
    ring = f.ring
    k = ring.ngens
    (cf, f) = f.primitive()
    (cg, g) = g.primitive()
    ch = ring.domain.gcd(cf, cg)
    gamma = ring.domain.gcd(f.LC, g.LC)
    badprimes = ring.domain.one
    for i in range(k):
        badprimes *= ring.domain.gcd(_swap(f, i).LC, _swap(g, i).LC)
    degbound = [min(fdeg, gdeg) for (fdeg, gdeg) in zip(f.degrees(), g.degrees())]
    contbound = list(degbound)
    m = 1
    p = 1
    while True:
        p = nextprime(p)
        while badprimes % p == 0:
            p = nextprime(p)
        fp = f.trunc_ground(p)
        gp = g.trunc_ground(p)
        try:
            hp = _modgcd_multivariate_p(fp, gp, p, degbound, contbound)
        except ModularGCDFailed:
            m = 1
            continue
        if hp is None:
            continue
        hp = hp.mul_ground(gamma).trunc_ground(p)
        if m == 1:
            m = p
            hlastm = hp
            continue
        hm = _chinese_remainder_reconstruction_multivariate(hp, hlastm, p, m)
        m *= p
        if not hm == hlastm:
            hlastm = hm
            continue
        h = hm.primitive()[1]
        (fquo, frem) = f.div(h)
        (gquo, grem) = g.div(h)
        if not frem and (not grem):
            if h.LC < 0:
                ch = -ch
            h = h.mul_ground(ch)
            cff = fquo.mul_ground(cf // ch)
            cfg = gquo.mul_ground(cg // ch)
            return (h, cff, cfg)

def _gf_div(f, g, p):
    if False:
        print('Hello World!')
    '\n    Compute `\\frac f g` modulo `p` for two univariate polynomials over\n    `\\mathbb Z_p`.\n    '
    ring = f.ring
    (densequo, denserem) = gf_div(f.to_dense(), g.to_dense(), p, ring.domain)
    return (ring.from_dense(densequo), ring.from_dense(denserem))

def _rational_function_reconstruction(c, p, m):
    if False:
        print('Hello World!')
    '\n    Reconstruct a rational function `\\frac a b` in `\\mathbb Z_p(t)` from\n\n    .. math::\n\n        c = \\frac a b \\; \\mathrm{mod} \\, m,\n\n    where `c` and `m` are polynomials in `\\mathbb Z_p[t]` and `m` has\n    positive degree.\n\n    The algorithm is based on the Euclidean Algorithm. In general, `m` is\n    not irreducible, so it is possible that `b` is not invertible modulo\n    `m`. In that case ``None`` is returned.\n\n    Parameters\n    ==========\n\n    c : PolyElement\n        univariate polynomial in `\\mathbb Z[t]`\n    p : Integer\n        prime number\n    m : PolyElement\n        modulus, not necessarily irreducible\n\n    Returns\n    =======\n\n    frac : FracElement\n        either `\\frac a b` in `\\mathbb Z(t)` or ``None``\n\n    References\n    ==========\n\n    1. [Hoeij04]_\n\n    '
    ring = c.ring
    domain = ring.domain
    M = m.degree()
    N = M // 2
    D = M - N - 1
    (r0, s0) = (m, ring.zero)
    (r1, s1) = (c, ring.one)
    while r1.degree() > N:
        quo = _gf_div(r0, r1, p)[0]
        (r0, r1) = (r1, (r0 - quo * r1).trunc_ground(p))
        (s0, s1) = (s1, (s0 - quo * s1).trunc_ground(p))
    (a, b) = (r1, s1)
    if b.degree() > D or _gf_gcd(b, m, p) != 1:
        return None
    lc = b.LC
    if lc != 1:
        lcinv = domain.invert(lc, p)
        a = a.mul_ground(lcinv).trunc_ground(p)
        b = b.mul_ground(lcinv).trunc_ground(p)
    field = ring.to_field()
    return field(a) / field(b)

def _rational_reconstruction_func_coeffs(hm, p, m, ring, k):
    if False:
        return 10
    '\n    Reconstruct every coefficient `c_h` of a polynomial `h` in\n    `\\mathbb Z_p(t_k)[t_1, \\ldots, t_{k-1}][x, z]` from the corresponding\n    coefficient `c_{h_m}` of a polynomial `h_m` in\n    `\\mathbb Z_p[t_1, \\ldots, t_k][x, z] \\cong \\mathbb Z_p[t_k][t_1, \\ldots, t_{k-1}][x, z]`\n    such that\n\n    .. math::\n\n        c_{h_m} = c_h \\; \\mathrm{mod} \\, m,\n\n    where `m \\in \\mathbb Z_p[t]`.\n\n    The reconstruction is based on the Euclidean Algorithm. In general, `m`\n    is not irreducible, so it is possible that this fails for some\n    coefficient. In that case ``None`` is returned.\n\n    Parameters\n    ==========\n\n    hm : PolyElement\n        polynomial in `\\mathbb Z[t_1, \\ldots, t_k][x, z]`\n    p : Integer\n        prime number, modulus of `\\mathbb Z_p`\n    m : PolyElement\n        modulus, polynomial in `\\mathbb Z[t]`, not necessarily irreducible\n    ring : PolyRing\n        `\\mathbb Z(t_k)[t_1, \\ldots, t_{k-1}][x, z]`, `h` will be an\n        element of this ring\n    k : Integer\n        index of the parameter `t_k` which will be reconstructed\n\n    Returns\n    =======\n\n    h : PolyElement\n        reconstructed polynomial in\n        `\\mathbb Z(t_k)[t_1, \\ldots, t_{k-1}][x, z]` or ``None``\n\n    See also\n    ========\n\n    _rational_function_reconstruction\n\n    '
    h = ring.zero
    for (monom, coeff) in hm.iterterms():
        if k == 0:
            coeffh = _rational_function_reconstruction(coeff, p, m)
            if not coeffh:
                return None
        else:
            coeffh = ring.domain.zero
            for (mon, c) in coeff.drop_to_ground(k).iterterms():
                ch = _rational_function_reconstruction(c, p, m)
                if not ch:
                    return None
                coeffh[mon] = ch
        h[monom] = coeffh
    return h

def _gf_gcdex(f, g, p):
    if False:
        while True:
            i = 10
    '\n    Extended Euclidean Algorithm for two univariate polynomials over\n    `\\mathbb Z_p`.\n\n    Returns polynomials `s, t` and `h`, such that `h` is the GCD of `f` and\n    `g` and `sf + tg = h \\; \\mathrm{mod} \\, p`.\n\n    '
    ring = f.ring
    (s, t, h) = gf_gcdex(f.to_dense(), g.to_dense(), p, ring.domain)
    return (ring.from_dense(s), ring.from_dense(t), ring.from_dense(h))

def _trunc(f, minpoly, p):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the reduced representation of a polynomial `f` in\n    `\\mathbb Z_p[z] / (\\check m_{\\alpha}(z))[x]`\n\n    Parameters\n    ==========\n\n    f : PolyElement\n        polynomial in `\\mathbb Z[x, z]`\n    minpoly : PolyElement\n        polynomial `\\check m_{\\alpha} \\in \\mathbb Z[z]`, not necessarily\n        irreducible\n    p : Integer\n        prime number, modulus of `\\mathbb Z_p`\n\n    Returns\n    =======\n\n    ftrunc : PolyElement\n        polynomial in `\\mathbb Z[x, z]`, reduced modulo\n        `\\check m_{\\alpha}(z)` and `p`\n\n    '
    ring = f.ring
    minpoly = minpoly.set_ring(ring)
    p_ = ring.ground_new(p)
    return f.trunc_ground(p).rem([minpoly, p_]).trunc_ground(p)

def _euclidean_algorithm(f, g, minpoly, p):
    if False:
        i = 10
        return i + 15
    '\n    Compute the monic GCD of two univariate polynomials in\n    `\\mathbb{Z}_p[z]/(\\check m_{\\alpha}(z))[x]` with the Euclidean\n    Algorithm.\n\n    In general, `\\check m_{\\alpha}(z)` is not irreducible, so it is possible\n    that some leading coefficient is not invertible modulo\n    `\\check m_{\\alpha}(z)`. In that case ``None`` is returned.\n\n    Parameters\n    ==========\n\n    f, g : PolyElement\n        polynomials in `\\mathbb Z[x, z]`\n    minpoly : PolyElement\n        polynomial in `\\mathbb Z[z]`, not necessarily irreducible\n    p : Integer\n        prime number, modulus of `\\mathbb Z_p`\n\n    Returns\n    =======\n\n    h : PolyElement\n        GCD of `f` and `g` in `\\mathbb Z[z, x]` or ``None``, coefficients\n        are in `\\left[ -\\frac{p-1} 2, \\frac{p-1} 2 \\right]`\n\n    '
    ring = f.ring
    f = _trunc(f, minpoly, p)
    g = _trunc(g, minpoly, p)
    while g:
        rem = f
        deg = g.degree(0)
        (lcinv, _, gcd) = _gf_gcdex(ring.dmp_LC(g), minpoly, p)
        if not gcd == 1:
            return None
        while True:
            degrem = rem.degree(0)
            if degrem < deg:
                break
            quo = (lcinv * ring.dmp_LC(rem)).set_ring(ring)
            rem = _trunc(rem - g.mul_monom((degrem - deg, 0)) * quo, minpoly, p)
        f = g
        g = rem
    lcfinv = _gf_gcdex(ring.dmp_LC(f), minpoly, p)[0].set_ring(ring)
    return _trunc(f * lcfinv, minpoly, p)

def _trial_division(f, h, minpoly, p=None):
    if False:
        while True:
            i = 10
    '\n    Check if `h` divides `f` in\n    `\\mathbb K[t_1, \\ldots, t_k][z]/(m_{\\alpha}(z))`, where `\\mathbb K` is\n    either `\\mathbb Q` or `\\mathbb Z_p`.\n\n    This algorithm is based on pseudo division and does not use any\n    fractions. By default `\\mathbb K` is `\\mathbb Q`, if a prime number `p`\n    is given, `\\mathbb Z_p` is chosen instead.\n\n    Parameters\n    ==========\n\n    f, h : PolyElement\n        polynomials in `\\mathbb Z[t_1, \\ldots, t_k][x, z]`\n    minpoly : PolyElement\n        polynomial `m_{\\alpha}(z)` in `\\mathbb Z[t_1, \\ldots, t_k][z]`\n    p : Integer or None\n        if `p` is given, `\\mathbb K` is set to `\\mathbb Z_p` instead of\n        `\\mathbb Q`, default is ``None``\n\n    Returns\n    =======\n\n    rem : PolyElement\n        remainder of `\\frac f h`\n\n    References\n    ==========\n\n    .. [1] [Hoeij02]_\n\n    '
    ring = f.ring
    zxring = ring.clone(symbols=(ring.symbols[1], ring.symbols[0]))
    minpoly = minpoly.set_ring(ring)
    rem = f
    degrem = rem.degree()
    degh = h.degree()
    degm = minpoly.degree(1)
    lch = _LC(h).set_ring(ring)
    lcm = minpoly.LC
    while rem and degrem >= degh:
        lcrem = _LC(rem).set_ring(ring)
        rem = rem * lch - h.mul_monom((degrem - degh, 0)) * lcrem
        if p:
            rem = rem.trunc_ground(p)
        degrem = rem.degree(1)
        while rem and degrem >= degm:
            lcrem = _LC(rem.set_ring(zxring)).set_ring(ring)
            rem = rem.mul_ground(lcm) - minpoly.mul_monom((0, degrem - degm)) * lcrem
            if p:
                rem = rem.trunc_ground(p)
            degrem = rem.degree(1)
        degrem = rem.degree()
    return rem

def _evaluate_ground(f, i, a):
    if False:
        for i in range(10):
            print('nop')
    '\n    Evaluate a polynomial `f` at `a` in the `i`-th variable of the ground\n    domain.\n    '
    ring = f.ring.clone(domain=f.ring.domain.ring.drop(i))
    fa = ring.zero
    for (monom, coeff) in f.iterterms():
        fa[monom] = coeff.evaluate(i, a)
    return fa

def _func_field_modgcd_p(f, g, minpoly, p):
    if False:
        i = 10
        return i + 15
    '\n    Compute the GCD of two polynomials `f` and `g` in\n    `\\mathbb Z_p(t_1, \\ldots, t_k)[z]/(\\check m_\\alpha(z))[x]`.\n\n    The algorithm reduces the problem step by step by evaluating the\n    polynomials `f` and `g` at `t_k = a` for suitable `a \\in \\mathbb Z_p`\n    and then calls itself recursively to compute the GCD in\n    `\\mathbb Z_p(t_1, \\ldots, t_{k-1})[z]/(\\check m_\\alpha(z))[x]`. If these\n    recursive calls are successful, the GCD over `k` variables is\n    interpolated, otherwise the algorithm returns ``None``. After\n    interpolation, Rational Function Reconstruction is used to obtain the\n    correct coefficients. If this fails, a new evaluation point has to be\n    chosen, otherwise the desired polynomial is obtained by clearing\n    denominators. The result is verified with a fraction free trial\n    division.\n\n    Parameters\n    ==========\n\n    f, g : PolyElement\n        polynomials in `\\mathbb Z[t_1, \\ldots, t_k][x, z]`\n    minpoly : PolyElement\n        polynomial in `\\mathbb Z[t_1, \\ldots, t_k][z]`, not necessarily\n        irreducible\n    p : Integer\n        prime number, modulus of `\\mathbb Z_p`\n\n    Returns\n    =======\n\n    h : PolyElement\n        primitive associate in `\\mathbb Z[t_1, \\ldots, t_k][x, z]` of the\n        GCD of the polynomials `f` and `g`  or ``None``, coefficients are\n        in `\\left[ -\\frac{p-1} 2, \\frac{p-1} 2 \\right]`\n\n    References\n    ==========\n\n    1. [Hoeij04]_\n\n    '
    ring = f.ring
    domain = ring.domain
    if isinstance(domain, PolynomialRing):
        k = domain.ngens
    else:
        return _euclidean_algorithm(f, g, minpoly, p)
    if k == 1:
        qdomain = domain.ring.to_field()
    else:
        qdomain = domain.ring.drop_to_ground(k - 1)
        qdomain = qdomain.clone(domain=qdomain.domain.ring.to_field())
    qring = ring.clone(domain=qdomain)
    n = 1
    d = 1
    gamma = ring.dmp_LC(f) * ring.dmp_LC(g)
    delta = minpoly.LC
    evalpoints = []
    heval = []
    LMlist = []
    points = list(range(p))
    while points:
        a = random.sample(points, 1)[0]
        points.remove(a)
        if k == 1:
            test = delta.evaluate(k - 1, a) % p == 0
        else:
            test = delta.evaluate(k - 1, a).trunc_ground(p) == 0
        if test:
            continue
        gammaa = _evaluate_ground(gamma, k - 1, a)
        minpolya = _evaluate_ground(minpoly, k - 1, a)
        if gammaa.rem([minpolya, gammaa.ring(p)]) == 0:
            continue
        fa = _evaluate_ground(f, k - 1, a)
        ga = _evaluate_ground(g, k - 1, a)
        ha = _func_field_modgcd_p(fa, ga, minpolya, p)
        if ha is None:
            d += 1
            if d > n:
                return None
            continue
        if ha == 1:
            return ha
        LM = [ha.degree()] + [0] * (k - 1)
        if k > 1:
            for (monom, coeff) in ha.iterterms():
                if monom[0] == LM[0] and coeff.LM > tuple(LM[1:]):
                    LM[1:] = coeff.LM
        evalpoints_a = [a]
        heval_a = [ha]
        if k == 1:
            m = qring.domain.get_ring().one
        else:
            m = qring.domain.domain.get_ring().one
        t = m.ring.gens[0]
        for (b, hb, LMhb) in zip(evalpoints, heval, LMlist):
            if LMhb == LM:
                evalpoints_a.append(b)
                heval_a.append(hb)
                m *= t - b
        m = m.trunc_ground(p)
        evalpoints.append(a)
        heval.append(ha)
        LMlist.append(LM)
        n += 1
        h = _interpolate_multivariate(evalpoints_a, heval_a, ring, k - 1, p, ground=True)
        h = _rational_reconstruction_func_coeffs(h, p, m, qring, k - 1)
        if h is None:
            continue
        if k == 1:
            dom = qring.domain.field
            den = dom.ring.one
            for coeff in h.itercoeffs():
                den = dom.ring.from_dense(gf_lcm(den.to_dense(), coeff.denom.to_dense(), p, dom.domain))
        else:
            dom = qring.domain.domain.field
            den = dom.ring.one
            for coeff in h.itercoeffs():
                for c in coeff.itercoeffs():
                    den = dom.ring.from_dense(gf_lcm(den.to_dense(), c.denom.to_dense(), p, dom.domain))
        den = qring.domain_new(den.trunc_ground(p))
        h = ring(h.mul_ground(den).as_expr()).trunc_ground(p)
        if not _trial_division(f, h, minpoly, p) and (not _trial_division(g, h, minpoly, p)):
            return h
    return None

def _integer_rational_reconstruction(c, m, domain):
    if False:
        for i in range(10):
            print('nop')
    '\n    Reconstruct a rational number `\\frac a b` from\n\n    .. math::\n\n        c = \\frac a b \\; \\mathrm{mod} \\, m,\n\n    where `c` and `m` are integers.\n\n    The algorithm is based on the Euclidean Algorithm. In general, `m` is\n    not a prime number, so it is possible that `b` is not invertible modulo\n    `m`. In that case ``None`` is returned.\n\n    Parameters\n    ==========\n\n    c : Integer\n        `c = \\frac a b \\; \\mathrm{mod} \\, m`\n    m : Integer\n        modulus, not necessarily prime\n    domain : IntegerRing\n        `a, b, c` are elements of ``domain``\n\n    Returns\n    =======\n\n    frac : Rational\n        either `\\frac a b` in `\\mathbb Q` or ``None``\n\n    References\n    ==========\n\n    1. [Wang81]_\n\n    '
    if c < 0:
        c += m
    (r0, s0) = (m, domain.zero)
    (r1, s1) = (c, domain.one)
    bound = sqrt(m / 2)
    while int(r1) >= bound:
        quo = r0 // r1
        (r0, r1) = (r1, r0 - quo * r1)
        (s0, s1) = (s1, s0 - quo * s1)
    if abs(int(s1)) >= bound:
        return None
    if s1 < 0:
        (a, b) = (-r1, -s1)
    elif s1 > 0:
        (a, b) = (r1, s1)
    else:
        return None
    field = domain.get_field()
    return field(a) / field(b)

def _rational_reconstruction_int_coeffs(hm, m, ring):
    if False:
        return 10
    '\n    Reconstruct every rational coefficient `c_h` of a polynomial `h` in\n    `\\mathbb Q[t_1, \\ldots, t_k][x, z]` from the corresponding integer\n    coefficient `c_{h_m}` of a polynomial `h_m` in\n    `\\mathbb Z[t_1, \\ldots, t_k][x, z]` such that\n\n    .. math::\n\n        c_{h_m} = c_h \\; \\mathrm{mod} \\, m,\n\n    where `m \\in \\mathbb Z`.\n\n    The reconstruction is based on the Euclidean Algorithm. In general,\n    `m` is not a prime number, so it is possible that this fails for some\n    coefficient. In that case ``None`` is returned.\n\n    Parameters\n    ==========\n\n    hm : PolyElement\n        polynomial in `\\mathbb Z[t_1, \\ldots, t_k][x, z]`\n    m : Integer\n        modulus, not necessarily prime\n    ring : PolyRing\n        `\\mathbb Q[t_1, \\ldots, t_k][x, z]`, `h` will be an element of this\n        ring\n\n    Returns\n    =======\n\n    h : PolyElement\n        reconstructed polynomial in `\\mathbb Q[t_1, \\ldots, t_k][x, z]` or\n        ``None``\n\n    See also\n    ========\n\n    _integer_rational_reconstruction\n\n    '
    h = ring.zero
    if isinstance(ring.domain, PolynomialRing):
        reconstruction = _rational_reconstruction_int_coeffs
        domain = ring.domain.ring
    else:
        reconstruction = _integer_rational_reconstruction
        domain = hm.ring.domain
    for (monom, coeff) in hm.iterterms():
        coeffh = reconstruction(coeff, m, domain)
        if not coeffh:
            return None
        h[monom] = coeffh
    return h

def _func_field_modgcd_m(f, g, minpoly):
    if False:
        while True:
            i = 10
    "\n    Compute the GCD of two polynomials in\n    `\\mathbb Q(t_1, \\ldots, t_k)[z]/(m_{\\alpha}(z))[x]` using a modular\n    algorithm.\n\n    The algorithm computes the GCD of two polynomials `f` and `g` by\n    calculating the GCD in\n    `\\mathbb Z_p(t_1, \\ldots, t_k)[z] / (\\check m_{\\alpha}(z))[x]` for\n    suitable primes `p` and the primitive associate `\\check m_{\\alpha}(z)`\n    of `m_{\\alpha}(z)`. Then the coefficients are reconstructed with the\n    Chinese Remainder Theorem and Rational Reconstruction. To compute the\n    GCD over `\\mathbb Z_p(t_1, \\ldots, t_k)[z] / (\\check m_{\\alpha})[x]`,\n    the recursive subroutine ``_func_field_modgcd_p`` is used. To verify the\n    result in `\\mathbb Q(t_1, \\ldots, t_k)[z] / (m_{\\alpha}(z))[x]`, a\n    fraction free trial division is used.\n\n    Parameters\n    ==========\n\n    f, g : PolyElement\n        polynomials in `\\mathbb Z[t_1, \\ldots, t_k][x, z]`\n    minpoly : PolyElement\n        irreducible polynomial in `\\mathbb Z[t_1, \\ldots, t_k][z]`\n\n    Returns\n    =======\n\n    h : PolyElement\n        the primitive associate in `\\mathbb Z[t_1, \\ldots, t_k][x, z]` of\n        the GCD of `f` and `g`\n\n    Examples\n    ========\n\n    >>> from sympy.polys.modulargcd import _func_field_modgcd_m\n    >>> from sympy.polys import ring, ZZ\n\n    >>> R, x, z = ring('x, z', ZZ)\n    >>> minpoly = (z**2 - 2).drop(0)\n\n    >>> f = x**2 + 2*x*z + 2\n    >>> g = x + z\n    >>> _func_field_modgcd_m(f, g, minpoly)\n    x + z\n\n    >>> D, t = ring('t', ZZ)\n    >>> R, x, z = ring('x, z', D)\n    >>> minpoly = (z**2-3).drop(0)\n\n    >>> f = x**2 + (t + 1)*x*z + 3*t\n    >>> g = x*z + 3*t\n    >>> _func_field_modgcd_m(f, g, minpoly)\n    x + t*z\n\n    References\n    ==========\n\n    1. [Hoeij04]_\n\n    See also\n    ========\n\n    _func_field_modgcd_p\n\n    "
    ring = f.ring
    domain = ring.domain
    if isinstance(domain, PolynomialRing):
        k = domain.ngens
        QQdomain = domain.ring.clone(domain=domain.domain.get_field())
        QQring = ring.clone(domain=QQdomain)
    else:
        k = 0
        QQring = ring.clone(domain=ring.domain.get_field())
    (cf, f) = f.primitive()
    (cg, g) = g.primitive()
    gamma = ring.dmp_LC(f) * ring.dmp_LC(g)
    delta = minpoly.LC
    p = 1
    primes = []
    hplist = []
    LMlist = []
    while True:
        p = nextprime(p)
        if gamma.trunc_ground(p) == 0:
            continue
        if k == 0:
            test = delta % p == 0
        else:
            test = delta.trunc_ground(p) == 0
        if test:
            continue
        fp = f.trunc_ground(p)
        gp = g.trunc_ground(p)
        minpolyp = minpoly.trunc_ground(p)
        hp = _func_field_modgcd_p(fp, gp, minpolyp, p)
        if hp is None:
            continue
        if hp == 1:
            return ring.one
        LM = [hp.degree()] + [0] * k
        if k > 0:
            for (monom, coeff) in hp.iterterms():
                if monom[0] == LM[0] and coeff.LM > tuple(LM[1:]):
                    LM[1:] = coeff.LM
        hm = hp
        m = p
        for (q, hq, LMhq) in zip(primes, hplist, LMlist):
            if LMhq == LM:
                hm = _chinese_remainder_reconstruction_multivariate(hq, hm, q, m)
                m *= q
        primes.append(p)
        hplist.append(hp)
        LMlist.append(LM)
        hm = _rational_reconstruction_int_coeffs(hm, m, QQring)
        if hm is None:
            continue
        if k == 0:
            h = hm.clear_denoms()[1]
        else:
            den = domain.domain.one
            for coeff in hm.itercoeffs():
                den = domain.domain.lcm(den, coeff.clear_denoms()[0])
            h = hm.mul_ground(den)
        h = h.set_ring(ring)
        h = h.primitive()[1]
        if not (_trial_division(f.mul_ground(cf), h, minpoly) or _trial_division(g.mul_ground(cg), h, minpoly)):
            return h

def _to_ZZ_poly(f, ring):
    if False:
        i = 10
        return i + 15
    '\n    Compute an associate of a polynomial\n    `f \\in \\mathbb Q(\\alpha)[x_0, \\ldots, x_{n-1}]` in\n    `\\mathbb Z[x_1, \\ldots, x_{n-1}][z] / (\\check m_{\\alpha}(z))[x_0]`,\n    where `\\check m_{\\alpha}(z) \\in \\mathbb Z[z]` is the primitive associate\n    of the minimal polynomial `m_{\\alpha}(z)` of `\\alpha` over\n    `\\mathbb Q`.\n\n    Parameters\n    ==========\n\n    f : PolyElement\n        polynomial in `\\mathbb Q(\\alpha)[x_0, \\ldots, x_{n-1}]`\n    ring : PolyRing\n        `\\mathbb Z[x_1, \\ldots, x_{n-1}][x_0, z]`\n\n    Returns\n    =======\n\n    f_ : PolyElement\n        associate of `f` in\n        `\\mathbb Z[x_1, \\ldots, x_{n-1}][x_0, z]`\n\n    '
    f_ = ring.zero
    if isinstance(ring.domain, PolynomialRing):
        domain = ring.domain.domain
    else:
        domain = ring.domain
    den = domain.one
    for coeff in f.itercoeffs():
        for c in coeff.to_list():
            if c:
                den = domain.lcm(den, c.denominator)
    for (monom, coeff) in f.iterterms():
        coeff = coeff.to_list()
        m = ring.domain.one
        if isinstance(ring.domain, PolynomialRing):
            m = m.mul_monom(monom[1:])
        n = len(coeff)
        for i in range(n):
            if coeff[i]:
                c = domain.convert(coeff[i] * den) * m
                if (monom[0], n - i - 1) not in f_:
                    f_[monom[0], n - i - 1] = c
                else:
                    f_[monom[0], n - i - 1] += c
    return f_

def _to_ANP_poly(f, ring):
    if False:
        print('Hello World!')
    '\n    Convert a polynomial\n    `f \\in \\mathbb Z[x_1, \\ldots, x_{n-1}][z]/(\\check m_{\\alpha}(z))[x_0]`\n    to a polynomial in `\\mathbb Q(\\alpha)[x_0, \\ldots, x_{n-1}]`,\n    where `\\check m_{\\alpha}(z) \\in \\mathbb Z[z]` is the primitive associate\n    of the minimal polynomial `m_{\\alpha}(z)` of `\\alpha` over\n    `\\mathbb Q`.\n\n    Parameters\n    ==========\n\n    f : PolyElement\n        polynomial in `\\mathbb Z[x_1, \\ldots, x_{n-1}][x_0, z]`\n    ring : PolyRing\n        `\\mathbb Q(\\alpha)[x_0, \\ldots, x_{n-1}]`\n\n    Returns\n    =======\n\n    f_ : PolyElement\n        polynomial in `\\mathbb Q(\\alpha)[x_0, \\ldots, x_{n-1}]`\n\n    '
    domain = ring.domain
    f_ = ring.zero
    if isinstance(f.ring.domain, PolynomialRing):
        for (monom, coeff) in f.iterterms():
            for (mon, coef) in coeff.iterterms():
                m = (monom[0],) + mon
                c = domain([domain.domain(coef)] + [0] * monom[1])
                if m not in f_:
                    f_[m] = c
                else:
                    f_[m] += c
    else:
        for (monom, coeff) in f.iterterms():
            m = (monom[0],)
            c = domain([domain.domain(coeff)] + [0] * monom[1])
            if m not in f_:
                f_[m] = c
            else:
                f_[m] += c
    return f_

def _minpoly_from_dense(minpoly, ring):
    if False:
        for i in range(10):
            print('nop')
    '\n    Change representation of the minimal polynomial from ``DMP`` to\n    ``PolyElement`` for a given ring.\n    '
    minpoly_ = ring.zero
    for (monom, coeff) in minpoly.terms():
        minpoly_[monom] = ring.domain(coeff)
    return minpoly_

def _primitive_in_x0(f):
    if False:
        i = 10
        return i + 15
    '\n    Compute the content in `x_0` and the primitive part of a polynomial `f`\n    in\n    `\\mathbb Q(\\alpha)[x_0, x_1, \\ldots, x_{n-1}] \\cong \\mathbb Q(\\alpha)[x_1, \\ldots, x_{n-1}][x_0]`.\n    '
    fring = f.ring
    ring = fring.drop_to_ground(*range(1, fring.ngens))
    dom = ring.domain.ring
    f_ = ring(f.as_expr())
    cont = dom.zero
    for coeff in f_.itercoeffs():
        cont = func_field_modgcd(cont, coeff)[0]
        if cont == dom.one:
            return (cont, f)
    return (cont, f.quo(cont.set_ring(fring)))

def func_field_modgcd(f, g):
    if False:
        return 10
    "\n    Compute the GCD of two polynomials `f` and `g` in\n    `\\mathbb Q(\\alpha)[x_0, \\ldots, x_{n-1}]` using a modular algorithm.\n\n    The algorithm first computes the primitive associate\n    `\\check m_{\\alpha}(z)` of the minimal polynomial `m_{\\alpha}` in\n    `\\mathbb{Z}[z]` and the primitive associates of `f` and `g` in\n    `\\mathbb{Z}[x_1, \\ldots, x_{n-1}][z]/(\\check m_{\\alpha})[x_0]`. Then it\n    computes the GCD in\n    `\\mathbb Q(x_1, \\ldots, x_{n-1})[z]/(m_{\\alpha}(z))[x_0]`.\n    This is done by calculating the GCD in\n    `\\mathbb{Z}_p(x_1, \\ldots, x_{n-1})[z]/(\\check m_{\\alpha}(z))[x_0]` for\n    suitable primes `p` and then reconstructing the coefficients with the\n    Chinese Remainder Theorem and Rational Reconstuction. The GCD over\n    `\\mathbb{Z}_p(x_1, \\ldots, x_{n-1})[z]/(\\check m_{\\alpha}(z))[x_0]` is\n    computed with a recursive subroutine, which evaluates the polynomials at\n    `x_{n-1} = a` for suitable evaluation points `a \\in \\mathbb Z_p` and\n    then calls itself recursively until the ground domain does no longer\n    contain any parameters. For\n    `\\mathbb{Z}_p[z]/(\\check m_{\\alpha}(z))[x_0]` the Euclidean Algorithm is\n    used. The results of those recursive calls are then interpolated and\n    Rational Function Reconstruction is used to obtain the correct\n    coefficients. The results, both in\n    `\\mathbb Q(x_1, \\ldots, x_{n-1})[z]/(m_{\\alpha}(z))[x_0]` and\n    `\\mathbb{Z}_p(x_1, \\ldots, x_{n-1})[z]/(\\check m_{\\alpha}(z))[x_0]`, are\n    verified by a fraction free trial division.\n\n    Apart from the above GCD computation some GCDs in\n    `\\mathbb Q(\\alpha)[x_1, \\ldots, x_{n-1}]` have to be calculated,\n    because treating the polynomials as univariate ones can result in\n    a spurious content of the GCD. For this ``func_field_modgcd`` is\n    called recursively.\n\n    Parameters\n    ==========\n\n    f, g : PolyElement\n        polynomials in `\\mathbb Q(\\alpha)[x_0, \\ldots, x_{n-1}]`\n\n    Returns\n    =======\n\n    h : PolyElement\n        monic GCD of the polynomials `f` and `g`\n    cff : PolyElement\n        cofactor of `f`, i.e. `\\frac f h`\n    cfg : PolyElement\n        cofactor of `g`, i.e. `\\frac g h`\n\n    Examples\n    ========\n\n    >>> from sympy.polys.modulargcd import func_field_modgcd\n    >>> from sympy.polys import AlgebraicField, QQ, ring\n    >>> from sympy import sqrt\n\n    >>> A = AlgebraicField(QQ, sqrt(2))\n    >>> R, x = ring('x', A)\n\n    >>> f = x**2 - 2\n    >>> g = x + sqrt(2)\n\n    >>> h, cff, cfg = func_field_modgcd(f, g)\n\n    >>> h == x + sqrt(2)\n    True\n    >>> cff * h == f\n    True\n    >>> cfg * h == g\n    True\n\n    >>> R, x, y = ring('x, y', A)\n\n    >>> f = x**2 + 2*sqrt(2)*x*y + 2*y**2\n    >>> g = x + sqrt(2)*y\n\n    >>> h, cff, cfg = func_field_modgcd(f, g)\n\n    >>> h == x + sqrt(2)*y\n    True\n    >>> cff * h == f\n    True\n    >>> cfg * h == g\n    True\n\n    >>> f = x + sqrt(2)*y\n    >>> g = x + y\n\n    >>> h, cff, cfg = func_field_modgcd(f, g)\n\n    >>> h == R.one\n    True\n    >>> cff * h == f\n    True\n    >>> cfg * h == g\n    True\n\n    References\n    ==========\n\n    1. [Hoeij04]_\n\n    "
    ring = f.ring
    domain = ring.domain
    n = ring.ngens
    assert ring == g.ring and domain.is_Algebraic
    result = _trivial_gcd(f, g)
    if result is not None:
        return result
    z = Dummy('z')
    ZZring = ring.clone(symbols=ring.symbols + (z,), domain=domain.domain.get_ring())
    if n == 1:
        f_ = _to_ZZ_poly(f, ZZring)
        g_ = _to_ZZ_poly(g, ZZring)
        minpoly = ZZring.drop(0).from_dense(domain.mod.to_list())
        h = _func_field_modgcd_m(f_, g_, minpoly)
        h = _to_ANP_poly(h, ring)
    else:
        (contx0f, f) = _primitive_in_x0(f)
        (contx0g, g) = _primitive_in_x0(g)
        contx0h = func_field_modgcd(contx0f, contx0g)[0]
        ZZring_ = ZZring.drop_to_ground(*range(1, n))
        f_ = _to_ZZ_poly(f, ZZring_)
        g_ = _to_ZZ_poly(g, ZZring_)
        minpoly = _minpoly_from_dense(domain.mod, ZZring_.drop(0))
        h = _func_field_modgcd_m(f_, g_, minpoly)
        h = _to_ANP_poly(h, ring)
        (contx0h_, h) = _primitive_in_x0(h)
        h *= contx0h.set_ring(ring)
        f *= contx0f.set_ring(ring)
        g *= contx0g.set_ring(ring)
    h = h.quo_ground(h.LC)
    return (h, f.quo(h), g.quo(h))