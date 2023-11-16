from sympy.core import S
from sympy.polys import Poly

def dispersionset(p, q=None, *gens, **args):
    if False:
        while True:
            i = 10
    "Compute the *dispersion set* of two polynomials.\n\n    For two polynomials `f(x)` and `g(x)` with `\\deg f > 0`\n    and `\\deg g > 0` the dispersion set `\\operatorname{J}(f, g)` is defined as:\n\n    .. math::\n        \\operatorname{J}(f, g)\n        & := \\{a \\in \\mathbb{N}_0 | \\gcd(f(x), g(x+a)) \\neq 1\\} \\\\\n        &  = \\{a \\in \\mathbb{N}_0 | \\deg \\gcd(f(x), g(x+a)) \\geq 1\\}\n\n    For a single polynomial one defines `\\operatorname{J}(f) := \\operatorname{J}(f, f)`.\n\n    Examples\n    ========\n\n    >>> from sympy import poly\n    >>> from sympy.polys.dispersion import dispersion, dispersionset\n    >>> from sympy.abc import x\n\n    Dispersion set and dispersion of a simple polynomial:\n\n    >>> fp = poly((x - 3)*(x + 3), x)\n    >>> sorted(dispersionset(fp))\n    [0, 6]\n    >>> dispersion(fp)\n    6\n\n    Note that the definition of the dispersion is not symmetric:\n\n    >>> fp = poly(x**4 - 3*x**2 + 1, x)\n    >>> gp = fp.shift(-3)\n    >>> sorted(dispersionset(fp, gp))\n    [2, 3, 4]\n    >>> dispersion(fp, gp)\n    4\n    >>> sorted(dispersionset(gp, fp))\n    []\n    >>> dispersion(gp, fp)\n    -oo\n\n    Computing the dispersion also works over field extensions:\n\n    >>> from sympy import sqrt\n    >>> fp = poly(x**2 + sqrt(5)*x - 1, x, domain='QQ<sqrt(5)>')\n    >>> gp = poly(x**2 + (2 + sqrt(5))*x + sqrt(5), x, domain='QQ<sqrt(5)>')\n    >>> sorted(dispersionset(fp, gp))\n    [2]\n    >>> sorted(dispersionset(gp, fp))\n    [1, 4]\n\n    We can even perform the computations for polynomials\n    having symbolic coefficients:\n\n    >>> from sympy.abc import a\n    >>> fp = poly(4*x**4 + (4*a + 8)*x**3 + (a**2 + 6*a + 4)*x**2 + (a**2 + 2*a)*x, x)\n    >>> sorted(dispersionset(fp))\n    [0, 1]\n\n    See Also\n    ========\n\n    dispersion\n\n    References\n    ==========\n\n    .. [1] [ManWright94]_\n    .. [2] [Koepf98]_\n    .. [3] [Abramov71]_\n    .. [4] [Man93]_\n    "
    same = False if q is not None else True
    if same:
        q = p
    p = Poly(p, *gens, **args)
    q = Poly(q, *gens, **args)
    if not p.is_univariate or not q.is_univariate:
        raise ValueError('Polynomials need to be univariate')
    if not p.gen == q.gen:
        raise ValueError('Polynomials must have the same generator')
    gen = p.gen
    if p.degree() < 1 or q.degree() < 1:
        return {0}
    fp = p.factor_list()
    fq = q.factor_list() if not same else fp
    J = set()
    for (s, unused) in fp[1]:
        for (t, unused) in fq[1]:
            m = s.degree()
            n = t.degree()
            if n != m:
                continue
            an = s.LC()
            bn = t.LC()
            if not (an - bn).is_zero:
                continue
            anm1 = s.coeff_monomial(gen ** (m - 1))
            bnm1 = t.coeff_monomial(gen ** (n - 1))
            alpha = (anm1 - bnm1) / S(n * bn)
            if not alpha.is_integer:
                continue
            if alpha < 0 or alpha in J:
                continue
            if n > 1 and (not (s - t.shift(alpha)).is_zero):
                continue
            J.add(alpha)
    return J

def dispersion(p, q=None, *gens, **args):
    if False:
        print('Hello World!')
    "Compute the *dispersion* of polynomials.\n\n    For two polynomials `f(x)` and `g(x)` with `\\deg f > 0`\n    and `\\deg g > 0` the dispersion `\\operatorname{dis}(f, g)` is defined as:\n\n    .. math::\n        \\operatorname{dis}(f, g)\n        & := \\max\\{ J(f,g) \\cup \\{0\\} \\} \\\\\n        &  = \\max\\{ \\{a \\in \\mathbb{N} | \\gcd(f(x), g(x+a)) \\neq 1\\} \\cup \\{0\\} \\}\n\n    and for a single polynomial `\\operatorname{dis}(f) := \\operatorname{dis}(f, f)`.\n    Note that we make the definition `\\max\\{\\} := -\\infty`.\n\n    Examples\n    ========\n\n    >>> from sympy import poly\n    >>> from sympy.polys.dispersion import dispersion, dispersionset\n    >>> from sympy.abc import x\n\n    Dispersion set and dispersion of a simple polynomial:\n\n    >>> fp = poly((x - 3)*(x + 3), x)\n    >>> sorted(dispersionset(fp))\n    [0, 6]\n    >>> dispersion(fp)\n    6\n\n    Note that the definition of the dispersion is not symmetric:\n\n    >>> fp = poly(x**4 - 3*x**2 + 1, x)\n    >>> gp = fp.shift(-3)\n    >>> sorted(dispersionset(fp, gp))\n    [2, 3, 4]\n    >>> dispersion(fp, gp)\n    4\n    >>> sorted(dispersionset(gp, fp))\n    []\n    >>> dispersion(gp, fp)\n    -oo\n\n    The maximum of an empty set is defined to be `-\\infty`\n    as seen in this example.\n\n    Computing the dispersion also works over field extensions:\n\n    >>> from sympy import sqrt\n    >>> fp = poly(x**2 + sqrt(5)*x - 1, x, domain='QQ<sqrt(5)>')\n    >>> gp = poly(x**2 + (2 + sqrt(5))*x + sqrt(5), x, domain='QQ<sqrt(5)>')\n    >>> sorted(dispersionset(fp, gp))\n    [2]\n    >>> sorted(dispersionset(gp, fp))\n    [1, 4]\n\n    We can even perform the computations for polynomials\n    having symbolic coefficients:\n\n    >>> from sympy.abc import a\n    >>> fp = poly(4*x**4 + (4*a + 8)*x**3 + (a**2 + 6*a + 4)*x**2 + (a**2 + 2*a)*x, x)\n    >>> sorted(dispersionset(fp))\n    [0, 1]\n\n    See Also\n    ========\n\n    dispersionset\n\n    References\n    ==========\n\n    .. [1] [ManWright94]_\n    .. [2] [Koepf98]_\n    .. [3] [Abramov71]_\n    .. [4] [Man93]_\n    "
    J = dispersionset(p, q, *gens, **args)
    if not J:
        j = S.NegativeInfinity
    else:
        j = max(J)
    return j