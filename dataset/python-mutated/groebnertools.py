"""Groebner bases algorithms. """
from sympy.core.symbol import Dummy
from sympy.polys.monomials import monomial_mul, monomial_lcm, monomial_divides, term_div
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import DomainError
from sympy.polys.polyconfig import query

def groebner(seq, ring, method=None):
    if False:
        print('Hello World!')
    '\n    Computes Groebner basis for a set of polynomials in `K[X]`.\n\n    Wrapper around the (default) improved Buchberger and the other algorithms\n    for computing Groebner bases. The choice of algorithm can be changed via\n    ``method`` argument or :func:`sympy.polys.polyconfig.setup`, where\n    ``method`` can be either ``buchberger`` or ``f5b``.\n\n    '
    if method is None:
        method = query('groebner')
    _groebner_methods = {'buchberger': _buchberger, 'f5b': _f5b}
    try:
        _groebner = _groebner_methods[method]
    except KeyError:
        raise ValueError("'%s' is not a valid Groebner bases algorithm (valid are 'buchberger' and 'f5b')" % method)
    (domain, orig) = (ring.domain, None)
    if not domain.is_Field or not domain.has_assoc_Field:
        try:
            (orig, ring) = (ring, ring.clone(domain=domain.get_field()))
        except DomainError:
            raise DomainError('Cannot compute a Groebner basis over %s' % domain)
        else:
            seq = [s.set_ring(ring) for s in seq]
    G = _groebner(seq, ring)
    if orig is not None:
        G = [g.clear_denoms()[1].set_ring(orig) for g in G]
    return G

def _buchberger(f, ring):
    if False:
        while True:
            i = 10
    "\n    Computes Groebner basis for a set of polynomials in `K[X]`.\n\n    Given a set of multivariate polynomials `F`, finds another\n    set `G`, such that Ideal `F = Ideal G` and `G` is a reduced\n    Groebner basis.\n\n    The resulting basis is unique and has monic generators if the\n    ground domains is a field. Otherwise the result is non-unique\n    but Groebner bases over e.g. integers can be computed (if the\n    input polynomials are monic).\n\n    Groebner bases can be used to choose specific generators for a\n    polynomial ideal. Because these bases are unique you can check\n    for ideal equality by comparing the Groebner bases.  To see if\n    one polynomial lies in an ideal, divide by the elements in the\n    base and see if the remainder vanishes.\n\n    They can also be used to solve systems of polynomial equations\n    as,  by choosing lexicographic ordering,  you can eliminate one\n    variable at a time, provided that the ideal is zero-dimensional\n    (finite number of solutions).\n\n    Notes\n    =====\n\n    Algorithm used: an improved version of Buchberger's algorithm\n    as presented in T. Becker, V. Weispfenning, Groebner Bases: A\n    Computational Approach to Commutative Algebra, Springer, 1993,\n    page 232.\n\n    References\n    ==========\n\n    .. [1] [Bose03]_\n    .. [2] [Giovini91]_\n    .. [3] [Ajwa95]_\n    .. [4] [Cox97]_\n\n    "
    order = ring.order
    monomial_mul = ring.monomial_mul
    monomial_div = ring.monomial_div
    monomial_lcm = ring.monomial_lcm

    def select(P):
        if False:
            i = 10
            return i + 15
        pr = min(P, key=lambda pair: order(monomial_lcm(f[pair[0]].LM, f[pair[1]].LM)))
        return pr

    def normal(g, J):
        if False:
            print('Hello World!')
        h = g.rem([f[j] for j in J])
        if not h:
            return None
        else:
            h = h.monic()
            if h not in I:
                I[h] = len(f)
                f.append(h)
            return (h.LM, I[h])

    def update(G, B, ih):
        if False:
            for i in range(10):
                print('nop')
        h = f[ih]
        mh = h.LM
        C = G.copy()
        D = set()
        while C:
            ig = C.pop()
            g = f[ig]
            mg = g.LM
            LCMhg = monomial_lcm(mh, mg)

            def lcm_divides(ip):
                if False:
                    i = 10
                    return i + 15
                m = monomial_lcm(mh, f[ip].LM)
                return monomial_div(LCMhg, m)
            if monomial_mul(mh, mg) == LCMhg or (not any((lcm_divides(ipx) for ipx in C)) and (not any((lcm_divides(pr[1]) for pr in D)))):
                D.add((ih, ig))
        E = set()
        while D:
            (ih, ig) = D.pop()
            mg = f[ig].LM
            LCMhg = monomial_lcm(mh, mg)
            if not monomial_mul(mh, mg) == LCMhg:
                E.add((ih, ig))
        B_new = set()
        while B:
            (ig1, ig2) = B.pop()
            mg1 = f[ig1].LM
            mg2 = f[ig2].LM
            LCM12 = monomial_lcm(mg1, mg2)
            if not monomial_div(LCM12, mh) or monomial_lcm(mg1, mh) == LCM12 or monomial_lcm(mg2, mh) == LCM12:
                B_new.add((ig1, ig2))
        B_new |= E
        G_new = set()
        while G:
            ig = G.pop()
            mg = f[ig].LM
            if not monomial_div(mg, mh):
                G_new.add(ig)
        G_new.add(ih)
        return (G_new, B_new)
    if not f:
        return []
    f1 = f[:]
    while True:
        f = f1[:]
        f1 = []
        for i in range(len(f)):
            p = f[i]
            r = p.rem(f[:i])
            if r:
                f1.append(r.monic())
        if f == f1:
            break
    I = {}
    F = set()
    G = set()
    CP = set()
    for (i, h) in enumerate(f):
        I[h] = i
        F.add(i)
    while F:
        h = min([f[x] for x in F], key=lambda f: order(f.LM))
        ih = I[h]
        F.remove(ih)
        (G, CP) = update(G, CP, ih)
    reductions_to_zero = 0
    while CP:
        (ig1, ig2) = select(CP)
        CP.remove((ig1, ig2))
        h = spoly(f[ig1], f[ig2], ring)
        G1 = sorted(G, key=lambda g: order(f[g].LM))
        ht = normal(h, G1)
        if ht:
            (G, CP) = update(G, CP, ht[1])
        else:
            reductions_to_zero += 1
    Gr = set()
    for ig in G:
        ht = normal(f[ig], G - {ig})
        if ht:
            Gr.add(ht[1])
    Gr = [f[ig] for ig in Gr]
    Gr = sorted(Gr, key=lambda f: order(f.LM), reverse=True)
    return Gr

def spoly(p1, p2, ring):
    if False:
        return 10
    '\n    Compute LCM(LM(p1), LM(p2))/LM(p1)*p1 - LCM(LM(p1), LM(p2))/LM(p2)*p2\n    This is the S-poly provided p1 and p2 are monic\n    '
    LM1 = p1.LM
    LM2 = p2.LM
    LCM12 = ring.monomial_lcm(LM1, LM2)
    m1 = ring.monomial_div(LCM12, LM1)
    m2 = ring.monomial_div(LCM12, LM2)
    s1 = p1.mul_monom(m1)
    s2 = p2.mul_monom(m2)
    s = s1 - s2
    return s

def Sign(f):
    if False:
        print('Hello World!')
    return f[0]

def Polyn(f):
    if False:
        print('Hello World!')
    return f[1]

def Num(f):
    if False:
        i = 10
        return i + 15
    return f[2]

def sig(monomial, index):
    if False:
        return 10
    return (monomial, index)

def lbp(signature, polynomial, number):
    if False:
        while True:
            i = 10
    return (signature, polynomial, number)

def sig_cmp(u, v, order):
    if False:
        print('Hello World!')
    '\n    Compare two signatures by extending the term order to K[X]^n.\n\n    u < v iff\n        - the index of v is greater than the index of u\n    or\n        - the index of v is equal to the index of u and u[0] < v[0] w.r.t. order\n\n    u > v otherwise\n    '
    if u[1] > v[1]:
        return -1
    if u[1] == v[1]:
        if order(u[0]) < order(v[0]):
            return -1
    return 1

def sig_key(s, order):
    if False:
        print('Hello World!')
    '\n    Key for comparing two signatures.\n\n    s = (m, k), t = (n, l)\n\n    s < t iff [k > l] or [k == l and m < n]\n    s > t otherwise\n    '
    return (-s[1], order(s[0]))

def sig_mult(s, m):
    if False:
        return 10
    '\n    Multiply a signature by a monomial.\n\n    The product of a signature (m, i) and a monomial n is defined as\n    (m * t, i).\n    '
    return sig(monomial_mul(s[0], m), s[1])

def lbp_sub(f, g):
    if False:
        print('Hello World!')
    '\n    Subtract labeled polynomial g from f.\n\n    The signature and number of the difference of f and g are signature\n    and number of the maximum of f and g, w.r.t. lbp_cmp.\n    '
    if sig_cmp(Sign(f), Sign(g), Polyn(f).ring.order) < 0:
        max_poly = g
    else:
        max_poly = f
    ret = Polyn(f) - Polyn(g)
    return lbp(Sign(max_poly), ret, Num(max_poly))

def lbp_mul_term(f, cx):
    if False:
        return 10
    '\n    Multiply a labeled polynomial with a term.\n\n    The product of a labeled polynomial (s, p, k) by a monomial is\n    defined as (m * s, m * p, k).\n    '
    return lbp(sig_mult(Sign(f), cx[0]), Polyn(f).mul_term(cx), Num(f))

def lbp_cmp(f, g):
    if False:
        print('Hello World!')
    '\n    Compare two labeled polynomials.\n\n    f < g iff\n        - Sign(f) < Sign(g)\n    or\n        - Sign(f) == Sign(g) and Num(f) > Num(g)\n\n    f > g otherwise\n    '
    if sig_cmp(Sign(f), Sign(g), Polyn(f).ring.order) == -1:
        return -1
    if Sign(f) == Sign(g):
        if Num(f) > Num(g):
            return -1
    return 1

def lbp_key(f):
    if False:
        for i in range(10):
            print('nop')
    '\n    Key for comparing two labeled polynomials.\n    '
    return (sig_key(Sign(f), Polyn(f).ring.order), -Num(f))

def critical_pair(f, g, ring):
    if False:
        return 10
    '\n    Compute the critical pair corresponding to two labeled polynomials.\n\n    A critical pair is a tuple (um, f, vm, g), where um and vm are\n    terms such that um * f - vm * g is the S-polynomial of f and g (so,\n    wlog assume um * f > vm * g).\n    For performance sake, a critical pair is represented as a tuple\n    (Sign(um * f), um, f, Sign(vm * g), vm, g), since um * f creates\n    a new, relatively expensive object in memory, whereas Sign(um *\n    f) and um are lightweight and f (in the tuple) is a reference to\n    an already existing object in memory.\n    '
    domain = ring.domain
    ltf = Polyn(f).LT
    ltg = Polyn(g).LT
    lt = (monomial_lcm(ltf[0], ltg[0]), domain.one)
    um = term_div(lt, ltf, domain)
    vm = term_div(lt, ltg, domain)
    fr = lbp_mul_term(lbp(Sign(f), Polyn(f).leading_term(), Num(f)), um)
    gr = lbp_mul_term(lbp(Sign(g), Polyn(g).leading_term(), Num(g)), vm)
    if lbp_cmp(fr, gr) == -1:
        return (Sign(gr), vm, g, Sign(fr), um, f)
    else:
        return (Sign(fr), um, f, Sign(gr), vm, g)

def cp_cmp(c, d):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compare two critical pairs c and d.\n\n    c < d iff\n        - lbp(c[0], _, Num(c[2]) < lbp(d[0], _, Num(d[2])) (this\n        corresponds to um_c * f_c and um_d * f_d)\n    or\n        - lbp(c[0], _, Num(c[2]) >< lbp(d[0], _, Num(d[2])) and\n        lbp(c[3], _, Num(c[5])) < lbp(d[3], _, Num(d[5])) (this\n        corresponds to vm_c * g_c and vm_d * g_d)\n\n    c > d otherwise\n    '
    zero = Polyn(c[2]).ring.zero
    c0 = lbp(c[0], zero, Num(c[2]))
    d0 = lbp(d[0], zero, Num(d[2]))
    r = lbp_cmp(c0, d0)
    if r == -1:
        return -1
    if r == 0:
        c1 = lbp(c[3], zero, Num(c[5]))
        d1 = lbp(d[3], zero, Num(d[5]))
        r = lbp_cmp(c1, d1)
        if r == -1:
            return -1
    return 1

def cp_key(c, ring):
    if False:
        print('Hello World!')
    '\n    Key for comparing critical pairs.\n    '
    return (lbp_key(lbp(c[0], ring.zero, Num(c[2]))), lbp_key(lbp(c[3], ring.zero, Num(c[5]))))

def s_poly(cp):
    if False:
        return 10
    '\n    Compute the S-polynomial of a critical pair.\n\n    The S-polynomial of a critical pair cp is cp[1] * cp[2] - cp[4] * cp[5].\n    '
    return lbp_sub(lbp_mul_term(cp[2], cp[1]), lbp_mul_term(cp[5], cp[4]))

def is_rewritable_or_comparable(sign, num, B):
    if False:
        return 10
    '\n    Check if a labeled polynomial is redundant by checking if its\n    signature and number imply rewritability or comparability.\n\n    (sign, num) is comparable if there exists a labeled polynomial\n    h in B, such that sign[1] (the index) is less than Sign(h)[1]\n    and sign[0] is divisible by the leading monomial of h.\n\n    (sign, num) is rewritable if there exists a labeled polynomial\n    h in B, such thatsign[1] is equal to Sign(h)[1], num < Num(h)\n    and sign[0] is divisible by Sign(h)[0].\n    '
    for h in B:
        if sign[1] < Sign(h)[1]:
            if monomial_divides(Polyn(h).LM, sign[0]):
                return True
        if sign[1] == Sign(h)[1]:
            if num < Num(h):
                if monomial_divides(Sign(h)[0], sign[0]):
                    return True
    return False

def f5_reduce(f, B):
    if False:
        while True:
            i = 10
    '\n    F5-reduce a labeled polynomial f by B.\n\n    Continuously searches for non-zero labeled polynomial h in B, such\n    that the leading term lt_h of h divides the leading term lt_f of\n    f and Sign(lt_h * h) < Sign(f). If such a labeled polynomial h is\n    found, f gets replaced by f - lt_f / lt_h * h. If no such h can be\n    found or f is 0, f is no further F5-reducible and f gets returned.\n\n    A polynomial that is reducible in the usual sense need not be\n    F5-reducible, e.g.:\n\n    >>> from sympy.polys.groebnertools import lbp, sig, f5_reduce, Polyn\n    >>> from sympy.polys import ring, QQ, lex\n\n    >>> R, x,y,z = ring("x,y,z", QQ, lex)\n\n    >>> f = lbp(sig((1, 1, 1), 4), x, 3)\n    >>> g = lbp(sig((0, 0, 0), 2), x, 2)\n\n    >>> Polyn(f).rem([Polyn(g)])\n    0\n    >>> f5_reduce(f, [g])\n    (((1, 1, 1), 4), x, 3)\n\n    '
    order = Polyn(f).ring.order
    domain = Polyn(f).ring.domain
    if not Polyn(f):
        return f
    while True:
        g = f
        for h in B:
            if Polyn(h):
                if monomial_divides(Polyn(h).LM, Polyn(f).LM):
                    t = term_div(Polyn(f).LT, Polyn(h).LT, domain)
                    if sig_cmp(sig_mult(Sign(h), t[0]), Sign(f), order) < 0:
                        hp = lbp_mul_term(h, t)
                        f = lbp_sub(f, hp)
                        break
        if g == f or not Polyn(f):
            return f

def _f5b(F, ring):
    if False:
        i = 10
        return i + 15
    '\n    Computes a reduced Groebner basis for the ideal generated by F.\n\n    f5b is an implementation of the F5B algorithm by Yao Sun and\n    Dingkang Wang. Similarly to Buchberger\'s algorithm, the algorithm\n    proceeds by computing critical pairs, computing the S-polynomial,\n    reducing it and adjoining the reduced S-polynomial if it is not 0.\n\n    Unlike Buchberger\'s algorithm, each polynomial contains additional\n    information, namely a signature and a number. The signature\n    specifies the path of computation (i.e. from which polynomial in\n    the original basis was it derived and how), the number says when\n    the polynomial was added to the basis.  With this information it\n    is (often) possible to decide if an S-polynomial will reduce to\n    0 and can be discarded.\n\n    Optimizations include: Reducing the generators before computing\n    a Groebner basis, removing redundant critical pairs when a new\n    polynomial enters the basis and sorting the critical pairs and\n    the current basis.\n\n    Once a Groebner basis has been found, it gets reduced.\n\n    References\n    ==========\n\n    .. [1] Yao Sun, Dingkang Wang: "A New Proof for the Correctness of F5\n           (F5-Like) Algorithm", https://arxiv.org/abs/1004.0084 (specifically\n           v4)\n\n    .. [2] Thomas Becker, Volker Weispfenning, Groebner bases: A computational\n           approach to commutative algebra, 1993, p. 203, 216\n    '
    order = ring.order
    B = F
    while True:
        F = B
        B = []
        for i in range(len(F)):
            p = F[i]
            r = p.rem(F[:i])
            if r:
                B.append(r)
        if F == B:
            break
    B = [lbp(sig(ring.zero_monom, i + 1), F[i], i + 1) for i in range(len(F))]
    B.sort(key=lambda f: order(Polyn(f).LM), reverse=True)
    CP = [critical_pair(B[i], B[j], ring) for i in range(len(B)) for j in range(i + 1, len(B))]
    CP.sort(key=lambda cp: cp_key(cp, ring), reverse=True)
    k = len(B)
    reductions_to_zero = 0
    while len(CP):
        cp = CP.pop()
        if is_rewritable_or_comparable(cp[0], Num(cp[2]), B):
            continue
        if is_rewritable_or_comparable(cp[3], Num(cp[5]), B):
            continue
        s = s_poly(cp)
        p = f5_reduce(s, B)
        p = lbp(Sign(p), Polyn(p).monic(), k + 1)
        if Polyn(p):
            indices = []
            for (i, cp) in enumerate(CP):
                if is_rewritable_or_comparable(cp[0], Num(cp[2]), [p]):
                    indices.append(i)
                elif is_rewritable_or_comparable(cp[3], Num(cp[5]), [p]):
                    indices.append(i)
            for i in reversed(indices):
                del CP[i]
            for g in B:
                if Polyn(g):
                    cp = critical_pair(p, g, ring)
                    if is_rewritable_or_comparable(cp[0], Num(cp[2]), [p]):
                        continue
                    elif is_rewritable_or_comparable(cp[3], Num(cp[5]), [p]):
                        continue
                    CP.append(cp)
            CP.sort(key=lambda cp: cp_key(cp, ring), reverse=True)
            m = Polyn(p).LM
            if order(m) <= order(Polyn(B[-1]).LM):
                B.append(p)
            else:
                for (i, q) in enumerate(B):
                    if order(m) > order(Polyn(q).LM):
                        B.insert(i, p)
                        break
            k += 1
        else:
            reductions_to_zero += 1
    H = [Polyn(g).monic() for g in B]
    H = red_groebner(H, ring)
    return sorted(H, key=lambda f: order(f.LM), reverse=True)

def red_groebner(G, ring):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute reduced Groebner basis, from BeckerWeispfenning93, p. 216\n\n    Selects a subset of generators, that already generate the ideal\n    and computes a reduced Groebner basis for them.\n    '

    def reduction(P):
        if False:
            i = 10
            return i + 15
        '\n        The actual reduction algorithm.\n        '
        Q = []
        for (i, p) in enumerate(P):
            h = p.rem(P[:i] + P[i + 1:])
            if h:
                Q.append(h)
        return [p.monic() for p in Q]
    F = G
    H = []
    while F:
        f0 = F.pop()
        if not any((monomial_divides(f.LM, f0.LM) for f in F + H)):
            H.append(f0)
    return reduction(H)

def is_groebner(G, ring):
    if False:
        i = 10
        return i + 15
    '\n    Check if G is a Groebner basis.\n    '
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            s = spoly(G[i], G[j], ring)
            s = s.rem(G)
            if s:
                return False
    return True

def is_minimal(G, ring):
    if False:
        return 10
    '\n    Checks if G is a minimal Groebner basis.\n    '
    order = ring.order
    domain = ring.domain
    G.sort(key=lambda g: order(g.LM))
    for (i, g) in enumerate(G):
        if g.LC != domain.one:
            return False
        for h in G[:i] + G[i + 1:]:
            if monomial_divides(h.LM, g.LM):
                return False
    return True

def is_reduced(G, ring):
    if False:
        for i in range(10):
            print('nop')
    '\n    Checks if G is a reduced Groebner basis.\n    '
    order = ring.order
    domain = ring.domain
    G.sort(key=lambda g: order(g.LM))
    for (i, g) in enumerate(G):
        if g.LC != domain.one:
            return False
        for term in g.terms():
            for h in G[:i] + G[i + 1:]:
                if monomial_divides(h.LM, term[0]):
                    return False
    return True

def groebner_lcm(f, g):
    if False:
        i = 10
        return i + 15
    '\n    Computes LCM of two polynomials using Groebner bases.\n\n    The LCM is computed as the unique generator of the intersection\n    of the two ideals generated by `f` and `g`. The approach is to\n    compute a Groebner basis with respect to lexicographic ordering\n    of `t*f` and `(1 - t)*g`, where `t` is an unrelated variable and\n    then filtering out the solution that does not contain `t`.\n\n    References\n    ==========\n\n    .. [1] [Cox97]_\n\n    '
    if f.ring != g.ring:
        raise ValueError('Values should be equal')
    ring = f.ring
    domain = ring.domain
    if not f or not g:
        return ring.zero
    if len(f) <= 1 and len(g) <= 1:
        monom = monomial_lcm(f.LM, g.LM)
        coeff = domain.lcm(f.LC, g.LC)
        return ring.term_new(monom, coeff)
    (fc, f) = f.primitive()
    (gc, g) = g.primitive()
    lcm = domain.lcm(fc, gc)
    f_terms = [((1,) + monom, coeff) for (monom, coeff) in f.terms()]
    g_terms = [((0,) + monom, coeff) for (monom, coeff) in g.terms()] + [((1,) + monom, -coeff) for (monom, coeff) in g.terms()]
    t = Dummy('t')
    t_ring = ring.clone(symbols=(t,) + ring.symbols, order=lex)
    F = t_ring.from_terms(f_terms)
    G = t_ring.from_terms(g_terms)
    basis = groebner([F, G], t_ring)

    def is_independent(h, j):
        if False:
            return 10
        return not any((monom[j] for monom in h.monoms()))
    H = [h for h in basis if is_independent(h, 0)]
    h_terms = [(monom[1:], coeff * lcm) for (monom, coeff) in H[0].terms()]
    h = ring.from_terms(h_terms)
    return h

def groebner_gcd(f, g):
    if False:
        for i in range(10):
            print('nop')
    'Computes GCD of two polynomials using Groebner bases. '
    if f.ring != g.ring:
        raise ValueError('Values should be equal')
    domain = f.ring.domain
    if not domain.is_Field:
        (fc, f) = f.primitive()
        (gc, g) = g.primitive()
        gcd = domain.gcd(fc, gc)
    H = (f * g).quo([groebner_lcm(f, g)])
    if len(H) != 1:
        raise ValueError('Length should be 1')
    h = H[0]
    if not domain.is_Field:
        return gcd * h
    else:
        return h.monic()