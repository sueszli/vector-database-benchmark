from sympy.combinatorics.permutations import Permutation, _af_rmul, _af_invert, _af_new
from sympy.combinatorics.perm_groups import PermutationGroup, _orbit, _orbit_transversal
from sympy.combinatorics.util import _distribute_gens_by_base, _orbits_transversals_from_bsgs
'\n    References for tensor canonicalization:\n\n    [1] R. Portugal "Algorithmic simplification of tensor expressions",\n        J. Phys. A 32 (1999) 7779-7789\n\n    [2] R. Portugal, B.F. Svaiter "Group-theoretic Approach for Symbolic\n        Tensor Manipulation: I. Free Indices"\n        arXiv:math-ph/0107031v1\n\n    [3] L.R.U. Manssur, R. Portugal "Group-theoretic Approach for Symbolic\n        Tensor Manipulation: II. Dummy Indices"\n        arXiv:math-ph/0107032v1\n\n    [4] xperm.c part of XPerm written by J. M. Martin-Garcia\n        http://www.xact.es/index.html\n'

def dummy_sgs(dummies, sym, n):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the strong generators for dummy indices.\n\n    Parameters\n    ==========\n\n    dummies : List of dummy indices.\n        `dummies[2k], dummies[2k+1]` are paired indices.\n        In base form, the dummy indices are always in\n        consecutive positions.\n    sym : symmetry under interchange of contracted dummies::\n        * None  no symmetry\n        * 0     commuting\n        * 1     anticommuting\n\n    n : number of indices\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.tensor_can import dummy_sgs\n    >>> dummy_sgs(list(range(2, 8)), 0, 8)\n    [[0, 1, 3, 2, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 5, 4, 6, 7, 8, 9],\n     [0, 1, 2, 3, 4, 5, 7, 6, 8, 9], [0, 1, 4, 5, 2, 3, 6, 7, 8, 9],\n     [0, 1, 2, 3, 6, 7, 4, 5, 8, 9]]\n    '
    if len(dummies) > n:
        raise ValueError('List too large')
    res = []
    if sym is not None:
        for j in dummies[::2]:
            a = list(range(n + 2))
            if sym == 1:
                a[n] = n + 1
                a[n + 1] = n
            (a[j], a[j + 1]) = (a[j + 1], a[j])
            res.append(a)
    for j in dummies[:-3:2]:
        a = list(range(n + 2))
        a[j:j + 4] = (a[j + 2], a[j + 3], a[j], a[j + 1])
        res.append(a)
    return res

def _min_dummies(dummies, sym, indices):
    if False:
        return 10
    '\n    Return list of minima of the orbits of indices in group of dummies.\n    See ``double_coset_can_rep`` for the description of ``dummies`` and ``sym``.\n    ``indices`` is the initial list of dummy indices.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.tensor_can import _min_dummies\n    >>> _min_dummies([list(range(2, 8))], [0], list(range(10)))\n    [0, 1, 2, 2, 2, 2, 2, 2, 8, 9]\n    '
    num_types = len(sym)
    m = [min(dx) if dx else None for dx in dummies]
    res = indices[:]
    for i in range(num_types):
        for (c, i) in enumerate(indices):
            for j in range(num_types):
                if i in dummies[j]:
                    res[c] = m[j]
                    break
    return res

def _trace_S(s, j, b, S_cosets):
    if False:
        print('Hello World!')
    '\n    Return the representative h satisfying s[h[b]] == j\n\n    If there is not such a representative return None\n    '
    for h in S_cosets[b]:
        if s[h[b]] == j:
            return h
    return None

def _trace_D(gj, p_i, Dxtrav):
    if False:
        return 10
    '\n    Return the representative h satisfying h[gj] == p_i\n\n    If there is not such a representative return None\n    '
    for h in Dxtrav:
        if h[gj] == p_i:
            return h
    return None

def _dumx_remove(dumx, dumx_flat, p0):
    if False:
        for i in range(10):
            print('nop')
    '\n    remove p0 from dumx\n    '
    res = []
    for dx in dumx:
        if p0 not in dx:
            res.append(dx)
            continue
        k = dx.index(p0)
        if k % 2 == 0:
            p0_paired = dx[k + 1]
        else:
            p0_paired = dx[k - 1]
        dx.remove(p0)
        dx.remove(p0_paired)
        dumx_flat.remove(p0)
        dumx_flat.remove(p0_paired)
        res.append(dx)

def transversal2coset(size, base, transversal):
    if False:
        print('Hello World!')
    a = []
    j = 0
    for i in range(size):
        if i in base:
            a.append(sorted(transversal[j].values()))
            j += 1
        else:
            a.append([list(range(size))])
    j = len(a) - 1
    while a[j] == [list(range(size))]:
        j -= 1
    return a[:j + 1]

def double_coset_can_rep(dummies, sym, b_S, sgens, S_transversals, g):
    if False:
        for i in range(10):
            print('nop')
    '\n    Butler-Portugal algorithm for tensor canonicalization with dummy indices.\n\n    Parameters\n    ==========\n\n      dummies\n        list of lists of dummy indices,\n        one list for each type of index;\n        the dummy indices are put in order contravariant, covariant\n        [d0, -d0, d1, -d1, ...].\n\n      sym\n        list of the symmetries of the index metric for each type.\n\n      possible symmetries of the metrics\n              * 0     symmetric\n              * 1     antisymmetric\n              * None  no symmetry\n\n      b_S\n        base of a minimal slot symmetry BSGS.\n\n      sgens\n        generators of the slot symmetry BSGS.\n\n      S_transversals\n        transversals for the slot BSGS.\n\n      g\n        permutation representing the tensor.\n\n    Returns\n    =======\n\n    Return 0 if the tensor is zero, else return the array form of\n    the permutation representing the canonical form of the tensor.\n\n    Notes\n    =====\n\n    A tensor with dummy indices can be represented in a number\n    of equivalent ways which typically grows exponentially with\n    the number of indices. To be able to establish if two tensors\n    with many indices are equal becomes computationally very slow\n    in absence of an efficient algorithm.\n\n    The Butler-Portugal algorithm [3] is an efficient algorithm to\n    put tensors in canonical form, solving the above problem.\n\n    Portugal observed that a tensor can be represented by a permutation,\n    and that the class of tensors equivalent to it under slot and dummy\n    symmetries is equivalent to the double coset `D*g*S`\n    (Note: in this documentation we use the conventions for multiplication\n    of permutations p, q with (p*q)(i) = p[q[i]] which is opposite\n    to the one used in the Permutation class)\n\n    Using the algorithm by Butler to find a representative of the\n    double coset one can find a canonical form for the tensor.\n\n    To see this correspondence,\n    let `g` be a permutation in array form; a tensor with indices `ind`\n    (the indices including both the contravariant and the covariant ones)\n    can be written as\n\n    `t = T(ind[g[0]], \\dots, ind[g[n-1]])`,\n\n    where `n = len(ind)`;\n    `g` has size `n + 2`, the last two indices for the sign of the tensor\n    (trick introduced in [4]).\n\n    A slot symmetry transformation `s` is a permutation acting on the slots\n    `t \\rightarrow T(ind[(g*s)[0]], \\dots, ind[(g*s)[n-1]])`\n\n    A dummy symmetry transformation acts on `ind`\n    `t \\rightarrow T(ind[(d*g)[0]], \\dots, ind[(d*g)[n-1]])`\n\n    Being interested only in the transformations of the tensor under\n    these symmetries, one can represent the tensor by `g`, which transforms\n    as\n\n    `g -> d*g*s`, so it belongs to the coset `D*g*S`, or in other words\n    to the set of all permutations allowed by the slot and dummy symmetries.\n\n    Let us explain the conventions by an example.\n\n    Given a tensor `T^{d3 d2 d1}{}_{d1 d2 d3}` with the slot symmetries\n          `T^{a0 a1 a2 a3 a4 a5} = -T^{a2 a1 a0 a3 a4 a5}`\n\n          `T^{a0 a1 a2 a3 a4 a5} = -T^{a4 a1 a2 a3 a0 a5}`\n\n    and symmetric metric, find the tensor equivalent to it which\n    is the lowest under the ordering of indices:\n    lexicographic ordering `d1, d2, d3` and then contravariant\n    before covariant index; that is the canonical form of the tensor.\n\n    The canonical form is `-T^{d1 d2 d3}{}_{d1 d2 d3}`\n    obtained using `T^{a0 a1 a2 a3 a4 a5} = -T^{a2 a1 a0 a3 a4 a5}`.\n\n    To convert this problem in the input for this function,\n    use the following ordering of the index names\n    (- for covariant for short) `d1, -d1, d2, -d2, d3, -d3`\n\n    `T^{d3 d2 d1}{}_{d1 d2 d3}` corresponds to `g = [4, 2, 0, 1, 3, 5, 6, 7]`\n    where the last two indices are for the sign\n\n    `sgens = [Permutation(0, 2)(6, 7), Permutation(0, 4)(6, 7)]`\n\n    sgens[0] is the slot symmetry `-(0, 2)`\n    `T^{a0 a1 a2 a3 a4 a5} = -T^{a2 a1 a0 a3 a4 a5}`\n\n    sgens[1] is the slot symmetry `-(0, 4)`\n    `T^{a0 a1 a2 a3 a4 a5} = -T^{a4 a1 a2 a3 a0 a5}`\n\n    The dummy symmetry group D is generated by the strong base generators\n    `[(0, 1), (2, 3), (4, 5), (0, 2)(1, 3), (0, 4)(1, 5)]`\n    where the first three interchange covariant and contravariant\n    positions of the same index (d1 <-> -d1) and the last two interchange\n    the dummy indices themselves (d1 <-> d2).\n\n    The dummy symmetry acts from the left\n    `d = [1, 0, 2, 3, 4, 5, 6, 7]`  exchange `d1 \\leftrightarrow -d1`\n    `T^{d3 d2 d1}{}_{d1 d2 d3} == T^{d3 d2}{}_{d1}{}^{d1}{}_{d2 d3}`\n\n    `g=[4, 2, 0, 1, 3, 5, 6, 7]  -> [4, 2, 1, 0, 3, 5, 6, 7] = _af_rmul(d, g)`\n    which differs from `_af_rmul(g, d)`.\n\n    The slot symmetry acts from the right\n    `s = [2, 1, 0, 3, 4, 5, 7, 6]`  exchanges slots 0 and 2 and changes sign\n    `T^{d3 d2 d1}{}_{d1 d2 d3} == -T^{d1 d2 d3}{}_{d1 d2 d3}`\n\n    `g=[4,2,0,1,3,5,6,7]  -> [0, 2, 4, 1, 3, 5, 7, 6] = _af_rmul(g, s)`\n\n    Example in which the tensor is zero, same slot symmetries as above:\n    `T^{d2}{}_{d1 d3}{}^{d1 d3}{}_{d2}`\n\n    `= -T^{d3}{}_{d1 d3}{}^{d1 d2}{}_{d2}`   under slot symmetry `-(0,4)`;\n\n    `= T_{d3 d1}{}^{d3}{}^{d1 d2}{}_{d2}`    under slot symmetry `-(0,2)`;\n\n    `= T^{d3}{}_{d1 d3}{}^{d1 d2}{}_{d2}`    symmetric metric;\n\n    `= 0`  since two of these lines have tensors differ only for the sign.\n\n    The double coset D*g*S consists of permutations `h = d*g*s` corresponding\n    to equivalent tensors; if there are two `h` which are the same apart\n    from the sign, return zero; otherwise\n    choose as representative the tensor with indices\n    ordered lexicographically according to `[d1, -d1, d2, -d2, d3, -d3]`\n    that is ``rep = min(D*g*S) = min([d*g*s for d in D for s in S])``\n\n    The indices are fixed one by one; first choose the lowest index\n    for slot 0, then the lowest remaining index for slot 1, etc.\n    Doing this one obtains a chain of stabilizers\n\n    `S \\rightarrow S_{b0} \\rightarrow S_{b0,b1} \\rightarrow \\dots` and\n    `D \\rightarrow D_{p0} \\rightarrow D_{p0,p1} \\rightarrow \\dots`\n\n    where ``[b0, b1, ...] = range(b)`` is a base of the symmetric group;\n    the strong base `b_S` of S is an ordered sublist of it;\n    therefore it is sufficient to compute once the\n    strong base generators of S using the Schreier-Sims algorithm;\n    the stabilizers of the strong base generators are the\n    strong base generators of the stabilizer subgroup.\n\n    ``dbase = [p0, p1, ...]`` is not in general in lexicographic order,\n    so that one must recompute the strong base generators each time;\n    however this is trivial, there is no need to use the Schreier-Sims\n    algorithm for D.\n\n    The algorithm keeps a TAB of elements `(s_i, d_i, h_i)`\n    where `h_i = d_i \\times g \\times s_i` satisfying `h_i[j] = p_j` for `0 \\le j < i`\n    starting from `s_0 = id, d_0 = id, h_0 = g`.\n\n    The equations `h_0[0] = p_0, h_1[1] = p_1, \\dots` are solved in this order,\n    choosing each time the lowest possible value of p_i\n\n    For `j < i`\n    `d_i*g*s_i*S_{b_0, \\dots, b_{i-1}}*b_j = D_{p_0, \\dots, p_{i-1}}*p_j`\n    so that for dx in `D_{p_0,\\dots,p_{i-1}}` and sx in\n    `S_{base[0], \\dots, base[i-1]}` one has `dx*d_i*g*s_i*sx*b_j = p_j`\n\n    Search for dx, sx such that this equation holds for `j = i`;\n    it can be written as `s_i*sx*b_j = J, dx*d_i*g*J = p_j`\n    `sx*b_j = s_i**-1*J; sx = trace(s_i**-1, S_{b_0,...,b_{i-1}})`\n    `dx**-1*p_j = d_i*g*J; dx = trace(d_i*g*J, D_{p_0,...,p_{i-1}})`\n\n    `s_{i+1} = s_i*trace(s_i**-1*J, S_{b_0,...,b_{i-1}})`\n    `d_{i+1} = trace(d_i*g*J, D_{p_0,...,p_{i-1}})**-1*d_i`\n    `h_{i+1}*b_i = d_{i+1}*g*s_{i+1}*b_i = p_i`\n\n    `h_n*b_j = p_j` for all j, so that `h_n` is the solution.\n\n    Add the found `(s, d, h)` to TAB1.\n\n    At the end of the iteration sort TAB1 with respect to the `h`;\n    if there are two consecutive `h` in TAB1 which differ only for the\n    sign, the tensor is zero, so return 0;\n    if there are two consecutive `h` which are equal, keep only one.\n\n    Then stabilize the slot generators under `i` and the dummy generators\n    under `p_i`.\n\n    Assign `TAB = TAB1` at the end of the iteration step.\n\n    At the end `TAB` contains a unique `(s, d, h)`, since all the slots\n    of the tensor `h` have been fixed to have the minimum value according\n    to the symmetries. The algorithm returns `h`.\n\n    It is important that the slot BSGS has lexicographic minimal base,\n    otherwise there is an `i` which does not belong to the slot base\n    for which `p_i` is fixed by the dummy symmetry only, while `i`\n    is not invariant from the slot stabilizer, so `p_i` is not in\n    general the minimal value.\n\n    This algorithm differs slightly from the original algorithm [3]:\n      the canonical form is minimal lexicographically, and\n      the BSGS has minimal base under lexicographic order.\n      Equal tensors `h` are eliminated from TAB.\n\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.permutations import Permutation\n    >>> from sympy.combinatorics.tensor_can import double_coset_can_rep, get_transversals\n    >>> gens = [Permutation(x) for x in [[2, 1, 0, 3, 4, 5, 7, 6], [4, 1, 2, 3, 0, 5, 7, 6]]]\n    >>> base = [0, 2]\n    >>> g = Permutation([4, 2, 0, 1, 3, 5, 6, 7])\n    >>> transversals = get_transversals(base, gens)\n    >>> double_coset_can_rep([list(range(6))], [0], base, gens, transversals, g)\n    [0, 1, 2, 3, 4, 5, 7, 6]\n\n    >>> g = Permutation([4, 1, 3, 0, 5, 2, 6, 7])\n    >>> double_coset_can_rep([list(range(6))], [0], base, gens, transversals, g)\n    0\n    '
    size = g.size
    g = g.array_form
    num_dummies = size - 2
    indices = list(range(num_dummies))
    all_metrics_with_sym = not any((_ is None for _ in sym))
    num_types = len(sym)
    dumx = dummies[:]
    dumx_flat = []
    for dx in dumx:
        dumx_flat.extend(dx)
    b_S = b_S[:]
    sgensx = [h._array_form for h in sgens]
    if b_S:
        S_transversals = transversal2coset(size, b_S, S_transversals)
    dsgsx = []
    for i in range(num_types):
        dsgsx.extend(dummy_sgs(dumx[i], sym[i], num_dummies))
    idn = list(range(size))
    TAB = [(idn, idn, g)]
    for i in range(size - 2):
        b = i
        testb = b in b_S and sgensx
        if testb:
            sgensx1 = [_af_new(_) for _ in sgensx]
            deltab = _orbit(size, sgensx1, b)
        else:
            deltab = {b}
        if all_metrics_with_sym:
            md = _min_dummies(dumx, sym, indices)
        else:
            md = [min(_orbit(size, [_af_new(ddx) for ddx in dsgsx], ii)) for ii in range(size - 2)]
        p_i = min([min([md[h[x]] for x in deltab]) for (s, d, h) in TAB])
        dsgsx1 = [_af_new(_) for _ in dsgsx]
        Dxtrav = _orbit_transversal(size, dsgsx1, p_i, False, af=True) if dsgsx else None
        if Dxtrav:
            Dxtrav = [_af_invert(x) for x in Dxtrav]
        for ii in range(num_types):
            if p_i in dumx[ii]:
                if sym[ii] is not None:
                    deltap = dumx[ii]
                else:
                    p_i_index = dumx[ii].index(p_i) % 2
                    deltap = dumx[ii][p_i_index::2]
                break
        else:
            deltap = [p_i]
        TAB1 = []
        while TAB:
            (s, d, h) = TAB.pop()
            if min([md[h[x]] for x in deltab]) != p_i:
                continue
            deltab1 = [x for x in deltab if md[h[x]] == p_i]
            dg = _af_rmul(d, g)
            dginv = _af_invert(dg)
            sdeltab = [s[x] for x in deltab1]
            gdeltap = [dginv[x] for x in deltap]
            NEXT = [x for x in sdeltab if x in gdeltap]
            for j in NEXT:
                if testb:
                    s1 = _trace_S(s, j, b, S_transversals)
                    if not s1:
                        continue
                    else:
                        s1 = [s[ix] for ix in s1]
                else:
                    s1 = s
                if Dxtrav:
                    d1 = _trace_D(dg[j], p_i, Dxtrav)
                    if not d1:
                        continue
                else:
                    if p_i != dg[j]:
                        continue
                    d1 = idn
                assert d1[dg[j]] == p_i
                d1 = [d1[ix] for ix in d]
                h1 = [d1[g[ix]] for ix in s1]
                TAB1.append((s1, d1, h1))
        TAB1.sort(key=lambda x: x[-1])
        prev = [0] * size
        while TAB1:
            (s, d, h) = TAB1.pop()
            if h[:-2] == prev[:-2]:
                if h[-1] != prev[-1]:
                    return 0
            else:
                TAB.append((s, d, h))
            prev = h
        sgensx = [h for h in sgensx if h[b] == b]
        if b in b_S:
            b_S.remove(b)
        _dumx_remove(dumx, dumx_flat, p_i)
        dsgsx = []
        for i in range(num_types):
            dsgsx.extend(dummy_sgs(dumx[i], sym[i], num_dummies))
    return TAB[0][-1]

def canonical_free(base, gens, g, num_free):
    if False:
        i = 10
        return i + 15
    '\n    Canonicalization of a tensor with respect to free indices\n    choosing the minimum with respect to lexicographical ordering\n    in the free indices.\n\n    Explanation\n    ===========\n\n    ``base``, ``gens``  BSGS for slot permutation group\n    ``g``               permutation representing the tensor\n    ``num_free``        number of free indices\n    The indices must be ordered with first the free indices\n\n    See explanation in double_coset_can_rep\n    The algorithm is a variation of the one given in [2].\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics import Permutation\n    >>> from sympy.combinatorics.tensor_can import canonical_free\n    >>> gens = [[1, 0, 2, 3, 5, 4], [2, 3, 0, 1, 4, 5],[0, 1, 3, 2, 5, 4]]\n    >>> gens = [Permutation(h) for h in gens]\n    >>> base = [0, 2]\n    >>> g = Permutation([2, 1, 0, 3, 4, 5])\n    >>> canonical_free(base, gens, g, 4)\n    [0, 3, 1, 2, 5, 4]\n\n    Consider the product of Riemann tensors\n    ``T = R^{a}_{d0}^{d1,d2}*R_{d2,d1}^{d0,b}``\n    The order of the indices is ``[a, b, d0, -d0, d1, -d1, d2, -d2]``\n    The permutation corresponding to the tensor is\n    ``g = [0, 3, 4, 6, 7, 5, 2, 1, 8, 9]``\n\n    In particular ``a`` is position ``0``, ``b`` is in position ``9``.\n    Use the slot symmetries to get `T` is a form which is the minimal\n    in lexicographic order in the free indices ``a`` and ``b``, e.g.\n    ``-R^{a}_{d0}^{d1,d2}*R^{b,d0}_{d2,d1}`` corresponding to\n    ``[0, 3, 4, 6, 1, 2, 7, 5, 9, 8]``\n\n    >>> from sympy.combinatorics.tensor_can import riemann_bsgs, tensor_gens\n    >>> base, gens = riemann_bsgs\n    >>> size, sbase, sgens = tensor_gens(base, gens, [[], []], 0)\n    >>> g = Permutation([0, 3, 4, 6, 7, 5, 2, 1, 8, 9])\n    >>> canonical_free(sbase, [Permutation(h) for h in sgens], g, 2)\n    [0, 3, 4, 6, 1, 2, 7, 5, 9, 8]\n    '
    g = g.array_form
    size = len(g)
    if not base:
        return g[:]
    transversals = get_transversals(base, gens)
    for x in sorted(g[:-2]):
        if x not in base:
            base.append(x)
    h = g
    for (i, transv) in enumerate(transversals):
        h_i = [size] * num_free
        s = None
        for sk in transv.values():
            h1 = _af_rmul(h, sk)
            hi = [h1.index(ix) for ix in range(num_free)]
            if hi < h_i:
                h_i = hi
                s = sk
        if s:
            h = _af_rmul(h, s)
    return h

def _get_map_slots(size, fixed_slots):
    if False:
        return 10
    res = list(range(size))
    pos = 0
    for i in range(size):
        if i in fixed_slots:
            continue
        res[i] = pos
        pos += 1
    return res

def _lift_sgens(size, fixed_slots, free, s):
    if False:
        print('Hello World!')
    a = []
    j = k = 0
    fd = list(zip(fixed_slots, free))
    fd = [y for (x, y) in sorted(fd)]
    num_free = len(free)
    for i in range(size):
        if i in fixed_slots:
            a.append(fd[k])
            k += 1
        else:
            a.append(s[j] + num_free)
            j += 1
    return a

def canonicalize(g, dummies, msym, *v):
    if False:
        while True:
            i = 10
    '\n    canonicalize tensor formed by tensors\n\n    Parameters\n    ==========\n\n    g : permutation representing the tensor\n\n    dummies : list representing the dummy indices\n      it can be a list of dummy indices of the same type\n      or a list of lists of dummy indices, one list for each\n      type of index;\n      the dummy indices must come after the free indices,\n      and put in order contravariant, covariant\n      [d0, -d0, d1,-d1,...]\n\n    msym :  symmetry of the metric(s)\n        it can be an integer or a list;\n        in the first case it is the symmetry of the dummy index metric;\n        in the second case it is the list of the symmetries of the\n        index metric for each type\n\n    v : list, (base_i, gens_i, n_i, sym_i) for tensors of type `i`\n\n    base_i, gens_i : BSGS for tensors of this type.\n        The BSGS should have minimal base under lexicographic ordering;\n        if not, an attempt is made do get the minimal BSGS;\n        in case of failure,\n        canonicalize_naive is used, which is much slower.\n\n    n_i :    number of tensors of type `i`.\n\n    sym_i :  symmetry under exchange of component tensors of type `i`.\n\n        Both for msym and sym_i the cases are\n            * None  no symmetry\n            * 0     commuting\n            * 1     anticommuting\n\n    Returns\n    =======\n\n    0 if the tensor is zero, else return the array form of\n    the permutation representing the canonical form of the tensor.\n\n    Algorithm\n    =========\n\n    First one uses canonical_free to get the minimum tensor under\n    lexicographic order, using only the slot symmetries.\n    If the component tensors have not minimal BSGS, it is attempted\n    to find it; if the attempt fails canonicalize_naive\n    is used instead.\n\n    Compute the residual slot symmetry keeping fixed the free indices\n    using tensor_gens(base, gens, list_free_indices, sym).\n\n    Reduce the problem eliminating the free indices.\n\n    Then use double_coset_can_rep and lift back the result reintroducing\n    the free indices.\n\n    Examples\n    ========\n\n    one type of index with commuting metric;\n\n    `A_{a b}` and `B_{a b}` antisymmetric and commuting\n\n    `T = A_{d0 d1} * B^{d0}{}_{d2} * B^{d2 d1}`\n\n    `ord = [d0,-d0,d1,-d1,d2,-d2]` order of the indices\n\n    g = [1, 3, 0, 5, 4, 2, 6, 7]\n\n    `T_c = 0`\n\n    >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, canonicalize, bsgs_direct_product\n    >>> from sympy.combinatorics import Permutation\n    >>> base2a, gens2a = get_symmetric_group_sgs(2, 1)\n    >>> t0 = (base2a, gens2a, 1, 0)\n    >>> t1 = (base2a, gens2a, 2, 0)\n    >>> g = Permutation([1, 3, 0, 5, 4, 2, 6, 7])\n    >>> canonicalize(g, range(6), 0, t0, t1)\n    0\n\n    same as above, but with `B_{a b}` anticommuting\n\n    `T_c = -A^{d0 d1} * B_{d0}{}^{d2} * B_{d1 d2}`\n\n    can = [0,2,1,4,3,5,7,6]\n\n    >>> t1 = (base2a, gens2a, 2, 1)\n    >>> canonicalize(g, range(6), 0, t0, t1)\n    [0, 2, 1, 4, 3, 5, 7, 6]\n\n    two types of indices `[a,b,c,d,e,f]` and `[m,n]`, in this order,\n    both with commuting metric\n\n    `f^{a b c}` antisymmetric, commuting\n\n    `A_{m a}` no symmetry, commuting\n\n    `T = f^c{}_{d a} * f^f{}_{e b} * A_m{}^d * A^{m b} * A_n{}^a * A^{n e}`\n\n    ord = [c,f,a,-a,b,-b,d,-d,e,-e,m,-m,n,-n]\n\n    g = [0,7,3, 1,9,5, 11,6, 10,4, 13,2, 12,8, 14,15]\n\n    The canonical tensor is\n    `T_c = -f^{c a b} * f^{f d e} * A^m{}_a * A_{m d} * A^n{}_b * A_{n e}`\n\n    can = [0,2,4, 1,6,8, 10,3, 11,7, 12,5, 13,9, 15,14]\n\n    >>> base_f, gens_f = get_symmetric_group_sgs(3, 1)\n    >>> base1, gens1 = get_symmetric_group_sgs(1)\n    >>> base_A, gens_A = bsgs_direct_product(base1, gens1, base1, gens1)\n    >>> t0 = (base_f, gens_f, 2, 0)\n    >>> t1 = (base_A, gens_A, 4, 0)\n    >>> dummies = [range(2, 10), range(10, 14)]\n    >>> g = Permutation([0, 7, 3, 1, 9, 5, 11, 6, 10, 4, 13, 2, 12, 8, 14, 15])\n    >>> canonicalize(g, dummies, [0, 0], t0, t1)\n    [0, 2, 4, 1, 6, 8, 10, 3, 11, 7, 12, 5, 13, 9, 15, 14]\n    '
    from sympy.combinatorics.testutil import canonicalize_naive
    if not isinstance(msym, list):
        if msym not in (0, 1, None):
            raise ValueError('msym must be 0, 1 or None')
        num_types = 1
    else:
        num_types = len(msym)
        if not all((msymx in (0, 1, None) for msymx in msym)):
            raise ValueError('msym entries must be 0, 1 or None')
        if len(dummies) != num_types:
            raise ValueError('dummies and msym must have the same number of elements')
    size = g.size
    num_tensors = 0
    v1 = []
    for (base_i, gens_i, n_i, sym_i) in v:
        if not _is_minimal_bsgs(base_i, gens_i):
            mbsgs = get_minimal_bsgs(base_i, gens_i)
            if not mbsgs:
                can = canonicalize_naive(g, dummies, msym, *v)
                return can
            (base_i, gens_i) = mbsgs
        v1.append((base_i, gens_i, [[]] * n_i, sym_i))
        num_tensors += n_i
    if num_types == 1 and (not isinstance(msym, list)):
        dummies = [dummies]
        msym = [msym]
    flat_dummies = []
    for dumx in dummies:
        flat_dummies.extend(dumx)
    if flat_dummies and flat_dummies != list(range(flat_dummies[0], flat_dummies[-1] + 1)):
        raise ValueError('dummies is not valid')
    (size1, sbase, sgens) = gens_products(*v1)
    if size != size1:
        raise ValueError('g has size %d, generators have size %d' % (size, size1))
    free = [i for i in range(size - 2) if i not in flat_dummies]
    num_free = len(free)
    g1 = canonical_free(sbase, sgens, g, num_free)
    if not flat_dummies:
        return g1
    sign = 0 if g1[-1] == size - 1 else 1
    start = 0
    for (i, (base_i, gens_i, n_i, sym_i)) in enumerate(v):
        free_i = []
        len_tens = gens_i[0].size - 2
        for j in range(n_i):
            h = g1[start:start + len_tens]
            fr = []
            for k in free:
                if k in h:
                    fr.append(h.index(k))
            free_i.append(fr)
            start += len_tens
        v1[i] = (base_i, gens_i, free_i, sym_i)
    (size, sbase, sgens) = gens_products(*v1)
    pos_free = [g1.index(x) for x in range(num_free)]
    size_red = size - num_free
    g1_red = [x - num_free for x in g1 if x in flat_dummies]
    if sign:
        g1_red.extend([size_red - 1, size_red - 2])
    else:
        g1_red.extend([size_red - 2, size_red - 1])
    map_slots = _get_map_slots(size, pos_free)
    sbase_red = [map_slots[i] for i in sbase if i not in pos_free]
    sgens_red = [_af_new([map_slots[i] for i in y._array_form if i not in pos_free]) for y in sgens]
    dummies_red = [[x - num_free for x in y] for y in dummies]
    transv_red = get_transversals(sbase_red, sgens_red)
    g1_red = _af_new(g1_red)
    g2 = double_coset_can_rep(dummies_red, msym, sbase_red, sgens_red, transv_red, g1_red)
    if g2 == 0:
        return 0
    g3 = _lift_sgens(size, pos_free, free, g2)
    return g3

def perm_af_direct_product(gens1, gens2, signed=True):
    if False:
        return 10
    '\n    Direct products of the generators gens1 and gens2.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.tensor_can import perm_af_direct_product\n    >>> gens1 = [[1, 0, 2, 3], [0, 1, 3, 2]]\n    >>> gens2 = [[1, 0]]\n    >>> perm_af_direct_product(gens1, gens2, False)\n    [[1, 0, 2, 3, 4, 5], [0, 1, 3, 2, 4, 5], [0, 1, 2, 3, 5, 4]]\n    >>> gens1 = [[1, 0, 2, 3, 5, 4], [0, 1, 3, 2, 4, 5]]\n    >>> gens2 = [[1, 0, 2, 3]]\n    >>> perm_af_direct_product(gens1, gens2, True)\n    [[1, 0, 2, 3, 4, 5, 7, 6], [0, 1, 3, 2, 4, 5, 6, 7], [0, 1, 2, 3, 5, 4, 6, 7]]\n    '
    gens1 = [list(x) for x in gens1]
    gens2 = [list(x) for x in gens2]
    s = 2 if signed else 0
    n1 = len(gens1[0]) - s
    n2 = len(gens2[0]) - s
    start = list(range(n1))
    end = list(range(n1, n1 + n2))
    if signed:
        gens1 = [gen[:-2] + end + [gen[-2] + n2, gen[-1] + n2] for gen in gens1]
        gens2 = [start + [x + n1 for x in gen] for gen in gens2]
    else:
        gens1 = [gen + end for gen in gens1]
        gens2 = [start + [x + n1 for x in gen] for gen in gens2]
    res = gens1 + gens2
    return res

def bsgs_direct_product(base1, gens1, base2, gens2, signed=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Direct product of two BSGS.\n\n    Parameters\n    ==========\n\n    base1 : base of the first BSGS.\n\n    gens1 : strong generating sequence of the first BSGS.\n\n    base2, gens2 : similarly for the second BSGS.\n\n    signed : flag for signed permutations.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.tensor_can import (get_symmetric_group_sgs, bsgs_direct_product)\n    >>> base1, gens1 = get_symmetric_group_sgs(1)\n    >>> base2, gens2 = get_symmetric_group_sgs(2)\n    >>> bsgs_direct_product(base1, gens1, base2, gens2)\n    ([1], [(4)(1 2)])\n    '
    s = 2 if signed else 0
    n1 = gens1[0].size - s
    base = list(base1)
    base += [x + n1 for x in base2]
    gens1 = [h._array_form for h in gens1]
    gens2 = [h._array_form for h in gens2]
    gens = perm_af_direct_product(gens1, gens2, signed)
    size = len(gens[0])
    id_af = list(range(size))
    gens = [h for h in gens if h != id_af]
    if not gens:
        gens = [id_af]
    return (base, [_af_new(h) for h in gens])

def get_symmetric_group_sgs(n, antisym=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return base, gens of the minimal BSGS for (anti)symmetric tensor\n\n    Parameters\n    ==========\n\n    n : rank of the tensor\n    antisym : bool\n        ``antisym = False`` symmetric tensor\n        ``antisym = True``  antisymmetric tensor\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs\n    >>> get_symmetric_group_sgs(3)\n    ([0, 1], [(4)(0 1), (4)(1 2)])\n    '
    if n == 1:
        return ([], [_af_new(list(range(3)))])
    gens = [Permutation(n - 1)(i, i + 1)._array_form for i in range(n - 1)]
    if antisym == 0:
        gens = [x + [n, n + 1] for x in gens]
    else:
        gens = [x + [n + 1, n] for x in gens]
    base = list(range(n - 1))
    return (base, [_af_new(h) for h in gens])
riemann_bsgs = ([0, 2], [Permutation(0, 1)(4, 5), Permutation(2, 3)(4, 5), Permutation(5)(0, 2)(1, 3)])

def get_transversals(base, gens):
    if False:
        while True:
            i = 10
    '\n    Return transversals for the group with BSGS base, gens\n    '
    if not base:
        return []
    stabs = _distribute_gens_by_base(base, gens)
    (orbits, transversals) = _orbits_transversals_from_bsgs(base, stabs)
    transversals = [{x: h._array_form for (x, h) in y.items()} for y in transversals]
    return transversals

def _is_minimal_bsgs(base, gens):
    if False:
        print('Hello World!')
    '\n    Check if the BSGS has minimal base under lexigographic order.\n\n    base, gens BSGS\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics import Permutation\n    >>> from sympy.combinatorics.tensor_can import riemann_bsgs, _is_minimal_bsgs\n    >>> _is_minimal_bsgs(*riemann_bsgs)\n    True\n    >>> riemann_bsgs1 = ([2, 0], ([Permutation(5)(0, 1)(4, 5), Permutation(5)(0, 2)(1, 3)]))\n    >>> _is_minimal_bsgs(*riemann_bsgs1)\n    False\n    '
    base1 = []
    sgs1 = gens[:]
    size = gens[0].size
    for i in range(size):
        if not all((h._array_form[i] == i for h in sgs1)):
            base1.append(i)
            sgs1 = [h for h in sgs1 if h._array_form[i] == i]
    return base1 == base

def get_minimal_bsgs(base, gens):
    if False:
        return 10
    '\n    Compute a minimal GSGS\n\n    base, gens BSGS\n\n    If base, gens is a minimal BSGS return it; else return a minimal BSGS\n    if it fails in finding one, it returns None\n\n    TODO: use baseswap in the case in which if it fails in finding a\n    minimal BSGS\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics import Permutation\n    >>> from sympy.combinatorics.tensor_can import get_minimal_bsgs\n    >>> riemann_bsgs1 = ([2, 0], ([Permutation(5)(0, 1)(4, 5), Permutation(5)(0, 2)(1, 3)]))\n    >>> get_minimal_bsgs(*riemann_bsgs1)\n    ([0, 2], [(0 1)(4 5), (5)(0 2)(1 3), (2 3)(4 5)])\n    '
    G = PermutationGroup(gens)
    (base, gens) = G.schreier_sims_incremental()
    if not _is_minimal_bsgs(base, gens):
        return None
    return (base, gens)

def tensor_gens(base, gens, list_free_indices, sym=0):
    if False:
        return 10
    '\n    Returns size, res_base, res_gens BSGS for n tensors of the\n    same type.\n\n    Explanation\n    ===========\n\n    base, gens BSGS for tensors of this type\n    list_free_indices  list of the slots occupied by fixed indices\n                       for each of the tensors\n\n    sym symmetry under commutation of two tensors\n    sym   None  no symmetry\n    sym   0     commuting\n    sym   1     anticommuting\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.tensor_can import tensor_gens, get_symmetric_group_sgs\n\n    two symmetric tensors with 3 indices without free indices\n\n    >>> base, gens = get_symmetric_group_sgs(3)\n    >>> tensor_gens(base, gens, [[], []])\n    (8, [0, 1, 3, 4], [(7)(0 1), (7)(1 2), (7)(3 4), (7)(4 5), (7)(0 3)(1 4)(2 5)])\n\n    two symmetric tensors with 3 indices with free indices in slot 1 and 0\n\n    >>> tensor_gens(base, gens, [[1], [0]])\n    (8, [0, 4], [(7)(0 2), (7)(4 5)])\n\n    four symmetric tensors with 3 indices, two of which with free indices\n\n    '

    def _get_bsgs(G, base, gens, free_indices):
        if False:
            return 10
        '\n        return the BSGS for G.pointwise_stabilizer(free_indices)\n        '
        if not free_indices:
            return (base[:], gens[:])
        else:
            H = G.pointwise_stabilizer(free_indices)
            (base, sgs) = H.schreier_sims_incremental()
            return (base, sgs)
    if not base and list_free_indices.count([]) < 2:
        n = len(list_free_indices)
        size = gens[0].size
        size = n * (size - 2) + 2
        return (size, [], [_af_new(list(range(size)))])
    if any(list_free_indices):
        G = PermutationGroup(gens)
    else:
        G = None
    no_free = []
    size = gens[0].size
    id_af = list(range(size))
    num_indices = size - 2
    if not list_free_indices[0]:
        no_free.append(list(range(num_indices)))
    (res_base, res_gens) = _get_bsgs(G, base, gens, list_free_indices[0])
    for i in range(1, len(list_free_indices)):
        (base1, gens1) = _get_bsgs(G, base, gens, list_free_indices[i])
        (res_base, res_gens) = bsgs_direct_product(res_base, res_gens, base1, gens1, 1)
        if not list_free_indices[i]:
            no_free.append(list(range(size - 2, size - 2 + num_indices)))
        size += num_indices
    nr = size - 2
    res_gens = [h for h in res_gens if h._array_form != id_af]
    if sym is None or not no_free:
        if not res_gens:
            res_gens = [_af_new(id_af)]
        return (size, res_base, res_gens)
    base_comm = []
    for i in range(len(no_free) - 1):
        ind1 = no_free[i]
        ind2 = no_free[i + 1]
        a = list(range(ind1[0]))
        a.extend(ind2)
        a.extend(ind1)
        base_comm.append(ind1[0])
        a.extend(list(range(ind2[-1] + 1, nr)))
        if sym == 0:
            a.extend([nr, nr + 1])
        else:
            a.extend([nr + 1, nr])
        res_gens.append(_af_new(a))
    res_base = list(res_base)
    for i in base_comm:
        if i not in res_base:
            res_base.append(i)
    res_base.sort()
    if not res_gens:
        res_gens = [_af_new(id_af)]
    return (size, res_base, res_gens)

def gens_products(*v):
    if False:
        i = 10
        return i + 15
    '\n    Returns size, res_base, res_gens BSGS for n tensors of different types.\n\n    Explanation\n    ===========\n\n    v is a sequence of (base_i, gens_i, free_i, sym_i)\n    where\n    base_i, gens_i  BSGS of tensor of type `i`\n    free_i          list of the fixed slots for each of the tensors\n                    of type `i`; if there are `n_i` tensors of type `i`\n                    and none of them have fixed slots, `free = [[]]*n_i`\n    sym   0 (1) if the tensors of type `i` (anti)commute among themselves\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, gens_products\n    >>> base, gens = get_symmetric_group_sgs(2)\n    >>> gens_products((base, gens, [[], []], 0))\n    (6, [0, 2], [(5)(0 1), (5)(2 3), (5)(0 2)(1 3)])\n    >>> gens_products((base, gens, [[1], []], 0))\n    (6, [2], [(5)(2 3)])\n    '
    (res_size, res_base, res_gens) = tensor_gens(*v[0])
    for i in range(1, len(v)):
        (size, base, gens) = tensor_gens(*v[i])
        (res_base, res_gens) = bsgs_direct_product(res_base, res_gens, base, gens, 1)
    res_size = res_gens[0].size
    id_af = list(range(res_size))
    res_gens = [h for h in res_gens if h != id_af]
    if not res_gens:
        res_gens = [id_af]
    return (res_size, res_base, res_gens)