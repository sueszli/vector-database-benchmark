"""
    Module to handle gamma matrices expressed as tensor objects.

    Examples
    ========

    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G, LorentzIndex
    >>> from sympy.tensor.tensor import tensor_indices
    >>> i = tensor_indices('i', LorentzIndex)
    >>> G(i)
    GammaMatrix(i)

    Note that there is already an instance of GammaMatrixHead in four dimensions:
    GammaMatrix, which is simply declare as

    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix
    >>> from sympy.tensor.tensor import tensor_indices
    >>> i = tensor_indices('i', LorentzIndex)
    >>> GammaMatrix(i)
    GammaMatrix(i)

    To access the metric tensor

    >>> LorentzIndex.metric
    metric(LorentzIndex,LorentzIndex)

"""
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.matrices.dense import eye
from sympy.matrices.expressions.trace import trace
from sympy.tensor.tensor import TensorIndexType, TensorIndex, TensMul, TensAdd, tensor_mul, Tensor, TensorHead, TensorSymmetry
LorentzIndex = TensorIndexType('LorentzIndex', dim=4, dummy_name='L')
GammaMatrix = TensorHead('GammaMatrix', [LorentzIndex], TensorSymmetry.no_symmetry(1), comm=None)

def extract_type_tens(expression, component):
    if False:
        for i in range(10):
            print('nop')
    '\n    Extract from a ``TensExpr`` all tensors with `component`.\n\n    Returns two tensor expressions:\n\n    * the first contains all ``Tensor`` of having `component`.\n    * the second contains all remaining.\n\n\n    '
    if isinstance(expression, Tensor):
        sp = [expression]
    elif isinstance(expression, TensMul):
        sp = expression.args
    else:
        raise ValueError('wrong type')
    new_expr = S.One
    residual_expr = S.One
    for i in sp:
        if isinstance(i, Tensor) and i.component == component:
            new_expr *= i
        else:
            residual_expr *= i
    return (new_expr, residual_expr)

def simplify_gamma_expression(expression):
    if False:
        i = 10
        return i + 15
    (extracted_expr, residual_expr) = extract_type_tens(expression, GammaMatrix)
    res_expr = _simplify_single_line(extracted_expr)
    return res_expr * residual_expr

def simplify_gpgp(ex, sort=True):
    if False:
        i = 10
        return i + 15
    "\n    simplify products ``G(i)*p(-i)*G(j)*p(-j) -> p(i)*p(-i)``\n\n    Examples\n    ========\n\n    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G,         LorentzIndex, simplify_gpgp\n    >>> from sympy.tensor.tensor import tensor_indices, tensor_heads\n    >>> p, q = tensor_heads('p, q', [LorentzIndex])\n    >>> i0,i1,i2,i3,i4,i5 = tensor_indices('i0:6', LorentzIndex)\n    >>> ps = p(i0)*G(-i0)\n    >>> qs = q(i0)*G(-i0)\n    >>> simplify_gpgp(ps*qs*qs)\n    GammaMatrix(-L_0)*p(L_0)*q(L_1)*q(-L_1)\n    "

    def _simplify_gpgp(ex):
        if False:
            while True:
                i = 10
        components = ex.components
        a = []
        comp_map = []
        for (i, comp) in enumerate(components):
            comp_map.extend([i] * comp.rank)
        dum = [(i[0], i[1], comp_map[i[0]], comp_map[i[1]]) for i in ex.dum]
        for i in range(len(components)):
            if components[i] != GammaMatrix:
                continue
            for dx in dum:
                if dx[2] == i:
                    p_pos1 = dx[3]
                elif dx[3] == i:
                    p_pos1 = dx[2]
                else:
                    continue
                comp1 = components[p_pos1]
                if comp1.comm == 0 and comp1.rank == 1:
                    a.append((i, p_pos1))
        if not a:
            return ex
        elim = set()
        tv = []
        hit = True
        coeff = S.One
        ta = None
        while hit:
            hit = False
            for (i, ai) in enumerate(a[:-1]):
                if ai[0] in elim:
                    continue
                if ai[0] != a[i + 1][0] - 1:
                    continue
                if components[ai[1]] != components[a[i + 1][1]]:
                    continue
                elim.add(ai[0])
                elim.add(ai[1])
                elim.add(a[i + 1][0])
                elim.add(a[i + 1][1])
                if not ta:
                    ta = ex.split()
                    mu = TensorIndex('mu', LorentzIndex)
                hit = True
                if i == 0:
                    coeff = ex.coeff
                tx = components[ai[1]](mu) * components[ai[1]](-mu)
                if len(a) == 2:
                    tx *= 4
                tv.append(tx)
                break
        if tv:
            a = [x for (j, x) in enumerate(ta) if j not in elim]
            a.extend(tv)
            t = tensor_mul(*a) * coeff
            return t
        else:
            return ex
    if sort:
        ex = ex.sorted_components()
    while 1:
        t = _simplify_gpgp(ex)
        if t != ex:
            ex = t
        else:
            return t

def gamma_trace(t):
    if False:
        while True:
            i = 10
    "\n    trace of a single line of gamma matrices\n\n    Examples\n    ========\n\n    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G,         gamma_trace, LorentzIndex\n    >>> from sympy.tensor.tensor import tensor_indices, tensor_heads\n    >>> p, q = tensor_heads('p, q', [LorentzIndex])\n    >>> i0,i1,i2,i3,i4,i5 = tensor_indices('i0:6', LorentzIndex)\n    >>> ps = p(i0)*G(-i0)\n    >>> qs = q(i0)*G(-i0)\n    >>> gamma_trace(G(i0)*G(i1))\n    4*metric(i0, i1)\n    >>> gamma_trace(ps*ps) - 4*p(i0)*p(-i0)\n    0\n    >>> gamma_trace(ps*qs + ps*ps) - 4*p(i0)*p(-i0) - 4*p(i0)*q(-i0)\n    0\n\n    "
    if isinstance(t, TensAdd):
        res = TensAdd(*[gamma_trace(x) for x in t.args])
        return res
    t = _simplify_single_line(t)
    res = _trace_single_line(t)
    return res

def _simplify_single_line(expression):
    if False:
        for i in range(10):
            print('nop')
    "\n    Simplify single-line product of gamma matrices.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G,         LorentzIndex, _simplify_single_line\n    >>> from sympy.tensor.tensor import tensor_indices, TensorHead\n    >>> p = TensorHead('p', [LorentzIndex])\n    >>> i0,i1 = tensor_indices('i0:2', LorentzIndex)\n    >>> _simplify_single_line(G(i0)*G(i1)*p(-i1)*G(-i0)) + 2*G(i0)*p(-i0)\n    0\n\n    "
    (t1, t2) = extract_type_tens(expression, GammaMatrix)
    if t1 != 1:
        t1 = kahane_simplify(t1)
    res = t1 * t2
    return res

def _trace_single_line(t):
    if False:
        for i in range(10):
            print('nop')
    "\n    Evaluate the trace of a single gamma matrix line inside a ``TensExpr``.\n\n    Notes\n    =====\n\n    If there are ``DiracSpinorIndex.auto_left`` and ``DiracSpinorIndex.auto_right``\n    indices trace over them; otherwise traces are not implied (explain)\n\n\n    Examples\n    ========\n\n    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G,         LorentzIndex, _trace_single_line\n    >>> from sympy.tensor.tensor import tensor_indices, TensorHead\n    >>> p = TensorHead('p', [LorentzIndex])\n    >>> i0,i1,i2,i3,i4,i5 = tensor_indices('i0:6', LorentzIndex)\n    >>> _trace_single_line(G(i0)*G(i1))\n    4*metric(i0, i1)\n    >>> _trace_single_line(G(i0)*p(-i0)*G(i1)*p(-i1)) - 4*p(i0)*p(-i0)\n    0\n\n    "

    def _trace_single_line1(t):
        if False:
            return 10
        t = t.sorted_components()
        components = t.components
        ncomps = len(components)
        g = LorentzIndex.metric
        hit = 0
        for i in range(ncomps):
            if components[i] == GammaMatrix:
                hit = 1
                break
        for j in range(i + hit, ncomps):
            if components[j] != GammaMatrix:
                break
        else:
            j = ncomps
        numG = j - i
        if numG == 0:
            tcoeff = t.coeff
            return t.nocoeff if tcoeff else t
        if numG % 2 == 1:
            return TensMul.from_data(S.Zero, [], [], [])
        elif numG > 4:
            a = t.split()
            ind1 = a[i].get_indices()[0]
            ind2 = a[i + 1].get_indices()[0]
            aa = a[:i] + a[i + 2:]
            t1 = tensor_mul(*aa) * g(ind1, ind2)
            t1 = t1.contract_metric(g)
            args = [t1]
            sign = 1
            for k in range(i + 2, j):
                sign = -sign
                ind2 = a[k].get_indices()[0]
                aa = a[:i] + a[i + 1:k] + a[k + 1:]
                t2 = sign * tensor_mul(*aa) * g(ind1, ind2)
                t2 = t2.contract_metric(g)
                t2 = simplify_gpgp(t2, False)
                args.append(t2)
            t3 = TensAdd(*args)
            t3 = _trace_single_line(t3)
            return t3
        else:
            a = t.split()
            t1 = _gamma_trace1(*a[i:j])
            a2 = a[:i] + a[j:]
            t2 = tensor_mul(*a2)
            t3 = t1 * t2
            if not t3:
                return t3
            t3 = t3.contract_metric(g)
            return t3
    t = t.expand()
    if isinstance(t, TensAdd):
        a = [_trace_single_line1(x) * x.coeff for x in t.args]
        return TensAdd(*a)
    elif isinstance(t, (Tensor, TensMul)):
        r = t.coeff * _trace_single_line1(t)
        return r
    else:
        return trace(t)

def _gamma_trace1(*a):
    if False:
        for i in range(10):
            print('nop')
    gctr = 4
    g = LorentzIndex.metric
    if not a:
        return gctr
    n = len(a)
    if n % 2 == 1:
        return S.Zero
    if n == 2:
        ind0 = a[0].get_indices()[0]
        ind1 = a[1].get_indices()[0]
        return gctr * g(ind0, ind1)
    if n == 4:
        ind0 = a[0].get_indices()[0]
        ind1 = a[1].get_indices()[0]
        ind2 = a[2].get_indices()[0]
        ind3 = a[3].get_indices()[0]
        return gctr * (g(ind0, ind1) * g(ind2, ind3) - g(ind0, ind2) * g(ind1, ind3) + g(ind0, ind3) * g(ind1, ind2))

def kahane_simplify(expression):
    if False:
        print('Hello World!')
    "\n    This function cancels contracted elements in a product of four\n    dimensional gamma matrices, resulting in an expression equal to the given\n    one, without the contracted gamma matrices.\n\n    Parameters\n    ==========\n\n    `expression`    the tensor expression containing the gamma matrices to simplify.\n\n    Notes\n    =====\n\n    If spinor indices are given, the matrices must be given in\n    the order given in the product.\n\n    Algorithm\n    =========\n\n    The idea behind the algorithm is to use some well-known identities,\n    i.e., for contractions enclosing an even number of `\\gamma` matrices\n\n    `\\gamma^\\mu \\gamma_{a_1} \\cdots \\gamma_{a_{2N}} \\gamma_\\mu = 2 (\\gamma_{a_{2N}} \\gamma_{a_1} \\cdots \\gamma_{a_{2N-1}} + \\gamma_{a_{2N-1}} \\cdots \\gamma_{a_1} \\gamma_{a_{2N}} )`\n\n    for an odd number of `\\gamma` matrices\n\n    `\\gamma^\\mu \\gamma_{a_1} \\cdots \\gamma_{a_{2N+1}} \\gamma_\\mu = -2 \\gamma_{a_{2N+1}} \\gamma_{a_{2N}} \\cdots \\gamma_{a_{1}}`\n\n    Instead of repeatedly applying these identities to cancel out all contracted indices,\n    it is possible to recognize the links that would result from such an operation,\n    the problem is thus reduced to a simple rearrangement of free gamma matrices.\n\n    Examples\n    ========\n\n    When using, always remember that the original expression coefficient\n    has to be handled separately\n\n    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G, LorentzIndex\n    >>> from sympy.physics.hep.gamma_matrices import kahane_simplify\n    >>> from sympy.tensor.tensor import tensor_indices\n    >>> i0, i1, i2 = tensor_indices('i0:3', LorentzIndex)\n    >>> ta = G(i0)*G(-i0)\n    >>> kahane_simplify(ta)\n    Matrix([\n    [4, 0, 0, 0],\n    [0, 4, 0, 0],\n    [0, 0, 4, 0],\n    [0, 0, 0, 4]])\n    >>> tb = G(i0)*G(i1)*G(-i0)\n    >>> kahane_simplify(tb)\n    -2*GammaMatrix(i1)\n    >>> t = G(i0)*G(-i0)\n    >>> kahane_simplify(t)\n    Matrix([\n    [4, 0, 0, 0],\n    [0, 4, 0, 0],\n    [0, 0, 4, 0],\n    [0, 0, 0, 4]])\n    >>> t = G(i0)*G(-i0)\n    >>> kahane_simplify(t)\n    Matrix([\n    [4, 0, 0, 0],\n    [0, 4, 0, 0],\n    [0, 0, 4, 0],\n    [0, 0, 0, 4]])\n\n    If there are no contractions, the same expression is returned\n\n    >>> tc = G(i0)*G(i1)\n    >>> kahane_simplify(tc)\n    GammaMatrix(i0)*GammaMatrix(i1)\n\n    References\n    ==========\n\n    [1] Algorithm for Reducing Contracted Products of gamma Matrices,\n    Joseph Kahane, Journal of Mathematical Physics, Vol. 9, No. 10, October 1968.\n    "
    if isinstance(expression, Mul):
        return expression
    if isinstance(expression, TensAdd):
        return TensAdd(*[kahane_simplify(arg) for arg in expression.args])
    if isinstance(expression, Tensor):
        return expression
    assert isinstance(expression, TensMul)
    gammas = expression.args
    for gamma in gammas:
        assert gamma.component == GammaMatrix
    free = expression.free
    dum = []
    for dum_pair in expression.dum:
        if expression.index_types[dum_pair[0]] == LorentzIndex:
            dum.append((dum_pair[0], dum_pair[1]))
    dum = sorted(dum)
    if len(dum) == 0:
        return expression
    first_dum_pos = min(map(min, dum))
    total_number = len(free) + len(dum) * 2
    number_of_contractions = len(dum)
    free_pos = [None] * total_number
    for i in free:
        free_pos[i[1]] = i[0]
    index_is_free = [False] * total_number
    for (i, indx) in enumerate(free):
        index_is_free[indx[1]] = True
    links = {i: [] for i in range(first_dum_pos, total_number)}
    cum_sign = -1
    cum_sign_list = [None] * total_number
    block_free_count = 0
    resulting_coeff = S.One
    resulting_indices = [[]]
    connected_components = 1
    for (i, is_free) in enumerate(index_is_free):
        if i < first_dum_pos:
            continue
        if is_free:
            block_free_count += 1
            if block_free_count > 1:
                links[i - 1].append(i)
                links[i].append(i - 1)
        else:
            cum_sign *= 1 if block_free_count % 2 else -1
            if block_free_count == 0 and i != first_dum_pos:
                if cum_sign == -1:
                    links[-1 - i] = [-1 - i + 1]
                    links[-1 - i + 1] = [-1 - i]
            if i - cum_sign in links:
                if i != first_dum_pos:
                    links[i].append(i - cum_sign)
                if block_free_count != 0:
                    if i - cum_sign < len(index_is_free):
                        if index_is_free[i - cum_sign]:
                            links[i - cum_sign].append(i)
            block_free_count = 0
        cum_sign_list[i] = cum_sign
    for i in dum:
        pos1 = i[0]
        pos2 = i[1]
        links[pos1].append(pos2)
        links[pos2].append(pos1)
        linkpos1 = pos1 + cum_sign_list[pos1]
        linkpos2 = pos2 + cum_sign_list[pos2]
        if linkpos1 >= total_number:
            continue
        if linkpos2 >= total_number:
            continue
        if linkpos1 < first_dum_pos:
            continue
        if linkpos2 < first_dum_pos:
            continue
        if -1 - linkpos1 in links:
            linkpos1 = -1 - linkpos1
        if -1 - linkpos2 in links:
            linkpos2 = -1 - linkpos2
        if linkpos1 >= 0 and (not index_is_free[linkpos1]):
            linkpos1 = pos1
        if linkpos2 >= 0 and (not index_is_free[linkpos2]):
            linkpos2 = pos2
        if linkpos2 not in links[linkpos1]:
            links[linkpos1].append(linkpos2)
        if linkpos1 not in links[linkpos2]:
            links[linkpos2].append(linkpos1)
    pointer = first_dum_pos
    previous_pointer = 0
    while True:
        if pointer in links:
            next_ones = links.pop(pointer)
        else:
            break
        if previous_pointer in next_ones:
            next_ones.remove(previous_pointer)
        previous_pointer = pointer
        if next_ones:
            pointer = next_ones[0]
        else:
            break
        if pointer == previous_pointer:
            break
        if pointer >= 0 and free_pos[pointer] is not None:
            for ri in resulting_indices:
                ri.append(free_pos[pointer])
    while links:
        connected_components += 1
        pointer = min(links.keys())
        previous_pointer = pointer
        prepend_indices = []
        while True:
            if pointer in links:
                next_ones = links.pop(pointer)
            else:
                break
            if previous_pointer in next_ones:
                if len(next_ones) > 1:
                    next_ones.remove(previous_pointer)
            previous_pointer = pointer
            if next_ones:
                pointer = next_ones[0]
            if pointer >= first_dum_pos and free_pos[pointer] is not None:
                prepend_indices.insert(0, free_pos[pointer])
        if len(prepend_indices) == 0:
            resulting_coeff *= 2
        else:
            expr1 = prepend_indices
            expr2 = list(reversed(prepend_indices))
            resulting_indices = [expri + ri for ri in resulting_indices for expri in (expr1, expr2)]
    resulting_coeff *= -1 if (number_of_contractions - connected_components + 1) % 2 else 1
    resulting_coeff *= 2 ** number_of_contractions
    resulting_indices = [free_pos[0:first_dum_pos] + ri for ri in resulting_indices]
    resulting_expr = S.Zero
    for i in resulting_indices:
        temp_expr = S.One
        for j in i:
            temp_expr *= GammaMatrix(j)
        resulting_expr += temp_expr
    t = resulting_coeff * resulting_expr
    t1 = None
    if isinstance(t, TensAdd):
        t1 = t.args[0]
    elif isinstance(t, TensMul):
        t1 = t
    if t1:
        pass
    else:
        t = eye(4) * t
    return t