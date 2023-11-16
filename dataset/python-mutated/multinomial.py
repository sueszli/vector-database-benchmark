from sympy.utilities.misc import as_int

def binomial_coefficients(n):
    if False:
        for i in range(10):
            print('nop')
    'Return a dictionary containing pairs :math:`{(k1,k2) : C_kn}` where\n    :math:`C_kn` are binomial coefficients and :math:`n=k1+k2`.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory import binomial_coefficients\n    >>> binomial_coefficients(9)\n    {(0, 9): 1, (1, 8): 9, (2, 7): 36, (3, 6): 84,\n     (4, 5): 126, (5, 4): 126, (6, 3): 84, (7, 2): 36, (8, 1): 9, (9, 0): 1}\n\n    See Also\n    ========\n\n    binomial_coefficients_list, multinomial_coefficients\n    '
    n = as_int(n)
    d = {(0, n): 1, (n, 0): 1}
    a = 1
    for k in range(1, n // 2 + 1):
        a = a * (n - k + 1) // k
        d[k, n - k] = d[n - k, k] = a
    return d

def binomial_coefficients_list(n):
    if False:
        while True:
            i = 10
    " Return a list of binomial coefficients as rows of the Pascal's\n    triangle.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory import binomial_coefficients_list\n    >>> binomial_coefficients_list(9)\n    [1, 9, 36, 84, 126, 126, 84, 36, 9, 1]\n\n    See Also\n    ========\n\n    binomial_coefficients, multinomial_coefficients\n    "
    n = as_int(n)
    d = [1] * (n + 1)
    a = 1
    for k in range(1, n // 2 + 1):
        a = a * (n - k + 1) // k
        d[k] = d[n - k] = a
    return d

def multinomial_coefficients(m, n):
    if False:
        while True:
            i = 10
    'Return a dictionary containing pairs ``{(k1,k2,..,km) : C_kn}``\n    where ``C_kn`` are multinomial coefficients such that\n    ``n=k1+k2+..+km``.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory import multinomial_coefficients\n    >>> multinomial_coefficients(2, 5) # indirect doctest\n    {(0, 5): 1, (1, 4): 5, (2, 3): 10, (3, 2): 10, (4, 1): 5, (5, 0): 1}\n\n    Notes\n    =====\n\n    The algorithm is based on the following result:\n\n    .. math::\n        \\binom{n}{k_1, \\ldots, k_m} =\n        \\frac{k_1 + 1}{n - k_1} \\sum_{i=2}^m \\binom{n}{k_1 + 1, \\ldots, k_i - 1, \\ldots}\n\n    Code contributed to Sage by Yann Laigle-Chapuy, copied with permission\n    of the author.\n\n    See Also\n    ========\n\n    binomial_coefficients_list, binomial_coefficients\n    '
    m = as_int(m)
    n = as_int(n)
    if not m:
        if n:
            return {}
        return {(): 1}
    if m == 2:
        return binomial_coefficients(n)
    if m >= 2 * n and n > 1:
        return dict(multinomial_coefficients_iterator(m, n))
    t = [n] + [0] * (m - 1)
    r = {tuple(t): 1}
    if n:
        j = 0
    else:
        j = m
    while j < m - 1:
        tj = t[j]
        if j:
            t[j] = 0
            t[0] = tj
        if tj > 1:
            t[j + 1] += 1
            j = 0
            start = 1
            v = 0
        else:
            j += 1
            start = j + 1
            v = r[tuple(t)]
            t[j] += 1
        for k in range(start, m):
            if t[k]:
                t[k] -= 1
                v += r[tuple(t)]
                t[k] += 1
        t[0] -= 1
        r[tuple(t)] = v * tj // (n - t[0])
    return r

def multinomial_coefficients_iterator(m, n, _tuple=tuple):
    if False:
        while True:
            i = 10
    'multinomial coefficient iterator\n\n    This routine has been optimized for `m` large with respect to `n` by taking\n    advantage of the fact that when the monomial tuples `t` are stripped of\n    zeros, their coefficient is the same as that of the monomial tuples from\n    ``multinomial_coefficients(n, n)``. Therefore, the latter coefficients are\n    precomputed to save memory and time.\n\n    >>> from sympy.ntheory.multinomial import multinomial_coefficients\n    >>> m53, m33 = multinomial_coefficients(5,3), multinomial_coefficients(3,3)\n    >>> m53[(0,0,0,1,2)] == m53[(0,0,1,0,2)] == m53[(1,0,2,0,0)] == m33[(0,1,2)]\n    True\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.multinomial import multinomial_coefficients_iterator\n    >>> it = multinomial_coefficients_iterator(20,3)\n    >>> next(it)\n    ((3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 1)\n    '
    m = as_int(m)
    n = as_int(n)
    if m < 2 * n or n == 1:
        mc = multinomial_coefficients(m, n)
        yield from mc.items()
    else:
        mc = multinomial_coefficients(n, n)
        mc1 = {}
        for (k, v) in mc.items():
            mc1[_tuple(filter(None, k))] = v
        mc = mc1
        t = [n] + [0] * (m - 1)
        t1 = _tuple(t)
        b = _tuple(filter(None, t1))
        yield (t1, mc[b])
        if n:
            j = 0
        else:
            j = m
        while j < m - 1:
            tj = t[j]
            if j:
                t[j] = 0
                t[0] = tj
            if tj > 1:
                t[j + 1] += 1
                j = 0
            else:
                j += 1
                t[j] += 1
            t[0] -= 1
            t1 = _tuple(t)
            b = _tuple(filter(None, t1))
            yield (t1, mc[b])