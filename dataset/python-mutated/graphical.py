"""Test sequences for graphiness.
"""
import heapq
import networkx as nx
__all__ = ['is_graphical', 'is_multigraphical', 'is_pseudographical', 'is_digraphical', 'is_valid_degree_sequence_erdos_gallai', 'is_valid_degree_sequence_havel_hakimi']

@nx._dispatch(graphs=None)
def is_graphical(sequence, method='eg'):
    if False:
        print('Hello World!')
    'Returns True if sequence is a valid degree sequence.\n\n    A degree sequence is valid if some graph can realize it.\n\n    Parameters\n    ----------\n    sequence : list or iterable container\n        A sequence of integer node degrees\n\n    method : "eg" | "hh"  (default: \'eg\')\n        The method used to validate the degree sequence.\n        "eg" corresponds to the Erdős-Gallai algorithm\n        [EG1960]_, [choudum1986]_, and\n        "hh" to the Havel-Hakimi algorithm\n        [havel1955]_, [hakimi1962]_, [CL1996]_.\n\n    Returns\n    -------\n    valid : bool\n        True if the sequence is a valid degree sequence and False if not.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4)\n    >>> sequence = (d for n, d in G.degree())\n    >>> nx.is_graphical(sequence)\n    True\n\n    To test a non-graphical sequence:\n    >>> sequence_list = [d for n, d in G.degree()]\n    >>> sequence_list[-1] += 1\n    >>> nx.is_graphical(sequence_list)\n    False\n\n    References\n    ----------\n    .. [EG1960] Erdős and Gallai, Mat. Lapok 11 264, 1960.\n    .. [choudum1986] S.A. Choudum. "A simple proof of the Erdős-Gallai theorem on\n       graph sequences." Bulletin of the Australian Mathematical Society, 33,\n       pp 67-70, 1986. https://doi.org/10.1017/S0004972700002872\n    .. [havel1955] Havel, V. "A Remark on the Existence of Finite Graphs"\n       Casopis Pest. Mat. 80, 477-480, 1955.\n    .. [hakimi1962] Hakimi, S. "On the Realizability of a Set of Integers as\n       Degrees of the Vertices of a Graph." SIAM J. Appl. Math. 10, 496-506, 1962.\n    .. [CL1996] G. Chartrand and L. Lesniak, "Graphs and Digraphs",\n       Chapman and Hall/CRC, 1996.\n    '
    if method == 'eg':
        valid = is_valid_degree_sequence_erdos_gallai(list(sequence))
    elif method == 'hh':
        valid = is_valid_degree_sequence_havel_hakimi(list(sequence))
    else:
        msg = "`method` must be 'eg' or 'hh'"
        raise nx.NetworkXException(msg)
    return valid

def _basic_graphical_tests(deg_sequence):
    if False:
        while True:
            i = 10
    deg_sequence = nx.utils.make_list_of_ints(deg_sequence)
    p = len(deg_sequence)
    num_degs = [0] * p
    (dmax, dmin, dsum, n) = (0, p, 0, 0)
    for d in deg_sequence:
        if d < 0 or d >= p:
            raise nx.NetworkXUnfeasible
        elif d > 0:
            (dmax, dmin, dsum, n) = (max(dmax, d), min(dmin, d), dsum + d, n + 1)
            num_degs[d] += 1
    if dsum % 2 or dsum > n * (n - 1):
        raise nx.NetworkXUnfeasible
    return (dmax, dmin, dsum, n, num_degs)

@nx._dispatch(graphs=None)
def is_valid_degree_sequence_havel_hakimi(deg_sequence):
    if False:
        return 10
    'Returns True if deg_sequence can be realized by a simple graph.\n\n    The validation proceeds using the Havel-Hakimi theorem\n    [havel1955]_, [hakimi1962]_, [CL1996]_.\n    Worst-case run time is $O(s)$ where $s$ is the sum of the sequence.\n\n    Parameters\n    ----------\n    deg_sequence : list\n        A list of integers where each element specifies the degree of a node\n        in a graph.\n\n    Returns\n    -------\n    valid : bool\n        True if deg_sequence is graphical and False if not.\n\n    Examples\n    --------\n    >>> G = nx.Graph([(1, 2), (1, 3), (2, 3), (3, 4), (4, 2), (5, 1), (5, 4)])\n    >>> sequence = (d for _, d in G.degree())\n    >>> nx.is_valid_degree_sequence_havel_hakimi(sequence)\n    True\n\n    To test a non-valid sequence:\n    >>> sequence_list = [d for _, d in G.degree()]\n    >>> sequence_list[-1] += 1\n    >>> nx.is_valid_degree_sequence_havel_hakimi(sequence_list)\n    False\n\n    Notes\n    -----\n    The ZZ condition says that for the sequence d if\n\n    .. math::\n        |d| >= \\frac{(\\max(d) + \\min(d) + 1)^2}{4*\\min(d)}\n\n    then d is graphical.  This was shown in Theorem 6 in [1]_.\n\n    References\n    ----------\n    .. [1] I.E. Zverovich and V.E. Zverovich. "Contributions to the theory\n       of graphic sequences", Discrete Mathematics, 105, pp. 292-303 (1992).\n    .. [havel1955] Havel, V. "A Remark on the Existence of Finite Graphs"\n       Casopis Pest. Mat. 80, 477-480, 1955.\n    .. [hakimi1962] Hakimi, S. "On the Realizability of a Set of Integers as\n       Degrees of the Vertices of a Graph." SIAM J. Appl. Math. 10, 496-506, 1962.\n    .. [CL1996] G. Chartrand and L. Lesniak, "Graphs and Digraphs",\n       Chapman and Hall/CRC, 1996.\n    '
    try:
        (dmax, dmin, dsum, n, num_degs) = _basic_graphical_tests(deg_sequence)
    except nx.NetworkXUnfeasible:
        return False
    if n == 0 or 4 * dmin * n >= (dmax + dmin + 1) * (dmax + dmin + 1):
        return True
    modstubs = [0] * (dmax + 1)
    while n > 0:
        while num_degs[dmax] == 0:
            dmax -= 1
        if dmax > n - 1:
            return False
        (num_degs[dmax], n) = (num_degs[dmax] - 1, n - 1)
        mslen = 0
        k = dmax
        for i in range(dmax):
            while num_degs[k] == 0:
                k -= 1
            (num_degs[k], n) = (num_degs[k] - 1, n - 1)
            if k > 1:
                modstubs[mslen] = k - 1
                mslen += 1
        for i in range(mslen):
            stub = modstubs[i]
            (num_degs[stub], n) = (num_degs[stub] + 1, n + 1)
    return True

@nx._dispatch(graphs=None)
def is_valid_degree_sequence_erdos_gallai(deg_sequence):
    if False:
        print('Hello World!')
    'Returns True if deg_sequence can be realized by a simple graph.\n\n    The validation is done using the Erdős-Gallai theorem [EG1960]_.\n\n    Parameters\n    ----------\n    deg_sequence : list\n        A list of integers\n\n    Returns\n    -------\n    valid : bool\n        True if deg_sequence is graphical and False if not.\n\n    Examples\n    --------\n    >>> G = nx.Graph([(1, 2), (1, 3), (2, 3), (3, 4), (4, 2), (5, 1), (5, 4)])\n    >>> sequence = (d for _, d in G.degree())\n    >>> nx.is_valid_degree_sequence_erdos_gallai(sequence)\n    True\n\n    To test a non-valid sequence:\n    >>> sequence_list = [d for _, d in G.degree()]\n    >>> sequence_list[-1] += 1\n    >>> nx.is_valid_degree_sequence_erdos_gallai(sequence_list)\n    False\n\n    Notes\n    -----\n\n    This implementation uses an equivalent form of the Erdős-Gallai criterion.\n    Worst-case run time is $O(n)$ where $n$ is the length of the sequence.\n\n    Specifically, a sequence d is graphical if and only if the\n    sum of the sequence is even and for all strong indices k in the sequence,\n\n     .. math::\n\n       \\sum_{i=1}^{k} d_i \\leq k(k-1) + \\sum_{j=k+1}^{n} \\min(d_i,k)\n             = k(n-1) - ( k \\sum_{j=0}^{k-1} n_j - \\sum_{j=0}^{k-1} j n_j )\n\n    A strong index k is any index where d_k >= k and the value n_j is the\n    number of occurrences of j in d.  The maximal strong index is called the\n    Durfee index.\n\n    This particular rearrangement comes from the proof of Theorem 3 in [2]_.\n\n    The ZZ condition says that for the sequence d if\n\n    .. math::\n        |d| >= \\frac{(\\max(d) + \\min(d) + 1)^2}{4*\\min(d)}\n\n    then d is graphical.  This was shown in Theorem 6 in [2]_.\n\n    References\n    ----------\n    .. [1] A. Tripathi and S. Vijay. "A note on a theorem of Erdős & Gallai",\n       Discrete Mathematics, 265, pp. 417-420 (2003).\n    .. [2] I.E. Zverovich and V.E. Zverovich. "Contributions to the theory\n       of graphic sequences", Discrete Mathematics, 105, pp. 292-303 (1992).\n    .. [EG1960] Erdős and Gallai, Mat. Lapok 11 264, 1960.\n    '
    try:
        (dmax, dmin, dsum, n, num_degs) = _basic_graphical_tests(deg_sequence)
    except nx.NetworkXUnfeasible:
        return False
    if n == 0 or 4 * dmin * n >= (dmax + dmin + 1) * (dmax + dmin + 1):
        return True
    (k, sum_deg, sum_nj, sum_jnj) = (0, 0, 0, 0)
    for dk in range(dmax, dmin - 1, -1):
        if dk < k + 1:
            return True
        if num_degs[dk] > 0:
            run_size = num_degs[dk]
            if dk < k + run_size:
                run_size = dk - k
            sum_deg += run_size * dk
            for v in range(run_size):
                sum_nj += num_degs[k + v]
                sum_jnj += (k + v) * num_degs[k + v]
            k += run_size
            if sum_deg > k * (n - 1) - k * sum_nj + sum_jnj:
                return False
    return True

@nx._dispatch(graphs=None)
def is_multigraphical(sequence):
    if False:
        i = 10
        return i + 15
    'Returns True if some multigraph can realize the sequence.\n\n    Parameters\n    ----------\n    sequence : list\n        A list of integers\n\n    Returns\n    -------\n    valid : bool\n        True if deg_sequence is a multigraphic degree sequence and False if not.\n\n    Examples\n    --------\n    >>> G = nx.MultiGraph([(1, 2), (1, 3), (2, 3), (3, 4), (4, 2), (5, 1), (5, 4)])\n    >>> sequence = (d for _, d in G.degree())\n    >>> nx.is_multigraphical(sequence)\n    True\n\n    To test a non-multigraphical sequence:\n    >>> sequence_list = [d for _, d in G.degree()]\n    >>> sequence_list[-1] += 1\n    >>> nx.is_multigraphical(sequence_list)\n    False\n\n    Notes\n    -----\n    The worst-case run time is $O(n)$ where $n$ is the length of the sequence.\n\n    References\n    ----------\n    .. [1] S. L. Hakimi. "On the realizability of a set of integers as\n       degrees of the vertices of a linear graph", J. SIAM, 10, pp. 496-506\n       (1962).\n    '
    try:
        deg_sequence = nx.utils.make_list_of_ints(sequence)
    except nx.NetworkXError:
        return False
    (dsum, dmax) = (0, 0)
    for d in deg_sequence:
        if d < 0:
            return False
        (dsum, dmax) = (dsum + d, max(dmax, d))
    if dsum % 2 or dsum < 2 * dmax:
        return False
    return True

@nx._dispatch(graphs=None)
def is_pseudographical(sequence):
    if False:
        print('Hello World!')
    'Returns True if some pseudograph can realize the sequence.\n\n    Every nonnegative integer sequence with an even sum is pseudographical\n    (see [1]_).\n\n    Parameters\n    ----------\n    sequence : list or iterable container\n        A sequence of integer node degrees\n\n    Returns\n    -------\n    valid : bool\n      True if the sequence is a pseudographic degree sequence and False if not.\n\n    Examples\n    --------\n    >>> G = nx.Graph([(1, 2), (1, 3), (2, 3), (3, 4), (4, 2), (5, 1), (5, 4)])\n    >>> sequence = (d for _, d in G.degree())\n    >>> nx.is_pseudographical(sequence)\n    True\n\n    To test a non-pseudographical sequence:\n    >>> sequence_list = [d for _, d in G.degree()]\n    >>> sequence_list[-1] += 1\n    >>> nx.is_pseudographical(sequence_list)\n    False\n\n    Notes\n    -----\n    The worst-case run time is $O(n)$ where n is the length of the sequence.\n\n    References\n    ----------\n    .. [1] F. Boesch and F. Harary. "Line removal algorithms for graphs\n       and their degree lists", IEEE Trans. Circuits and Systems, CAS-23(12),\n       pp. 778-782 (1976).\n    '
    try:
        deg_sequence = nx.utils.make_list_of_ints(sequence)
    except nx.NetworkXError:
        return False
    return sum(deg_sequence) % 2 == 0 and min(deg_sequence) >= 0

@nx._dispatch(graphs=None)
def is_digraphical(in_sequence, out_sequence):
    if False:
        i = 10
        return i + 15
    'Returns True if some directed graph can realize the in- and out-degree\n    sequences.\n\n    Parameters\n    ----------\n    in_sequence : list or iterable container\n        A sequence of integer node in-degrees\n\n    out_sequence : list or iterable container\n        A sequence of integer node out-degrees\n\n    Returns\n    -------\n    valid : bool\n      True if in and out-sequences are digraphic False if not.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 3), (3, 4), (4, 2), (5, 1), (5, 4)])\n    >>> in_seq = (d for n, d in G.in_degree())\n    >>> out_seq = (d for n, d in G.out_degree())\n    >>> nx.is_digraphical(in_seq, out_seq)\n    True\n\n    To test a non-digraphical scenario:\n    >>> in_seq_list = [d for n, d in G.in_degree()]\n    >>> in_seq_list[-1] += 1\n    >>> nx.is_digraphical(in_seq_list, out_seq)\n    False\n\n    Notes\n    -----\n    This algorithm is from Kleitman and Wang [1]_.\n    The worst case runtime is $O(s \\times \\log n)$ where $s$ and $n$ are the\n    sum and length of the sequences respectively.\n\n    References\n    ----------\n    .. [1] D.J. Kleitman and D.L. Wang\n       Algorithms for Constructing Graphs and Digraphs with Given Valences\n       and Factors, Discrete Mathematics, 6(1), pp. 79-88 (1973)\n    '
    try:
        in_deg_sequence = nx.utils.make_list_of_ints(in_sequence)
        out_deg_sequence = nx.utils.make_list_of_ints(out_sequence)
    except nx.NetworkXError:
        return False
    (sumin, sumout, nin, nout) = (0, 0, len(in_deg_sequence), len(out_deg_sequence))
    maxn = max(nin, nout)
    maxin = 0
    if maxn == 0:
        return True
    (stubheap, zeroheap) = ([], [])
    for n in range(maxn):
        (in_deg, out_deg) = (0, 0)
        if n < nout:
            out_deg = out_deg_sequence[n]
        if n < nin:
            in_deg = in_deg_sequence[n]
        if in_deg < 0 or out_deg < 0:
            return False
        (sumin, sumout, maxin) = (sumin + in_deg, sumout + out_deg, max(maxin, in_deg))
        if in_deg > 0:
            stubheap.append((-1 * out_deg, -1 * in_deg))
        elif out_deg > 0:
            zeroheap.append(-1 * out_deg)
    if sumin != sumout:
        return False
    heapq.heapify(stubheap)
    heapq.heapify(zeroheap)
    modstubs = [(0, 0)] * (maxin + 1)
    while stubheap:
        (freeout, freein) = heapq.heappop(stubheap)
        freein *= -1
        if freein > len(stubheap) + len(zeroheap):
            return False
        mslen = 0
        for i in range(freein):
            if zeroheap and (not stubheap or stubheap[0][0] > zeroheap[0]):
                stubout = heapq.heappop(zeroheap)
                stubin = 0
            else:
                (stubout, stubin) = heapq.heappop(stubheap)
            if stubout == 0:
                return False
            if stubout + 1 < 0 or stubin < 0:
                modstubs[mslen] = (stubout + 1, stubin)
                mslen += 1
        for i in range(mslen):
            stub = modstubs[i]
            if stub[1] < 0:
                heapq.heappush(stubheap, stub)
            else:
                heapq.heappush(zeroheap, stub[0])
        if freeout < 0:
            heapq.heappush(zeroheap, freeout)
    return True