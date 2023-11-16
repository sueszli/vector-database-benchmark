import numpy as np
import operator
from . import linear_sum_assignment, OptimizeResult
from ._optimize import _check_unknown_options
from scipy._lib._util import check_random_state
import itertools
QUADRATIC_ASSIGNMENT_METHODS = ['faq', '2opt']

def quadratic_assignment(A, B, method='faq', options=None):
    if False:
        print('Hello World!')
    '\n    Approximates solution to the quadratic assignment problem and\n    the graph matching problem.\n\n    Quadratic assignment solves problems of the following form:\n\n    .. math::\n\n        \\min_P & \\ {\\ \\text{trace}(A^T P B P^T)}\\\\\n        \\mbox{s.t. } & {P \\ \\epsilon \\ \\mathcal{P}}\\\\\n\n    where :math:`\\mathcal{P}` is the set of all permutation matrices,\n    and :math:`A` and :math:`B` are square matrices.\n\n    Graph matching tries to *maximize* the same objective function.\n    This algorithm can be thought of as finding the alignment of the\n    nodes of two graphs that minimizes the number of induced edge\n    disagreements, or, in the case of weighted graphs, the sum of squared\n    edge weight differences.\n\n    Note that the quadratic assignment problem is NP-hard. The results given\n    here are approximations and are not guaranteed to be optimal.\n\n\n    Parameters\n    ----------\n    A : 2-D array, square\n        The square matrix :math:`A` in the objective function above.\n\n    B : 2-D array, square\n        The square matrix :math:`B` in the objective function above.\n\n    method :  str in {\'faq\', \'2opt\'} (default: \'faq\')\n        The algorithm used to solve the problem.\n        :ref:`\'faq\' <optimize.qap-faq>` (default) and\n        :ref:`\'2opt\' <optimize.qap-2opt>` are available.\n\n    options : dict, optional\n        A dictionary of solver options. All solvers support the following:\n\n        maximize : bool (default: False)\n            Maximizes the objective function if ``True``.\n\n        partial_match : 2-D array of integers, optional (default: None)\n            Fixes part of the matching. Also known as a "seed" [2]_.\n\n            Each row of `partial_match` specifies a pair of matched nodes:\n            node ``partial_match[i, 0]`` of `A` is matched to node\n            ``partial_match[i, 1]`` of `B`. The array has shape ``(m, 2)``,\n            where ``m`` is not greater than the number of nodes, :math:`n`.\n\n        rng : {None, int, `numpy.random.Generator`,\n               `numpy.random.RandomState`}, optional\n\n            If `seed` is None (or `np.random`), the `numpy.random.RandomState`\n            singleton is used.\n            If `seed` is an int, a new ``RandomState`` instance is used,\n            seeded with `seed`.\n            If `seed` is already a ``Generator`` or ``RandomState`` instance then\n            that instance is used.\n\n        For method-specific options, see\n        :func:`show_options(\'quadratic_assignment\') <show_options>`.\n\n    Returns\n    -------\n    res : OptimizeResult\n        `OptimizeResult` containing the following fields.\n\n        col_ind : 1-D array\n            Column indices corresponding to the best permutation found of the\n            nodes of `B`.\n        fun : float\n            The objective value of the solution.\n        nit : int\n            The number of iterations performed during optimization.\n\n    Notes\n    -----\n    The default method :ref:`\'faq\' <optimize.qap-faq>` uses the Fast\n    Approximate QAP algorithm [1]_; it typically offers the best combination of\n    speed and accuracy.\n    Method :ref:`\'2opt\' <optimize.qap-2opt>` can be computationally expensive,\n    but may be a useful alternative, or it can be used to refine the solution\n    returned by another method.\n\n    References\n    ----------\n    .. [1] J.T. Vogelstein, J.M. Conroy, V. Lyzinski, L.J. Podrazik,\n           S.G. Kratzer, E.T. Harley, D.E. Fishkind, R.J. Vogelstein, and\n           C.E. Priebe, "Fast approximate quadratic programming for graph\n           matching," PLOS one, vol. 10, no. 4, p. e0121002, 2015,\n           :doi:`10.1371/journal.pone.0121002`\n\n    .. [2] D. Fishkind, S. Adali, H. Patsolic, L. Meng, D. Singh, V. Lyzinski,\n           C. Priebe, "Seeded graph matching", Pattern Recognit. 87 (2019):\n           203-215, :doi:`10.1016/j.patcog.2018.09.014`\n\n    .. [3] "2-opt," Wikipedia.\n           https://en.wikipedia.org/wiki/2-opt\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.optimize import quadratic_assignment\n    >>> A = np.array([[0, 80, 150, 170], [80, 0, 130, 100],\n    ...               [150, 130, 0, 120], [170, 100, 120, 0]])\n    >>> B = np.array([[0, 5, 2, 7], [0, 0, 3, 8],\n    ...               [0, 0, 0, 3], [0, 0, 0, 0]])\n    >>> res = quadratic_assignment(A, B)\n    >>> print(res)\n         fun: 3260\n     col_ind: [0 3 2 1]\n         nit: 9\n\n    The see the relationship between the returned ``col_ind`` and ``fun``,\n    use ``col_ind`` to form the best permutation matrix found, then evaluate\n    the objective function :math:`f(P) = trace(A^T P B P^T )`.\n\n    >>> perm = res[\'col_ind\']\n    >>> P = np.eye(len(A), dtype=int)[perm]\n    >>> fun = np.trace(A.T @ P @ B @ P.T)\n    >>> print(fun)\n    3260\n\n    Alternatively, to avoid constructing the permutation matrix explicitly,\n    directly permute the rows and columns of the distance matrix.\n\n    >>> fun = np.trace(A.T @ B[perm][:, perm])\n    >>> print(fun)\n    3260\n\n    Although not guaranteed in general, ``quadratic_assignment`` happens to\n    have found the globally optimal solution.\n\n    >>> from itertools import permutations\n    >>> perm_opt, fun_opt = None, np.inf\n    >>> for perm in permutations([0, 1, 2, 3]):\n    ...     perm = np.array(perm)\n    ...     fun = np.trace(A.T @ B[perm][:, perm])\n    ...     if fun < fun_opt:\n    ...         fun_opt, perm_opt = fun, perm\n    >>> print(np.array_equal(perm_opt, res[\'col_ind\']))\n    True\n\n    Here is an example for which the default method,\n    :ref:`\'faq\' <optimize.qap-faq>`, does not find the global optimum.\n\n    >>> A = np.array([[0, 5, 8, 6], [5, 0, 5, 1],\n    ...               [8, 5, 0, 2], [6, 1, 2, 0]])\n    >>> B = np.array([[0, 1, 8, 4], [1, 0, 5, 2],\n    ...               [8, 5, 0, 5], [4, 2, 5, 0]])\n    >>> res = quadratic_assignment(A, B)\n    >>> print(res)\n         fun: 178\n     col_ind: [1 0 3 2]\n         nit: 13\n\n    If accuracy is important, consider using  :ref:`\'2opt\' <optimize.qap-2opt>`\n    to refine the solution.\n\n    >>> guess = np.array([np.arange(len(A)), res.col_ind]).T\n    >>> res = quadratic_assignment(A, B, method="2opt",\n    ...                            options = {\'partial_guess\': guess})\n    >>> print(res)\n         fun: 176\n     col_ind: [1 2 3 0]\n         nit: 17\n\n    '
    if options is None:
        options = {}
    method = method.lower()
    methods = {'faq': _quadratic_assignment_faq, '2opt': _quadratic_assignment_2opt}
    if method not in methods:
        raise ValueError(f'method {method} must be in {methods}.')
    res = methods[method](A, B, **options)
    return res

def _calc_score(A, B, perm):
    if False:
        return 10
    return np.sum(A * B[perm][:, perm])

def _common_input_validation(A, B, partial_match):
    if False:
        while True:
            i = 10
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    if partial_match is None:
        partial_match = np.array([[], []]).T
    partial_match = np.atleast_2d(partial_match).astype(int)
    msg = None
    if A.shape[0] != A.shape[1]:
        msg = '`A` must be square'
    elif B.shape[0] != B.shape[1]:
        msg = '`B` must be square'
    elif A.ndim != 2 or B.ndim != 2:
        msg = '`A` and `B` must have exactly two dimensions'
    elif A.shape != B.shape:
        msg = '`A` and `B` matrices must be of equal size'
    elif partial_match.shape[0] > A.shape[0]:
        msg = '`partial_match` can have only as many seeds as there are nodes'
    elif partial_match.shape[1] != 2:
        msg = '`partial_match` must have two columns'
    elif partial_match.ndim != 2:
        msg = '`partial_match` must have exactly two dimensions'
    elif (partial_match < 0).any():
        msg = '`partial_match` must contain only positive indices'
    elif (partial_match >= len(A)).any():
        msg = '`partial_match` entries must be less than number of nodes'
    elif not len(set(partial_match[:, 0])) == len(partial_match[:, 0]) or not len(set(partial_match[:, 1])) == len(partial_match[:, 1]):
        msg = '`partial_match` column entries must be unique'
    if msg is not None:
        raise ValueError(msg)
    return (A, B, partial_match)

def _quadratic_assignment_faq(A, B, maximize=False, partial_match=None, rng=None, P0='barycenter', shuffle_input=False, maxiter=30, tol=0.03, **unknown_options):
    if False:
        for i in range(10):
            print('nop')
    'Solve the quadratic assignment problem (approximately).\n\n    This function solves the Quadratic Assignment Problem (QAP) and the\n    Graph Matching Problem (GMP) using the Fast Approximate QAP Algorithm\n    (FAQ) [1]_.\n\n    Quadratic assignment solves problems of the following form:\n\n    .. math::\n\n        \\min_P & \\ {\\ \\text{trace}(A^T P B P^T)}\\\\\n        \\mbox{s.t. } & {P \\ \\epsilon \\ \\mathcal{P}}\\\\\n\n    where :math:`\\mathcal{P}` is the set of all permutation matrices,\n    and :math:`A` and :math:`B` are square matrices.\n\n    Graph matching tries to *maximize* the same objective function.\n    This algorithm can be thought of as finding the alignment of the\n    nodes of two graphs that minimizes the number of induced edge\n    disagreements, or, in the case of weighted graphs, the sum of squared\n    edge weight differences.\n\n    Note that the quadratic assignment problem is NP-hard. The results given\n    here are approximations and are not guaranteed to be optimal.\n\n    Parameters\n    ----------\n    A : 2-D array, square\n        The square matrix :math:`A` in the objective function above.\n    B : 2-D array, square\n        The square matrix :math:`B` in the objective function above.\n    method :  str in {\'faq\', \'2opt\'} (default: \'faq\')\n        The algorithm used to solve the problem. This is the method-specific\n        documentation for \'faq\'.\n        :ref:`\'2opt\' <optimize.qap-2opt>` is also available.\n\n    Options\n    -------\n    maximize : bool (default: False)\n        Maximizes the objective function if ``True``.\n    partial_match : 2-D array of integers, optional (default: None)\n        Fixes part of the matching. Also known as a "seed" [2]_.\n\n        Each row of `partial_match` specifies a pair of matched nodes:\n        node ``partial_match[i, 0]`` of `A` is matched to node\n        ``partial_match[i, 1]`` of `B`. The array has shape ``(m, 2)``, where\n        ``m`` is not greater than the number of nodes, :math:`n`.\n\n    rng : {None, int, `numpy.random.Generator`,\n           `numpy.random.RandomState`}, optional\n\n        If `seed` is None (or `np.random`), the `numpy.random.RandomState`\n        singleton is used.\n        If `seed` is an int, a new ``RandomState`` instance is used,\n        seeded with `seed`.\n        If `seed` is already a ``Generator`` or ``RandomState`` instance then\n        that instance is used.\n    P0 : 2-D array, "barycenter", or "randomized" (default: "barycenter")\n        Initial position. Must be a doubly-stochastic matrix [3]_.\n\n        If the initial position is an array, it must be a doubly stochastic\n        matrix of size :math:`m\' \\times m\'` where :math:`m\' = n - m`.\n\n        If ``"barycenter"`` (default), the initial position is the barycenter\n        of the Birkhoff polytope (the space of doubly stochastic matrices).\n        This is a :math:`m\' \\times m\'` matrix with all entries equal to\n        :math:`1 / m\'`.\n\n        If ``"randomized"`` the initial search position is\n        :math:`P_0 = (J + K) / 2`, where :math:`J` is the barycenter and\n        :math:`K` is a random doubly stochastic matrix.\n    shuffle_input : bool (default: False)\n        Set to `True` to resolve degenerate gradients randomly. For\n        non-degenerate gradients this option has no effect.\n    maxiter : int, positive (default: 30)\n        Integer specifying the max number of Frank-Wolfe iterations performed.\n    tol : float (default: 0.03)\n        Tolerance for termination. Frank-Wolfe iteration terminates when\n        :math:`\\frac{||P_{i}-P_{i+1}||_F}{\\sqrt{m\')}} \\leq tol`,\n        where :math:`i` is the iteration number.\n\n    Returns\n    -------\n    res : OptimizeResult\n        `OptimizeResult` containing the following fields.\n\n        col_ind : 1-D array\n            Column indices corresponding to the best permutation found of the\n            nodes of `B`.\n        fun : float\n            The objective value of the solution.\n        nit : int\n            The number of Frank-Wolfe iterations performed.\n\n    Notes\n    -----\n    The algorithm may be sensitive to the initial permutation matrix (or\n    search "position") due to the possibility of several local minima\n    within the feasible region. A barycenter initialization is more likely to\n    result in a better solution than a single random initialization. However,\n    calling ``quadratic_assignment`` several times with different random\n    initializations may result in a better optimum at the cost of longer\n    total execution time.\n\n    Examples\n    --------\n    As mentioned above, a barycenter initialization often results in a better\n    solution than a single random initialization.\n\n    >>> from numpy.random import default_rng\n    >>> rng = default_rng()\n    >>> n = 15\n    >>> A = rng.random((n, n))\n    >>> B = rng.random((n, n))\n    >>> res = quadratic_assignment(A, B)  # FAQ is default method\n    >>> print(res.fun)\n    46.871483385480545  # may vary\n\n    >>> options = {"P0": "randomized"}  # use randomized initialization\n    >>> res = quadratic_assignment(A, B, options=options)\n    >>> print(res.fun)\n    47.224831071310625 # may vary\n\n    However, consider running from several randomized initializations and\n    keeping the best result.\n\n    >>> res = min([quadratic_assignment(A, B, options=options)\n    ...            for i in range(30)], key=lambda x: x.fun)\n    >>> print(res.fun)\n    46.671852533681516 # may vary\n\n    The \'2-opt\' method can be used to further refine the results.\n\n    >>> options = {"partial_guess": np.array([np.arange(n), res.col_ind]).T}\n    >>> res = quadratic_assignment(A, B, method="2opt", options=options)\n    >>> print(res.fun)\n    46.47160735721583 # may vary\n\n    References\n    ----------\n    .. [1] J.T. Vogelstein, J.M. Conroy, V. Lyzinski, L.J. Podrazik,\n           S.G. Kratzer, E.T. Harley, D.E. Fishkind, R.J. Vogelstein, and\n           C.E. Priebe, "Fast approximate quadratic programming for graph\n           matching," PLOS one, vol. 10, no. 4, p. e0121002, 2015,\n           :doi:`10.1371/journal.pone.0121002`\n\n    .. [2] D. Fishkind, S. Adali, H. Patsolic, L. Meng, D. Singh, V. Lyzinski,\n           C. Priebe, "Seeded graph matching", Pattern Recognit. 87 (2019):\n           203-215, :doi:`10.1016/j.patcog.2018.09.014`\n\n    .. [3] "Doubly stochastic Matrix," Wikipedia.\n           https://en.wikipedia.org/wiki/Doubly_stochastic_matrix\n\n    '
    _check_unknown_options(unknown_options)
    maxiter = operator.index(maxiter)
    (A, B, partial_match) = _common_input_validation(A, B, partial_match)
    msg = None
    if isinstance(P0, str) and P0 not in {'barycenter', 'randomized'}:
        msg = "Invalid 'P0' parameter string"
    elif maxiter <= 0:
        msg = "'maxiter' must be a positive integer"
    elif tol <= 0:
        msg = "'tol' must be a positive float"
    if msg is not None:
        raise ValueError(msg)
    rng = check_random_state(rng)
    n = len(A)
    n_seeds = len(partial_match)
    n_unseed = n - n_seeds
    if not isinstance(P0, str):
        P0 = np.atleast_2d(P0)
        if P0.shape != (n_unseed, n_unseed):
            msg = "`P0` matrix must have shape m' x m', where m'=n-m"
        elif (P0 < 0).any() or not np.allclose(np.sum(P0, axis=0), 1) or (not np.allclose(np.sum(P0, axis=1), 1)):
            msg = '`P0` matrix must be doubly stochastic'
        if msg is not None:
            raise ValueError(msg)
    elif P0 == 'barycenter':
        P0 = np.ones((n_unseed, n_unseed)) / n_unseed
    elif P0 == 'randomized':
        J = np.ones((n_unseed, n_unseed)) / n_unseed
        K = _doubly_stochastic(rng.uniform(size=(n_unseed, n_unseed)))
        P0 = (J + K) / 2
    if n == 0 or n_seeds == n:
        score = _calc_score(A, B, partial_match[:, 1])
        res = {'col_ind': partial_match[:, 1], 'fun': score, 'nit': 0}
        return OptimizeResult(res)
    obj_func_scalar = 1
    if maximize:
        obj_func_scalar = -1
    nonseed_B = np.setdiff1d(range(n), partial_match[:, 1])
    if shuffle_input:
        nonseed_B = rng.permutation(nonseed_B)
    nonseed_A = np.setdiff1d(range(n), partial_match[:, 0])
    perm_A = np.concatenate([partial_match[:, 0], nonseed_A])
    perm_B = np.concatenate([partial_match[:, 1], nonseed_B])
    (A11, A12, A21, A22) = _split_matrix(A[perm_A][:, perm_A], n_seeds)
    (B11, B12, B21, B22) = _split_matrix(B[perm_B][:, perm_B], n_seeds)
    const_sum = A21 @ B21.T + A12.T @ B12
    P = P0
    for n_iter in range(1, maxiter + 1):
        grad_fp = const_sum + A22 @ P @ B22.T + A22.T @ P @ B22
        (_, cols) = linear_sum_assignment(grad_fp, maximize=maximize)
        Q = np.eye(n_unseed)[cols]
        R = P - Q
        b21 = (R.T @ A21 * B21).sum()
        b12 = (R.T @ A12.T * B12.T).sum()
        AR22 = A22.T @ R
        BR22 = B22 @ R.T
        b22a = (AR22 * B22.T[cols]).sum()
        b22b = (A22 * BR22[cols]).sum()
        a = (AR22.T * BR22).sum()
        b = b21 + b12 + b22a + b22b
        if a * obj_func_scalar > 0 and 0 <= -b / (2 * a) <= 1:
            alpha = -b / (2 * a)
        else:
            alpha = np.argmin([0, (b + a) * obj_func_scalar])
        P_i1 = alpha * P + (1 - alpha) * Q
        if np.linalg.norm(P - P_i1) / np.sqrt(n_unseed) < tol:
            P = P_i1
            break
        P = P_i1
    (_, col) = linear_sum_assignment(P, maximize=True)
    perm = np.concatenate((np.arange(n_seeds), col + n_seeds))
    unshuffled_perm = np.zeros(n, dtype=int)
    unshuffled_perm[perm_A] = perm_B[perm]
    score = _calc_score(A, B, unshuffled_perm)
    res = {'col_ind': unshuffled_perm, 'fun': score, 'nit': n_iter}
    return OptimizeResult(res)

def _split_matrix(X, n):
    if False:
        while True:
            i = 10
    (upper, lower) = (X[:n], X[n:])
    return (upper[:, :n], upper[:, n:], lower[:, :n], lower[:, n:])

def _doubly_stochastic(P, tol=0.001):
    if False:
        return 10
    max_iter = 1000
    c = 1 / P.sum(axis=0)
    r = 1 / (P @ c)
    P_eps = P
    for it in range(max_iter):
        if (np.abs(P_eps.sum(axis=1) - 1) < tol).all() and (np.abs(P_eps.sum(axis=0) - 1) < tol).all():
            break
        c = 1 / (r @ P)
        r = 1 / (P @ c)
        P_eps = r[:, None] * P * c
    return P_eps

def _quadratic_assignment_2opt(A, B, maximize=False, rng=None, partial_match=None, partial_guess=None, **unknown_options):
    if False:
        for i in range(10):
            print('nop')
    'Solve the quadratic assignment problem (approximately).\n\n    This function solves the Quadratic Assignment Problem (QAP) and the\n    Graph Matching Problem (GMP) using the 2-opt algorithm [1]_.\n\n    Quadratic assignment solves problems of the following form:\n\n    .. math::\n\n        \\min_P & \\ {\\ \\text{trace}(A^T P B P^T)}\\\\\n        \\mbox{s.t. } & {P \\ \\epsilon \\ \\mathcal{P}}\\\\\n\n    where :math:`\\mathcal{P}` is the set of all permutation matrices,\n    and :math:`A` and :math:`B` are square matrices.\n\n    Graph matching tries to *maximize* the same objective function.\n    This algorithm can be thought of as finding the alignment of the\n    nodes of two graphs that minimizes the number of induced edge\n    disagreements, or, in the case of weighted graphs, the sum of squared\n    edge weight differences.\n\n    Note that the quadratic assignment problem is NP-hard. The results given\n    here are approximations and are not guaranteed to be optimal.\n\n    Parameters\n    ----------\n    A : 2-D array, square\n        The square matrix :math:`A` in the objective function above.\n    B : 2-D array, square\n        The square matrix :math:`B` in the objective function above.\n    method :  str in {\'faq\', \'2opt\'} (default: \'faq\')\n        The algorithm used to solve the problem. This is the method-specific\n        documentation for \'2opt\'.\n        :ref:`\'faq\' <optimize.qap-faq>` is also available.\n\n    Options\n    -------\n    maximize : bool (default: False)\n        Maximizes the objective function if ``True``.\n    rng : {None, int, `numpy.random.Generator`,\n           `numpy.random.RandomState`}, optional\n\n        If `seed` is None (or `np.random`), the `numpy.random.RandomState`\n        singleton is used.\n        If `seed` is an int, a new ``RandomState`` instance is used,\n        seeded with `seed`.\n        If `seed` is already a ``Generator`` or ``RandomState`` instance then\n        that instance is used.\n    partial_match : 2-D array of integers, optional (default: None)\n        Fixes part of the matching. Also known as a "seed" [2]_.\n\n        Each row of `partial_match` specifies a pair of matched nodes: node\n        ``partial_match[i, 0]`` of `A` is matched to node\n        ``partial_match[i, 1]`` of `B`. The array has shape ``(m, 2)``,\n        where ``m`` is not greater than the number of nodes, :math:`n`.\n    partial_guess : 2-D array of integers, optional (default: None)\n        A guess for the matching between the two matrices. Unlike\n        `partial_match`, `partial_guess` does not fix the indices; they are\n        still free to be optimized.\n\n        Each row of `partial_guess` specifies a pair of matched nodes: node\n        ``partial_guess[i, 0]`` of `A` is matched to node\n        ``partial_guess[i, 1]`` of `B`. The array has shape ``(m, 2)``,\n        where ``m`` is not greater than the number of nodes, :math:`n`.\n\n    Returns\n    -------\n    res : OptimizeResult\n        `OptimizeResult` containing the following fields.\n\n        col_ind : 1-D array\n            Column indices corresponding to the best permutation found of the\n            nodes of `B`.\n        fun : float\n            The objective value of the solution.\n        nit : int\n            The number of iterations performed during optimization.\n\n    Notes\n    -----\n    This is a greedy algorithm that works similarly to bubble sort: beginning\n    with an initial permutation, it iteratively swaps pairs of indices to\n    improve the objective function until no such improvements are possible.\n\n    References\n    ----------\n    .. [1] "2-opt," Wikipedia.\n           https://en.wikipedia.org/wiki/2-opt\n\n    .. [2] D. Fishkind, S. Adali, H. Patsolic, L. Meng, D. Singh, V. Lyzinski,\n           C. Priebe, "Seeded graph matching", Pattern Recognit. 87 (2019):\n           203-215, https://doi.org/10.1016/j.patcog.2018.09.014\n\n    '
    _check_unknown_options(unknown_options)
    rng = check_random_state(rng)
    (A, B, partial_match) = _common_input_validation(A, B, partial_match)
    N = len(A)
    if N == 0 or partial_match.shape[0] == N:
        score = _calc_score(A, B, partial_match[:, 1])
        res = {'col_ind': partial_match[:, 1], 'fun': score, 'nit': 0}
        return OptimizeResult(res)
    if partial_guess is None:
        partial_guess = np.array([[], []]).T
    partial_guess = np.atleast_2d(partial_guess).astype(int)
    msg = None
    if partial_guess.shape[0] > A.shape[0]:
        msg = '`partial_guess` can have only as many entries as there are nodes'
    elif partial_guess.shape[1] != 2:
        msg = '`partial_guess` must have two columns'
    elif partial_guess.ndim != 2:
        msg = '`partial_guess` must have exactly two dimensions'
    elif (partial_guess < 0).any():
        msg = '`partial_guess` must contain only positive indices'
    elif (partial_guess >= len(A)).any():
        msg = '`partial_guess` entries must be less than number of nodes'
    elif not len(set(partial_guess[:, 0])) == len(partial_guess[:, 0]) or not len(set(partial_guess[:, 1])) == len(partial_guess[:, 1]):
        msg = '`partial_guess` column entries must be unique'
    if msg is not None:
        raise ValueError(msg)
    fixed_rows = None
    if partial_match.size or partial_guess.size:
        guess_rows = np.zeros(N, dtype=bool)
        guess_cols = np.zeros(N, dtype=bool)
        fixed_rows = np.zeros(N, dtype=bool)
        fixed_cols = np.zeros(N, dtype=bool)
        perm = np.zeros(N, dtype=int)
        (rg, cg) = partial_guess.T
        guess_rows[rg] = True
        guess_cols[cg] = True
        perm[guess_rows] = cg
        (rf, cf) = partial_match.T
        fixed_rows[rf] = True
        fixed_cols[cf] = True
        perm[fixed_rows] = cf
        random_rows = ~fixed_rows & ~guess_rows
        random_cols = ~fixed_cols & ~guess_cols
        perm[random_rows] = rng.permutation(np.arange(N)[random_cols])
    else:
        perm = rng.permutation(np.arange(N))
    best_score = _calc_score(A, B, perm)
    i_free = np.arange(N)
    if fixed_rows is not None:
        i_free = i_free[~fixed_rows]
    better = operator.gt if maximize else operator.lt
    n_iter = 0
    done = False
    while not done:
        for (i, j) in itertools.combinations_with_replacement(i_free, 2):
            n_iter += 1
            (perm[i], perm[j]) = (perm[j], perm[i])
            score = _calc_score(A, B, perm)
            if better(score, best_score):
                best_score = score
                break
            (perm[i], perm[j]) = (perm[j], perm[i])
        else:
            done = True
    res = {'col_ind': perm, 'fun': best_score, 'nit': n_iter}
    return OptimizeResult(res)