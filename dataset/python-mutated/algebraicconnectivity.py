"""
Algebraic connectivity and Fiedler vectors of undirected graphs.
"""
from functools import partial
import networkx as nx
from networkx.utils import not_implemented_for, np_random_state, reverse_cuthill_mckee_ordering
__all__ = ['algebraic_connectivity', 'fiedler_vector', 'spectral_ordering', 'spectral_bisection']

class _PCGSolver:
    """Preconditioned conjugate gradient method.

    To solve Ax = b:
        M = A.diagonal() # or some other preconditioner
        solver = _PCGSolver(lambda x: A * x, lambda x: M * x)
        x = solver.solve(b)

    The inputs A and M are functions which compute
    matrix multiplication on the argument.
    A - multiply by the matrix A in Ax=b
    M - multiply by M, the preconditioner surrogate for A

    Warning: There is no limit on number of iterations.
    """

    def __init__(self, A, M):
        if False:
            print('Hello World!')
        self._A = A
        self._M = M

    def solve(self, B, tol):
        if False:
            for i in range(10):
                print('nop')
        import numpy as np
        B = np.asarray(B)
        X = np.ndarray(B.shape, order='F')
        for j in range(B.shape[1]):
            X[:, j] = self._solve(B[:, j], tol)
        return X

    def _solve(self, b, tol):
        if False:
            return 10
        import numpy as np
        import scipy as sp
        A = self._A
        M = self._M
        tol *= sp.linalg.blas.dasum(b)
        x = np.zeros(b.shape)
        r = b.copy()
        z = M(r)
        rz = sp.linalg.blas.ddot(r, z)
        p = z.copy()
        while True:
            Ap = A(p)
            alpha = rz / sp.linalg.blas.ddot(p, Ap)
            x = sp.linalg.blas.daxpy(p, x, a=alpha)
            r = sp.linalg.blas.daxpy(Ap, r, a=-alpha)
            if sp.linalg.blas.dasum(r) < tol:
                return x
            z = M(r)
            beta = sp.linalg.blas.ddot(r, z)
            (beta, rz) = (beta / rz, beta)
            p = sp.linalg.blas.daxpy(p, z, a=beta)

class _LUSolver:
    """LU factorization.

    To solve Ax = b:
        solver = _LUSolver(A)
        x = solver.solve(b)

    optional argument `tol` on solve method is ignored but included
    to match _PCGsolver API.
    """

    def __init__(self, A):
        if False:
            return 10
        import scipy as sp
        self._LU = sp.sparse.linalg.splu(A, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0, options={'Equil': True, 'SymmetricMode': True})

    def solve(self, B, tol=None):
        if False:
            i = 10
            return i + 15
        import numpy as np
        B = np.asarray(B)
        X = np.ndarray(B.shape, order='F')
        for j in range(B.shape[1]):
            X[:, j] = self._LU.solve(B[:, j])
        return X

def _preprocess_graph(G, weight):
    if False:
        i = 10
        return i + 15
    'Compute edge weights and eliminate zero-weight edges.'
    if G.is_directed():
        H = nx.MultiGraph()
        H.add_nodes_from(G)
        H.add_weighted_edges_from(((u, v, e.get(weight, 1.0)) for (u, v, e) in G.edges(data=True) if u != v), weight=weight)
        G = H
    if not G.is_multigraph():
        edges = ((u, v, abs(e.get(weight, 1.0))) for (u, v, e) in G.edges(data=True) if u != v)
    else:
        edges = ((u, v, sum((abs(e.get(weight, 1.0)) for e in G[u][v].values()))) for (u, v) in G.edges() if u != v)
    H = nx.Graph()
    H.add_nodes_from(G)
    H.add_weighted_edges_from(((u, v, e) for (u, v, e) in edges if e != 0))
    return H

def _rcm_estimate(G, nodelist):
    if False:
        while True:
            i = 10
    'Estimate the Fiedler vector using the reverse Cuthill-McKee ordering.'
    import numpy as np
    G = G.subgraph(nodelist)
    order = reverse_cuthill_mckee_ordering(G)
    n = len(nodelist)
    index = dict(zip(nodelist, range(n)))
    x = np.ndarray(n, dtype=float)
    for (i, u) in enumerate(order):
        x[index[u]] = i
    x -= (n - 1) / 2.0
    return x

def _tracemin_fiedler(L, X, normalized, tol, method):
    if False:
        for i in range(10):
            print('nop')
    "Compute the Fiedler vector of L using the TraceMIN-Fiedler algorithm.\n\n    The Fiedler vector of a connected undirected graph is the eigenvector\n    corresponding to the second smallest eigenvalue of the Laplacian matrix\n    of the graph. This function starts with the Laplacian L, not the Graph.\n\n    Parameters\n    ----------\n    L : Laplacian of a possibly weighted or normalized, but undirected graph\n\n    X : Initial guess for a solution. Usually a matrix of random numbers.\n        This function allows more than one column in X to identify more than\n        one eigenvector if desired.\n\n    normalized : bool\n        Whether the normalized Laplacian matrix is used.\n\n    tol : float\n        Tolerance of relative residual in eigenvalue computation.\n        Warning: There is no limit on number of iterations.\n\n    method : string\n        Should be 'tracemin_pcg' or 'tracemin_lu'.\n        Otherwise exception is raised.\n\n    Returns\n    -------\n    sigma, X : Two NumPy arrays of floats.\n        The lowest eigenvalues and corresponding eigenvectors of L.\n        The size of input X determines the size of these outputs.\n        As this is for Fiedler vectors, the zero eigenvalue (and\n        constant eigenvector) are avoided.\n    "
    import numpy as np
    import scipy as sp
    n = X.shape[0]
    if normalized:
        e = np.sqrt(L.diagonal())
        D = sp.sparse.csr_array(sp.sparse.spdiags(1 / e, 0, n, n, format='csr'))
        L = D @ L @ D
        e *= 1.0 / np.linalg.norm(e, 2)
    if normalized:

        def project(X):
            if False:
                return 10
            'Make X orthogonal to the nullspace of L.'
            X = np.asarray(X)
            for j in range(X.shape[1]):
                X[:, j] -= X[:, j] @ e * e
    else:

        def project(X):
            if False:
                i = 10
                return i + 15
            'Make X orthogonal to the nullspace of L.'
            X = np.asarray(X)
            for j in range(X.shape[1]):
                X[:, j] -= X[:, j].sum() / n
    if method == 'tracemin_pcg':
        D = L.diagonal().astype(float)
        solver = _PCGSolver(lambda x: L @ x, lambda x: D * x)
    elif method == 'tracemin_lu':
        A = sp.sparse.csc_array(L, dtype=float, copy=True)
        i = (A.indptr[1:] - A.indptr[:-1]).argmax()
        A[i, i] = float('inf')
        solver = _LUSolver(A)
    else:
        raise nx.NetworkXError(f'Unknown linear system solver: {method}')
    Lnorm = abs(L).sum(axis=1).flatten().max()
    project(X)
    W = np.ndarray(X.shape, order='F')
    while True:
        X = np.linalg.qr(X)[0]
        W[:, :] = L @ X
        H = X.T @ W
        (sigma, Y) = sp.linalg.eigh(H, overwrite_a=True)
        X = X @ Y
        res = sp.linalg.blas.dasum(W @ Y[:, 0] - sigma[0] * X[:, 0]) / Lnorm
        if res < tol:
            break
        W[:, :] = solver.solve(X, tol)
        X = (sp.linalg.inv(W.T @ X) @ W.T).T
        project(X)
    return (sigma, np.asarray(X))

def _get_fiedler_func(method):
    if False:
        for i in range(10):
            print('nop')
    'Returns a function that solves the Fiedler eigenvalue problem.'
    import numpy as np
    if method == 'tracemin':
        method = 'tracemin_pcg'
    if method in ('tracemin_pcg', 'tracemin_lu'):

        def find_fiedler(L, x, normalized, tol, seed):
            if False:
                for i in range(10):
                    print('nop')
            q = 1 if method == 'tracemin_pcg' else min(4, L.shape[0] - 1)
            X = np.asarray(seed.normal(size=(q, L.shape[0]))).T
            (sigma, X) = _tracemin_fiedler(L, X, normalized, tol, method)
            return (sigma[0], X[:, 0])
    elif method == 'lanczos' or method == 'lobpcg':

        def find_fiedler(L, x, normalized, tol, seed):
            if False:
                return 10
            import scipy as sp
            L = sp.sparse.csc_array(L, dtype=float)
            n = L.shape[0]
            if normalized:
                D = sp.sparse.csc_array(sp.sparse.spdiags(1.0 / np.sqrt(L.diagonal()), [0], n, n, format='csc'))
                L = D @ L @ D
            if method == 'lanczos' or n < 10:
                (sigma, X) = sp.sparse.linalg.eigsh(L, 2, which='SM', tol=tol, return_eigenvectors=True)
                return (sigma[1], X[:, 1])
            else:
                X = np.asarray(np.atleast_2d(x).T)
                M = sp.sparse.csr_array(sp.sparse.spdiags(1.0 / L.diagonal(), 0, n, n))
                Y = np.ones(n)
                if normalized:
                    Y /= D.diagonal()
                (sigma, X) = sp.sparse.linalg.lobpcg(L, X, M=M, Y=np.atleast_2d(Y).T, tol=tol, maxiter=n, largest=False)
                return (sigma[0], X[:, 0])
    else:
        raise nx.NetworkXError(f'unknown method {method!r}.')
    return find_fiedler

@not_implemented_for('directed')
@np_random_state(5)
@nx._dispatch(edge_attrs='weight')
def algebraic_connectivity(G, weight='weight', normalized=False, tol=1e-08, method='tracemin_pcg', seed=None):
    if False:
        for i in range(10):
            print('nop')
    "Returns the algebraic connectivity of an undirected graph.\n\n    The algebraic connectivity of a connected undirected graph is the second\n    smallest eigenvalue of its Laplacian matrix.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        An undirected graph.\n\n    weight : object, optional (default: None)\n        The data key used to determine the weight of each edge. If None, then\n        each edge has unit weight.\n\n    normalized : bool, optional (default: False)\n        Whether the normalized Laplacian matrix is used.\n\n    tol : float, optional (default: 1e-8)\n        Tolerance of relative residual in eigenvalue computation.\n\n    method : string, optional (default: 'tracemin_pcg')\n        Method of eigenvalue computation. It must be one of the tracemin\n        options shown below (TraceMIN), 'lanczos' (Lanczos iteration)\n        or 'lobpcg' (LOBPCG).\n\n        The TraceMIN algorithm uses a linear system solver. The following\n        values allow specifying the solver to be used.\n\n        =============== ========================================\n        Value           Solver\n        =============== ========================================\n        'tracemin_pcg'  Preconditioned conjugate gradient method\n        'tracemin_lu'   LU factorization\n        =============== ========================================\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    algebraic_connectivity : float\n        Algebraic connectivity.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If G is directed.\n\n    NetworkXError\n        If G has less than two nodes.\n\n    Notes\n    -----\n    Edge weights are interpreted by their absolute values. For MultiGraph's,\n    weights of parallel edges are summed. Zero-weighted edges are ignored.\n\n    See Also\n    --------\n    laplacian_matrix\n\n    Examples\n    --------\n    For undirected graphs algebraic connectivity can tell us if a graph is connected or not\n    `G` is connected iff  ``algebraic_connectivity(G) > 0``:\n\n    >>> G = nx.complete_graph(5)\n    >>> nx.algebraic_connectivity(G) > 0\n    True\n    >>> G.add_node(10)  # G is no longer connected\n    >>> nx.algebraic_connectivity(G) > 0\n    False\n\n    "
    if len(G) < 2:
        raise nx.NetworkXError('graph has less than two nodes.')
    G = _preprocess_graph(G, weight)
    if not nx.is_connected(G):
        return 0.0
    L = nx.laplacian_matrix(G)
    if L.shape[0] == 2:
        return 2.0 * L[0, 0] if not normalized else 2.0
    find_fiedler = _get_fiedler_func(method)
    x = None if method != 'lobpcg' else _rcm_estimate(G, G)
    (sigma, fiedler) = find_fiedler(L, x, normalized, tol, seed)
    return sigma

@not_implemented_for('directed')
@np_random_state(5)
@nx._dispatch(edge_attrs='weight')
def fiedler_vector(G, weight='weight', normalized=False, tol=1e-08, method='tracemin_pcg', seed=None):
    if False:
        while True:
            i = 10
    "Returns the Fiedler vector of a connected undirected graph.\n\n    The Fiedler vector of a connected undirected graph is the eigenvector\n    corresponding to the second smallest eigenvalue of the Laplacian matrix\n    of the graph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        An undirected graph.\n\n    weight : object, optional (default: None)\n        The data key used to determine the weight of each edge. If None, then\n        each edge has unit weight.\n\n    normalized : bool, optional (default: False)\n        Whether the normalized Laplacian matrix is used.\n\n    tol : float, optional (default: 1e-8)\n        Tolerance of relative residual in eigenvalue computation.\n\n    method : string, optional (default: 'tracemin_pcg')\n        Method of eigenvalue computation. It must be one of the tracemin\n        options shown below (TraceMIN), 'lanczos' (Lanczos iteration)\n        or 'lobpcg' (LOBPCG).\n\n        The TraceMIN algorithm uses a linear system solver. The following\n        values allow specifying the solver to be used.\n\n        =============== ========================================\n        Value           Solver\n        =============== ========================================\n        'tracemin_pcg'  Preconditioned conjugate gradient method\n        'tracemin_lu'   LU factorization\n        =============== ========================================\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    fiedler_vector : NumPy array of floats.\n        Fiedler vector.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If G is directed.\n\n    NetworkXError\n        If G has less than two nodes or is not connected.\n\n    Notes\n    -----\n    Edge weights are interpreted by their absolute values. For MultiGraph's,\n    weights of parallel edges are summed. Zero-weighted edges are ignored.\n\n    See Also\n    --------\n    laplacian_matrix\n\n    Examples\n    --------\n    Given a connected graph the signs of the values in the Fiedler vector can be\n    used to partition the graph into two components.\n\n    >>> G = nx.barbell_graph(5, 0)\n    >>> nx.fiedler_vector(G, normalized=True, seed=1)\n    array([-0.32864129, -0.32864129, -0.32864129, -0.32864129, -0.26072899,\n            0.26072899,  0.32864129,  0.32864129,  0.32864129,  0.32864129])\n\n    The connected components are the two 5-node cliques of the barbell graph.\n    "
    import numpy as np
    if len(G) < 2:
        raise nx.NetworkXError('graph has less than two nodes.')
    G = _preprocess_graph(G, weight)
    if not nx.is_connected(G):
        raise nx.NetworkXError('graph is not connected.')
    if len(G) == 2:
        return np.array([1.0, -1.0])
    find_fiedler = _get_fiedler_func(method)
    L = nx.laplacian_matrix(G)
    x = None if method != 'lobpcg' else _rcm_estimate(G, G)
    (sigma, fiedler) = find_fiedler(L, x, normalized, tol, seed)
    return fiedler

@np_random_state(5)
@nx._dispatch(edge_attrs='weight')
def spectral_ordering(G, weight='weight', normalized=False, tol=1e-08, method='tracemin_pcg', seed=None):
    if False:
        return 10
    "Compute the spectral_ordering of a graph.\n\n    The spectral ordering of a graph is an ordering of its nodes where nodes\n    in the same weakly connected components appear contiguous and ordered by\n    their corresponding elements in the Fiedler vector of the component.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        A graph.\n\n    weight : object, optional (default: None)\n        The data key used to determine the weight of each edge. If None, then\n        each edge has unit weight.\n\n    normalized : bool, optional (default: False)\n        Whether the normalized Laplacian matrix is used.\n\n    tol : float, optional (default: 1e-8)\n        Tolerance of relative residual in eigenvalue computation.\n\n    method : string, optional (default: 'tracemin_pcg')\n        Method of eigenvalue computation. It must be one of the tracemin\n        options shown below (TraceMIN), 'lanczos' (Lanczos iteration)\n        or 'lobpcg' (LOBPCG).\n\n        The TraceMIN algorithm uses a linear system solver. The following\n        values allow specifying the solver to be used.\n\n        =============== ========================================\n        Value           Solver\n        =============== ========================================\n        'tracemin_pcg'  Preconditioned conjugate gradient method\n        'tracemin_lu'   LU factorization\n        =============== ========================================\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    spectral_ordering : NumPy array of floats.\n        Spectral ordering of nodes.\n\n    Raises\n    ------\n    NetworkXError\n        If G is empty.\n\n    Notes\n    -----\n    Edge weights are interpreted by their absolute values. For MultiGraph's,\n    weights of parallel edges are summed. Zero-weighted edges are ignored.\n\n    See Also\n    --------\n    laplacian_matrix\n    "
    if len(G) == 0:
        raise nx.NetworkXError('graph is empty.')
    G = _preprocess_graph(G, weight)
    find_fiedler = _get_fiedler_func(method)
    order = []
    for component in nx.connected_components(G):
        size = len(component)
        if size > 2:
            L = nx.laplacian_matrix(G, component)
            x = None if method != 'lobpcg' else _rcm_estimate(G, component)
            (sigma, fiedler) = find_fiedler(L, x, normalized, tol, seed)
            sort_info = zip(fiedler, range(size), component)
            order.extend((u for (x, c, u) in sorted(sort_info)))
        else:
            order.extend(component)
    return order

@nx._dispatch(edge_attrs='weight')
def spectral_bisection(G, weight='weight', normalized=False, tol=1e-08, method='tracemin_pcg', seed=None):
    if False:
        return 10
    "Bisect the graph using the Fiedler vector.\n\n    This method uses the Fiedler vector to bisect a graph.\n    The partition is defined by the nodes which are associated with\n    either positive or negative values in the vector.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n\n    weight : str, optional (default: weight)\n        The data key used to determine the weight of each edge. If None, then\n        each edge has unit weight.\n\n    normalized : bool, optional (default: False)\n        Whether the normalized Laplacian matrix is used.\n\n    tol : float, optional (default: 1e-8)\n        Tolerance of relative residual in eigenvalue computation.\n\n    method : string, optional (default: 'tracemin_pcg')\n        Method of eigenvalue computation. It must be one of the tracemin\n        options shown below (TraceMIN), 'lanczos' (Lanczos iteration)\n        or 'lobpcg' (LOBPCG).\n\n        The TraceMIN algorithm uses a linear system solver. The following\n        values allow specifying the solver to be used.\n\n        =============== ========================================\n        Value           Solver\n        =============== ========================================\n        'tracemin_pcg'  Preconditioned conjugate gradient method\n        'tracemin_lu'   LU factorization\n        =============== ========================================\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    bisection : tuple of sets\n        Sets with the bisection of nodes\n\n    Examples\n    --------\n    >>> G = nx.barbell_graph(3, 0)\n    >>> nx.spectral_bisection(G)\n    ({0, 1, 2}, {3, 4, 5})\n\n    References\n    ----------\n    .. [1] M. E. J Newman 'Networks: An Introduction', pages 364-370\n       Oxford University Press 2011.\n    "
    import numpy as np
    v = nx.fiedler_vector(G, weight, normalized, tol, method, seed)
    nodes = np.array(list(G))
    pos_vals = v >= 0
    return (set(nodes[~pos_vals]), set(nodes[pos_vals]))