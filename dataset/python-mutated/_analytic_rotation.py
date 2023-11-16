"""
This file contains analytic implementations of rotation methods.
"""
import numpy as np
import scipy as sp

def target_rotation(A, H, full_rank=False):
    if False:
        print('Hello World!')
    '\n    Analytically performs orthogonal rotations towards a target matrix,\n    i.e., we minimize:\n\n    .. math::\n        \\phi(L) =\\frac{1}{2}\\|AT-H\\|^2.\n\n    where :math:`T` is an orthogonal matrix. This problem is also known as\n    an orthogonal Procrustes problem.\n\n    Under the assumption that :math:`A^*H` has full rank, the analytical\n    solution :math:`T` is given by:\n\n    .. math::\n        T = (A^*HH^*A)^{-\\frac{1}{2}}A^*H,\n\n    see Green (1952). In other cases the solution is given by :math:`T = UV`,\n    where :math:`U` and :math:`V` result from the singular value decomposition\n    of :math:`A^*H`:\n\n    .. math::\n        A^*H = U\\Sigma V,\n\n    see Schonemann (1966).\n\n    Parameters\n    ----------\n    A : numpy matrix (default None)\n        non rotated factors\n    H : numpy matrix\n        target matrix\n    full_rank : bool (default FAlse)\n        if set to true full rank is assumed\n\n    Returns\n    -------\n    The matrix :math:`T`.\n\n    References\n    ----------\n    [1] Green (1952, Psychometrika) - The orthogonal approximation of an\n    oblique structure in factor analysis\n\n    [2] Schonemann (1966) - A generalized solution of the orthogonal\n    procrustes problem\n\n    [3] Gower, Dijksterhuis (2004) - Procrustes problems\n    '
    ATH = A.T.dot(H)
    if full_rank or np.linalg.matrix_rank(ATH) == A.shape[1]:
        T = sp.linalg.fractional_matrix_power(ATH.dot(ATH.T), -1 / 2).dot(ATH)
    else:
        (U, D, V) = np.linalg.svd(ATH, full_matrices=False)
        T = U.dot(V)
    return T

def procrustes(A, H):
    if False:
        i = 10
        return i + 15
    '\n    Analytically solves the following Procrustes problem:\n\n    .. math::\n        \\phi(L) =\\frac{1}{2}\\|AT-H\\|^2.\n\n    (With no further conditions on :math:`H`)\n\n    Under the assumption that :math:`A^*H` has full rank, the analytical\n    solution :math:`T` is given by:\n\n    .. math::\n        T = (A^*HH^*A)^{-\\frac{1}{2}}A^*H,\n\n    see Navarra, Simoncini (2010).\n\n    Parameters\n    ----------\n    A : numpy matrix\n        non rotated factors\n    H : numpy matrix\n        target matrix\n    full_rank : bool (default False)\n        if set to true full rank is assumed\n\n    Returns\n    -------\n    The matrix :math:`T`.\n\n    References\n    ----------\n    [1] Navarra, Simoncini (2010) - A guide to empirical orthogonal functions\n    for climate data analysis\n    '
    return np.linalg.inv(A.T.dot(A)).dot(A.T).dot(H)

def promax(A, k=2):
    if False:
        while True:
            i = 10
    '\n    Performs promax rotation of the matrix :math:`A`.\n\n    This method was not very clear to me from the literature, this\n    implementation is as I understand it should work.\n\n    Promax rotation is performed in the following steps:\n\n    * Determine varimax rotated patterns :math:`V`.\n\n    * Construct a rotation target matrix :math:`|V_{ij}|^k/V_{ij}`\n\n    * Perform procrustes rotation towards the target to obtain T\n\n    * Determine the patterns\n\n    First, varimax rotation a target matrix :math:`H` is determined with\n    orthogonal varimax rotation.\n    Then, oblique target rotation is performed towards the target.\n\n    Parameters\n    ----------\n    A : numpy matrix\n        non rotated factors\n    k : float\n        parameter, should be positive\n\n    References\n    ----------\n    [1] Browne (2001) - An overview of analytic rotation in exploratory\n    factor analysis\n\n    [2] Navarra, Simoncini (2010) - A guide to empirical orthogonal functions\n    for climate data analysis\n    '
    assert k > 0
    from ._wrappers import rotate_factors
    (V, T) = rotate_factors(A, 'varimax')
    H = np.abs(V) ** k / V
    S = procrustes(A, H)
    d = np.sqrt(np.diag(np.linalg.inv(S.T.dot(S))))
    D = np.diag(d)
    T = np.linalg.inv(S.dot(D)).T
    return (A.dot(T), T)