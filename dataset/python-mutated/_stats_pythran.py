import numpy as np

def _Aij(A, i, j):
    if False:
        for i in range(10):
            print('nop')
    'Sum of upper-left and lower right blocks of contingency table.'
    return A[:i, :j].sum() + A[i + 1:, j + 1:].sum()

def _Dij(A, i, j):
    if False:
        i = 10
        return i + 15
    'Sum of lower-left and upper-right blocks of contingency table.'
    return A[i + 1:, :j].sum() + A[:i, j + 1:].sum()

def _concordant_pairs(A):
    if False:
        for i in range(10):
            print('nop')
    'Twice the number of concordant pairs, excluding ties.'
    (m, n) = A.shape
    count = 0
    for i in range(m):
        for j in range(n):
            count += A[i, j] * _Aij(A, i, j)
    return count

def _discordant_pairs(A):
    if False:
        i = 10
        return i + 15
    'Twice the number of discordant pairs, excluding ties.'
    (m, n) = A.shape
    count = 0
    for i in range(m):
        for j in range(n):
            count += A[i, j] * _Dij(A, i, j)
    return count

def _a_ij_Aij_Dij2(A):
    if False:
        i = 10
        return i + 15
    "A term that appears in the ASE of Kendall's tau and Somers' D."
    (m, n) = A.shape
    count = 0
    for i in range(m):
        for j in range(n):
            count += A[i, j] * (_Aij(A, i, j) - _Dij(A, i, j)) ** 2
    return count

def _compute_outer_prob_inside_method(m, n, g, h):
    if False:
        i = 10
        return i + 15
    '\n    Count the proportion of paths that do not stay strictly inside two\n    diagonal lines.\n\n    Parameters\n    ----------\n    m : integer\n        m > 0\n    n : integer\n        n > 0\n    g : integer\n        g is greatest common divisor of m and n\n    h : integer\n        0 <= h <= lcm(m,n)\n\n    Returns\n    -------\n    p : float\n        The proportion of paths that do not stay inside the two lines.\n\n    The classical algorithm counts the integer lattice paths from (0, 0)\n    to (m, n) which satisfy |x/m - y/n| < h / lcm(m, n).\n    The paths make steps of size +1 in either positive x or positive y\n    directions.\n    We are, however, interested in 1 - proportion to computes p-values,\n    so we change the recursion to compute 1 - p directly while staying\n    within the "inside method" a described by Hodges.\n\n    We generally follow Hodges\' treatment of Drion/Gnedenko/Korolyuk.\n    Hodges, J.L. Jr.,\n    "The Significance Probability of the Smirnov Two-Sample Test,"\n    Arkiv fiur Matematik, 3, No. 43 (1958), 469-86.\n\n    For the recursion for 1-p see\n    Viehmann, T.: "Numerically more stable computation of the p-values\n    for the two-sample Kolmogorov-Smirnov test," arXiv: 2102.08037\n\n    '
    if m < n:
        (m, n) = (n, m)
    mg = m // g
    ng = n // g
    (minj, maxj) = (0, min(int(np.ceil(h / mg)), n + 1))
    curlen = maxj - minj
    lenA = min(2 * maxj + 2, n + 1)
    dtype = np.float64
    A = np.ones(lenA, dtype=dtype)
    A[minj:maxj] = 0.0
    for i in range(1, m + 1):
        (lastminj, lastlen) = (minj, curlen)
        minj = max(int(np.floor((ng * i - h) / mg)) + 1, 0)
        minj = min(minj, n)
        maxj = min(int(np.ceil((ng * i + h) / mg)), n + 1)
        if maxj <= minj:
            return 1.0
        val = 0.0 if minj == 0 else 1.0
        for jj in range(maxj - minj):
            j = jj + minj
            val = (A[jj + minj - lastminj] * i + val * j) / (i + j)
            A[jj] = val
        curlen = maxj - minj
        if lastlen > curlen:
            A[maxj - minj:maxj - minj + (lastlen - curlen)] = 1
    return A[maxj - minj - 1]

def siegelslopes(y, x, method):
    if False:
        for i in range(10):
            print('nop')
    deltax = np.expand_dims(x, 1) - x
    deltay = np.expand_dims(y, 1) - y
    (slopes, intercepts) = ([], [])
    for j in range(len(x)):
        (id_nonzero,) = np.nonzero(deltax[j, :])
        slopes_j = deltay[j, id_nonzero] / deltax[j, id_nonzero]
        medslope_j = np.median(slopes_j)
        slopes.append(medslope_j)
        if method == 'separate':
            z = y * x[j] - y[j] * x
            medintercept_j = np.median(z[id_nonzero] / deltax[j, id_nonzero])
            intercepts.append(medintercept_j)
    medslope = np.median(np.asarray(slopes))
    if method == 'separate':
        medinter = np.median(np.asarray(intercepts))
    else:
        medinter = np.median(y - medslope * x)
    return (medslope, medinter)