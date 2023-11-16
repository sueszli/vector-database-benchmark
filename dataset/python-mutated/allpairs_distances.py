import numpy as np

def allpairs_distances(A, B):
    if False:
        i = 10
        return i + 15
    'This returns the euclidean distances squared\n    dist2(x, y) = dot(x, x) - 2 * dot(x, y) + dot(y, y)\n    '
    A2 = np.einsum('ij,ij->i', A, A)
    B2 = np.einsum('ij,ij->i', B, B)
    return A2[:, None] + B2[None, :] - 2 * np.dot(A, B.T)