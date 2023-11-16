import numpy as np

def compact(vertices, indices, tolerance=0.001):
    if False:
        print('Hello World!')
    'Compact vertices and indices within given tolerance'
    n = len(vertices)
    V = np.zeros(n, dtype=[('pos', np.float32, 3)])
    V['pos'][:, 0] = vertices[:, 0]
    V['pos'][:, 1] = vertices[:, 1]
    V['pos'][:, 2] = vertices[:, 2]
    epsilon = 0.001
    decimals = int(np.log(epsilon) / np.log(1 / 10.0))
    V_ = np.zeros_like(V)
    X = V['pos'][:, 0].round(decimals=decimals)
    X[np.where(abs(X) < epsilon)] = 0
    V_['pos'][:, 0] = X
    Y = V['pos'][:, 1].round(decimals=decimals)
    Y[np.where(abs(Y) < epsilon)] = 0
    V_['pos'][:, 1] = Y
    Z = V['pos'][:, 2].round(decimals=decimals)
    Z[np.where(abs(Z) < epsilon)] = 0
    V_['pos'][:, 2] = Z
    (U, RI) = np.unique(V_, return_inverse=True)
    indices = indices.ravel()
    I_ = indices.copy().ravel()
    for i in range(len(indices)):
        I_[i] = RI[indices[i]]
    I_ = I_.reshape(len(indices) // 3, 3)
    return (U.view(np.float32).reshape(len(U), 3), I_, RI)

def normals(vertices, indices):
    if False:
        for i in range(10):
            print('nop')
    'Compute normals over a triangulated surface\n\n    Parameters\n    ----------\n    vertices : ndarray (n,3)\n        triangles vertices\n\n    indices : ndarray (p,3)\n        triangles indices\n    '
    (vertices, indices, mapping) = compact(vertices, indices)
    T = vertices[indices]
    N = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    L = np.sqrt(np.sum(N * N, axis=1))
    L[L == 0] = 1.0
    N /= L[:, np.newaxis]
    normals = np.zeros_like(vertices)
    normals[indices[:, 0]] += N
    normals[indices[:, 1]] += N
    normals[indices[:, 2]] += N
    L = np.sqrt(np.sum(normals * normals, axis=1))
    L[L == 0] = 1.0
    normals /= L[:, np.newaxis]
    return normals[mapping]