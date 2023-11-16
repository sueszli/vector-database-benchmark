def harris(X):
    if False:
        print('Hello World!')
    (m, n) = X.shape
    dx = (X[1:, :] - X[:m - 1, :])[:, 1:]
    dy = (X[:, 1:] - X[:, :n - 1])[1:, :]
    A = dx * dx
    B = dy * dy
    C = dx * dy
    tr = A + B
    det = A * B - C * C
    k = 0.05
    return det - k * tr * tr