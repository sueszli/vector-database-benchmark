import numpy as np

def hyantes(xmin, ymin, xmax, ymax, step, range_, range_x, range_y, t):
    if False:
        for i in range(10):
            print('nop')
    (X, Y) = t.shape
    pt = np.zeros((X, Y))
    for i in range(X):
        for j in range(Y):
            for k in t:
                tmp = 6368.0 * np.arccos(np.cos(xmin + step * i) * np.cos(k[0]) * np.cos(ymin + step * j - k[1]) + np.sin(xmin + step * i) * np.sin(k[0]))
                if tmp < range_:
                    pt[i, j] += k[2] / (1 + tmp)
    return pt