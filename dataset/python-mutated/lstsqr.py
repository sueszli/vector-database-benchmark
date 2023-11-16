import numpy as np

def lstsqr(x, y):
    if False:
        while True:
            i = 10
    'Computes the least-squares solution to a linear matrix equation.'
    x_avg = np.average(x)
    y_avg = np.average(y)
    dx = x - x_avg
    var_x = np.sum(dx ** 2)
    cov_xy = np.sum(dx * (y - y_avg))
    slope = cov_xy / var_x
    y_interc = y_avg - slope * x_avg
    return (slope, y_interc)