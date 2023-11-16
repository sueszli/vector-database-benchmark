import numpy as np

def mandelbrot(X, Y, max_iterations=1000, verbose=True):
    if False:
        return 10
    'Computes the Mandelbrot set.\n\n    Returns a matrix with the escape iteration number of the mandelbrot\n    sequence. The matrix contains a cell for every (x, y) couple of the\n    X and Y vectors elements given in input. Maximum max_iterations are\n    performed for each point\n    :param X: set of x coordinates\n    :param Y: set of y coordinates\n    :param max_iterations: maximum number of iterations to perform before\n        forcing to stop the sequence\n    :param show_out: flag indicating whether to print on console which line\n        number is being computed\n    :return: Matrix containing the escape iteration number for every point\n        specified in input\n    '
    out_arr = np.zeros((len(Y), len(X)))
    for (i, y) in enumerate(Y):
        for (j, x) in enumerate(X):
            n = 0
            c = x + 1j * y
            z = c
            while n < max_iterations and abs(z) <= 2:
                z = z * z + c
                n += 1
            out_arr[i, j] = n
    return out_arr