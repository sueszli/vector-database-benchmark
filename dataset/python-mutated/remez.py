import math
import numpy as np
from matplotlib import pyplot as plt

def _get_chebyshev_nodes(n, a, b):
    if False:
        i = 10
        return i + 15
    nodes = [0.5 * (a + b) + 0.5 * (b - a) * math.cos((2 * k + 1) / (2.0 * n) * math.pi) for k in range(n)]
    return nodes

def _get_errors(exact_values, poly_coeff, nodes):
    if False:
        return 10
    ys = np.polyval(poly_coeff, nodes)
    for i in range(len(ys)):
        ys[i] = abs(ys[i] - exact_values[i])
    return ys
'\nReturn the coefficients of a polynomial of degree d approximating\nthe function fun on the interval (a,b).\n\nArgs:\n    fun: Function to approximate\n    a: Left interval border\n    b: Right interval border\n    d: The polynomial degree will be d, 2*d or 2*d + 1 depending\n        on the values of odd and even below\n    odd: If True, use odd polynomial of degree 2*d+1\n    even: If True, use even polynomial of degree 2*d\n    tol: Tolerance to use when checking for convergence\n\nReturns: Tuple where the first entry is the achieved absolute error\n    and the second entry is a list of the polynomial coefficients in\n    the order that is required by the QDK Numerics library. This is\n    the inverse order compared to what np.polyval expects.\n'

def run_remez(fun, a, b, d=5, odd=False, even=False, tol=1e-13):
    if False:
        while True:
            i = 10
    finished = False
    cn = _get_chebyshev_nodes(d + 2, a, b)
    cn2 = _get_chebyshev_nodes(100 * d, a, b)
    it = 0
    while not finished and len(cn) == d + 2 and (it < 50):
        it += 1
        b = np.array([fun(c) for c in cn])
        A = np.matrix(np.zeros([d + 2, d + 2]))
        for i in range(d + 2):
            x = 1.0
            if odd:
                x *= cn[i]
            for j in range(d + 2):
                A[i, j] = x
                x *= cn[i]
                if odd or even:
                    x *= cn[i]
            A[i, -1] = (-1) ** (i + 1)
        res = np.linalg.solve(A, b)
        revlist = reversed(res[0:-1])
        sc_coeff = []
        for c in revlist:
            sc_coeff.append(c)
            if odd or even:
                sc_coeff.append(0)
        if even:
            sc_coeff = sc_coeff[0:-1]
        errs = _get_errors([fun(c) for c in cn2], sc_coeff, cn2)
        maximum_indices = []
        if errs[0] > errs[1]:
            maximum_indices.append(0)
        for i in range(1, len(errs) - 1):
            if errs[i] > errs[i - 1] and errs[i] > errs[i + 1]:
                maximum_indices.append(i)
        if errs[-1] > errs[-2]:
            maximum_indices.append(-1)
        finished = True
        for idx in maximum_indices[1:]:
            if abs(errs[idx] - errs[maximum_indices[0]]) > tol:
                finished = False
        cn = [cn2[i] for i in maximum_indices]
    plt.plot(cn2, abs(errs))
    plt.title('Plot of the approximation error')
    plt.xlabel('x')
    plt.ylabel('|poly_fit(x) - f(x)|')
    plt.show()
    return (max(abs(errs)), list(reversed(res[0:-1])))
if __name__ == '__main__':

    def f(x):
        if False:
            print('Hello World!')
        return math.sin(x)
    odd = True
    even = False
    a = 0.0
    b = math.pi
    degree = 3
    (err, coeffs) = run_remez(f, a, b, degree, odd, even)
    oddEvenStr = ''
    if odd:
        oddEvenStr = ' for odd powers of x'
    if even:
        oddEvenStr = ' for even powers of x'
    print('Coefficients{}: {}'.format(oddEvenStr, list(reversed(coeffs))))
    print('The polynomial achieves an L_inf error of {}.'.format(err))