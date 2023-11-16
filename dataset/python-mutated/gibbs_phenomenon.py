"""
This example illustrates the Gibbs phenomenon.

It also calculates the Wilbraham-Gibbs constant by two approaches:

1) calculating the fourier series of the step function and determining the
first maximum.
2) evaluating the integral for si(pi).

See:
 * https://en.wikipedia.org/wiki/Gibbs_phenomena
"""
from sympy import var, sqrt, integrate, conjugate, seterr, Abs, pprint, I, pi, sin, cos, sign, lambdify, Integral, S
x = var('x', real=True)

def l2_norm(f, lim):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculates L2 norm of the function "f", over the domain lim=(x, a, b).\n\n    x ...... the independent variable in f over which to integrate\n    a, b ... the limits of the interval\n\n    Examples\n    ========\n\n    >>> from sympy import Symbol\n    >>> from gibbs_phenomenon import l2_norm\n    >>> x = Symbol(\'x\', real=True)\n    >>> l2_norm(1, (x, -1, 1))\n    sqrt(2)\n    >>> l2_norm(x, (x, -1, 1))\n    sqrt(6)/3\n\n    '
    return sqrt(integrate(Abs(f) ** 2, lim))

def l2_inner_product(a, b, lim):
    if False:
        return 10
    '\n    Calculates the L2 inner product (a, b) over the domain lim.\n    '
    return integrate(conjugate(a) * b, lim)

def l2_projection(f, basis, lim):
    if False:
        for i in range(10):
            print('nop')
    '\n    L2 projects the function f on the basis over the domain lim.\n    '
    r = 0
    for b in basis:
        r += l2_inner_product(f, b, lim) * b
    return r

def l2_gram_schmidt(list, lim):
    if False:
        return 10
    '\n    Orthonormalizes the "list" of functions using the Gram-Schmidt process.\n\n    Examples\n    ========\n\n    >>> from sympy import Symbol\n    >>> from gibbs_phenomenon import l2_gram_schmidt\n\n    >>> x = Symbol(\'x\', real=True)    # perform computations over reals to save time\n    >>> l2_gram_schmidt([1, x, x**2], (x, -1, 1))\n    [sqrt(2)/2, sqrt(6)*x/2, 3*sqrt(10)*(x**2 - 1/3)/4]\n\n    '
    r = []
    for a in list:
        if r == []:
            v = a
        else:
            v = a - l2_projection(a, r, lim)
        v_norm = l2_norm(v, lim)
        if v_norm == 0:
            raise ValueError('The sequence is not linearly independent.')
        r.append(v / v_norm)
    return r

def integ(f):
    if False:
        return 10
    return integrate(f, (x, -pi, 0)) + integrate(-f, (x, 0, pi))

def series(L):
    if False:
        for i in range(10):
            print('nop')
    '\n    Normalizes the series.\n    '
    r = 0
    for b in L:
        r += integ(b) * b
    return r

def msolve(f, x):
    if False:
        print('Hello World!')
    '\n    Finds the first root of f(x) to the left of 0.\n\n    The x0 and dx below are tailored to get the correct result for our\n    particular function --- the general solver often overshoots the first\n    solution.\n    '
    f = lambdify(x, f)
    x0 = -0.001
    dx = 0.001
    while f(x0 - dx) * f(x0) > 0:
        x0 = x0 - dx
    x_max = x0 - dx
    x_min = x0
    assert f(x_max) > 0
    assert f(x_min) < 0
    for n in range(100):
        x0 = (x_max + x_min) / 2
        if f(x0) > 0:
            x_max = x0
        else:
            x_min = x0
    return x0

def main():
    if False:
        for i in range(10):
            print('nop')
    L = [1]
    for i in range(1, 100):
        L.append(cos(i * x))
        L.append(sin(i * x))
    L[0] /= sqrt(2)
    L = [f / sqrt(pi) for f in L]
    f = series(L)
    print('Fourier series of the step function')
    pprint(f)
    x0 = msolve(f.diff(x), x)
    print('x-value of the maximum:', x0)
    max = f.subs(x, x0).evalf()
    print('y-value of the maximum:', max)
    g = max * pi / 2
    print('Wilbraham-Gibbs constant        :', g.evalf())
    print('Wilbraham-Gibbs constant (exact):', Integral(sin(x) / x, (x, 0, pi)).evalf())
if __name__ == '__main__':
    main()