"""Precompute coefficients of several series expansions
of Wright's generalized Bessel function Phi(a, b, x).

See https://dlmf.nist.gov/10.46.E1 with rho=a, beta=b, z=x.
"""
from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, curve_fit
from time import time
try:
    import sympy
    from sympy import EulerGamma, Rational, S, Sum, factorial, gamma, gammasimp, pi, polygamma, symbols, zeta
    from sympy.polys.polyfuncs import horner
except ImportError:
    pass

def series_small_a():
    if False:
        i = 10
        return i + 15
    'Tylor series expansion of Phi(a, b, x) in a=0 up to order 5.\n    '
    order = 5
    (a, b, x, k) = symbols('a b x k')
    A = []
    X = []
    B = []
    expression = Sum(x ** k / factorial(k) / gamma(a * k + b), (k, 0, S.Infinity))
    expression = gamma(b) / sympy.exp(x) * expression
    for n in range(0, order + 1):
        term = expression.diff(a, n).subs(a, 0).simplify().doit()
        x_part = term.subs(polygamma(0, b), 1).replace(polygamma, lambda *args: 0)
        x_part *= (-1) ** n
        A.append(a ** n / factorial(n))
        X.append(horner(x_part))
        B.append(horner((term / x_part).simplify()))
    s = 'Tylor series expansion of Phi(a, b, x) in a=0 up to order 5.\n'
    s += 'Phi(a, b, x) = exp(x)/gamma(b) * sum(A[i] * X[i] * B[i], i=0..5)\n'
    for (name, c) in zip(['A', 'X', 'B'], [A, X, B]):
        for i in range(len(c)):
            s += f'\n{name}[{i}] = ' + str(c[i])
    return s

def dg_series(z, n):
    if False:
        for i in range(10):
            print('nop')
    'Symbolic expansion of digamma(z) in z=0 to order n.\n\n    See https://dlmf.nist.gov/5.7.E4 and with https://dlmf.nist.gov/5.5.E2\n    '
    k = symbols('k')
    return -1 / z - EulerGamma + sympy.summation((-1) ** k * zeta(k) * z ** (k - 1), (k, 2, n + 1))

def pg_series(k, z, n):
    if False:
        print('Hello World!')
    'Symbolic expansion of polygamma(k, z) in z=0 to order n.'
    return sympy.diff(dg_series(z, n + k), z, k)

def series_small_a_small_b():
    if False:
        print('Hello World!')
    'Tylor series expansion of Phi(a, b, x) in a=0 and b=0 up to order 5.\n\n    Be aware of cancellation of poles in b=0 of digamma(b)/Gamma(b) and\n    polygamma functions.\n\n    digamma(b)/Gamma(b) = -1 - 2*M_EG*b + O(b^2)\n    digamma(b)^2/Gamma(b) = 1/b + 3*M_EG + b*(-5/12*PI^2+7/2*M_EG^2) + O(b^2)\n    polygamma(1, b)/Gamma(b) = 1/b + M_EG + b*(1/12*PI^2 + 1/2*M_EG^2) + O(b^2)\n    and so on.\n    '
    order = 5
    (a, b, x, k) = symbols('a b x k')
    (M_PI, M_EG, M_Z3) = symbols('M_PI M_EG M_Z3')
    c_subs = {pi: M_PI, EulerGamma: M_EG, zeta(3): M_Z3}
    A = []
    X = []
    B = []
    C = []
    expression = gamma(b) / sympy.exp(x) * Sum(x ** k / factorial(k) / gamma(a * k + b), (k, 0, S.Infinity))
    for n in range(0, order + 1):
        term = expression.diff(a, n).subs(a, 0).simplify().doit()
        x_part = term.subs(polygamma(0, b), 1).replace(polygamma, lambda *args: 0)
        x_part *= (-1) ** n
        pg_part = term / x_part / gamma(b)
        if n >= 1:
            pg_part = pg_part.replace(polygamma, lambda k, x: pg_series(k, x, order + 1 + n))
            pg_part = pg_part.series(b, 0, n=order + 1 - n).removeO().subs(polygamma(2, 1), -2 * zeta(3)).simplify()
        A.append(a ** n / factorial(n))
        X.append(horner(x_part))
        B.append(pg_part)
    C = sympy.Poly(B[1].subs(c_subs), b).coeffs()
    C.reverse()
    for i in range(len(C)):
        C[i] = (C[i] * factorial(i)).simplify()
    s = 'Tylor series expansion of Phi(a, b, x) in a=0 and b=0 up to order 5.'
    s += '\nPhi(a, b, x) = exp(x) * sum(A[i] * X[i] * B[i], i=0..5)\n'
    s += 'B[0] = 1\n'
    s += 'B[i] = sum(C[k+i-1] * b**k/k!, k=0..)\n'
    s += '\nM_PI = pi'
    s += '\nM_EG = EulerGamma'
    s += '\nM_Z3 = zeta(3)'
    for (name, c) in zip(['A', 'X'], [A, X]):
        for i in range(len(c)):
            s += f'\n{name}[{i}] = '
            s += str(c[i])
    for i in range(len(C)):
        s += f'\n# C[{i}] = '
        s += str(C[i])
        s += f'\nC[{i}] = '
        s += str(C[i].subs({M_EG: EulerGamma, M_PI: pi, M_Z3: zeta(3)}).evalf(17))
    s += '\n\nTest if B[i] does have the assumed structure.'
    s += '\nC[i] are derived from B[1] alone.'
    s += '\nTest B[2] == C[1] + b*C[2] + b^2/2*C[3] + b^3/6*C[4] + ..'
    test = sum([b ** k / factorial(k) * C[k + 1] for k in range(order - 1)])
    test = (test - B[2].subs(c_subs)).simplify()
    s += f'\ntest successful = {test == S(0)}'
    s += '\nTest B[3] == C[2] + b*C[3] + b^2/2*C[4] + ..'
    test = sum([b ** k / factorial(k) * C[k + 2] for k in range(order - 2)])
    test = (test - B[3].subs(c_subs)).simplify()
    s += f'\ntest successful = {test == S(0)}'
    return s

def asymptotic_series():
    if False:
        for i in range(10):
            print('nop')
    'Asymptotic expansion for large x.\n\n    Phi(a, b, x) ~ Z^(1/2-b) * exp((1+a)/a * Z) * sum_k (-1)^k * C_k / Z^k\n    Z = (a*x)^(1/(1+a))\n\n    Wright (1935) lists the coefficients C_0 and C_1 (he calls them a_0 and\n    a_1). With slightly different notation, Paris (2017) lists coefficients\n    c_k up to order k=3.\n    Paris (2017) uses ZP = (1+a)/a * Z  (ZP = Z of Paris) and\n    C_k = C_0 * (-a/(1+a))^k * c_k\n    '
    order = 8

    class g(sympy.Function):
        """Helper function g according to Wright (1935)

        g(n, rho, v) = (1 + (rho+2)/3 * v + (rho+2)*(rho+3)/(2*3) * v^2 + ...)

        Note: Wright (1935) uses square root of above definition.
        """
        nargs = 3

        @classmethod
        def eval(cls, n, rho, v):
            if False:
                print('Hello World!')
            if not n >= 0:
                raise ValueError('must have n >= 0')
            elif n == 0:
                return 1
            else:
                return g(n - 1, rho, v) + gammasimp(gamma(rho + 2 + n) / gamma(rho + 2)) / gammasimp(gamma(3 + n) / gamma(3)) * v ** n

    class coef_C(sympy.Function):
        """Calculate coefficients C_m for integer m.

        C_m is the coefficient of v^(2*m) in the Taylor expansion in v=0 of
        Gamma(m+1/2)/(2*pi) * (2/(rho+1))^(m+1/2) * (1-v)^(-b)
            * g(rho, v)^(-m-1/2)
        """
        nargs = 3

        @classmethod
        def eval(cls, m, rho, beta):
            if False:
                i = 10
                return i + 15
            if not m >= 0:
                raise ValueError('must have m >= 0')
            v = symbols('v')
            expression = (1 - v) ** (-beta) * g(2 * m, rho, v) ** (-m - Rational(1, 2))
            res = expression.diff(v, 2 * m).subs(v, 0) / factorial(2 * m)
            res = res * (gamma(m + Rational(1, 2)) / (2 * pi) * (2 / (rho + 1)) ** (m + Rational(1, 2)))
            return res
    (xa, b, xap1) = symbols('xa b xap1')
    C0 = coef_C(0, xa, b)
    s = 'Asymptotic expansion for large x\n'
    s += 'Phi(a, b, x) = Z**(1/2-b) * exp((1+a)/a * Z) \n'
    s += '               * sum((-1)**k * C[k]/Z**k, k=0..6)\n\n'
    s += 'Z      = pow(a * x, 1/(1+a))\n'
    s += 'A[k]   = pow(a, k)\n'
    s += 'B[k]   = pow(b, k)\n'
    s += 'Ap1[k] = pow(1+a, k)\n\n'
    s += 'C[0] = 1./sqrt(2. * M_PI * Ap1[1])\n'
    for i in range(1, order + 1):
        expr = (coef_C(i, xa, b) / (C0 / (1 + xa) ** i)).simplify()
        factor = [x.denominator() for x in sympy.Poly(expr).coeffs()]
        factor = sympy.lcm(factor)
        expr = (expr * factor).simplify().collect(b, sympy.factor)
        expr = expr.xreplace({xa + 1: xap1})
        s += f'C[{i}] = C[0] / ({factor} * Ap1[{i}])\n'
        s += f'C[{i}] *= {str(expr)}\n\n'
    import re
    re_a = re.compile('xa\\*\\*(\\d+)')
    s = re_a.sub('A[\\1]', s)
    re_b = re.compile('b\\*\\*(\\d+)')
    s = re_b.sub('B[\\1]', s)
    s = s.replace('xap1', 'Ap1[1]')
    s = s.replace('xa', 'a')
    re_digits = re.compile('(\\d{10,})')
    s = re_digits.sub('\\1.', s)
    return s

def optimal_epsilon_integral():
    if False:
        print('Hello World!')
    'Fit optimal choice of epsilon for integral representation.\n\n    The integrand of\n        int_0^pi P(eps, a, b, x, phi) * dphi\n    can exhibit oscillatory behaviour. It stems from the cosine of P and can be\n    minimized by minimizing the arc length of the argument\n        f(phi) = eps * sin(phi) - x * eps^(-a) * sin(a * phi) + (1 - b) * phi\n    of cos(f(phi)).\n    We minimize the arc length in eps for a grid of values (a, b, x) and fit a\n    parametric function to it.\n    '

    def fp(eps, a, b, x, phi):
        if False:
            return 10
        'Derivative of f w.r.t. phi.'
        eps_a = np.power(1.0 * eps, -a)
        return eps * np.cos(phi) - a * x * eps_a * np.cos(a * phi) + 1 - b

    def arclength(eps, a, b, x, epsrel=0.01, limit=100):
        if False:
            print('Hello World!')
        "Compute Arc length of f.\n\n        Note that the arc length of a function f from t0 to t1 is given by\n            int_t0^t1 sqrt(1 + f'(t)^2) dt\n        "
        return quad(lambda phi: np.sqrt(1 + fp(eps, a, b, x, phi) ** 2), 0, np.pi, epsrel=epsrel, limit=100)[0]
    data_a = [0.001, 0.1, 0.5, 0.9, 1, 2, 4, 5, 6, 8]
    data_b = [0, 1, 4, 7, 10]
    data_x = [1, 1.5, 2, 4, 10, 20, 50, 100, 200, 500, 1000.0, 5000.0, 10000.0]
    (data_a, data_b, data_x) = np.meshgrid(data_a, data_b, data_x)
    (data_a, data_b, data_x) = (data_a.flatten(), data_b.flatten(), data_x.flatten())
    best_eps = []
    for i in range(data_x.size):
        best_eps.append(minimize_scalar(lambda eps: arclength(eps, data_a[i], data_b[i], data_x[i]), bounds=(0.001, 1000), method='Bounded', options={'xatol': 0.001}).x)
    best_eps = np.array(best_eps)
    df = {'a': data_a, 'b': data_b, 'x': data_x, 'eps': best_eps}

    def func(data, A0, A1, A2, A3, A4, A5):
        if False:
            i = 10
            return i + 15
        'Compute parametric function to fit.'
        a = data['a']
        b = data['b']
        x = data['x']
        return A0 * b * np.exp(-0.5 * a) + np.exp(A1 + 1 / (1 + a) * np.log(x) - A2 * np.exp(-A3 * a) + A4 / (1 + np.exp(A5 * a)))
    func_params = list(curve_fit(func, df, df['eps'], method='trf')[0])
    s = 'Fit optimal eps for integrand P via minimal arc length\n'
    s += 'with parametric function:\n'
    s += 'optimal_eps = (A0 * b * exp(-a/2) + exp(A1 + 1 / (1 + a) * log(x)\n'
    s += '              - A2 * exp(-A3 * a) + A4 / (1 + exp(A5 * a)))\n\n'
    s += 'Fitted parameters A0 to A5 are:\n'
    s += ', '.join([f'{x:.5g}' for x in func_params])
    return s

def main():
    if False:
        while True:
            i = 10
    t0 = time()
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument('action', type=int, choices=[1, 2, 3, 4], help='chose what expansion to precompute\n1 : Series for small a\n2 : Series for small a and small b\n3 : Asymptotic series for large x\n    This may take some time (>4h).\n4 : Fit optimal eps for integral representation.')
    args = parser.parse_args()
    switch = {1: lambda : print(series_small_a()), 2: lambda : print(series_small_a_small_b()), 3: lambda : print(asymptotic_series()), 4: lambda : print(optimal_epsilon_integral())}
    switch.get(args.action, lambda : print('Invalid input.'))()
    print(f'\n{(time() - t0) / 60:.1f} minutes elapsed.\n')
if __name__ == '__main__':
    main()