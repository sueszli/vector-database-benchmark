from sympy.core import S, pi, oo, symbols, Function, Rational, Integer, Tuple, Symbol, EulerGamma, GoldenRatio, Catalan, Lambda, Mul, Pow, Mod, Eq, Ne, Le, Lt, Gt, Ge
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.functions import arg, atan2, bernoulli, beta, ceiling, chebyshevu, chebyshevt, conjugate, DiracDelta, exp, expint, factorial, floor, harmonic, Heaviside, im, laguerre, LambertW, log, Max, Min, Piecewise, polylog, re, RisingFactorial, sign, sinc, sqrt, zeta, binomial, legendre, dirichlet_eta, riemann_xi
from sympy.functions import sin, cos, tan, cot, sec, csc, asin, acos, acot, atan, asec, acsc, sinh, cosh, tanh, coth, csch, sech, asinh, acosh, atanh, acoth, asech, acsch
from sympy.testing.pytest import raises, XFAIL
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import eye, Matrix, MatrixSymbol, Identity, HadamardProduct, SparseMatrix, HadamardPower
from sympy.functions.special.bessel import jn, yn, besselj, bessely, besseli, besselk, hankel1, hankel2, airyai, airybi, airyaiprime, airybiprime
from sympy.functions.special.gamma_functions import gamma, lowergamma, uppergamma, loggamma, polygamma
from sympy.functions.special.error_functions import Chi, Ci, erf, erfc, erfi, erfcinv, erfinv, fresnelc, fresnels, li, Shi, Si, Li, erf2, Ei
from sympy.printing.octave import octave_code, octave_code as mcode
(x, y, z) = symbols('x,y,z')

def test_Integer():
    if False:
        while True:
            i = 10
    assert mcode(Integer(67)) == '67'
    assert mcode(Integer(-1)) == '-1'

def test_Rational():
    if False:
        while True:
            i = 10
    assert mcode(Rational(3, 7)) == '3/7'
    assert mcode(Rational(18, 9)) == '2'
    assert mcode(Rational(3, -7)) == '-3/7'
    assert mcode(Rational(-3, -7)) == '3/7'
    assert mcode(x + Rational(3, 7)) == 'x + 3/7'
    assert mcode(Rational(3, 7) * x) == '3*x/7'

def test_Relational():
    if False:
        for i in range(10):
            print('nop')
    assert mcode(Eq(x, y)) == 'x == y'
    assert mcode(Ne(x, y)) == 'x != y'
    assert mcode(Le(x, y)) == 'x <= y'
    assert mcode(Lt(x, y)) == 'x < y'
    assert mcode(Gt(x, y)) == 'x > y'
    assert mcode(Ge(x, y)) == 'x >= y'

def test_Function():
    if False:
        i = 10
        return i + 15
    assert mcode(sin(x) ** cos(x)) == 'sin(x).^cos(x)'
    assert mcode(sign(x)) == 'sign(x)'
    assert mcode(exp(x)) == 'exp(x)'
    assert mcode(log(x)) == 'log(x)'
    assert mcode(factorial(x)) == 'factorial(x)'
    assert mcode(floor(x)) == 'floor(x)'
    assert mcode(atan2(y, x)) == 'atan2(y, x)'
    assert mcode(beta(x, y)) == 'beta(x, y)'
    assert mcode(polylog(x, y)) == 'polylog(x, y)'
    assert mcode(harmonic(x)) == 'harmonic(x)'
    assert mcode(bernoulli(x)) == 'bernoulli(x)'
    assert mcode(bernoulli(x, y)) == 'bernoulli(x, y)'
    assert mcode(legendre(x, y)) == 'legendre(x, y)'

def test_Function_change_name():
    if False:
        for i in range(10):
            print('nop')
    assert mcode(abs(x)) == 'abs(x)'
    assert mcode(ceiling(x)) == 'ceil(x)'
    assert mcode(arg(x)) == 'angle(x)'
    assert mcode(im(x)) == 'imag(x)'
    assert mcode(re(x)) == 'real(x)'
    assert mcode(conjugate(x)) == 'conj(x)'
    assert mcode(chebyshevt(y, x)) == 'chebyshevT(y, x)'
    assert mcode(chebyshevu(y, x)) == 'chebyshevU(y, x)'
    assert mcode(laguerre(x, y)) == 'laguerreL(x, y)'
    assert mcode(Chi(x)) == 'coshint(x)'
    assert mcode(Shi(x)) == 'sinhint(x)'
    assert mcode(Ci(x)) == 'cosint(x)'
    assert mcode(Si(x)) == 'sinint(x)'
    assert mcode(li(x)) == 'logint(x)'
    assert mcode(loggamma(x)) == 'gammaln(x)'
    assert mcode(polygamma(x, y)) == 'psi(x, y)'
    assert mcode(RisingFactorial(x, y)) == 'pochhammer(x, y)'
    assert mcode(DiracDelta(x)) == 'dirac(x)'
    assert mcode(DiracDelta(x, 3)) == 'dirac(3, x)'
    assert mcode(Heaviside(x)) == 'heaviside(x, 1/2)'
    assert mcode(Heaviside(x, y)) == 'heaviside(x, y)'
    assert mcode(binomial(x, y)) == 'bincoeff(x, y)'
    assert mcode(Mod(x, y)) == 'mod(x, y)'

def test_minmax():
    if False:
        while True:
            i = 10
    assert mcode(Max(x, y) + Min(x, y)) == 'max(x, y) + min(x, y)'
    assert mcode(Max(x, y, z)) == 'max(x, max(y, z))'
    assert mcode(Min(x, y, z)) == 'min(x, min(y, z))'

def test_Pow():
    if False:
        while True:
            i = 10
    assert mcode(x ** 3) == 'x.^3'
    assert mcode(x ** y ** 3) == 'x.^(y.^3)'
    assert mcode(x ** Rational(2, 3)) == 'x.^(2/3)'
    g = implemented_function('g', Lambda(x, 2 * x))
    assert mcode(1 / (g(x) * 3.5) ** (x - y ** x) / (x ** 2 + y)) == '(3.5*2*x).^(-x + y.^x)./(x.^2 + y)'
    assert mcode(Mul(-2, x, Pow(Mul(y, y, evaluate=False), -1, evaluate=False), evaluate=False)) == '-2*x./(y.*y)'

def test_basic_ops():
    if False:
        for i in range(10):
            print('nop')
    assert mcode(x * y) == 'x.*y'
    assert mcode(x + y) == 'x + y'
    assert mcode(x - y) == 'x - y'
    assert mcode(-x) == '-x'

def test_1_over_x_and_sqrt():
    if False:
        print('Hello World!')
    assert mcode(1 / x) == '1./x'
    assert mcode(x ** (-1)) == mcode(x ** (-1.0)) == '1./x'
    assert mcode(1 / sqrt(x)) == '1./sqrt(x)'
    assert mcode(x ** (-S.Half)) == mcode(x ** (-0.5)) == '1./sqrt(x)'
    assert mcode(sqrt(x)) == 'sqrt(x)'
    assert mcode(x ** S.Half) == mcode(x ** 0.5) == 'sqrt(x)'
    assert mcode(1 / pi) == '1/pi'
    assert mcode(pi ** (-1)) == mcode(pi ** (-1.0)) == '1/pi'
    assert mcode(pi ** (-0.5)) == '1/sqrt(pi)'

def test_mix_number_mult_symbols():
    if False:
        for i in range(10):
            print('nop')
    assert mcode(3 * x) == '3*x'
    assert mcode(pi * x) == 'pi*x'
    assert mcode(3 / x) == '3./x'
    assert mcode(pi / x) == 'pi./x'
    assert mcode(x / 3) == 'x/3'
    assert mcode(x / pi) == 'x/pi'
    assert mcode(x * y) == 'x.*y'
    assert mcode(3 * x * y) == '3*x.*y'
    assert mcode(3 * pi * x * y) == '3*pi*x.*y'
    assert mcode(x / y) == 'x./y'
    assert mcode(3 * x / y) == '3*x./y'
    assert mcode(x * y / z) == 'x.*y./z'
    assert mcode(x / y * z) == 'x.*z./y'
    assert mcode(1 / x / y) == '1./(x.*y)'
    assert mcode(2 * pi * x / y / z) == '2*pi*x./(y.*z)'
    assert mcode(3 * pi / x) == '3*pi./x'
    assert mcode(S(3) / 5) == '3/5'
    assert mcode(S(3) / 5 * x) == '3*x/5'
    assert mcode(x / y / z) == 'x./(y.*z)'
    assert mcode((x + y) / z) == '(x + y)./z'
    assert mcode((x + y) / (z + x)) == '(x + y)./(x + z)'
    assert mcode((x + y) / EulerGamma) == '(x + y)/%s' % EulerGamma.evalf(17)
    assert mcode(x / 3 / pi) == 'x/(3*pi)'
    assert mcode(S(3) / 5 * x * y / pi) == '3*x.*y/(5*pi)'

def test_mix_number_pow_symbols():
    if False:
        return 10
    assert mcode(pi ** 3) == 'pi^3'
    assert mcode(x ** 2) == 'x.^2'
    assert mcode(x ** pi ** 3) == 'x.^(pi^3)'
    assert mcode(x ** y) == 'x.^y'
    assert mcode(x ** y ** z) == 'x.^(y.^z)'
    assert mcode((x ** y) ** z) == '(x.^y).^z'

def test_imag():
    if False:
        return 10
    I = S('I')
    assert mcode(I) == '1i'
    assert mcode(5 * I) == '5i'
    assert mcode(S(3) / 2 * I) == '3*1i/2'
    assert mcode(3 + 4 * I) == '3 + 4i'
    assert mcode(sqrt(3) * I) == 'sqrt(3)*1i'

def test_constants():
    if False:
        while True:
            i = 10
    assert mcode(pi) == 'pi'
    assert mcode(oo) == 'inf'
    assert mcode(-oo) == '-inf'
    assert mcode(S.NegativeInfinity) == '-inf'
    assert mcode(S.NaN) == 'NaN'
    assert mcode(S.Exp1) == 'exp(1)'
    assert mcode(exp(1)) == 'exp(1)'

def test_constants_other():
    if False:
        i = 10
        return i + 15
    assert mcode(2 * GoldenRatio) == '2*(1+sqrt(5))/2'
    assert mcode(2 * Catalan) == '2*%s' % Catalan.evalf(17)
    assert mcode(2 * EulerGamma) == '2*%s' % EulerGamma.evalf(17)

def test_boolean():
    if False:
        for i in range(10):
            print('nop')
    assert mcode(x & y) == 'x & y'
    assert mcode(x | y) == 'x | y'
    assert mcode(~x) == '~x'
    assert mcode(x & y & z) == 'x & y & z'
    assert mcode(x | y | z) == 'x | y | z'
    assert mcode(x & y | z) == 'z | x & y'
    assert mcode((x | y) & z) == 'z & (x | y)'

def test_KroneckerDelta():
    if False:
        i = 10
        return i + 15
    from sympy.functions import KroneckerDelta
    assert mcode(KroneckerDelta(x, y)) == 'double(x == y)'
    assert mcode(KroneckerDelta(x, y + 1)) == 'double(x == (y + 1))'
    assert mcode(KroneckerDelta(2 ** x, y)) == 'double((2.^x) == y)'

def test_Matrices():
    if False:
        for i in range(10):
            print('nop')
    assert mcode(Matrix(1, 1, [10])) == '10'
    A = Matrix([[1, sin(x / 2), abs(x)], [0, 1, pi], [0, exp(1), ceiling(x)]])
    expected = '[1 sin(x/2) abs(x); 0 1 pi; 0 exp(1) ceil(x)]'
    assert mcode(A) == expected
    assert mcode(A[:, 0]) == '[1; 0; 0]'
    assert mcode(A[0, :]) == '[1 sin(x/2) abs(x)]'
    assert mcode(Matrix(0, 0, [])) == '[]'
    assert mcode(Matrix(0, 3, [])) == 'zeros(0, 3)'
    assert mcode(Matrix([[x, x - y, -y]])) == '[x x - y -y]'

def test_vector_entries_hadamard():
    if False:
        return 10
    A = Matrix([[1, sin(2 / x), 3 * pi / x / 5]])
    assert mcode(A) == '[1 sin(2./x) 3*pi./(5*x)]'
    assert mcode(A.T) == '[1; sin(2./x); 3*pi./(5*x)]'

@XFAIL
def test_Matrices_entries_not_hadamard():
    if False:
        i = 10
        return i + 15
    A = Matrix([[1, sin(2 / x), 3 * pi / x / 5], [1, 2, x * y]])
    expected = '[1 sin(2/x) 3*pi/(5*x);\n1        2        x*y]'
    assert mcode(A) == expected

def test_MatrixSymbol():
    if False:
        print('Hello World!')
    n = Symbol('n', integer=True)
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, n)
    assert mcode(A * B) == 'A*B'
    assert mcode(B * A) == 'B*A'
    assert mcode(2 * A * B) == '2*A*B'
    assert mcode(B * 2 * A) == '2*B*A'
    assert mcode(A * (B + 3 * Identity(n))) == 'A*(3*eye(n) + B)'
    assert mcode(A ** x ** 2) == 'A^(x.^2)'
    assert mcode(A ** 3) == 'A^3'
    assert mcode(A ** S.Half) == 'A^(1/2)'

def test_MatrixSolve():
    if False:
        return 10
    n = Symbol('n', integer=True)
    A = MatrixSymbol('A', n, n)
    x = MatrixSymbol('x', n, 1)
    assert mcode(MatrixSolve(A, x)) == 'A \\ x'

def test_special_matrices():
    if False:
        return 10
    assert mcode(6 * Identity(3)) == '6*eye(3)'

def test_containers():
    if False:
        for i in range(10):
            print('nop')
    assert mcode([1, 2, 3, [4, 5, [6, 7]], 8, [9, 10], 11]) == '{1, 2, 3, {4, 5, {6, 7}}, 8, {9, 10}, 11}'
    assert mcode((1, 2, (3, 4))) == '{1, 2, {3, 4}}'
    assert mcode([1]) == '{1}'
    assert mcode((1,)) == '{1}'
    assert mcode(Tuple(*[1, 2, 3])) == '{1, 2, 3}'
    assert mcode((1, x * y, (3, x ** 2))) == '{1, x.*y, {3, x.^2}}'
    assert mcode((1, eye(3), Matrix(0, 0, []), [])) == '{1, [1 0 0; 0 1 0; 0 0 1], [], {}}'

def test_octave_noninline():
    if False:
        print('Hello World!')
    source = mcode((x + y) / Catalan, assign_to='me', inline=False)
    expected = 'Catalan = %s;\nme = (x + y)/Catalan;' % Catalan.evalf(17)
    assert source == expected

def test_octave_piecewise():
    if False:
        for i in range(10):
            print('nop')
    expr = Piecewise((x, x < 1), (x ** 2, True))
    assert mcode(expr) == '((x < 1).*(x) + (~(x < 1)).*(x.^2))'
    assert mcode(expr, assign_to='r') == 'r = ((x < 1).*(x) + (~(x < 1)).*(x.^2));'
    assert mcode(expr, assign_to='r', inline=False) == 'if (x < 1)\n  r = x;\nelse\n  r = x.^2;\nend'
    expr = Piecewise((x ** 2, x < 1), (x ** 3, x < 2), (x ** 4, x < 3), (x ** 5, True))
    expected = '((x < 1).*(x.^2) + (~(x < 1)).*( ...\n(x < 2).*(x.^3) + (~(x < 2)).*( ...\n(x < 3).*(x.^4) + (~(x < 3)).*(x.^5))))'
    assert mcode(expr) == expected
    assert mcode(expr, assign_to='r') == 'r = ' + expected + ';'
    assert mcode(expr, assign_to='r', inline=False) == 'if (x < 1)\n  r = x.^2;\nelseif (x < 2)\n  r = x.^3;\nelseif (x < 3)\n  r = x.^4;\nelse\n  r = x.^5;\nend'
    expr = Piecewise((x, x < 1), (x ** 2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda : mcode(expr))

def test_octave_piecewise_times_const():
    if False:
        return 10
    pw = Piecewise((x, x < 1), (x ** 2, True))
    assert mcode(2 * pw) == '2*((x < 1).*(x) + (~(x < 1)).*(x.^2))'
    assert mcode(pw / x) == '((x < 1).*(x) + (~(x < 1)).*(x.^2))./x'
    assert mcode(pw / (x * y)) == '((x < 1).*(x) + (~(x < 1)).*(x.^2))./(x.*y)'
    assert mcode(pw / 3) == '((x < 1).*(x) + (~(x < 1)).*(x.^2))/3'

def test_octave_matrix_assign_to():
    if False:
        i = 10
        return i + 15
    A = Matrix([[1, 2, 3]])
    assert mcode(A, assign_to='a') == 'a = [1 2 3];'
    A = Matrix([[1, 2], [3, 4]])
    assert mcode(A, assign_to='A') == 'A = [1 2; 3 4];'

def test_octave_matrix_assign_to_more():
    if False:
        print('Hello World!')
    A = Matrix([[1, 2, 3]])
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 2, 3)
    assert mcode(A, assign_to=B) == 'B = [1 2 3];'
    raises(ValueError, lambda : mcode(A, assign_to=x))
    raises(ValueError, lambda : mcode(A, assign_to=C))

def test_octave_matrix_1x1():
    if False:
        return 10
    A = Matrix([[3]])
    B = MatrixSymbol('B', 1, 1)
    C = MatrixSymbol('C', 1, 2)
    assert mcode(A, assign_to=B) == 'B = 3;'
    raises(ValueError, lambda : mcode(A, assign_to=C))

def test_octave_matrix_elements():
    if False:
        print('Hello World!')
    A = Matrix([[x, 2, x * y]])
    assert mcode(A[0, 0] ** 2 + A[0, 1] + A[0, 2]) == 'x.^2 + x.*y + 2'
    A = MatrixSymbol('AA', 1, 3)
    assert mcode(A) == 'AA'
    assert mcode(A[0, 0] ** 2 + sin(A[0, 1]) + A[0, 2]) == 'sin(AA(1, 2)) + AA(1, 1).^2 + AA(1, 3)'
    assert mcode(sum(A)) == 'AA(1, 1) + AA(1, 2) + AA(1, 3)'

def test_octave_boolean():
    if False:
        i = 10
        return i + 15
    assert mcode(True) == 'true'
    assert mcode(S.true) == 'true'
    assert mcode(False) == 'false'
    assert mcode(S.false) == 'false'

def test_octave_not_supported():
    if False:
        while True:
            i = 10
    assert mcode(S.ComplexInfinity) == '% Not supported in Octave:\n% ComplexInfinity\nzoo'
    f = Function('f')
    assert mcode(f(x).diff(x)) == '% Not supported in Octave:\n% Derivative\nDerivative(f(x), x)'

def test_octave_not_supported_not_on_whitelist():
    if False:
        return 10
    from sympy.functions.special.polynomials import assoc_laguerre
    assert mcode(assoc_laguerre(x, y, z)) == '% Not supported in Octave:\n% assoc_laguerre\nassoc_laguerre(x, y, z)'

def test_octave_expint():
    if False:
        for i in range(10):
            print('nop')
    assert mcode(expint(1, x)) == 'expint(x)'
    assert mcode(expint(2, x)) == '% Not supported in Octave:\n% expint\nexpint(2, x)'
    assert mcode(expint(y, x)) == '% Not supported in Octave:\n% expint\nexpint(y, x)'

def test_trick_indent_with_end_else_words():
    if False:
        for i in range(10):
            print('nop')
    t1 = S('endless')
    t2 = S('elsewhere')
    pw = Piecewise((t1, x < 0), (t2, x <= 1), (1, True))
    assert mcode(pw, inline=False) == 'if (x < 0)\n  endless\nelseif (x <= 1)\n  elsewhere\nelse\n  1\nend'

def test_hadamard():
    if False:
        for i in range(10):
            print('nop')
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 3, 3)
    v = MatrixSymbol('v', 3, 1)
    h = MatrixSymbol('h', 1, 3)
    C = HadamardProduct(A, B)
    n = Symbol('n')
    assert mcode(C) == 'A.*B'
    assert mcode(C * v) == '(A.*B)*v'
    assert mcode(h * C * v) == 'h*(A.*B)*v'
    assert mcode(C * A) == '(A.*B)*A'
    assert mcode(C * x * y) == '(x.*y)*(A.*B)'
    assert mcode(HadamardPower(A, n)) == 'A.**n'
    assert mcode(HadamardPower(A, 1 + n)) == 'A.**(n + 1)'
    assert mcode(HadamardPower(A * B.T, 1 + n)) == '(A*B.T).**(n + 1)'

def test_sparse():
    if False:
        while True:
            i = 10
    M = SparseMatrix(5, 6, {})
    M[2, 2] = 10
    M[1, 2] = 20
    M[1, 3] = 22
    M[0, 3] = 30
    M[3, 0] = x * y
    assert mcode(M) == 'sparse([4 2 3 1 2], [1 3 3 4 4], [x.*y 20 10 30 22], 5, 6)'

def test_sinc():
    if False:
        for i in range(10):
            print('nop')
    assert mcode(sinc(x)) == 'sinc(x/pi)'
    assert mcode(sinc(x + 3)) == 'sinc((x + 3)/pi)'
    assert mcode(sinc(pi * (x + 3))) == 'sinc(x + 3)'

def test_trigfun():
    if False:
        return 10
    for f in (sin, cos, tan, cot, sec, csc, asin, acos, acot, atan, asec, acsc, sinh, cosh, tanh, coth, csch, sech, asinh, acosh, atanh, acoth, asech, acsch):
        assert octave_code(f(x) == f.__name__ + '(x)')

def test_specfun():
    if False:
        return 10
    n = Symbol('n')
    for f in [besselj, bessely, besseli, besselk]:
        assert octave_code(f(n, x)) == f.__name__ + '(n, x)'
    for f in (erfc, erfi, erf, erfinv, erfcinv, fresnelc, fresnels, gamma):
        assert octave_code(f(x)) == f.__name__ + '(x)'
    assert octave_code(hankel1(n, x)) == 'besselh(n, 1, x)'
    assert octave_code(hankel2(n, x)) == 'besselh(n, 2, x)'
    assert octave_code(airyai(x)) == 'airy(0, x)'
    assert octave_code(airyaiprime(x)) == 'airy(1, x)'
    assert octave_code(airybi(x)) == 'airy(2, x)'
    assert octave_code(airybiprime(x)) == 'airy(3, x)'
    assert octave_code(uppergamma(n, x)) == "(gammainc(x, n, 'upper').*gamma(n))"
    assert octave_code(lowergamma(n, x)) == '(gammainc(x, n).*gamma(n))'
    assert octave_code(z ** lowergamma(n, x)) == 'z.^(gammainc(x, n).*gamma(n))'
    assert octave_code(jn(n, x)) == 'sqrt(2)*sqrt(pi)*sqrt(1./x).*besselj(n + 1/2, x)/2'
    assert octave_code(yn(n, x)) == 'sqrt(2)*sqrt(pi)*sqrt(1./x).*bessely(n + 1/2, x)/2'
    assert octave_code(LambertW(x)) == 'lambertw(x)'
    assert octave_code(LambertW(x, n)) == 'lambertw(n, x)'
    assert octave_code(Ei(x)) == '(logint(exp(x)))'
    assert octave_code(dirichlet_eta(x)) == '(((x == 1).*(log(2)) + (~(x == 1)).*((1 - 2.^(1 - x)).*zeta(x))))'
    assert octave_code(riemann_xi(x)) == '(pi.^(-x/2).*x.*(x - 1).*gamma(x/2).*zeta(x)/2)'

def test_MatrixElement_printing():
    if False:
        for i in range(10):
            print('nop')
    A = MatrixSymbol('A', 1, 3)
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 1, 3)
    assert mcode(A[0, 0]) == 'A(1, 1)'
    assert mcode(3 * A[0, 0]) == '3*A(1, 1)'
    F = C[0, 0].subs(C, A - B)
    assert mcode(F) == '(A - B)(1, 1)'

def test_zeta_printing_issue_14820():
    if False:
        print('Hello World!')
    assert octave_code(zeta(x)) == 'zeta(x)'
    assert octave_code(zeta(x, y)) == '% Not supported in Octave:\n% zeta\nzeta(x, y)'

def test_automatic_rewrite():
    if False:
        i = 10
        return i + 15
    assert octave_code(Li(x)) == '(logint(x) - logint(2))'
    assert octave_code(erf2(x, y)) == '(-erf(x) + erf(y))'