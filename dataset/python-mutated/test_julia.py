from sympy.core import S, pi, oo, symbols, Function, Rational, Integer, Tuple, Symbol, Eq, Ne, Le, Lt, Gt, Ge
from sympy.core import EulerGamma, GoldenRatio, Catalan, Lambda, Mul, Pow
from sympy.functions import Piecewise, sqrt, ceiling, exp, sin, cos
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import eye, Matrix, MatrixSymbol, Identity, HadamardProduct, SparseMatrix
from sympy.functions.special.bessel import jn, yn, besselj, bessely, besseli, besselk, hankel1, hankel2, airyai, airybi, airyaiprime, airybiprime
from sympy.testing.pytest import XFAIL
from sympy.printing.julia import julia_code
(x, y, z) = symbols('x,y,z')

def test_Integer():
    if False:
        i = 10
        return i + 15
    assert julia_code(Integer(67)) == '67'
    assert julia_code(Integer(-1)) == '-1'

def test_Rational():
    if False:
        for i in range(10):
            print('nop')
    assert julia_code(Rational(3, 7)) == '3 // 7'
    assert julia_code(Rational(18, 9)) == '2'
    assert julia_code(Rational(3, -7)) == '-3 // 7'
    assert julia_code(Rational(-3, -7)) == '3 // 7'
    assert julia_code(x + Rational(3, 7)) == 'x + 3 // 7'
    assert julia_code(Rational(3, 7) * x) == '(3 // 7) * x'

def test_Relational():
    if False:
        for i in range(10):
            print('nop')
    assert julia_code(Eq(x, y)) == 'x == y'
    assert julia_code(Ne(x, y)) == 'x != y'
    assert julia_code(Le(x, y)) == 'x <= y'
    assert julia_code(Lt(x, y)) == 'x < y'
    assert julia_code(Gt(x, y)) == 'x > y'
    assert julia_code(Ge(x, y)) == 'x >= y'

def test_Function():
    if False:
        while True:
            i = 10
    assert julia_code(sin(x) ** cos(x)) == 'sin(x) .^ cos(x)'
    assert julia_code(abs(x)) == 'abs(x)'
    assert julia_code(ceiling(x)) == 'ceil(x)'

def test_Pow():
    if False:
        while True:
            i = 10
    assert julia_code(x ** 3) == 'x .^ 3'
    assert julia_code(x ** y ** 3) == 'x .^ (y .^ 3)'
    assert julia_code(x ** Rational(2, 3)) == 'x .^ (2 // 3)'
    g = implemented_function('g', Lambda(x, 2 * x))
    assert julia_code(1 / (g(x) * 3.5) ** (x - y ** x) / (x ** 2 + y)) == '(3.5 * 2 * x) .^ (-x + y .^ x) ./ (x .^ 2 + y)'
    assert julia_code(Mul(-2, x, Pow(Mul(y, y, evaluate=False), -1, evaluate=False), evaluate=False)) == '-2 * x ./ (y .* y)'

def test_basic_ops():
    if False:
        while True:
            i = 10
    assert julia_code(x * y) == 'x .* y'
    assert julia_code(x + y) == 'x + y'
    assert julia_code(x - y) == 'x - y'
    assert julia_code(-x) == '-x'

def test_1_over_x_and_sqrt():
    if False:
        i = 10
        return i + 15
    assert julia_code(1 / x) == '1 ./ x'
    assert julia_code(x ** (-1)) == julia_code(x ** (-1.0)) == '1 ./ x'
    assert julia_code(1 / sqrt(x)) == '1 ./ sqrt(x)'
    assert julia_code(x ** (-S.Half)) == julia_code(x ** (-0.5)) == '1 ./ sqrt(x)'
    assert julia_code(sqrt(x)) == 'sqrt(x)'
    assert julia_code(x ** S.Half) == julia_code(x ** 0.5) == 'sqrt(x)'
    assert julia_code(1 / pi) == '1 / pi'
    assert julia_code(pi ** (-1)) == julia_code(pi ** (-1.0)) == '1 / pi'
    assert julia_code(pi ** (-0.5)) == '1 / sqrt(pi)'

def test_mix_number_mult_symbols():
    if False:
        i = 10
        return i + 15
    assert julia_code(3 * x) == '3 * x'
    assert julia_code(pi * x) == 'pi * x'
    assert julia_code(3 / x) == '3 ./ x'
    assert julia_code(pi / x) == 'pi ./ x'
    assert julia_code(x / 3) == 'x / 3'
    assert julia_code(x / pi) == 'x / pi'
    assert julia_code(x * y) == 'x .* y'
    assert julia_code(3 * x * y) == '3 * x .* y'
    assert julia_code(3 * pi * x * y) == '3 * pi * x .* y'
    assert julia_code(x / y) == 'x ./ y'
    assert julia_code(3 * x / y) == '3 * x ./ y'
    assert julia_code(x * y / z) == 'x .* y ./ z'
    assert julia_code(x / y * z) == 'x .* z ./ y'
    assert julia_code(1 / x / y) == '1 ./ (x .* y)'
    assert julia_code(2 * pi * x / y / z) == '2 * pi * x ./ (y .* z)'
    assert julia_code(3 * pi / x) == '3 * pi ./ x'
    assert julia_code(S(3) / 5) == '3 // 5'
    assert julia_code(S(3) / 5 * x) == '(3 // 5) * x'
    assert julia_code(x / y / z) == 'x ./ (y .* z)'
    assert julia_code((x + y) / z) == '(x + y) ./ z'
    assert julia_code((x + y) / (z + x)) == '(x + y) ./ (x + z)'
    assert julia_code((x + y) / EulerGamma) == '(x + y) / eulergamma'
    assert julia_code(x / 3 / pi) == 'x / (3 * pi)'
    assert julia_code(S(3) / 5 * x * y / pi) == '(3 // 5) * x .* y / pi'

def test_mix_number_pow_symbols():
    if False:
        for i in range(10):
            print('nop')
    assert julia_code(pi ** 3) == 'pi ^ 3'
    assert julia_code(x ** 2) == 'x .^ 2'
    assert julia_code(x ** pi ** 3) == 'x .^ (pi ^ 3)'
    assert julia_code(x ** y) == 'x .^ y'
    assert julia_code(x ** y ** z) == 'x .^ (y .^ z)'
    assert julia_code((x ** y) ** z) == '(x .^ y) .^ z'

def test_imag():
    if False:
        i = 10
        return i + 15
    I = S('I')
    assert julia_code(I) == 'im'
    assert julia_code(5 * I) == '5im'
    assert julia_code(S(3) / 2 * I) == '(3 // 2) * im'
    assert julia_code(3 + 4 * I) == '3 + 4im'

def test_constants():
    if False:
        return 10
    assert julia_code(pi) == 'pi'
    assert julia_code(oo) == 'Inf'
    assert julia_code(-oo) == '-Inf'
    assert julia_code(S.NegativeInfinity) == '-Inf'
    assert julia_code(S.NaN) == 'NaN'
    assert julia_code(S.Exp1) == 'e'
    assert julia_code(exp(1)) == 'e'

def test_constants_other():
    if False:
        i = 10
        return i + 15
    assert julia_code(2 * GoldenRatio) == '2 * golden'
    assert julia_code(2 * Catalan) == '2 * catalan'
    assert julia_code(2 * EulerGamma) == '2 * eulergamma'

def test_boolean():
    if False:
        i = 10
        return i + 15
    assert julia_code(x & y) == 'x && y'
    assert julia_code(x | y) == 'x || y'
    assert julia_code(~x) == '!x'
    assert julia_code(x & y & z) == 'x && y && z'
    assert julia_code(x | y | z) == 'x || y || z'
    assert julia_code(x & y | z) == 'z || x && y'
    assert julia_code((x | y) & z) == 'z && (x || y)'

def test_Matrices():
    if False:
        print('Hello World!')
    assert julia_code(Matrix(1, 1, [10])) == '[10]'
    A = Matrix([[1, sin(x / 2), abs(x)], [0, 1, pi], [0, exp(1), ceiling(x)]])
    expected = '[1 sin(x / 2)  abs(x);\n0          1      pi;\n0          e ceil(x)]'
    assert julia_code(A) == expected
    assert julia_code(A[:, 0]) == '[1, 0, 0]'
    assert julia_code(A[0, :]) == '[1 sin(x / 2) abs(x)]'
    assert julia_code(Matrix(0, 0, [])) == 'zeros(0, 0)'
    assert julia_code(Matrix(0, 3, [])) == 'zeros(0, 3)'
    assert julia_code(Matrix([[x, x - y, -y]])) == '[x x - y -y]'

def test_vector_entries_hadamard():
    if False:
        return 10
    A = Matrix([[1, sin(2 / x), 3 * pi / x / 5]])
    assert julia_code(A) == '[1 sin(2 ./ x) (3 // 5) * pi ./ x]'
    assert julia_code(A.T) == '[1, sin(2 ./ x), (3 // 5) * pi ./ x]'

@XFAIL
def test_Matrices_entries_not_hadamard():
    if False:
        i = 10
        return i + 15
    A = Matrix([[1, sin(2 / x), 3 * pi / x / 5], [1, 2, x * y]])
    expected = '[1 sin(2/x) 3*pi/(5*x);\n1        2        x*y]'
    assert julia_code(A) == expected

def test_MatrixSymbol():
    if False:
        for i in range(10):
            print('nop')
    n = Symbol('n', integer=True)
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, n)
    assert julia_code(A * B) == 'A * B'
    assert julia_code(B * A) == 'B * A'
    assert julia_code(2 * A * B) == '2 * A * B'
    assert julia_code(B * 2 * A) == '2 * B * A'
    assert julia_code(A * (B + 3 * Identity(n))) == 'A * (3 * eye(n) + B)'
    assert julia_code(A ** x ** 2) == 'A ^ (x .^ 2)'
    assert julia_code(A ** 3) == 'A ^ 3'
    assert julia_code(A ** S.Half) == 'A ^ (1 // 2)'

def test_special_matrices():
    if False:
        for i in range(10):
            print('nop')
    assert julia_code(6 * Identity(3)) == '6 * eye(3)'

def test_containers():
    if False:
        while True:
            i = 10
    assert julia_code([1, 2, 3, [4, 5, [6, 7]], 8, [9, 10], 11]) == 'Any[1, 2, 3, Any[4, 5, Any[6, 7]], 8, Any[9, 10], 11]'
    assert julia_code((1, 2, (3, 4))) == '(1, 2, (3, 4))'
    assert julia_code([1]) == 'Any[1]'
    assert julia_code((1,)) == '(1,)'
    assert julia_code(Tuple(*[1, 2, 3])) == '(1, 2, 3)'
    assert julia_code((1, x * y, (3, x ** 2))) == '(1, x .* y, (3, x .^ 2))'
    assert julia_code((1, eye(3), Matrix(0, 0, []), [])) == '(1, [1 0 0;\n0 1 0;\n0 0 1], zeros(0, 0), Any[])'

def test_julia_noninline():
    if False:
        while True:
            i = 10
    source = julia_code((x + y) / Catalan, assign_to='me', inline=False)
    expected = 'const Catalan = %s\nme = (x + y) / Catalan' % Catalan.evalf(17)
    assert source == expected

def test_julia_piecewise():
    if False:
        for i in range(10):
            print('nop')
    expr = Piecewise((x, x < 1), (x ** 2, True))
    assert julia_code(expr) == '((x < 1) ? (x) : (x .^ 2))'
    assert julia_code(expr, assign_to='r') == 'r = ((x < 1) ? (x) : (x .^ 2))'
    assert julia_code(expr, assign_to='r', inline=False) == 'if (x < 1)\n    r = x\nelse\n    r = x .^ 2\nend'
    expr = Piecewise((x ** 2, x < 1), (x ** 3, x < 2), (x ** 4, x < 3), (x ** 5, True))
    expected = '((x < 1) ? (x .^ 2) :\n(x < 2) ? (x .^ 3) :\n(x < 3) ? (x .^ 4) : (x .^ 5))'
    assert julia_code(expr) == expected
    assert julia_code(expr, assign_to='r') == 'r = ' + expected
    assert julia_code(expr, assign_to='r', inline=False) == 'if (x < 1)\n    r = x .^ 2\nelseif (x < 2)\n    r = x .^ 3\nelseif (x < 3)\n    r = x .^ 4\nelse\n    r = x .^ 5\nend'
    expr = Piecewise((x, x < 1), (x ** 2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda : julia_code(expr))

def test_julia_piecewise_times_const():
    if False:
        i = 10
        return i + 15
    pw = Piecewise((x, x < 1), (x ** 2, True))
    assert julia_code(2 * pw) == '2 * ((x < 1) ? (x) : (x .^ 2))'
    assert julia_code(pw / x) == '((x < 1) ? (x) : (x .^ 2)) ./ x'
    assert julia_code(pw / (x * y)) == '((x < 1) ? (x) : (x .^ 2)) ./ (x .* y)'
    assert julia_code(pw / 3) == '((x < 1) ? (x) : (x .^ 2)) / 3'

def test_julia_matrix_assign_to():
    if False:
        print('Hello World!')
    A = Matrix([[1, 2, 3]])
    assert julia_code(A, assign_to='a') == 'a = [1 2 3]'
    A = Matrix([[1, 2], [3, 4]])
    assert julia_code(A, assign_to='A') == 'A = [1 2;\n3 4]'

def test_julia_matrix_assign_to_more():
    if False:
        for i in range(10):
            print('nop')
    A = Matrix([[1, 2, 3]])
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 2, 3)
    assert julia_code(A, assign_to=B) == 'B = [1 2 3]'
    raises(ValueError, lambda : julia_code(A, assign_to=x))
    raises(ValueError, lambda : julia_code(A, assign_to=C))

def test_julia_matrix_1x1():
    if False:
        i = 10
        return i + 15
    A = Matrix([[3]])
    B = MatrixSymbol('B', 1, 1)
    C = MatrixSymbol('C', 1, 2)
    assert julia_code(A, assign_to=B) == 'B = [3]'
    raises(ValueError, lambda : julia_code(A, assign_to=C))

def test_julia_matrix_elements():
    if False:
        for i in range(10):
            print('nop')
    A = Matrix([[x, 2, x * y]])
    assert julia_code(A[0, 0] ** 2 + A[0, 1] + A[0, 2]) == 'x .^ 2 + x .* y + 2'
    A = MatrixSymbol('AA', 1, 3)
    assert julia_code(A) == 'AA'
    assert julia_code(A[0, 0] ** 2 + sin(A[0, 1]) + A[0, 2]) == 'sin(AA[1,2]) + AA[1,1] .^ 2 + AA[1,3]'
    assert julia_code(sum(A)) == 'AA[1,1] + AA[1,2] + AA[1,3]'

def test_julia_boolean():
    if False:
        print('Hello World!')
    assert julia_code(True) == 'true'
    assert julia_code(S.true) == 'true'
    assert julia_code(False) == 'false'
    assert julia_code(S.false) == 'false'

def test_julia_not_supported():
    if False:
        for i in range(10):
            print('nop')
    assert julia_code(S.ComplexInfinity) == '# Not supported in Julia:\n# ComplexInfinity\nzoo'
    f = Function('f')
    assert julia_code(f(x).diff(x)) == '# Not supported in Julia:\n# Derivative\nDerivative(f(x), x)'

def test_trick_indent_with_end_else_words():
    if False:
        for i in range(10):
            print('nop')
    t1 = S('endless')
    t2 = S('elsewhere')
    pw = Piecewise((t1, x < 0), (t2, x <= 1), (1, True))
    assert julia_code(pw, inline=False) == 'if (x < 0)\n    endless\nelseif (x <= 1)\n    elsewhere\nelse\n    1\nend'

def test_haramard():
    if False:
        for i in range(10):
            print('nop')
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 3, 3)
    v = MatrixSymbol('v', 3, 1)
    h = MatrixSymbol('h', 1, 3)
    C = HadamardProduct(A, B)
    assert julia_code(C) == 'A .* B'
    assert julia_code(C * v) == '(A .* B) * v'
    assert julia_code(h * C * v) == 'h * (A .* B) * v'
    assert julia_code(C * A) == '(A .* B) * A'
    assert julia_code(C * x * y) == '(x .* y) * (A .* B)'

def test_sparse():
    if False:
        i = 10
        return i + 15
    M = SparseMatrix(5, 6, {})
    M[2, 2] = 10
    M[1, 2] = 20
    M[1, 3] = 22
    M[0, 3] = 30
    M[3, 0] = x * y
    assert julia_code(M) == 'sparse([4, 2, 3, 1, 2], [1, 3, 3, 4, 4], [x .* y, 20, 10, 30, 22], 5, 6)'

def test_specfun():
    if False:
        while True:
            i = 10
    n = Symbol('n')
    for f in [besselj, bessely, besseli, besselk]:
        assert julia_code(f(n, x)) == f.__name__ + '(n, x)'
    for f in [airyai, airyaiprime, airybi, airybiprime]:
        assert julia_code(f(x)) == f.__name__ + '(x)'
    assert julia_code(hankel1(n, x)) == 'hankelh1(n, x)'
    assert julia_code(hankel2(n, x)) == 'hankelh2(n, x)'
    assert julia_code(jn(n, x)) == 'sqrt(2) * sqrt(pi) * sqrt(1 ./ x) .* besselj(n + 1 // 2, x) / 2'
    assert julia_code(yn(n, x)) == 'sqrt(2) * sqrt(pi) * sqrt(1 ./ x) .* bessely(n + 1 // 2, x) / 2'

def test_MatrixElement_printing():
    if False:
        for i in range(10):
            print('nop')
    A = MatrixSymbol('A', 1, 3)
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 1, 3)
    assert julia_code(A[0, 0]) == 'A[1,1]'
    assert julia_code(3 * A[0, 0]) == '3 * A[1,1]'
    F = C[0, 0].subs(C, A - B)
    assert julia_code(F) == '(A - B)[1,1]'