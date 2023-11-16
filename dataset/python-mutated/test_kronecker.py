from sympy.core.mod import Mod
from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.functions.elementary.integers import floor
from sympy.matrices.dense import Matrix, eye
from sympy.matrices import MatrixSymbol, Identity
from sympy.matrices.expressions import det, trace
from sympy.matrices.expressions.kronecker import KroneckerProduct, kronecker_product, combine_kronecker
mat1 = Matrix([[1, 2 * I], [1 + I, 3]])
mat2 = Matrix([[2 * I, 3], [4 * I, 2]])
(i, j, k, n, m, o, p, x) = symbols('i,j,k,n,m,o,p,x')
Z = MatrixSymbol('Z', n, n)
W = MatrixSymbol('W', m, m)
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', n, m)
C = MatrixSymbol('C', m, k)

def test_KroneckerProduct():
    if False:
        while True:
            i = 10
    assert isinstance(KroneckerProduct(A, B), KroneckerProduct)
    assert KroneckerProduct(A, B).subs(A, C) == KroneckerProduct(C, B)
    assert KroneckerProduct(A, C).shape == (n * m, m * k)
    assert (KroneckerProduct(A, C) + KroneckerProduct(-A, C)).is_ZeroMatrix
    assert (KroneckerProduct(W, Z) * KroneckerProduct(W.I, Z.I)).is_Identity

def test_KroneckerProduct_identity():
    if False:
        print('Hello World!')
    assert KroneckerProduct(Identity(m), Identity(n)) == Identity(m * n)
    assert KroneckerProduct(eye(2), eye(3)) == eye(6)

def test_KroneckerProduct_explicit():
    if False:
        print('Hello World!')
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    kp = KroneckerProduct(X, Y)
    assert kp.shape == (4, 4)
    assert kp.as_explicit() == Matrix([[X[0, 0] * Y[0, 0], X[0, 0] * Y[0, 1], X[0, 1] * Y[0, 0], X[0, 1] * Y[0, 1]], [X[0, 0] * Y[1, 0], X[0, 0] * Y[1, 1], X[0, 1] * Y[1, 0], X[0, 1] * Y[1, 1]], [X[1, 0] * Y[0, 0], X[1, 0] * Y[0, 1], X[1, 1] * Y[0, 0], X[1, 1] * Y[0, 1]], [X[1, 0] * Y[1, 0], X[1, 0] * Y[1, 1], X[1, 1] * Y[1, 0], X[1, 1] * Y[1, 1]]])

def test_tensor_product_adjoint():
    if False:
        while True:
            i = 10
    assert KroneckerProduct(I * A, B).adjoint() == -I * KroneckerProduct(A.adjoint(), B.adjoint())
    assert KroneckerProduct(mat1, mat2).adjoint() == kronecker_product(mat1.adjoint(), mat2.adjoint())

def test_tensor_product_conjugate():
    if False:
        while True:
            i = 10
    assert KroneckerProduct(I * A, B).conjugate() == -I * KroneckerProduct(A.conjugate(), B.conjugate())
    assert KroneckerProduct(mat1, mat2).conjugate() == kronecker_product(mat1.conjugate(), mat2.conjugate())

def test_tensor_product_transpose():
    if False:
        return 10
    assert KroneckerProduct(I * A, B).transpose() == I * KroneckerProduct(A.transpose(), B.transpose())
    assert KroneckerProduct(mat1, mat2).transpose() == kronecker_product(mat1.transpose(), mat2.transpose())

def test_KroneckerProduct_is_associative():
    if False:
        return 10
    assert kronecker_product(A, kronecker_product(B, C)) == kronecker_product(kronecker_product(A, B), C)
    assert kronecker_product(A, kronecker_product(B, C)) == KroneckerProduct(A, B, C)

def test_KroneckerProduct_is_bilinear():
    if False:
        for i in range(10):
            print('nop')
    assert kronecker_product(x * A, B) == x * kronecker_product(A, B)
    assert kronecker_product(A, x * B) == x * kronecker_product(A, B)

def test_KroneckerProduct_determinant():
    if False:
        i = 10
        return i + 15
    kp = kronecker_product(W, Z)
    assert det(kp) == det(W) ** n * det(Z) ** m

def test_KroneckerProduct_trace():
    if False:
        print('Hello World!')
    kp = kronecker_product(W, Z)
    assert trace(kp) == trace(W) * trace(Z)

def test_KroneckerProduct_isnt_commutative():
    if False:
        i = 10
        return i + 15
    assert KroneckerProduct(A, B) != KroneckerProduct(B, A)
    assert KroneckerProduct(A, B).is_commutative is False

def test_KroneckerProduct_extracts_commutative_part():
    if False:
        i = 10
        return i + 15
    assert kronecker_product(x * A, 2 * B) == x * 2 * KroneckerProduct(A, B)

def test_KroneckerProduct_inverse():
    if False:
        while True:
            i = 10
    kp = kronecker_product(W, Z)
    assert kp.inverse() == kronecker_product(W.inverse(), Z.inverse())

def test_KroneckerProduct_combine_add():
    if False:
        for i in range(10):
            print('nop')
    kp1 = kronecker_product(A, B)
    kp2 = kronecker_product(C, W)
    assert combine_kronecker(kp1 * kp2) == kronecker_product(A * C, B * W)

def test_KroneckerProduct_combine_mul():
    if False:
        while True:
            i = 10
    X = MatrixSymbol('X', m, n)
    Y = MatrixSymbol('Y', m, n)
    kp1 = kronecker_product(A, X)
    kp2 = kronecker_product(B, Y)
    assert combine_kronecker(kp1 + kp2) == kronecker_product(A + B, X + Y)

def test_KroneckerProduct_combine_pow():
    if False:
        return 10
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', n, n)
    assert combine_kronecker(KroneckerProduct(X, Y) ** x) == KroneckerProduct(X ** x, Y ** x)
    assert combine_kronecker(x * KroneckerProduct(X, Y) ** 2) == x * KroneckerProduct(X ** 2, Y ** 2)
    assert combine_kronecker(x * KroneckerProduct(X, Y) ** 2 * KroneckerProduct(A, B)) == x * KroneckerProduct(X ** 2 * A, Y ** 2 * B)
    assert combine_kronecker(KroneckerProduct(A, B.T) ** m) == KroneckerProduct(A, B.T) ** m

def test_KroneckerProduct_expand():
    if False:
        while True:
            i = 10
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', n, n)
    assert KroneckerProduct(X + Y, Y + Z).expand(kroneckerproduct=True) == KroneckerProduct(X, Y) + KroneckerProduct(X, Z) + KroneckerProduct(Y, Y) + KroneckerProduct(Y, Z)

def test_KroneckerProduct_entry():
    if False:
        for i in range(10):
            print('nop')
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', o, p)
    assert KroneckerProduct(A, B)._entry(i, j) == A[Mod(floor(i / o), n), Mod(floor(j / p), m)] * B[Mod(i, o), Mod(j, p)]