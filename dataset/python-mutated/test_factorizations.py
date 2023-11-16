from sympy.matrices.expressions.factorizations import lu, LofCholesky, qr, svd
from sympy.assumptions.ask import Q, ask
from sympy.core.symbol import Symbol
from sympy.matrices.expressions.matexpr import MatrixSymbol
n = Symbol('n')
X = MatrixSymbol('X', n, n)

def test_LU():
    if False:
        print('Hello World!')
    (L, U) = lu(X)
    assert L.shape == U.shape == X.shape
    assert ask(Q.lower_triangular(L))
    assert ask(Q.upper_triangular(U))

def test_Cholesky():
    if False:
        i = 10
        return i + 15
    LofCholesky(X)

def test_QR():
    if False:
        return 10
    (Q_, R) = qr(X)
    assert Q_.shape == R.shape == X.shape
    assert ask(Q.orthogonal(Q_))
    assert ask(Q.upper_triangular(R))

def test_svd():
    if False:
        while True:
            i = 10
    (U, S, V) = svd(X)
    assert U.shape == S.shape == V.shape == X.shape
    assert ask(Q.orthogonal(U))
    assert ask(Q.orthogonal(V))
    assert ask(Q.diagonal(S))