from sympy.core import S, symbols
from sympy.matrices import eye, ones, Matrix, ShapeError
from sympy.matrices.expressions import Identity, MatrixExpr, MatrixSymbol, Determinant, det, per, ZeroMatrix, Transpose, Permanent, MatMul
from sympy.matrices.expressions.special import OneMatrix
from sympy.testing.pytest import raises
from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine
n = symbols('n', integer=True)
A = MatrixSymbol('A', n, n)
B = MatrixSymbol('B', n, n)
C = MatrixSymbol('C', 3, 4)

def test_det():
    if False:
        return 10
    assert isinstance(Determinant(A), Determinant)
    assert not isinstance(Determinant(A), MatrixExpr)
    raises(ShapeError, lambda : Determinant(C))
    assert det(eye(3)) == 1
    assert det(Matrix(3, 3, [1, 3, 2, 4, 1, 3, 2, 5, 2])) == 17
    _ = A / det(A)
    raises(TypeError, lambda : Determinant(S.One))
    assert Determinant(A).arg is A

def test_eval_determinant():
    if False:
        while True:
            i = 10
    assert det(Identity(n)) == 1
    assert det(ZeroMatrix(n, n)) == 0
    assert det(OneMatrix(n, n)) == Determinant(OneMatrix(n, n))
    assert det(OneMatrix(1, 1)) == 1
    assert det(OneMatrix(2, 2)) == 0
    assert det(Transpose(A)) == det(A)
    assert Determinant(MatMul(eye(2), eye(2))).doit(deep=True) == 1

def test_refine():
    if False:
        for i in range(10):
            print('nop')
    assert refine(det(A), Q.orthogonal(A)) == 1
    assert refine(det(A), Q.singular(A)) == 0
    assert refine(det(A), Q.unit_triangular(A)) == 1
    assert refine(det(A), Q.normal(A)) == det(A)

def test_commutative():
    if False:
        return 10
    det_a = Determinant(A)
    det_b = Determinant(B)
    assert det_a.is_commutative
    assert det_b.is_commutative
    assert det_a * det_b == det_b * det_a

def test_permanent():
    if False:
        print('Hello World!')
    assert isinstance(Permanent(A), Permanent)
    assert not isinstance(Permanent(A), MatrixExpr)
    assert isinstance(Permanent(C), Permanent)
    assert Permanent(ones(3, 3)).doit() == 6
    _ = C / per(C)
    assert per(Matrix(3, 3, [1, 3, 2, 4, 1, 3, 2, 5, 2])) == 103
    raises(TypeError, lambda : Permanent(S.One))
    assert Permanent(A).arg is A