from sympy.testing.pytest import raises
from sympy.core.symbol import S
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.domainscalar import DomainScalar
from sympy.polys.matrices.domainmatrix import DomainMatrix

def test_DomainScalar___new__():
    if False:
        i = 10
        return i + 15
    raises(TypeError, lambda : DomainScalar(ZZ(1), QQ))
    raises(TypeError, lambda : DomainScalar(ZZ(1), 1))

def test_DomainScalar_new():
    if False:
        for i in range(10):
            print('nop')
    A = DomainScalar(ZZ(1), ZZ)
    B = A.new(ZZ(4), ZZ)
    assert B == DomainScalar(ZZ(4), ZZ)

def test_DomainScalar_repr():
    if False:
        return 10
    A = DomainScalar(ZZ(1), ZZ)
    assert repr(A) in {'1', 'mpz(1)'}

def test_DomainScalar_from_sympy():
    if False:
        print('Hello World!')
    expr = S(1)
    B = DomainScalar.from_sympy(expr)
    assert B == DomainScalar(ZZ(1), ZZ)

def test_DomainScalar_to_sympy():
    if False:
        print('Hello World!')
    B = DomainScalar(ZZ(1), ZZ)
    expr = B.to_sympy()
    assert expr.is_Integer and expr == 1

def test_DomainScalar_to_domain():
    if False:
        while True:
            i = 10
    A = DomainScalar(ZZ(1), ZZ)
    B = A.to_domain(QQ)
    assert B == DomainScalar(QQ(1), QQ)

def test_DomainScalar_convert_to():
    if False:
        while True:
            i = 10
    A = DomainScalar(ZZ(1), ZZ)
    B = A.convert_to(QQ)
    assert B == DomainScalar(QQ(1), QQ)

def test_DomainScalar_unify():
    if False:
        for i in range(10):
            print('nop')
    A = DomainScalar(ZZ(1), ZZ)
    B = DomainScalar(QQ(2), QQ)
    (A, B) = A.unify(B)
    assert A.domain == B.domain == QQ

def test_DomainScalar_add():
    if False:
        return 10
    A = DomainScalar(ZZ(1), ZZ)
    B = DomainScalar(QQ(2), QQ)
    assert A + B == DomainScalar(QQ(3), QQ)
    raises(TypeError, lambda : A + 1.5)

def test_DomainScalar_sub():
    if False:
        print('Hello World!')
    A = DomainScalar(ZZ(1), ZZ)
    B = DomainScalar(QQ(2), QQ)
    assert A - B == DomainScalar(QQ(-1), QQ)
    raises(TypeError, lambda : A - 1.5)

def test_DomainScalar_mul():
    if False:
        for i in range(10):
            print('nop')
    A = DomainScalar(ZZ(1), ZZ)
    B = DomainScalar(QQ(2), QQ)
    dm = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A * B == DomainScalar(QQ(2), QQ)
    assert A * dm == dm
    assert B * 2 == DomainScalar(QQ(4), QQ)
    raises(TypeError, lambda : A * 1.5)

def test_DomainScalar_floordiv():
    if False:
        while True:
            i = 10
    A = DomainScalar(ZZ(-5), ZZ)
    B = DomainScalar(QQ(2), QQ)
    assert A // B == DomainScalar(QQ(-5, 2), QQ)
    C = DomainScalar(ZZ(2), ZZ)
    assert A // C == DomainScalar(ZZ(-3), ZZ)
    raises(TypeError, lambda : A // 1.5)

def test_DomainScalar_mod():
    if False:
        return 10
    A = DomainScalar(ZZ(5), ZZ)
    B = DomainScalar(QQ(2), QQ)
    assert A % B == DomainScalar(QQ(0), QQ)
    C = DomainScalar(ZZ(2), ZZ)
    assert A % C == DomainScalar(ZZ(1), ZZ)
    raises(TypeError, lambda : A % 1.5)

def test_DomainScalar_divmod():
    if False:
        print('Hello World!')
    A = DomainScalar(ZZ(5), ZZ)
    B = DomainScalar(QQ(2), QQ)
    assert divmod(A, B) == (DomainScalar(QQ(5, 2), QQ), DomainScalar(QQ(0), QQ))
    C = DomainScalar(ZZ(2), ZZ)
    assert divmod(A, C) == (DomainScalar(ZZ(2), ZZ), DomainScalar(ZZ(1), ZZ))
    raises(TypeError, lambda : divmod(A, 1.5))

def test_DomainScalar_pow():
    if False:
        print('Hello World!')
    A = DomainScalar(ZZ(-5), ZZ)
    B = A ** 2
    assert B == DomainScalar(ZZ(25), ZZ)
    raises(TypeError, lambda : A ** 1.5)

def test_DomainScalar_pos():
    if False:
        print('Hello World!')
    A = DomainScalar(QQ(2), QQ)
    B = DomainScalar(QQ(2), QQ)
    assert +A == B

def test_DomainScalar_neg():
    if False:
        return 10
    A = DomainScalar(QQ(2), QQ)
    B = DomainScalar(QQ(-2), QQ)
    assert -A == B

def test_DomainScalar_eq():
    if False:
        for i in range(10):
            print('nop')
    A = DomainScalar(QQ(2), QQ)
    assert A == A
    B = DomainScalar(ZZ(-5), ZZ)
    assert A != B
    C = DomainScalar(ZZ(2), ZZ)
    assert A != C
    D = [1]
    assert A != D

def test_DomainScalar_isZero():
    if False:
        while True:
            i = 10
    A = DomainScalar(ZZ(0), ZZ)
    assert A.is_zero() == True
    B = DomainScalar(ZZ(1), ZZ)
    assert B.is_zero() == False

def test_DomainScalar_isOne():
    if False:
        i = 10
        return i + 15
    A = DomainScalar(ZZ(1), ZZ)
    assert A.is_one() == True
    B = DomainScalar(ZZ(0), ZZ)
    assert B.is_one() == False