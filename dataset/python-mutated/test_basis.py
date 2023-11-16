from sympy.abc import x
from sympy.core import S
from sympy.core.numbers import AlgebraicNumber
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.basis import round_two
from sympy.testing.pytest import raises

def test_round_two():
    if False:
        print('Hello World!')
    raises(ValueError, lambda : round_two(Poly(x ** 2 - 1)))
    raises(ValueError, lambda : round_two(Poly(x ** 2 + sqrt(2))))
    cases = ((cyclotomic_poly(5), DomainMatrix.eye(4, QQ), 125), (cyclotomic_poly(7), DomainMatrix.eye(6, QQ), -16807), (x ** 2 - 5, DM([[1, (1, 2)], [0, (1, 2)]], QQ), 5), (x ** 2 - 7, DM([[1, 0], [0, 1]], QQ), 28), (x ** 3 + x ** 2 - 2 * x + 8, DM([[1, 0, 0], [0, 1, 0], [0, (1, 2), (1, 2)]], QQ).transpose(), -503), (x ** 3 + 3 * x ** 2 - 4 * x + 4, DM([((1, 2), (1, 4), (1, 4)), (0, (1, 2), (1, 2)), (0, 0, 1)], QQ).transpose(), -83), (x ** 3 + 3 * x ** 2 + 3 * x - 3, DM([((1, 2), 0, (1, 2)), (0, 1, 0), (0, 0, 1)], QQ).transpose(), -108), (x ** 3 + 5 * x ** 2 - x + 3, DM([((1, 4), 0, (3, 4)), (0, (1, 2), (1, 2)), (0, 0, 1)], QQ).transpose(), -31), (x ** 3 + 5 * x ** 2 - 5 * x - 5, DM([((1, 2), 0, (1, 2)), (0, 1, 0), (0, 0, 1)], QQ).transpose(), 1300), (x ** 3 + 3 * x ** 2 + 5, DM([((1, 3), (1, 3), (1, 3)), (0, 1, 0), (0, 0, 1)], QQ).transpose(), -135), (x ** 3 + 6 * x ** 2 + 3 * x - 1, DM([((1, 3), (1, 3), (1, 3)), (0, 1, 0), (0, 0, 1)], QQ).transpose(), 81), (x ** 3 + 6 * x ** 2 + 4, DM([((1, 3), (2, 3), (1, 3)), (0, 1, 0), (0, 0, (1, 2))], QQ).transpose(), -108), (x ** 3 + 7 * x ** 2 + 7 * x - 7, DM([((1, 4), 0, (3, 4)), (0, (1, 2), (1, 2)), (0, 0, 1)], QQ).transpose(), 49), (x ** 3 + 7 * x ** 2 - x + 5, DM([((1, 2), 0, (1, 2)), (0, 1, 0), (0, 0, 1)], QQ).transpose(), -2028), (x ** 3 + 7 * x ** 2 - 5 * x + 5, DM([((1, 4), 0, (3, 4)), (0, (1, 2), (1, 2)), (0, 0, 1)], QQ).transpose(), -140), (x ** 3 + 4 * x ** 2 - 3 * x + 7, DM([((1, 5), (4, 5), (4, 5)), (0, 1, 0), (0, 0, 1)], QQ).transpose(), -175), (x ** 3 + 8 * x ** 2 + 5 * x - 1, DM([((1, 7), (6, 7), (2, 7)), (0, 1, 0), (0, 0, 1)], QQ).transpose(), 49), (x ** 3 + 8 * x ** 2 - 2 * x + 6, DM([(1, 0, 0), (0, 1, 0), (0, 0, 1)], QQ).transpose(), -14700), (x ** 3 + 6 * x ** 2 - 3 * x + 8, DM([(1, 0, 0), (0, (1, 4), (1, 4)), (0, 0, 1)], QQ).transpose(), -675), (x ** 3 + 9 * x ** 2 + 6 * x - 8, DM([(1, 0, 0), (0, (1, 2), (1, 2)), (0, 0, 1)], QQ).transpose(), 3969), (x ** 3 + 15 * x ** 2 - 9 * x + 13, DM([((1, 6), (1, 3), (1, 6)), (0, 1, 0), (0, 0, 1)], QQ).transpose(), -5292), (5 * x ** 3 + 5 * x ** 2 - 10 * x + 40, DM([[1, 0, 0], [0, 1, 0], [0, (1, 2), (1, 2)]], QQ).transpose(), -503), (QQ(5, 3) * x ** 3 + QQ(5, 3) * x ** 2 - QQ(10, 3) * x + QQ(40, 3), DM([[1, 0, 0], [0, 1, 0], [0, (1, 2), (1, 2)]], QQ).transpose(), -503))
    for (f, B_exp, d_exp) in cases:
        K = QQ.alg_field_from_poly(f)
        B = K.maximal_order().QQ_matrix
        d = K.discriminant()
        assert d == d_exp
        assert (B.inv() * B_exp).det() ** 2 == 1

def test_AlgebraicField_integral_basis():
    if False:
        for i in range(10):
            print('nop')
    alpha = AlgebraicNumber(sqrt(5), alias='alpha')
    k = QQ.algebraic_field(alpha)
    B0 = k.integral_basis()
    B1 = k.integral_basis(fmt='sympy')
    B2 = k.integral_basis(fmt='alg')
    assert B0 == [k([1]), k([S.Half, S.Half])]
    assert B1 == [1, S.Half + alpha / 2]
    assert B2 == [k.ext.field_element([1]), k.ext.field_element([S.Half, S.Half])]