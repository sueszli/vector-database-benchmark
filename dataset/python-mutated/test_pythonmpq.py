"""
test_pythonmpq.py

Test the PythonMPQ class for consistency with gmpy2's mpq type. If gmpy2 is
installed run the same tests for both.
"""
from fractions import Fraction
from decimal import Decimal
import pickle
from typing import Callable, List, Tuple, Type
from sympy.testing.pytest import raises
from sympy.external.pythonmpq import PythonMPQ
rational_types: List[Tuple[Callable, Type, Callable, Type]]
rational_types = [(PythonMPQ, PythonMPQ, int, int)]
try:
    from gmpy2 import mpq, mpz
    rational_types.append((mpq, type(mpq(1)), mpz, type(mpz(1))))
except ImportError:
    pass

def test_PythonMPQ():
    if False:
        while True:
            i = 10
    for (Q, TQ, Z, TZ) in rational_types:

        def check_Q(q):
            if False:
                print('Hello World!')
            assert isinstance(q, TQ)
            assert isinstance(q.numerator, TZ)
            assert isinstance(q.denominator, TZ)
            return (q.numerator, q.denominator)
        assert check_Q(Q(3)) == (3, 1)
        assert check_Q(Q(3, 5)) == (3, 5)
        assert check_Q(Q(Q(3, 5))) == (3, 5)
        assert check_Q(Q(0.5)) == (1, 2)
        assert check_Q(Q('0.5')) == (1, 2)
        assert check_Q(Q(Fraction(3, 5))) == (3, 5)
        if Q is PythonMPQ:
            assert check_Q(Q(Decimal('0.6'))) == (3, 5)
        raises(TypeError, lambda : Q([]))
        raises(TypeError, lambda : Q([], []))
        assert check_Q(Q(2, 3)) == (2, 3)
        assert check_Q(Q(-2, 3)) == (-2, 3)
        assert check_Q(Q(2, -3)) == (-2, 3)
        assert check_Q(Q(-2, -3)) == (2, 3)
        assert check_Q(Q(12, 8)) == (3, 2)
        assert int(Q(5, 3)) == 1
        assert int(Q(-5, 3)) == -1
        assert float(Q(5, 2)) == 2.5
        assert float(Q(-5, 2)) == -2.5
        assert str(Q(2, 1)) == '2'
        assert str(Q(1, 2)) == '1/2'
        if Q is PythonMPQ:
            assert repr(Q(2, 1)) == 'MPQ(2,1)'
            assert repr(Q(1, 2)) == 'MPQ(1,2)'
        else:
            assert repr(Q(2, 1)) == 'mpq(2,1)'
            assert repr(Q(1, 2)) == 'mpq(1,2)'
        assert bool(Q(1, 2)) is True
        assert bool(Q(0)) is False
        assert (Q(2, 3) == Q(2, 3)) is True
        assert (Q(2, 3) == Q(2, 5)) is False
        assert (Q(2, 3) != Q(2, 3)) is False
        assert (Q(2, 3) != Q(2, 5)) is True
        assert hash(Q(3, 5)) == hash(Fraction(3, 5))
        q = Q(2, 3)
        assert pickle.loads(pickle.dumps(q)) == q
        assert (Q(1, 3) < Q(2, 3)) is True
        assert (Q(2, 3) < Q(2, 3)) is False
        assert (Q(2, 3) < Q(1, 3)) is False
        assert (Q(-2, 3) < Q(1, 3)) is True
        assert (Q(1, 3) < Q(-2, 3)) is False
        assert (Q(1, 3) <= Q(2, 3)) is True
        assert (Q(2, 3) <= Q(2, 3)) is True
        assert (Q(2, 3) <= Q(1, 3)) is False
        assert (Q(-2, 3) <= Q(1, 3)) is True
        assert (Q(1, 3) <= Q(-2, 3)) is False
        assert (Q(1, 3) > Q(2, 3)) is False
        assert (Q(2, 3) > Q(2, 3)) is False
        assert (Q(2, 3) > Q(1, 3)) is True
        assert (Q(-2, 3) > Q(1, 3)) is False
        assert (Q(1, 3) > Q(-2, 3)) is True
        assert (Q(1, 3) >= Q(2, 3)) is False
        assert (Q(2, 3) >= Q(2, 3)) is True
        assert (Q(2, 3) >= Q(1, 3)) is True
        assert (Q(-2, 3) >= Q(1, 3)) is False
        assert (Q(1, 3) >= Q(-2, 3)) is True
        assert abs(Q(2, 3)) == abs(Q(-2, 3)) == Q(2, 3)
        assert +Q(2, 3) == Q(2, 3)
        assert -Q(2, 3) == Q(-2, 3)
        assert Q(2, 3) + Q(5, 7) == Q(29, 21)
        assert Q(2, 3) + 1 == Q(5, 3)
        assert 1 + Q(2, 3) == Q(5, 3)
        raises(TypeError, lambda : [] + Q(1))
        raises(TypeError, lambda : Q(1) + [])
        assert Q(2, 3) - Q(5, 7) == Q(-1, 21)
        assert Q(2, 3) - 1 == Q(-1, 3)
        assert 1 - Q(2, 3) == Q(1, 3)
        raises(TypeError, lambda : [] - Q(1))
        raises(TypeError, lambda : Q(1) - [])
        assert Q(2, 3) * Q(5, 7) == Q(10, 21)
        assert Q(2, 3) * 1 == Q(2, 3)
        assert 1 * Q(2, 3) == Q(2, 3)
        raises(TypeError, lambda : [] * Q(1))
        raises(TypeError, lambda : Q(1) * [])
        assert Q(2, 3) ** 2 == Q(4, 9)
        assert Q(2, 3) ** 1 == Q(2, 3)
        assert Q(-2, 3) ** 2 == Q(4, 9)
        assert Q(-2, 3) ** (-1) == Q(-3, 2)
        if Q is PythonMPQ:
            raises(TypeError, lambda : 1 ** Q(2, 3))
            raises(TypeError, lambda : Q(1, 4) ** Q(1, 2))
        raises(TypeError, lambda : [] ** Q(1))
        raises(TypeError, lambda : Q(1) ** [])
        assert Q(2, 3) / Q(5, 7) == Q(14, 15)
        assert Q(2, 3) / 1 == Q(2, 3)
        assert 1 / Q(2, 3) == Q(3, 2)
        raises(TypeError, lambda : [] / Q(1))
        raises(TypeError, lambda : Q(1) / [])
        raises(ZeroDivisionError, lambda : Q(1, 2) / Q(0))
        if Q is PythonMPQ:
            raises(TypeError, lambda : Q(2, 3) // Q(1, 3))
            raises(TypeError, lambda : Q(2, 3) % Q(1, 3))
            raises(TypeError, lambda : 1 // Q(1, 3))
            raises(TypeError, lambda : 1 % Q(1, 3))
            raises(TypeError, lambda : Q(2, 3) // 1)
            raises(TypeError, lambda : Q(2, 3) % 1)