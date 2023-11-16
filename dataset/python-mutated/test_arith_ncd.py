from fractions import Fraction
from math import isclose
import pytest
import textdistance
ALG = textdistance.arith_ncd

@pytest.mark.parametrize('left, right, expected', [('test', 'test', 1), ('test', 'nani', 2.1666666666666665)])
def test_similarity(left, right, expected):
    if False:
        while True:
            i = 10
    actual = ALG(left, right)
    assert isclose(actual, expected)

def test_make_probs():
    if False:
        return 10
    alg = textdistance.ArithNCD(terminator='\x00')
    probs = alg._make_probs('lol', 'lal')
    assert probs['l'] == (Fraction(0, 1), Fraction(4, 7))
    assert probs['o'][1] == Fraction(1, 7)
    assert probs['a'][1] == Fraction(1, 7)

def test_arith_output():
    if False:
        print('Hello World!')
    alg = textdistance.ArithNCD(terminator='\x00')
    fraction = alg._compress('BANANA')
    assert fraction.numerator == 1525