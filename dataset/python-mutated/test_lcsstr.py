import pytest
import textdistance
ALG = textdistance.LCSStr

@pytest.mark.parametrize('left, right, expected', [('ab', 'abcd', 'ab'), ('abcd', 'ab', 'ab'), ('abcd', 'bc', 'bc'), ('bc', 'abcd', 'bc'), ('abcd', 'cd', 'cd'), ('abcd', 'cd', 'cd'), ('abcd', 'ef', ''), ('ef', 'abcd', ''), ('MYTEST' * 100, 'TEST', 'TEST'), ('TEST', 'MYTEST' * 100, 'TEST')])
def test_distance(left, right, expected):
    if False:
        i = 10
        return i + 15
    actual = ALG(external=False)(left, right)
    assert actual == expected
    actual = ALG(external=True)(left, right)
    assert actual == expected