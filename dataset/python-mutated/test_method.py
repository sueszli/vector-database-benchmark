from sympy.physics.mechanics.method import _Methods
from sympy.testing.pytest import raises

def test_method():
    if False:
        i = 10
        return i + 15
    raises(TypeError, lambda : _Methods())