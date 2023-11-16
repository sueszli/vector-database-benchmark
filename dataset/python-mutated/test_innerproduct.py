from sympy.core.numbers import I, Integer
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.state import Bra, Ket, StateBase

def test_innerproduct():
    if False:
        print('Hello World!')
    k = Ket('k')
    b = Bra('b')
    ip = InnerProduct(b, k)
    assert isinstance(ip, InnerProduct)
    assert ip.bra == b
    assert ip.ket == k
    assert b * k == InnerProduct(b, k)
    assert k * (b * k) * b == k * InnerProduct(b, k) * b
    assert InnerProduct(b, k).subs(b, Dagger(k)) == Dagger(k) * k

def test_innerproduct_dagger():
    if False:
        return 10
    k = Ket('k')
    b = Bra('b')
    ip = b * k
    assert Dagger(ip) == Dagger(k) * Dagger(b)

class FooState(StateBase):
    pass

class FooKet(Ket, FooState):

    @classmethod
    def dual_class(self):
        if False:
            print('Hello World!')
        return FooBra

    def _eval_innerproduct_FooBra(self, bra):
        if False:
            i = 10
            return i + 15
        return Integer(1)

    def _eval_innerproduct_BarBra(self, bra):
        if False:
            for i in range(10):
                print('nop')
        return I

class FooBra(Bra, FooState):

    @classmethod
    def dual_class(self):
        if False:
            print('Hello World!')
        return FooKet

class BarState(StateBase):
    pass

class BarKet(Ket, BarState):

    @classmethod
    def dual_class(self):
        if False:
            print('Hello World!')
        return BarBra

class BarBra(Bra, BarState):

    @classmethod
    def dual_class(self):
        if False:
            while True:
                i = 10
        return BarKet

def test_doit():
    if False:
        for i in range(10):
            print('nop')
    f = FooKet('foo')
    b = BarBra('bar')
    assert InnerProduct(b, f).doit() == I
    assert InnerProduct(Dagger(f), Dagger(b)).doit() == -I
    assert InnerProduct(Dagger(f), f).doit() == Integer(1)