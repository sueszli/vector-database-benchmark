import unittest
import pytype_extensions
from pytype_extensions import instrumentation_for_testing as i4t
assert_type = pytype_extensions.assert_type

class NoCtor:

    def __init__(self):
        if False:
            print('Hello World!')
        raise RuntimeError('Meant to be inaccessible')

    def Mul100(self, i):
        if False:
            i = 10
            return i + 15
        return i * 100

class FakeNoCtor(i4t.ProductionType[NoCtor]):

    def __init__(self, state):
        if False:
            i = 10
            return i + 15
        self.state = state
        self.call_count = 0

    def Mul100(self, i):
        if False:
            while True:
                i = 10
        self.call_count += 1
        return self.state * i * 100

class FakeNoCtorInitArgUnsealed(i4t.ProductionType[NoCtor]):

    def __init__(self, state):
        if False:
            while True:
                i = 10
        self.state = state

    def Mul100(self, i):
        if False:
            while True:
                i = 10
        return self.state * i * 103

class FakeNoCtorDefaultInitUnsealed(i4t.ProductionType[NoCtor]):

    def Mul100(self, i):
        if False:
            for i in range(10):
                print('nop')
        return i * 104
FakeNoCtorDefaultInitSealed = FakeNoCtorDefaultInitUnsealed.SealType()

class FakeNoCtorInitNoArgsUnsealed(i4t.ProductionType[NoCtor]):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.state = 8

    def Mul100(self, i):
        if False:
            for i in range(10):
                print('nop')
        return self.state * i * 105
FakeNoCtorInitNoArgsSealed = FakeNoCtorInitNoArgsUnsealed.SealType()

@i4t.SealAsProductionType(NoCtor)
class FakeNoCtorSealedAs:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.state = 3

    def Mul100(self, i):
        if False:
            for i in range(10):
                print('nop')
        return self.state * i * 102

def ProductionCodePassNoCtor(obj: NoCtor):
    if False:
        while True:
            i = 10
    return obj.Mul100(2)

class WithCtor:

    def __init__(self, state):
        if False:
            while True:
                i = 10
        self.state = state

    def Mul100(self, i):
        if False:
            while True:
                i = 10
        return self.state * i * 100

class FakeWithCtor(WithCtor, i4t.ProductionType[WithCtor]):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.state = 5

def ProductionCodePassWithCtor(obj: WithCtor):
    if False:
        while True:
            i = 10
    return obj.Mul100(7)

class InstrumentationForTestingTest(unittest.TestCase):

    def testFakeNoCtor(self):
        if False:
            i = 10
            return i + 15
        orig_fake_obj = FakeNoCtor(3)
        obj = orig_fake_obj.Seal()
        assert_type(obj, NoCtor)
        for expected_call_count in (1, 2):
            self.assertEqual(ProductionCodePassNoCtor(obj), 600)
            fake_obj = i4t.Unseal(obj, FakeNoCtor)
            assert fake_obj is orig_fake_obj
            assert_type(fake_obj, FakeNoCtor)
            self.assertEqual(fake_obj.call_count, expected_call_count)

    def testFakeNoCtorInitArg(self):
        if False:
            return 10
        obj = FakeNoCtorInitArgUnsealed(5).Seal()
        assert_type(obj, NoCtor)
        self.assertEqual(ProductionCodePassNoCtor(obj), 1030)
        fake_obj = i4t.Unseal(obj, FakeNoCtorInitArgUnsealed)
        assert_type(fake_obj, FakeNoCtorInitArgUnsealed)
        self.assertEqual(fake_obj.state, 5)

    def testFakeNoCtorDefaultInit(self):
        if False:
            for i in range(10):
                print('nop')
        obj = FakeNoCtorDefaultInitSealed()
        assert_type(obj, NoCtor)
        self.assertEqual(ProductionCodePassNoCtor(obj), 208)
        fake_obj = i4t.Unseal(obj, FakeNoCtorDefaultInitUnsealed)
        assert_type(fake_obj, FakeNoCtorDefaultInitUnsealed)

    def testFakeNoCtorInitNoArgs(self):
        if False:
            return 10
        obj = FakeNoCtorInitNoArgsSealed()
        assert_type(obj, NoCtor)
        self.assertEqual(ProductionCodePassNoCtor(obj), 1680)
        fake_obj = i4t.Unseal(obj, FakeNoCtorInitNoArgsUnsealed)
        assert_type(fake_obj, FakeNoCtorInitNoArgsUnsealed)
        self.assertEqual(fake_obj.state, 8)

    def testFakeNoCtorSealedAs(self):
        if False:
            return 10
        obj = FakeNoCtorSealedAs()
        assert_type(obj, NoCtor)
        self.assertEqual(ProductionCodePassNoCtor(obj), 612)

    def testFakeWithCtor(self):
        if False:
            print('Hello World!')
        orig_fake_obj = FakeWithCtor()
        obj = orig_fake_obj.Seal()
        assert_type(obj, WithCtor)
        self.assertEqual(ProductionCodePassWithCtor(obj), 3500)
        fake_obj = i4t.Unseal(obj, FakeWithCtor)
        assert fake_obj is orig_fake_obj
        assert_type(fake_obj, FakeWithCtor)
if __name__ == '__main__':
    unittest.main()