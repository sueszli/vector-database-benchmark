import unittest
import numpy as np
from numba import njit
from numba.core import types
from numba.experimental import structref
from numba.tests.support import skip_unless_scipy

@structref.register
class MyStructType(types.StructRef):

    def preprocess_fields(self, fields):
        if False:
            for i in range(10):
                print('nop')
        return tuple(((name, types.unliteral(typ)) for (name, typ) in fields))

class MyStruct(structref.StructRefProxy):

    def __new__(cls, name, vector):
        if False:
            print('Hello World!')
        return structref.StructRefProxy.__new__(cls, name, vector)

    @property
    def name(self):
        if False:
            return 10
        return MyStruct_get_name(self)

    @property
    def vector(self):
        if False:
            i = 10
            return i + 15
        return MyStruct_get_vector(self)

@njit
def MyStruct_get_name(self):
    if False:
        print('Hello World!')
    return self.name

@njit
def MyStruct_get_vector(self):
    if False:
        i = 10
        return i + 15
    return self.vector
structref.define_proxy(MyStruct, MyStructType, ['name', 'vector'])

@skip_unless_scipy
class TestStructRefUsage(unittest.TestCase):

    def test_type_definition(self):
        if False:
            return 10
        np.random.seed(0)
        buf = []

        def print(*args):
            if False:
                return 10
            buf.append(args)
        alice = MyStruct('Alice', vector=np.random.random(3))

        @njit
        def make_bob():
            if False:
                while True:
                    i = 10
            bob = MyStruct('unnamed', vector=np.zeros(3))
            bob.name = 'Bob'
            bob.vector = np.random.random(3)
            return bob
        bob = make_bob()
        print(f'{alice.name}: {alice.vector}')
        print(f'{bob.name}: {bob.vector}')

        @njit
        def distance(a, b):
            if False:
                return 10
            return np.linalg.norm(a.vector - b.vector)
        print(distance(alice, bob))
        self.assertEqual(len(buf), 3)

    def test_overload_method(self):
        if False:
            i = 10
            return i + 15
        from numba.core.extending import overload_method
        from numba.core.errors import TypingError

        @overload_method(MyStructType, 'distance')
        def ol_distance(self, other):
            if False:
                return 10
            if not isinstance(other, MyStructType):
                raise TypingError(f'*other* must be a {MyStructType}; got {other}')

            def impl(self, other):
                if False:
                    while True:
                        i = 10
                return np.linalg.norm(self.vector - other.vector)
            return impl

        @njit
        def test():
            if False:
                print('Hello World!')
            alice = MyStruct('Alice', vector=np.random.random(3))
            bob = MyStruct('Bob', vector=np.random.random(3))
            return alice.distance(bob)
        self.assertIsInstance(test(), float)