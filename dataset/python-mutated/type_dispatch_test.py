"""Tests for type_dispatch."""
from typing import Optional
from tensorflow.core.function.polymorphism import function_type
from tensorflow.core.function.polymorphism import type_dispatch
from tensorflow.python.platform import test
from tensorflow.python.types import trace

class MockShape(trace.TraceType):

    def __init__(self, *shape: Optional[int]):
        if False:
            while True:
                i = 10
        self.shape = shape

    def is_subtype_of(self, other: 'MockShape') -> bool:
        if False:
            i = 10
            return i + 15
        if len(self.shape) != len(other.shape):
            return False
        return all((o is None or s == o for (s, o) in zip(self.shape, other.shape)))

    def most_specific_common_supertype(self, others):
        if False:
            while True:
                i = 10
        if any((len(other.shape) != len(self.shape) for other in others)):
            return None
        dims = [dim if all((dim == other.shape[i] for other in others)) else None for (i, dim) in enumerate(self.shape)]
        return MockShape(*dims)

    def placeholder_value(self, placeholder_context=None):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return str(self.shape)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self)

    def __hash__(self) -> int:
        if False:
            while True:
                i = 10
        return hash(self.shape)

    def __eq__(self, other: 'MockShape') -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.shape == other.shape

def make_shape_function_type(*shape):
    if False:
        return 10
    return function_type.FunctionType([function_type.Parameter('x', function_type.Parameter.POSITIONAL_ONLY, False, MockShape(*shape))])

class TypeDispatchTableTest(test.TestCase):

    def testVertical(self):
        if False:
            print('Hello World!')
        table = type_dispatch.TypeDispatchTable()
        table.add_target(make_shape_function_type(None, None, None))
        table.add_target(make_shape_function_type(None, None, 1))
        table.add_target(make_shape_function_type(None, 1, 1))
        table.add_target(make_shape_function_type(1, 1, 1))
        self.assertEqual(list(table.targets), [make_shape_function_type(None, None, None), make_shape_function_type(None, None, 1), make_shape_function_type(None, 1, 1), make_shape_function_type(1, 1, 1)])

    def testHorizontal(self):
        if False:
            return 10
        table = type_dispatch.TypeDispatchTable()
        table.add_target(make_shape_function_type(1))
        table.add_target(make_shape_function_type(1, 2))
        table.add_target(make_shape_function_type(1, 2, 3))
        self.assertEqual(list(table.targets), [make_shape_function_type(1), make_shape_function_type(1, 2), make_shape_function_type(1, 2, 3)])

    def testDuplicateNodes(self):
        if False:
            while True:
                i = 10
        table = type_dispatch.TypeDispatchTable()
        table.add_target(make_shape_function_type(None, None))
        table.add_target(make_shape_function_type(1, None))
        table.add_target(make_shape_function_type(None, 2))
        table.add_target(make_shape_function_type(None, None))
        self.assertEqual(list(table.targets), [make_shape_function_type(None, None), make_shape_function_type(1, None), make_shape_function_type(None, 2)])

    def testDeletion(self):
        if False:
            while True:
                i = 10
        table = type_dispatch.TypeDispatchTable()
        table.add_target(make_shape_function_type(None, None))
        table.add_target(make_shape_function_type(None, 1))
        table.add_target(make_shape_function_type(None, 2))
        self.assertEqual(list(table.targets), [make_shape_function_type(None, None), make_shape_function_type(None, 1), make_shape_function_type(None, 2)])
        table.delete(make_shape_function_type(None, 2))
        self.assertEqual(list(table.targets), [make_shape_function_type(None, None), make_shape_function_type(None, 1)])
        table.delete(make_shape_function_type(None, 2))
        self.assertEqual(list(table.targets), [make_shape_function_type(None, None), make_shape_function_type(None, 1)])

    def testContains(self):
        if False:
            for i in range(10):
                print('nop')
        table = type_dispatch.TypeDispatchTable()
        table.add_target(make_shape_function_type(None, None, None))
        table.add_target(make_shape_function_type(None, 1))
        table.add_target(make_shape_function_type(1, 1))
        table.add_target(make_shape_function_type(None, 2, 1))
        self.assertIn(make_shape_function_type(None, None, None), table.targets)
        self.assertIn(make_shape_function_type(None, 1), table.targets)
        self.assertIn(make_shape_function_type(1, 1), table.targets)
        self.assertIn(make_shape_function_type(None, 2, 1), table.targets)
        self.assertNotIn(make_shape_function_type(None, None, 1), table.targets)
        self.assertNotIn(make_shape_function_type(1, None), table.targets)
        self.assertNotIn(make_shape_function_type(1, 2), table.targets)
        self.assertNotIn(make_shape_function_type(None, 2, None), table.targets)

    def testDispatchExactMatches(self):
        if False:
            while True:
                i = 10
        table = type_dispatch.TypeDispatchTable()
        table.add_target(make_shape_function_type(None, None, None))
        table.add_target(make_shape_function_type(None, 1, None))
        table.add_target(make_shape_function_type(None, 1, 2))
        table.add_target(make_shape_function_type(None, 2, 2))
        self.assertEqual(table.dispatch(make_shape_function_type(None, 1, 2)), make_shape_function_type(None, 1, 2))
        self.assertEqual(table.dispatch(make_shape_function_type(None, 1, None)), make_shape_function_type(None, 1, None))
        self.assertEqual(table.dispatch(make_shape_function_type(None, None, None)), make_shape_function_type(None, None, None))
        self.assertEqual(table.dispatch(make_shape_function_type(None, 2, 2)), make_shape_function_type(None, 2, 2))

    def testDispatchMoreSpecific(self):
        if False:
            for i in range(10):
                print('nop')
        table = type_dispatch.TypeDispatchTable()
        table.add_target(make_shape_function_type(None, None, None))
        table.add_target(make_shape_function_type(None, 1, None))
        table.add_target(make_shape_function_type(None, 1, 2))
        table.add_target(make_shape_function_type(None, 2, 2))
        self.assertEqual(table.dispatch(make_shape_function_type(1, 1, 2)), make_shape_function_type(None, 1, 2))
        self.assertEqual(table.dispatch(make_shape_function_type(1, 1, 3)), make_shape_function_type(None, 1, None))
        self.assertEqual(table.dispatch(make_shape_function_type(1, 3, 3)), make_shape_function_type(None, None, None))
        self.assertEqual(table.dispatch(make_shape_function_type(1, 2, 2)), make_shape_function_type(None, 2, 2))

    def testDispatchNoMatches(self):
        if False:
            print('Hello World!')
        table = type_dispatch.TypeDispatchTable()
        table.add_target(make_shape_function_type(None, 1, None))
        table.add_target(make_shape_function_type(None, 1, 2))
        table.add_target(make_shape_function_type(None, 2, 2))
        self.assertIsNone(table.dispatch(make_shape_function_type(1, 2)))
        self.assertIsNone(table.dispatch(make_shape_function_type(1, 2, 3)))
        self.assertIsNone(table.dispatch(make_shape_function_type(1, 2, 3, 4)))

    def testDispatchCachedAddUpdates(self):
        if False:
            for i in range(10):
                print('nop')
        table = type_dispatch.TypeDispatchTable()
        table.add_target(make_shape_function_type(None, None, None))
        self.assertEqual(table.dispatch(make_shape_function_type(1, 1, 2)), make_shape_function_type(None, None, None))
        table.add_target(make_shape_function_type(None, 1, None))
        self.assertEqual(table.dispatch(make_shape_function_type(1, 1, 2)), make_shape_function_type(None, 1, None))
        table.add_target(make_shape_function_type(None, 1, 2))
        self.assertEqual(table.dispatch(make_shape_function_type(1, 1, 2)), make_shape_function_type(None, 1, 2))
        table.add_target(make_shape_function_type(1, 1, 2))
        self.assertEqual(table.dispatch(make_shape_function_type(1, 1, 2)), make_shape_function_type(1, 1, 2))

    def testDispatchCachedDeleteUpdates(self):
        if False:
            print('Hello World!')
        table = type_dispatch.TypeDispatchTable()
        table.add_target(make_shape_function_type(None, None, None))
        table.add_target(make_shape_function_type(None, 1, None))
        table.add_target(make_shape_function_type(None, 1, 2))
        table.add_target(make_shape_function_type(1, 1, 2))
        self.assertEqual(table.dispatch(make_shape_function_type(1, 1, 2)), make_shape_function_type(1, 1, 2))
        table.delete(make_shape_function_type(1, 1, 2))
        self.assertEqual(table.dispatch(make_shape_function_type(1, 1, 2)), make_shape_function_type(None, 1, 2))
        table.delete(make_shape_function_type(None, 1, 2))
        self.assertEqual(table.dispatch(make_shape_function_type(1, 1, 2)), make_shape_function_type(None, 1, None))
        table.delete(make_shape_function_type(None, 1, None))
        self.assertEqual(table.dispatch(make_shape_function_type(1, 1, 2)), make_shape_function_type(None, None, None))

    def testDispatchCacheOrderingDeterminism(self):
        if False:
            return 10
        table_1 = type_dispatch.TypeDispatchTable()
        table_1.add_target(make_shape_function_type(1, None, None))
        table_1.add_target(make_shape_function_type(None, 2, None))
        table_1.add_target(make_shape_function_type(None, None, 3))
        table_2 = type_dispatch.TypeDispatchTable()
        table_2.add_target(make_shape_function_type(None, 2, None))
        table_2.add_target(make_shape_function_type(1, None, None))
        table_2.add_target(make_shape_function_type(None, None, 3))
        table_3 = type_dispatch.TypeDispatchTable()
        table_3.add_target(make_shape_function_type(None, None, 3))
        table_3.add_target(make_shape_function_type(1, None, None))
        table_3.add_target(make_shape_function_type(None, 2, None))
        self.assertEqual(set(table_1.targets), set(table_2.targets))
        self.assertEqual(set(table_2.targets), set(table_3.targets))
        shape = make_shape_function_type(1, 2, 3)
        self.assertEqual(table_1.dispatch(shape), make_shape_function_type(1, None, None))
        self.assertEqual(table_2.dispatch(shape), make_shape_function_type(None, 2, None))
        self.assertEqual(table_3.dispatch(shape), make_shape_function_type(None, None, 3))

    def testGeneralizedExisting(self):
        if False:
            while True:
                i = 10
        table = type_dispatch.TypeDispatchTable()
        table.add_target(make_shape_function_type(None, None, None))
        table.add_target(make_shape_function_type(None, 1, None))
        table.add_target(make_shape_function_type(None, 1, 2))
        self.assertEqual(table.try_generalizing_function_type(make_shape_function_type(None, 1, 3)), make_shape_function_type(None, None, None))

    def testGeneralizedNovel(self):
        if False:
            i = 10
            return i + 15
        table = type_dispatch.TypeDispatchTable()
        table.add_target(make_shape_function_type(None, 1, None))
        table.add_target(make_shape_function_type(None, 1, 2))
        self.assertEqual(table.try_generalizing_function_type(make_shape_function_type(None, 2, 3)), make_shape_function_type(None, None, None))

    def testGeneralizedUnknown(self):
        if False:
            i = 10
            return i + 15
        table = type_dispatch.TypeDispatchTable()
        table.add_target(make_shape_function_type(None, 1))
        table.add_target(make_shape_function_type(None, 2))
        table.add_target(make_shape_function_type(None, 3))
        self.assertEqual(table.try_generalizing_function_type(make_shape_function_type(None, 4, 3)), make_shape_function_type(None, 4, 3))
if __name__ == '__main__':
    test.main()