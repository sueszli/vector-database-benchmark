"""Tests for default_types."""
import collections
from tensorflow.core.function.trace_type import default_types
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.platform import test
from tensorflow.python.types import trace

class MockSupertypes2With3(trace.TraceType):

    def __init__(self, obj):
        if False:
            for i in range(10):
                print('nop')
        self._object = obj

    def is_subtype_of(self, other):
        if False:
            i = 10
            return i + 15
        return self._object == 2 and other._object == 3

    def most_specific_common_supertype(self, others):
        if False:
            return 10
        if not others:
            return self
        if self._object == 2 and isinstance(others[0]._object, int):
            return MockSupertypes2With3(3)
        else:
            return None

    def placeholder_value(self, placeholder_context=None):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        if False:
            i = 10
            return i + 15
        return isinstance(other, type(self)) and self._object == other._object

    def __hash__(self) -> int:
        if False:
            return 10
        return self._object_hash

    def __repr__(self) -> str:
        if False:
            return 10
        return 'MockSupertypes2With3'

class Mock2AsTopType(MockSupertypes2With3):

    def is_subtype_of(self, other):
        if False:
            while True:
                i = 10
        return other._object == 2

    def most_specific_common_supertype(self, others):
        if False:
            while True:
                i = 10
        if not all((isinstance(other, Mock2AsTopType) for other in others)):
            return None
        return self if all((self._object == other._object for other in others)) else Mock2AsTopType(2)

    def __repr__(self) -> str:
        if False:
            return 10
        return 'Mock2AsTopType'

class TestAttr:
    """Helps test attrs collections."""

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        self.name = name

class TestAttrsClass:
    """Helps test attrs collections."""
    __attrs_attrs__ = (TestAttr('a'), TestAttr('b'))

    def __init__(self, a, b):
        if False:
            while True:
                i = 10
        self.a = a
        self.b = b

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return isinstance(other, TestAttrsClass) and self.a == other.a and (self.b == other.b)

class DefaultTypesTest(test.TestCase):

    def testLiteralNan(self):
        if False:
            i = 10
            return i + 15
        nan_literal = default_types.Literal(float('nan'))
        complex_nan = default_types.Literal(complex(float('nan'), 1))
        complex_nan_other = default_types.Literal(complex(1, float('nan')))
        self.assertEqual(nan_literal, nan_literal)
        self.assertEqual(nan_literal, complex_nan)
        self.assertEqual(nan_literal, complex_nan_other)

    def testLiteralSupertypes(self):
        if False:
            i = 10
            return i + 15
        literal_a = default_types.Literal(1)
        literal_b = default_types.Literal(2)
        literal_c = default_types.Literal(1)
        self.assertEqual(literal_a, literal_a.most_specific_common_supertype([]))
        self.assertEqual(literal_a, literal_a.most_specific_common_supertype([literal_a]))
        self.assertEqual(literal_a, literal_a.most_specific_common_supertype([literal_c]))
        self.assertIsNone(literal_a.most_specific_common_supertype([literal_b]))

    def testLiteralSerialization(self):
        if False:
            i = 10
            return i + 15
        literal_bool = default_types.Literal(True)
        literal_int = default_types.Literal(1)
        literal_float = default_types.Literal(1.2)
        literal_str = default_types.Literal('a')
        literal_none = default_types.Literal(None)
        self.assertEqual(str(literal_bool), 'Literal[True]')
        self.assertEqual(serialization.deserialize(serialization.serialize(literal_bool)), literal_bool)
        self.assertEqual(serialization.deserialize(serialization.serialize(literal_int)), literal_int)
        self.assertEqual(serialization.deserialize(serialization.serialize(literal_float)), literal_float)
        self.assertEqual(serialization.deserialize(serialization.serialize(literal_str)), literal_str)
        self.assertEqual(serialization.deserialize(serialization.serialize(literal_none)), literal_none)

    def testListSupertype(self):
        if False:
            return 10
        list_a = default_types.List(MockSupertypes2With3(1), MockSupertypes2With3(2), MockSupertypes2With3(3))
        list_b = default_types.List(MockSupertypes2With3(2), MockSupertypes2With3(2), MockSupertypes2With3(2))
        self.assertEqual(list_a, list_a.most_specific_common_supertype([]))
        self.assertIsNone(list_a.most_specific_common_supertype([list_b]))
        self.assertEqual(list_b.most_specific_common_supertype([list_a]), default_types.List(MockSupertypes2With3(3), MockSupertypes2With3(3), MockSupertypes2With3(3)))

    def testListSerialization(self):
        if False:
            print('Hello World!')
        list_original = default_types.List(default_types.Literal(1), default_types.Literal(2), default_types.Literal(3))
        self.assertEqual(str(list_original), 'List[Literal[1], Literal[2], Literal[3]]')
        self.assertEqual(serialization.deserialize(serialization.serialize(list_original)), list_original)

    def testTupleSupertype(self):
        if False:
            i = 10
            return i + 15
        tuple_a = default_types.Tuple(MockSupertypes2With3(1), MockSupertypes2With3(2), MockSupertypes2With3(3))
        tuple_b = default_types.Tuple(MockSupertypes2With3(2), MockSupertypes2With3(2), MockSupertypes2With3(2))
        self.assertEqual(tuple_a, tuple_a.most_specific_common_supertype([]))
        self.assertIsNone(tuple_a.most_specific_common_supertype([tuple_b]))
        self.assertEqual(tuple_b.most_specific_common_supertype([tuple_a]), default_types.Tuple(MockSupertypes2With3(3), MockSupertypes2With3(3), MockSupertypes2With3(3)))

    def testTupleSerialization(self):
        if False:
            for i in range(10):
                print('nop')
        tuple_original = default_types.Tuple(default_types.Literal(1), default_types.Literal(2), default_types.Literal(3))
        self.assertEqual(str(tuple_original), 'Tuple[Literal[1], Literal[2], Literal[3]]')
        self.assertEqual(serialization.deserialize(serialization.serialize(tuple_original)), tuple_original)

    def testNamedTupleSupertype(self):
        if False:
            return 10
        named_tuple_type = collections.namedtuple('MyNamedTuple', 'x y z')
        tuple_a = default_types.NamedTuple.from_type_and_attributes(named_tuple_type, (MockSupertypes2With3(1), MockSupertypes2With3(2), MockSupertypes2With3(3)))
        tuple_b = default_types.NamedTuple.from_type_and_attributes(named_tuple_type, (MockSupertypes2With3(2), MockSupertypes2With3(2), MockSupertypes2With3(2)))
        self.assertEqual(str(tuple_a), "MyNamedTuple[['x', MockSupertypes2With3], ['y', MockSupertypes2With3], ['z', MockSupertypes2With3]]")
        self.assertEqual(tuple_a, tuple_a.most_specific_common_supertype([]))
        self.assertIsNone(tuple_a.most_specific_common_supertype([tuple_b]))
        self.assertEqual(tuple_b.most_specific_common_supertype([tuple_a]), default_types.NamedTuple.from_type_and_attributes(named_tuple_type, (MockSupertypes2With3(3), MockSupertypes2With3(3), MockSupertypes2With3(3))))

    def testAttrsSupertype(self):
        if False:
            for i in range(10):
                print('nop')
        attrs_a = default_types.Attrs.from_type_and_attributes(TestAttrsClass, (MockSupertypes2With3(1), MockSupertypes2With3(2), MockSupertypes2With3(3)))
        attrs_b = default_types.Attrs.from_type_and_attributes(TestAttrsClass, (MockSupertypes2With3(2), MockSupertypes2With3(2), MockSupertypes2With3(2)))
        self.assertEqual(str(attrs_a), "TestAttrsClass[['a', MockSupertypes2With3], ['b', MockSupertypes2With3]]")
        self.assertEqual(attrs_a, attrs_a.most_specific_common_supertype([]))
        self.assertIsNone(attrs_a.most_specific_common_supertype([attrs_b]))
        self.assertEqual(attrs_b.most_specific_common_supertype([attrs_a]), default_types.Attrs.from_type_and_attributes(TestAttrsClass, (MockSupertypes2With3(3), MockSupertypes2With3(3), MockSupertypes2With3(3))))

    def testDictTypeSubtype(self):
        if False:
            while True:
                i = 10
        dict_type = default_types.Dict
        dict_a = dict_type({'a': Mock2AsTopType(1), 'b': Mock2AsTopType(1), 'c': Mock2AsTopType(1)})
        dict_b = dict_type({'a': Mock2AsTopType(2), 'b': Mock2AsTopType(2), 'c': Mock2AsTopType(2)})
        dict_c = dict_type({'a': Mock2AsTopType(1), 'b': Mock2AsTopType(1)})
        self.assertTrue(dict_a.is_subtype_of(dict_b))
        self.assertFalse(dict_c.is_subtype_of(dict_b))
        self.assertFalse(dict_c.is_subtype_of(dict_a))

    def testDictTypeSupertype(self):
        if False:
            while True:
                i = 10
        dict_type = default_types.Dict
        dict_a = dict_type({'a': MockSupertypes2With3(1), 'b': MockSupertypes2With3(2), 'c': MockSupertypes2With3(3)})
        dict_b = dict_type({'a': MockSupertypes2With3(2), 'b': MockSupertypes2With3(2), 'c': MockSupertypes2With3(2)})
        self.assertEqual(dict_a, dict_a.most_specific_common_supertype([]))
        self.assertIsNone(dict_a.most_specific_common_supertype([dict_b]))
        self.assertEqual(dict_b.most_specific_common_supertype([dict_a]), dict_type({'a': MockSupertypes2With3(3), 'b': MockSupertypes2With3(3), 'c': MockSupertypes2With3(3)}))

    def testDictSerialization(self):
        if False:
            return 10
        dict_original = default_types.Dict({'a': default_types.Literal(1), 'b': default_types.Literal(2), 'c': default_types.Literal(3)})
        self.assertEqual(str(dict_original), "Dict[['a', Literal[1]], ['b', Literal[2]], ['c', Literal[3]]]")
        self.assertEqual(serialization.deserialize(serialization.serialize(dict_original)), dict_original)

    def testListTupleInequality(self):
        if False:
            i = 10
            return i + 15
        literal = default_types.Literal
        list_a = default_types.List(literal(1), literal(2), literal(3))
        list_b = default_types.List(literal(1), literal(2), literal(3))
        tuple_a = default_types.Tuple(literal(1), literal(2), literal(3))
        tuple_b = default_types.Tuple(literal(1), literal(2), literal(3))
        self.assertEqual(list_a, list_b)
        self.assertEqual(tuple_a, tuple_b)
        self.assertNotEqual(list_a, tuple_a)
        self.assertNotEqual(tuple_a, list_a)

    def testDictTypeEquality(self):
        if False:
            return 10
        dict_type = default_types.Dict
        literal = default_types.Literal
        dict_a = dict_type({literal(1): literal(2), literal(3): literal(4)})
        dict_b = dict_type({literal(1): literal(2)})
        dict_c = dict_type({literal(3): literal(4), literal(1): literal(2)})
        self.assertEqual(dict_a, dict_c)
        self.assertNotEqual(dict_a, dict_b)

    def testCastLazy(self):
        if False:
            for i in range(10):
                print('nop')
        list_type = default_types.List(default_types.Literal('a'), default_types.Literal('b'))
        tuple_type = default_types.Tuple(default_types.Literal('c'), list_type)
        dict_type = default_types.Dict({'key': tuple_type, 'other_key': list_type}, placeholder_type=dict)
        value = dict_type.placeholder_value(None)
        casted_value = dict_type.cast(value, None)
        self.assertIs(value, casted_value)
if __name__ == '__main__':
    test.main()