"""Unit tests for the ShardedKeyTypeConstraint."""
from apache_beam.typehints import Tuple
from apache_beam.typehints import typehints
from apache_beam.typehints.sharded_key_type import ShardedKeyType
from apache_beam.typehints.typehints_test import TypeHintTestCase
from apache_beam.utils.sharded_key import ShardedKey

class ShardedKeyTypeConstraintTest(TypeHintTestCase):

    def test_compatibility(self):
        if False:
            i = 10
            return i + 15
        constraint1 = ShardedKeyType[int]
        constraint2 = ShardedKeyType[str]
        self.assertCompatible(constraint1, constraint1)
        self.assertCompatible(constraint2, constraint2)
        self.assertNotCompatible(constraint1, constraint2)

    def test_repr(self):
        if False:
            return 10
        constraint = ShardedKeyType[int]
        self.assertEqual("ShardedKey[<class 'int'>]", repr(constraint))

    def test_type_check_not_sharded_key(self):
        if False:
            for i in range(10):
                print('nop')
        constraint = ShardedKeyType[int]
        obj = 5
        with self.assertRaises(TypeError) as e:
            constraint.type_check(obj)
        self.assertEqual("ShardedKey type-constraint violated. Valid object instance must be of type 'ShardedKey'. Instead, an instance of 'int' was received.", e.exception.args[0])

    def test_type_check_invalid_key_type(self):
        if False:
            i = 10
            return i + 15
        constraint = ShardedKeyType[int]
        obj = ShardedKey(key='abc', shard_id=b'123')
        with self.assertRaises((TypeError, TypeError)) as e:
            constraint.type_check(obj)
        self.assertEqual("ShardedKey[<class 'int'>] type-constraint violated. The type of key in 'ShardedKey' is incorrect. Expected an instance of type '<class 'int'>', instead received an instance of type 'str'.", e.exception.args[0])

    def test_type_check_valid_simple_type(self):
        if False:
            print('Hello World!')
        constraint = ShardedKeyType[str]
        obj = ShardedKey(key='abc', shard_id=b'123')
        self.assertIsNone(constraint.type_check(obj))

    def test_type_check_valid_composite_type(self):
        if False:
            return 10
        constraint = ShardedKeyType[Tuple[int, str]]
        obj = ShardedKey(key=(1, 'a'), shard_id=b'123')
        self.assertIsNone(constraint.type_check(obj))

    def test_match_type_variables(self):
        if False:
            for i in range(10):
                print('nop')
        K = typehints.TypeVariable('K')
        constraint = ShardedKeyType[K]
        self.assertEqual({K: int}, constraint.match_type_variables(ShardedKeyType[int]))

    def test_getitem(self):
        if False:
            i = 10
            return i + 15
        K = typehints.TypeVariable('K')
        T = typehints.TypeVariable('T')
        with self.assertRaisesRegex(TypeError, 'Parameter to ShardedKeyType hint.*'):
            _ = ShardedKeyType[K, T]
        with self.assertRaisesRegex(TypeError, 'Parameter to ShardedKeyType hint.*'):
            _ = ShardedKeyType[K, T]