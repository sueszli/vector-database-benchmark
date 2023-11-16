"""Unittest for reflection.py, which also indirectly tests the output of the
pure-Python protocol compiler.
"""
import copy
import gc
import operator
import six
import struct
try:
    import unittest2 as unittest
except ImportError:
    import unittest
from google.protobuf import unittest_import_pb2
from google.protobuf import unittest_mset_pb2
from google.protobuf import unittest_pb2
from google.protobuf import descriptor_pb2
from google.protobuf import descriptor
from google.protobuf import message
from google.protobuf import reflection
from google.protobuf import text_format
from google.protobuf.internal import api_implementation
from google.protobuf.internal import more_extensions_pb2
from google.protobuf.internal import more_messages_pb2
from google.protobuf.internal import message_set_extensions_pb2
from google.protobuf.internal import wire_format
from google.protobuf.internal import test_util
from google.protobuf.internal import testing_refleaks
from google.protobuf.internal import decoder
BaseTestCase = testing_refleaks.BaseTestCase

class _MiniDecoder(object):
    """Decodes a stream of values from a string.

  Once upon a time we actually had a class called decoder.Decoder.  Then we
  got rid of it during a redesign that made decoding much, much faster overall.
  But a couple tests in this file used it to check that the serialized form of
  a message was correct.  So, this class implements just the methods that were
  used by said tests, so that we don't have to rewrite the tests.
  """

    def __init__(self, bytes):
        if False:
            return 10
        self._bytes = bytes
        self._pos = 0

    def ReadVarint(self):
        if False:
            for i in range(10):
                print('nop')
        (result, self._pos) = decoder._DecodeVarint(self._bytes, self._pos)
        return result
    ReadInt32 = ReadVarint
    ReadInt64 = ReadVarint
    ReadUInt32 = ReadVarint
    ReadUInt64 = ReadVarint

    def ReadSInt64(self):
        if False:
            print('Hello World!')
        return wire_format.ZigZagDecode(self.ReadVarint())
    ReadSInt32 = ReadSInt64

    def ReadFieldNumberAndWireType(self):
        if False:
            print('Hello World!')
        return wire_format.UnpackTag(self.ReadVarint())

    def ReadFloat(self):
        if False:
            print('Hello World!')
        result = struct.unpack('<f', self._bytes[self._pos:self._pos + 4])[0]
        self._pos += 4
        return result

    def ReadDouble(self):
        if False:
            return 10
        result = struct.unpack('<d', self._bytes[self._pos:self._pos + 8])[0]
        self._pos += 8
        return result

    def EndOfStream(self):
        if False:
            print('Hello World!')
        return self._pos == len(self._bytes)

class ReflectionTest(BaseTestCase):

    def assertListsEqual(self, values, others):
        if False:
            while True:
                i = 10
        self.assertEqual(len(values), len(others))
        for i in range(len(values)):
            self.assertEqual(values[i], others[i])

    def testScalarConstructor(self):
        if False:
            for i in range(10):
                print('nop')
        proto = unittest_pb2.TestAllTypes(optional_int32=24, optional_double=54.321, optional_string='optional_string', optional_float=None)
        self.assertEqual(24, proto.optional_int32)
        self.assertEqual(54.321, proto.optional_double)
        self.assertEqual('optional_string', proto.optional_string)
        self.assertFalse(proto.HasField('optional_float'))

    def testRepeatedScalarConstructor(self):
        if False:
            for i in range(10):
                print('nop')
        proto = unittest_pb2.TestAllTypes(repeated_int32=[1, 2, 3, 4], repeated_double=[1.23, 54.321], repeated_bool=[True, False, False], repeated_string=['optional_string'], repeated_float=None)
        self.assertEqual([1, 2, 3, 4], list(proto.repeated_int32))
        self.assertEqual([1.23, 54.321], list(proto.repeated_double))
        self.assertEqual([True, False, False], list(proto.repeated_bool))
        self.assertEqual(['optional_string'], list(proto.repeated_string))
        self.assertEqual([], list(proto.repeated_float))

    def testRepeatedCompositeConstructor(self):
        if False:
            for i in range(10):
                print('nop')
        proto = unittest_pb2.TestAllTypes(repeated_nested_message=[unittest_pb2.TestAllTypes.NestedMessage(bb=unittest_pb2.TestAllTypes.FOO), unittest_pb2.TestAllTypes.NestedMessage(bb=unittest_pb2.TestAllTypes.BAR)], repeated_foreign_message=[unittest_pb2.ForeignMessage(c=-43), unittest_pb2.ForeignMessage(c=45324), unittest_pb2.ForeignMessage(c=12)], repeatedgroup=[unittest_pb2.TestAllTypes.RepeatedGroup(), unittest_pb2.TestAllTypes.RepeatedGroup(a=1), unittest_pb2.TestAllTypes.RepeatedGroup(a=2)])
        self.assertEqual([unittest_pb2.TestAllTypes.NestedMessage(bb=unittest_pb2.TestAllTypes.FOO), unittest_pb2.TestAllTypes.NestedMessage(bb=unittest_pb2.TestAllTypes.BAR)], list(proto.repeated_nested_message))
        self.assertEqual([unittest_pb2.ForeignMessage(c=-43), unittest_pb2.ForeignMessage(c=45324), unittest_pb2.ForeignMessage(c=12)], list(proto.repeated_foreign_message))
        self.assertEqual([unittest_pb2.TestAllTypes.RepeatedGroup(), unittest_pb2.TestAllTypes.RepeatedGroup(a=1), unittest_pb2.TestAllTypes.RepeatedGroup(a=2)], list(proto.repeatedgroup))

    def testMixedConstructor(self):
        if False:
            i = 10
            return i + 15
        proto = unittest_pb2.TestAllTypes(optional_int32=24, optional_string='optional_string', repeated_double=[1.23, 54.321], repeated_bool=[True, False, False], repeated_nested_message=[unittest_pb2.TestAllTypes.NestedMessage(bb=unittest_pb2.TestAllTypes.FOO), unittest_pb2.TestAllTypes.NestedMessage(bb=unittest_pb2.TestAllTypes.BAR)], repeated_foreign_message=[unittest_pb2.ForeignMessage(c=-43), unittest_pb2.ForeignMessage(c=45324), unittest_pb2.ForeignMessage(c=12)], optional_nested_message=None)
        self.assertEqual(24, proto.optional_int32)
        self.assertEqual('optional_string', proto.optional_string)
        self.assertEqual([1.23, 54.321], list(proto.repeated_double))
        self.assertEqual([True, False, False], list(proto.repeated_bool))
        self.assertEqual([unittest_pb2.TestAllTypes.NestedMessage(bb=unittest_pb2.TestAllTypes.FOO), unittest_pb2.TestAllTypes.NestedMessage(bb=unittest_pb2.TestAllTypes.BAR)], list(proto.repeated_nested_message))
        self.assertEqual([unittest_pb2.ForeignMessage(c=-43), unittest_pb2.ForeignMessage(c=45324), unittest_pb2.ForeignMessage(c=12)], list(proto.repeated_foreign_message))
        self.assertFalse(proto.HasField('optional_nested_message'))

    def testConstructorTypeError(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, unittest_pb2.TestAllTypes, optional_int32='foo')
        self.assertRaises(TypeError, unittest_pb2.TestAllTypes, optional_string=1234)
        self.assertRaises(TypeError, unittest_pb2.TestAllTypes, optional_nested_message=1234)
        self.assertRaises(TypeError, unittest_pb2.TestAllTypes, repeated_int32=1234)
        self.assertRaises(TypeError, unittest_pb2.TestAllTypes, repeated_int32=['foo'])
        self.assertRaises(TypeError, unittest_pb2.TestAllTypes, repeated_string=1234)
        self.assertRaises(TypeError, unittest_pb2.TestAllTypes, repeated_string=[1234])
        self.assertRaises(TypeError, unittest_pb2.TestAllTypes, repeated_nested_message=1234)
        self.assertRaises(TypeError, unittest_pb2.TestAllTypes, repeated_nested_message=[1234])

    def testConstructorInvalidatesCachedByteSize(self):
        if False:
            while True:
                i = 10
        message = unittest_pb2.TestAllTypes(optional_int32=12)
        self.assertEqual(2, message.ByteSize())
        message = unittest_pb2.TestAllTypes(optional_nested_message=unittest_pb2.TestAllTypes.NestedMessage())
        self.assertEqual(3, message.ByteSize())
        message = unittest_pb2.TestAllTypes(repeated_int32=[12])
        self.assertEqual(3, message.ByteSize())
        message = unittest_pb2.TestAllTypes(repeated_nested_message=[unittest_pb2.TestAllTypes.NestedMessage()])
        self.assertEqual(3, message.ByteSize())

    def testSimpleHasBits(self):
        if False:
            return 10
        proto = unittest_pb2.TestAllTypes()
        self.assertTrue(not proto.HasField('optional_int32'))
        self.assertEqual(0, proto.optional_int32)
        self.assertTrue(not proto.HasField('optional_int32'))
        proto.optional_int32 = 1
        self.assertTrue(proto.HasField('optional_int32'))
        proto.ClearField('optional_int32')
        self.assertTrue(not proto.HasField('optional_int32'))

    def testHasBitsWithSinglyNestedScalar(self):
        if False:
            while True:
                i = 10

        def TestCompositeHasBits(composite_field_name, scalar_field_name):
            if False:
                print('Hello World!')
            proto = unittest_pb2.TestAllTypes()
            composite_field = getattr(proto, composite_field_name)
            original_scalar_value = getattr(composite_field, scalar_field_name)
            self.assertEqual(0, original_scalar_value)
            self.assertTrue(not composite_field.HasField(scalar_field_name))
            self.assertTrue(not proto.HasField(composite_field_name))
            new_val = 20
            setattr(composite_field, scalar_field_name, new_val)
            self.assertEqual(new_val, getattr(composite_field, scalar_field_name))
            old_composite_field = composite_field
            self.assertTrue(composite_field.HasField(scalar_field_name))
            self.assertTrue(proto.HasField(composite_field_name))
            proto.ClearField(composite_field_name)
            composite_field = getattr(proto, composite_field_name)
            self.assertTrue(not composite_field.HasField(scalar_field_name))
            self.assertTrue(not proto.HasField(composite_field_name))
            self.assertEqual(0, getattr(composite_field, scalar_field_name))
            self.assertTrue(old_composite_field is not composite_field)
            setattr(old_composite_field, scalar_field_name, new_val)
            self.assertTrue(not composite_field.HasField(scalar_field_name))
            self.assertTrue(not proto.HasField(composite_field_name))
            self.assertEqual(0, getattr(composite_field, scalar_field_name))
        TestCompositeHasBits('optionalgroup', 'a')
        TestCompositeHasBits('optional_nested_message', 'bb')
        TestCompositeHasBits('optional_foreign_message', 'c')
        TestCompositeHasBits('optional_import_message', 'd')

    def testReferencesToNestedMessage(self):
        if False:
            return 10
        proto = unittest_pb2.TestAllTypes()
        nested = proto.optional_nested_message
        del proto
        nested.bb = 23

    def testDisconnectingNestedMessageBeforeSettingField(self):
        if False:
            for i in range(10):
                print('nop')
        proto = unittest_pb2.TestAllTypes()
        nested = proto.optional_nested_message
        proto.ClearField('optional_nested_message')
        self.assertTrue(nested is not proto.optional_nested_message)
        nested.bb = 23
        self.assertTrue(not proto.HasField('optional_nested_message'))
        self.assertEqual(0, proto.optional_nested_message.bb)

    def testGetDefaultMessageAfterDisconnectingDefaultMessage(self):
        if False:
            i = 10
            return i + 15
        proto = unittest_pb2.TestAllTypes()
        nested = proto.optional_nested_message
        proto.ClearField('optional_nested_message')
        del proto
        del nested
        gc.collect()
        proto = unittest_pb2.TestAllTypes()
        nested = proto.optional_nested_message

    def testDisconnectingNestedMessageAfterSettingField(self):
        if False:
            while True:
                i = 10
        proto = unittest_pb2.TestAllTypes()
        nested = proto.optional_nested_message
        nested.bb = 5
        self.assertTrue(proto.HasField('optional_nested_message'))
        proto.ClearField('optional_nested_message')
        self.assertEqual(5, nested.bb)
        self.assertEqual(0, proto.optional_nested_message.bb)
        self.assertTrue(nested is not proto.optional_nested_message)
        nested.bb = 23
        self.assertTrue(not proto.HasField('optional_nested_message'))
        self.assertEqual(0, proto.optional_nested_message.bb)

    def testDisconnectingNestedMessageBeforeGettingField(self):
        if False:
            print('Hello World!')
        proto = unittest_pb2.TestAllTypes()
        self.assertTrue(not proto.HasField('optional_nested_message'))
        proto.ClearField('optional_nested_message')
        self.assertTrue(not proto.HasField('optional_nested_message'))

    def testDisconnectingNestedMessageAfterMerge(self):
        if False:
            return 10
        proto1 = unittest_pb2.TestAllTypes()
        proto2 = unittest_pb2.TestAllTypes()
        proto2.optional_nested_message.bb = 5
        proto1.MergeFrom(proto2)
        self.assertTrue(proto1.HasField('optional_nested_message'))
        proto1.ClearField('optional_nested_message')
        self.assertTrue(not proto1.HasField('optional_nested_message'))

    def testDisconnectingLazyNestedMessage(self):
        if False:
            i = 10
            return i + 15
        if api_implementation.Type() != 'python':
            return
        proto = unittest_pb2.TestAllTypes()
        proto.optional_lazy_message.bb = 5
        proto.ClearField('optional_lazy_message')
        del proto
        gc.collect()

    def testHasBitsWhenModifyingRepeatedFields(self):
        if False:
            return 10
        proto = unittest_pb2.TestNestedMessageHasBits()
        proto.optional_nested_message.nestedmessage_repeated_int32.append(5)
        self.assertEqual([5], proto.optional_nested_message.nestedmessage_repeated_int32)
        self.assertTrue(proto.HasField('optional_nested_message'))
        proto.ClearField('optional_nested_message')
        self.assertTrue(not proto.HasField('optional_nested_message'))
        proto.optional_nested_message.nestedmessage_repeated_foreignmessage.add()
        self.assertTrue(proto.HasField('optional_nested_message'))

    def testHasBitsForManyLevelsOfNesting(self):
        if False:
            while True:
                i = 10
        recursive_proto = unittest_pb2.TestMutualRecursionA()
        self.assertTrue(not recursive_proto.HasField('bb'))
        self.assertEqual(0, recursive_proto.bb.a.bb.a.bb.optional_int32)
        self.assertTrue(not recursive_proto.HasField('bb'))
        recursive_proto.bb.a.bb.a.bb.optional_int32 = 5
        self.assertEqual(5, recursive_proto.bb.a.bb.a.bb.optional_int32)
        self.assertTrue(recursive_proto.HasField('bb'))
        self.assertTrue(recursive_proto.bb.HasField('a'))
        self.assertTrue(recursive_proto.bb.a.HasField('bb'))
        self.assertTrue(recursive_proto.bb.a.bb.HasField('a'))
        self.assertTrue(recursive_proto.bb.a.bb.a.HasField('bb'))
        self.assertTrue(not recursive_proto.bb.a.bb.a.bb.HasField('a'))
        self.assertTrue(recursive_proto.bb.a.bb.a.bb.HasField('optional_int32'))

    def testSingularListFields(self):
        if False:
            while True:
                i = 10
        proto = unittest_pb2.TestAllTypes()
        proto.optional_fixed32 = 1
        proto.optional_int32 = 5
        proto.optional_string = 'foo'
        nested_message = proto.optional_nested_message
        self.assertEqual([(proto.DESCRIPTOR.fields_by_name['optional_int32'], 5), (proto.DESCRIPTOR.fields_by_name['optional_fixed32'], 1), (proto.DESCRIPTOR.fields_by_name['optional_string'], 'foo')], proto.ListFields())
        proto.optional_nested_message.bb = 123
        self.assertEqual([(proto.DESCRIPTOR.fields_by_name['optional_int32'], 5), (proto.DESCRIPTOR.fields_by_name['optional_fixed32'], 1), (proto.DESCRIPTOR.fields_by_name['optional_string'], 'foo'), (proto.DESCRIPTOR.fields_by_name['optional_nested_message'], nested_message)], proto.ListFields())

    def testRepeatedListFields(self):
        if False:
            i = 10
            return i + 15
        proto = unittest_pb2.TestAllTypes()
        proto.repeated_fixed32.append(1)
        proto.repeated_int32.append(5)
        proto.repeated_int32.append(11)
        proto.repeated_string.extend(['foo', 'bar'])
        proto.repeated_string.extend([])
        proto.repeated_string.append('baz')
        proto.repeated_string.extend((str(x) for x in range(2)))
        proto.optional_int32 = 21
        proto.repeated_bool
        self.assertEqual([(proto.DESCRIPTOR.fields_by_name['optional_int32'], 21), (proto.DESCRIPTOR.fields_by_name['repeated_int32'], [5, 11]), (proto.DESCRIPTOR.fields_by_name['repeated_fixed32'], [1]), (proto.DESCRIPTOR.fields_by_name['repeated_string'], ['foo', 'bar', 'baz', '0', '1'])], proto.ListFields())

    def testSingularListExtensions(self):
        if False:
            for i in range(10):
                print('nop')
        proto = unittest_pb2.TestAllExtensions()
        proto.Extensions[unittest_pb2.optional_fixed32_extension] = 1
        proto.Extensions[unittest_pb2.optional_int32_extension] = 5
        proto.Extensions[unittest_pb2.optional_string_extension] = 'foo'
        self.assertEqual([(unittest_pb2.optional_int32_extension, 5), (unittest_pb2.optional_fixed32_extension, 1), (unittest_pb2.optional_string_extension, 'foo')], proto.ListFields())

    def testRepeatedListExtensions(self):
        if False:
            return 10
        proto = unittest_pb2.TestAllExtensions()
        proto.Extensions[unittest_pb2.repeated_fixed32_extension].append(1)
        proto.Extensions[unittest_pb2.repeated_int32_extension].append(5)
        proto.Extensions[unittest_pb2.repeated_int32_extension].append(11)
        proto.Extensions[unittest_pb2.repeated_string_extension].append('foo')
        proto.Extensions[unittest_pb2.repeated_string_extension].append('bar')
        proto.Extensions[unittest_pb2.repeated_string_extension].append('baz')
        proto.Extensions[unittest_pb2.optional_int32_extension] = 21
        self.assertEqual([(unittest_pb2.optional_int32_extension, 21), (unittest_pb2.repeated_int32_extension, [5, 11]), (unittest_pb2.repeated_fixed32_extension, [1]), (unittest_pb2.repeated_string_extension, ['foo', 'bar', 'baz'])], proto.ListFields())

    def testListFieldsAndExtensions(self):
        if False:
            i = 10
            return i + 15
        proto = unittest_pb2.TestFieldOrderings()
        test_util.SetAllFieldsAndExtensions(proto)
        unittest_pb2.my_extension_int
        self.assertEqual([(proto.DESCRIPTOR.fields_by_name['my_int'], 1), (unittest_pb2.my_extension_int, 23), (proto.DESCRIPTOR.fields_by_name['my_string'], 'foo'), (unittest_pb2.my_extension_string, 'bar'), (proto.DESCRIPTOR.fields_by_name['my_float'], 1.0)], proto.ListFields())

    def testDefaultValues(self):
        if False:
            i = 10
            return i + 15
        proto = unittest_pb2.TestAllTypes()
        self.assertEqual(0, proto.optional_int32)
        self.assertEqual(0, proto.optional_int64)
        self.assertEqual(0, proto.optional_uint32)
        self.assertEqual(0, proto.optional_uint64)
        self.assertEqual(0, proto.optional_sint32)
        self.assertEqual(0, proto.optional_sint64)
        self.assertEqual(0, proto.optional_fixed32)
        self.assertEqual(0, proto.optional_fixed64)
        self.assertEqual(0, proto.optional_sfixed32)
        self.assertEqual(0, proto.optional_sfixed64)
        self.assertEqual(0.0, proto.optional_float)
        self.assertEqual(0.0, proto.optional_double)
        self.assertEqual(False, proto.optional_bool)
        self.assertEqual('', proto.optional_string)
        self.assertEqual(b'', proto.optional_bytes)
        self.assertEqual(41, proto.default_int32)
        self.assertEqual(42, proto.default_int64)
        self.assertEqual(43, proto.default_uint32)
        self.assertEqual(44, proto.default_uint64)
        self.assertEqual(-45, proto.default_sint32)
        self.assertEqual(46, proto.default_sint64)
        self.assertEqual(47, proto.default_fixed32)
        self.assertEqual(48, proto.default_fixed64)
        self.assertEqual(49, proto.default_sfixed32)
        self.assertEqual(-50, proto.default_sfixed64)
        self.assertEqual(51.5, proto.default_float)
        self.assertEqual(52000.0, proto.default_double)
        self.assertEqual(True, proto.default_bool)
        self.assertEqual('hello', proto.default_string)
        self.assertEqual(b'world', proto.default_bytes)
        self.assertEqual(unittest_pb2.TestAllTypes.BAR, proto.default_nested_enum)
        self.assertEqual(unittest_pb2.FOREIGN_BAR, proto.default_foreign_enum)
        self.assertEqual(unittest_import_pb2.IMPORT_BAR, proto.default_import_enum)
        proto = unittest_pb2.TestExtremeDefaultValues()
        self.assertEqual(u'ሴ', proto.utf8_string)

    def testHasFieldWithUnknownFieldName(self):
        if False:
            return 10
        proto = unittest_pb2.TestAllTypes()
        self.assertRaises(ValueError, proto.HasField, 'nonexistent_field')

    def testClearFieldWithUnknownFieldName(self):
        if False:
            return 10
        proto = unittest_pb2.TestAllTypes()
        self.assertRaises(ValueError, proto.ClearField, 'nonexistent_field')

    def testClearRemovesChildren(self):
        if False:
            i = 10
            return i + 15
        proto = unittest_pb2.TestRequiredForeign()
        for i in range(10):
            proto.repeated_message.add()
        proto2 = unittest_pb2.TestRequiredForeign()
        proto.CopyFrom(proto2)
        self.assertRaises(IndexError, lambda : proto.repeated_message[5])

    def testDisallowedAssignments(self):
        if False:
            return 10
        proto = unittest_pb2.TestAllTypes()
        self.assertRaises(AttributeError, setattr, proto, 'repeated_int32', 10)
        self.assertRaises(AttributeError, setattr, proto, 'repeated_int32', [10])
        self.assertRaises(AttributeError, setattr, proto, 'optional_nested_message', 23)
        self.assertRaises(AttributeError, setattr, proto.repeated_nested_message, 'bb', 34)
        self.assertRaises(AttributeError, setattr, proto.repeated_float, 'some_attribute', 34)
        self.assertRaises(AttributeError, setattr, proto, 'nonexistent_field', 23)

    def testSingleScalarTypeSafety(self):
        if False:
            for i in range(10):
                print('nop')
        proto = unittest_pb2.TestAllTypes()
        self.assertRaises(TypeError, setattr, proto, 'optional_int32', 1.1)
        self.assertRaises(TypeError, setattr, proto, 'optional_int32', 'foo')
        self.assertRaises(TypeError, setattr, proto, 'optional_string', 10)
        self.assertRaises(TypeError, setattr, proto, 'optional_bytes', 10)

    def assertIntegerTypes(self, integer_fn):
        if False:
            print('Hello World!')
        'Verifies setting of scalar integers.\n\n    Args:\n      integer_fn: A function to wrap the integers that will be assigned.\n    '

        def TestGetAndDeserialize(field_name, value, expected_type):
            if False:
                return 10
            proto = unittest_pb2.TestAllTypes()
            value = integer_fn(value)
            setattr(proto, field_name, value)
            self.assertIsInstance(getattr(proto, field_name), expected_type)
            proto2 = unittest_pb2.TestAllTypes()
            proto2.ParseFromString(proto.SerializeToString())
            self.assertIsInstance(getattr(proto2, field_name), expected_type)
        TestGetAndDeserialize('optional_int32', 1, int)
        TestGetAndDeserialize('optional_int32', 1 << 30, int)
        TestGetAndDeserialize('optional_uint32', 1 << 30, int)
        try:
            integer_64 = long
        except NameError:
            integer_64 = int
        if struct.calcsize('L') == 4:
            TestGetAndDeserialize('optional_uint32', 1 << 31, integer_64)
        else:
            TestGetAndDeserialize('optional_uint32', 1 << 31, int)
        TestGetAndDeserialize('optional_int64', 1 << 30, integer_64)
        TestGetAndDeserialize('optional_int64', 1 << 60, integer_64)
        TestGetAndDeserialize('optional_uint64', 1 << 30, integer_64)
        TestGetAndDeserialize('optional_uint64', 1 << 60, integer_64)

    def testIntegerTypes(self):
        if False:
            i = 10
            return i + 15
        self.assertIntegerTypes(lambda x: x)

    def testNonStandardIntegerTypes(self):
        if False:
            while True:
                i = 10
        self.assertIntegerTypes(test_util.NonStandardInteger)

    def testIllegalValuesForIntegers(self):
        if False:
            while True:
                i = 10
        pb = unittest_pb2.TestAllTypes()
        with self.assertRaises(TypeError):
            pb.optional_uint64 = '2'
        with self.assertRaisesRegexp(RuntimeError, 'my_error'):
            pb.optional_uint64 = test_util.NonStandardInteger(5, 'my_error')

    def assetIntegerBoundsChecking(self, integer_fn):
        if False:
            i = 10
            return i + 15
        'Verifies bounds checking for scalar integer fields.\n\n    Args:\n      integer_fn: A function to wrap the integers that will be assigned.\n    '

        def TestMinAndMaxIntegers(field_name, expected_min, expected_max):
            if False:
                while True:
                    i = 10
            pb = unittest_pb2.TestAllTypes()
            expected_min = integer_fn(expected_min)
            expected_max = integer_fn(expected_max)
            setattr(pb, field_name, expected_min)
            self.assertEqual(expected_min, getattr(pb, field_name))
            setattr(pb, field_name, expected_max)
            self.assertEqual(expected_max, getattr(pb, field_name))
            self.assertRaises(ValueError, setattr, pb, field_name, expected_min - 1)
            self.assertRaises(ValueError, setattr, pb, field_name, expected_max + 1)
        TestMinAndMaxIntegers('optional_int32', -(1 << 31), (1 << 31) - 1)
        TestMinAndMaxIntegers('optional_uint32', 0, 4294967295)
        TestMinAndMaxIntegers('optional_int64', -(1 << 63), (1 << 63) - 1)
        TestMinAndMaxIntegers('optional_uint64', 0, 18446744073709551615)
        pb = unittest_pb2.TestAllTypes()
        with self.assertRaises(ValueError):
            pb.optional_uint64 = integer_fn(-(1 << 63))
        pb = unittest_pb2.TestAllTypes()
        pb.optional_nested_enum = integer_fn(1)
        self.assertEqual(1, pb.optional_nested_enum)

    def testSingleScalarBoundsChecking(self):
        if False:
            while True:
                i = 10
        self.assetIntegerBoundsChecking(lambda x: x)

    def testNonStandardSingleScalarBoundsChecking(self):
        if False:
            while True:
                i = 10
        self.assetIntegerBoundsChecking(test_util.NonStandardInteger)

    def testRepeatedScalarTypeSafety(self):
        if False:
            while True:
                i = 10
        proto = unittest_pb2.TestAllTypes()
        self.assertRaises(TypeError, proto.repeated_int32.append, 1.1)
        self.assertRaises(TypeError, proto.repeated_int32.append, 'foo')
        self.assertRaises(TypeError, proto.repeated_string, 10)
        self.assertRaises(TypeError, proto.repeated_bytes, 10)
        proto.repeated_int32.append(10)
        proto.repeated_int32[0] = 23
        self.assertRaises(IndexError, proto.repeated_int32.__setitem__, 500, 23)
        self.assertRaises(TypeError, proto.repeated_int32.__setitem__, 0, 'abc')

    def testSingleScalarGettersAndSetters(self):
        if False:
            return 10
        proto = unittest_pb2.TestAllTypes()
        self.assertEqual(0, proto.optional_int32)
        proto.optional_int32 = 1
        self.assertEqual(1, proto.optional_int32)
        proto.optional_uint64 = 281474976710655
        self.assertEqual(281474976710655, proto.optional_uint64)
        proto.optional_uint64 = 18446744073709551615
        self.assertEqual(18446744073709551615, proto.optional_uint64)

    def testSingleScalarClearField(self):
        if False:
            for i in range(10):
                print('nop')
        proto = unittest_pb2.TestAllTypes()
        proto.ClearField('optional_int32')
        proto.optional_int32 = 1
        self.assertTrue(proto.HasField('optional_int32'))
        proto.ClearField('optional_int32')
        self.assertEqual(0, proto.optional_int32)
        self.assertTrue(not proto.HasField('optional_int32'))

    def testEnums(self):
        if False:
            while True:
                i = 10
        proto = unittest_pb2.TestAllTypes()
        self.assertEqual(1, proto.FOO)
        self.assertEqual(1, unittest_pb2.TestAllTypes.FOO)
        self.assertEqual(2, proto.BAR)
        self.assertEqual(2, unittest_pb2.TestAllTypes.BAR)
        self.assertEqual(3, proto.BAZ)
        self.assertEqual(3, unittest_pb2.TestAllTypes.BAZ)

    def testEnum_Name(self):
        if False:
            print('Hello World!')
        self.assertEqual('FOREIGN_FOO', unittest_pb2.ForeignEnum.Name(unittest_pb2.FOREIGN_FOO))
        self.assertEqual('FOREIGN_BAR', unittest_pb2.ForeignEnum.Name(unittest_pb2.FOREIGN_BAR))
        self.assertEqual('FOREIGN_BAZ', unittest_pb2.ForeignEnum.Name(unittest_pb2.FOREIGN_BAZ))
        self.assertRaises(ValueError, unittest_pb2.ForeignEnum.Name, 11312)
        proto = unittest_pb2.TestAllTypes()
        self.assertEqual('FOO', proto.NestedEnum.Name(proto.FOO))
        self.assertEqual('FOO', unittest_pb2.TestAllTypes.NestedEnum.Name(proto.FOO))
        self.assertEqual('BAR', proto.NestedEnum.Name(proto.BAR))
        self.assertEqual('BAR', unittest_pb2.TestAllTypes.NestedEnum.Name(proto.BAR))
        self.assertEqual('BAZ', proto.NestedEnum.Name(proto.BAZ))
        self.assertEqual('BAZ', unittest_pb2.TestAllTypes.NestedEnum.Name(proto.BAZ))
        self.assertRaises(ValueError, proto.NestedEnum.Name, 11312)
        self.assertRaises(ValueError, unittest_pb2.TestAllTypes.NestedEnum.Name, 11312)

    def testEnum_Value(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(unittest_pb2.FOREIGN_FOO, unittest_pb2.ForeignEnum.Value('FOREIGN_FOO'))
        self.assertEqual(unittest_pb2.FOREIGN_BAR, unittest_pb2.ForeignEnum.Value('FOREIGN_BAR'))
        self.assertEqual(unittest_pb2.FOREIGN_BAZ, unittest_pb2.ForeignEnum.Value('FOREIGN_BAZ'))
        self.assertRaises(ValueError, unittest_pb2.ForeignEnum.Value, 'FO')
        proto = unittest_pb2.TestAllTypes()
        self.assertEqual(proto.FOO, proto.NestedEnum.Value('FOO'))
        self.assertEqual(proto.FOO, unittest_pb2.TestAllTypes.NestedEnum.Value('FOO'))
        self.assertEqual(proto.BAR, proto.NestedEnum.Value('BAR'))
        self.assertEqual(proto.BAR, unittest_pb2.TestAllTypes.NestedEnum.Value('BAR'))
        self.assertEqual(proto.BAZ, proto.NestedEnum.Value('BAZ'))
        self.assertEqual(proto.BAZ, unittest_pb2.TestAllTypes.NestedEnum.Value('BAZ'))
        self.assertRaises(ValueError, proto.NestedEnum.Value, 'Foo')
        self.assertRaises(ValueError, unittest_pb2.TestAllTypes.NestedEnum.Value, 'Foo')

    def testEnum_KeysAndValues(self):
        if False:
            return 10
        self.assertEqual(['FOREIGN_FOO', 'FOREIGN_BAR', 'FOREIGN_BAZ'], list(unittest_pb2.ForeignEnum.keys()))
        self.assertEqual([4, 5, 6], list(unittest_pb2.ForeignEnum.values()))
        self.assertEqual([('FOREIGN_FOO', 4), ('FOREIGN_BAR', 5), ('FOREIGN_BAZ', 6)], list(unittest_pb2.ForeignEnum.items()))
        proto = unittest_pb2.TestAllTypes()
        self.assertEqual(['FOO', 'BAR', 'BAZ', 'NEG'], list(proto.NestedEnum.keys()))
        self.assertEqual([1, 2, 3, -1], list(proto.NestedEnum.values()))
        self.assertEqual([('FOO', 1), ('BAR', 2), ('BAZ', 3), ('NEG', -1)], list(proto.NestedEnum.items()))

    def testRepeatedScalars(self):
        if False:
            print('Hello World!')
        proto = unittest_pb2.TestAllTypes()
        self.assertTrue(not proto.repeated_int32)
        self.assertEqual(0, len(proto.repeated_int32))
        proto.repeated_int32.append(5)
        proto.repeated_int32.append(10)
        proto.repeated_int32.append(15)
        self.assertTrue(proto.repeated_int32)
        self.assertEqual(3, len(proto.repeated_int32))
        self.assertEqual([5, 10, 15], proto.repeated_int32)
        self.assertEqual(5, proto.repeated_int32[0])
        self.assertEqual(15, proto.repeated_int32[-1])
        self.assertRaises(IndexError, proto.repeated_int32.__getitem__, 1234)
        self.assertRaises(IndexError, proto.repeated_int32.__getitem__, -1234)
        self.assertRaises(TypeError, proto.repeated_int32.__getitem__, 'foo')
        self.assertRaises(TypeError, proto.repeated_int32.__getitem__, None)
        proto.repeated_int32[1] = 20
        self.assertEqual([5, 20, 15], proto.repeated_int32)
        proto.repeated_int32.insert(1, 25)
        self.assertEqual([5, 25, 20, 15], proto.repeated_int32)
        proto.repeated_int32.append(30)
        self.assertEqual([25, 20, 15], proto.repeated_int32[1:4])
        self.assertEqual([5, 25, 20, 15, 30], proto.repeated_int32[:])
        proto.repeated_int32[1:4] = (i for i in range(3))
        self.assertEqual([5, 0, 1, 2, 30], proto.repeated_int32)
        proto.repeated_int32[1:4] = [35, 40, 45]
        self.assertEqual([5, 35, 40, 45, 30], proto.repeated_int32)
        result = []
        for i in proto.repeated_int32:
            result.append(i)
        self.assertEqual([5, 35, 40, 45, 30], result)
        del proto.repeated_int32[2]
        self.assertEqual([5, 35, 45, 30], proto.repeated_int32)
        del proto.repeated_int32[2:]
        self.assertEqual([5, 35], proto.repeated_int32)
        proto.repeated_int32.extend([3, 13])
        self.assertEqual([5, 35, 3, 13], proto.repeated_int32)
        proto.ClearField('repeated_int32')
        self.assertTrue(not proto.repeated_int32)
        self.assertEqual(0, len(proto.repeated_int32))
        proto.repeated_int32.append(1)
        self.assertEqual(1, proto.repeated_int32[-1])
        proto.repeated_int32[-1] = 2
        self.assertEqual(2, proto.repeated_int32[-1])
        proto.repeated_int32[:] = [0, 1, 2, 3]
        del proto.repeated_int32[-1]
        self.assertEqual([0, 1, 2], proto.repeated_int32)
        del proto.repeated_int32[-2]
        self.assertEqual([0, 2], proto.repeated_int32)
        self.assertRaises(IndexError, proto.repeated_int32.__delitem__, -3)
        self.assertRaises(IndexError, proto.repeated_int32.__delitem__, 300)
        del proto.repeated_int32[-2:-1]
        self.assertEqual([2], proto.repeated_int32)
        del proto.repeated_int32[100:10000]
        self.assertEqual([2], proto.repeated_int32)

    def testRepeatedScalarsRemove(self):
        if False:
            while True:
                i = 10
        proto = unittest_pb2.TestAllTypes()
        self.assertTrue(not proto.repeated_int32)
        self.assertEqual(0, len(proto.repeated_int32))
        proto.repeated_int32.append(5)
        proto.repeated_int32.append(10)
        proto.repeated_int32.append(5)
        proto.repeated_int32.append(5)
        self.assertEqual(4, len(proto.repeated_int32))
        proto.repeated_int32.remove(5)
        self.assertEqual(3, len(proto.repeated_int32))
        self.assertEqual(10, proto.repeated_int32[0])
        self.assertEqual(5, proto.repeated_int32[1])
        self.assertEqual(5, proto.repeated_int32[2])
        proto.repeated_int32.remove(5)
        self.assertEqual(2, len(proto.repeated_int32))
        self.assertEqual(10, proto.repeated_int32[0])
        self.assertEqual(5, proto.repeated_int32[1])
        proto.repeated_int32.remove(10)
        self.assertEqual(1, len(proto.repeated_int32))
        self.assertEqual(5, proto.repeated_int32[0])
        self.assertRaises(ValueError, proto.repeated_int32.remove, 123)

    def testRepeatedComposites(self):
        if False:
            i = 10
            return i + 15
        proto = unittest_pb2.TestAllTypes()
        self.assertTrue(not proto.repeated_nested_message)
        self.assertEqual(0, len(proto.repeated_nested_message))
        m0 = proto.repeated_nested_message.add()
        m1 = proto.repeated_nested_message.add()
        self.assertTrue(proto.repeated_nested_message)
        self.assertEqual(2, len(proto.repeated_nested_message))
        self.assertListsEqual([m0, m1], proto.repeated_nested_message)
        self.assertIsInstance(m0, unittest_pb2.TestAllTypes.NestedMessage)
        self.assertRaises(IndexError, proto.repeated_nested_message.__getitem__, 1234)
        self.assertRaises(IndexError, proto.repeated_nested_message.__getitem__, -1234)
        self.assertRaises(TypeError, proto.repeated_nested_message.__getitem__, 'foo')
        self.assertRaises(TypeError, proto.repeated_nested_message.__getitem__, None)
        m2 = proto.repeated_nested_message.add()
        m3 = proto.repeated_nested_message.add()
        m4 = proto.repeated_nested_message.add()
        self.assertListsEqual([m1, m2, m3], proto.repeated_nested_message[1:4])
        self.assertListsEqual([m0, m1, m2, m3, m4], proto.repeated_nested_message[:])
        self.assertListsEqual([m0, m1], proto.repeated_nested_message[:2])
        self.assertListsEqual([m2, m3, m4], proto.repeated_nested_message[2:])
        self.assertEqual(m0, proto.repeated_nested_message[0])
        self.assertListsEqual([m0], proto.repeated_nested_message[:1])
        result = []
        for i in proto.repeated_nested_message:
            result.append(i)
        self.assertListsEqual([m0, m1, m2, m3, m4], result)
        del proto.repeated_nested_message[2]
        self.assertListsEqual([m0, m1, m3, m4], proto.repeated_nested_message)
        del proto.repeated_nested_message[2:]
        self.assertListsEqual([m0, m1], proto.repeated_nested_message)
        n1 = unittest_pb2.TestAllTypes.NestedMessage(bb=1)
        n2 = unittest_pb2.TestAllTypes.NestedMessage(bb=2)
        proto.repeated_nested_message.extend([n1, n2])
        self.assertEqual(4, len(proto.repeated_nested_message))
        self.assertEqual(n1, proto.repeated_nested_message[2])
        self.assertEqual(n2, proto.repeated_nested_message[3])
        proto.ClearField('repeated_nested_message')
        self.assertTrue(not proto.repeated_nested_message)
        self.assertEqual(0, len(proto.repeated_nested_message))
        proto.repeated_nested_message.add(bb=23)
        self.assertEqual(1, len(proto.repeated_nested_message))
        self.assertEqual(23, proto.repeated_nested_message[0].bb)
        self.assertRaises(TypeError, proto.repeated_nested_message.add, 23)

    def testRepeatedCompositeRemove(self):
        if False:
            print('Hello World!')
        proto = unittest_pb2.TestAllTypes()
        self.assertEqual(0, len(proto.repeated_nested_message))
        m0 = proto.repeated_nested_message.add()
        m0.bb = len(proto.repeated_nested_message)
        m1 = proto.repeated_nested_message.add()
        m1.bb = len(proto.repeated_nested_message)
        self.assertTrue(m0 != m1)
        m2 = proto.repeated_nested_message.add()
        m2.bb = len(proto.repeated_nested_message)
        self.assertListsEqual([m0, m1, m2], proto.repeated_nested_message)
        self.assertEqual(3, len(proto.repeated_nested_message))
        proto.repeated_nested_message.remove(m0)
        self.assertEqual(2, len(proto.repeated_nested_message))
        self.assertEqual(m1, proto.repeated_nested_message[0])
        self.assertEqual(m2, proto.repeated_nested_message[1])
        self.assertRaises(ValueError, proto.repeated_nested_message.remove, m0)
        self.assertRaises(ValueError, proto.repeated_nested_message.remove, None)
        self.assertEqual(2, len(proto.repeated_nested_message))
        proto.repeated_nested_message.remove(m2)
        self.assertEqual(1, len(proto.repeated_nested_message))
        self.assertEqual(m1, proto.repeated_nested_message[0])

    def testHandWrittenReflection(self):
        if False:
            for i in range(10):
                print('nop')
        if api_implementation.Type() != 'python':
            return
        FieldDescriptor = descriptor.FieldDescriptor
        foo_field_descriptor = FieldDescriptor(name='foo_field', full_name='MyProto.foo_field', index=0, number=1, type=FieldDescriptor.TYPE_INT64, cpp_type=FieldDescriptor.CPPTYPE_INT64, label=FieldDescriptor.LABEL_OPTIONAL, default_value=0, containing_type=None, message_type=None, enum_type=None, is_extension=False, extension_scope=None, options=descriptor_pb2.FieldOptions())
        mydescriptor = descriptor.Descriptor(name='MyProto', full_name='MyProto', filename='ignored', containing_type=None, nested_types=[], enum_types=[], fields=[foo_field_descriptor], extensions=[], options=descriptor_pb2.MessageOptions())

        class MyProtoClass(six.with_metaclass(reflection.GeneratedProtocolMessageType, message.Message)):
            DESCRIPTOR = mydescriptor
        myproto_instance = MyProtoClass()
        self.assertEqual(0, myproto_instance.foo_field)
        self.assertTrue(not myproto_instance.HasField('foo_field'))
        myproto_instance.foo_field = 23
        self.assertEqual(23, myproto_instance.foo_field)
        self.assertTrue(myproto_instance.HasField('foo_field'))

    def testDescriptorProtoSupport(self):
        if False:
            print('Hello World!')
        if api_implementation.Type() != 'python':
            return

        def AddDescriptorField(proto, field_name, field_type):
            if False:
                for i in range(10):
                    print('nop')
            AddDescriptorField.field_index += 1
            new_field = proto.field.add()
            new_field.name = field_name
            new_field.type = field_type
            new_field.number = AddDescriptorField.field_index
            new_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        AddDescriptorField.field_index = 0
        desc_proto = descriptor_pb2.DescriptorProto()
        desc_proto.name = 'Car'
        fdp = descriptor_pb2.FieldDescriptorProto
        AddDescriptorField(desc_proto, 'name', fdp.TYPE_STRING)
        AddDescriptorField(desc_proto, 'year', fdp.TYPE_INT64)
        AddDescriptorField(desc_proto, 'automatic', fdp.TYPE_BOOL)
        AddDescriptorField(desc_proto, 'price', fdp.TYPE_DOUBLE)
        AddDescriptorField.field_index += 1
        new_field = desc_proto.field.add()
        new_field.name = 'owners'
        new_field.type = fdp.TYPE_STRING
        new_field.number = AddDescriptorField.field_index
        new_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
        desc = descriptor.MakeDescriptor(desc_proto)
        self.assertTrue('name' in desc.fields_by_name)
        self.assertTrue('year' in desc.fields_by_name)
        self.assertTrue('automatic' in desc.fields_by_name)
        self.assertTrue('price' in desc.fields_by_name)
        self.assertTrue('owners' in desc.fields_by_name)

        class CarMessage(six.with_metaclass(reflection.GeneratedProtocolMessageType, message.Message)):
            DESCRIPTOR = desc
        prius = CarMessage()
        prius.name = 'prius'
        prius.year = 2010
        prius.automatic = True
        prius.price = 25134.75
        prius.owners.extend(['bob', 'susan'])
        serialized_prius = prius.SerializeToString()
        new_prius = reflection.ParseMessage(desc, serialized_prius)
        self.assertTrue(new_prius is not prius)
        self.assertEqual(prius, new_prius)
        self.assertEqual(prius.name, new_prius.name)
        self.assertEqual(prius.year, new_prius.year)
        self.assertEqual(prius.automatic, new_prius.automatic)
        self.assertEqual(prius.price, new_prius.price)
        self.assertEqual(prius.owners, new_prius.owners)

    def testTopLevelExtensionsForOptionalScalar(self):
        if False:
            return 10
        extendee_proto = unittest_pb2.TestAllExtensions()
        extension = unittest_pb2.optional_int32_extension
        self.assertTrue(not extendee_proto.HasExtension(extension))
        self.assertEqual(0, extendee_proto.Extensions[extension])
        self.assertTrue(not extendee_proto.HasExtension(extension))
        extendee_proto.Extensions[extension] = 23
        self.assertEqual(23, extendee_proto.Extensions[extension])
        self.assertTrue(extendee_proto.HasExtension(extension))
        extendee_proto.ClearExtension(extension)
        self.assertEqual(0, extendee_proto.Extensions[extension])
        self.assertTrue(not extendee_proto.HasExtension(extension))

    def testTopLevelExtensionsForRepeatedScalar(self):
        if False:
            print('Hello World!')
        extendee_proto = unittest_pb2.TestAllExtensions()
        extension = unittest_pb2.repeated_string_extension
        self.assertEqual(0, len(extendee_proto.Extensions[extension]))
        extendee_proto.Extensions[extension].append('foo')
        self.assertEqual(['foo'], extendee_proto.Extensions[extension])
        string_list = extendee_proto.Extensions[extension]
        extendee_proto.ClearExtension(extension)
        self.assertEqual(0, len(extendee_proto.Extensions[extension]))
        self.assertTrue(string_list is not extendee_proto.Extensions[extension])
        self.assertRaises(TypeError, operator.setitem, extendee_proto.Extensions, extension, 'a')

    def testTopLevelExtensionsForOptionalMessage(self):
        if False:
            return 10
        extendee_proto = unittest_pb2.TestAllExtensions()
        extension = unittest_pb2.optional_foreign_message_extension
        self.assertTrue(not extendee_proto.HasExtension(extension))
        self.assertEqual(0, extendee_proto.Extensions[extension].c)
        self.assertTrue(not extendee_proto.HasExtension(extension))
        extendee_proto.Extensions[extension].c = 23
        self.assertEqual(23, extendee_proto.Extensions[extension].c)
        self.assertTrue(extendee_proto.HasExtension(extension))
        foreign_message = extendee_proto.Extensions[extension]
        extendee_proto.ClearExtension(extension)
        self.assertTrue(foreign_message is not extendee_proto.Extensions[extension])
        foreign_message.c = 42
        self.assertEqual(42, foreign_message.c)
        self.assertTrue(foreign_message.HasField('c'))
        self.assertTrue(not extendee_proto.HasExtension(extension))
        self.assertRaises(TypeError, operator.setitem, extendee_proto.Extensions, extension, 'a')

    def testTopLevelExtensionsForRepeatedMessage(self):
        if False:
            i = 10
            return i + 15
        extendee_proto = unittest_pb2.TestAllExtensions()
        extension = unittest_pb2.repeatedgroup_extension
        self.assertEqual(0, len(extendee_proto.Extensions[extension]))
        group = extendee_proto.Extensions[extension].add()
        group.a = 23
        self.assertEqual(23, extendee_proto.Extensions[extension][0].a)
        group.a = 42
        self.assertEqual(42, extendee_proto.Extensions[extension][0].a)
        group_list = extendee_proto.Extensions[extension]
        extendee_proto.ClearExtension(extension)
        self.assertEqual(0, len(extendee_proto.Extensions[extension]))
        self.assertTrue(group_list is not extendee_proto.Extensions[extension])
        self.assertRaises(TypeError, operator.setitem, extendee_proto.Extensions, extension, 'a')

    def testNestedExtensions(self):
        if False:
            print('Hello World!')
        extendee_proto = unittest_pb2.TestAllExtensions()
        extension = unittest_pb2.TestRequired.single
        self.assertTrue(not extendee_proto.HasExtension(extension))
        required = extendee_proto.Extensions[extension]
        self.assertEqual(0, required.a)
        self.assertTrue(not extendee_proto.HasExtension(extension))
        required.a = 23
        self.assertEqual(23, extendee_proto.Extensions[extension].a)
        self.assertTrue(extendee_proto.HasExtension(extension))
        extendee_proto.ClearExtension(extension)
        self.assertTrue(required is not extendee_proto.Extensions[extension])
        self.assertTrue(not extendee_proto.HasExtension(extension))

    def testRegisteredExtensions(self):
        if False:
            print('Hello World!')
        pool = unittest_pb2.DESCRIPTOR.pool
        self.assertTrue(pool.FindExtensionByNumber(unittest_pb2.TestAllExtensions.DESCRIPTOR, 1))
        self.assertIs(pool.FindExtensionByName('protobuf_unittest.optional_int32_extension').containing_type, unittest_pb2.TestAllExtensions.DESCRIPTOR)
        self.assertEqual(0, len(pool.FindAllExtensions(unittest_pb2.TestAllTypes.DESCRIPTOR)))

    def testHasBitsForAncestorsOfExtendedMessage(self):
        if False:
            i = 10
            return i + 15
        toplevel = more_extensions_pb2.TopLevelMessage()
        self.assertTrue(not toplevel.HasField('submessage'))
        self.assertEqual(0, toplevel.submessage.Extensions[more_extensions_pb2.optional_int_extension])
        self.assertTrue(not toplevel.HasField('submessage'))
        toplevel.submessage.Extensions[more_extensions_pb2.optional_int_extension] = 23
        self.assertEqual(23, toplevel.submessage.Extensions[more_extensions_pb2.optional_int_extension])
        self.assertTrue(toplevel.HasField('submessage'))
        toplevel = more_extensions_pb2.TopLevelMessage()
        self.assertTrue(not toplevel.HasField('submessage'))
        self.assertEqual([], toplevel.submessage.Extensions[more_extensions_pb2.repeated_int_extension])
        self.assertTrue(not toplevel.HasField('submessage'))
        toplevel.submessage.Extensions[more_extensions_pb2.repeated_int_extension].append(23)
        self.assertEqual([23], toplevel.submessage.Extensions[more_extensions_pb2.repeated_int_extension])
        self.assertTrue(toplevel.HasField('submessage'))
        toplevel = more_extensions_pb2.TopLevelMessage()
        self.assertTrue(not toplevel.HasField('submessage'))
        self.assertEqual(0, toplevel.submessage.Extensions[more_extensions_pb2.optional_message_extension].foreign_message_int)
        self.assertTrue(not toplevel.HasField('submessage'))
        toplevel.submessage.Extensions[more_extensions_pb2.optional_message_extension].foreign_message_int = 23
        self.assertEqual(23, toplevel.submessage.Extensions[more_extensions_pb2.optional_message_extension].foreign_message_int)
        self.assertTrue(toplevel.HasField('submessage'))
        toplevel = more_extensions_pb2.TopLevelMessage()
        self.assertTrue(not toplevel.HasField('submessage'))
        self.assertEqual(0, len(toplevel.submessage.Extensions[more_extensions_pb2.repeated_message_extension]))
        self.assertTrue(not toplevel.HasField('submessage'))
        foreign = toplevel.submessage.Extensions[more_extensions_pb2.repeated_message_extension].add()
        self.assertEqual(foreign, toplevel.submessage.Extensions[more_extensions_pb2.repeated_message_extension][0])
        self.assertTrue(toplevel.HasField('submessage'))

    def testDisconnectionAfterClearingEmptyMessage(self):
        if False:
            for i in range(10):
                print('nop')
        toplevel = more_extensions_pb2.TopLevelMessage()
        extendee_proto = toplevel.submessage
        extension = more_extensions_pb2.optional_message_extension
        extension_proto = extendee_proto.Extensions[extension]
        extendee_proto.ClearExtension(extension)
        extension_proto.foreign_message_int = 23
        self.assertTrue(extension_proto is not extendee_proto.Extensions[extension])

    def testExtensionFailureModes(self):
        if False:
            for i in range(10):
                print('nop')
        extendee_proto = unittest_pb2.TestAllExtensions()
        self.assertRaises(KeyError, extendee_proto.HasExtension, 1234)
        self.assertRaises(KeyError, extendee_proto.ClearExtension, 1234)
        self.assertRaises(KeyError, extendee_proto.Extensions.__getitem__, 1234)
        self.assertRaises(KeyError, extendee_proto.Extensions.__setitem__, 1234, 5)
        for unknown_handle in (more_extensions_pb2.optional_int_extension, more_extensions_pb2.optional_message_extension, more_extensions_pb2.repeated_int_extension, more_extensions_pb2.repeated_message_extension):
            self.assertRaises(KeyError, extendee_proto.HasExtension, unknown_handle)
            self.assertRaises(KeyError, extendee_proto.ClearExtension, unknown_handle)
            self.assertRaises(KeyError, extendee_proto.Extensions.__getitem__, unknown_handle)
            self.assertRaises(KeyError, extendee_proto.Extensions.__setitem__, unknown_handle, 5)
        self.assertRaises(KeyError, extendee_proto.HasExtension, unittest_pb2.repeated_string_extension)

    def testStaticParseFrom(self):
        if False:
            i = 10
            return i + 15
        proto1 = unittest_pb2.TestAllTypes()
        test_util.SetAllFields(proto1)
        string1 = proto1.SerializeToString()
        proto2 = unittest_pb2.TestAllTypes.FromString(string1)
        self.assertEqual(proto2, proto1)

    def testMergeFromSingularField(self):
        if False:
            print('Hello World!')
        proto1 = unittest_pb2.TestAllTypes()
        proto1.optional_int32 = 1
        proto2 = unittest_pb2.TestAllTypes()
        proto2.optional_string = 'value'
        proto2.MergeFrom(proto1)
        self.assertEqual(1, proto2.optional_int32)
        self.assertEqual('value', proto2.optional_string)

    def testMergeFromRepeatedField(self):
        if False:
            while True:
                i = 10
        proto1 = unittest_pb2.TestAllTypes()
        proto1.repeated_int32.append(1)
        proto1.repeated_int32.append(2)
        proto2 = unittest_pb2.TestAllTypes()
        proto2.repeated_int32.append(0)
        proto2.MergeFrom(proto1)
        self.assertEqual(0, proto2.repeated_int32[0])
        self.assertEqual(1, proto2.repeated_int32[1])
        self.assertEqual(2, proto2.repeated_int32[2])

    def testMergeFromOptionalGroup(self):
        if False:
            i = 10
            return i + 15
        proto1 = unittest_pb2.TestAllTypes()
        proto1.optionalgroup.a = 12
        proto2 = unittest_pb2.TestAllTypes()
        proto2.MergeFrom(proto1)
        self.assertEqual(12, proto2.optionalgroup.a)

    def testMergeFromRepeatedNestedMessage(self):
        if False:
            for i in range(10):
                print('nop')
        proto1 = unittest_pb2.TestAllTypes()
        m = proto1.repeated_nested_message.add()
        m.bb = 123
        m = proto1.repeated_nested_message.add()
        m.bb = 321
        proto2 = unittest_pb2.TestAllTypes()
        m = proto2.repeated_nested_message.add()
        m.bb = 999
        proto2.MergeFrom(proto1)
        self.assertEqual(999, proto2.repeated_nested_message[0].bb)
        self.assertEqual(123, proto2.repeated_nested_message[1].bb)
        self.assertEqual(321, proto2.repeated_nested_message[2].bb)
        proto3 = unittest_pb2.TestAllTypes()
        proto3.repeated_nested_message.MergeFrom(proto2.repeated_nested_message)
        self.assertEqual(999, proto3.repeated_nested_message[0].bb)
        self.assertEqual(123, proto3.repeated_nested_message[1].bb)
        self.assertEqual(321, proto3.repeated_nested_message[2].bb)

    def testMergeFromAllFields(self):
        if False:
            i = 10
            return i + 15
        proto1 = unittest_pb2.TestAllTypes()
        test_util.SetAllFields(proto1)
        proto2 = unittest_pb2.TestAllTypes()
        proto2.MergeFrom(proto1)
        self.assertEqual(proto2, proto1)
        string1 = proto1.SerializeToString()
        string2 = proto2.SerializeToString()
        self.assertEqual(string1, string2)

    def testMergeFromExtensionsSingular(self):
        if False:
            print('Hello World!')
        proto1 = unittest_pb2.TestAllExtensions()
        proto1.Extensions[unittest_pb2.optional_int32_extension] = 1
        proto2 = unittest_pb2.TestAllExtensions()
        proto2.MergeFrom(proto1)
        self.assertEqual(1, proto2.Extensions[unittest_pb2.optional_int32_extension])

    def testMergeFromExtensionsRepeated(self):
        if False:
            while True:
                i = 10
        proto1 = unittest_pb2.TestAllExtensions()
        proto1.Extensions[unittest_pb2.repeated_int32_extension].append(1)
        proto1.Extensions[unittest_pb2.repeated_int32_extension].append(2)
        proto2 = unittest_pb2.TestAllExtensions()
        proto2.Extensions[unittest_pb2.repeated_int32_extension].append(0)
        proto2.MergeFrom(proto1)
        self.assertEqual(3, len(proto2.Extensions[unittest_pb2.repeated_int32_extension]))
        self.assertEqual(0, proto2.Extensions[unittest_pb2.repeated_int32_extension][0])
        self.assertEqual(1, proto2.Extensions[unittest_pb2.repeated_int32_extension][1])
        self.assertEqual(2, proto2.Extensions[unittest_pb2.repeated_int32_extension][2])

    def testMergeFromExtensionsNestedMessage(self):
        if False:
            print('Hello World!')
        proto1 = unittest_pb2.TestAllExtensions()
        ext1 = proto1.Extensions[unittest_pb2.repeated_nested_message_extension]
        m = ext1.add()
        m.bb = 222
        m = ext1.add()
        m.bb = 333
        proto2 = unittest_pb2.TestAllExtensions()
        ext2 = proto2.Extensions[unittest_pb2.repeated_nested_message_extension]
        m = ext2.add()
        m.bb = 111
        proto2.MergeFrom(proto1)
        ext2 = proto2.Extensions[unittest_pb2.repeated_nested_message_extension]
        self.assertEqual(3, len(ext2))
        self.assertEqual(111, ext2[0].bb)
        self.assertEqual(222, ext2[1].bb)
        self.assertEqual(333, ext2[2].bb)

    def testMergeFromBug(self):
        if False:
            while True:
                i = 10
        message1 = unittest_pb2.TestAllTypes()
        message2 = unittest_pb2.TestAllTypes()
        message1.optional_nested_message
        self.assertFalse(message1.HasField('optional_nested_message'))
        message2.MergeFrom(message1)
        self.assertFalse(message2.HasField('optional_nested_message'))

    def testCopyFromSingularField(self):
        if False:
            while True:
                i = 10
        proto1 = unittest_pb2.TestAllTypes()
        proto1.optional_int32 = 1
        proto1.optional_string = 'important-text'
        proto2 = unittest_pb2.TestAllTypes()
        proto2.optional_string = 'value'
        proto2.CopyFrom(proto1)
        self.assertEqual(1, proto2.optional_int32)
        self.assertEqual('important-text', proto2.optional_string)

    def testCopyFromRepeatedField(self):
        if False:
            for i in range(10):
                print('nop')
        proto1 = unittest_pb2.TestAllTypes()
        proto1.repeated_int32.append(1)
        proto1.repeated_int32.append(2)
        proto2 = unittest_pb2.TestAllTypes()
        proto2.repeated_int32.append(0)
        proto2.CopyFrom(proto1)
        self.assertEqual(1, proto2.repeated_int32[0])
        self.assertEqual(2, proto2.repeated_int32[1])

    def testCopyFromAllFields(self):
        if False:
            while True:
                i = 10
        proto1 = unittest_pb2.TestAllTypes()
        test_util.SetAllFields(proto1)
        proto2 = unittest_pb2.TestAllTypes()
        proto2.CopyFrom(proto1)
        self.assertEqual(proto2, proto1)
        string1 = proto1.SerializeToString()
        string2 = proto2.SerializeToString()
        self.assertEqual(string1, string2)

    def testCopyFromSelf(self):
        if False:
            while True:
                i = 10
        proto1 = unittest_pb2.TestAllTypes()
        proto1.repeated_int32.append(1)
        proto1.optional_int32 = 2
        proto1.optional_string = 'important-text'
        proto1.CopyFrom(proto1)
        self.assertEqual(1, proto1.repeated_int32[0])
        self.assertEqual(2, proto1.optional_int32)
        self.assertEqual('important-text', proto1.optional_string)

    def testCopyFromBadType(self):
        if False:
            i = 10
            return i + 15
        if api_implementation.Type() == 'python':
            return
        proto1 = unittest_pb2.TestAllTypes()
        proto2 = unittest_pb2.TestAllExtensions()
        self.assertRaises(TypeError, proto1.CopyFrom, proto2)

    def testDeepCopy(self):
        if False:
            while True:
                i = 10
        proto1 = unittest_pb2.TestAllTypes()
        proto1.optional_int32 = 1
        proto2 = copy.deepcopy(proto1)
        self.assertEqual(1, proto2.optional_int32)
        proto1.repeated_int32.append(2)
        proto1.repeated_int32.append(3)
        container = copy.deepcopy(proto1.repeated_int32)
        self.assertEqual([2, 3], container)
        message1 = proto1.repeated_nested_message.add()
        message1.bb = 1
        messages = copy.deepcopy(proto1.repeated_nested_message)
        self.assertEqual(proto1.repeated_nested_message, messages)
        message1.bb = 2
        self.assertNotEqual(proto1.repeated_nested_message, messages)

    def testClear(self):
        if False:
            while True:
                i = 10
        proto = unittest_pb2.TestAllTypes()
        if api_implementation.Type() == 'python':
            test_util.SetAllFields(proto)
        else:
            test_util.SetAllNonLazyFields(proto)
        proto.Clear()
        self.assertEqual(proto.ByteSize(), 0)
        empty_proto = unittest_pb2.TestAllTypes()
        self.assertEqual(proto, empty_proto)
        proto = unittest_pb2.TestAllExtensions()
        test_util.SetAllExtensions(proto)
        proto.Clear()
        self.assertEqual(proto.ByteSize(), 0)
        empty_proto = unittest_pb2.TestAllExtensions()
        self.assertEqual(proto, empty_proto)

    def testDisconnectingBeforeClear(self):
        if False:
            while True:
                i = 10
        proto = unittest_pb2.TestAllTypes()
        nested = proto.optional_nested_message
        proto.Clear()
        self.assertTrue(nested is not proto.optional_nested_message)
        nested.bb = 23
        self.assertTrue(not proto.HasField('optional_nested_message'))
        self.assertEqual(0, proto.optional_nested_message.bb)
        proto = unittest_pb2.TestAllTypes()
        nested = proto.optional_nested_message
        nested.bb = 5
        foreign = proto.optional_foreign_message
        foreign.c = 6
        proto.Clear()
        self.assertTrue(nested is not proto.optional_nested_message)
        self.assertTrue(foreign is not proto.optional_foreign_message)
        self.assertEqual(5, nested.bb)
        self.assertEqual(6, foreign.c)
        nested.bb = 15
        foreign.c = 16
        self.assertFalse(proto.HasField('optional_nested_message'))
        self.assertEqual(0, proto.optional_nested_message.bb)
        self.assertFalse(proto.HasField('optional_foreign_message'))
        self.assertEqual(0, proto.optional_foreign_message.c)

    def testDisconnectingInOneof(self):
        if False:
            i = 10
            return i + 15
        m = unittest_pb2.TestOneof2()
        m.foo_message.qux_int = 5
        sub_message = m.foo_message
        self.assertEqual(m.foo_lazy_message.qux_int, 0)
        self.assertEqual(m.foo_message.qux_int, 5)
        m.foo_lazy_message.qux_int = 6
        self.assertEqual(m.foo_message.qux_int, 0)
        self.assertEqual(sub_message.qux_int, 5)
        sub_message.qux_int = 7

    def testOneOf(self):
        if False:
            while True:
                i = 10
        proto = unittest_pb2.TestAllTypes()
        proto.oneof_uint32 = 10
        proto.oneof_nested_message.bb = 11
        self.assertEqual(11, proto.oneof_nested_message.bb)
        self.assertFalse(proto.HasField('oneof_uint32'))
        nested = proto.oneof_nested_message
        proto.oneof_string = 'abc'
        self.assertEqual('abc', proto.oneof_string)
        self.assertEqual(11, nested.bb)
        self.assertFalse(proto.HasField('oneof_nested_message'))

    def assertInitialized(self, proto):
        if False:
            print('Hello World!')
        self.assertTrue(proto.IsInitialized())
        proto.SerializeToString()
        proto.SerializePartialToString()

    def assertNotInitialized(self, proto):
        if False:
            print('Hello World!')
        self.assertFalse(proto.IsInitialized())
        self.assertRaises(message.EncodeError, proto.SerializeToString)
        proto.SerializePartialToString()

    def testIsInitialized(self):
        if False:
            i = 10
            return i + 15
        proto = unittest_pb2.TestAllTypes()
        self.assertInitialized(proto)
        proto = unittest_pb2.TestAllExtensions()
        self.assertInitialized(proto)
        proto = unittest_pb2.TestRequired()
        self.assertNotInitialized(proto)
        proto.a = proto.b = proto.c = 2
        self.assertInitialized(proto)
        proto = unittest_pb2.TestRequiredForeign()
        self.assertInitialized(proto)
        proto.optional_message.a = 1
        self.assertNotInitialized(proto)
        proto.optional_message.b = 0
        proto.optional_message.c = 0
        self.assertInitialized(proto)
        message1 = proto.repeated_message.add()
        self.assertNotInitialized(proto)
        message1.a = message1.b = message1.c = 0
        self.assertInitialized(proto)
        proto = unittest_pb2.TestAllExtensions()
        extension = unittest_pb2.TestRequired.multi
        message1 = proto.Extensions[extension].add()
        message2 = proto.Extensions[extension].add()
        self.assertNotInitialized(proto)
        message1.a = 1
        message1.b = 1
        message1.c = 1
        self.assertNotInitialized(proto)
        message2.a = 2
        message2.b = 2
        message2.c = 2
        self.assertInitialized(proto)
        proto = unittest_pb2.TestAllExtensions()
        extension = unittest_pb2.TestRequired.single
        proto.Extensions[extension].a = 1
        self.assertNotInitialized(proto)
        proto.Extensions[extension].b = 2
        proto.Extensions[extension].c = 3
        self.assertInitialized(proto)
        errors = []
        proto = unittest_pb2.TestRequired()
        self.assertFalse(proto.IsInitialized(errors))
        self.assertEqual(errors, ['a', 'b', 'c'])

    @unittest.skipIf(api_implementation.Type() != 'cpp' or api_implementation.Version() != 2, 'Errors are only available from the most recent C++ implementation.')
    def testFileDescriptorErrors(self):
        if False:
            while True:
                i = 10
        file_name = 'test_file_descriptor_errors.proto'
        package_name = 'test_file_descriptor_errors.proto'
        file_descriptor_proto = descriptor_pb2.FileDescriptorProto()
        file_descriptor_proto.name = file_name
        file_descriptor_proto.package = package_name
        m1 = file_descriptor_proto.message_type.add()
        m1.name = 'msg1'
        descriptor.FileDescriptor(file_name, package_name, serialized_pb=file_descriptor_proto.SerializeToString())
        another_file_name = 'another_test_file_descriptor_errors.proto'
        file_descriptor_proto.name = another_file_name
        m2 = file_descriptor_proto.message_type.add()
        m2.name = 'msg2'
        with self.assertRaises(TypeError) as cm:
            descriptor.FileDescriptor(another_file_name, package_name, serialized_pb=file_descriptor_proto.SerializeToString())
            self.assertTrue(hasattr(cm, 'exception'), '%s not raised' % getattr(cm.expected, '__name__', cm.expected))
            self.assertIn('test_file_descriptor_errors.proto', str(cm.exception))
            self.assertIn('test_file_descriptor_errors.msg1', str(cm.exception))

    def testStringUTF8Encoding(self):
        if False:
            return 10
        proto = unittest_pb2.TestAllTypes()
        self.assertRaises(TypeError, setattr, proto, 'optional_bytes', u'unicode object')
        self.assertEqual(type(proto.optional_string), six.text_type)
        proto.optional_string = six.text_type('Testing')
        self.assertEqual(proto.optional_string, str('Testing'))
        proto.optional_string = str('Testing')
        self.assertEqual(proto.optional_string, six.text_type('Testing'))
        self.assertRaises(ValueError, setattr, proto, 'optional_string', b'a\x80a')
        utf8_bytes = u'Тест'.encode('utf-8')
        proto.optional_string = utf8_bytes
        proto.optional_string = u'Тест'
        proto.optional_string = 'abc'

    def testStringUTF8Serialization(self):
        if False:
            for i in range(10):
                print('nop')
        proto = message_set_extensions_pb2.TestMessageSet()
        extension_message = message_set_extensions_pb2.TestMessageSetExtension2
        extension = extension_message.message_set_extension
        test_utf8 = u'Тест'
        test_utf8_bytes = test_utf8.encode('utf-8')
        proto.Extensions[extension].str = test_utf8
        serialized = proto.SerializeToString()
        self.assertEqual(proto.ByteSize(), len(serialized))
        raw = unittest_mset_pb2.RawMessageSet()
        bytes_read = raw.MergeFromString(serialized)
        self.assertEqual(len(serialized), bytes_read)
        message2 = message_set_extensions_pb2.TestMessageSetExtension2()
        self.assertEqual(1, len(raw.item))
        self.assertEqual(raw.item[0].type_id, 98418634)
        self.assertTrue(raw.item[0].message.endswith(test_utf8_bytes))
        bytes_read = message2.MergeFromString(raw.item[0].message)
        self.assertEqual(len(raw.item[0].message), bytes_read)
        self.assertEqual(type(message2.str), six.text_type)
        self.assertEqual(message2.str, test_utf8)
        badbytes = raw.item[0].message.replace(test_utf8_bytes, len(test_utf8_bytes) * b'\xff')
        unicode_decode_failed = False
        try:
            message2.MergeFromString(badbytes)
        except UnicodeDecodeError:
            unicode_decode_failed = True
        string_field = message2.str
        self.assertTrue(unicode_decode_failed or type(string_field) is bytes)

    def testBytesInTextFormat(self):
        if False:
            return 10
        proto = unittest_pb2.TestAllTypes(optional_bytes=b'\x00\x7f\x80\xff')
        self.assertEqual(u'optional_bytes: "\\000\\177\\200\\377"\n', six.text_type(proto))

    def testEmptyNestedMessage(self):
        if False:
            while True:
                i = 10
        proto = unittest_pb2.TestAllTypes()
        proto.optional_nested_message.MergeFrom(unittest_pb2.TestAllTypes.NestedMessage())
        self.assertTrue(proto.HasField('optional_nested_message'))
        proto = unittest_pb2.TestAllTypes()
        proto.optional_nested_message.CopyFrom(unittest_pb2.TestAllTypes.NestedMessage())
        self.assertTrue(proto.HasField('optional_nested_message'))
        proto = unittest_pb2.TestAllTypes()
        bytes_read = proto.optional_nested_message.MergeFromString(b'')
        self.assertEqual(0, bytes_read)
        self.assertTrue(proto.HasField('optional_nested_message'))
        proto = unittest_pb2.TestAllTypes()
        proto.optional_nested_message.ParseFromString(b'')
        self.assertTrue(proto.HasField('optional_nested_message'))
        serialized = proto.SerializeToString()
        proto2 = unittest_pb2.TestAllTypes()
        self.assertEqual(len(serialized), proto2.MergeFromString(serialized))
        self.assertTrue(proto2.HasField('optional_nested_message'))

    def testSetInParent(self):
        if False:
            for i in range(10):
                print('nop')
        proto = unittest_pb2.TestAllTypes()
        self.assertFalse(proto.HasField('optionalgroup'))
        proto.optionalgroup.SetInParent()
        self.assertTrue(proto.HasField('optionalgroup'))

    def testPackageInitializationImport(self):
        if False:
            i = 10
            return i + 15
        "Test that we can import nested messages from their __init__.py.\n\n    Such setup is not trivial since at the time of processing of __init__.py one\n    can't refer to its submodules by name in code, so expressions like\n    google.protobuf.internal.import_test_package.inner_pb2\n    don't work. They do work in imports, so we have assign an alias at import\n    and then use that alias in generated code.\n    "
        from google.protobuf.internal import import_test_package
        msg = import_test_package.myproto.Outer()
        self.assertEqual(57, msg.inner.value)

class TestAllTypesEqualityTest(BaseTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.first_proto = unittest_pb2.TestAllTypes()
        self.second_proto = unittest_pb2.TestAllTypes()

    def testNotHashable(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, hash, self.first_proto)

    def testSelfEquality(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.first_proto, self.first_proto)

    def testEmptyProtosEqual(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.first_proto, self.second_proto)

class FullProtosEqualityTest(BaseTestCase):
    """Equality tests using completely-full protos as a starting point."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.first_proto = unittest_pb2.TestAllTypes()
        self.second_proto = unittest_pb2.TestAllTypes()
        test_util.SetAllFields(self.first_proto)
        test_util.SetAllFields(self.second_proto)

    def testNotHashable(self):
        if False:
            return 10
        self.assertRaises(TypeError, hash, self.first_proto)

    def testNoneNotEqual(self):
        if False:
            return 10
        self.assertNotEqual(self.first_proto, None)
        self.assertNotEqual(None, self.second_proto)

    def testNotEqualToOtherMessage(self):
        if False:
            for i in range(10):
                print('nop')
        third_proto = unittest_pb2.TestRequired()
        self.assertNotEqual(self.first_proto, third_proto)
        self.assertNotEqual(third_proto, self.second_proto)

    def testAllFieldsFilledEquality(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.first_proto, self.second_proto)

    def testNonRepeatedScalar(self):
        if False:
            for i in range(10):
                print('nop')
        self.first_proto.optional_int32 += 1
        self.assertNotEqual(self.first_proto, self.second_proto)
        self.first_proto.ClearField('optional_int32')
        self.assertNotEqual(self.first_proto, self.second_proto)

    def testNonRepeatedComposite(self):
        if False:
            return 10
        self.first_proto.optional_nested_message.bb += 1
        self.assertNotEqual(self.first_proto, self.second_proto)
        self.first_proto.optional_nested_message.bb -= 1
        self.assertEqual(self.first_proto, self.second_proto)
        self.first_proto.optional_nested_message.ClearField('bb')
        self.assertNotEqual(self.first_proto, self.second_proto)
        self.first_proto.optional_nested_message.bb = self.second_proto.optional_nested_message.bb
        self.assertEqual(self.first_proto, self.second_proto)
        self.first_proto.ClearField('optional_nested_message')
        self.assertNotEqual(self.first_proto, self.second_proto)

    def testRepeatedScalar(self):
        if False:
            for i in range(10):
                print('nop')
        self.first_proto.repeated_int32.append(5)
        self.assertNotEqual(self.first_proto, self.second_proto)
        self.first_proto.ClearField('repeated_int32')
        self.assertNotEqual(self.first_proto, self.second_proto)

    def testRepeatedComposite(self):
        if False:
            print('Hello World!')
        self.first_proto.repeated_nested_message[0].bb += 1
        self.assertNotEqual(self.first_proto, self.second_proto)
        self.first_proto.repeated_nested_message[0].bb -= 1
        self.assertEqual(self.first_proto, self.second_proto)
        self.first_proto.repeated_nested_message.add()
        self.assertNotEqual(self.first_proto, self.second_proto)
        self.second_proto.repeated_nested_message.add()
        self.assertEqual(self.first_proto, self.second_proto)

    def testNonRepeatedScalarHasBits(self):
        if False:
            i = 10
            return i + 15
        self.first_proto.ClearField('optional_int32')
        self.second_proto.optional_int32 = 0
        self.assertNotEqual(self.first_proto, self.second_proto)

    def testNonRepeatedCompositeHasBits(self):
        if False:
            print('Hello World!')
        self.first_proto.ClearField('optional_nested_message')
        self.second_proto.optional_nested_message.ClearField('bb')
        self.assertNotEqual(self.first_proto, self.second_proto)
        self.first_proto.optional_nested_message.bb = 0
        self.first_proto.optional_nested_message.ClearField('bb')
        self.assertEqual(self.first_proto, self.second_proto)

class ExtensionEqualityTest(BaseTestCase):

    def testExtensionEquality(self):
        if False:
            for i in range(10):
                print('nop')
        first_proto = unittest_pb2.TestAllExtensions()
        second_proto = unittest_pb2.TestAllExtensions()
        self.assertEqual(first_proto, second_proto)
        test_util.SetAllExtensions(first_proto)
        self.assertNotEqual(first_proto, second_proto)
        test_util.SetAllExtensions(second_proto)
        self.assertEqual(first_proto, second_proto)
        first_proto.Extensions[unittest_pb2.optional_int32_extension] += 1
        self.assertNotEqual(first_proto, second_proto)
        first_proto.Extensions[unittest_pb2.optional_int32_extension] -= 1
        self.assertEqual(first_proto, second_proto)
        first_proto.ClearExtension(unittest_pb2.optional_int32_extension)
        second_proto.Extensions[unittest_pb2.optional_int32_extension] = 0
        self.assertNotEqual(first_proto, second_proto)
        first_proto.Extensions[unittest_pb2.optional_int32_extension] = 0
        self.assertEqual(first_proto, second_proto)
        first_proto = unittest_pb2.TestAllExtensions()
        second_proto = unittest_pb2.TestAllExtensions()
        self.assertEqual(0, first_proto.Extensions[unittest_pb2.optional_int32_extension])
        self.assertEqual(first_proto, second_proto)

class MutualRecursionEqualityTest(BaseTestCase):

    def testEqualityWithMutualRecursion(self):
        if False:
            return 10
        first_proto = unittest_pb2.TestMutualRecursionA()
        second_proto = unittest_pb2.TestMutualRecursionA()
        self.assertEqual(first_proto, second_proto)
        first_proto.bb.a.bb.optional_int32 = 23
        self.assertNotEqual(first_proto, second_proto)
        second_proto.bb.a.bb.optional_int32 = 23
        self.assertEqual(first_proto, second_proto)

class ByteSizeTest(BaseTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.proto = unittest_pb2.TestAllTypes()
        self.extended_proto = more_extensions_pb2.ExtendedMessage()
        self.packed_proto = unittest_pb2.TestPackedTypes()
        self.packed_extended_proto = unittest_pb2.TestPackedExtensions()

    def Size(self):
        if False:
            for i in range(10):
                print('nop')
        return self.proto.ByteSize()

    def testEmptyMessage(self):
        if False:
            print('Hello World!')
        self.assertEqual(0, self.proto.ByteSize())

    def testSizedOnKwargs(self):
        if False:
            return 10
        proto = unittest_pb2.TestAllTypes()
        self.assertEqual(0, proto.ByteSize())
        proto_kwargs = unittest_pb2.TestAllTypes(optional_int64=1)
        self.assertEqual(2, proto_kwargs.ByteSize())

    def testVarints(self):
        if False:
            for i in range(10):
                print('nop')

        def Test(i, expected_varint_size):
            if False:
                while True:
                    i = 10
            self.proto.Clear()
            self.proto.optional_int64 = i
            self.assertEqual(expected_varint_size + 1, self.Size())
        Test(0, 1)
        Test(1, 1)
        for (i, num_bytes) in zip(range(7, 63, 7), range(1, 10000)):
            Test((1 << i) - 1, num_bytes)
        Test(-1, 10)
        Test(-2, 10)
        Test(-(1 << 63), 10)

    def testStrings(self):
        if False:
            print('Hello World!')
        self.proto.optional_string = ''
        self.assertEqual(2, self.Size())
        self.proto.optional_string = 'abc'
        self.assertEqual(2 + len(self.proto.optional_string), self.Size())
        self.proto.optional_string = 'x' * 128
        self.assertEqual(3 + len(self.proto.optional_string), self.Size())

    def testOtherNumerics(self):
        if False:
            print('Hello World!')
        self.proto.optional_fixed32 = 1234
        self.assertEqual(5, self.Size())
        self.proto = unittest_pb2.TestAllTypes()
        self.proto.optional_fixed64 = 1234
        self.assertEqual(9, self.Size())
        self.proto = unittest_pb2.TestAllTypes()
        self.proto.optional_float = 1.234
        self.assertEqual(5, self.Size())
        self.proto = unittest_pb2.TestAllTypes()
        self.proto.optional_double = 1.234
        self.assertEqual(9, self.Size())
        self.proto = unittest_pb2.TestAllTypes()
        self.proto.optional_sint32 = 64
        self.assertEqual(3, self.Size())
        self.proto = unittest_pb2.TestAllTypes()

    def testComposites(self):
        if False:
            return 10
        self.proto.optional_nested_message.bb = 1 << 14
        self.assertEqual(3 + 1 + 1 + 2, self.Size())

    def testGroups(self):
        if False:
            for i in range(10):
                print('nop')
        self.proto.optionalgroup.a = 1 << 21
        self.assertEqual(4 + 2 + 2 * 2, self.Size())

    def testRepeatedScalars(self):
        if False:
            print('Hello World!')
        self.proto.repeated_int32.append(10)
        self.proto.repeated_int32.append(128)
        self.assertEqual(1 + 2 + 2 * 2, self.Size())

    def testRepeatedScalarsExtend(self):
        if False:
            for i in range(10):
                print('nop')
        self.proto.repeated_int32.extend([10, 128])
        self.assertEqual(1 + 2 + 2 * 2, self.Size())

    def testRepeatedScalarsRemove(self):
        if False:
            for i in range(10):
                print('nop')
        self.proto.repeated_int32.append(10)
        self.proto.repeated_int32.append(128)
        self.assertEqual(1 + 2 + 2 * 2, self.Size())
        self.proto.repeated_int32.remove(128)
        self.assertEqual(1 + 2, self.Size())

    def testRepeatedComposites(self):
        if False:
            for i in range(10):
                print('nop')
        foreign_message_0 = self.proto.repeated_nested_message.add()
        foreign_message_1 = self.proto.repeated_nested_message.add()
        foreign_message_1.bb = 7
        self.assertEqual(2 + 1 + 2 + 1 + 1 + 1, self.Size())

    def testRepeatedCompositesDelete(self):
        if False:
            print('Hello World!')
        foreign_message_0 = self.proto.repeated_nested_message.add()
        foreign_message_1 = self.proto.repeated_nested_message.add()
        foreign_message_1.bb = 9
        self.assertEqual(2 + 1 + 2 + 1 + 1 + 1, self.Size())
        del self.proto.repeated_nested_message[0]
        self.assertEqual(2 + 1 + 1 + 1, self.Size())
        foreign_message_2 = self.proto.repeated_nested_message.add()
        foreign_message_2.bb = 12
        self.assertEqual(2 + 1 + 1 + 1 + 2 + 1 + 1 + 1, self.Size())
        del self.proto.repeated_nested_message[1]
        self.assertEqual(2 + 1 + 1 + 1, self.Size())
        del self.proto.repeated_nested_message[0]
        self.assertEqual(0, self.Size())

    def testRepeatedGroups(self):
        if False:
            return 10
        group_0 = self.proto.repeatedgroup.add()
        group_1 = self.proto.repeatedgroup.add()
        group_1.a = 7
        self.assertEqual(2 + 2 + 2 + 2 + 1 + 2, self.Size())

    def testExtensions(self):
        if False:
            return 10
        proto = unittest_pb2.TestAllExtensions()
        self.assertEqual(0, proto.ByteSize())
        extension = unittest_pb2.optional_int32_extension
        proto.Extensions[extension] = 23
        self.assertEqual(2, proto.ByteSize())

    def testCacheInvalidationForNonrepeatedScalar(self):
        if False:
            i = 10
            return i + 15
        self.proto.optional_int32 = 1
        self.assertEqual(2, self.proto.ByteSize())
        self.proto.optional_int32 = 128
        self.assertEqual(3, self.proto.ByteSize())
        self.proto.ClearField('optional_int32')
        self.assertEqual(0, self.proto.ByteSize())
        extension = more_extensions_pb2.optional_int_extension
        self.extended_proto.Extensions[extension] = 1
        self.assertEqual(2, self.extended_proto.ByteSize())
        self.extended_proto.Extensions[extension] = 128
        self.assertEqual(3, self.extended_proto.ByteSize())
        self.extended_proto.ClearExtension(extension)
        self.assertEqual(0, self.extended_proto.ByteSize())

    def testCacheInvalidationForRepeatedScalar(self):
        if False:
            while True:
                i = 10
        self.proto.repeated_int32.append(1)
        self.assertEqual(3, self.proto.ByteSize())
        self.proto.repeated_int32.append(1)
        self.assertEqual(6, self.proto.ByteSize())
        self.proto.repeated_int32[1] = 128
        self.assertEqual(7, self.proto.ByteSize())
        self.proto.ClearField('repeated_int32')
        self.assertEqual(0, self.proto.ByteSize())
        extension = more_extensions_pb2.repeated_int_extension
        repeated = self.extended_proto.Extensions[extension]
        repeated.append(1)
        self.assertEqual(2, self.extended_proto.ByteSize())
        repeated.append(1)
        self.assertEqual(4, self.extended_proto.ByteSize())
        repeated[1] = 128
        self.assertEqual(5, self.extended_proto.ByteSize())
        self.extended_proto.ClearExtension(extension)
        self.assertEqual(0, self.extended_proto.ByteSize())

    def testCacheInvalidationForNonrepeatedMessage(self):
        if False:
            for i in range(10):
                print('nop')
        self.proto.optional_foreign_message.c = 1
        self.assertEqual(5, self.proto.ByteSize())
        self.proto.optional_foreign_message.c = 128
        self.assertEqual(6, self.proto.ByteSize())
        self.proto.optional_foreign_message.ClearField('c')
        self.assertEqual(3, self.proto.ByteSize())
        self.proto.ClearField('optional_foreign_message')
        self.assertEqual(0, self.proto.ByteSize())
        if api_implementation.Type() == 'python':
            child = self.proto.optional_foreign_message
            self.proto.ClearField('optional_foreign_message')
            child.c = 128
            self.assertEqual(0, self.proto.ByteSize())
        extension = more_extensions_pb2.optional_message_extension
        child = self.extended_proto.Extensions[extension]
        self.assertEqual(0, self.extended_proto.ByteSize())
        child.foreign_message_int = 1
        self.assertEqual(4, self.extended_proto.ByteSize())
        child.foreign_message_int = 128
        self.assertEqual(5, self.extended_proto.ByteSize())
        self.extended_proto.ClearExtension(extension)
        self.assertEqual(0, self.extended_proto.ByteSize())

    def testCacheInvalidationForRepeatedMessage(self):
        if False:
            i = 10
            return i + 15
        child0 = self.proto.repeated_foreign_message.add()
        self.assertEqual(3, self.proto.ByteSize())
        self.proto.repeated_foreign_message.add()
        self.assertEqual(6, self.proto.ByteSize())
        child0.c = 1
        self.assertEqual(8, self.proto.ByteSize())
        self.proto.ClearField('repeated_foreign_message')
        self.assertEqual(0, self.proto.ByteSize())
        extension = more_extensions_pb2.repeated_message_extension
        child_list = self.extended_proto.Extensions[extension]
        child0 = child_list.add()
        self.assertEqual(2, self.extended_proto.ByteSize())
        child_list.add()
        self.assertEqual(4, self.extended_proto.ByteSize())
        child0.foreign_message_int = 1
        self.assertEqual(6, self.extended_proto.ByteSize())
        child0.ClearField('foreign_message_int')
        self.assertEqual(4, self.extended_proto.ByteSize())
        self.extended_proto.ClearExtension(extension)
        self.assertEqual(0, self.extended_proto.ByteSize())

    def testPackedRepeatedScalars(self):
        if False:
            while True:
                i = 10
        self.assertEqual(0, self.packed_proto.ByteSize())
        self.packed_proto.packed_int32.append(10)
        self.packed_proto.packed_int32.append(128)
        int_size = 1 + 2 + 3
        self.assertEqual(int_size, self.packed_proto.ByteSize())
        self.packed_proto.packed_double.append(4.2)
        self.packed_proto.packed_double.append(3.25)
        double_size = 8 + 8 + 3
        self.assertEqual(int_size + double_size, self.packed_proto.ByteSize())
        self.packed_proto.ClearField('packed_int32')
        self.assertEqual(double_size, self.packed_proto.ByteSize())

    def testPackedExtensions(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(0, self.packed_extended_proto.ByteSize())
        extension = self.packed_extended_proto.Extensions[unittest_pb2.packed_fixed32_extension]
        extension.extend([1, 2, 3, 4])
        self.assertEqual(19, self.packed_extended_proto.ByteSize())

class SerializationTest(BaseTestCase):

    def testSerializeEmtpyMessage(self):
        if False:
            for i in range(10):
                print('nop')
        first_proto = unittest_pb2.TestAllTypes()
        second_proto = unittest_pb2.TestAllTypes()
        serialized = first_proto.SerializeToString()
        self.assertEqual(first_proto.ByteSize(), len(serialized))
        self.assertEqual(len(serialized), second_proto.MergeFromString(serialized))
        self.assertEqual(first_proto, second_proto)

    def testSerializeAllFields(self):
        if False:
            i = 10
            return i + 15
        first_proto = unittest_pb2.TestAllTypes()
        second_proto = unittest_pb2.TestAllTypes()
        test_util.SetAllFields(first_proto)
        serialized = first_proto.SerializeToString()
        self.assertEqual(first_proto.ByteSize(), len(serialized))
        self.assertEqual(len(serialized), second_proto.MergeFromString(serialized))
        self.assertEqual(first_proto, second_proto)

    def testSerializeAllExtensions(self):
        if False:
            return 10
        first_proto = unittest_pb2.TestAllExtensions()
        second_proto = unittest_pb2.TestAllExtensions()
        test_util.SetAllExtensions(first_proto)
        serialized = first_proto.SerializeToString()
        self.assertEqual(len(serialized), second_proto.MergeFromString(serialized))
        self.assertEqual(first_proto, second_proto)

    def testSerializeWithOptionalGroup(self):
        if False:
            i = 10
            return i + 15
        first_proto = unittest_pb2.TestAllTypes()
        second_proto = unittest_pb2.TestAllTypes()
        first_proto.optionalgroup.a = 242
        serialized = first_proto.SerializeToString()
        self.assertEqual(len(serialized), second_proto.MergeFromString(serialized))
        self.assertEqual(first_proto, second_proto)

    def testSerializeNegativeValues(self):
        if False:
            i = 10
            return i + 15
        first_proto = unittest_pb2.TestAllTypes()
        first_proto.optional_int32 = -1
        first_proto.optional_int64 = -(2 << 40)
        first_proto.optional_sint32 = -3
        first_proto.optional_sint64 = -(4 << 40)
        first_proto.optional_sfixed32 = -5
        first_proto.optional_sfixed64 = -(6 << 40)
        second_proto = unittest_pb2.TestAllTypes.FromString(first_proto.SerializeToString())
        self.assertEqual(first_proto, second_proto)

    def testParseTruncated(self):
        if False:
            while True:
                i = 10
        if api_implementation.Type() != 'python':
            return
        first_proto = unittest_pb2.TestAllTypes()
        test_util.SetAllFields(first_proto)
        serialized = first_proto.SerializeToString()
        for truncation_point in range(len(serialized) + 1):
            try:
                second_proto = unittest_pb2.TestAllTypes()
                unknown_fields = unittest_pb2.TestEmptyMessage()
                pos = second_proto._InternalParse(serialized, 0, truncation_point)
                self.assertEqual(truncation_point, pos)
                try:
                    pos2 = unknown_fields._InternalParse(serialized, 0, truncation_point)
                    self.assertEqual(truncation_point, pos2)
                except message.DecodeError:
                    self.fail('Parsing unknown fields failed when parsing known fields did not.')
            except message.DecodeError:
                self.assertRaises(message.DecodeError, unknown_fields._InternalParse, serialized, 0, truncation_point)

    def testCanonicalSerializationOrder(self):
        if False:
            return 10
        proto = more_messages_pb2.OutOfOrderFields()
        proto.optional_sint32 = 5
        proto.Extensions[more_messages_pb2.optional_uint64] = 4
        proto.optional_uint32 = 3
        proto.Extensions[more_messages_pb2.optional_int64] = 2
        proto.optional_int32 = 1
        serialized = proto.SerializeToString()
        self.assertEqual(proto.ByteSize(), len(serialized))
        d = _MiniDecoder(serialized)
        ReadTag = d.ReadFieldNumberAndWireType
        self.assertEqual((1, wire_format.WIRETYPE_VARINT), ReadTag())
        self.assertEqual(1, d.ReadInt32())
        self.assertEqual((2, wire_format.WIRETYPE_VARINT), ReadTag())
        self.assertEqual(2, d.ReadInt64())
        self.assertEqual((3, wire_format.WIRETYPE_VARINT), ReadTag())
        self.assertEqual(3, d.ReadUInt32())
        self.assertEqual((4, wire_format.WIRETYPE_VARINT), ReadTag())
        self.assertEqual(4, d.ReadUInt64())
        self.assertEqual((5, wire_format.WIRETYPE_VARINT), ReadTag())
        self.assertEqual(5, d.ReadSInt32())

    def testCanonicalSerializationOrderSameAsCpp(self):
        if False:
            return 10
        proto = unittest_pb2.TestFieldOrderings()
        test_util.SetAllFieldsAndExtensions(proto)
        serialized = proto.SerializeToString()
        test_util.ExpectAllFieldsAndExtensionsInOrder(serialized)

    def testMergeFromStringWhenFieldsAlreadySet(self):
        if False:
            print('Hello World!')
        first_proto = unittest_pb2.TestAllTypes()
        first_proto.repeated_string.append('foobar')
        first_proto.optional_int32 = 23
        first_proto.optional_nested_message.bb = 42
        serialized = first_proto.SerializeToString()
        second_proto = unittest_pb2.TestAllTypes()
        second_proto.repeated_string.append('baz')
        second_proto.optional_int32 = 100
        second_proto.optional_nested_message.bb = 999
        bytes_parsed = second_proto.MergeFromString(serialized)
        self.assertEqual(len(serialized), bytes_parsed)
        self.assertEqual(['baz', 'foobar'], list(second_proto.repeated_string))
        self.assertEqual(23, second_proto.optional_int32)
        self.assertEqual(42, second_proto.optional_nested_message.bb)

    def testMessageSetWireFormat(self):
        if False:
            for i in range(10):
                print('nop')
        proto = message_set_extensions_pb2.TestMessageSet()
        extension_message1 = message_set_extensions_pb2.TestMessageSetExtension1
        extension_message2 = message_set_extensions_pb2.TestMessageSetExtension2
        extension1 = extension_message1.message_set_extension
        extension2 = extension_message2.message_set_extension
        extension3 = message_set_extensions_pb2.message_set_extension3
        proto.Extensions[extension1].i = 123
        proto.Extensions[extension2].str = 'foo'
        proto.Extensions[extension3].text = 'bar'
        serialized = proto.SerializeToString()
        raw = unittest_mset_pb2.RawMessageSet()
        self.assertEqual(False, raw.DESCRIPTOR.GetOptions().message_set_wire_format)
        self.assertEqual(len(serialized), raw.MergeFromString(serialized))
        self.assertEqual(3, len(raw.item))
        message1 = message_set_extensions_pb2.TestMessageSetExtension1()
        self.assertEqual(len(raw.item[0].message), message1.MergeFromString(raw.item[0].message))
        self.assertEqual(123, message1.i)
        message2 = message_set_extensions_pb2.TestMessageSetExtension2()
        self.assertEqual(len(raw.item[1].message), message2.MergeFromString(raw.item[1].message))
        self.assertEqual('foo', message2.str)
        message3 = message_set_extensions_pb2.TestMessageSetExtension3()
        self.assertEqual(len(raw.item[2].message), message3.MergeFromString(raw.item[2].message))
        self.assertEqual('bar', message3.text)
        proto2 = message_set_extensions_pb2.TestMessageSet()
        self.assertEqual(len(serialized), proto2.MergeFromString(serialized))
        self.assertEqual(123, proto2.Extensions[extension1].i)
        self.assertEqual('foo', proto2.Extensions[extension2].str)
        self.assertEqual('bar', proto2.Extensions[extension3].text)
        self.assertEqual(proto2.ByteSize(), len(serialized))
        self.assertEqual(proto.ByteSize(), len(serialized))

    def testMessageSetWireFormatUnknownExtension(self):
        if False:
            while True:
                i = 10
        raw = unittest_mset_pb2.RawMessageSet()
        item = raw.item.add()
        item.type_id = 98418603
        extension_message1 = message_set_extensions_pb2.TestMessageSetExtension1
        message1 = message_set_extensions_pb2.TestMessageSetExtension1()
        message1.i = 12345
        item.message = message1.SerializeToString()
        item = raw.item.add()
        item.type_id = 98418604
        extension_message1 = message_set_extensions_pb2.TestMessageSetExtension1
        message1 = message_set_extensions_pb2.TestMessageSetExtension1()
        message1.i = 12346
        item.message = message1.SerializeToString()
        item = raw.item.add()
        item.type_id = 98418605
        message1 = message_set_extensions_pb2.TestMessageSetExtension2()
        message1.str = 'foo'
        item.message = message1.SerializeToString()
        serialized = raw.SerializeToString()
        proto = message_set_extensions_pb2.TestMessageSet()
        self.assertEqual(len(serialized), proto.MergeFromString(serialized))
        extension_message1 = message_set_extensions_pb2.TestMessageSetExtension1
        extension1 = extension_message1.message_set_extension
        self.assertEqual(12345, proto.Extensions[extension1].i)

    def testUnknownFields(self):
        if False:
            for i in range(10):
                print('nop')
        proto = unittest_pb2.TestAllTypes()
        test_util.SetAllFields(proto)
        serialized = proto.SerializeToString()
        proto2 = unittest_pb2.TestEmptyMessage()
        self.assertEqual(len(serialized), proto2.MergeFromString(serialized))
        proto = unittest_pb2.TestAllTypes()
        proto.optional_int64 = 1152921504606846975
        serialized = proto.SerializeToString()
        proto2 = unittest_pb2.TestEmptyMessage()
        self.assertEqual(len(serialized), proto2.MergeFromString(serialized))

    def _CheckRaises(self, exc_class, callable_obj, exception):
        if False:
            for i in range(10):
                print('nop')
        'This method checks if the excpetion type and message are as expected.'
        try:
            callable_obj()
        except exc_class as ex:
            self.assertEqual(exception, str(ex))
            return
        else:
            raise self.failureException('%s not raised' % str(exc_class))

    def testSerializeUninitialized(self):
        if False:
            while True:
                i = 10
        proto = unittest_pb2.TestRequired()
        self._CheckRaises(message.EncodeError, proto.SerializeToString, 'Message protobuf_unittest.TestRequired is missing required fields: a,b,c')
        partial = proto.SerializePartialToString()
        proto2 = unittest_pb2.TestRequired()
        self.assertFalse(proto2.HasField('a'))
        proto2.ParseFromString(partial)
        self.assertFalse(proto2.HasField('a'))
        proto.a = 1
        self._CheckRaises(message.EncodeError, proto.SerializeToString, 'Message protobuf_unittest.TestRequired is missing required fields: b,c')
        partial = proto.SerializePartialToString()
        proto.b = 2
        self._CheckRaises(message.EncodeError, proto.SerializeToString, 'Message protobuf_unittest.TestRequired is missing required fields: c')
        partial = proto.SerializePartialToString()
        proto.c = 3
        serialized = proto.SerializeToString()
        partial = proto.SerializePartialToString()
        proto2 = unittest_pb2.TestRequired()
        self.assertEqual(len(serialized), proto2.MergeFromString(serialized))
        self.assertEqual(1, proto2.a)
        self.assertEqual(2, proto2.b)
        self.assertEqual(3, proto2.c)
        self.assertEqual(len(partial), proto2.MergeFromString(partial))
        self.assertEqual(1, proto2.a)
        self.assertEqual(2, proto2.b)
        self.assertEqual(3, proto2.c)

    def testSerializeUninitializedSubMessage(self):
        if False:
            return 10
        proto = unittest_pb2.TestRequiredForeign()
        proto.SerializeToString()
        proto.optional_message.a = 1
        self._CheckRaises(message.EncodeError, proto.SerializeToString, 'Message protobuf_unittest.TestRequiredForeign is missing required fields: optional_message.b,optional_message.c')
        proto.optional_message.b = 2
        proto.optional_message.c = 3
        proto.SerializeToString()
        proto.repeated_message.add().a = 1
        proto.repeated_message.add().b = 2
        self._CheckRaises(message.EncodeError, proto.SerializeToString, 'Message protobuf_unittest.TestRequiredForeign is missing required fields: repeated_message[0].b,repeated_message[0].c,repeated_message[1].a,repeated_message[1].c')
        proto.repeated_message[0].b = 2
        proto.repeated_message[0].c = 3
        proto.repeated_message[1].a = 1
        proto.repeated_message[1].c = 3
        proto.SerializeToString()

    def testSerializeAllPackedFields(self):
        if False:
            print('Hello World!')
        first_proto = unittest_pb2.TestPackedTypes()
        second_proto = unittest_pb2.TestPackedTypes()
        test_util.SetAllPackedFields(first_proto)
        serialized = first_proto.SerializeToString()
        self.assertEqual(first_proto.ByteSize(), len(serialized))
        bytes_read = second_proto.MergeFromString(serialized)
        self.assertEqual(second_proto.ByteSize(), bytes_read)
        self.assertEqual(first_proto, second_proto)

    def testSerializeAllPackedExtensions(self):
        if False:
            i = 10
            return i + 15
        first_proto = unittest_pb2.TestPackedExtensions()
        second_proto = unittest_pb2.TestPackedExtensions()
        test_util.SetAllPackedExtensions(first_proto)
        serialized = first_proto.SerializeToString()
        bytes_read = second_proto.MergeFromString(serialized)
        self.assertEqual(second_proto.ByteSize(), bytes_read)
        self.assertEqual(first_proto, second_proto)

    def testMergePackedFromStringWhenSomeFieldsAlreadySet(self):
        if False:
            i = 10
            return i + 15
        first_proto = unittest_pb2.TestPackedTypes()
        first_proto.packed_int32.extend([1, 2])
        first_proto.packed_double.append(3.0)
        serialized = first_proto.SerializeToString()
        second_proto = unittest_pb2.TestPackedTypes()
        second_proto.packed_int32.append(3)
        second_proto.packed_double.extend([1.0, 2.0])
        second_proto.packed_sint32.append(4)
        self.assertEqual(len(serialized), second_proto.MergeFromString(serialized))
        self.assertEqual([3, 1, 2], second_proto.packed_int32)
        self.assertEqual([1.0, 2.0, 3.0], second_proto.packed_double)
        self.assertEqual([4], second_proto.packed_sint32)

    def testPackedFieldsWireFormat(self):
        if False:
            return 10
        proto = unittest_pb2.TestPackedTypes()
        proto.packed_int32.extend([1, 2, 150, 3])
        proto.packed_double.extend([1.0, 1000.0])
        proto.packed_float.append(2.0)
        serialized = proto.SerializeToString()
        self.assertEqual(proto.ByteSize(), len(serialized))
        d = _MiniDecoder(serialized)
        ReadTag = d.ReadFieldNumberAndWireType
        self.assertEqual((90, wire_format.WIRETYPE_LENGTH_DELIMITED), ReadTag())
        self.assertEqual(1 + 1 + 1 + 2, d.ReadInt32())
        self.assertEqual(1, d.ReadInt32())
        self.assertEqual(2, d.ReadInt32())
        self.assertEqual(150, d.ReadInt32())
        self.assertEqual(3, d.ReadInt32())
        self.assertEqual((100, wire_format.WIRETYPE_LENGTH_DELIMITED), ReadTag())
        self.assertEqual(4, d.ReadInt32())
        self.assertEqual(2.0, d.ReadFloat())
        self.assertEqual((101, wire_format.WIRETYPE_LENGTH_DELIMITED), ReadTag())
        self.assertEqual(8 + 8, d.ReadInt32())
        self.assertEqual(1.0, d.ReadDouble())
        self.assertEqual(1000.0, d.ReadDouble())
        self.assertTrue(d.EndOfStream())

    def testParsePackedFromUnpacked(self):
        if False:
            for i in range(10):
                print('nop')
        unpacked = unittest_pb2.TestUnpackedTypes()
        test_util.SetAllUnpackedFields(unpacked)
        packed = unittest_pb2.TestPackedTypes()
        serialized = unpacked.SerializeToString()
        self.assertEqual(len(serialized), packed.MergeFromString(serialized))
        expected = unittest_pb2.TestPackedTypes()
        test_util.SetAllPackedFields(expected)
        self.assertEqual(expected, packed)

    def testParseUnpackedFromPacked(self):
        if False:
            i = 10
            return i + 15
        packed = unittest_pb2.TestPackedTypes()
        test_util.SetAllPackedFields(packed)
        unpacked = unittest_pb2.TestUnpackedTypes()
        serialized = packed.SerializeToString()
        self.assertEqual(len(serialized), unpacked.MergeFromString(serialized))
        expected = unittest_pb2.TestUnpackedTypes()
        test_util.SetAllUnpackedFields(expected)
        self.assertEqual(expected, unpacked)

    def testFieldNumbers(self):
        if False:
            for i in range(10):
                print('nop')
        proto = unittest_pb2.TestAllTypes()
        self.assertEqual(unittest_pb2.TestAllTypes.NestedMessage.BB_FIELD_NUMBER, 1)
        self.assertEqual(unittest_pb2.TestAllTypes.OPTIONAL_INT32_FIELD_NUMBER, 1)
        self.assertEqual(unittest_pb2.TestAllTypes.OPTIONALGROUP_FIELD_NUMBER, 16)
        self.assertEqual(unittest_pb2.TestAllTypes.OPTIONAL_NESTED_MESSAGE_FIELD_NUMBER, 18)
        self.assertEqual(unittest_pb2.TestAllTypes.OPTIONAL_NESTED_ENUM_FIELD_NUMBER, 21)
        self.assertEqual(unittest_pb2.TestAllTypes.REPEATED_INT32_FIELD_NUMBER, 31)
        self.assertEqual(unittest_pb2.TestAllTypes.REPEATEDGROUP_FIELD_NUMBER, 46)
        self.assertEqual(unittest_pb2.TestAllTypes.REPEATED_NESTED_MESSAGE_FIELD_NUMBER, 48)
        self.assertEqual(unittest_pb2.TestAllTypes.REPEATED_NESTED_ENUM_FIELD_NUMBER, 51)

    def testExtensionFieldNumbers(self):
        if False:
            while True:
                i = 10
        self.assertEqual(unittest_pb2.TestRequired.single.number, 1000)
        self.assertEqual(unittest_pb2.TestRequired.SINGLE_FIELD_NUMBER, 1000)
        self.assertEqual(unittest_pb2.TestRequired.multi.number, 1001)
        self.assertEqual(unittest_pb2.TestRequired.MULTI_FIELD_NUMBER, 1001)
        self.assertEqual(unittest_pb2.optional_int32_extension.number, 1)
        self.assertEqual(unittest_pb2.OPTIONAL_INT32_EXTENSION_FIELD_NUMBER, 1)
        self.assertEqual(unittest_pb2.optionalgroup_extension.number, 16)
        self.assertEqual(unittest_pb2.OPTIONALGROUP_EXTENSION_FIELD_NUMBER, 16)
        self.assertEqual(unittest_pb2.optional_nested_message_extension.number, 18)
        self.assertEqual(unittest_pb2.OPTIONAL_NESTED_MESSAGE_EXTENSION_FIELD_NUMBER, 18)
        self.assertEqual(unittest_pb2.optional_nested_enum_extension.number, 21)
        self.assertEqual(unittest_pb2.OPTIONAL_NESTED_ENUM_EXTENSION_FIELD_NUMBER, 21)
        self.assertEqual(unittest_pb2.repeated_int32_extension.number, 31)
        self.assertEqual(unittest_pb2.REPEATED_INT32_EXTENSION_FIELD_NUMBER, 31)
        self.assertEqual(unittest_pb2.repeatedgroup_extension.number, 46)
        self.assertEqual(unittest_pb2.REPEATEDGROUP_EXTENSION_FIELD_NUMBER, 46)
        self.assertEqual(unittest_pb2.repeated_nested_message_extension.number, 48)
        self.assertEqual(unittest_pb2.REPEATED_NESTED_MESSAGE_EXTENSION_FIELD_NUMBER, 48)
        self.assertEqual(unittest_pb2.repeated_nested_enum_extension.number, 51)
        self.assertEqual(unittest_pb2.REPEATED_NESTED_ENUM_EXTENSION_FIELD_NUMBER, 51)

    def testInitKwargs(self):
        if False:
            i = 10
            return i + 15
        proto = unittest_pb2.TestAllTypes(optional_int32=1, optional_string='foo', optional_bool=True, optional_bytes=b'bar', optional_nested_message=unittest_pb2.TestAllTypes.NestedMessage(bb=1), optional_foreign_message=unittest_pb2.ForeignMessage(c=1), optional_nested_enum=unittest_pb2.TestAllTypes.FOO, optional_foreign_enum=unittest_pb2.FOREIGN_FOO, repeated_int32=[1, 2, 3])
        self.assertTrue(proto.IsInitialized())
        self.assertTrue(proto.HasField('optional_int32'))
        self.assertTrue(proto.HasField('optional_string'))
        self.assertTrue(proto.HasField('optional_bool'))
        self.assertTrue(proto.HasField('optional_bytes'))
        self.assertTrue(proto.HasField('optional_nested_message'))
        self.assertTrue(proto.HasField('optional_foreign_message'))
        self.assertTrue(proto.HasField('optional_nested_enum'))
        self.assertTrue(proto.HasField('optional_foreign_enum'))
        self.assertEqual(1, proto.optional_int32)
        self.assertEqual('foo', proto.optional_string)
        self.assertEqual(True, proto.optional_bool)
        self.assertEqual(b'bar', proto.optional_bytes)
        self.assertEqual(1, proto.optional_nested_message.bb)
        self.assertEqual(1, proto.optional_foreign_message.c)
        self.assertEqual(unittest_pb2.TestAllTypes.FOO, proto.optional_nested_enum)
        self.assertEqual(unittest_pb2.FOREIGN_FOO, proto.optional_foreign_enum)
        self.assertEqual([1, 2, 3], proto.repeated_int32)

    def testInitArgsUnknownFieldName(self):
        if False:
            for i in range(10):
                print('nop')

        def InitalizeEmptyMessageWithExtraKeywordArg():
            if False:
                i = 10
                return i + 15
            unused_proto = unittest_pb2.TestEmptyMessage(unknown='unknown')
        self._CheckRaises(ValueError, InitalizeEmptyMessageWithExtraKeywordArg, 'Protocol message TestEmptyMessage has no "unknown" field.')

    def testInitRequiredKwargs(self):
        if False:
            return 10
        proto = unittest_pb2.TestRequired(a=1, b=1, c=1)
        self.assertTrue(proto.IsInitialized())
        self.assertTrue(proto.HasField('a'))
        self.assertTrue(proto.HasField('b'))
        self.assertTrue(proto.HasField('c'))
        self.assertTrue(not proto.HasField('dummy2'))
        self.assertEqual(1, proto.a)
        self.assertEqual(1, proto.b)
        self.assertEqual(1, proto.c)

    def testInitRequiredForeignKwargs(self):
        if False:
            for i in range(10):
                print('nop')
        proto = unittest_pb2.TestRequiredForeign(optional_message=unittest_pb2.TestRequired(a=1, b=1, c=1))
        self.assertTrue(proto.IsInitialized())
        self.assertTrue(proto.HasField('optional_message'))
        self.assertTrue(proto.optional_message.IsInitialized())
        self.assertTrue(proto.optional_message.HasField('a'))
        self.assertTrue(proto.optional_message.HasField('b'))
        self.assertTrue(proto.optional_message.HasField('c'))
        self.assertTrue(not proto.optional_message.HasField('dummy2'))
        self.assertEqual(unittest_pb2.TestRequired(a=1, b=1, c=1), proto.optional_message)
        self.assertEqual(1, proto.optional_message.a)
        self.assertEqual(1, proto.optional_message.b)
        self.assertEqual(1, proto.optional_message.c)

    def testInitRepeatedKwargs(self):
        if False:
            while True:
                i = 10
        proto = unittest_pb2.TestAllTypes(repeated_int32=[1, 2, 3])
        self.assertTrue(proto.IsInitialized())
        self.assertEqual(1, proto.repeated_int32[0])
        self.assertEqual(2, proto.repeated_int32[1])
        self.assertEqual(3, proto.repeated_int32[2])

class OptionsTest(BaseTestCase):

    def testMessageOptions(self):
        if False:
            print('Hello World!')
        proto = message_set_extensions_pb2.TestMessageSet()
        self.assertEqual(True, proto.DESCRIPTOR.GetOptions().message_set_wire_format)
        proto = unittest_pb2.TestAllTypes()
        self.assertEqual(False, proto.DESCRIPTOR.GetOptions().message_set_wire_format)

    def testPackedOptions(self):
        if False:
            print('Hello World!')
        proto = unittest_pb2.TestAllTypes()
        proto.optional_int32 = 1
        proto.optional_double = 3.0
        for (field_descriptor, _) in proto.ListFields():
            self.assertEqual(False, field_descriptor.GetOptions().packed)
        proto = unittest_pb2.TestPackedTypes()
        proto.packed_int32.append(1)
        proto.packed_double.append(3.0)
        for (field_descriptor, _) in proto.ListFields():
            self.assertEqual(True, field_descriptor.GetOptions().packed)
            self.assertEqual(descriptor.FieldDescriptor.LABEL_REPEATED, field_descriptor.label)

class ClassAPITest(BaseTestCase):

    @unittest.skipIf(api_implementation.Type() == 'cpp' and api_implementation.Version() == 2, 'C++ implementation requires a call to MakeDescriptor()')
    def testMakeClassWithNestedDescriptor(self):
        if False:
            i = 10
            return i + 15
        leaf_desc = descriptor.Descriptor('leaf', 'package.parent.child.leaf', '', containing_type=None, fields=[], nested_types=[], enum_types=[], extensions=[])
        child_desc = descriptor.Descriptor('child', 'package.parent.child', '', containing_type=None, fields=[], nested_types=[leaf_desc], enum_types=[], extensions=[])
        sibling_desc = descriptor.Descriptor('sibling', 'package.parent.sibling', '', containing_type=None, fields=[], nested_types=[], enum_types=[], extensions=[])
        parent_desc = descriptor.Descriptor('parent', 'package.parent', '', containing_type=None, fields=[], nested_types=[child_desc, sibling_desc], enum_types=[], extensions=[])
        message_class = reflection.MakeClass(parent_desc)
        self.assertIn('child', message_class.__dict__)
        self.assertIn('sibling', message_class.__dict__)
        self.assertIn('leaf', message_class.child.__dict__)

    def _GetSerializedFileDescriptor(self, name):
        if False:
            i = 10
            return i + 15
        'Get a serialized representation of a test FileDescriptorProto.\n\n    Args:\n      name: All calls to this must use a unique message name, to avoid\n          collisions in the cpp descriptor pool.\n    Returns:\n      A string containing the serialized form of a test FileDescriptorProto.\n    '
        file_descriptor_str = 'message_type {  name: "' + name + '"  field {    name: "flat"    number: 1    label: LABEL_REPEATED    type: TYPE_UINT32  }  field {    name: "bar"    number: 2    label: LABEL_OPTIONAL    type: TYPE_MESSAGE    type_name: "Bar"  }  nested_type {    name: "Bar"    field {      name: "baz"      number: 3      label: LABEL_OPTIONAL      type: TYPE_MESSAGE      type_name: "Baz"    }    nested_type {      name: "Baz"      enum_type {        name: "deep_enum"        value {          name: "VALUE_A"          number: 0        }      }      field {        name: "deep"        number: 4        label: LABEL_OPTIONAL        type: TYPE_UINT32      }    }  }}'
        file_descriptor = descriptor_pb2.FileDescriptorProto()
        text_format.Merge(file_descriptor_str, file_descriptor)
        return file_descriptor.SerializeToString()

    @testing_refleaks.SkipReferenceLeakChecker('MakeDescriptor is not repeatable')
    def testParsingFlatClassWithExplicitClassDeclaration(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that the generated class can parse a flat message.'
        if api_implementation.Type() != 'python':
            return
        file_descriptor = descriptor_pb2.FileDescriptorProto()
        file_descriptor.ParseFromString(self._GetSerializedFileDescriptor('A'))
        msg_descriptor = descriptor.MakeDescriptor(file_descriptor.message_type[0])

        class MessageClass(six.with_metaclass(reflection.GeneratedProtocolMessageType, message.Message)):
            DESCRIPTOR = msg_descriptor
        msg = MessageClass()
        msg_str = 'flat: 0 flat: 1 flat: 2 '
        text_format.Merge(msg_str, msg)
        self.assertEqual(msg.flat, [0, 1, 2])

    @testing_refleaks.SkipReferenceLeakChecker('MakeDescriptor is not repeatable')
    def testParsingFlatClass(self):
        if False:
            print('Hello World!')
        'Test that the generated class can parse a flat message.'
        file_descriptor = descriptor_pb2.FileDescriptorProto()
        file_descriptor.ParseFromString(self._GetSerializedFileDescriptor('B'))
        msg_descriptor = descriptor.MakeDescriptor(file_descriptor.message_type[0])
        msg_class = reflection.MakeClass(msg_descriptor)
        msg = msg_class()
        msg_str = 'flat: 0 flat: 1 flat: 2 '
        text_format.Merge(msg_str, msg)
        self.assertEqual(msg.flat, [0, 1, 2])

    @testing_refleaks.SkipReferenceLeakChecker('MakeDescriptor is not repeatable')
    def testParsingNestedClass(self):
        if False:
            return 10
        'Test that the generated class can parse a nested message.'
        file_descriptor = descriptor_pb2.FileDescriptorProto()
        file_descriptor.ParseFromString(self._GetSerializedFileDescriptor('C'))
        msg_descriptor = descriptor.MakeDescriptor(file_descriptor.message_type[0])
        msg_class = reflection.MakeClass(msg_descriptor)
        msg = msg_class()
        msg_str = 'bar {  baz {    deep: 4  }}'
        text_format.Merge(msg_str, msg)
        self.assertEqual(msg.bar.baz.deep, 4)
if __name__ == '__main__':
    unittest.main()