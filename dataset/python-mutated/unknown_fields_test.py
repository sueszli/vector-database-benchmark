"""Test for preservation of unknown fields in the pure Python implementation."""
__author__ = 'bohdank@google.com (Bohdan Koval)'
try:
    import unittest2 as unittest
except ImportError:
    import unittest
from google.protobuf import unittest_mset_pb2
from google.protobuf import unittest_pb2
from google.protobuf import unittest_proto3_arena_pb2
from google.protobuf.internal import api_implementation
from google.protobuf.internal import encoder
from google.protobuf.internal import message_set_extensions_pb2
from google.protobuf.internal import missing_enum_values_pb2
from google.protobuf.internal import test_util
from google.protobuf.internal import testing_refleaks
from google.protobuf.internal import type_checkers
BaseTestCase = testing_refleaks.BaseTestCase

def SkipIfCppImplementation(func):
    if False:
        while True:
            i = 10
    return unittest.skipIf(api_implementation.Type() == 'cpp' and api_implementation.Version() == 2, 'C++ implementation does not expose unknown fields to Python')(func)

class UnknownFieldsTest(BaseTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.descriptor = unittest_pb2.TestAllTypes.DESCRIPTOR
        self.all_fields = unittest_pb2.TestAllTypes()
        test_util.SetAllFields(self.all_fields)
        self.all_fields_data = self.all_fields.SerializeToString()
        self.empty_message = unittest_pb2.TestEmptyMessage()
        self.empty_message.ParseFromString(self.all_fields_data)

    def testSerialize(self):
        if False:
            return 10
        data = self.empty_message.SerializeToString()
        self.assertTrue(data == self.all_fields_data)

    def testSerializeProto3(self):
        if False:
            i = 10
            return i + 15
        message = unittest_proto3_arena_pb2.TestEmptyMessage()
        message.ParseFromString(self.all_fields_data)
        self.assertEqual(0, len(message.SerializeToString()))

    def testByteSize(self):
        if False:
            return 10
        self.assertEqual(self.all_fields.ByteSize(), self.empty_message.ByteSize())

    def testListFields(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(0, len(self.empty_message.ListFields()))

    def testSerializeMessageSetWireFormatUnknownExtension(self):
        if False:
            while True:
                i = 10
        raw = unittest_mset_pb2.RawMessageSet()
        item = raw.item.add()
        item.type_id = 98418603
        message1 = message_set_extensions_pb2.TestMessageSetExtension1()
        message1.i = 12345
        item.message = message1.SerializeToString()
        serialized = raw.SerializeToString()
        proto = message_set_extensions_pb2.TestMessageSet()
        proto.MergeFromString(serialized)
        reserialized = proto.SerializeToString()
        new_raw = unittest_mset_pb2.RawMessageSet()
        new_raw.MergeFromString(reserialized)
        self.assertEqual(raw, new_raw)

    def testEquals(self):
        if False:
            print('Hello World!')
        message = unittest_pb2.TestEmptyMessage()
        message.ParseFromString(self.all_fields_data)
        self.assertEqual(self.empty_message, message)
        self.all_fields.ClearField('optional_string')
        message.ParseFromString(self.all_fields.SerializeToString())
        self.assertNotEqual(self.empty_message, message)

    def testDiscardUnknownFields(self):
        if False:
            for i in range(10):
                print('nop')
        self.empty_message.DiscardUnknownFields()
        self.assertEqual(b'', self.empty_message.SerializeToString())
        message = unittest_pb2.TestAllTypes()
        other_message = unittest_pb2.TestAllTypes()
        other_message.optional_string = 'discard'
        message.optional_nested_message.ParseFromString(other_message.SerializeToString())
        message.repeated_nested_message.add().ParseFromString(other_message.SerializeToString())
        self.assertNotEqual(b'', message.optional_nested_message.SerializeToString())
        self.assertNotEqual(b'', message.repeated_nested_message[0].SerializeToString())
        message.DiscardUnknownFields()
        self.assertEqual(b'', message.optional_nested_message.SerializeToString())
        self.assertEqual(b'', message.repeated_nested_message[0].SerializeToString())

class UnknownFieldsAccessorsTest(BaseTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.descriptor = unittest_pb2.TestAllTypes.DESCRIPTOR
        self.all_fields = unittest_pb2.TestAllTypes()
        test_util.SetAllFields(self.all_fields)
        self.all_fields_data = self.all_fields.SerializeToString()
        self.empty_message = unittest_pb2.TestEmptyMessage()
        self.empty_message.ParseFromString(self.all_fields_data)

    def GetUnknownField(self, name):
        if False:
            print('Hello World!')
        field_descriptor = self.descriptor.fields_by_name[name]
        wire_type = type_checkers.FIELD_TYPE_TO_WIRE_TYPE[field_descriptor.type]
        field_tag = encoder.TagBytes(field_descriptor.number, wire_type)
        result_dict = {}
        for (tag_bytes, value) in self.empty_message._unknown_fields:
            if tag_bytes == field_tag:
                decoder = unittest_pb2.TestAllTypes._decoders_by_tag[tag_bytes][0]
                decoder(value, 0, len(value), self.all_fields, result_dict)
        return result_dict[field_descriptor]

    @SkipIfCppImplementation
    def testEnum(self):
        if False:
            i = 10
            return i + 15
        value = self.GetUnknownField('optional_nested_enum')
        self.assertEqual(self.all_fields.optional_nested_enum, value)

    @SkipIfCppImplementation
    def testRepeatedEnum(self):
        if False:
            i = 10
            return i + 15
        value = self.GetUnknownField('repeated_nested_enum')
        self.assertEqual(self.all_fields.repeated_nested_enum, value)

    @SkipIfCppImplementation
    def testVarint(self):
        if False:
            while True:
                i = 10
        value = self.GetUnknownField('optional_int32')
        self.assertEqual(self.all_fields.optional_int32, value)

    @SkipIfCppImplementation
    def testFixed32(self):
        if False:
            i = 10
            return i + 15
        value = self.GetUnknownField('optional_fixed32')
        self.assertEqual(self.all_fields.optional_fixed32, value)

    @SkipIfCppImplementation
    def testFixed64(self):
        if False:
            while True:
                i = 10
        value = self.GetUnknownField('optional_fixed64')
        self.assertEqual(self.all_fields.optional_fixed64, value)

    @SkipIfCppImplementation
    def testLengthDelimited(self):
        if False:
            while True:
                i = 10
        value = self.GetUnknownField('optional_string')
        self.assertEqual(self.all_fields.optional_string, value)

    @SkipIfCppImplementation
    def testGroup(self):
        if False:
            print('Hello World!')
        value = self.GetUnknownField('optionalgroup')
        self.assertEqual(self.all_fields.optionalgroup, value)

    def testCopyFrom(self):
        if False:
            return 10
        message = unittest_pb2.TestEmptyMessage()
        message.CopyFrom(self.empty_message)
        self.assertEqual(message.SerializeToString(), self.all_fields_data)

    def testMergeFrom(self):
        if False:
            print('Hello World!')
        message = unittest_pb2.TestAllTypes()
        message.optional_int32 = 1
        message.optional_uint32 = 2
        source = unittest_pb2.TestEmptyMessage()
        source.ParseFromString(message.SerializeToString())
        message.ClearField('optional_int32')
        message.optional_int64 = 3
        message.optional_uint32 = 4
        destination = unittest_pb2.TestEmptyMessage()
        destination.ParseFromString(message.SerializeToString())
        destination.MergeFrom(source)
        message.ParseFromString(destination.SerializeToString())
        self.assertEqual(message.optional_int32, 1)
        self.assertEqual(message.optional_uint32, 2)
        self.assertEqual(message.optional_int64, 3)

    def testClear(self):
        if False:
            for i in range(10):
                print('nop')
        self.empty_message.Clear()
        self.assertEqual(self.empty_message.SerializeToString(), b'')

    def testUnknownExtensions(self):
        if False:
            print('Hello World!')
        message = unittest_pb2.TestEmptyMessageWithExtensions()
        message.ParseFromString(self.all_fields_data)
        self.assertEqual(message.SerializeToString(), self.all_fields_data)

class UnknownEnumValuesTest(BaseTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.descriptor = missing_enum_values_pb2.TestEnumValues.DESCRIPTOR
        self.message = missing_enum_values_pb2.TestEnumValues()
        self.message.optional_nested_enum = missing_enum_values_pb2.TestEnumValues.ZERO
        self.message.repeated_nested_enum.extend([missing_enum_values_pb2.TestEnumValues.ZERO, missing_enum_values_pb2.TestEnumValues.ONE])
        self.message.packed_nested_enum.extend([missing_enum_values_pb2.TestEnumValues.ZERO, missing_enum_values_pb2.TestEnumValues.ONE])
        self.message_data = self.message.SerializeToString()
        self.missing_message = missing_enum_values_pb2.TestMissingEnumValues()
        self.missing_message.ParseFromString(self.message_data)

    def GetUnknownField(self, name):
        if False:
            return 10
        field_descriptor = self.descriptor.fields_by_name[name]
        wire_type = type_checkers.FIELD_TYPE_TO_WIRE_TYPE[field_descriptor.type]
        field_tag = encoder.TagBytes(field_descriptor.number, wire_type)
        result_dict = {}
        for (tag_bytes, value) in self.missing_message._unknown_fields:
            if tag_bytes == field_tag:
                decoder = missing_enum_values_pb2.TestEnumValues._decoders_by_tag[tag_bytes][0]
                decoder(value, 0, len(value), self.message, result_dict)
        return result_dict[field_descriptor]

    def testUnknownParseMismatchEnumValue(self):
        if False:
            i = 10
            return i + 15
        just_string = missing_enum_values_pb2.JustString()
        just_string.dummy = 'blah'
        missing = missing_enum_values_pb2.TestEnumValues()
        missing.ParseFromString(just_string.SerializeToString())
        self.assertEqual(missing.optional_nested_enum, 0)

    def testUnknownEnumValue(self):
        if False:
            print('Hello World!')
        if api_implementation.Type() == 'cpp':
            self.assertTrue(self.missing_message.HasField('optional_nested_enum'))
            self.assertEqual(self.message.optional_nested_enum, self.missing_message.optional_nested_enum)
        else:
            self.assertFalse(self.missing_message.HasField('optional_nested_enum'))
            value = self.GetUnknownField('optional_nested_enum')
            self.assertEqual(self.message.optional_nested_enum, value)
        self.missing_message.ClearField('optional_nested_enum')
        self.assertFalse(self.missing_message.HasField('optional_nested_enum'))

    def testUnknownRepeatedEnumValue(self):
        if False:
            while True:
                i = 10
        if api_implementation.Type() == 'cpp':
            self.assertEqual([], self.missing_message.repeated_nested_enum)
        else:
            self.assertEqual([], self.missing_message.repeated_nested_enum)
            value = self.GetUnknownField('repeated_nested_enum')
            self.assertEqual(self.message.repeated_nested_enum, value)

    def testUnknownPackedEnumValue(self):
        if False:
            while True:
                i = 10
        if api_implementation.Type() == 'cpp':
            self.assertEqual([], self.missing_message.packed_nested_enum)
        else:
            self.assertEqual([], self.missing_message.packed_nested_enum)
            value = self.GetUnknownField('packed_nested_enum')
            self.assertEqual(self.message.packed_nested_enum, value)

    def testRoundTrip(self):
        if False:
            while True:
                i = 10
        new_message = missing_enum_values_pb2.TestEnumValues()
        new_message.ParseFromString(self.missing_message.SerializeToString())
        self.assertEqual(self.message, new_message)
if __name__ == '__main__':
    unittest.main()