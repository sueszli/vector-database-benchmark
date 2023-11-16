"""Tests for google.protobuf.descriptor_pool."""
__author__ = 'matthewtoia@google.com (Matt Toia)'
import os
import sys
try:
    import unittest2 as unittest
except ImportError:
    import unittest
from google.protobuf import unittest_import_pb2
from google.protobuf import unittest_import_public_pb2
from google.protobuf import unittest_pb2
from google.protobuf import descriptor_pb2
from google.protobuf.internal import api_implementation
from google.protobuf.internal import descriptor_pool_test1_pb2
from google.protobuf.internal import descriptor_pool_test2_pb2
from google.protobuf.internal import factory_test1_pb2
from google.protobuf.internal import factory_test2_pb2
from google.protobuf.internal import file_options_test_pb2
from google.protobuf.internal import more_messages_pb2
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import descriptor_pool
from google.protobuf import message_factory
from google.protobuf import symbol_database

class DescriptorPoolTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.pool = descriptor_pool.DescriptorPool()
        self.factory_test1_fd = descriptor_pb2.FileDescriptorProto.FromString(factory_test1_pb2.DESCRIPTOR.serialized_pb)
        self.factory_test2_fd = descriptor_pb2.FileDescriptorProto.FromString(factory_test2_pb2.DESCRIPTOR.serialized_pb)
        self.pool.Add(self.factory_test1_fd)
        self.pool.Add(self.factory_test2_fd)
        self.pool.Add(descriptor_pb2.FileDescriptorProto.FromString(unittest_import_public_pb2.DESCRIPTOR.serialized_pb))
        self.pool.Add(descriptor_pb2.FileDescriptorProto.FromString(unittest_import_pb2.DESCRIPTOR.serialized_pb))
        self.pool.Add(descriptor_pb2.FileDescriptorProto.FromString(unittest_pb2.DESCRIPTOR.serialized_pb))

    def testFindFileByName(self):
        if False:
            i = 10
            return i + 15
        name1 = 'google/protobuf/internal/factory_test1.proto'
        file_desc1 = self.pool.FindFileByName(name1)
        self.assertIsInstance(file_desc1, descriptor.FileDescriptor)
        self.assertEqual(name1, file_desc1.name)
        self.assertEqual('google.protobuf.python.internal', file_desc1.package)
        self.assertIn('Factory1Message', file_desc1.message_types_by_name)
        name2 = 'google/protobuf/internal/factory_test2.proto'
        file_desc2 = self.pool.FindFileByName(name2)
        self.assertIsInstance(file_desc2, descriptor.FileDescriptor)
        self.assertEqual(name2, file_desc2.name)
        self.assertEqual('google.protobuf.python.internal', file_desc2.package)
        self.assertIn('Factory2Message', file_desc2.message_types_by_name)

    def testFindFileByNameFailure(self):
        if False:
            print('Hello World!')
        with self.assertRaises(KeyError):
            self.pool.FindFileByName('Does not exist')

    def testFindFileContainingSymbol(self):
        if False:
            print('Hello World!')
        file_desc1 = self.pool.FindFileContainingSymbol('google.protobuf.python.internal.Factory1Message')
        self.assertIsInstance(file_desc1, descriptor.FileDescriptor)
        self.assertEqual('google/protobuf/internal/factory_test1.proto', file_desc1.name)
        self.assertEqual('google.protobuf.python.internal', file_desc1.package)
        self.assertIn('Factory1Message', file_desc1.message_types_by_name)
        file_desc2 = self.pool.FindFileContainingSymbol('google.protobuf.python.internal.Factory2Message')
        self.assertIsInstance(file_desc2, descriptor.FileDescriptor)
        self.assertEqual('google/protobuf/internal/factory_test2.proto', file_desc2.name)
        self.assertEqual('google.protobuf.python.internal', file_desc2.package)
        self.assertIn('Factory2Message', file_desc2.message_types_by_name)
        file_desc3 = self.pool.FindFileContainingSymbol('google.protobuf.python.internal.another_field')
        self.assertIsInstance(file_desc3, descriptor.FileDescriptor)
        self.assertEqual('google/protobuf/internal/factory_test2.proto', file_desc3.name)
        file_desc4 = self.pool.FindFileContainingSymbol('google.protobuf.python.internal.Factory2Message.one_more_field')
        self.assertIsInstance(file_desc4, descriptor.FileDescriptor)
        self.assertEqual('google/protobuf/internal/factory_test2.proto', file_desc4.name)
        assert descriptor_pool.Default().FindFileContainingSymbol('google.protobuf.python.internal.Factory2Message.one_more_field')
        assert descriptor_pool.Default().FindFileContainingSymbol('google.protobuf.python.internal.another_field')

    def testFindFileContainingSymbolFailure(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(KeyError):
            self.pool.FindFileContainingSymbol('Does not exist')

    def testFindMessageTypeByName(self):
        if False:
            print('Hello World!')
        msg1 = self.pool.FindMessageTypeByName('google.protobuf.python.internal.Factory1Message')
        self.assertIsInstance(msg1, descriptor.Descriptor)
        self.assertEqual('Factory1Message', msg1.name)
        self.assertEqual('google.protobuf.python.internal.Factory1Message', msg1.full_name)
        self.assertEqual(None, msg1.containing_type)
        self.assertFalse(msg1.has_options)
        nested_msg1 = msg1.nested_types[0]
        self.assertEqual('NestedFactory1Message', nested_msg1.name)
        self.assertEqual(msg1, nested_msg1.containing_type)
        nested_enum1 = msg1.enum_types[0]
        self.assertEqual('NestedFactory1Enum', nested_enum1.name)
        self.assertEqual(msg1, nested_enum1.containing_type)
        self.assertEqual(nested_msg1, msg1.fields_by_name['nested_factory_1_message'].message_type)
        self.assertEqual(nested_enum1, msg1.fields_by_name['nested_factory_1_enum'].enum_type)
        msg2 = self.pool.FindMessageTypeByName('google.protobuf.python.internal.Factory2Message')
        self.assertIsInstance(msg2, descriptor.Descriptor)
        self.assertEqual('Factory2Message', msg2.name)
        self.assertEqual('google.protobuf.python.internal.Factory2Message', msg2.full_name)
        self.assertIsNone(msg2.containing_type)
        nested_msg2 = msg2.nested_types[0]
        self.assertEqual('NestedFactory2Message', nested_msg2.name)
        self.assertEqual(msg2, nested_msg2.containing_type)
        nested_enum2 = msg2.enum_types[0]
        self.assertEqual('NestedFactory2Enum', nested_enum2.name)
        self.assertEqual(msg2, nested_enum2.containing_type)
        self.assertEqual(nested_msg2, msg2.fields_by_name['nested_factory_2_message'].message_type)
        self.assertEqual(nested_enum2, msg2.fields_by_name['nested_factory_2_enum'].enum_type)
        self.assertTrue(msg2.fields_by_name['int_with_default'].has_default_value)
        self.assertEqual(1776, msg2.fields_by_name['int_with_default'].default_value)
        self.assertTrue(msg2.fields_by_name['double_with_default'].has_default_value)
        self.assertEqual(9.99, msg2.fields_by_name['double_with_default'].default_value)
        self.assertTrue(msg2.fields_by_name['string_with_default'].has_default_value)
        self.assertEqual('hello world', msg2.fields_by_name['string_with_default'].default_value)
        self.assertTrue(msg2.fields_by_name['bool_with_default'].has_default_value)
        self.assertFalse(msg2.fields_by_name['bool_with_default'].default_value)
        self.assertTrue(msg2.fields_by_name['enum_with_default'].has_default_value)
        self.assertEqual(1, msg2.fields_by_name['enum_with_default'].default_value)
        msg3 = self.pool.FindMessageTypeByName('google.protobuf.python.internal.Factory2Message.NestedFactory2Message')
        self.assertEqual(nested_msg2, msg3)
        self.assertTrue(msg2.fields_by_name['bytes_with_default'].has_default_value)
        self.assertEqual(b'a\xfb\x00c', msg2.fields_by_name['bytes_with_default'].default_value)
        self.assertEqual(1, len(msg2.oneofs))
        self.assertEqual(1, len(msg2.oneofs_by_name))
        self.assertEqual(2, len(msg2.oneofs[0].fields))
        for name in ['oneof_int', 'oneof_string']:
            self.assertEqual(msg2.oneofs[0], msg2.fields_by_name[name].containing_oneof)
            self.assertIn(msg2.fields_by_name[name], msg2.oneofs[0].fields)

    def testFindMessageTypeByNameFailure(self):
        if False:
            return 10
        with self.assertRaises(KeyError):
            self.pool.FindMessageTypeByName('Does not exist')

    def testFindEnumTypeByName(self):
        if False:
            for i in range(10):
                print('nop')
        enum1 = self.pool.FindEnumTypeByName('google.protobuf.python.internal.Factory1Enum')
        self.assertIsInstance(enum1, descriptor.EnumDescriptor)
        self.assertEqual(0, enum1.values_by_name['FACTORY_1_VALUE_0'].number)
        self.assertEqual(1, enum1.values_by_name['FACTORY_1_VALUE_1'].number)
        self.assertFalse(enum1.has_options)
        nested_enum1 = self.pool.FindEnumTypeByName('google.protobuf.python.internal.Factory1Message.NestedFactory1Enum')
        self.assertIsInstance(nested_enum1, descriptor.EnumDescriptor)
        self.assertEqual(0, nested_enum1.values_by_name['NESTED_FACTORY_1_VALUE_0'].number)
        self.assertEqual(1, nested_enum1.values_by_name['NESTED_FACTORY_1_VALUE_1'].number)
        enum2 = self.pool.FindEnumTypeByName('google.protobuf.python.internal.Factory2Enum')
        self.assertIsInstance(enum2, descriptor.EnumDescriptor)
        self.assertEqual(0, enum2.values_by_name['FACTORY_2_VALUE_0'].number)
        self.assertEqual(1, enum2.values_by_name['FACTORY_2_VALUE_1'].number)
        nested_enum2 = self.pool.FindEnumTypeByName('google.protobuf.python.internal.Factory2Message.NestedFactory2Enum')
        self.assertIsInstance(nested_enum2, descriptor.EnumDescriptor)
        self.assertEqual(0, nested_enum2.values_by_name['NESTED_FACTORY_2_VALUE_0'].number)
        self.assertEqual(1, nested_enum2.values_by_name['NESTED_FACTORY_2_VALUE_1'].number)

    def testFindEnumTypeByNameFailure(self):
        if False:
            print('Hello World!')
        with self.assertRaises(KeyError):
            self.pool.FindEnumTypeByName('Does not exist')

    def testFindFieldByName(self):
        if False:
            print('Hello World!')
        field = self.pool.FindFieldByName('google.protobuf.python.internal.Factory1Message.list_value')
        self.assertEqual(field.name, 'list_value')
        self.assertEqual(field.label, field.LABEL_REPEATED)
        self.assertFalse(field.has_options)
        with self.assertRaises(KeyError):
            self.pool.FindFieldByName('Does not exist')

    def testFindExtensionByName(self):
        if False:
            while True:
                i = 10
        extension = self.pool.FindExtensionByName('google.protobuf.python.internal.Factory2Message.one_more_field')
        self.assertEqual(extension.name, 'one_more_field')
        extension = self.pool.FindExtensionByName('google.protobuf.python.internal.another_field')
        self.assertEqual(extension.name, 'another_field')
        self.assertEqual(extension.number, 1002)
        with self.assertRaises(KeyError):
            self.pool.FindFieldByName('Does not exist')

    def testFindAllExtensions(self):
        if False:
            i = 10
            return i + 15
        factory1_message = self.pool.FindMessageTypeByName('google.protobuf.python.internal.Factory1Message')
        factory2_message = self.pool.FindMessageTypeByName('google.protobuf.python.internal.Factory2Message')
        one_more_field = factory2_message.extensions_by_name['one_more_field']
        self.pool.AddExtensionDescriptor(one_more_field)
        factory_test2 = self.pool.FindFileByName('google/protobuf/internal/factory_test2.proto')
        another_field = factory_test2.extensions_by_name['another_field']
        self.pool.AddExtensionDescriptor(another_field)
        extensions = self.pool.FindAllExtensions(factory1_message)
        expected_extension_numbers = set([one_more_field, another_field])
        self.assertEqual(expected_extension_numbers, set(extensions))
        extensions.append('unexpected_element')
        extensions = self.pool.FindAllExtensions(factory1_message)
        self.assertEqual(expected_extension_numbers, set(extensions))

    def testFindExtensionByNumber(self):
        if False:
            for i in range(10):
                print('nop')
        factory1_message = self.pool.FindMessageTypeByName('google.protobuf.python.internal.Factory1Message')
        factory2_message = self.pool.FindMessageTypeByName('google.protobuf.python.internal.Factory2Message')
        one_more_field = factory2_message.extensions_by_name['one_more_field']
        self.pool.AddExtensionDescriptor(one_more_field)
        factory_test2 = self.pool.FindFileByName('google/protobuf/internal/factory_test2.proto')
        another_field = factory_test2.extensions_by_name['another_field']
        self.pool.AddExtensionDescriptor(another_field)
        extension = self.pool.FindExtensionByNumber(factory1_message, 1001)
        self.assertEqual(extension.name, 'one_more_field')
        extension = self.pool.FindExtensionByNumber(factory1_message, 1002)
        self.assertEqual(extension.name, 'another_field')
        with self.assertRaises(KeyError):
            extension = self.pool.FindExtensionByNumber(factory1_message, 1234567)

    def testExtensionsAreNotFields(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(KeyError):
            self.pool.FindFieldByName('google.protobuf.python.internal.another_field')
        with self.assertRaises(KeyError):
            self.pool.FindFieldByName('google.protobuf.python.internal.Factory2Message.one_more_field')
        with self.assertRaises(KeyError):
            self.pool.FindExtensionByName('google.protobuf.python.internal.Factory1Message.list_value')

    def testFindService(self):
        if False:
            return 10
        service = self.pool.FindServiceByName('protobuf_unittest.TestService')
        self.assertEqual(service.full_name, 'protobuf_unittest.TestService')

    def testUserDefinedDB(self):
        if False:
            print('Hello World!')
        db = descriptor_database.DescriptorDatabase()
        self.pool = descriptor_pool.DescriptorPool(db)
        db.Add(self.factory_test1_fd)
        db.Add(self.factory_test2_fd)
        self.testFindMessageTypeByName()

    def testAddSerializedFile(self):
        if False:
            while True:
                i = 10
        self.pool = descriptor_pool.DescriptorPool()
        self.pool.AddSerializedFile(self.factory_test1_fd.SerializeToString())
        self.pool.AddSerializedFile(self.factory_test2_fd.SerializeToString())
        self.testFindMessageTypeByName()

    def testComplexNesting(self):
        if False:
            print('Hello World!')
        more_messages_desc = descriptor_pb2.FileDescriptorProto.FromString(more_messages_pb2.DESCRIPTOR.serialized_pb)
        test1_desc = descriptor_pb2.FileDescriptorProto.FromString(descriptor_pool_test1_pb2.DESCRIPTOR.serialized_pb)
        test2_desc = descriptor_pb2.FileDescriptorProto.FromString(descriptor_pool_test2_pb2.DESCRIPTOR.serialized_pb)
        self.pool.Add(more_messages_desc)
        self.pool.Add(test1_desc)
        self.pool.Add(test2_desc)
        TEST1_FILE.CheckFile(self, self.pool)
        TEST2_FILE.CheckFile(self, self.pool)

    def testEnumDefaultValue(self):
        if False:
            return 10
        "Test the default value of enums which don't start at zero."

        def _CheckDefaultValue(file_descriptor):
            if False:
                i = 10
                return i + 15
            default_value = file_descriptor.message_types_by_name['DescriptorPoolTest1'].fields_by_name['nested_enum'].default_value
            self.assertEqual(default_value, descriptor_pool_test1_pb2.DescriptorPoolTest1.BETA)
        _CheckDefaultValue(descriptor_pool_test1_pb2.DESCRIPTOR)
        file_descriptor = symbol_database.Default().pool.FindFileByName('google/protobuf/internal/descriptor_pool_test1.proto')
        self.assertIs(file_descriptor, descriptor_pool_test1_pb2.DESCRIPTOR)
        _CheckDefaultValue(file_descriptor)
        descriptor_proto = descriptor_pb2.FileDescriptorProto.FromString(descriptor_pool_test1_pb2.DESCRIPTOR.serialized_pb)
        self.pool.Add(descriptor_proto)
        file_descriptor = self.pool.FindFileByName('google/protobuf/internal/descriptor_pool_test1.proto')
        _CheckDefaultValue(file_descriptor)

    def testDefaultValueForCustomMessages(self):
        if False:
            for i in range(10):
                print('nop')
        'Check the value returned by non-existent fields.'

        def _CheckValueAndType(value, expected_value, expected_type):
            if False:
                print('Hello World!')
            self.assertEqual(value, expected_value)
            self.assertIsInstance(value, expected_type)

        def _CheckDefaultValues(msg):
            if False:
                i = 10
                return i + 15
            try:
                int64 = long
            except NameError:
                int64 = int
            try:
                unicode_type = unicode
            except NameError:
                unicode_type = str
            _CheckValueAndType(msg.optional_int32, 0, int)
            _CheckValueAndType(msg.optional_uint64, 0, (int64, int))
            _CheckValueAndType(msg.optional_float, 0, (float, int))
            _CheckValueAndType(msg.optional_double, 0, (float, int))
            _CheckValueAndType(msg.optional_bool, False, bool)
            _CheckValueAndType(msg.optional_string, u'', unicode_type)
            _CheckValueAndType(msg.optional_bytes, b'', bytes)
            _CheckValueAndType(msg.optional_nested_enum, msg.FOO, int)
        _CheckDefaultValues(unittest_pb2.TestAllTypes())
        pool = descriptor_pool.DescriptorPool()
        pool.Add(descriptor_pb2.FileDescriptorProto.FromString(unittest_import_public_pb2.DESCRIPTOR.serialized_pb))
        pool.Add(descriptor_pb2.FileDescriptorProto.FromString(unittest_import_pb2.DESCRIPTOR.serialized_pb))
        pool.Add(descriptor_pb2.FileDescriptorProto.FromString(unittest_pb2.DESCRIPTOR.serialized_pb))
        message_class = message_factory.MessageFactory(pool).GetPrototype(pool.FindMessageTypeByName(unittest_pb2.TestAllTypes.DESCRIPTOR.full_name))
        _CheckDefaultValues(message_class())

class ProtoFile(object):

    def __init__(self, name, package, messages, dependencies=None, public_dependencies=None):
        if False:
            return 10
        self.name = name
        self.package = package
        self.messages = messages
        self.dependencies = dependencies or []
        self.public_dependencies = public_dependencies or []

    def CheckFile(self, test, pool):
        if False:
            i = 10
            return i + 15
        file_desc = pool.FindFileByName(self.name)
        test.assertEqual(self.name, file_desc.name)
        test.assertEqual(self.package, file_desc.package)
        dependencies_names = [f.name for f in file_desc.dependencies]
        test.assertEqual(self.dependencies, dependencies_names)
        public_dependencies_names = [f.name for f in file_desc.public_dependencies]
        test.assertEqual(self.public_dependencies, public_dependencies_names)
        for (name, msg_type) in self.messages.items():
            msg_type.CheckType(test, None, name, file_desc)

class EnumType(object):

    def __init__(self, values):
        if False:
            print('Hello World!')
        self.values = values

    def CheckType(self, test, msg_desc, name, file_desc):
        if False:
            for i in range(10):
                print('nop')
        enum_desc = msg_desc.enum_types_by_name[name]
        test.assertEqual(name, enum_desc.name)
        expected_enum_full_name = '.'.join([msg_desc.full_name, name])
        test.assertEqual(expected_enum_full_name, enum_desc.full_name)
        test.assertEqual(msg_desc, enum_desc.containing_type)
        test.assertEqual(file_desc, enum_desc.file)
        for (index, (value, number)) in enumerate(self.values):
            value_desc = enum_desc.values_by_name[value]
            test.assertEqual(value, value_desc.name)
            test.assertEqual(index, value_desc.index)
            test.assertEqual(number, value_desc.number)
            test.assertEqual(enum_desc, value_desc.type)
            test.assertIn(value, msg_desc.enum_values_by_name)

class MessageType(object):

    def __init__(self, type_dict, field_list, is_extendable=False, extensions=None):
        if False:
            print('Hello World!')
        self.type_dict = type_dict
        self.field_list = field_list
        self.is_extendable = is_extendable
        self.extensions = extensions or []

    def CheckType(self, test, containing_type_desc, name, file_desc):
        if False:
            return 10
        if containing_type_desc is None:
            desc = file_desc.message_types_by_name[name]
            expected_full_name = '.'.join([file_desc.package, name])
        else:
            desc = containing_type_desc.nested_types_by_name[name]
            expected_full_name = '.'.join([containing_type_desc.full_name, name])
        test.assertEqual(name, desc.name)
        test.assertEqual(expected_full_name, desc.full_name)
        test.assertEqual(containing_type_desc, desc.containing_type)
        test.assertEqual(desc.file, file_desc)
        test.assertEqual(self.is_extendable, desc.is_extendable)
        for (name, subtype) in self.type_dict.items():
            subtype.CheckType(test, desc, name, file_desc)
        for (index, (name, field)) in enumerate(self.field_list):
            field.CheckField(test, desc, name, index)
        for (index, (name, field)) in enumerate(self.extensions):
            field.CheckField(test, desc, name, index)

class EnumField(object):

    def __init__(self, number, type_name, default_value):
        if False:
            print('Hello World!')
        self.number = number
        self.type_name = type_name
        self.default_value = default_value

    def CheckField(self, test, msg_desc, name, index):
        if False:
            while True:
                i = 10
        field_desc = msg_desc.fields_by_name[name]
        enum_desc = msg_desc.enum_types_by_name[self.type_name]
        test.assertEqual(name, field_desc.name)
        expected_field_full_name = '.'.join([msg_desc.full_name, name])
        test.assertEqual(expected_field_full_name, field_desc.full_name)
        test.assertEqual(index, field_desc.index)
        test.assertEqual(self.number, field_desc.number)
        test.assertEqual(descriptor.FieldDescriptor.TYPE_ENUM, field_desc.type)
        test.assertEqual(descriptor.FieldDescriptor.CPPTYPE_ENUM, field_desc.cpp_type)
        test.assertTrue(field_desc.has_default_value)
        test.assertEqual(enum_desc.values_by_name[self.default_value].number, field_desc.default_value)
        test.assertFalse(enum_desc.values_by_name[self.default_value].has_options)
        test.assertEqual(msg_desc, field_desc.containing_type)
        test.assertEqual(enum_desc, field_desc.enum_type)

class MessageField(object):

    def __init__(self, number, type_name):
        if False:
            while True:
                i = 10
        self.number = number
        self.type_name = type_name

    def CheckField(self, test, msg_desc, name, index):
        if False:
            print('Hello World!')
        field_desc = msg_desc.fields_by_name[name]
        field_type_desc = msg_desc.nested_types_by_name[self.type_name]
        test.assertEqual(name, field_desc.name)
        expected_field_full_name = '.'.join([msg_desc.full_name, name])
        test.assertEqual(expected_field_full_name, field_desc.full_name)
        test.assertEqual(index, field_desc.index)
        test.assertEqual(self.number, field_desc.number)
        test.assertEqual(descriptor.FieldDescriptor.TYPE_MESSAGE, field_desc.type)
        test.assertEqual(descriptor.FieldDescriptor.CPPTYPE_MESSAGE, field_desc.cpp_type)
        test.assertFalse(field_desc.has_default_value)
        test.assertEqual(msg_desc, field_desc.containing_type)
        test.assertEqual(field_type_desc, field_desc.message_type)

class StringField(object):

    def __init__(self, number, default_value):
        if False:
            while True:
                i = 10
        self.number = number
        self.default_value = default_value

    def CheckField(self, test, msg_desc, name, index):
        if False:
            while True:
                i = 10
        field_desc = msg_desc.fields_by_name[name]
        test.assertEqual(name, field_desc.name)
        expected_field_full_name = '.'.join([msg_desc.full_name, name])
        test.assertEqual(expected_field_full_name, field_desc.full_name)
        test.assertEqual(index, field_desc.index)
        test.assertEqual(self.number, field_desc.number)
        test.assertEqual(descriptor.FieldDescriptor.TYPE_STRING, field_desc.type)
        test.assertEqual(descriptor.FieldDescriptor.CPPTYPE_STRING, field_desc.cpp_type)
        test.assertTrue(field_desc.has_default_value)
        test.assertEqual(self.default_value, field_desc.default_value)

class ExtensionField(object):

    def __init__(self, number, extended_type):
        if False:
            while True:
                i = 10
        self.number = number
        self.extended_type = extended_type

    def CheckField(self, test, msg_desc, name, index):
        if False:
            print('Hello World!')
        field_desc = msg_desc.extensions_by_name[name]
        test.assertEqual(name, field_desc.name)
        expected_field_full_name = '.'.join([msg_desc.full_name, name])
        test.assertEqual(expected_field_full_name, field_desc.full_name)
        test.assertEqual(self.number, field_desc.number)
        test.assertEqual(index, field_desc.index)
        test.assertEqual(descriptor.FieldDescriptor.TYPE_MESSAGE, field_desc.type)
        test.assertEqual(descriptor.FieldDescriptor.CPPTYPE_MESSAGE, field_desc.cpp_type)
        test.assertFalse(field_desc.has_default_value)
        test.assertTrue(field_desc.is_extension)
        test.assertEqual(msg_desc, field_desc.extension_scope)
        test.assertEqual(msg_desc, field_desc.message_type)
        test.assertEqual(self.extended_type, field_desc.containing_type.name)

class AddDescriptorTest(unittest.TestCase):

    def _TestMessage(self, prefix):
        if False:
            print('Hello World!')
        pool = descriptor_pool.DescriptorPool()
        pool.AddDescriptor(unittest_pb2.TestAllTypes.DESCRIPTOR)
        self.assertEqual('protobuf_unittest.TestAllTypes', pool.FindMessageTypeByName(prefix + 'protobuf_unittest.TestAllTypes').full_name)
        with self.assertRaises(KeyError):
            pool.FindMessageTypeByName(prefix + 'protobuf_unittest.TestAllTypes.NestedMessage')
        pool.AddDescriptor(unittest_pb2.TestAllTypes.NestedMessage.DESCRIPTOR)
        self.assertEqual('protobuf_unittest.TestAllTypes.NestedMessage', pool.FindMessageTypeByName(prefix + 'protobuf_unittest.TestAllTypes.NestedMessage').full_name)
        self.assertEqual('google/protobuf/unittest.proto', pool.FindFileByName('google/protobuf/unittest.proto').name)
        self.assertEqual('google/protobuf/unittest.proto', pool.FindFileContainingSymbol(prefix + 'protobuf_unittest.TestAllTypes.NestedMessage').name)

    @unittest.skipIf(api_implementation.Type() == 'cpp', 'With the cpp implementation, Add() must be called first')
    def testMessage(self):
        if False:
            return 10
        self._TestMessage('')
        self._TestMessage('.')

    def _TestEnum(self, prefix):
        if False:
            return 10
        pool = descriptor_pool.DescriptorPool()
        pool.AddEnumDescriptor(unittest_pb2.ForeignEnum.DESCRIPTOR)
        self.assertEqual('protobuf_unittest.ForeignEnum', pool.FindEnumTypeByName(prefix + 'protobuf_unittest.ForeignEnum').full_name)
        with self.assertRaises(KeyError):
            pool.FindEnumTypeByName(prefix + 'protobuf_unittest.ForeignEnum.NestedEnum')
        pool.AddEnumDescriptor(unittest_pb2.TestAllTypes.NestedEnum.DESCRIPTOR)
        self.assertEqual('protobuf_unittest.TestAllTypes.NestedEnum', pool.FindEnumTypeByName(prefix + 'protobuf_unittest.TestAllTypes.NestedEnum').full_name)
        self.assertEqual('google/protobuf/unittest.proto', pool.FindFileByName('google/protobuf/unittest.proto').name)
        self.assertEqual('google/protobuf/unittest.proto', pool.FindFileContainingSymbol(prefix + 'protobuf_unittest.TestAllTypes.NestedEnum').name)

    @unittest.skipIf(api_implementation.Type() == 'cpp', 'With the cpp implementation, Add() must be called first')
    def testEnum(self):
        if False:
            for i in range(10):
                print('nop')
        self._TestEnum('')
        self._TestEnum('.')

    @unittest.skipIf(api_implementation.Type() == 'cpp', 'With the cpp implementation, Add() must be called first')
    def testService(self):
        if False:
            print('Hello World!')
        pool = descriptor_pool.DescriptorPool()
        with self.assertRaises(KeyError):
            pool.FindServiceByName('protobuf_unittest.TestService')
        pool.AddServiceDescriptor(unittest_pb2._TESTSERVICE)
        self.assertEqual('protobuf_unittest.TestService', pool.FindServiceByName('protobuf_unittest.TestService').full_name)

    @unittest.skipIf(api_implementation.Type() == 'cpp', 'With the cpp implementation, Add() must be called first')
    def testFile(self):
        if False:
            return 10
        pool = descriptor_pool.DescriptorPool()
        pool.AddFileDescriptor(unittest_pb2.DESCRIPTOR)
        self.assertEqual('google/protobuf/unittest.proto', pool.FindFileByName('google/protobuf/unittest.proto').name)
        with self.assertRaises(KeyError):
            pool.FindFileContainingSymbol('protobuf_unittest.TestAllTypes')

    def testEmptyDescriptorPool(self):
        if False:
            for i in range(10):
                print('nop')
        pool = descriptor_pool.DescriptorPool()
        proto_file_name = descriptor_pb2.DESCRIPTOR.name
        self.assertRaises(KeyError, pool.FindFileByName, proto_file_name)
        file_descriptor = descriptor_pb2.FileDescriptorProto()
        descriptor_pb2.DESCRIPTOR.CopyToProto(file_descriptor)
        pool.Add(file_descriptor)
        self.assertTrue(pool.FindFileByName(proto_file_name))

    def testCustomDescriptorPool(self):
        if False:
            i = 10
            return i + 15
        pool = descriptor_pool.DescriptorPool()
        file_desc = descriptor_pb2.FileDescriptorProto(name='some/file.proto', package='package')
        file_desc.message_type.add(name='Message')
        pool.Add(file_desc)
        self.assertEqual(pool.FindFileByName('some/file.proto').name, 'some/file.proto')
        self.assertEqual(pool.FindMessageTypeByName('package.Message').name, 'Message')

    def testFileDescriptorOptionsWithCustomDescriptorPool(self):
        if False:
            for i in range(10):
                print('nop')
        pool = descriptor_pool.DescriptorPool()
        file_name = 'file_descriptor_options_with_custom_descriptor_pool.proto'
        file_descriptor_proto = descriptor_pb2.FileDescriptorProto(name=file_name)
        extension_id = file_options_test_pb2.foo_options
        file_descriptor_proto.options.Extensions[extension_id].foo_name = 'foo'
        pool.Add(file_descriptor_proto)
        file_descriptor = pool.FindFileByName(file_name)
        options = file_descriptor.GetOptions()
        self.assertEqual('foo', options.Extensions[extension_id].foo_name)
        self.assertIs(options, file_descriptor.GetOptions())

@unittest.skipIf(api_implementation.Type() != 'cpp', 'default_pool is only supported by the C++ implementation')
class DefaultPoolTest(unittest.TestCase):

    def testFindMethods(self):
        if False:
            for i in range(10):
                print('nop')
        from google.protobuf.pyext import _message
        pool = _message.default_pool
        self.assertIs(pool.FindFileByName('google/protobuf/unittest.proto'), unittest_pb2.DESCRIPTOR)
        self.assertIs(pool.FindMessageTypeByName('protobuf_unittest.TestAllTypes'), unittest_pb2.TestAllTypes.DESCRIPTOR)
        self.assertIs(pool.FindFieldByName('protobuf_unittest.TestAllTypes.optional_int32'), unittest_pb2.TestAllTypes.DESCRIPTOR.fields_by_name['optional_int32'])
        self.assertIs(pool.FindExtensionByName('protobuf_unittest.optional_int32_extension'), unittest_pb2.DESCRIPTOR.extensions_by_name['optional_int32_extension'])
        self.assertIs(pool.FindEnumTypeByName('protobuf_unittest.ForeignEnum'), unittest_pb2.ForeignEnum.DESCRIPTOR)
        self.assertIs(pool.FindOneofByName('protobuf_unittest.TestAllTypes.oneof_field'), unittest_pb2.TestAllTypes.DESCRIPTOR.oneofs_by_name['oneof_field'])

    def testAddFileDescriptor(self):
        if False:
            print('Hello World!')
        from google.protobuf.pyext import _message
        pool = _message.default_pool
        file_desc = descriptor_pb2.FileDescriptorProto(name='some/file.proto')
        pool.Add(file_desc)
        pool.AddSerializedFile(file_desc.SerializeToString())
TEST1_FILE = ProtoFile('google/protobuf/internal/descriptor_pool_test1.proto', 'google.protobuf.python.internal', {'DescriptorPoolTest1': MessageType({'NestedEnum': EnumType([('ALPHA', 1), ('BETA', 2)]), 'NestedMessage': MessageType({'NestedEnum': EnumType([('EPSILON', 5), ('ZETA', 6)]), 'DeepNestedMessage': MessageType({'NestedEnum': EnumType([('ETA', 7), ('THETA', 8)])}, [('nested_enum', EnumField(1, 'NestedEnum', 'ETA')), ('nested_field', StringField(2, 'theta'))])}, [('nested_enum', EnumField(1, 'NestedEnum', 'ZETA')), ('nested_field', StringField(2, 'beta')), ('deep_nested_message', MessageField(3, 'DeepNestedMessage'))])}, [('nested_enum', EnumField(1, 'NestedEnum', 'BETA')), ('nested_message', MessageField(2, 'NestedMessage'))], is_extendable=True), 'DescriptorPoolTest2': MessageType({'NestedEnum': EnumType([('GAMMA', 3), ('DELTA', 4)]), 'NestedMessage': MessageType({'NestedEnum': EnumType([('IOTA', 9), ('KAPPA', 10)]), 'DeepNestedMessage': MessageType({'NestedEnum': EnumType([('LAMBDA', 11), ('MU', 12)])}, [('nested_enum', EnumField(1, 'NestedEnum', 'MU')), ('nested_field', StringField(2, 'lambda'))])}, [('nested_enum', EnumField(1, 'NestedEnum', 'IOTA')), ('nested_field', StringField(2, 'delta')), ('deep_nested_message', MessageField(3, 'DeepNestedMessage'))])}, [('nested_enum', EnumField(1, 'NestedEnum', 'GAMMA')), ('nested_message', MessageField(2, 'NestedMessage'))])})
TEST2_FILE = ProtoFile('google/protobuf/internal/descriptor_pool_test2.proto', 'google.protobuf.python.internal', {'DescriptorPoolTest3': MessageType({'NestedEnum': EnumType([('NU', 13), ('XI', 14)]), 'NestedMessage': MessageType({'NestedEnum': EnumType([('OMICRON', 15), ('PI', 16)]), 'DeepNestedMessage': MessageType({'NestedEnum': EnumType([('RHO', 17), ('SIGMA', 18)])}, [('nested_enum', EnumField(1, 'NestedEnum', 'RHO')), ('nested_field', StringField(2, 'sigma'))])}, [('nested_enum', EnumField(1, 'NestedEnum', 'PI')), ('nested_field', StringField(2, 'nu')), ('deep_nested_message', MessageField(3, 'DeepNestedMessage'))])}, [('nested_enum', EnumField(1, 'NestedEnum', 'XI')), ('nested_message', MessageField(2, 'NestedMessage'))], extensions=[('descriptor_pool_test', ExtensionField(1001, 'DescriptorPoolTest1'))])}, dependencies=['google/protobuf/internal/descriptor_pool_test1.proto', 'google/protobuf/internal/more_messages.proto'], public_dependencies=['google/protobuf/internal/more_messages.proto'])
if __name__ == '__main__':
    unittest.main()