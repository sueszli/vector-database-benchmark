"""Unittest that directly tests the output of the pure-Python protocol
compiler.  See //google/protobuf/reflection_test.py for a test which
further ensures that we can use Python protocol message objects as we expect.
"""
__author__ = 'robinson@google.com (Will Robinson)'
import unittest
from google.protobuf.internal import test_bad_identifiers_pb2
from google.protobuf import unittest_custom_options_pb2
from google.protobuf import unittest_import_pb2
from google.protobuf import unittest_import_public_pb2
from google.protobuf import unittest_mset_pb2
from google.protobuf import unittest_pb2
from google.protobuf import unittest_no_generic_services_pb2
from google.protobuf import service
MAX_EXTENSION = 536870912

class GeneratorTest(unittest.TestCase):

    def testNestedMessageDescriptor(self):
        if False:
            for i in range(10):
                print('nop')
        field_name = 'optional_nested_message'
        proto_type = unittest_pb2.TestAllTypes
        self.assertEqual(proto_type.NestedMessage.DESCRIPTOR, proto_type.DESCRIPTOR.fields_by_name[field_name].message_type)

    def testEnums(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(4, unittest_pb2.FOREIGN_FOO)
        self.assertEqual(5, unittest_pb2.FOREIGN_BAR)
        self.assertEqual(6, unittest_pb2.FOREIGN_BAZ)
        proto = unittest_pb2.TestAllTypes()
        self.assertEqual(1, proto.FOO)
        self.assertEqual(1, unittest_pb2.TestAllTypes.FOO)
        self.assertEqual(2, proto.BAR)
        self.assertEqual(2, unittest_pb2.TestAllTypes.BAR)
        self.assertEqual(3, proto.BAZ)
        self.assertEqual(3, unittest_pb2.TestAllTypes.BAZ)

    def testExtremeDefaultValues(self):
        if False:
            print('Hello World!')
        message = unittest_pb2.TestExtremeDefaultValues()

        def isnan(val):
            if False:
                return 10
            return val != val

        def isinf(val):
            if False:
                i = 10
                return i + 15
            return not isnan(val) and isnan(val * 0)
        self.assertTrue(isinf(message.inf_double))
        self.assertTrue(message.inf_double > 0)
        self.assertTrue(isinf(message.neg_inf_double))
        self.assertTrue(message.neg_inf_double < 0)
        self.assertTrue(isnan(message.nan_double))
        self.assertTrue(isinf(message.inf_float))
        self.assertTrue(message.inf_float > 0)
        self.assertTrue(isinf(message.neg_inf_float))
        self.assertTrue(message.neg_inf_float < 0)
        self.assertTrue(isnan(message.nan_float))
        self.assertEqual('? ? ?? ?? ??? ??/ ??-', message.cpp_trigraph)

    def testHasDefaultValues(self):
        if False:
            print('Hello World!')
        desc = unittest_pb2.TestAllTypes.DESCRIPTOR
        expected_has_default_by_name = {'optional_int32': False, 'repeated_int32': False, 'optional_nested_message': False, 'default_int32': True}
        has_default_by_name = dict([(f.name, f.has_default_value) for f in desc.fields if f.name in expected_has_default_by_name])
        self.assertEqual(expected_has_default_by_name, has_default_by_name)

    def testContainingTypeBehaviorForExtensions(self):
        if False:
            print('Hello World!')
        self.assertEqual(unittest_pb2.optional_int32_extension.containing_type, unittest_pb2.TestAllExtensions.DESCRIPTOR)
        self.assertEqual(unittest_pb2.TestRequired.single.containing_type, unittest_pb2.TestAllExtensions.DESCRIPTOR)

    def testExtensionScope(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(unittest_pb2.optional_int32_extension.extension_scope, None)
        self.assertEqual(unittest_pb2.TestRequired.single.extension_scope, unittest_pb2.TestRequired.DESCRIPTOR)

    def testIsExtension(self):
        if False:
            while True:
                i = 10
        self.assertTrue(unittest_pb2.optional_int32_extension.is_extension)
        self.assertTrue(unittest_pb2.TestRequired.single.is_extension)
        message_descriptor = unittest_pb2.TestRequired.DESCRIPTOR
        non_extension_descriptor = message_descriptor.fields_by_name['a']
        self.assertTrue(not non_extension_descriptor.is_extension)

    def testOptions(self):
        if False:
            i = 10
            return i + 15
        proto = unittest_mset_pb2.TestMessageSet()
        self.assertTrue(proto.DESCRIPTOR.GetOptions().message_set_wire_format)

    def testMessageWithCustomOptions(self):
        if False:
            i = 10
            return i + 15
        proto = unittest_custom_options_pb2.TestMessageWithCustomOptions()
        enum_options = proto.DESCRIPTOR.enum_types_by_name['AnEnum'].GetOptions()
        self.assertTrue(enum_options is not None)

    def testNestedTypes(self):
        if False:
            return 10
        self.assertEquals(set(unittest_pb2.TestAllTypes.DESCRIPTOR.nested_types), set([unittest_pb2.TestAllTypes.NestedMessage.DESCRIPTOR, unittest_pb2.TestAllTypes.OptionalGroup.DESCRIPTOR, unittest_pb2.TestAllTypes.RepeatedGroup.DESCRIPTOR]))
        self.assertEqual(unittest_pb2.TestEmptyMessage.DESCRIPTOR.nested_types, [])
        self.assertEqual(unittest_pb2.TestAllTypes.NestedMessage.DESCRIPTOR.nested_types, [])

    def testContainingType(self):
        if False:
            while True:
                i = 10
        self.assertTrue(unittest_pb2.TestEmptyMessage.DESCRIPTOR.containing_type is None)
        self.assertTrue(unittest_pb2.TestAllTypes.DESCRIPTOR.containing_type is None)
        self.assertEqual(unittest_pb2.TestAllTypes.NestedMessage.DESCRIPTOR.containing_type, unittest_pb2.TestAllTypes.DESCRIPTOR)
        self.assertEqual(unittest_pb2.TestAllTypes.NestedMessage.DESCRIPTOR.containing_type, unittest_pb2.TestAllTypes.DESCRIPTOR)
        self.assertEqual(unittest_pb2.TestAllTypes.RepeatedGroup.DESCRIPTOR.containing_type, unittest_pb2.TestAllTypes.DESCRIPTOR)

    def testContainingTypeInEnumDescriptor(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(unittest_pb2._FOREIGNENUM.containing_type is None)
        self.assertEqual(unittest_pb2._TESTALLTYPES_NESTEDENUM.containing_type, unittest_pb2.TestAllTypes.DESCRIPTOR)

    def testPackage(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(unittest_pb2.TestAllTypes.DESCRIPTOR.file.package, 'protobuf_unittest')
        desc = unittest_pb2.TestAllTypes.NestedMessage.DESCRIPTOR
        self.assertEqual(desc.file.package, 'protobuf_unittest')
        self.assertEqual(unittest_import_pb2.ImportMessage.DESCRIPTOR.file.package, 'protobuf_unittest_import')
        self.assertEqual(unittest_pb2._FOREIGNENUM.file.package, 'protobuf_unittest')
        self.assertEqual(unittest_pb2._TESTALLTYPES_NESTEDENUM.file.package, 'protobuf_unittest')
        self.assertEqual(unittest_import_pb2._IMPORTENUM.file.package, 'protobuf_unittest_import')

    def testExtensionRange(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(unittest_pb2.TestAllTypes.DESCRIPTOR.extension_ranges, [])
        self.assertEqual(unittest_pb2.TestAllExtensions.DESCRIPTOR.extension_ranges, [(1, MAX_EXTENSION)])
        self.assertEqual(unittest_pb2.TestMultipleExtensionRanges.DESCRIPTOR.extension_ranges, [(42, 43), (4143, 4244), (65536, MAX_EXTENSION)])

    def testFileDescriptor(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(unittest_pb2.DESCRIPTOR.name, 'google/protobuf/unittest.proto')
        self.assertEqual(unittest_pb2.DESCRIPTOR.package, 'protobuf_unittest')
        self.assertFalse(unittest_pb2.DESCRIPTOR.serialized_pb is None)

    def testNoGenericServices(self):
        if False:
            print('Hello World!')
        self.assertTrue(hasattr(unittest_no_generic_services_pb2, 'TestMessage'))
        self.assertTrue(hasattr(unittest_no_generic_services_pb2, 'FOO'))
        self.assertTrue(hasattr(unittest_no_generic_services_pb2, 'test_extension'))
        if hasattr(unittest_no_generic_services_pb2, 'TestService'):
            self.assertFalse(issubclass(unittest_no_generic_services_pb2.TestService, service.Service))

    def testMessageTypesByName(self):
        if False:
            print('Hello World!')
        file_type = unittest_pb2.DESCRIPTOR
        self.assertEqual(unittest_pb2._TESTALLTYPES, file_type.message_types_by_name[unittest_pb2._TESTALLTYPES.name])
        self.assertFalse(unittest_pb2._TESTALLTYPES_NESTEDMESSAGE.name in file_type.message_types_by_name)

    def testPublicImports(self):
        if False:
            print('Hello World!')
        all_type_proto = unittest_pb2.TestAllTypes()
        self.assertEqual(0, all_type_proto.optional_public_import_message.e)
        public_import_proto = unittest_import_pb2.PublicImportMessage()
        self.assertEqual(0, public_import_proto.e)
        self.assertTrue(unittest_import_public_pb2.PublicImportMessage is unittest_import_pb2.PublicImportMessage)

    def testBadIdentifiers(self):
        if False:
            while True:
                i = 10
        message = test_bad_identifiers_pb2.TestBadIdentifiers()
        self.assertEqual(message.Extensions[test_bad_identifiers_pb2.message], 'foo')
        self.assertEqual(message.Extensions[test_bad_identifiers_pb2.descriptor], 'bar')
        self.assertEqual(message.Extensions[test_bad_identifiers_pb2.reflection], 'baz')
        self.assertEqual(message.Extensions[test_bad_identifiers_pb2.service], 'qux')
if __name__ == '__main__':
    unittest.main()