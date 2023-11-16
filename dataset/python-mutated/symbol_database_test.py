"""Tests for google.protobuf.symbol_database."""
try:
    import unittest2 as unittest
except ImportError:
    import unittest
from google.protobuf import unittest_pb2
from google.protobuf import descriptor
from google.protobuf import descriptor_pool
from google.protobuf import symbol_database

class SymbolDatabaseTest(unittest.TestCase):

    def _Database(self):
        if False:
            print('Hello World!')
        if descriptor._USE_C_DESCRIPTORS:
            db = symbol_database.SymbolDatabase(pool=descriptor_pool.Default())
        else:
            db = symbol_database.SymbolDatabase()
        db.RegisterFileDescriptor(unittest_pb2.DESCRIPTOR)
        db.RegisterMessage(unittest_pb2.TestAllTypes)
        db.RegisterMessage(unittest_pb2.TestAllTypes.NestedMessage)
        db.RegisterMessage(unittest_pb2.TestAllTypes.OptionalGroup)
        db.RegisterMessage(unittest_pb2.TestAllTypes.RepeatedGroup)
        db.RegisterEnumDescriptor(unittest_pb2.ForeignEnum.DESCRIPTOR)
        db.RegisterEnumDescriptor(unittest_pb2.TestAllTypes.NestedEnum.DESCRIPTOR)
        db.RegisterServiceDescriptor(unittest_pb2._TESTSERVICE)
        return db

    def testGetPrototype(self):
        if False:
            i = 10
            return i + 15
        instance = self._Database().GetPrototype(unittest_pb2.TestAllTypes.DESCRIPTOR)
        self.assertTrue(instance is unittest_pb2.TestAllTypes)

    def testGetMessages(self):
        if False:
            print('Hello World!')
        messages = self._Database().GetMessages(['google/protobuf/unittest.proto'])
        self.assertTrue(unittest_pb2.TestAllTypes is messages['protobuf_unittest.TestAllTypes'])

    def testGetSymbol(self):
        if False:
            return 10
        self.assertEqual(unittest_pb2.TestAllTypes, self._Database().GetSymbol('protobuf_unittest.TestAllTypes'))
        self.assertEqual(unittest_pb2.TestAllTypes.NestedMessage, self._Database().GetSymbol('protobuf_unittest.TestAllTypes.NestedMessage'))
        self.assertEqual(unittest_pb2.TestAllTypes.OptionalGroup, self._Database().GetSymbol('protobuf_unittest.TestAllTypes.OptionalGroup'))
        self.assertEqual(unittest_pb2.TestAllTypes.RepeatedGroup, self._Database().GetSymbol('protobuf_unittest.TestAllTypes.RepeatedGroup'))

    def testEnums(self):
        if False:
            while True:
                i = 10
        self.assertEqual('protobuf_unittest.ForeignEnum', self._Database().pool.FindEnumTypeByName('protobuf_unittest.ForeignEnum').full_name)
        self.assertEqual('protobuf_unittest.TestAllTypes.NestedEnum', self._Database().pool.FindEnumTypeByName('protobuf_unittest.TestAllTypes.NestedEnum').full_name)

    def testFindMessageTypeByName(self):
        if False:
            print('Hello World!')
        self.assertEqual('protobuf_unittest.TestAllTypes', self._Database().pool.FindMessageTypeByName('protobuf_unittest.TestAllTypes').full_name)
        self.assertEqual('protobuf_unittest.TestAllTypes.NestedMessage', self._Database().pool.FindMessageTypeByName('protobuf_unittest.TestAllTypes.NestedMessage').full_name)

    def testFindServiceByName(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('protobuf_unittest.TestService', self._Database().pool.FindServiceByName('protobuf_unittest.TestService').full_name)

    def testFindFileContainingSymbol(self):
        if False:
            while True:
                i = 10
        self.assertEqual('google/protobuf/unittest.proto', self._Database().pool.FindFileContainingSymbol('protobuf_unittest.TestAllTypes.NestedEnum').name)
        self.assertEqual('google/protobuf/unittest.proto', self._Database().pool.FindFileContainingSymbol('protobuf_unittest.TestAllTypes').name)

    def testFindFileByName(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('google/protobuf/unittest.proto', self._Database().pool.FindFileByName('google/protobuf/unittest.proto').name)
if __name__ == '__main__':
    unittest.main()