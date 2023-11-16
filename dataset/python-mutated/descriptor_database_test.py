"""Tests for google.protobuf.descriptor_database."""
__author__ = 'matthewtoia@google.com (Matt Toia)'
try:
    import unittest2 as unittest
except ImportError:
    import unittest
from google.protobuf import descriptor_pb2
from google.protobuf.internal import factory_test2_pb2
from google.protobuf import descriptor_database

class DescriptorDatabaseTest(unittest.TestCase):

    def testAdd(self):
        if False:
            i = 10
            return i + 15
        db = descriptor_database.DescriptorDatabase()
        file_desc_proto = descriptor_pb2.FileDescriptorProto.FromString(factory_test2_pb2.DESCRIPTOR.serialized_pb)
        db.Add(file_desc_proto)
        self.assertEqual(file_desc_proto, db.FindFileByName('google/protobuf/internal/factory_test2.proto'))
        self.assertEqual(file_desc_proto, db.FindFileContainingSymbol('google.protobuf.python.internal.Factory2Message'))
        self.assertEqual(file_desc_proto, db.FindFileContainingSymbol('google.protobuf.python.internal.Factory2Message.NestedFactory2Message'))
        self.assertEqual(file_desc_proto, db.FindFileContainingSymbol('google.protobuf.python.internal.Factory2Enum'))
        self.assertEqual(file_desc_proto, db.FindFileContainingSymbol('google.protobuf.python.internal.Factory2Message.NestedFactory2Enum'))
        self.assertEqual(file_desc_proto, db.FindFileContainingSymbol('google.protobuf.python.internal.MessageWithNestedEnumOnly.NestedEnum'))
if __name__ == '__main__':
    unittest.main()