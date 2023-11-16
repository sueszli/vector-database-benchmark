"""Tests for proto ops reading descriptors from other sources."""
import os
import numpy as np
from google.protobuf.descriptor_pb2 import FieldDescriptorProto
from google.protobuf.descriptor_pb2 import FileDescriptorSet
from tensorflow.python.framework import dtypes
from tensorflow.python.kernel_tests.proto import proto_op_test_base as test_base
from tensorflow.python.platform import test

class DescriptorSourceTestBase(test.TestCase):
    """Base class for testing descriptor sources."""

    def __init__(self, decode_module, encode_module, methodName='runTest'):
        if False:
            while True:
                i = 10
        'DescriptorSourceTestBase initializer.\n\n    Args:\n      decode_module: a module containing the `decode_proto_op` method\n      encode_module: a module containing the `encode_proto_op` method\n      methodName: the name of the test method (same as for test.TestCase)\n    '
        super(DescriptorSourceTestBase, self).__init__(methodName)
        self._decode_module = decode_module
        self._encode_module = encode_module

    def _createDescriptorProto(self):
        if False:
            while True:
                i = 10
        proto = FileDescriptorSet()
        file_proto = proto.file.add(name='types.proto', package='tensorflow', syntax='proto3')
        enum_proto = file_proto.enum_type.add(name='DataType')
        enum_proto.value.add(name='DT_DOUBLE', number=0)
        enum_proto.value.add(name='DT_BOOL', number=1)
        file_proto = proto.file.add(name='test_example.proto', package='tensorflow.contrib.proto', dependency=['types.proto'])
        message_proto = file_proto.message_type.add(name='TestCase')
        message_proto.field.add(name='values', number=1, type=FieldDescriptorProto.TYPE_MESSAGE, type_name='.tensorflow.contrib.proto.TestValue', label=FieldDescriptorProto.LABEL_REPEATED)
        message_proto.field.add(name='shapes', number=2, type=FieldDescriptorProto.TYPE_INT32, label=FieldDescriptorProto.LABEL_REPEATED)
        message_proto.field.add(name='sizes', number=3, type=FieldDescriptorProto.TYPE_INT32, label=FieldDescriptorProto.LABEL_REPEATED)
        message_proto.field.add(name='fields', number=4, type=FieldDescriptorProto.TYPE_MESSAGE, type_name='.tensorflow.contrib.proto.FieldSpec', label=FieldDescriptorProto.LABEL_REPEATED)
        message_proto = file_proto.message_type.add(name='TestValue')
        message_proto.field.add(name='double_value', number=1, type=FieldDescriptorProto.TYPE_DOUBLE, label=FieldDescriptorProto.LABEL_REPEATED)
        message_proto.field.add(name='bool_value', number=2, type=FieldDescriptorProto.TYPE_BOOL, label=FieldDescriptorProto.LABEL_REPEATED)
        message_proto = file_proto.message_type.add(name='FieldSpec')
        message_proto.field.add(name='name', number=1, type=FieldDescriptorProto.TYPE_STRING, label=FieldDescriptorProto.LABEL_OPTIONAL)
        message_proto.field.add(name='dtype', number=2, type=FieldDescriptorProto.TYPE_ENUM, type_name='.tensorflow.DataType', label=FieldDescriptorProto.LABEL_OPTIONAL)
        message_proto.field.add(name='value', number=3, type=FieldDescriptorProto.TYPE_MESSAGE, type_name='.tensorflow.contrib.proto.TestValue', label=FieldDescriptorProto.LABEL_OPTIONAL)
        return proto

    def _writeProtoToFile(self, proto):
        if False:
            for i in range(10):
                print('nop')
        fn = os.path.join(self.get_temp_dir(), 'descriptor.pb')
        with open(fn, 'wb') as f:
            f.write(proto.SerializeToString())
        return fn

    def _testRoundtrip(self, descriptor_source):
        if False:
            while True:
                i = 10
        in_bufs = np.array([test_base.ProtoOpTestBase.simple_test_case().SerializeToString()], dtype=object)
        message_type = 'tensorflow.contrib.proto.TestCase'
        field_names = ['values', 'shapes', 'sizes', 'fields']
        tensor_types = [dtypes.string, dtypes.int32, dtypes.int32, dtypes.string]
        with self.cached_session() as sess:
            (sizes, field_tensors) = self._decode_module.decode_proto(in_bufs, message_type=message_type, field_names=field_names, output_types=tensor_types, descriptor_source=descriptor_source)
            out_tensors = self._encode_module.encode_proto(sizes, field_tensors, message_type=message_type, field_names=field_names, descriptor_source=descriptor_source)
            (out_bufs,) = sess.run([out_tensors])
            self.assertEqual(in_bufs.shape, out_bufs.shape)
            for (in_buf, out_buf) in zip(in_bufs.flat, out_bufs.flat):
                self.assertEqual(in_buf, out_buf)

    def testWithFileDescriptorSet(self):
        if False:
            print('Hello World!')
        with self.assertRaisesOpError('No descriptor found for message type'):
            self._testRoundtrip(b'local://')
        proto = self._createDescriptorProto()
        proto_file = self._writeProtoToFile(proto)
        self._testRoundtrip(proto_file)
        self._testRoundtrip(b'bytes://' + proto.SerializeToString())