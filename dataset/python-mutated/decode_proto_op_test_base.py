"""Tests for decode_proto op."""
import itertools
from absl.testing import parameterized
import numpy as np
from google.protobuf import text_format
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.kernel_tests.proto import proto_op_test_base as test_base
from tensorflow.python.kernel_tests.proto import test_example_pb2

class DecodeProtoOpTestBase(test_base.ProtoOpTestBase, parameterized.TestCase):
    """Base class for testing proto decoding ops."""

    def __init__(self, decode_module, methodName='runTest'):
        if False:
            print('Hello World!')
        'DecodeProtoOpTestBase initializer.\n\n    Args:\n      decode_module: a module containing the `decode_proto_op` method\n      methodName: the name of the test method (same as for test.TestCase)\n    '
        super(DecodeProtoOpTestBase, self).__init__(methodName)
        self._decode_module = decode_module

    def _compareValues(self, fd, vs, evs):
        if False:
            i = 10
            return i + 15
        'Compare lists/arrays of field values.'
        if len(vs) != len(evs):
            self.fail('Field %s decoded %d outputs, expected %d' % (fd.name, len(vs), len(evs)))
        for (i, ev) in enumerate(evs):
            if fd.cpp_type == fd.CPPTYPE_FLOAT:
                self.assertTrue(np.isclose(vs[i], ev), 'expected %r, actual %r' % (ev, vs[i]))
            elif fd.cpp_type == fd.CPPTYPE_STRING:
                self.assertEqual(vs[i], ev.encode('ascii'))
            else:
                self.assertEqual(vs[i], ev)

    def _compareProtos(self, batch_shape, sizes, fields, field_dict):
        if False:
            return 10
        'Compare protos of type TestValue.\n\n    Args:\n      batch_shape: the shape of the input tensor of serialized messages.\n      sizes: int matrix of repeat counts returned by decode_proto\n      fields: list of test_example_pb2.FieldSpec (types and expected values)\n      field_dict: map from field names to decoded numpy tensors of values\n    '
        for field in fields:
            values = field_dict[field.name]
            self.assertEqual(dtypes.as_dtype(values.dtype), field.dtype)
            if 'ext_value' in field.name:
                fd = test_example_pb2.PrimitiveValue()
            else:
                fd = field.value.DESCRIPTOR.fields_by_name[field.name]
            self.assertEqual(list(values.shape)[:-1], batch_shape)
            if field.name == 'message_value' or 'ext_value' in field.name:
                vs = []
                for buf in values.flat:
                    msg = test_example_pb2.PrimitiveValue()
                    msg.ParseFromString(buf)
                    vs.append(msg)
                if 'ext_value' in field.name:
                    evs = field.value.Extensions[test_example_pb2.ext_value]
                else:
                    evs = getattr(field.value, field.name)
                if len(vs) != len(evs):
                    self.fail('Field %s decoded %d outputs, expected %d' % (fd.name, len(vs), len(evs)))
                for (v, ev) in zip(vs, evs):
                    self.assertEqual(v, ev)
                continue
            tf_type_to_primitive_value_field = {dtypes.bool: 'bool_value', dtypes.float32: 'float_value', dtypes.float64: 'double_value', dtypes.int8: 'int8_value', dtypes.int32: 'int32_value', dtypes.int64: 'int64_value', dtypes.string: 'string_value', dtypes.uint8: 'uint8_value', dtypes.uint32: 'uint32_value', dtypes.uint64: 'uint64_value'}
            if field.name in ['enum_value', 'enum_value_with_default']:
                tf_field_name = 'enum_value'
            else:
                tf_field_name = tf_type_to_primitive_value_field.get(field.dtype)
            if tf_field_name is None:
                self.fail('Unhandled tensorflow type %d' % field.dtype)
            self._compareValues(fd, values.flat, getattr(field.value, tf_field_name))

    def _runDecodeProtoTests(self, fields, case_sizes, batch_shape, batch, message_type, message_format, sanitize, force_disordered=False):
        if False:
            print('Hello World!')
        "Run decode tests on a batch of messages.\n\n    Args:\n      fields: list of test_example_pb2.FieldSpec (types and expected values)\n      case_sizes: expected sizes array\n      batch_shape: the shape of the input tensor of serialized messages\n      batch: list of serialized messages\n      message_type: descriptor name for messages\n      message_format: format of messages, 'text' or 'binary'\n      sanitize: whether to sanitize binary protobuf inputs\n      force_disordered: whether to force fields encoded out of order.\n    "
        if force_disordered:
            assert not sanitize
            extra_fields = test_example_pb2.ExtraFields()
            extra_fields.string_value = 'IGNORE ME'
            extra_fields.bool_value = False
            extra_msg = extra_fields.SerializeToString()
            batch = [extra_msg + msg for msg in batch]
        batch = np.array(batch, dtype=object)
        batch = np.reshape(batch, batch_shape)
        field_names = [f.name for f in fields]
        output_types = [f.dtype for f in fields]
        with self.cached_session() as sess:
            (sizes, vtensor) = self._decode_module.decode_proto(batch, message_type=message_type, field_names=field_names, output_types=output_types, message_format=message_format, sanitize=sanitize)
            vlist = sess.run([sizes] + vtensor)
            sizes = vlist[0]
            value_tensors = vlist[1:]
            self.assertTrue(np.all(np.array(sizes.shape) == batch_shape + [len(field_names)]))
            self.assertEqual(len(sizes.flat), len(case_sizes))
            self.assertTrue(np.all(sizes.flat == np.array(case_sizes, dtype=np.int32)))
            field_dict = dict(zip(field_names, value_tensors))
            self._compareProtos(batch_shape, sizes, fields, field_dict)

    @parameterized.named_parameters(*test_base.ProtoOpTestBase.named_parameters())
    def testBinary(self, case):
        if False:
            print('Hello World!')
        batch = [value.SerializeToString() for value in case.values]
        self._runDecodeProtoTests(case.fields, case.sizes, list(case.shapes), batch, 'tensorflow.contrib.proto.TestValue', 'binary', sanitize=False)

    @parameterized.named_parameters(*test_base.ProtoOpTestBase.named_parameters())
    def testBinaryDisordered(self, case):
        if False:
            while True:
                i = 10
        batch = [value.SerializeToString() for value in case.values]
        self._runDecodeProtoTests(case.fields, case.sizes, list(case.shapes), batch, 'tensorflow.contrib.proto.TestValue', 'binary', sanitize=False, force_disordered=True)

    @parameterized.named_parameters(*test_base.ProtoOpTestBase.named_parameters(extension=False))
    def testPacked(self, case):
        if False:
            i = 10
            return i + 15
        packed_batch = [text_format.Parse(text_format.MessageToString(value, float_format='.17g'), test_example_pb2.PackedTestValue()).SerializeToString() for value in case.values]
        self._runDecodeProtoTests(case.fields, case.sizes, list(case.shapes), packed_batch, 'tensorflow.contrib.proto.PackedTestValue', 'binary', sanitize=False)

    @parameterized.named_parameters(*test_base.ProtoOpTestBase.named_parameters())
    def testText(self, case):
        if False:
            while True:
                i = 10
        text_batch = [text_format.MessageToString(value, float_format='.17g') for value in case.values]
        self._runDecodeProtoTests(case.fields, case.sizes, list(case.shapes), text_batch, 'tensorflow.contrib.proto.TestValue', 'text', sanitize=False)

    @parameterized.named_parameters(*test_base.ProtoOpTestBase.named_parameters())
    def testSanitizerGood(self, case):
        if False:
            return 10
        batch = [value.SerializeToString() for value in case.values]
        self._runDecodeProtoTests(case.fields, case.sizes, list(case.shapes), batch, 'tensorflow.contrib.proto.TestValue', 'binary', sanitize=True)

    @parameterized.parameters(False, True)
    def testCorruptProtobuf(self, sanitize):
        if False:
            print('Hello World!')
        corrupt_proto = 'This is not a binary protobuf'
        batch = np.array(corrupt_proto, dtype=object)
        msg_type = 'tensorflow.contrib.proto.TestCase'
        field_names = ['sizes']
        field_types = [dtypes.int32]
        with self.assertRaisesRegexp(errors.DataLossError, 'Unable to parse binary protobuf|Failed to consume entire buffer'):
            self.evaluate(self._decode_module.decode_proto(batch, message_type=msg_type, field_names=field_names, output_types=field_types, sanitize=sanitize))

    def testOutOfOrderRepeated(self):
        if False:
            while True:
                i = 10
        fragments = [test_example_pb2.TestValue(double_value=[1.0]).SerializeToString(), test_example_pb2.TestValue(message_value=[test_example_pb2.PrimitiveValue(string_value='abc')]).SerializeToString(), test_example_pb2.TestValue(message_value=[test_example_pb2.PrimitiveValue(string_value='def')]).SerializeToString()]
        all_fields_to_parse = ['double_value', 'message_value']
        field_types = {'double_value': dtypes.double, 'message_value': dtypes.string}
        for indices in itertools.permutations(range(len(fragments))):
            proto = b''.join((fragments[i] for i in indices))
            for i in indices:
                if i == 1:
                    expected_message_values = [test_example_pb2.PrimitiveValue(string_value='abc').SerializeToString(), test_example_pb2.PrimitiveValue(string_value='def').SerializeToString()]
                    break
                if i == 2:
                    expected_message_values = [test_example_pb2.PrimitiveValue(string_value='def').SerializeToString(), test_example_pb2.PrimitiveValue(string_value='abc').SerializeToString()]
                    break
            expected_field_values = {'double_value': [[1.0]], 'message_value': [expected_message_values]}
            for num_fields_to_parse in range(len(all_fields_to_parse)):
                for comb in itertools.combinations(all_fields_to_parse, num_fields_to_parse):
                    parsed_values = self.evaluate(self._decode_module.decode_proto([proto], message_type='tensorflow.contrib.proto.TestValue', field_names=comb, output_types=[field_types[f] for f in comb], sanitize=False)).values
                    self.assertLen(parsed_values, len(comb))
                    for (field_name, parsed) in zip(comb, parsed_values):
                        self.assertAllEqual(parsed, expected_field_values[field_name], 'perm: {}, comb: {}'.format(indices, comb))