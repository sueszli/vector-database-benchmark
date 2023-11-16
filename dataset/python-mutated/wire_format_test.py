"""Test for google.protobuf.internal.wire_format."""
__author__ = 'robinson@google.com (Will Robinson)'
import unittest
from google.protobuf import message
from google.protobuf.internal import wire_format

class WireFormatTest(unittest.TestCase):

    def testPackTag(self):
        if False:
            while True:
                i = 10
        field_number = 2748
        tag_type = 2
        self.assertEqual(field_number << 3 | tag_type, wire_format.PackTag(field_number, tag_type))
        PackTag = wire_format.PackTag
        self.assertRaises(message.EncodeError, PackTag, field_number, 6)
        self.assertRaises(message.EncodeError, PackTag, field_number, -1)

    def testUnpackTag(self):
        if False:
            while True:
                i = 10
        for expected_field_number in (1, 15, 16, 2047, 2048):
            for expected_wire_type in range(6):
                (field_number, wire_type) = wire_format.UnpackTag(wire_format.PackTag(expected_field_number, expected_wire_type))
                self.assertEqual(expected_field_number, field_number)
                self.assertEqual(expected_wire_type, wire_type)
        self.assertRaises(TypeError, wire_format.UnpackTag, None)
        self.assertRaises(TypeError, wire_format.UnpackTag, 'abc')
        self.assertRaises(TypeError, wire_format.UnpackTag, 0.0)
        self.assertRaises(TypeError, wire_format.UnpackTag, object())

    def testZigZagEncode(self):
        if False:
            i = 10
            return i + 15
        Z = wire_format.ZigZagEncode
        self.assertEqual(0, Z(0))
        self.assertEqual(1, Z(-1))
        self.assertEqual(2, Z(1))
        self.assertEqual(3, Z(-2))
        self.assertEqual(4, Z(2))
        self.assertEqual(4294967294, Z(2147483647))
        self.assertEqual(4294967295, Z(-2147483648))
        self.assertEqual(18446744073709551614, Z(9223372036854775807))
        self.assertEqual(18446744073709551615, Z(-9223372036854775808))
        self.assertRaises(TypeError, Z, None)
        self.assertRaises(TypeError, Z, 'abcd')
        self.assertRaises(TypeError, Z, 0.0)
        self.assertRaises(TypeError, Z, object())

    def testZigZagDecode(self):
        if False:
            while True:
                i = 10
        Z = wire_format.ZigZagDecode
        self.assertEqual(0, Z(0))
        self.assertEqual(-1, Z(1))
        self.assertEqual(1, Z(2))
        self.assertEqual(-2, Z(3))
        self.assertEqual(2, Z(4))
        self.assertEqual(2147483647, Z(4294967294))
        self.assertEqual(-2147483648, Z(4294967295))
        self.assertEqual(9223372036854775807, Z(18446744073709551614))
        self.assertEqual(-9223372036854775808, Z(18446744073709551615))
        self.assertRaises(TypeError, Z, None)
        self.assertRaises(TypeError, Z, 'abcd')
        self.assertRaises(TypeError, Z, 0.0)
        self.assertRaises(TypeError, Z, object())

    def NumericByteSizeTestHelper(self, byte_size_fn, value, expected_value_size):
        if False:
            i = 10
            return i + 15
        for (field_number, tag_bytes) in ((15, 1), (16, 2), (2047, 2), (2048, 3)):
            expected_size = expected_value_size + tag_bytes
            actual_size = byte_size_fn(field_number, value)
            self.assertEqual(expected_size, actual_size, 'byte_size_fn: %s, field_number: %d, value: %r\nExpected: %d, Actual: %d' % (byte_size_fn, field_number, value, expected_size, actual_size))

    def testByteSizeFunctions(self):
        if False:
            print('Hello World!')
        NUMERIC_ARGS = [[wire_format.Int32ByteSize, 0, 1], [wire_format.Int32ByteSize, 127, 1], [wire_format.Int32ByteSize, 128, 2], [wire_format.Int32ByteSize, -1, 10], [wire_format.Int64ByteSize, 0, 1], [wire_format.Int64ByteSize, 127, 1], [wire_format.Int64ByteSize, 128, 2], [wire_format.Int64ByteSize, -1, 10], [wire_format.UInt32ByteSize, 0, 1], [wire_format.UInt32ByteSize, 127, 1], [wire_format.UInt32ByteSize, 128, 2], [wire_format.UInt32ByteSize, wire_format.UINT32_MAX, 5], [wire_format.UInt64ByteSize, 0, 1], [wire_format.UInt64ByteSize, 127, 1], [wire_format.UInt64ByteSize, 128, 2], [wire_format.UInt64ByteSize, wire_format.UINT64_MAX, 10], [wire_format.SInt32ByteSize, 0, 1], [wire_format.SInt32ByteSize, -1, 1], [wire_format.SInt32ByteSize, 1, 1], [wire_format.SInt32ByteSize, -63, 1], [wire_format.SInt32ByteSize, 63, 1], [wire_format.SInt32ByteSize, -64, 1], [wire_format.SInt32ByteSize, 64, 2], [wire_format.SInt64ByteSize, 0, 1], [wire_format.SInt64ByteSize, -1, 1], [wire_format.SInt64ByteSize, 1, 1], [wire_format.SInt64ByteSize, -63, 1], [wire_format.SInt64ByteSize, 63, 1], [wire_format.SInt64ByteSize, -64, 1], [wire_format.SInt64ByteSize, 64, 2], [wire_format.Fixed32ByteSize, 0, 4], [wire_format.Fixed32ByteSize, wire_format.UINT32_MAX, 4], [wire_format.Fixed64ByteSize, 0, 8], [wire_format.Fixed64ByteSize, wire_format.UINT64_MAX, 8], [wire_format.SFixed32ByteSize, 0, 4], [wire_format.SFixed32ByteSize, wire_format.INT32_MIN, 4], [wire_format.SFixed32ByteSize, wire_format.INT32_MAX, 4], [wire_format.SFixed64ByteSize, 0, 8], [wire_format.SFixed64ByteSize, wire_format.INT64_MIN, 8], [wire_format.SFixed64ByteSize, wire_format.INT64_MAX, 8], [wire_format.FloatByteSize, 0.0, 4], [wire_format.FloatByteSize, 1000000000.0, 4], [wire_format.FloatByteSize, -1000000000.0, 4], [wire_format.DoubleByteSize, 0.0, 8], [wire_format.DoubleByteSize, 1000000000.0, 8], [wire_format.DoubleByteSize, -1000000000.0, 8], [wire_format.BoolByteSize, False, 1], [wire_format.BoolByteSize, True, 1], [wire_format.EnumByteSize, 0, 1], [wire_format.EnumByteSize, 127, 1], [wire_format.EnumByteSize, 128, 2], [wire_format.EnumByteSize, wire_format.UINT32_MAX, 5]]
        for args in NUMERIC_ARGS:
            self.NumericByteSizeTestHelper(*args)
        for byte_size_fn in (wire_format.StringByteSize, wire_format.BytesByteSize):
            self.assertEqual(5, byte_size_fn(10, 'abc'))
            self.assertEqual(6, byte_size_fn(16, 'abc'))
            self.assertEqual(132, byte_size_fn(16, 'a' * 128))
        self.assertEqual(10, wire_format.StringByteSize(5, unicode('Ð¢ÐµÑ\x81Ñ\x82', 'utf-8')))

        class MockMessage(object):

            def __init__(self, byte_size):
                if False:
                    i = 10
                    return i + 15
                self.byte_size = byte_size

            def ByteSize(self):
                if False:
                    i = 10
                    return i + 15
                return self.byte_size
        message_byte_size = 10
        mock_message = MockMessage(byte_size=message_byte_size)
        self.assertEqual(2 + message_byte_size, wire_format.GroupByteSize(1, mock_message))
        self.assertEqual(4 + message_byte_size, wire_format.GroupByteSize(16, mock_message))
        self.assertEqual(2 + mock_message.byte_size, wire_format.MessageByteSize(1, mock_message))
        self.assertEqual(3 + mock_message.byte_size, wire_format.MessageByteSize(16, mock_message))
        mock_message.byte_size = 128
        self.assertEqual(4 + mock_message.byte_size, wire_format.MessageByteSize(16, mock_message))
        mock_message.byte_size = 10
        self.assertEqual(mock_message.byte_size + 6, wire_format.MessageSetItemByteSize(1, mock_message))
        mock_message.byte_size = 128
        self.assertEqual(mock_message.byte_size + 7, wire_format.MessageSetItemByteSize(1, mock_message))
        self.assertEqual(mock_message.byte_size + 8, wire_format.MessageSetItemByteSize(128, mock_message))
        self.assertRaises(message.EncodeError, wire_format.UInt64ByteSize, 1, 1 << 128)
if __name__ == '__main__':
    unittest.main()