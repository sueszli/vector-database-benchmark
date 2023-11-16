"""
pika.data tests

"""
import datetime
import decimal
import unittest
from collections import OrderedDict
from pika.compat import PY2, PY3
from pika import data, exceptions
from pika.compat import long

class DataTests(unittest.TestCase):
    FIELD_TBL_ENCODED = b'\x00\x00\x00\xef\x05arrayA\x00\x00\x00\x0fI\x00\x00\x00\x01I\x00\x00\x00\x02I\x00\x00\x00\x03\x07boolvalt\x01\x07decimalD\x02\x00\x00\x01:\x0bdecimal_tooD\x00\x00\x00\x00d\x07dictvalF\x00\x00\x00\x0c\x03fooS\x00\x00\x00\x03bar\x06intvalI\x00\x00\x00\x01\x06bigintl\x00\x00\x00\x00\x9a~\xc8\x00\x07longvall\x00\x00\x00\x006e&U\tmaxLLUINTl\xff\xff\xff\xff\xff\xff\xff\xff\x04nullV\x06strvalS\x00\x00\x00\x04Test\x0ctimestampvalT\x00\x00\x00\x00Ec)\x92\x07unicodeS\x00\x00\x00\x08utf8=\xe2\x9c\x93'
    FIELD_TBL_ENCODED += b'\x05bytesx\x00\x00\x00\x06foobar' if PY3 else b'\x05bytesS\x00\x00\x00\x06foobar'
    FIELD_TBL_VALUE = OrderedDict([('array', [1, 2, 3]), ('boolval', True), ('decimal', decimal.Decimal('3.14')), ('decimal_too', decimal.Decimal('100')), ('dictval', {'foo': 'bar'}), ('intval', 1), ('bigint', 2592000000), ('longval', long(912598613)), ('maxLLUINT', long(2 ** 64 - 1)), ('null', None), ('strval', 'Test'), ('timestampval', datetime.datetime(2006, 11, 21, 16, 30, 10)), ('unicode', u'utf8=✓'), ('bytes', b'foobar')])

    def test_decode_bytes(self):
        if False:
            return 10
        input = b'\x00\x00\x00\x01\x05bytesx\x00\x00\x00\x06foobar'
        result = data.decode_table(input, 0)
        self.assertEqual(result, ({'bytes': b'foobar'}, 21))

    def test_decode_shortint(self):
        if False:
            return 10
        input = b'\x00\x00\x00\x01\x08shortints\x04\xd2'
        result = data.decode_table(input, 0)
        self.assertEqual(result, ({'shortint': 1234}, 16))

    def test_encode_table(self):
        if False:
            while True:
                i = 10
        result = []
        data.encode_table(result, self.FIELD_TBL_VALUE)
        self.assertEqual(b''.join(result), self.FIELD_TBL_ENCODED)

    def test_encode_table_bytes(self):
        if False:
            for i in range(10):
                print('nop')
        result = []
        byte_count = data.encode_table(result, self.FIELD_TBL_VALUE)
        self.assertEqual(byte_count, 243)

    def test_decode_table(self):
        if False:
            i = 10
            return i + 15
        (value, byte_count) = data.decode_table(self.FIELD_TBL_ENCODED, 0)
        self.assertDictEqual(value, self.FIELD_TBL_VALUE)

    def test_decode_table_bytes(self):
        if False:
            print('Hello World!')
        (value, byte_count) = data.decode_table(self.FIELD_TBL_ENCODED, 0)
        self.assertEqual(byte_count, 243)

    def test_encode_raises(self):
        if False:
            return 10
        self.assertRaises(exceptions.UnsupportedAMQPFieldException, data.encode_table, [], {'foo': {1, 2, 3}})

    def test_decode_raises(self):
        if False:
            return 10
        self.assertRaises(exceptions.InvalidFieldTypeException, data.decode_table, b'\x00\x00\x00\t\x03fooZ\x00\x00\x04\xd2', 0)

    def test_long_repr(self):
        if False:
            return 10
        value = long(912598613)
        self.assertEqual(repr(value), '912598613L')