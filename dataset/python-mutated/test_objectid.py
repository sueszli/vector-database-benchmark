"""Tests for the objectid module."""
from __future__ import annotations
import datetime
import pickle
import struct
import sys
sys.path[0:0] = ['']
from test import SkipTest, unittest
from test.utils import oid_generated_on_process
from bson.errors import InvalidId
from bson.objectid import _MAX_COUNTER_VALUE, ObjectId
from bson.tz_util import FixedOffset, utc

def oid(x):
    if False:
        while True:
            i = 10
    return ObjectId()

class TestObjectId(unittest.TestCase):

    def test_creation(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, ObjectId, 4)
        self.assertRaises(TypeError, ObjectId, 175.0)
        self.assertRaises(TypeError, ObjectId, {'test': 4})
        self.assertRaises(TypeError, ObjectId, ['something'])
        self.assertRaises(InvalidId, ObjectId, '')
        self.assertRaises(InvalidId, ObjectId, '12345678901')
        self.assertRaises(InvalidId, ObjectId, '1234567890123')
        self.assertTrue(ObjectId())
        self.assertTrue(ObjectId(b'123456789012'))
        a = ObjectId()
        self.assertTrue(ObjectId(a))

    def test_unicode(self):
        if False:
            for i in range(10):
                print('nop')
        a = ObjectId()
        self.assertEqual(a, ObjectId(a))
        self.assertRaises(InvalidId, ObjectId, 'hello')

    def test_from_hex(self):
        if False:
            print('Hello World!')
        ObjectId('123456789012123456789012')
        self.assertRaises(InvalidId, ObjectId, '123456789012123456789G12')

    def test_repr_str(self):
        if False:
            return 10
        self.assertEqual(repr(ObjectId('1234567890abcdef12345678')), "ObjectId('1234567890abcdef12345678')")
        self.assertEqual(str(ObjectId('1234567890abcdef12345678')), '1234567890abcdef12345678')
        self.assertEqual(str(ObjectId(b'123456789012')), '313233343536373839303132')
        self.assertEqual(ObjectId('1234567890abcdef12345678').binary, b'\x124Vx\x90\xab\xcd\xef\x124Vx')
        self.assertEqual(str(ObjectId(b'\x124Vx\x90\xab\xcd\xef\x124Vx')), '1234567890abcdef12345678')

    def test_equality(self):
        if False:
            for i in range(10):
                print('nop')
        a = ObjectId()
        self.assertEqual(a, ObjectId(a))
        self.assertEqual(ObjectId(b'123456789012'), ObjectId(b'123456789012'))
        self.assertNotEqual(ObjectId(), ObjectId())
        self.assertNotEqual(ObjectId(b'123456789012'), b'123456789012')
        self.assertFalse(a != ObjectId(a))
        self.assertFalse(ObjectId(b'123456789012') != ObjectId(b'123456789012'))

    def test_binary_str_equivalence(self):
        if False:
            return 10
        a = ObjectId()
        self.assertEqual(a, ObjectId(a.binary))
        self.assertEqual(a, ObjectId(str(a)))

    def test_generation_time(self):
        if False:
            return 10
        d1 = datetime.datetime.now(tz=datetime.timezone.utc).replace(tzinfo=None)
        d2 = ObjectId().generation_time
        self.assertEqual(utc, d2.tzinfo)
        d2 = d2.replace(tzinfo=None)
        self.assertTrue(d2 - d1 < datetime.timedelta(seconds=2))

    def test_from_datetime(self):
        if False:
            print('Hello World!')
        if 'PyPy 1.8.0' in sys.version:
            raise SkipTest('datetime.timedelta is broken in pypy 1.8.0')
        d = datetime.datetime.now(tz=datetime.timezone.utc).replace(tzinfo=None)
        d = d - datetime.timedelta(microseconds=d.microsecond)
        oid = ObjectId.from_datetime(d)
        self.assertEqual(d, oid.generation_time.replace(tzinfo=None))
        self.assertEqual('0' * 16, str(oid)[8:])
        aware = datetime.datetime(1993, 4, 4, 2, tzinfo=FixedOffset(555, 'SomeZone'))
        offset = aware.utcoffset()
        assert offset is not None
        as_utc = (aware - offset).replace(tzinfo=utc)
        oid = ObjectId.from_datetime(aware)
        self.assertEqual(as_utc, oid.generation_time)

    def test_pickling(self):
        if False:
            return 10
        orig = ObjectId()
        for protocol in [0, 1, 2, -1]:
            pkl = pickle.dumps(orig, protocol=protocol)
            self.assertEqual(orig, pickle.loads(pkl))

    def test_pickle_backwards_compatability(self):
        if False:
            for i in range(10):
                print('nop')
        pickled_with_1_9 = b"ccopy_reg\n_reconstructor\np0\n(cbson.objectid\nObjectId\np1\nc__builtin__\nobject\np2\nNtp3\nRp4\n(dp5\nS'_ObjectId__id'\np6\nS'M\\x9afV\\x13v\\xc0\\x0b\\x88\\x00\\x00\\x00'\np7\nsb."
        pickled_with_1_10 = b"ccopy_reg\n_reconstructor\np0\n(cbson.objectid\nObjectId\np1\nc__builtin__\nobject\np2\nNtp3\nRp4\nS'M\\x9afV\\x13v\\xc0\\x0b\\x88\\x00\\x00\\x00'\np5\nb."
        oid_1_9 = pickle.loads(pickled_with_1_9, encoding='latin-1')
        oid_1_10 = pickle.loads(pickled_with_1_10, encoding='latin-1')
        self.assertEqual(oid_1_9, ObjectId('4d9a66561376c00b88000000'))
        self.assertEqual(oid_1_9, oid_1_10)

    def test_random_bytes(self):
        if False:
            print('Hello World!')
        self.assertTrue(oid_generated_on_process(ObjectId()))

    def test_is_valid(self):
        if False:
            while True:
                i = 10
        self.assertFalse(ObjectId.is_valid(None))
        self.assertFalse(ObjectId.is_valid(4))
        self.assertFalse(ObjectId.is_valid(175.0))
        self.assertFalse(ObjectId.is_valid({'test': 4}))
        self.assertFalse(ObjectId.is_valid(['something']))
        self.assertFalse(ObjectId.is_valid(''))
        self.assertFalse(ObjectId.is_valid('12345678901'))
        self.assertFalse(ObjectId.is_valid('1234567890123'))
        self.assertTrue(ObjectId.is_valid(b'123456789012'))
        self.assertTrue(ObjectId.is_valid('123456789012123456789012'))

    def test_counter_overflow(self):
        if False:
            for i in range(10):
                print('nop')
        ObjectId._inc = _MAX_COUNTER_VALUE
        ObjectId()
        self.assertEqual(ObjectId._inc, 0)

    def test_timestamp_values(self):
        if False:
            return 10
        TEST_DATA = {0: (1970, 1, 1, 0, 0, 0), 2147483647: (2038, 1, 19, 3, 14, 7), 2147483648: (2038, 1, 19, 3, 14, 8), 4294967295: (2106, 2, 7, 6, 28, 15)}

        def generate_objectid_with_timestamp(timestamp):
            if False:
                print('Hello World!')
            oid = ObjectId()
            (_, trailing_bytes) = struct.unpack('>IQ', oid.binary)
            new_oid = struct.pack('>IQ', timestamp, trailing_bytes)
            return ObjectId(new_oid)
        for (tstamp, exp_datetime_args) in TEST_DATA.items():
            oid = generate_objectid_with_timestamp(tstamp)
            if tstamp > 2147483647 and sys.maxsize < 2 ** 32:
                try:
                    oid.generation_time
                except (OverflowError, ValueError):
                    continue
            self.assertEqual(oid.generation_time, datetime.datetime(*exp_datetime_args, tzinfo=utc))

    def test_random_regenerated_on_pid_change(self):
        if False:
            for i in range(10):
                print('nop')
        random_original = ObjectId._random()
        ObjectId._pid += 1
        random_new = ObjectId._random()
        self.assertNotEqual(random_original, random_new)
if __name__ == '__main__':
    unittest.main()