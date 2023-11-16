import string
import ctypes
import decimal
import datetime
import platform
from . import testutils
import unittest
from .testutils import ConnectingTestCase, restore_types
from .testutils import skip_if_crdb
import psycopg2
from psycopg2.extensions import AsIs, adapt, register_adapter

class TypesBasicTests(ConnectingTestCase):
    """Test that all type conversions are working."""

    def execute(self, *args):
        if False:
            while True:
                i = 10
        curs = self.conn.cursor()
        curs.execute(*args)
        return curs.fetchone()[0]

    def testQuoting(self):
        if False:
            i = 10
            return i + 15
        s = "Quote'this\\! ''ok?''"
        self.failUnless(self.execute('SELECT %s AS foo', (s,)) == s, 'wrong quoting: ' + s)

    def testUnicode(self):
        if False:
            while True:
                i = 10
        s = "Quote'this\\! ''ok?''"
        self.failUnless(self.execute('SELECT %s AS foo', (s,)) == s, 'wrong unicode quoting: ' + s)

    def testNumber(self):
        if False:
            print('Hello World!')
        s = self.execute('SELECT %s AS foo', (1971,))
        self.failUnless(s == 1971, 'wrong integer quoting: ' + str(s))

    def testBoolean(self):
        if False:
            print('Hello World!')
        x = self.execute('SELECT %s as foo', (False,))
        self.assert_(x is False)
        x = self.execute('SELECT %s as foo', (True,))
        self.assert_(x is True)

    def testDecimal(self):
        if False:
            while True:
                i = 10
        s = self.execute('SELECT %s AS foo', (decimal.Decimal('19.10'),))
        self.failUnless(s - decimal.Decimal('19.10') == 0, 'wrong decimal quoting: ' + str(s))
        s = self.execute('SELECT %s AS foo', (decimal.Decimal('NaN'),))
        self.failUnless(str(s) == 'NaN', 'wrong decimal quoting: ' + str(s))
        self.failUnless(type(s) == decimal.Decimal, 'wrong decimal conversion: ' + repr(s))
        s = self.execute('SELECT %s AS foo', (decimal.Decimal('infinity'),))
        self.failUnless(str(s) == 'NaN', 'wrong decimal quoting: ' + str(s))
        self.failUnless(type(s) == decimal.Decimal, 'wrong decimal conversion: ' + repr(s))
        s = self.execute('SELECT %s AS foo', (decimal.Decimal('-infinity'),))
        self.failUnless(str(s) == 'NaN', 'wrong decimal quoting: ' + str(s))
        self.failUnless(type(s) == decimal.Decimal, 'wrong decimal conversion: ' + repr(s))

    def testFloatNan(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            float('nan')
        except ValueError:
            return self.skipTest('nan not available on this platform')
        s = self.execute('SELECT %s AS foo', (float('nan'),))
        self.failUnless(str(s) == 'nan', 'wrong float quoting: ' + str(s))
        self.failUnless(type(s) == float, 'wrong float conversion: ' + repr(s))

    def testFloatInf(self):
        if False:
            i = 10
            return i + 15
        try:
            self.execute("select 'inf'::float")
        except psycopg2.DataError:
            return self.skipTest('inf::float not available on the server')
        except ValueError:
            return self.skipTest('inf not available on this platform')
        s = self.execute('SELECT %s AS foo', (float('inf'),))
        self.failUnless(str(s) == 'inf', 'wrong float quoting: ' + str(s))
        self.failUnless(type(s) == float, 'wrong float conversion: ' + repr(s))
        s = self.execute('SELECT %s AS foo', (float('-inf'),))
        self.failUnless(str(s) == '-inf', 'wrong float quoting: ' + str(s))

    def testBinary(self):
        if False:
            print('Hello World!')
        s = bytes(range(256))
        b = psycopg2.Binary(s)
        buf = self.execute('SELECT %s::bytea AS foo', (b,))
        self.assertEqual(s, buf.tobytes())

    def testBinaryNone(self):
        if False:
            for i in range(10):
                print('nop')
        b = psycopg2.Binary(None)
        buf = self.execute('SELECT %s::bytea AS foo', (b,))
        self.assertEqual(buf, None)

    def testBinaryEmptyString(self):
        if False:
            print('Hello World!')
        b = psycopg2.Binary(bytes([]))
        self.assertEqual(str(b), "''::bytea")

    def testBinaryRoundTrip(self):
        if False:
            while True:
                i = 10
        s = bytes(range(256))
        buf = self.execute('SELECT %s::bytea AS foo', (psycopg2.Binary(s),))
        buf2 = self.execute('SELECT %s::bytea AS foo', (buf,))
        self.assertEqual(s, buf2.tobytes())

    @skip_if_crdb('nested array')
    def testArray(self):
        if False:
            for i in range(10):
                print('nop')
        s = self.execute('SELECT %s AS foo', ([[1, 2], [3, 4]],))
        self.failUnlessEqual(s, [[1, 2], [3, 4]])
        s = self.execute('SELECT %s AS foo', (['one', 'two', 'three'],))
        self.failUnlessEqual(s, ['one', 'two', 'three'])

    @skip_if_crdb('nested array')
    def testEmptyArrayRegression(self):
        if False:
            return 10
        curs = self.conn.cursor()
        curs.execute('create table array_test (id integer, col timestamp without time zone[])')
        curs.execute('insert into array_test values (%s, %s)', (1, [datetime.date(2011, 2, 14)]))
        curs.execute('select col from array_test where id = 1')
        self.assertEqual(curs.fetchone()[0], [datetime.datetime(2011, 2, 14, 0, 0)])
        curs.execute('insert into array_test values (%s, %s)', (2, []))
        curs.execute('select col from array_test where id = 2')
        self.assertEqual(curs.fetchone()[0], [])

    @skip_if_crdb('nested array')
    @testutils.skip_before_postgres(8, 4)
    def testNestedEmptyArray(self):
        if False:
            while True:
                i = 10
        curs = self.conn.cursor()
        curs.execute('select 10 = any(%s::int[])', ([[]],))
        self.assertFalse(curs.fetchone()[0])

    def testEmptyArrayNoCast(self):
        if False:
            print('Hello World!')
        s = self.execute("SELECT '{}' AS foo")
        self.assertEqual(s, '{}')
        s = self.execute('SELECT %s AS foo', ([],))
        self.assertEqual(s, '{}')

    def testEmptyArray(self):
        if False:
            for i in range(10):
                print('nop')
        s = self.execute("SELECT '{}'::text[] AS foo")
        self.failUnlessEqual(s, [])
        s = self.execute('SELECT 1 != ALL(%s)', ([],))
        self.failUnlessEqual(s, True)
        s = self.execute("SELECT '{}'::text AS foo")
        self.failUnlessEqual(s, '{}')

    def testArrayEscape(self):
        if False:
            for i in range(10):
                print('nop')
        ss = ['', '\\', '"', '\\\\', '\\"']
        for s in ss:
            r = self.execute('SELECT %s AS foo', (s,))
            self.failUnlessEqual(s, r)
            r = self.execute('SELECT %s AS foo', ([s],))
            self.failUnlessEqual([s], r)
        r = self.execute('SELECT %s AS foo', (ss,))
        self.failUnlessEqual(ss, r)

    def testArrayMalformed(self):
        if False:
            print('Hello World!')
        curs = self.conn.cursor()
        ss = ['', '{', '{}}', '{' * 20 + '}' * 20]
        for s in ss:
            self.assertRaises(psycopg2.DataError, psycopg2.extensions.STRINGARRAY, s.encode('utf8'), curs)

    def testTextArray(self):
        if False:
            print('Hello World!')
        curs = self.conn.cursor()
        curs.execute("select '{a,b,c}'::text[]")
        x = curs.fetchone()[0]
        self.assert_(isinstance(x[0], str))
        self.assertEqual(x, ['a', 'b', 'c'])

    def testUnicodeArray(self):
        if False:
            print('Hello World!')
        psycopg2.extensions.register_type(psycopg2.extensions.UNICODEARRAY, self.conn)
        curs = self.conn.cursor()
        curs.execute("select '{a,b,c}'::text[]")
        x = curs.fetchone()[0]
        self.assert_(isinstance(x[0], str))
        self.assertEqual(x, ['a', 'b', 'c'])

    def testBytesArray(self):
        if False:
            i = 10
            return i + 15
        psycopg2.extensions.register_type(psycopg2.extensions.BYTESARRAY, self.conn)
        curs = self.conn.cursor()
        curs.execute("select '{a,b,c}'::text[]")
        x = curs.fetchone()[0]
        self.assert_(isinstance(x[0], bytes))
        self.assertEqual(x, [b'a', b'b', b'c'])

    @skip_if_crdb('nested array')
    @testutils.skip_before_postgres(8, 2)
    def testArrayOfNulls(self):
        if False:
            while True:
                i = 10
        curs = self.conn.cursor()
        curs.execute('\n            create table na (\n              texta text[],\n              inta int[],\n              boola boolean[],\n\n              textaa text[][],\n              intaa int[][],\n              boolaa boolean[][]\n            )')
        curs.execute('insert into na (texta) values (%s)', ([None],))
        curs.execute('insert into na (texta) values (%s)', (['a', None],))
        curs.execute('insert into na (texta) values (%s)', ([None, None],))
        curs.execute('insert into na (inta) values (%s)', ([None],))
        curs.execute('insert into na (inta) values (%s)', ([42, None],))
        curs.execute('insert into na (inta) values (%s)', ([None, None],))
        curs.execute('insert into na (boola) values (%s)', ([None],))
        curs.execute('insert into na (boola) values (%s)', ([True, None],))
        curs.execute('insert into na (boola) values (%s)', ([None, None],))
        curs.execute('insert into na (textaa) values (%s)', ([[None]],))
        curs.execute('insert into na (textaa) values (%s)', ([['a', None]],))
        curs.execute('insert into na (textaa) values (%s)', ([[None, None]],))
        curs.execute('insert into na (intaa) values (%s)', ([[None]],))
        curs.execute('insert into na (intaa) values (%s)', ([[42, None]],))
        curs.execute('insert into na (intaa) values (%s)', ([[None, None]],))
        curs.execute('insert into na (boolaa) values (%s)', ([[None]],))
        curs.execute('insert into na (boolaa) values (%s)', ([[True, None]],))
        curs.execute('insert into na (boolaa) values (%s)', ([[None, None]],))

    @skip_if_crdb('nested array')
    @testutils.skip_before_postgres(8, 2)
    def testNestedArrays(self):
        if False:
            i = 10
            return i + 15
        curs = self.conn.cursor()
        for a in [[[1]], [[None]], [[None, None, None]], [[None, None], [1, None]], [[None, None], [None, None]], [[[None, None], [None, None]]]]:
            curs.execute('select %s::int[]', (a,))
            self.assertEqual(curs.fetchone()[0], a)
            curs.execute('select array[%s::int[]]', (a,))
            self.assertEqual(curs.fetchone()[0], [a])

    def testTypeRoundtripBytes(self):
        if False:
            while True:
                i = 10
        o1 = bytes(range(256))
        o2 = self.execute('select %s;', (o1,))
        self.assertEqual(memoryview, type(o2))
        o1 = bytes([])
        o2 = self.execute('select %s;', (o1,))
        self.assertEqual(memoryview, type(o2))

    def testTypeRoundtripBytesArray(self):
        if False:
            i = 10
            return i + 15
        o1 = bytes(range(256))
        o1 = [o1]
        o2 = self.execute('select %s;', (o1,))
        self.assertEqual(memoryview, type(o2[0]))

    def testAdaptBytearray(self):
        if False:
            print('Hello World!')
        o1 = bytearray(range(256))
        o2 = self.execute('select %s;', (o1,))
        self.assertEqual(memoryview, type(o2))
        self.assertEqual(len(o1), len(o2))
        for (c1, c2) in zip(o1, o2):
            self.assertEqual(c1, ord(c2))
        o1 = bytearray([])
        o2 = self.execute('select %s;', (o1,))
        self.assertEqual(len(o2), 0)
        self.assertEqual(memoryview, type(o2))

    def testAdaptMemoryview(self):
        if False:
            while True:
                i = 10
        o1 = memoryview(bytearray(range(256)))
        o2 = self.execute('select %s;', (o1,))
        self.assertEqual(memoryview, type(o2))
        o1 = memoryview(bytearray([]))
        o2 = self.execute('select %s;', (o1,))
        self.assertEqual(memoryview, type(o2))

    def testByteaHexCheckFalsePositive(self):
        if False:
            while True:
                i = 10
        o1 = psycopg2.Binary(b'x')
        o2 = self.execute('SELECT %s::bytea AS foo', (o1,))
        self.assertEqual(b'x', o2[0])

    def testNegNumber(self):
        if False:
            return 10
        d1 = self.execute('select -%s;', (decimal.Decimal('-1.0'),))
        self.assertEqual(1, d1)
        f1 = self.execute('select -%s;', (-1.0,))
        self.assertEqual(1, f1)
        i1 = self.execute('select -%s;', (-1,))
        self.assertEqual(1, i1)

    def testGenericArray(self):
        if False:
            while True:
                i = 10
        a = self.execute("select '{1, 2, 3}'::int4[]")
        self.assertEqual(a, [1, 2, 3])
        a = self.execute("select array['a', 'b', '''']::text[]")
        self.assertEqual(a, ['a', 'b', "'"])

    @testutils.skip_before_postgres(8, 2)
    def testGenericArrayNull(self):
        if False:
            return 10

        def caster(s, cur):
            if False:
                return 10
            if s is None:
                return 'nada'
            return int(s) * 2
        base = psycopg2.extensions.new_type((23,), 'INT4', caster)
        array = psycopg2.extensions.new_array_type((1007,), 'INT4ARRAY', base)
        psycopg2.extensions.register_type(array, self.conn)
        a = self.execute("select '{1, 2, 3}'::int4[]")
        self.assertEqual(a, [2, 4, 6])
        a = self.execute("select '{1, 2, NULL}'::int4[]")
        self.assertEqual(a, [2, 4, 'nada'])

    @skip_if_crdb('cidr')
    @testutils.skip_before_postgres(8, 2)
    def testNetworkArray(self):
        if False:
            while True:
                i = 10
        a = self.execute("select '{192.168.0.1/24}'::inet[]")
        self.assertEqual(a, ['192.168.0.1/24'])
        a = self.execute("select '{192.168.0.0/24}'::cidr[]")
        self.assertEqual(a, ['192.168.0.0/24'])
        a = self.execute("select '{10:20:30:40:50:60}'::macaddr[]")
        self.assertEqual(a, ['10:20:30:40:50:60'])

    def testIntEnum(self):
        if False:
            return 10
        from enum import IntEnum

        class Color(IntEnum):
            RED = 1
            GREEN = 2
            BLUE = 4
        a = self.execute('select %s', (Color.GREEN,))
        self.assertEqual(a, Color.GREEN)

class AdaptSubclassTest(unittest.TestCase):

    def test_adapt_subtype(self):
        if False:
            print('Hello World!')

        class Sub(str):
            pass
        s1 = "hel'lo"
        s2 = Sub(s1)
        self.assertEqual(adapt(s1).getquoted(), adapt(s2).getquoted())

    @restore_types
    def test_adapt_most_specific(self):
        if False:
            for i in range(10):
                print('nop')

        class A:
            pass

        class B(A):
            pass

        class C(B):
            pass
        register_adapter(A, lambda a: AsIs('a'))
        register_adapter(B, lambda b: AsIs('b'))
        self.assertEqual(b'b', adapt(C()).getquoted())

    @restore_types
    def test_adapt_subtype_3(self):
        if False:
            while True:
                i = 10

        class A:
            pass

        class B(A):
            pass
        register_adapter(A, lambda a: AsIs('a'))
        self.assertEqual(b'a', adapt(B()).getquoted())

    def test_conform_subclass_precedence(self):
        if False:
            return 10

        class foo(tuple):

            def __conform__(self, proto):
                if False:
                    while True:
                        i = 10
                return self

            def getquoted(self):
                if False:
                    print('Hello World!')
                return 'bar'
        self.assertEqual(adapt(foo((1, 2, 3))).getquoted(), 'bar')

@unittest.skipIf(platform.system() == 'Windows', 'Not testing because we are useless with ctypes on Windows')
class ByteaParserTest(unittest.TestCase):
    """Unit test for our bytea format parser."""

    def setUp(self):
        if False:
            while True:
                i = 10
        self._cast = self._import_cast()

    def _import_cast(self):
        if False:
            return 10
        'Use ctypes to access the C function.'
        lib = ctypes.pydll.LoadLibrary(psycopg2._psycopg.__file__)
        cast = lib.typecast_BINARY_cast
        cast.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.py_object]
        cast.restype = ctypes.py_object
        return cast

    def cast(self, buffer):
        if False:
            i = 10
            return i + 15
        'Cast a buffer from the output format'
        l = buffer and len(buffer) or 0
        rv = self._cast(buffer, l, None)
        if rv is None:
            return None
        return rv.tobytes()

    def test_null(self):
        if False:
            while True:
                i = 10
        rv = self.cast(None)
        self.assertEqual(rv, None)

    def test_blank(self):
        if False:
            for i in range(10):
                print('nop')
        rv = self.cast(b'')
        self.assertEqual(rv, b'')

    def test_blank_hex(self):
        if False:
            print('Hello World!')
        rv = self.cast(b'\\x')
        self.assertEqual(rv, b'')

    def test_full_hex(self, upper=False):
        if False:
            print('Hello World!')
        buf = ''.join(('%02x' % i for i in range(256)))
        if upper:
            buf = buf.upper()
        buf = '\\x' + buf
        rv = self.cast(buf.encode('utf8'))
        self.assertEqual(rv, bytes(range(256)))

    def test_full_hex_upper(self):
        if False:
            i = 10
            return i + 15
        return self.test_full_hex(upper=True)

    def test_full_escaped_octal(self):
        if False:
            i = 10
            return i + 15
        buf = ''.join(('\\%03o' % i for i in range(256)))
        rv = self.cast(buf.encode('utf8'))
        self.assertEqual(rv, bytes(range(256)))

    def test_escaped_mixed(self):
        if False:
            return 10
        buf = ''.join(('\\%03o' % i for i in range(32)))
        buf += string.ascii_letters
        buf += ''.join(('\\' + c for c in string.ascii_letters))
        buf += '\\\\'
        rv = self.cast(buf.encode('utf8'))
        tgt = bytes(range(32)) + (string.ascii_letters * 2 + '\\').encode('ascii')
        self.assertEqual(rv, tgt)

def test_suite():
    if False:
        for i in range(10):
            print('nop')
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main()