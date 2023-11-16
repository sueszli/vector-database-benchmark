from test import support
from test.support import os_helper
import array
import io
import marshal
import sys
import unittest
import os
import types
try:
    import _testcapi
except ImportError:
    _testcapi = None

class HelperMixin:

    def helper(self, sample, *extra):
        if False:
            i = 10
            return i + 15
        new = marshal.loads(marshal.dumps(sample, *extra))
        self.assertEqual(sample, new)
        try:
            with open(os_helper.TESTFN, 'wb') as f:
                marshal.dump(sample, f, *extra)
            with open(os_helper.TESTFN, 'rb') as f:
                new = marshal.load(f)
            self.assertEqual(sample, new)
        finally:
            os_helper.unlink(os_helper.TESTFN)

class IntTestCase(unittest.TestCase, HelperMixin):

    def test_ints(self):
        if False:
            return 10
        n = sys.maxsize ** 2
        while n:
            for expected in (-n, n):
                self.helper(expected)
            n = n >> 1

    def test_int64(self):
        if False:
            i = 10
            return i + 15
        maxint64 = (1 << 63) - 1
        minint64 = -maxint64 - 1
        for base in (maxint64, minint64, -maxint64, -(minint64 >> 1)):
            while base:
                s = b'I' + int.to_bytes(base, 8, 'little', signed=True)
                got = marshal.loads(s)
                self.assertEqual(base, got)
                if base == -1:
                    base = 0
                else:
                    base >>= 1
        got = marshal.loads(b'I\xfe\xdc\xba\x98vT2\x10')
        self.assertEqual(got, 1167088121787636990)
        got = marshal.loads(b'I\x01#Eg\x89\xab\xcd\xef')
        self.assertEqual(got, -1167088121787636991)
        got = marshal.loads(b'I\x08\x19*;L]n\x7f')
        self.assertEqual(got, 9182379272246532360)
        got = marshal.loads(b'I\xf7\xe6\xd5\xc4\xb3\xa2\x91\x80')
        self.assertEqual(got, -9182379272246532361)

    def test_bool(self):
        if False:
            return 10
        for b in (True, False):
            self.helper(b)

class FloatTestCase(unittest.TestCase, HelperMixin):

    def test_floats(self):
        if False:
            return 10
        small = 1e-25
        n = sys.maxsize * 3.7e+250
        while n > small:
            for expected in (-n, n):
                self.helper(float(expected))
            n /= 123.4567
        f = 0.0
        s = marshal.dumps(f, 2)
        got = marshal.loads(s)
        self.assertEqual(f, got)
        s = marshal.dumps(f, 1)
        got = marshal.loads(s)
        self.assertEqual(f, got)
        n = sys.maxsize * 3.7e-250
        while n < small:
            for expected in (-n, n):
                f = float(expected)
                self.helper(f)
                self.helper(f, 1)
            n *= 123.4567

class StringTestCase(unittest.TestCase, HelperMixin):

    def test_unicode(self):
        if False:
            for i in range(10):
                print('nop')
        for s in ['', 'Andrè Previn', 'abc', ' ' * 10000]:
            self.helper(marshal.loads(marshal.dumps(s)))

    def test_string(self):
        if False:
            return 10
        for s in ['', 'Andrè Previn', 'abc', ' ' * 10000]:
            self.helper(s)

    def test_bytes(self):
        if False:
            i = 10
            return i + 15
        for s in [b'', b'Andr\xe8 Previn', b'abc', b' ' * 10000]:
            self.helper(s)

class ExceptionTestCase(unittest.TestCase):

    def test_exceptions(self):
        if False:
            i = 10
            return i + 15
        new = marshal.loads(marshal.dumps(StopIteration))
        self.assertEqual(StopIteration, new)

class CodeTestCase(unittest.TestCase):

    def test_code(self):
        if False:
            return 10
        co = ExceptionTestCase.test_exceptions.__code__
        new = marshal.loads(marshal.dumps(co))
        self.assertEqual(co, new)

    def test_many_codeobjects(self):
        if False:
            for i in range(10):
                print('nop')
        count = 5000
        codes = (ExceptionTestCase.test_exceptions.__code__,) * count
        marshal.loads(marshal.dumps(codes))

    def test_different_filenames(self):
        if False:
            while True:
                i = 10
        co1 = compile('x', 'f1', 'exec')
        co2 = compile('y', 'f2', 'exec')
        (co1, co2) = marshal.loads(marshal.dumps((co1, co2)))
        self.assertEqual(co1.co_filename, 'f1')
        self.assertEqual(co2.co_filename, 'f2')

    @support.cpython_only
    def test_same_filename_used(self):
        if False:
            i = 10
            return i + 15
        s = 'def f(): pass\ndef g(): pass'
        co = compile(s, 'myfile', 'exec')
        co = marshal.loads(marshal.dumps(co))
        for obj in co.co_consts:
            if isinstance(obj, types.CodeType):
                self.assertIs(co.co_filename, obj.co_filename)

class ContainerTestCase(unittest.TestCase, HelperMixin):
    d = {'astring': 'foo@bar.baz.spam', 'afloat': 7283.43, 'anint': 2 ** 20, 'ashortlong': 2, 'alist': ['.zyx.41'], 'atuple': ('.zyx.41',) * 10, 'aboolean': False, 'aunicode': 'Andrè Previn'}

    def test_dict(self):
        if False:
            i = 10
            return i + 15
        self.helper(self.d)

    def test_list(self):
        if False:
            while True:
                i = 10
        self.helper(list(self.d.items()))

    def test_tuple(self):
        if False:
            return 10
        self.helper(tuple(self.d.keys()))

    def test_sets(self):
        if False:
            i = 10
            return i + 15
        for constructor in (set, frozenset):
            self.helper(constructor(self.d.keys()))

class BufferTestCase(unittest.TestCase, HelperMixin):

    def test_bytearray(self):
        if False:
            while True:
                i = 10
        b = bytearray(b'abc')
        self.helper(b)
        new = marshal.loads(marshal.dumps(b))
        self.assertEqual(type(new), bytes)

    def test_memoryview(self):
        if False:
            for i in range(10):
                print('nop')
        b = memoryview(b'abc')
        self.helper(b)
        new = marshal.loads(marshal.dumps(b))
        self.assertEqual(type(new), bytes)

    def test_array(self):
        if False:
            print('Hello World!')
        a = array.array('B', b'abc')
        new = marshal.loads(marshal.dumps(a))
        self.assertEqual(new, b'abc')

class BugsTestCase(unittest.TestCase):

    def test_bug_5888452(self):
        if False:
            i = 10
            return i + 15
        marshal.dumps([128] * 1000)

    def test_patch_873224(self):
        if False:
            print('Hello World!')
        self.assertRaises(Exception, marshal.loads, b'0')
        self.assertRaises(Exception, marshal.loads, b'f')
        self.assertRaises(Exception, marshal.loads, marshal.dumps(2 ** 65)[:-1])

    def test_version_argument(self):
        if False:
            print('Hello World!')
        self.assertEqual(marshal.loads(marshal.dumps(5, 0)), 5)
        self.assertEqual(marshal.loads(marshal.dumps(5, 1)), 5)

    def test_fuzz(self):
        if False:
            return 10
        for i in range(256):
            c = bytes([i])
            try:
                marshal.loads(c)
            except Exception:
                pass

    def test_loads_recursion(self):
        if False:
            for i in range(10):
                print('nop')

        def run_tests(N, check):
            if False:
                for i in range(10):
                    print('nop')
            check(b')\x01' * N + b'N')
            check(b'(\x01\x00\x00\x00' * N + b'N')
            check(b'[\x01\x00\x00\x00' * N + b'N')
            check(b'{N' * N + b'N' + b'0' * N)
            check(b'>\x01\x00\x00\x00' * N + b'N')
        run_tests(100, marshal.loads)

        def check(s):
            if False:
                return 10
            self.assertRaises(ValueError, marshal.loads, s)
        run_tests(2 ** 20, check)

    def test_recursion_limit(self):
        if False:
            for i in range(10):
                print('nop')
        head = last = []
        if os.name == 'nt':
            MAX_MARSHAL_STACK_DEPTH = 1000
        else:
            MAX_MARSHAL_STACK_DEPTH = 2000
        for i in range(MAX_MARSHAL_STACK_DEPTH - 2):
            last.append([0])
            last = last[-1]
        data = marshal.dumps(head)
        new_head = marshal.loads(data)
        self.assertEqual(len(new_head), len(head))
        self.assertEqual(len(new_head[0]), len(head[0]))
        self.assertEqual(len(new_head[-1]), len(head[-1]))
        last.append([0])
        self.assertRaises(ValueError, marshal.dumps, head)

    def test_exact_type_match(self):
        if False:
            while True:
                i = 10
        for typ in (int, float, complex, tuple, list, dict, set, frozenset):
            subtyp = type('subtyp', (typ,), {})
            self.assertRaises(ValueError, marshal.dumps, subtyp())

    def test_large_marshal(self):
        if False:
            print('Hello World!')
        size = int(1000000.0)
        testString = 'abc' * size
        marshal.dumps(testString)

    def test_invalid_longs(self):
        if False:
            i = 10
            return i + 15
        invalid_string = b'l\x02\x00\x00\x00\x00\x00\x00\x00'
        self.assertRaises(ValueError, marshal.loads, invalid_string)

    def test_multiple_dumps_and_loads(self):
        if False:
            i = 10
            return i + 15
        data = (1, 'abc', b'def', 1.0, (2, 'a', ['b', b'c']))
        for interleaved in (b'', b'0123'):
            ilen = len(interleaved)
            positions = []
            try:
                with open(os_helper.TESTFN, 'wb') as f:
                    for d in data:
                        marshal.dump(d, f)
                        if ilen:
                            f.write(interleaved)
                        positions.append(f.tell())
                with open(os_helper.TESTFN, 'rb') as f:
                    for (i, d) in enumerate(data):
                        self.assertEqual(d, marshal.load(f))
                        if ilen:
                            f.read(ilen)
                        self.assertEqual(positions[i], f.tell())
            finally:
                os_helper.unlink(os_helper.TESTFN)

    def test_loads_reject_unicode_strings(self):
        if False:
            print('Hello World!')
        unicode_string = 'T'
        self.assertRaises(TypeError, marshal.loads, unicode_string)

    def test_bad_reader(self):
        if False:
            return 10

        class BadReader(io.BytesIO):

            def readinto(self, buf):
                if False:
                    print('Hello World!')
                n = super().readinto(buf)
                if n is not None and n > 4:
                    n += 10 ** 6
                return n
        for value in (1.0, 1j, b'0123456789', '0123456789'):
            self.assertRaises(ValueError, marshal.load, BadReader(marshal.dumps(value)))

    def test_eof(self):
        if False:
            while True:
                i = 10
        data = marshal.dumps(('hello', 'dolly', None))
        for i in range(len(data)):
            self.assertRaises(EOFError, marshal.loads, data[0:i])
LARGE_SIZE = 2 ** 31
pointer_size = 8 if sys.maxsize > 4294967295 else 4

class NullWriter:

    def write(self, s):
        if False:
            return 10
        pass

@unittest.skipIf(LARGE_SIZE > sys.maxsize, 'test cannot run on 32-bit systems')
class LargeValuesTestCase(unittest.TestCase):

    def check_unmarshallable(self, data):
        if False:
            while True:
                i = 10
        self.assertRaises(ValueError, marshal.dump, data, NullWriter())

    @support.bigmemtest(size=LARGE_SIZE, memuse=2, dry_run=False)
    def test_bytes(self, size):
        if False:
            i = 10
            return i + 15
        self.check_unmarshallable(b'x' * size)

    @support.bigmemtest(size=LARGE_SIZE, memuse=2, dry_run=False)
    def test_str(self, size):
        if False:
            while True:
                i = 10
        self.check_unmarshallable('x' * size)

    @support.bigmemtest(size=LARGE_SIZE, memuse=pointer_size + 1, dry_run=False)
    def test_tuple(self, size):
        if False:
            return 10
        self.check_unmarshallable((None,) * size)

    @support.bigmemtest(size=LARGE_SIZE, memuse=pointer_size + 1, dry_run=False)
    def test_list(self, size):
        if False:
            i = 10
            return i + 15
        self.check_unmarshallable([None] * size)

    @support.bigmemtest(size=LARGE_SIZE, memuse=pointer_size * 12 + sys.getsizeof(LARGE_SIZE - 1), dry_run=False)
    def test_set(self, size):
        if False:
            while True:
                i = 10
        self.check_unmarshallable(set(range(size)))

    @support.bigmemtest(size=LARGE_SIZE, memuse=pointer_size * 12 + sys.getsizeof(LARGE_SIZE - 1), dry_run=False)
    def test_frozenset(self, size):
        if False:
            print('Hello World!')
        self.check_unmarshallable(frozenset(range(size)))

    @support.bigmemtest(size=LARGE_SIZE, memuse=2, dry_run=False)
    def test_bytearray(self, size):
        if False:
            return 10
        self.check_unmarshallable(bytearray(size))

def CollectObjectIDs(ids, obj):
    if False:
        return 10
    'Collect object ids seen in a structure'
    if id(obj) in ids:
        return
    ids.add(id(obj))
    if isinstance(obj, (list, tuple, set, frozenset)):
        for e in obj:
            CollectObjectIDs(ids, e)
    elif isinstance(obj, dict):
        for (k, v) in obj.items():
            CollectObjectIDs(ids, k)
            CollectObjectIDs(ids, v)
    return len(ids)

class InstancingTestCase(unittest.TestCase, HelperMixin):
    keys = (123, 1.2345, 'abc', (123, 'abc'), frozenset({123, 'abc'}))

    def helper3(self, rsample, recursive=False, simple=False):
        if False:
            i = 10
            return i + 15
        sample = (rsample, rsample)
        n0 = CollectObjectIDs(set(), sample)
        for v in range(3, marshal.version + 1):
            s3 = marshal.dumps(sample, v)
            n3 = CollectObjectIDs(set(), marshal.loads(s3))
            self.assertEqual(n3, n0)
        if not recursive:
            s2 = marshal.dumps(sample, 2)
            n2 = CollectObjectIDs(set(), marshal.loads(s2))
            self.assertGreater(n2, n0)
            if not simple:
                self.assertGreater(len(s2), len(s3))
            else:
                self.assertGreaterEqual(len(s2), len(s3))

    def testInt(self):
        if False:
            return 10
        intobj = 123321
        self.helper(intobj)
        self.helper3(intobj, simple=True)

    def testFloat(self):
        if False:
            for i in range(10):
                print('nop')
        floatobj = 1.2345
        self.helper(floatobj)
        self.helper3(floatobj)

    def testStr(self):
        if False:
            while True:
                i = 10
        strobj = 'abcde' * 3
        self.helper(strobj)
        self.helper3(strobj)

    def testBytes(self):
        if False:
            return 10
        bytesobj = b'abcde' * 3
        self.helper(bytesobj)
        self.helper3(bytesobj)

    def testList(self):
        if False:
            i = 10
            return i + 15
        for obj in self.keys:
            listobj = [obj, obj]
            self.helper(listobj)
            self.helper3(listobj)

    def testTuple(self):
        if False:
            return 10
        for obj in self.keys:
            tupleobj = (obj, obj)
            self.helper(tupleobj)
            self.helper3(tupleobj)

    def testSet(self):
        if False:
            for i in range(10):
                print('nop')
        for obj in self.keys:
            setobj = {(obj, 1), (obj, 2)}
            self.helper(setobj)
            self.helper3(setobj)

    def testFrozenSet(self):
        if False:
            while True:
                i = 10
        for obj in self.keys:
            frozensetobj = frozenset({(obj, 1), (obj, 2)})
            self.helper(frozensetobj)
            self.helper3(frozensetobj)

    def testDict(self):
        if False:
            print('Hello World!')
        for obj in self.keys:
            dictobj = {'hello': obj, 'goodbye': obj, obj: 'hello'}
            self.helper(dictobj)
            self.helper3(dictobj)

    def testModule(self):
        if False:
            return 10
        with open(__file__, 'rb') as f:
            code = f.read()
        if __file__.endswith('.py'):
            code = compile(code, __file__, 'exec')
        self.helper(code)
        self.helper3(code)

    def testRecursion(self):
        if False:
            print('Hello World!')
        obj = 1.2345
        d = {'hello': obj, 'goodbye': obj, obj: 'hello'}
        d['self'] = d
        self.helper3(d, recursive=True)
        l = [obj, obj]
        l.append(l)
        self.helper3(l, recursive=True)

class CompatibilityTestCase(unittest.TestCase):

    def _test(self, version):
        if False:
            i = 10
            return i + 15
        with open(__file__, 'rb') as f:
            code = f.read()
        if __file__.endswith('.py'):
            code = compile(code, __file__, 'exec')
        data = marshal.dumps(code, version)
        marshal.loads(data)

    def test0To3(self):
        if False:
            while True:
                i = 10
        self._test(0)

    def test1To3(self):
        if False:
            return 10
        self._test(1)

    def test2To3(self):
        if False:
            for i in range(10):
                print('nop')
        self._test(2)

    def test3To3(self):
        if False:
            i = 10
            return i + 15
        self._test(3)

class InterningTestCase(unittest.TestCase, HelperMixin):
    strobj = 'this is an interned string'
    strobj = sys.intern(strobj)

    def testIntern(self):
        if False:
            for i in range(10):
                print('nop')
        s = marshal.loads(marshal.dumps(self.strobj))
        self.assertEqual(s, self.strobj)
        self.assertEqual(id(s), id(self.strobj))
        s2 = sys.intern(s)
        self.assertEqual(id(s2), id(s))

    def testNoIntern(self):
        if False:
            while True:
                i = 10
        s = marshal.loads(marshal.dumps(self.strobj, 2))
        self.assertEqual(s, self.strobj)
        self.assertNotEqual(id(s), id(self.strobj))
        s2 = sys.intern(s)
        self.assertNotEqual(id(s2), id(s))

@support.cpython_only
@unittest.skipUnless(_testcapi, 'requires _testcapi')
class CAPI_TestCase(unittest.TestCase, HelperMixin):

    def test_write_long_to_file(self):
        if False:
            i = 10
            return i + 15
        for v in range(marshal.version + 1):
            _testcapi.pymarshal_write_long_to_file(305419896, os_helper.TESTFN, v)
            with open(os_helper.TESTFN, 'rb') as f:
                data = f.read()
            os_helper.unlink(os_helper.TESTFN)
            self.assertEqual(data, b'xV4\x12')

    def test_write_object_to_file(self):
        if False:
            while True:
                i = 10
        obj = ('€', b'abc', 123, 45.6, 7 + 8j, 'long line ' * 1000)
        for v in range(marshal.version + 1):
            _testcapi.pymarshal_write_object_to_file(obj, os_helper.TESTFN, v)
            with open(os_helper.TESTFN, 'rb') as f:
                data = f.read()
            os_helper.unlink(os_helper.TESTFN)
            self.assertEqual(marshal.loads(data), obj)

    def test_read_short_from_file(self):
        if False:
            print('Hello World!')
        with open(os_helper.TESTFN, 'wb') as f:
            f.write(b'4\x12xxxx')
        (r, p) = _testcapi.pymarshal_read_short_from_file(os_helper.TESTFN)
        os_helper.unlink(os_helper.TESTFN)
        self.assertEqual(r, 4660)
        self.assertEqual(p, 2)
        with open(os_helper.TESTFN, 'wb') as f:
            f.write(b'\x12')
        with self.assertRaises(EOFError):
            _testcapi.pymarshal_read_short_from_file(os_helper.TESTFN)
        os_helper.unlink(os_helper.TESTFN)

    def test_read_long_from_file(self):
        if False:
            while True:
                i = 10
        with open(os_helper.TESTFN, 'wb') as f:
            f.write(b'xV4\x12xxxx')
        (r, p) = _testcapi.pymarshal_read_long_from_file(os_helper.TESTFN)
        os_helper.unlink(os_helper.TESTFN)
        self.assertEqual(r, 305419896)
        self.assertEqual(p, 4)
        with open(os_helper.TESTFN, 'wb') as f:
            f.write(b'V4\x12')
        with self.assertRaises(EOFError):
            _testcapi.pymarshal_read_long_from_file(os_helper.TESTFN)
        os_helper.unlink(os_helper.TESTFN)

    def test_read_last_object_from_file(self):
        if False:
            return 10
        obj = ('€', b'abc', 123, 45.6, 7 + 8j)
        for v in range(marshal.version + 1):
            data = marshal.dumps(obj, v)
            with open(os_helper.TESTFN, 'wb') as f:
                f.write(data + b'xxxx')
            (r, p) = _testcapi.pymarshal_read_last_object_from_file(os_helper.TESTFN)
            os_helper.unlink(os_helper.TESTFN)
            self.assertEqual(r, obj)
            with open(os_helper.TESTFN, 'wb') as f:
                f.write(data[:1])
            with self.assertRaises(EOFError):
                _testcapi.pymarshal_read_last_object_from_file(os_helper.TESTFN)
            os_helper.unlink(os_helper.TESTFN)

    def test_read_object_from_file(self):
        if False:
            while True:
                i = 10
        obj = ('€', b'abc', 123, 45.6, 7 + 8j)
        for v in range(marshal.version + 1):
            data = marshal.dumps(obj, v)
            with open(os_helper.TESTFN, 'wb') as f:
                f.write(data + b'xxxx')
            (r, p) = _testcapi.pymarshal_read_object_from_file(os_helper.TESTFN)
            os_helper.unlink(os_helper.TESTFN)
            self.assertEqual(r, obj)
            self.assertEqual(p, len(data))
            with open(os_helper.TESTFN, 'wb') as f:
                f.write(data[:1])
            with self.assertRaises(EOFError):
                _testcapi.pymarshal_read_object_from_file(os_helper.TESTFN)
            os_helper.unlink(os_helper.TESTFN)
if __name__ == '__main__':
    unittest.main()