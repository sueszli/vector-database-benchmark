import os
import shutil
import tempfile
from functools import wraps
import psycopg2
import psycopg2.extensions
import unittest
from .testutils import decorate_all_tests, skip_if_tpc_disabled, skip_before_postgres, ConnectingTestCase, skip_if_green, skip_if_crdb, slow

def skip_if_no_lo(f):
    if False:
        i = 10
        return i + 15
    f = skip_before_postgres(8, 1, 'large objects only supported from PG 8.1')(f)
    f = skip_if_green("libpq doesn't support LO in async mode")(f)
    f = skip_if_crdb('large objects')(f)
    return f

class LargeObjectTestCase(ConnectingTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        ConnectingTestCase.setUp(self)
        self.lo_oid = None
        self.tmpdir = None

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        if self.tmpdir:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
        if self.conn.closed:
            return
        if self.lo_oid is not None:
            self.conn.rollback()
            try:
                lo = self.conn.lobject(self.lo_oid, 'n')
            except psycopg2.OperationalError:
                pass
            else:
                lo.unlink()
        ConnectingTestCase.tearDown(self)

@skip_if_no_lo
class LargeObjectTests(LargeObjectTestCase):

    def test_create(self):
        if False:
            for i in range(10):
                print('nop')
        lo = self.conn.lobject()
        self.assertNotEqual(lo, None)
        self.assertEqual(lo.mode[0], 'w')

    def test_connection_needed(self):
        if False:
            return 10
        self.assertRaises(TypeError, psycopg2.extensions.lobject, [])

    def test_open_non_existent(self):
        if False:
            return 10
        lo = self.conn.lobject()
        lo.unlink()
        self.assertRaises(psycopg2.OperationalError, self.conn.lobject, lo.oid)

    def test_open_existing(self):
        if False:
            i = 10
            return i + 15
        lo = self.conn.lobject()
        lo2 = self.conn.lobject(lo.oid)
        self.assertNotEqual(lo2, None)
        self.assertEqual(lo2.oid, lo.oid)
        self.assertEqual(lo2.mode[0], 'r')

    def test_open_for_write(self):
        if False:
            print('Hello World!')
        lo = self.conn.lobject()
        lo2 = self.conn.lobject(lo.oid, 'w')
        self.assertEqual(lo2.mode[0], 'w')
        lo2.write(b'some data')

    def test_open_mode_n(self):
        if False:
            print('Hello World!')
        lo = self.conn.lobject()
        lo.close()
        lo2 = self.conn.lobject(lo.oid, 'n')
        self.assertEqual(lo2.oid, lo.oid)
        self.assertEqual(lo2.closed, True)

    def test_mode_defaults(self):
        if False:
            i = 10
            return i + 15
        lo = self.conn.lobject()
        lo2 = self.conn.lobject(mode=None)
        lo3 = self.conn.lobject(mode='')
        self.assertEqual(lo.mode, lo2.mode)
        self.assertEqual(lo.mode, lo3.mode)

    def test_close_connection_gone(self):
        if False:
            for i in range(10):
                print('nop')
        lo = self.conn.lobject()
        self.conn.close()
        lo.close()

    def test_create_with_oid(self):
        if False:
            i = 10
            return i + 15
        lo = self.conn.lobject()
        oid = lo.oid
        lo.unlink()
        lo = self.conn.lobject(0, 'w', oid)
        self.assertEqual(lo.oid, oid)

    def test_create_with_existing_oid(self):
        if False:
            i = 10
            return i + 15
        lo = self.conn.lobject()
        lo.close()
        self.assertRaises(psycopg2.OperationalError, self.conn.lobject, 0, 'w', lo.oid)
        self.assert_(not self.conn.closed)

    def test_import(self):
        if False:
            while True:
                i = 10
        self.tmpdir = tempfile.mkdtemp()
        filename = os.path.join(self.tmpdir, 'data.txt')
        fp = open(filename, 'wb')
        fp.write(b'some data')
        fp.close()
        lo = self.conn.lobject(0, 'r', 0, filename)
        self.assertEqual(lo.read(), 'some data')

    def test_close(self):
        if False:
            for i in range(10):
                print('nop')
        lo = self.conn.lobject()
        self.assertEqual(lo.closed, False)
        lo.close()
        self.assertEqual(lo.closed, True)

    def test_write(self):
        if False:
            return 10
        lo = self.conn.lobject()
        self.assertEqual(lo.write(b'some data'), len('some data'))

    def test_write_large(self):
        if False:
            i = 10
            return i + 15
        lo = self.conn.lobject()
        data = 'data' * 1000000
        self.assertEqual(lo.write(data), len(data))

    def test_read(self):
        if False:
            while True:
                i = 10
        lo = self.conn.lobject()
        lo.write(b'some data')
        lo.close()
        lo = self.conn.lobject(lo.oid)
        x = lo.read(4)
        self.assertEqual(type(x), type(''))
        self.assertEqual(x, 'some')
        self.assertEqual(lo.read(), ' data')

    def test_read_binary(self):
        if False:
            return 10
        lo = self.conn.lobject()
        lo.write(b'some data')
        lo.close()
        lo = self.conn.lobject(lo.oid, 'rb')
        x = lo.read(4)
        self.assertEqual(type(x), type(b''))
        self.assertEqual(x, b'some')
        self.assertEqual(lo.read(), b' data')

    def test_read_text(self):
        if False:
            for i in range(10):
                print('nop')
        lo = self.conn.lobject()
        snowman = '☃'
        lo.write('some data ' + snowman)
        lo.close()
        lo = self.conn.lobject(lo.oid, 'rt')
        x = lo.read(4)
        self.assertEqual(type(x), type(''))
        self.assertEqual(x, 'some')
        self.assertEqual(lo.read(), ' data ' + snowman)

    @slow
    def test_read_large(self):
        if False:
            for i in range(10):
                print('nop')
        lo = self.conn.lobject()
        data = 'data' * 1000000
        lo.write('some' + data)
        lo.close()
        lo = self.conn.lobject(lo.oid)
        self.assertEqual(lo.read(4), 'some')
        data1 = lo.read()
        self.assert_(data == data1, f'{data[:100]!r}... != {data1[:100]!r}...')

    def test_seek_tell(self):
        if False:
            return 10
        lo = self.conn.lobject()
        length = lo.write(b'some data')
        self.assertEqual(lo.tell(), length)
        lo.close()
        lo = self.conn.lobject(lo.oid)
        self.assertEqual(lo.seek(5, 0), 5)
        self.assertEqual(lo.tell(), 5)
        self.assertEqual(lo.read(), 'data')
        lo.seek(5)
        self.assertEqual(lo.seek(2, 1), 7)
        self.assertEqual(lo.tell(), 7)
        self.assertEqual(lo.read(), 'ta')
        self.assertEqual(lo.seek(-2, 2), length - 2)
        self.assertEqual(lo.read(), 'ta')

    def test_unlink(self):
        if False:
            i = 10
            return i + 15
        lo = self.conn.lobject()
        lo.unlink()
        self.assertRaises(psycopg2.OperationalError, self.conn.lobject, lo.oid)
        self.assertEquals(lo.closed, True)

    def test_export(self):
        if False:
            print('Hello World!')
        lo = self.conn.lobject()
        lo.write(b'some data')
        self.tmpdir = tempfile.mkdtemp()
        filename = os.path.join(self.tmpdir, 'data.txt')
        lo.export(filename)
        self.assertTrue(os.path.exists(filename))
        f = open(filename, 'rb')
        try:
            self.assertEqual(f.read(), b'some data')
        finally:
            f.close()

    def test_close_twice(self):
        if False:
            return 10
        lo = self.conn.lobject()
        lo.close()
        lo.close()

    def test_write_after_close(self):
        if False:
            i = 10
            return i + 15
        lo = self.conn.lobject()
        lo.close()
        self.assertRaises(psycopg2.InterfaceError, lo.write, b'some data')

    def test_read_after_close(self):
        if False:
            while True:
                i = 10
        lo = self.conn.lobject()
        lo.close()
        self.assertRaises(psycopg2.InterfaceError, lo.read, 5)

    def test_seek_after_close(self):
        if False:
            print('Hello World!')
        lo = self.conn.lobject()
        lo.close()
        self.assertRaises(psycopg2.InterfaceError, lo.seek, 0)

    def test_tell_after_close(self):
        if False:
            return 10
        lo = self.conn.lobject()
        lo.close()
        self.assertRaises(psycopg2.InterfaceError, lo.tell)

    def test_unlink_after_close(self):
        if False:
            print('Hello World!')
        lo = self.conn.lobject()
        lo.close()
        lo.unlink()

    def test_export_after_close(self):
        if False:
            print('Hello World!')
        lo = self.conn.lobject()
        lo.write(b'some data')
        lo.close()
        self.tmpdir = tempfile.mkdtemp()
        filename = os.path.join(self.tmpdir, 'data.txt')
        lo.export(filename)
        self.assertTrue(os.path.exists(filename))
        f = open(filename, 'rb')
        try:
            self.assertEqual(f.read(), b'some data')
        finally:
            f.close()

    def test_close_after_commit(self):
        if False:
            print('Hello World!')
        lo = self.conn.lobject()
        self.lo_oid = lo.oid
        self.conn.commit()
        lo.close()

    def test_write_after_commit(self):
        if False:
            print('Hello World!')
        lo = self.conn.lobject()
        self.lo_oid = lo.oid
        self.conn.commit()
        self.assertRaises(psycopg2.ProgrammingError, lo.write, b'some data')

    def test_read_after_commit(self):
        if False:
            for i in range(10):
                print('nop')
        lo = self.conn.lobject()
        self.lo_oid = lo.oid
        self.conn.commit()
        self.assertRaises(psycopg2.ProgrammingError, lo.read, 5)

    def test_seek_after_commit(self):
        if False:
            i = 10
            return i + 15
        lo = self.conn.lobject()
        self.lo_oid = lo.oid
        self.conn.commit()
        self.assertRaises(psycopg2.ProgrammingError, lo.seek, 0)

    def test_tell_after_commit(self):
        if False:
            return 10
        lo = self.conn.lobject()
        self.lo_oid = lo.oid
        self.conn.commit()
        self.assertRaises(psycopg2.ProgrammingError, lo.tell)

    def test_unlink_after_commit(self):
        if False:
            i = 10
            return i + 15
        lo = self.conn.lobject()
        self.lo_oid = lo.oid
        self.conn.commit()
        lo.unlink()

    def test_export_after_commit(self):
        if False:
            while True:
                i = 10
        lo = self.conn.lobject()
        lo.write(b'some data')
        self.conn.commit()
        self.tmpdir = tempfile.mkdtemp()
        filename = os.path.join(self.tmpdir, 'data.txt')
        lo.export(filename)
        self.assertTrue(os.path.exists(filename))
        f = open(filename, 'rb')
        try:
            self.assertEqual(f.read(), b'some data')
        finally:
            f.close()

    @skip_if_tpc_disabled
    def test_read_after_tpc_commit(self):
        if False:
            while True:
                i = 10
        self.conn.tpc_begin('test_lobject')
        lo = self.conn.lobject()
        self.lo_oid = lo.oid
        self.conn.tpc_commit()
        self.assertRaises(psycopg2.ProgrammingError, lo.read, 5)

    @skip_if_tpc_disabled
    def test_read_after_tpc_prepare(self):
        if False:
            return 10
        self.conn.tpc_begin('test_lobject')
        lo = self.conn.lobject()
        self.lo_oid = lo.oid
        self.conn.tpc_prepare()
        try:
            self.assertRaises(psycopg2.ProgrammingError, lo.read, 5)
        finally:
            self.conn.tpc_commit()

    def test_large_oid(self):
        if False:
            return 10
        try:
            self.conn.lobject(4294967294)
        except psycopg2.OperationalError:
            pass

    def test_factory(self):
        if False:
            return 10

        class lobject_subclass(psycopg2.extensions.lobject):
            pass
        lo = self.conn.lobject(lobject_factory=lobject_subclass)
        self.assert_(isinstance(lo, lobject_subclass))

@decorate_all_tests
def skip_if_no_truncate(f):
    if False:
        for i in range(10):
            print('nop')

    @wraps(f)
    def skip_if_no_truncate_(self):
        if False:
            print('Hello World!')
        if self.conn.info.server_version < 80300:
            return self.skipTest("the server doesn't support large object truncate")
        if not hasattr(psycopg2.extensions.lobject, 'truncate'):
            return self.skipTest('psycopg2 has been built against a libpq without large object truncate support.')
        return f(self)
    return skip_if_no_truncate_

@skip_if_no_lo
@skip_if_no_truncate
class LargeObjectTruncateTests(LargeObjectTestCase):

    def test_truncate(self):
        if False:
            i = 10
            return i + 15
        lo = self.conn.lobject()
        lo.write('some data')
        lo.close()
        lo = self.conn.lobject(lo.oid, 'w')
        lo.truncate(4)
        self.assertEqual(lo.tell(), 0)
        self.assertEqual(lo.read(), 'some')
        lo.truncate(6)
        lo.seek(0)
        self.assertEqual(lo.read(), 'some\x00\x00')
        lo.truncate()
        lo.seek(0)
        self.assertEqual(lo.read(), '')

    def test_truncate_after_close(self):
        if False:
            i = 10
            return i + 15
        lo = self.conn.lobject()
        lo.close()
        self.assertRaises(psycopg2.InterfaceError, lo.truncate)

    def test_truncate_after_commit(self):
        if False:
            return 10
        lo = self.conn.lobject()
        self.lo_oid = lo.oid
        self.conn.commit()
        self.assertRaises(psycopg2.ProgrammingError, lo.truncate)

def _has_lo64(conn):
    if False:
        i = 10
        return i + 15
    'Return (bool, msg) about the lo64 support'
    if conn.info.server_version < 90300:
        return (False, "server version %s doesn't support the lo64 API" % conn.info.server_version)
    if 'lo64' not in psycopg2.__version__:
        return (False, "this psycopg build doesn't support the lo64 API")
    return (True, 'this server and build support the lo64 API')

@decorate_all_tests
def skip_if_no_lo64(f):
    if False:
        i = 10
        return i + 15

    @wraps(f)
    def skip_if_no_lo64_(self):
        if False:
            return 10
        (lo64, msg) = _has_lo64(self.conn)
        if not lo64:
            return self.skipTest(msg)
        else:
            return f(self)
    return skip_if_no_lo64_

@skip_if_no_lo
@skip_if_no_truncate
@skip_if_no_lo64
class LargeObject64Tests(LargeObjectTestCase):

    def test_seek_tell_truncate_greater_than_2gb(self):
        if False:
            i = 10
            return i + 15
        lo = self.conn.lobject()
        length = (1 << 31) + (1 << 30)
        lo.truncate(length)
        self.assertEqual(lo.seek(length, 0), length)
        self.assertEqual(lo.tell(), length)

@decorate_all_tests
def skip_if_lo64(f):
    if False:
        print('Hello World!')

    @wraps(f)
    def skip_if_lo64_(self):
        if False:
            print('Hello World!')
        (lo64, msg) = _has_lo64(self.conn)
        if lo64:
            return self.skipTest(msg)
        else:
            return f(self)
    return skip_if_lo64_

@skip_if_no_lo
@skip_if_no_truncate
@skip_if_lo64
class LargeObjectNot64Tests(LargeObjectTestCase):

    def test_seek_larger_than_2gb(self):
        if False:
            while True:
                i = 10
        lo = self.conn.lobject()
        offset = 1 << 32
        self.assertRaises((OverflowError, psycopg2.InterfaceError, psycopg2.NotSupportedError), lo.seek, offset, 0)

    def test_truncate_larger_than_2gb(self):
        if False:
            while True:
                i = 10
        lo = self.conn.lobject()
        length = 1 << 32
        self.assertRaises((OverflowError, psycopg2.InterfaceError, psycopg2.NotSupportedError), lo.truncate, length)

def test_suite():
    if False:
        print('Hello World!')
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main()