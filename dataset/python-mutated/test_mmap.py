from test.support import requires, _2G, _4G, gc_collect, cpython_only
from test.support.import_helper import import_module
from test.support.os_helper import TESTFN, unlink
import unittest
import os
import re
import itertools
import socket
import sys
import weakref
mmap = import_module('mmap')
PAGESIZE = mmap.PAGESIZE

class MmapTests(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        if os.path.exists(TESTFN):
            os.unlink(TESTFN)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            os.unlink(TESTFN)
        except OSError:
            pass

    def test_basic(self):
        if False:
            while True:
                i = 10
        f = open(TESTFN, 'bw+')
        try:
            f.write(b'\x00' * PAGESIZE)
            f.write(b'foo')
            f.write(b'\x00' * (PAGESIZE - 3))
            f.flush()
            m = mmap.mmap(f.fileno(), 2 * PAGESIZE)
        finally:
            f.close()
        tp = str(type(m))
        self.assertEqual(m.find(b'foo'), PAGESIZE)
        self.assertEqual(len(m), 2 * PAGESIZE)
        self.assertEqual(m[0], 0)
        self.assertEqual(m[0:3], b'\x00\x00\x00')
        self.assertRaises(IndexError, m.__getitem__, len(m))
        self.assertRaises(IndexError, m.__setitem__, len(m), b'\x00')
        m[0] = b'3'[0]
        m[PAGESIZE + 3:PAGESIZE + 3 + 3] = b'bar'
        self.assertEqual(m[0], b'3'[0])
        self.assertEqual(m[0:3], b'3\x00\x00')
        self.assertEqual(m[PAGESIZE - 1:PAGESIZE + 7], b'\x00foobar\x00')
        m.flush()
        match = re.search(b'[A-Za-z]+', m)
        if match is None:
            self.fail('regex match on mmap failed!')
        else:
            (start, end) = match.span(0)
            length = end - start
            self.assertEqual(start, PAGESIZE)
            self.assertEqual(end, PAGESIZE + 6)
        m.seek(0, 0)
        self.assertEqual(m.tell(), 0)
        m.seek(42, 1)
        self.assertEqual(m.tell(), 42)
        m.seek(0, 2)
        self.assertEqual(m.tell(), len(m))
        self.assertRaises(ValueError, m.seek, -1)
        self.assertRaises(ValueError, m.seek, 1, 2)
        self.assertRaises(ValueError, m.seek, -len(m) - 1, 2)
        try:
            m.resize(512)
        except SystemError:
            pass
        else:
            self.assertEqual(len(m), 512)
            self.assertRaises(ValueError, m.seek, 513, 0)
            f = open(TESTFN, 'rb')
            try:
                f.seek(0, 2)
                self.assertEqual(f.tell(), 512)
            finally:
                f.close()
            self.assertEqual(m.size(), 512)
        m.close()

    def test_access_parameter(self):
        if False:
            return 10
        mapsize = 10
        with open(TESTFN, 'wb') as fp:
            fp.write(b'a' * mapsize)
        with open(TESTFN, 'rb') as f:
            m = mmap.mmap(f.fileno(), mapsize, access=mmap.ACCESS_READ)
            self.assertEqual(m[:], b'a' * mapsize, 'Readonly memory map data incorrect.')
            try:
                m[:] = b'b' * mapsize
            except TypeError:
                pass
            else:
                self.fail('Able to write to readonly memory map')
            try:
                m[0] = b'b'
            except TypeError:
                pass
            else:
                self.fail('Able to write to readonly memory map')
            try:
                m.seek(0, 0)
                m.write(b'abc')
            except TypeError:
                pass
            else:
                self.fail('Able to write to readonly memory map')
            try:
                m.seek(0, 0)
                m.write_byte(b'd')
            except TypeError:
                pass
            else:
                self.fail('Able to write to readonly memory map')
            try:
                m.resize(2 * mapsize)
            except SystemError:
                pass
            except TypeError:
                pass
            else:
                self.fail('Able to resize readonly memory map')
            with open(TESTFN, 'rb') as fp:
                self.assertEqual(fp.read(), b'a' * mapsize, 'Readonly memory map data file was modified')
        with open(TESTFN, 'r+b') as f:
            try:
                m = mmap.mmap(f.fileno(), mapsize + 1)
            except ValueError:
                if sys.platform.startswith('win'):
                    self.fail('Opening mmap with size+1 should work on Windows.')
            else:
                if not sys.platform.startswith('win'):
                    self.fail('Opening mmap with size+1 should raise ValueError.')
                m.close()
            if sys.platform.startswith('win'):
                with open(TESTFN, 'r+b') as f:
                    f.truncate(mapsize)
        with open(TESTFN, 'r+b') as f:
            m = mmap.mmap(f.fileno(), mapsize, access=mmap.ACCESS_WRITE)
            m[:] = b'c' * mapsize
            self.assertEqual(m[:], b'c' * mapsize, 'Write-through memory map memory not updated properly.')
            m.flush()
            m.close()
        with open(TESTFN, 'rb') as f:
            stuff = f.read()
        self.assertEqual(stuff, b'c' * mapsize, 'Write-through memory map data file not updated properly.')
        with open(TESTFN, 'r+b') as f:
            m = mmap.mmap(f.fileno(), mapsize, access=mmap.ACCESS_COPY)
            m[:] = b'd' * mapsize
            self.assertEqual(m[:], b'd' * mapsize, 'Copy-on-write memory map data not written correctly.')
            m.flush()
            with open(TESTFN, 'rb') as fp:
                self.assertEqual(fp.read(), b'c' * mapsize, 'Copy-on-write test data file should not be modified.')
            self.assertRaises(TypeError, m.resize, 2 * mapsize)
            m.close()
        with open(TESTFN, 'r+b') as f:
            self.assertRaises(ValueError, mmap.mmap, f.fileno(), mapsize, access=4)
        if os.name == 'posix':
            with open(TESTFN, 'r+b') as f:
                self.assertRaises(ValueError, mmap.mmap, f.fileno(), mapsize, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ, access=mmap.ACCESS_WRITE)
            prot = mmap.PROT_READ | getattr(mmap, 'PROT_EXEC', 0)
            with open(TESTFN, 'r+b') as f:
                m = mmap.mmap(f.fileno(), mapsize, prot=prot)
                self.assertRaises(TypeError, m.write, b'abcdef')
                self.assertRaises(TypeError, m.write_byte, 0)
                m.close()

    def test_bad_file_desc(self):
        if False:
            while True:
                i = 10
        self.assertRaises(OSError, mmap.mmap, -2, 4096)

    def test_tougher_find(self):
        if False:
            for i in range(10):
                print('nop')
        with open(TESTFN, 'wb+') as f:
            data = b'aabaac\x00deef\x00\x00aa\x00'
            n = len(data)
            f.write(data)
            f.flush()
            m = mmap.mmap(f.fileno(), n)
        for start in range(n + 1):
            for finish in range(start, n + 1):
                slice = data[start:finish]
                self.assertEqual(m.find(slice), data.find(slice))
                self.assertEqual(m.find(slice + b'x'), -1)
        m.close()

    def test_find_end(self):
        if False:
            return 10
        with open(TESTFN, 'wb+') as f:
            data = b'one two ones'
            n = len(data)
            f.write(data)
            f.flush()
            m = mmap.mmap(f.fileno(), n)
        self.assertEqual(m.find(b'one'), 0)
        self.assertEqual(m.find(b'ones'), 8)
        self.assertEqual(m.find(b'one', 0, -1), 0)
        self.assertEqual(m.find(b'one', 1), 8)
        self.assertEqual(m.find(b'one', 1, -1), 8)
        self.assertEqual(m.find(b'one', 1, -2), -1)
        self.assertEqual(m.find(bytearray(b'one')), 0)

    def test_rfind(self):
        if False:
            for i in range(10):
                print('nop')
        with open(TESTFN, 'wb+') as f:
            data = b'one two ones'
            n = len(data)
            f.write(data)
            f.flush()
            m = mmap.mmap(f.fileno(), n)
        self.assertEqual(m.rfind(b'one'), 8)
        self.assertEqual(m.rfind(b'one '), 0)
        self.assertEqual(m.rfind(b'one', 0, -1), 8)
        self.assertEqual(m.rfind(b'one', 0, -2), 0)
        self.assertEqual(m.rfind(b'one', 1, -1), 8)
        self.assertEqual(m.rfind(b'one', 1, -2), -1)
        self.assertEqual(m.rfind(bytearray(b'one')), 8)

    def test_double_close(self):
        if False:
            print('Hello World!')
        with open(TESTFN, 'wb+') as f:
            f.write(2 ** 16 * b'a')
        with open(TESTFN, 'rb') as f:
            mf = mmap.mmap(f.fileno(), 2 ** 16, access=mmap.ACCESS_READ)
            mf.close()
            mf.close()

    def test_entire_file(self):
        if False:
            return 10
        with open(TESTFN, 'wb+') as f:
            f.write(2 ** 16 * b'm')
        with open(TESTFN, 'rb+') as f, mmap.mmap(f.fileno(), 0) as mf:
            self.assertEqual(len(mf), 2 ** 16, 'Map size should equal file size.')
            self.assertEqual(mf.read(2 ** 16), 2 ** 16 * b'm')

    def test_length_0_offset(self):
        if False:
            for i in range(10):
                print('nop')
        with open(TESTFN, 'wb') as f:
            f.write(65536 * 2 * b'm')
        with open(TESTFN, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, offset=65536, access=mmap.ACCESS_READ) as mf:
                self.assertRaises(IndexError, mf.__getitem__, 80000)

    def test_length_0_large_offset(self):
        if False:
            for i in range(10):
                print('nop')
        with open(TESTFN, 'wb') as f:
            f.write(115699 * b'm')
        with open(TESTFN, 'w+b') as f:
            self.assertRaises(ValueError, mmap.mmap, f.fileno(), 0, offset=2147418112)

    def test_move(self):
        if False:
            i = 10
            return i + 15
        with open(TESTFN, 'wb+') as f:
            f.write(b'ABCDEabcde')
            f.flush()
            mf = mmap.mmap(f.fileno(), 10)
            mf.move(5, 0, 5)
            self.assertEqual(mf[:], b'ABCDEABCDE', 'Map move should have duplicated front 5')
            mf.close()
        data = b'0123456789'
        for dest in range(len(data)):
            for src in range(len(data)):
                for count in range(len(data) - max(dest, src)):
                    expected = data[:dest] + data[src:src + count] + data[dest + count:]
                    m = mmap.mmap(-1, len(data))
                    m[:] = data
                    m.move(dest, src, count)
                    self.assertEqual(m[:], expected)
                    m.close()
        m = mmap.mmap(-1, 100)
        offsets = [-100, -1, 0, 1, 100]
        for (source, dest, size) in itertools.product(offsets, offsets, offsets):
            try:
                m.move(source, dest, size)
            except ValueError:
                pass
        offsets = [(-1, -1, -1), (-1, -1, 0), (-1, 0, -1), (0, -1, -1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
        for (source, dest, size) in offsets:
            self.assertRaises(ValueError, m.move, source, dest, size)
        m.close()
        m = mmap.mmap(-1, 1)
        self.assertRaises(ValueError, m.move, 0, 0, 2)
        self.assertRaises(ValueError, m.move, 1, 0, 1)
        self.assertRaises(ValueError, m.move, 0, 1, 1)
        m.move(0, 0, 1)
        m.move(0, 0, 0)

    def test_anonymous(self):
        if False:
            i = 10
            return i + 15
        m = mmap.mmap(-1, PAGESIZE)
        for x in range(PAGESIZE):
            self.assertEqual(m[x], 0, "anonymously mmap'ed contents should be zero")
        for x in range(PAGESIZE):
            b = x & 255
            m[x] = b
            self.assertEqual(m[x], b)

    def test_read_all(self):
        if False:
            return 10
        m = mmap.mmap(-1, 16)
        self.addCleanup(m.close)
        m.write(bytes(range(16)))
        m.seek(0)
        self.assertEqual(m.read(), bytes(range(16)))
        m.seek(8)
        self.assertEqual(m.read(), bytes(range(8, 16)))
        m.seek(16)
        self.assertEqual(m.read(), b'')
        m.seek(3)
        self.assertEqual(m.read(None), bytes(range(3, 16)))
        m.seek(4)
        self.assertEqual(m.read(-1), bytes(range(4, 16)))
        m.seek(5)
        self.assertEqual(m.read(-2), bytes(range(5, 16)))
        m.seek(9)
        self.assertEqual(m.read(-42), bytes(range(9, 16)))

    def test_read_invalid_arg(self):
        if False:
            i = 10
            return i + 15
        m = mmap.mmap(-1, 16)
        self.addCleanup(m.close)
        self.assertRaises(TypeError, m.read, 'foo')
        self.assertRaises(TypeError, m.read, 5.5)
        self.assertRaises(TypeError, m.read, [1, 2, 3])

    def test_extended_getslice(self):
        if False:
            print('Hello World!')
        s = bytes(reversed(range(256)))
        m = mmap.mmap(-1, len(s))
        m[:] = s
        self.assertEqual(m[:], s)
        indices = (0, None, 1, 3, 19, 300, sys.maxsize, -1, -2, -31, -300)
        for start in indices:
            for stop in indices:
                for step in indices[1:]:
                    self.assertEqual(m[start:stop:step], s[start:stop:step])

    def test_extended_set_del_slice(self):
        if False:
            i = 10
            return i + 15
        s = bytes(reversed(range(256)))
        m = mmap.mmap(-1, len(s))
        indices = (0, None, 1, 3, 19, 300, sys.maxsize, -1, -2, -31, -300)
        for start in indices:
            for stop in indices:
                for step in indices[1:]:
                    m[:] = s
                    self.assertEqual(m[:], s)
                    L = list(s)
                    data = L[start:stop:step]
                    data = bytes(reversed(data))
                    L[start:stop:step] = data
                    m[start:stop:step] = data
                    self.assertEqual(m[:], bytes(L))

    def make_mmap_file(self, f, halfsize):
        if False:
            while True:
                i = 10
        f.write(b'\x00' * halfsize)
        f.write(b'foo')
        f.write(b'\x00' * (halfsize - 3))
        f.flush()
        return mmap.mmap(f.fileno(), 0)

    def test_empty_file(self):
        if False:
            while True:
                i = 10
        f = open(TESTFN, 'w+b')
        f.close()
        with open(TESTFN, 'rb') as f:
            self.assertRaisesRegex(ValueError, 'cannot mmap an empty file', mmap.mmap, f.fileno(), 0, access=mmap.ACCESS_READ)

    def test_offset(self):
        if False:
            i = 10
            return i + 15
        f = open(TESTFN, 'w+b')
        try:
            halfsize = mmap.ALLOCATIONGRANULARITY
            m = self.make_mmap_file(f, halfsize)
            m.close()
            f.close()
            mapsize = halfsize * 2
            f = open(TESTFN, 'r+b')
            for offset in [-2, -1, None]:
                try:
                    m = mmap.mmap(f.fileno(), mapsize, offset=offset)
                    self.assertEqual(0, 1)
                except (ValueError, TypeError, OverflowError):
                    pass
                else:
                    self.assertEqual(0, 0)
            f.close()
            f = open(TESTFN, 'r+b')
            m = mmap.mmap(f.fileno(), mapsize - halfsize, offset=halfsize)
            self.assertEqual(m[0:3], b'foo')
            f.close()
            try:
                m.resize(512)
            except SystemError:
                pass
            else:
                self.assertEqual(len(m), 512)
                self.assertRaises(ValueError, m.seek, 513, 0)
                self.assertEqual(m[0:3], b'foo')
                f = open(TESTFN, 'rb')
                f.seek(0, 2)
                self.assertEqual(f.tell(), halfsize + 512)
                f.close()
                self.assertEqual(m.size(), halfsize + 512)
            m.close()
        finally:
            f.close()
            try:
                os.unlink(TESTFN)
            except OSError:
                pass

    def test_subclass(self):
        if False:
            print('Hello World!')

        class anon_mmap(mmap.mmap):

            def __new__(klass, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                return mmap.mmap.__new__(klass, -1, *args, **kwargs)
        anon_mmap(PAGESIZE)

    @unittest.skipUnless(hasattr(mmap, 'PROT_READ'), 'needs mmap.PROT_READ')
    def test_prot_readonly(self):
        if False:
            for i in range(10):
                print('nop')
        mapsize = 10
        with open(TESTFN, 'wb') as fp:
            fp.write(b'a' * mapsize)
        with open(TESTFN, 'rb') as f:
            m = mmap.mmap(f.fileno(), mapsize, prot=mmap.PROT_READ)
            self.assertRaises(TypeError, m.write, 'foo')

    def test_error(self):
        if False:
            i = 10
            return i + 15
        self.assertIs(mmap.error, OSError)

    def test_io_methods(self):
        if False:
            print('Hello World!')
        data = b'0123456789'
        with open(TESTFN, 'wb') as fp:
            fp.write(b'x' * len(data))
        with open(TESTFN, 'r+b') as f:
            m = mmap.mmap(f.fileno(), len(data))
        for i in range(len(data)):
            self.assertEqual(m.tell(), i)
            m.write_byte(data[i])
            self.assertEqual(m.tell(), i + 1)
        self.assertRaises(ValueError, m.write_byte, b'x'[0])
        self.assertEqual(m[:], data)
        m.seek(0)
        for i in range(len(data)):
            self.assertEqual(m.tell(), i)
            self.assertEqual(m.read_byte(), data[i])
            self.assertEqual(m.tell(), i + 1)
        self.assertRaises(ValueError, m.read_byte)
        m.seek(3)
        self.assertEqual(m.read(3), b'345')
        self.assertEqual(m.tell(), 6)
        m.seek(3)
        m.write(b'bar')
        self.assertEqual(m.tell(), 6)
        self.assertEqual(m[:], b'012bar6789')
        m.write(bytearray(b'baz'))
        self.assertEqual(m.tell(), 9)
        self.assertEqual(m[:], b'012barbaz9')
        self.assertRaises(ValueError, m.write, b'ba')

    def test_non_ascii_byte(self):
        if False:
            print('Hello World!')
        for b in (129, 200, 255):
            m = mmap.mmap(-1, 1)
            m.write_byte(b)
            self.assertEqual(m[0], b)
            m.seek(0)
            self.assertEqual(m.read_byte(), b)
            m.close()

    @unittest.skipUnless(os.name == 'nt', 'requires Windows')
    def test_tagname(self):
        if False:
            while True:
                i = 10
        data1 = b'0123456789'
        data2 = b'abcdefghij'
        assert len(data1) == len(data2)
        m1 = mmap.mmap(-1, len(data1), tagname='foo')
        m1[:] = data1
        m2 = mmap.mmap(-1, len(data2), tagname='foo')
        m2[:] = data2
        self.assertEqual(m1[:], data2)
        self.assertEqual(m2[:], data2)
        m2.close()
        m1.close()
        m1 = mmap.mmap(-1, len(data1), tagname='foo')
        m1[:] = data1
        m2 = mmap.mmap(-1, len(data2), tagname='boo')
        m2[:] = data2
        self.assertEqual(m1[:], data1)
        self.assertEqual(m2[:], data2)
        m2.close()
        m1.close()

    @cpython_only
    @unittest.skipUnless(os.name == 'nt', 'requires Windows')
    def test_sizeof(self):
        if False:
            i = 10
            return i + 15
        m1 = mmap.mmap(-1, 100)
        tagname = 'foo'
        m2 = mmap.mmap(-1, 100, tagname=tagname)
        self.assertEqual(sys.getsizeof(m2), sys.getsizeof(m1) + len(tagname) + 1)

    @unittest.skipUnless(os.name == 'nt', 'requires Windows')
    def test_crasher_on_windows(self):
        if False:
            while True:
                i = 10
        m = mmap.mmap(-1, 1000, tagname='foo')
        try:
            mmap.mmap(-1, 5000, tagname='foo')[:]
        except:
            pass
        m.close()
        with open(TESTFN, 'wb') as fp:
            fp.write(b'x' * 10)
        f = open(TESTFN, 'r+b')
        m = mmap.mmap(f.fileno(), 0)
        f.close()
        try:
            m.resize(0)
        except:
            pass
        try:
            m[:]
        except:
            pass
        m.close()

    @unittest.skipUnless(os.name == 'nt', 'requires Windows')
    def test_invalid_descriptor(self):
        if False:
            while True:
                i = 10
        s = socket.socket()
        try:
            with self.assertRaises(OSError):
                m = mmap.mmap(s.fileno(), 10)
        finally:
            s.close()

    def test_context_manager(self):
        if False:
            i = 10
            return i + 15
        with mmap.mmap(-1, 10) as m:
            self.assertFalse(m.closed)
        self.assertTrue(m.closed)

    def test_context_manager_exception(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(Exception) as exc:
            with mmap.mmap(-1, 10) as m:
                raise OSError
        self.assertIsInstance(exc.exception, OSError, 'wrong exception raised in context manager')
        self.assertTrue(m.closed, 'context manager failed')

    def test_weakref(self):
        if False:
            i = 10
            return i + 15
        mm = mmap.mmap(-1, 16)
        wr = weakref.ref(mm)
        self.assertIs(wr(), mm)
        del mm
        gc_collect()
        self.assertIs(wr(), None)

    def test_write_returning_the_number_of_bytes_written(self):
        if False:
            return 10
        mm = mmap.mmap(-1, 16)
        self.assertEqual(mm.write(b''), 0)
        self.assertEqual(mm.write(b'x'), 1)
        self.assertEqual(mm.write(b'yz'), 2)
        self.assertEqual(mm.write(b'python'), 6)

    @unittest.skipIf(os.name == 'nt', 'cannot resize anonymous mmaps on Windows')
    def test_resize_past_pos(self):
        if False:
            for i in range(10):
                print('nop')
        m = mmap.mmap(-1, 8192)
        self.addCleanup(m.close)
        m.read(5000)
        try:
            m.resize(4096)
        except SystemError:
            self.skipTest('resizing not supported')
        self.assertEqual(m.read(14), b'')
        self.assertRaises(ValueError, m.read_byte)
        self.assertRaises(ValueError, m.write_byte, 42)
        self.assertRaises(ValueError, m.write, b'abc')

    def test_concat_repeat_exception(self):
        if False:
            return 10
        m = mmap.mmap(-1, 16)
        with self.assertRaises(TypeError):
            m + m
        with self.assertRaises(TypeError):
            m * 2

    def test_flush_return_value(self):
        if False:
            i = 10
            return i + 15
        mm = mmap.mmap(-1, 16)
        self.addCleanup(mm.close)
        mm.write(b'python')
        result = mm.flush()
        self.assertIsNone(result)
        if sys.platform.startswith('linux'):
            self.assertRaises(OSError, mm.flush, 1, len(b'python'))

    def test_repr(self):
        if False:
            while True:
                i = 10
        open_mmap_repr_pat = re.compile('<mmap.mmap closed=False, access=(?P<access>\\S+), length=(?P<length>\\d+), pos=(?P<pos>\\d+), offset=(?P<offset>\\d+)>')
        closed_mmap_repr_pat = re.compile('<mmap.mmap closed=True>')
        mapsizes = (50, 100, 1000, 1000000, 10000000)
        offsets = tuple((mapsize // 2 // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY for mapsize in mapsizes))
        for (offset, mapsize) in zip(offsets, mapsizes):
            data = b'a' * mapsize
            length = mapsize - offset
            accesses = ('ACCESS_DEFAULT', 'ACCESS_READ', 'ACCESS_COPY', 'ACCESS_WRITE')
            positions = (0, length // 10, length // 5, length // 4)
            with open(TESTFN, 'wb+') as fp:
                fp.write(data)
                fp.flush()
                for (access, pos) in itertools.product(accesses, positions):
                    accint = getattr(mmap, access)
                    with mmap.mmap(fp.fileno(), length, access=accint, offset=offset) as mm:
                        mm.seek(pos)
                        match = open_mmap_repr_pat.match(repr(mm))
                        self.assertIsNotNone(match)
                        self.assertEqual(match.group('access'), access)
                        self.assertEqual(match.group('length'), str(length))
                        self.assertEqual(match.group('pos'), str(pos))
                        self.assertEqual(match.group('offset'), str(offset))
                    match = closed_mmap_repr_pat.match(repr(mm))
                    self.assertIsNotNone(match)

    @unittest.skipUnless(hasattr(mmap.mmap, 'madvise'), 'needs madvise')
    def test_madvise(self):
        if False:
            for i in range(10):
                print('nop')
        size = 2 * PAGESIZE
        m = mmap.mmap(-1, size)
        with self.assertRaisesRegex(ValueError, 'madvise start out of bounds'):
            m.madvise(mmap.MADV_NORMAL, size)
        with self.assertRaisesRegex(ValueError, 'madvise start out of bounds'):
            m.madvise(mmap.MADV_NORMAL, -1)
        with self.assertRaisesRegex(ValueError, 'madvise length invalid'):
            m.madvise(mmap.MADV_NORMAL, 0, -1)
        with self.assertRaisesRegex(OverflowError, 'madvise length too large'):
            m.madvise(mmap.MADV_NORMAL, PAGESIZE, sys.maxsize)
        self.assertEqual(m.madvise(mmap.MADV_NORMAL), None)
        self.assertEqual(m.madvise(mmap.MADV_NORMAL, PAGESIZE), None)
        self.assertEqual(m.madvise(mmap.MADV_NORMAL, PAGESIZE, size), None)
        self.assertEqual(m.madvise(mmap.MADV_NORMAL, 0, 2), None)
        self.assertEqual(m.madvise(mmap.MADV_NORMAL, 0, size), None)

class LargeMmapTests(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        unlink(TESTFN)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        unlink(TESTFN)

    def _make_test_file(self, num_zeroes, tail):
        if False:
            return 10
        if sys.platform[:3] == 'win' or sys.platform == 'darwin':
            requires('largefile', 'test requires %s bytes and a long time to run' % str(6442450944))
        f = open(TESTFN, 'w+b')
        try:
            f.seek(num_zeroes)
            f.write(tail)
            f.flush()
        except (OSError, OverflowError, ValueError):
            try:
                f.close()
            except (OSError, OverflowError):
                pass
            raise unittest.SkipTest('filesystem does not have largefile support')
        return f

    def test_large_offset(self):
        if False:
            while True:
                i = 10
        with self._make_test_file(5637144575, b' ') as f:
            with mmap.mmap(f.fileno(), 0, offset=5368709120, access=mmap.ACCESS_READ) as m:
                self.assertEqual(m[268435455], 32)

    def test_large_filesize(self):
        if False:
            return 10
        with self._make_test_file(6442450943, b' ') as f:
            if sys.maxsize < 6442450944:
                with self.assertRaises(OverflowError):
                    mmap.mmap(f.fileno(), 6442450944, access=mmap.ACCESS_READ)
                with self.assertRaises(ValueError):
                    mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            with mmap.mmap(f.fileno(), 65536, access=mmap.ACCESS_READ) as m:
                self.assertEqual(m.size(), 6442450944)

    def _test_around_boundary(self, boundary):
        if False:
            return 10
        tail = b'  DEARdear  '
        start = boundary - len(tail) // 2
        end = start + len(tail)
        with self._make_test_file(start, tail) as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                self.assertEqual(m[start:end], tail)

    @unittest.skipUnless(sys.maxsize > _4G, 'test cannot run on 32-bit systems')
    def test_around_2GB(self):
        if False:
            i = 10
            return i + 15
        self._test_around_boundary(_2G)

    @unittest.skipUnless(sys.maxsize > _4G, 'test cannot run on 32-bit systems')
    def test_around_4GB(self):
        if False:
            while True:
                i = 10
        self._test_around_boundary(_4G)
if __name__ == '__main__':
    unittest.main()