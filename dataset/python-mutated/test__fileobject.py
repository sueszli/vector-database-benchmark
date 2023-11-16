from __future__ import print_function
from __future__ import absolute_import
import functools
import gc
import io
import os
import sys
import tempfile
import unittest
import gevent
from gevent import fileobject
from gevent._fileobjectcommon import OpenDescriptor
try:
    from gevent._fileobjectposix import GreenOpenDescriptor
except ImportError:
    GreenOpenDescriptor = None
import gevent.testing as greentest
from gevent.testing import sysinfo

def Writer(fobj, line):
    if False:
        while True:
            i = 10
    for character in line:
        fobj.write(character)
        fobj.flush()
    fobj.close()

def close_fd_quietly(fd):
    if False:
        return 10
    try:
        os.close(fd)
    except OSError:
        pass

def skipUnlessWorksWithRegularFiles(func):
    if False:
        return 10

    @functools.wraps(func)
    def f(self):
        if False:
            while True:
                i = 10
        if not self.WORKS_WITH_REGULAR_FILES:
            self.skipTest("Doesn't work with regular files")
        func(self)
    return f

class CleanupMixin(object):

    def _mkstemp(self, suffix):
        if False:
            while True:
                i = 10
        (fileno, path) = tempfile.mkstemp(suffix)
        self.addCleanup(os.remove, path)
        self.addCleanup(close_fd_quietly, fileno)
        return (fileno, path)

    def _pipe(self):
        if False:
            return 10
        (r, w) = os.pipe()
        self.addCleanup(close_fd_quietly, r)
        self.addCleanup(close_fd_quietly, w)
        return (r, w)

class TestFileObjectBlock(CleanupMixin, greentest.TestCase):
    WORKS_WITH_REGULAR_FILES = True

    def _getTargetClass(self):
        if False:
            while True:
                i = 10
        return fileobject.FileObjectBlock

    def _makeOne(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return self._getTargetClass()(*args, **kwargs)

    def _test_del(self, **kwargs):
        if False:
            while True:
                i = 10
        (r, w) = self._pipe()
        self._do_test_del((r, w), **kwargs)

    def _do_test_del(self, pipe, **kwargs):
        if False:
            i = 10
            return i + 15
        (r, w) = pipe
        s = self._makeOne(w, 'wb', **kwargs)
        s.write(b'x')
        try:
            s.flush()
        except IOError:
            print('Failed flushing fileobject', repr(s), file=sys.stderr)
            import traceback
            traceback.print_exc()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ResourceWarning)
            del s
            gc.collect()
        if kwargs.get('close', True):
            with self.assertRaises((OSError, IOError)):
                os.close(w)
        else:
            os.close(w)
        with self._makeOne(r, 'rb') as fobj:
            self.assertEqual(fobj.read(), b'x')

    def test_del(self):
        if False:
            print('Hello World!')
        self._test_del()

    def test_del_close(self):
        if False:
            return 10
        self._test_del(close=True)

    @skipUnlessWorksWithRegularFiles
    def test_seek(self):
        if False:
            while True:
                i = 10
        (fileno, path) = self._mkstemp('.gevent.test__fileobject.test_seek')
        s = b'a' * 1024
        os.write(fileno, b'B' * 15)
        os.write(fileno, s)
        os.close(fileno)
        with open(path, 'rb') as f:
            f.seek(15)
            native_data = f.read(1024)
        with open(path, 'rb') as f_raw:
            f = self._makeOne(f_raw, 'rb', close=False)
            self.assertTrue(f.seekable())
            f.seek(15)
            self.assertEqual(15, f.tell())
            fileobj_data = f.read(1024)
        self.assertEqual(native_data, s)
        self.assertEqual(native_data, fileobj_data)

    def __check_native_matches(self, byte_data, open_mode, meth='read', open_path=True, **open_kwargs):
        if False:
            print('Hello World!')
        (fileno, path) = self._mkstemp('.gevent_test_' + open_mode)
        os.write(fileno, byte_data)
        os.close(fileno)
        with io.open(path, open_mode, **open_kwargs) as f:
            native_data = getattr(f, meth)()
        if open_path:
            with self._makeOne(path, open_mode, **open_kwargs) as f:
                gevent_data = getattr(f, meth)()
        else:
            opener = io.open
            with opener(path, open_mode, **open_kwargs) as raw:
                with self._makeOne(raw) as f:
                    gevent_data = getattr(f, meth)()
        self.assertEqual(native_data, gevent_data)
        return gevent_data

    @skipUnlessWorksWithRegularFiles
    def test_str_default_to_native(self):
        if False:
            for i in range(10):
                print('nop')
        gevent_data = self.__check_native_matches(b'abcdefg', 'r')
        self.assertIsInstance(gevent_data, str)

    @skipUnlessWorksWithRegularFiles
    def test_text_encoding(self):
        if False:
            for i in range(10):
                print('nop')
        gevent_data = self.__check_native_matches(u'☃'.encode('utf-8'), 'r+', buffering=5, encoding='utf-8')
        self.assertIsInstance(gevent_data, str)

    @skipUnlessWorksWithRegularFiles
    def test_does_not_leak_on_exception(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @skipUnlessWorksWithRegularFiles
    def test_rbU_produces_bytes_readline(self):
        if False:
            print('Hello World!')
        if sys.version_info > (3, 11):
            self.skipTest('U file mode was removed in 3.11')
        gevent_data = self.__check_native_matches(b'line1\nline2\r\nline3\rlastline\n\n', 'rbU', meth='readlines')
        self.assertIsInstance(gevent_data[0], bytes)
        self.assertEqual(len(gevent_data), 4)

    @skipUnlessWorksWithRegularFiles
    def test_rU_produces_native(self):
        if False:
            i = 10
            return i + 15
        if sys.version_info > (3, 11):
            self.skipTest('U file mode was removed in 3.11')
        gevent_data = self.__check_native_matches(b'line1\nline2\r\nline3\rlastline\n\n', 'rU', meth='readlines')
        self.assertIsInstance(gevent_data[0], str)

    @skipUnlessWorksWithRegularFiles
    def test_r_readline_produces_native(self):
        if False:
            while True:
                i = 10
        gevent_data = self.__check_native_matches(b'line1\n', 'r', meth='readline')
        self.assertIsInstance(gevent_data, str)

    @skipUnlessWorksWithRegularFiles
    def test_r_readline_on_fobject_produces_native(self):
        if False:
            return 10
        gevent_data = self.__check_native_matches(b'line1\n', 'r', meth='readline', open_path=False)
        self.assertIsInstance(gevent_data, str)

    def test_close_pipe(self):
        if False:
            i = 10
            return i + 15
        (r, w) = os.pipe()
        x = self._makeOne(r)
        y = self._makeOne(w, 'w')
        x.close()
        y.close()

    @skipUnlessWorksWithRegularFiles
    @greentest.ignores_leakcheck
    def test_name_after_close(self):
        if False:
            return 10
        (fileno, path) = self._mkstemp('.gevent_test_named_path_after_close')
        f = self._makeOne(fileno)
        nf = os.fdopen(fileno)
        nf_name = '<fdopen>' if greentest.PY2 else fileno
        self.assertEqual(f.name, fileno)
        self.assertEqual(nf.name, nf_name)

        class Nameless(object):

            def fileno(self):
                if False:
                    i = 10
                    return i + 15
                return fileno
            close = flush = isatty = closed = writable = lambda self: False
            seekable = readable = lambda self: True
        nameless = self._makeOne(Nameless(), 'rb')
        with self.assertRaises(AttributeError):
            getattr(nameless, 'name')
        nameless.close()
        with self.assertRaises(AttributeError):
            getattr(nameless, 'name')
        f.close()
        try:
            nf.close()
        except OSError:
            pass
        self.assertEqual(f.name, fileno)
        self.assertEqual(nf.name, nf_name)

        def check(arg):
            if False:
                print('Hello World!')
            f = self._makeOne(arg)
            self.assertEqual(f.name, path)
            f.close()
            self.assertEqual(f.name, path)
        check(path)
        with open(path) as nf:
            check(nf)
        with io.open(path) as nf:
            check(nf)

    @skipUnlessWorksWithRegularFiles
    def test_readinto_serial(self):
        if False:
            i = 10
            return i + 15
        (fileno, path) = self._mkstemp('.gevent_test_readinto')
        os.write(fileno, b'hello world')
        os.close(fileno)
        buf = bytearray(32)
        mbuf = memoryview(buf)

        def assertReadInto(byte_count, expected_data):
            if False:
                i = 10
                return i + 15
            bytes_read = f.readinto(mbuf[:byte_count])
            self.assertEqual(bytes_read, len(expected_data))
            self.assertEqual(buf[:bytes_read], expected_data)
        with self._makeOne(path, 'rb') as f:
            assertReadInto(2, b'he')
            assertReadInto(1, b'l')
            assertReadInto(32, b'lo world')
            assertReadInto(32, b'')

class ConcurrentFileObjectMixin(object):

    def test_read1_binary_present(self):
        if False:
            i = 10
            return i + 15
        (r, w) = self._pipe()
        reader = self._makeOne(r, 'rb')
        self._close_on_teardown(reader)
        writer = self._makeOne(w, 'w')
        self._close_on_teardown(writer)
        self.assertTrue(hasattr(reader, 'read1'), dir(reader))

    def test_read1_text_not_present(self):
        if False:
            while True:
                i = 10
        (r, w) = self._pipe()
        reader = self._makeOne(r, 'rt')
        self._close_on_teardown(reader)
        self.addCleanup(os.close, w)
        self.assertFalse(hasattr(reader, 'read1'), dir(reader))

    def test_read1_default(self):
        if False:
            while True:
                i = 10
        (r, w) = self._pipe()
        self.addCleanup(os.close, w)
        reader = self._makeOne(r)
        self._close_on_teardown(reader)
        self.assertFalse(hasattr(reader, 'read1'))

    def test_bufsize_0(self):
        if False:
            print('Hello World!')
        (r, w) = self._pipe()
        x = self._makeOne(r, 'rb', bufsize=0)
        y = self._makeOne(w, 'wb', bufsize=0)
        self._close_on_teardown(x)
        self._close_on_teardown(y)
        y.write(b'a')
        b = x.read(1)
        self.assertEqual(b, b'a')
        y.writelines([b'2'])
        b = x.read(1)
        self.assertEqual(b, b'2')

    def test_newlines(self):
        if False:
            for i in range(10):
                print('nop')
        import warnings
        (r, w) = self._pipe()
        lines = [b'line1\n', b'line2\r', b'line3\r\n', b'line4\r\nline5', b'\nline6']
        g = gevent.spawn(Writer, self._makeOne(w, 'wb'), lines)
        try:
            with warnings.catch_warnings():
                if sys.version_info > (3, 11):
                    mode = 'r'
                    self.skipTest('U file mode was removed in 3.11')
                else:
                    warnings.simplefilter('ignore', DeprecationWarning)
                    mode = 'rU'
                fobj = self._makeOne(r, mode)
            result = fobj.read()
            fobj.close()
            self.assertEqual('line1\nline2\nline3\nline4\nline5\nline6', result)
        finally:
            g.kill()

    def test_readinto(self):
        if False:
            i = 10
            return i + 15
        (r, w) = self._pipe()
        rf = self._close_on_teardown(self._makeOne(r, 'rb'))
        wf = self._close_on_teardown(self._makeOne(w, 'wb'))
        g = gevent.spawn(Writer, wf, [b'hello'])
        try:
            buf1 = bytearray(32)
            buf2 = bytearray(32)
            n1 = rf.readinto(buf1)
            n2 = rf.readinto(buf2)
            self.assertEqual(n1, 5)
            self.assertEqual(buf1[:n1], b'hello')
            self.assertEqual(n2, 0)
        finally:
            g.kill()

class TestFileObjectThread(ConcurrentFileObjectMixin, TestFileObjectBlock):

    def _getTargetClass(self):
        if False:
            while True:
                i = 10
        return fileobject.FileObjectThread

    def test_del_noclose(self):
        if False:
            i = 10
            return i + 15
        self._test_del(close=False)

    def test_del(self):
        if False:
            while True:
                i = 10
        raise unittest.SkipTest('Race conditions')

    def test_del_close(self):
        if False:
            for i in range(10):
                print('nop')
        raise unittest.SkipTest('Race conditions')

@unittest.skipUnless(hasattr(fileobject, 'FileObjectPosix'), 'Needs FileObjectPosix')
class TestFileObjectPosix(ConcurrentFileObjectMixin, TestFileObjectBlock):
    if sysinfo.LIBUV and sysinfo.LINUX:
        WORKS_WITH_REGULAR_FILES = False

    def _getTargetClass(self):
        if False:
            i = 10
            return i + 15
        return fileobject.FileObjectPosix

    def test_seek_raises_ioerror(self):
        if False:
            print('Hello World!')
        (r, _w) = self._pipe()
        with self.assertRaises(OSError) as ctx:
            os.lseek(r, 0, os.SEEK_SET)
        os_ex = ctx.exception
        with self.assertRaises(IOError) as ctx:
            f = self._makeOne(r, 'r', close=False)
            f.fileio.seek(0)
        io_ex = ctx.exception
        self.assertEqual(io_ex.errno, os_ex.errno)
        self.assertEqual(io_ex.strerror, os_ex.strerror)
        self.assertEqual(io_ex.args, os_ex.args)
        self.assertEqual(str(io_ex), str(os_ex))

class TestTextMode(CleanupMixin, unittest.TestCase):

    def test_default_mode_writes_linesep(self):
        if False:
            print('Hello World!')
        gevent.get_hub()
        (fileno, path) = self._mkstemp('.gevent.test__fileobject.test_default')
        os.close(fileno)
        with open(path, 'w') as f:
            f.write('\n')
        with open(path, 'rb') as f:
            data = f.read()
        self.assertEqual(data, os.linesep.encode('ascii'))

class TestOpenDescriptor(CleanupMixin, greentest.TestCase):

    def _getTargetClass(self):
        if False:
            while True:
                i = 10
        return OpenDescriptor

    def _makeOne(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return self._getTargetClass()(*args, **kwargs)

    def _check(self, regex, kind, *args, **kwargs):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(kind, regex):
            self._makeOne(*args, **kwargs)
    case = lambda re, **kwargs: (re, TypeError, kwargs)
    vase = lambda re, **kwargs: (re, ValueError, kwargs)
    CASES = (case('mode', mode=42), case('buffering', buffering='nope'), case('encoding', encoding=42), case('errors', errors=42), vase('mode', mode='aoeug'), vase('mode U cannot be combined', mode='wU'), vase('text and binary', mode='rtb'), vase('append mode at once', mode='rw'), vase('exactly one', mode='+'), vase('take an encoding', mode='rb', encoding='ascii'), vase('take an errors', mode='rb', errors='strict'), vase('take a newline', mode='rb', newline='\n'))

    def test_atomicwrite_fd(self):
        if False:
            for i in range(10):
                print('nop')
        from gevent._fileobjectcommon import WriteallMixin
        (fileno, _w) = self._pipe()
        desc = self._makeOne(fileno, 'wb', buffering=0, closefd=False, atomic_write=True)
        self.assertTrue(desc.atomic_write)
        fobj = desc.opened()
        self.assertIsInstance(fobj, WriteallMixin)
        os.close(fileno)

def pop():
    if False:
        for i in range(10):
            print('nop')
    for (regex, kind, kwargs) in TestOpenDescriptor.CASES:
        setattr(TestOpenDescriptor, 'test_' + regex.replace(' ', '_'), lambda self, _re=regex, _kind=kind, _kw=kwargs: self._check(_re, _kind, 1, **_kw))
pop()

@unittest.skipIf(GreenOpenDescriptor is None, 'No support for non-blocking IO')
class TestGreenOpenDescripton(TestOpenDescriptor):

    def _getTargetClass(self):
        if False:
            return 10
        return GreenOpenDescriptor
if __name__ == '__main__':
    greentest.main()