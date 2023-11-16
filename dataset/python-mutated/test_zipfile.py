import array
import contextlib
import importlib.util
import io
import itertools
import os
import pathlib
import posixpath
import string
import struct
import subprocess
import sys
import time
import unittest
import unittest.mock as mock
import zipfile
import functools
from tempfile import TemporaryFile
from random import randint, random, randbytes
from test.support import script_helper
from test.support import findfile, requires_zlib, requires_bz2, requires_lzma, captured_stdout
from test.support.os_helper import TESTFN, unlink, rmtree, temp_dir, temp_cwd
TESTFN2 = TESTFN + '2'
TESTFNDIR = TESTFN + 'd'
FIXEDTEST_SIZE = 1000
DATAFILES_DIR = 'zipfile_datafiles'
SMALL_TEST_DATA = [('_ziptest1', '1q2w3e4r5t'), ('ziptest2dir/_ziptest2', 'qawsedrftg'), ('ziptest2dir/ziptest3dir/_ziptest3', 'azsxdcfvgb'), ('ziptest2dir/ziptest3dir/ziptest4dir/_ziptest3', '6y7u8i9o0p')]

def get_files(test):
    if False:
        while True:
            i = 10
    yield TESTFN2
    with TemporaryFile() as f:
        yield f
        test.assertFalse(f.closed)
    with io.BytesIO() as f:
        yield f
        test.assertFalse(f.closed)

class AbstractTestsWithSourceFile:

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.line_gen = [bytes('Zipfile test line %d. random float: %f\n' % (i, random()), 'ascii') for i in range(FIXEDTEST_SIZE)]
        cls.data = b''.join(cls.line_gen)

    def setUp(self):
        if False:
            while True:
                i = 10
        with open(TESTFN, 'wb') as fp:
            fp.write(self.data)

    def make_test_archive(self, f, compression, compresslevel=None):
        if False:
            i = 10
            return i + 15
        kwargs = {'compression': compression, 'compresslevel': compresslevel}
        with zipfile.ZipFile(f, 'w', **kwargs) as zipfp:
            zipfp.write(TESTFN, 'another.name')
            zipfp.write(TESTFN, TESTFN)
            zipfp.writestr('strfile', self.data)
            with zipfp.open('written-open-w', mode='w') as f:
                for line in self.line_gen:
                    f.write(line)

    def zip_test(self, f, compression, compresslevel=None):
        if False:
            for i in range(10):
                print('nop')
        self.make_test_archive(f, compression, compresslevel)
        with zipfile.ZipFile(f, 'r', compression) as zipfp:
            self.assertEqual(zipfp.read(TESTFN), self.data)
            self.assertEqual(zipfp.read('another.name'), self.data)
            self.assertEqual(zipfp.read('strfile'), self.data)
            fp = io.StringIO()
            zipfp.printdir(file=fp)
            directory = fp.getvalue()
            lines = directory.splitlines()
            self.assertEqual(len(lines), 5)
            self.assertIn('File Name', lines[0])
            self.assertIn('Modified', lines[0])
            self.assertIn('Size', lines[0])
            (fn, date, time_, size) = lines[1].split()
            self.assertEqual(fn, 'another.name')
            self.assertTrue(time.strptime(date, '%Y-%m-%d'))
            self.assertTrue(time.strptime(time_, '%H:%M:%S'))
            self.assertEqual(size, str(len(self.data)))
            names = zipfp.namelist()
            self.assertEqual(len(names), 4)
            self.assertIn(TESTFN, names)
            self.assertIn('another.name', names)
            self.assertIn('strfile', names)
            self.assertIn('written-open-w', names)
            infos = zipfp.infolist()
            names = [i.filename for i in infos]
            self.assertEqual(len(names), 4)
            self.assertIn(TESTFN, names)
            self.assertIn('another.name', names)
            self.assertIn('strfile', names)
            self.assertIn('written-open-w', names)
            for i in infos:
                self.assertEqual(i.file_size, len(self.data))
            for nm in (TESTFN, 'another.name', 'strfile', 'written-open-w'):
                info = zipfp.getinfo(nm)
                self.assertEqual(info.filename, nm)
                self.assertEqual(info.file_size, len(self.data))
            zipfp.testzip()

    def test_basic(self):
        if False:
            print('Hello World!')
        for f in get_files(self):
            self.zip_test(f, self.compression)

    def zip_open_test(self, f, compression):
        if False:
            print('Hello World!')
        self.make_test_archive(f, compression)
        with zipfile.ZipFile(f, 'r', compression) as zipfp:
            zipdata1 = []
            with zipfp.open(TESTFN) as zipopen1:
                while True:
                    read_data = zipopen1.read(256)
                    if not read_data:
                        break
                    zipdata1.append(read_data)
            zipdata2 = []
            with zipfp.open('another.name') as zipopen2:
                while True:
                    read_data = zipopen2.read(256)
                    if not read_data:
                        break
                    zipdata2.append(read_data)
            self.assertEqual(b''.join(zipdata1), self.data)
            self.assertEqual(b''.join(zipdata2), self.data)

    def test_open(self):
        if False:
            for i in range(10):
                print('nop')
        for f in get_files(self):
            self.zip_open_test(f, self.compression)

    def test_open_with_pathlike(self):
        if False:
            i = 10
            return i + 15
        path = pathlib.Path(TESTFN2)
        self.zip_open_test(path, self.compression)
        with zipfile.ZipFile(path, 'r', self.compression) as zipfp:
            self.assertIsInstance(zipfp.filename, str)

    def zip_random_open_test(self, f, compression):
        if False:
            print('Hello World!')
        self.make_test_archive(f, compression)
        with zipfile.ZipFile(f, 'r', compression) as zipfp:
            zipdata1 = []
            with zipfp.open(TESTFN) as zipopen1:
                while True:
                    read_data = zipopen1.read(randint(1, 1024))
                    if not read_data:
                        break
                    zipdata1.append(read_data)
            self.assertEqual(b''.join(zipdata1), self.data)

    def test_random_open(self):
        if False:
            for i in range(10):
                print('nop')
        for f in get_files(self):
            self.zip_random_open_test(f, self.compression)

    def zip_read1_test(self, f, compression):
        if False:
            for i in range(10):
                print('nop')
        self.make_test_archive(f, compression)
        with zipfile.ZipFile(f, 'r') as zipfp, zipfp.open(TESTFN) as zipopen:
            zipdata = []
            while True:
                read_data = zipopen.read1(-1)
                if not read_data:
                    break
                zipdata.append(read_data)
        self.assertEqual(b''.join(zipdata), self.data)

    def test_read1(self):
        if False:
            i = 10
            return i + 15
        for f in get_files(self):
            self.zip_read1_test(f, self.compression)

    def zip_read1_10_test(self, f, compression):
        if False:
            i = 10
            return i + 15
        self.make_test_archive(f, compression)
        with zipfile.ZipFile(f, 'r') as zipfp, zipfp.open(TESTFN) as zipopen:
            zipdata = []
            while True:
                read_data = zipopen.read1(10)
                self.assertLessEqual(len(read_data), 10)
                if not read_data:
                    break
                zipdata.append(read_data)
        self.assertEqual(b''.join(zipdata), self.data)

    def test_read1_10(self):
        if False:
            return 10
        for f in get_files(self):
            self.zip_read1_10_test(f, self.compression)

    def zip_readline_read_test(self, f, compression):
        if False:
            return 10
        self.make_test_archive(f, compression)
        with zipfile.ZipFile(f, 'r') as zipfp, zipfp.open(TESTFN) as zipopen:
            data = b''
            while True:
                read = zipopen.readline()
                if not read:
                    break
                data += read
                read = zipopen.read(100)
                if not read:
                    break
                data += read
        self.assertEqual(data, self.data)

    def test_readline_read(self):
        if False:
            return 10
        for f in get_files(self):
            self.zip_readline_read_test(f, self.compression)

    def zip_readline_test(self, f, compression):
        if False:
            for i in range(10):
                print('nop')
        self.make_test_archive(f, compression)
        with zipfile.ZipFile(f, 'r') as zipfp:
            with zipfp.open(TESTFN) as zipopen:
                for line in self.line_gen:
                    linedata = zipopen.readline()
                    self.assertEqual(linedata, line)

    def test_readline(self):
        if False:
            print('Hello World!')
        for f in get_files(self):
            self.zip_readline_test(f, self.compression)

    def zip_readlines_test(self, f, compression):
        if False:
            return 10
        self.make_test_archive(f, compression)
        with zipfile.ZipFile(f, 'r') as zipfp:
            with zipfp.open(TESTFN) as zipopen:
                ziplines = zipopen.readlines()
            for (line, zipline) in zip(self.line_gen, ziplines):
                self.assertEqual(zipline, line)

    def test_readlines(self):
        if False:
            print('Hello World!')
        for f in get_files(self):
            self.zip_readlines_test(f, self.compression)

    def zip_iterlines_test(self, f, compression):
        if False:
            i = 10
            return i + 15
        self.make_test_archive(f, compression)
        with zipfile.ZipFile(f, 'r') as zipfp:
            with zipfp.open(TESTFN) as zipopen:
                for (line, zipline) in zip(self.line_gen, zipopen):
                    self.assertEqual(zipline, line)

    def test_iterlines(self):
        if False:
            while True:
                i = 10
        for f in get_files(self):
            self.zip_iterlines_test(f, self.compression)

    def test_low_compression(self):
        if False:
            return 10
        'Check for cases where compressed data is larger than original.'
        with zipfile.ZipFile(TESTFN2, 'w', self.compression) as zipfp:
            zipfp.writestr('strfile', '12')
        with zipfile.ZipFile(TESTFN2, 'r', self.compression) as zipfp:
            with zipfp.open('strfile') as openobj:
                self.assertEqual(openobj.read(1), b'1')
                self.assertEqual(openobj.read(1), b'2')

    def test_writestr_compression(self):
        if False:
            while True:
                i = 10
        zipfp = zipfile.ZipFile(TESTFN2, 'w')
        zipfp.writestr('b.txt', 'hello world', compress_type=self.compression)
        info = zipfp.getinfo('b.txt')
        self.assertEqual(info.compress_type, self.compression)

    def test_writestr_compresslevel(self):
        if False:
            return 10
        zipfp = zipfile.ZipFile(TESTFN2, 'w', compresslevel=1)
        zipfp.writestr('a.txt', 'hello world', compress_type=self.compression)
        zipfp.writestr('b.txt', 'hello world', compress_type=self.compression, compresslevel=2)
        a_info = zipfp.getinfo('a.txt')
        self.assertEqual(a_info.compress_type, self.compression)
        self.assertEqual(a_info._compresslevel, 1)
        b_info = zipfp.getinfo('b.txt')
        self.assertEqual(b_info.compress_type, self.compression)
        self.assertEqual(b_info._compresslevel, 2)

    def test_read_return_size(self):
        if False:
            print('Hello World!')
        for test_size in (1, 4095, 4096, 4097, 16384):
            file_size = test_size + 1
            junk = randbytes(file_size)
            with zipfile.ZipFile(io.BytesIO(), 'w', self.compression) as zipf:
                zipf.writestr('foo', junk)
                with zipf.open('foo', 'r') as fp:
                    buf = fp.read(test_size)
                    self.assertEqual(len(buf), test_size)

    def test_truncated_zipfile(self):
        if False:
            return 10
        fp = io.BytesIO()
        with zipfile.ZipFile(fp, mode='w') as zipf:
            zipf.writestr('strfile', self.data, compress_type=self.compression)
            end_offset = fp.tell()
        zipfiledata = fp.getvalue()
        fp = io.BytesIO(zipfiledata)
        with zipfile.ZipFile(fp) as zipf:
            with zipf.open('strfile') as zipopen:
                fp.truncate(end_offset - 20)
                with self.assertRaises(EOFError):
                    zipopen.read()
        fp = io.BytesIO(zipfiledata)
        with zipfile.ZipFile(fp) as zipf:
            with zipf.open('strfile') as zipopen:
                fp.truncate(end_offset - 20)
                with self.assertRaises(EOFError):
                    while zipopen.read(100):
                        pass
        fp = io.BytesIO(zipfiledata)
        with zipfile.ZipFile(fp) as zipf:
            with zipf.open('strfile') as zipopen:
                fp.truncate(end_offset - 20)
                with self.assertRaises(EOFError):
                    while zipopen.read1(100):
                        pass

    def test_repr(self):
        if False:
            for i in range(10):
                print('nop')
        fname = 'file.name'
        for f in get_files(self):
            with zipfile.ZipFile(f, 'w', self.compression) as zipfp:
                zipfp.write(TESTFN, fname)
                r = repr(zipfp)
                self.assertIn("mode='w'", r)
            with zipfile.ZipFile(f, 'r') as zipfp:
                r = repr(zipfp)
                if isinstance(f, str):
                    self.assertIn('filename=%r' % f, r)
                else:
                    self.assertIn('file=%r' % f, r)
                self.assertIn("mode='r'", r)
                r = repr(zipfp.getinfo(fname))
                self.assertIn('filename=%r' % fname, r)
                self.assertIn('filemode=', r)
                self.assertIn('file_size=', r)
                if self.compression != zipfile.ZIP_STORED:
                    self.assertIn('compress_type=', r)
                    self.assertIn('compress_size=', r)
                with zipfp.open(fname) as zipopen:
                    r = repr(zipopen)
                    self.assertIn('name=%r' % fname, r)
                    self.assertIn("mode='r'", r)
                    if self.compression != zipfile.ZIP_STORED:
                        self.assertIn('compress_type=', r)
                self.assertIn('[closed]', repr(zipopen))
            self.assertIn('[closed]', repr(zipfp))

    def test_compresslevel_basic(self):
        if False:
            while True:
                i = 10
        for f in get_files(self):
            self.zip_test(f, self.compression, compresslevel=9)

    def test_per_file_compresslevel(self):
        if False:
            for i in range(10):
                print('nop')
        'Check that files within a Zip archive can have different\n        compression levels.'
        with zipfile.ZipFile(TESTFN2, 'w', compresslevel=1) as zipfp:
            zipfp.write(TESTFN, 'compress_1')
            zipfp.write(TESTFN, 'compress_9', compresslevel=9)
            one_info = zipfp.getinfo('compress_1')
            nine_info = zipfp.getinfo('compress_9')
            self.assertEqual(one_info._compresslevel, 1)
            self.assertEqual(nine_info._compresslevel, 9)

    def test_writing_errors(self):
        if False:
            while True:
                i = 10

        class BrokenFile(io.BytesIO):

            def write(self, data):
                if False:
                    while True:
                        i = 10
                nonlocal count
                if count is not None:
                    if count == stop:
                        raise OSError
                    count += 1
                super().write(data)
        stop = 0
        while True:
            testfile = BrokenFile()
            count = None
            with zipfile.ZipFile(testfile, 'w', self.compression) as zipfp:
                with zipfp.open('file1', 'w') as f:
                    f.write(b'data1')
                count = 0
                try:
                    with zipfp.open('file2', 'w') as f:
                        f.write(b'data2')
                except OSError:
                    stop += 1
                else:
                    break
                finally:
                    count = None
            with zipfile.ZipFile(io.BytesIO(testfile.getvalue())) as zipfp:
                self.assertEqual(zipfp.namelist(), ['file1'])
                self.assertEqual(zipfp.read('file1'), b'data1')
        with zipfile.ZipFile(io.BytesIO(testfile.getvalue())) as zipfp:
            self.assertEqual(zipfp.namelist(), ['file1', 'file2'])
            self.assertEqual(zipfp.read('file1'), b'data1')
            self.assertEqual(zipfp.read('file2'), b'data2')

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        unlink(TESTFN)
        unlink(TESTFN2)

class StoredTestsWithSourceFile(AbstractTestsWithSourceFile, unittest.TestCase):
    compression = zipfile.ZIP_STORED
    test_low_compression = None

    def zip_test_writestr_permissions(self, f, compression):
        if False:
            for i in range(10):
                print('nop')
        self.make_test_archive(f, compression)
        with zipfile.ZipFile(f, 'r') as zipfp:
            zinfo = zipfp.getinfo('strfile')
            self.assertEqual(zinfo.external_attr, 384 << 16)
            zinfo2 = zipfp.getinfo('written-open-w')
            self.assertEqual(zinfo2.external_attr, 384 << 16)

    def test_writestr_permissions(self):
        if False:
            while True:
                i = 10
        for f in get_files(self):
            self.zip_test_writestr_permissions(f, zipfile.ZIP_STORED)

    def test_absolute_arcnames(self):
        if False:
            for i in range(10):
                print('nop')
        with zipfile.ZipFile(TESTFN2, 'w', zipfile.ZIP_STORED) as zipfp:
            zipfp.write(TESTFN, '/absolute')
        with zipfile.ZipFile(TESTFN2, 'r', zipfile.ZIP_STORED) as zipfp:
            self.assertEqual(zipfp.namelist(), ['absolute'])

    def test_append_to_zip_file(self):
        if False:
            while True:
                i = 10
        'Test appending to an existing zipfile.'
        with zipfile.ZipFile(TESTFN2, 'w', zipfile.ZIP_STORED) as zipfp:
            zipfp.write(TESTFN, TESTFN)
        with zipfile.ZipFile(TESTFN2, 'a', zipfile.ZIP_STORED) as zipfp:
            zipfp.writestr('strfile', self.data)
            self.assertEqual(zipfp.namelist(), [TESTFN, 'strfile'])

    def test_append_to_non_zip_file(self):
        if False:
            i = 10
            return i + 15
        'Test appending to an existing file that is not a zipfile.'
        data = b'I am not a ZipFile!' * 10
        with open(TESTFN2, 'wb') as f:
            f.write(data)
        with zipfile.ZipFile(TESTFN2, 'a', zipfile.ZIP_STORED) as zipfp:
            zipfp.write(TESTFN, TESTFN)
        with open(TESTFN2, 'rb') as f:
            f.seek(len(data))
            with zipfile.ZipFile(f, 'r') as zipfp:
                self.assertEqual(zipfp.namelist(), [TESTFN])
                self.assertEqual(zipfp.read(TESTFN), self.data)
        with open(TESTFN2, 'rb') as f:
            self.assertEqual(f.read(len(data)), data)
            zipfiledata = f.read()
        with io.BytesIO(zipfiledata) as bio, zipfile.ZipFile(bio) as zipfp:
            self.assertEqual(zipfp.namelist(), [TESTFN])
            self.assertEqual(zipfp.read(TESTFN), self.data)

    def test_read_concatenated_zip_file(self):
        if False:
            print('Hello World!')
        with io.BytesIO() as bio:
            with zipfile.ZipFile(bio, 'w', zipfile.ZIP_STORED) as zipfp:
                zipfp.write(TESTFN, TESTFN)
            zipfiledata = bio.getvalue()
        data = b'I am not a ZipFile!' * 10
        with open(TESTFN2, 'wb') as f:
            f.write(data)
            f.write(zipfiledata)
        with zipfile.ZipFile(TESTFN2) as zipfp:
            self.assertEqual(zipfp.namelist(), [TESTFN])
            self.assertEqual(zipfp.read(TESTFN), self.data)

    def test_append_to_concatenated_zip_file(self):
        if False:
            i = 10
            return i + 15
        with io.BytesIO() as bio:
            with zipfile.ZipFile(bio, 'w', zipfile.ZIP_STORED) as zipfp:
                zipfp.write(TESTFN, TESTFN)
            zipfiledata = bio.getvalue()
        data = b'I am not a ZipFile!' * 1000000
        with open(TESTFN2, 'wb') as f:
            f.write(data)
            f.write(zipfiledata)
        with zipfile.ZipFile(TESTFN2, 'a') as zipfp:
            self.assertEqual(zipfp.namelist(), [TESTFN])
            zipfp.writestr('strfile', self.data)
        with open(TESTFN2, 'rb') as f:
            self.assertEqual(f.read(len(data)), data)
            zipfiledata = f.read()
        with io.BytesIO(zipfiledata) as bio, zipfile.ZipFile(bio) as zipfp:
            self.assertEqual(zipfp.namelist(), [TESTFN, 'strfile'])
            self.assertEqual(zipfp.read(TESTFN), self.data)
            self.assertEqual(zipfp.read('strfile'), self.data)

    def test_ignores_newline_at_end(self):
        if False:
            i = 10
            return i + 15
        with zipfile.ZipFile(TESTFN2, 'w', zipfile.ZIP_STORED) as zipfp:
            zipfp.write(TESTFN, TESTFN)
        with open(TESTFN2, 'a', encoding='utf-8') as f:
            f.write('\r\n\x00\x00\x00')
        with zipfile.ZipFile(TESTFN2, 'r') as zipfp:
            self.assertIsInstance(zipfp, zipfile.ZipFile)

    def test_ignores_stuff_appended_past_comments(self):
        if False:
            print('Hello World!')
        with zipfile.ZipFile(TESTFN2, 'w', zipfile.ZIP_STORED) as zipfp:
            zipfp.comment = b'this is a comment'
            zipfp.write(TESTFN, TESTFN)
        with open(TESTFN2, 'a', encoding='utf-8') as f:
            f.write('abcdef\r\n')
        with zipfile.ZipFile(TESTFN2, 'r') as zipfp:
            self.assertIsInstance(zipfp, zipfile.ZipFile)
            self.assertEqual(zipfp.comment, b'this is a comment')

    def test_write_default_name(self):
        if False:
            return 10
        'Check that calling ZipFile.write without arcname specified\n        produces the expected result.'
        with zipfile.ZipFile(TESTFN2, 'w') as zipfp:
            zipfp.write(TESTFN)
            with open(TESTFN, 'rb') as f:
                self.assertEqual(zipfp.read(TESTFN), f.read())

    def test_io_on_closed_zipextfile(self):
        if False:
            while True:
                i = 10
        fname = 'somefile.txt'
        with zipfile.ZipFile(TESTFN2, mode='w') as zipfp:
            zipfp.writestr(fname, 'bogus')
        with zipfile.ZipFile(TESTFN2, mode='r') as zipfp:
            with zipfp.open(fname) as fid:
                fid.close()
                self.assertRaises(ValueError, fid.read)
                self.assertRaises(ValueError, fid.seek, 0)
                self.assertRaises(ValueError, fid.tell)
                self.assertRaises(ValueError, fid.readable)
                self.assertRaises(ValueError, fid.seekable)

    def test_write_to_readonly(self):
        if False:
            return 10
        'Check that trying to call write() on a readonly ZipFile object\n        raises a ValueError.'
        with zipfile.ZipFile(TESTFN2, mode='w') as zipfp:
            zipfp.writestr('somefile.txt', 'bogus')
        with zipfile.ZipFile(TESTFN2, mode='r') as zipfp:
            self.assertRaises(ValueError, zipfp.write, TESTFN)
        with zipfile.ZipFile(TESTFN2, mode='r') as zipfp:
            with self.assertRaises(ValueError):
                zipfp.open(TESTFN, mode='w')

    def test_add_file_before_1980(self):
        if False:
            i = 10
            return i + 15
        os.utime(TESTFN, (0, 0))
        with zipfile.ZipFile(TESTFN2, 'w') as zipfp:
            self.assertRaises(ValueError, zipfp.write, TESTFN)
        with zipfile.ZipFile(TESTFN2, 'w', strict_timestamps=False) as zipfp:
            zipfp.write(TESTFN)
            zinfo = zipfp.getinfo(TESTFN)
            self.assertEqual(zinfo.date_time, (1980, 1, 1, 0, 0, 0))

    def test_add_file_after_2107(self):
        if False:
            print('Hello World!')
        ts = 4386268800
        try:
            time.localtime(ts)
        except OverflowError:
            self.skipTest(f'time.localtime({ts}) raises OverflowError')
        try:
            os.utime(TESTFN, (ts, ts))
        except OverflowError:
            self.skipTest('Host fs cannot set timestamp to required value.')
        mtime_ns = os.stat(TESTFN).st_mtime_ns
        if mtime_ns != 4386268800 * 10 ** 9:
            self.skipTest(f'Linux VFS/XFS kernel bug detected: mtime_ns={mtime_ns!r}')
        with zipfile.ZipFile(TESTFN2, 'w') as zipfp:
            self.assertRaises(struct.error, zipfp.write, TESTFN)
        with zipfile.ZipFile(TESTFN2, 'w', strict_timestamps=False) as zipfp:
            zipfp.write(TESTFN)
            zinfo = zipfp.getinfo(TESTFN)
            self.assertEqual(zinfo.date_time, (2107, 12, 31, 23, 59, 59))

@requires_zlib()
class DeflateTestsWithSourceFile(AbstractTestsWithSourceFile, unittest.TestCase):
    compression = zipfile.ZIP_DEFLATED

    def test_per_file_compression(self):
        if False:
            for i in range(10):
                print('nop')
        'Check that files within a Zip archive can have different\n        compression options.'
        with zipfile.ZipFile(TESTFN2, 'w') as zipfp:
            zipfp.write(TESTFN, 'storeme', zipfile.ZIP_STORED)
            zipfp.write(TESTFN, 'deflateme', zipfile.ZIP_DEFLATED)
            sinfo = zipfp.getinfo('storeme')
            dinfo = zipfp.getinfo('deflateme')
            self.assertEqual(sinfo.compress_type, zipfile.ZIP_STORED)
            self.assertEqual(dinfo.compress_type, zipfile.ZIP_DEFLATED)

@requires_bz2()
class Bzip2TestsWithSourceFile(AbstractTestsWithSourceFile, unittest.TestCase):
    compression = zipfile.ZIP_BZIP2

@requires_lzma()
class LzmaTestsWithSourceFile(AbstractTestsWithSourceFile, unittest.TestCase):
    compression = zipfile.ZIP_LZMA

class AbstractTestZip64InSmallFiles:

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        line_gen = (bytes('Test of zipfile line %d.' % i, 'ascii') for i in range(0, FIXEDTEST_SIZE))
        cls.data = b'\n'.join(line_gen)

    def setUp(self):
        if False:
            return 10
        self._limit = zipfile.ZIP64_LIMIT
        self._filecount_limit = zipfile.ZIP_FILECOUNT_LIMIT
        zipfile.ZIP64_LIMIT = 1000
        zipfile.ZIP_FILECOUNT_LIMIT = 9
        with open(TESTFN, 'wb') as fp:
            fp.write(self.data)

    def zip_test(self, f, compression):
        if False:
            for i in range(10):
                print('nop')
        with zipfile.ZipFile(f, 'w', compression, allowZip64=True) as zipfp:
            zipfp.write(TESTFN, 'another.name')
            zipfp.write(TESTFN, TESTFN)
            zipfp.writestr('strfile', self.data)
        with zipfile.ZipFile(f, 'r', compression) as zipfp:
            self.assertEqual(zipfp.read(TESTFN), self.data)
            self.assertEqual(zipfp.read('another.name'), self.data)
            self.assertEqual(zipfp.read('strfile'), self.data)
            fp = io.StringIO()
            zipfp.printdir(fp)
            directory = fp.getvalue()
            lines = directory.splitlines()
            self.assertEqual(len(lines), 4)
            self.assertIn('File Name', lines[0])
            self.assertIn('Modified', lines[0])
            self.assertIn('Size', lines[0])
            (fn, date, time_, size) = lines[1].split()
            self.assertEqual(fn, 'another.name')
            self.assertTrue(time.strptime(date, '%Y-%m-%d'))
            self.assertTrue(time.strptime(time_, '%H:%M:%S'))
            self.assertEqual(size, str(len(self.data)))
            names = zipfp.namelist()
            self.assertEqual(len(names), 3)
            self.assertIn(TESTFN, names)
            self.assertIn('another.name', names)
            self.assertIn('strfile', names)
            infos = zipfp.infolist()
            names = [i.filename for i in infos]
            self.assertEqual(len(names), 3)
            self.assertIn(TESTFN, names)
            self.assertIn('another.name', names)
            self.assertIn('strfile', names)
            for i in infos:
                self.assertEqual(i.file_size, len(self.data))
            for nm in (TESTFN, 'another.name', 'strfile'):
                info = zipfp.getinfo(nm)
                self.assertEqual(info.filename, nm)
                self.assertEqual(info.file_size, len(self.data))
            zipfp.testzip()

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        for f in get_files(self):
            self.zip_test(f, self.compression)

    def test_too_many_files(self):
        if False:
            print('Hello World!')
        zipf = zipfile.ZipFile(TESTFN, 'w', self.compression, allowZip64=True)
        zipf.debug = 100
        numfiles = 15
        for i in range(numfiles):
            zipf.writestr('foo%08d' % i, '%d' % (i ** 3 % 57))
        self.assertEqual(len(zipf.namelist()), numfiles)
        zipf.close()
        zipf2 = zipfile.ZipFile(TESTFN, 'r', self.compression)
        self.assertEqual(len(zipf2.namelist()), numfiles)
        for i in range(numfiles):
            content = zipf2.read('foo%08d' % i).decode('ascii')
            self.assertEqual(content, '%d' % (i ** 3 % 57))
        zipf2.close()

    def test_too_many_files_append(self):
        if False:
            return 10
        zipf = zipfile.ZipFile(TESTFN, 'w', self.compression, allowZip64=False)
        zipf.debug = 100
        numfiles = 9
        for i in range(numfiles):
            zipf.writestr('foo%08d' % i, '%d' % (i ** 3 % 57))
        self.assertEqual(len(zipf.namelist()), numfiles)
        with self.assertRaises(zipfile.LargeZipFile):
            zipf.writestr('foo%08d' % numfiles, b'')
        self.assertEqual(len(zipf.namelist()), numfiles)
        zipf.close()
        zipf = zipfile.ZipFile(TESTFN, 'a', self.compression, allowZip64=False)
        zipf.debug = 100
        self.assertEqual(len(zipf.namelist()), numfiles)
        with self.assertRaises(zipfile.LargeZipFile):
            zipf.writestr('foo%08d' % numfiles, b'')
        self.assertEqual(len(zipf.namelist()), numfiles)
        zipf.close()
        zipf = zipfile.ZipFile(TESTFN, 'a', self.compression, allowZip64=True)
        zipf.debug = 100
        self.assertEqual(len(zipf.namelist()), numfiles)
        numfiles2 = 15
        for i in range(numfiles, numfiles2):
            zipf.writestr('foo%08d' % i, '%d' % (i ** 3 % 57))
        self.assertEqual(len(zipf.namelist()), numfiles2)
        zipf.close()
        zipf2 = zipfile.ZipFile(TESTFN, 'r', self.compression)
        self.assertEqual(len(zipf2.namelist()), numfiles2)
        for i in range(numfiles2):
            content = zipf2.read('foo%08d' % i).decode('ascii')
            self.assertEqual(content, '%d' % (i ** 3 % 57))
        zipf2.close()

    def tearDown(self):
        if False:
            print('Hello World!')
        zipfile.ZIP64_LIMIT = self._limit
        zipfile.ZIP_FILECOUNT_LIMIT = self._filecount_limit
        unlink(TESTFN)
        unlink(TESTFN2)

class StoredTestZip64InSmallFiles(AbstractTestZip64InSmallFiles, unittest.TestCase):
    compression = zipfile.ZIP_STORED

    def large_file_exception_test(self, f, compression):
        if False:
            for i in range(10):
                print('nop')
        with zipfile.ZipFile(f, 'w', compression, allowZip64=False) as zipfp:
            self.assertRaises(zipfile.LargeZipFile, zipfp.write, TESTFN, 'another.name')

    def large_file_exception_test2(self, f, compression):
        if False:
            i = 10
            return i + 15
        with zipfile.ZipFile(f, 'w', compression, allowZip64=False) as zipfp:
            self.assertRaises(zipfile.LargeZipFile, zipfp.writestr, 'another.name', self.data)

    def test_large_file_exception(self):
        if False:
            return 10
        for f in get_files(self):
            self.large_file_exception_test(f, zipfile.ZIP_STORED)
            self.large_file_exception_test2(f, zipfile.ZIP_STORED)

    def test_absolute_arcnames(self):
        if False:
            while True:
                i = 10
        with zipfile.ZipFile(TESTFN2, 'w', zipfile.ZIP_STORED, allowZip64=True) as zipfp:
            zipfp.write(TESTFN, '/absolute')
        with zipfile.ZipFile(TESTFN2, 'r', zipfile.ZIP_STORED) as zipfp:
            self.assertEqual(zipfp.namelist(), ['absolute'])

    def test_append(self):
        if False:
            print('Hello World!')
        with zipfile.ZipFile(TESTFN2, 'w', allowZip64=True) as zipfp:
            zipfp.writestr('strfile', self.data)
        with zipfile.ZipFile(TESTFN2, 'r', allowZip64=True) as zipfp:
            zinfo = zipfp.getinfo('strfile')
            extra = zinfo.extra
        with zipfile.ZipFile(TESTFN2, 'a', allowZip64=True) as zipfp:
            zipfp.writestr('strfile2', self.data)
        with zipfile.ZipFile(TESTFN2, 'r', allowZip64=True) as zipfp:
            zinfo = zipfp.getinfo('strfile')
            self.assertEqual(zinfo.extra, extra)

    def make_zip64_file(self, file_size_64_set=False, file_size_extra=False, compress_size_64_set=False, compress_size_extra=False, header_offset_64_set=False, header_offset_extra=False):
        if False:
            i = 10
            return i + 15
        'Generate bytes sequence for a zip with (incomplete) zip64 data.\n\n        The actual values (not the zip 64 0xffffffff values) stored in the file\n        are:\n        file_size: 8\n        compress_size: 8\n        header_offset: 0\n        '
        actual_size = 8
        actual_header_offset = 0
        local_zip64_fields = []
        central_zip64_fields = []
        file_size = actual_size
        if file_size_64_set:
            file_size = 4294967295
            if file_size_extra:
                local_zip64_fields.append(actual_size)
                central_zip64_fields.append(actual_size)
        file_size = struct.pack('<L', file_size)
        compress_size = actual_size
        if compress_size_64_set:
            compress_size = 4294967295
            if compress_size_extra:
                local_zip64_fields.append(actual_size)
                central_zip64_fields.append(actual_size)
        compress_size = struct.pack('<L', compress_size)
        header_offset = actual_header_offset
        if header_offset_64_set:
            header_offset = 4294967295
            if header_offset_extra:
                central_zip64_fields.append(actual_header_offset)
        header_offset = struct.pack('<L', header_offset)
        local_extra = struct.pack('<HH' + 'Q' * len(local_zip64_fields), 1, 8 * len(local_zip64_fields), *local_zip64_fields)
        central_extra = struct.pack('<HH' + 'Q' * len(central_zip64_fields), 1, 8 * len(central_zip64_fields), *central_zip64_fields)
        central_dir_size = struct.pack('<Q', 58 + 8 * len(central_zip64_fields))
        offset_to_central_dir = struct.pack('<Q', 50 + 8 * len(local_zip64_fields))
        local_extra_length = struct.pack('<H', 4 + 8 * len(local_zip64_fields))
        central_extra_length = struct.pack('<H', 4 + 8 * len(central_zip64_fields))
        filename = b'test.txt'
        content = b'test1234'
        filename_length = struct.pack('<H', len(filename))
        zip64_contents = b'PK\x03\x04\x14\x00\x00\x00\x00\x00\x00\x00!\x00\x9e%\xf5\xaf' + compress_size + file_size + filename_length + local_extra_length + filename + local_extra + content + b'PK\x01\x02-\x03-\x00\x00\x00\x00\x00\x00\x00!\x00\x9e%\xf5\xaf' + compress_size + file_size + filename_length + central_extra_length + b'\x00\x00\x00\x00\x00\x00\x00\x00\x80\x01' + header_offset + filename + central_extra + b'PK\x06\x06,\x00\x00\x00\x00\x00\x00\x00-\x00-' + b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00' + b'\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00' + central_dir_size + offset_to_central_dir + b'PK\x06\x07\x00\x00\x00\x00l\x00\x00\x00\x00\x00\x00\x00\x01' + b'\x00\x00\x00' + b'PK\x05\x06\x00\x00\x00\x00\x01\x00\x01\x00:\x00\x00\x002\x00' + b'\x00\x00\x00\x00'
        return zip64_contents

    def test_bad_zip64_extra(self):
        if False:
            return 10
        "Missing zip64 extra records raises an exception.\n\n        There are 4 fields that the zip64 format handles (the disk number is\n        not used in this module and so is ignored here). According to the zip\n        spec:\n              The order of the fields in the zip64 extended\n              information record is fixed, but the fields MUST\n              only appear if the corresponding Local or Central\n              directory record field is set to 0xFFFF or 0xFFFFFFFF.\n\n        If the zip64 extra content doesn't contain enough entries for the\n        number of fields marked with 0xFFFF or 0xFFFFFFFF, we raise an error.\n        This test mismatches the length of the zip64 extra field and the number\n        of fields set to indicate the presence of zip64 data.\n        "
        missing_file_size_extra = self.make_zip64_file(file_size_64_set=True)
        with self.assertRaises(zipfile.BadZipFile) as e:
            zipfile.ZipFile(io.BytesIO(missing_file_size_extra))
        self.assertIn('file size', str(e.exception).lower())
        missing_compress_size_extra = self.make_zip64_file(file_size_64_set=True, file_size_extra=True, compress_size_64_set=True)
        with self.assertRaises(zipfile.BadZipFile) as e:
            zipfile.ZipFile(io.BytesIO(missing_compress_size_extra))
        self.assertIn('compress size', str(e.exception).lower())
        missing_compress_size_extra = self.make_zip64_file(compress_size_64_set=True)
        with self.assertRaises(zipfile.BadZipFile) as e:
            zipfile.ZipFile(io.BytesIO(missing_compress_size_extra))
        self.assertIn('compress size', str(e.exception).lower())
        missing_header_offset_extra = self.make_zip64_file(file_size_64_set=True, file_size_extra=True, compress_size_64_set=True, compress_size_extra=True, header_offset_64_set=True)
        with self.assertRaises(zipfile.BadZipFile) as e:
            zipfile.ZipFile(io.BytesIO(missing_header_offset_extra))
        self.assertIn('header offset', str(e.exception).lower())
        missing_header_offset_extra = self.make_zip64_file(file_size_64_set=False, compress_size_64_set=True, compress_size_extra=True, header_offset_64_set=True)
        with self.assertRaises(zipfile.BadZipFile) as e:
            zipfile.ZipFile(io.BytesIO(missing_header_offset_extra))
        self.assertIn('header offset', str(e.exception).lower())
        missing_header_offset_extra = self.make_zip64_file(file_size_64_set=True, file_size_extra=True, compress_size_64_set=False, header_offset_64_set=True)
        with self.assertRaises(zipfile.BadZipFile) as e:
            zipfile.ZipFile(io.BytesIO(missing_header_offset_extra))
        self.assertIn('header offset', str(e.exception).lower())
        missing_header_offset_extra = self.make_zip64_file(file_size_64_set=False, compress_size_64_set=False, header_offset_64_set=True)
        with self.assertRaises(zipfile.BadZipFile) as e:
            zipfile.ZipFile(io.BytesIO(missing_header_offset_extra))
        self.assertIn('header offset', str(e.exception).lower())

    def test_generated_valid_zip64_extra(self):
        if False:
            i = 10
            return i + 15
        expected_file_size = 8
        expected_compress_size = 8
        expected_header_offset = 0
        expected_content = b'test1234'
        params = ({'file_size_64_set': True, 'file_size_extra': True}, {'compress_size_64_set': True, 'compress_size_extra': True}, {'header_offset_64_set': True, 'header_offset_extra': True})
        for r in range(1, len(params) + 1):
            for combo in itertools.combinations(params, r):
                kwargs = {}
                for c in combo:
                    kwargs.update(c)
                with zipfile.ZipFile(io.BytesIO(self.make_zip64_file(**kwargs))) as zf:
                    zinfo = zf.infolist()[0]
                    self.assertEqual(zinfo.file_size, expected_file_size)
                    self.assertEqual(zinfo.compress_size, expected_compress_size)
                    self.assertEqual(zinfo.header_offset, expected_header_offset)
                    self.assertEqual(zf.read(zinfo), expected_content)

@requires_zlib()
class DeflateTestZip64InSmallFiles(AbstractTestZip64InSmallFiles, unittest.TestCase):
    compression = zipfile.ZIP_DEFLATED

@requires_bz2()
class Bzip2TestZip64InSmallFiles(AbstractTestZip64InSmallFiles, unittest.TestCase):
    compression = zipfile.ZIP_BZIP2

@requires_lzma()
class LzmaTestZip64InSmallFiles(AbstractTestZip64InSmallFiles, unittest.TestCase):
    compression = zipfile.ZIP_LZMA

class AbstractWriterTests:

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        unlink(TESTFN2)

    def test_close_after_close(self):
        if False:
            print('Hello World!')
        data = b'content'
        with zipfile.ZipFile(TESTFN2, 'w', self.compression) as zipf:
            w = zipf.open('test', 'w')
            w.write(data)
            w.close()
            self.assertTrue(w.closed)
            w.close()
            self.assertTrue(w.closed)
            self.assertEqual(zipf.read('test'), data)

    def test_write_after_close(self):
        if False:
            print('Hello World!')
        data = b'content'
        with zipfile.ZipFile(TESTFN2, 'w', self.compression) as zipf:
            w = zipf.open('test', 'w')
            w.write(data)
            w.close()
            self.assertTrue(w.closed)
            self.assertRaises(ValueError, w.write, b'')
            self.assertEqual(zipf.read('test'), data)

    def test_issue44439(self):
        if False:
            for i in range(10):
                print('nop')
        q = array.array('Q', [1, 2, 3, 4, 5])
        LENGTH = len(q) * q.itemsize
        with zipfile.ZipFile(io.BytesIO(), 'w', self.compression) as zip:
            with zip.open('data', 'w') as data:
                self.assertEqual(data.write(q), LENGTH)
            self.assertEqual(zip.getinfo('data').file_size, LENGTH)

class StoredWriterTests(AbstractWriterTests, unittest.TestCase):
    compression = zipfile.ZIP_STORED

@requires_zlib()
class DeflateWriterTests(AbstractWriterTests, unittest.TestCase):
    compression = zipfile.ZIP_DEFLATED

@requires_bz2()
class Bzip2WriterTests(AbstractWriterTests, unittest.TestCase):
    compression = zipfile.ZIP_BZIP2

@requires_lzma()
class LzmaWriterTests(AbstractWriterTests, unittest.TestCase):
    compression = zipfile.ZIP_LZMA

class PyZipFileTests(unittest.TestCase):

    def assertCompiledIn(self, name, namelist):
        if False:
            return 10
        if name + 'o' not in namelist:
            self.assertIn(name + 'c', namelist)

    def requiresWriteAccess(self, path):
        if False:
            print('Hello World!')
        if not os.access(path, os.W_OK, effective_ids=os.access in os.supports_effective_ids):
            self.skipTest('requires write access to the installed location')
        filename = os.path.join(path, 'test_zipfile.try')
        try:
            fd = os.open(filename, os.O_WRONLY | os.O_CREAT)
            os.close(fd)
        except Exception:
            self.skipTest('requires write access to the installed location')
        unlink(filename)

    def test_write_pyfile(self):
        if False:
            i = 10
            return i + 15
        self.requiresWriteAccess(os.path.dirname(__file__))
        with TemporaryFile() as t, zipfile.PyZipFile(t, 'w') as zipfp:
            fn = __file__
            if fn.endswith('.pyc'):
                path_split = fn.split(os.sep)
                if os.altsep is not None:
                    path_split.extend(fn.split(os.altsep))
                if '__pycache__' in path_split:
                    fn = importlib.util.source_from_cache(fn)
                else:
                    fn = fn[:-1]
            zipfp.writepy(fn)
            bn = os.path.basename(fn)
            self.assertNotIn(bn, zipfp.namelist())
            self.assertCompiledIn(bn, zipfp.namelist())
        with TemporaryFile() as t, zipfile.PyZipFile(t, 'w') as zipfp:
            fn = __file__
            if fn.endswith('.pyc'):
                fn = fn[:-1]
            zipfp.writepy(fn, 'testpackage')
            bn = '%s/%s' % ('testpackage', os.path.basename(fn))
            self.assertNotIn(bn, zipfp.namelist())
            self.assertCompiledIn(bn, zipfp.namelist())

    def test_write_python_package(self):
        if False:
            print('Hello World!')
        import email
        packagedir = os.path.dirname(email.__file__)
        self.requiresWriteAccess(packagedir)
        with TemporaryFile() as t, zipfile.PyZipFile(t, 'w') as zipfp:
            zipfp.writepy(packagedir)
            names = zipfp.namelist()
            self.assertCompiledIn('email/__init__.py', names)
            self.assertCompiledIn('email/mime/text.py', names)

    def test_write_filtered_python_package(self):
        if False:
            i = 10
            return i + 15
        import test
        packagedir = os.path.dirname(test.__file__)
        self.requiresWriteAccess(packagedir)
        with TemporaryFile() as t, zipfile.PyZipFile(t, 'w') as zipfp:
            with captured_stdout() as reportSIO:
                zipfp.writepy(packagedir)
            reportStr = reportSIO.getvalue()
            self.assertTrue('SyntaxError' in reportStr)
            with captured_stdout() as reportSIO:
                zipfp.writepy(packagedir, filterfunc=lambda whatever: False)
            reportStr = reportSIO.getvalue()
            self.assertTrue('SyntaxError' not in reportStr)

            def filter(path):
                if False:
                    for i in range(10):
                        print('nop')
                return not os.path.basename(path).startswith('bad')
            with captured_stdout() as reportSIO, self.assertWarns(UserWarning):
                zipfp.writepy(packagedir, filterfunc=filter)
            reportStr = reportSIO.getvalue()
            if reportStr:
                print(reportStr)
            self.assertTrue('SyntaxError' not in reportStr)

    def test_write_with_optimization(self):
        if False:
            i = 10
            return i + 15
        import email
        packagedir = os.path.dirname(email.__file__)
        self.requiresWriteAccess(packagedir)
        optlevel = 1 if __debug__ else 0
        ext = '.pyc'
        with TemporaryFile() as t, zipfile.PyZipFile(t, 'w', optimize=optlevel) as zipfp:
            zipfp.writepy(packagedir)
            names = zipfp.namelist()
            self.assertIn('email/__init__' + ext, names)
            self.assertIn('email/mime/text' + ext, names)

    def test_write_python_directory(self):
        if False:
            while True:
                i = 10
        os.mkdir(TESTFN2)
        try:
            with open(os.path.join(TESTFN2, 'mod1.py'), 'w', encoding='utf-8') as fp:
                fp.write('print(42)\n')
            with open(os.path.join(TESTFN2, 'mod2.py'), 'w', encoding='utf-8') as fp:
                fp.write('print(42 * 42)\n')
            with open(os.path.join(TESTFN2, 'mod2.txt'), 'w', encoding='utf-8') as fp:
                fp.write('bla bla bla\n')
            with TemporaryFile() as t, zipfile.PyZipFile(t, 'w') as zipfp:
                zipfp.writepy(TESTFN2)
                names = zipfp.namelist()
                self.assertCompiledIn('mod1.py', names)
                self.assertCompiledIn('mod2.py', names)
                self.assertNotIn('mod2.txt', names)
        finally:
            rmtree(TESTFN2)

    def test_write_python_directory_filtered(self):
        if False:
            while True:
                i = 10
        os.mkdir(TESTFN2)
        try:
            with open(os.path.join(TESTFN2, 'mod1.py'), 'w', encoding='utf-8') as fp:
                fp.write('print(42)\n')
            with open(os.path.join(TESTFN2, 'mod2.py'), 'w', encoding='utf-8') as fp:
                fp.write('print(42 * 42)\n')
            with TemporaryFile() as t, zipfile.PyZipFile(t, 'w') as zipfp:
                zipfp.writepy(TESTFN2, filterfunc=lambda fn: not fn.endswith('mod2.py'))
                names = zipfp.namelist()
                self.assertCompiledIn('mod1.py', names)
                self.assertNotIn('mod2.py', names)
        finally:
            rmtree(TESTFN2)

    def test_write_non_pyfile(self):
        if False:
            for i in range(10):
                print('nop')
        with TemporaryFile() as t, zipfile.PyZipFile(t, 'w') as zipfp:
            with open(TESTFN, 'w', encoding='utf-8') as f:
                f.write('most definitely not a python file')
            self.assertRaises(RuntimeError, zipfp.writepy, TESTFN)
            unlink(TESTFN)

    def test_write_pyfile_bad_syntax(self):
        if False:
            while True:
                i = 10
        os.mkdir(TESTFN2)
        try:
            with open(os.path.join(TESTFN2, 'mod1.py'), 'w', encoding='utf-8') as fp:
                fp.write('Bad syntax in python file\n')
            with TemporaryFile() as t, zipfile.PyZipFile(t, 'w') as zipfp:
                with captured_stdout() as s:
                    zipfp.writepy(os.path.join(TESTFN2, 'mod1.py'))
                self.assertIn('SyntaxError', s.getvalue())
                names = zipfp.namelist()
                self.assertIn('mod1.py', names)
                self.assertNotIn('mod1.pyc', names)
        finally:
            rmtree(TESTFN2)

    def test_write_pathlike(self):
        if False:
            return 10
        os.mkdir(TESTFN2)
        try:
            with open(os.path.join(TESTFN2, 'mod1.py'), 'w', encoding='utf-8') as fp:
                fp.write('print(42)\n')
            with TemporaryFile() as t, zipfile.PyZipFile(t, 'w') as zipfp:
                zipfp.writepy(pathlib.Path(TESTFN2) / 'mod1.py')
                names = zipfp.namelist()
                self.assertCompiledIn('mod1.py', names)
        finally:
            rmtree(TESTFN2)

class ExtractTests(unittest.TestCase):

    def make_test_file(self):
        if False:
            return 10
        with zipfile.ZipFile(TESTFN2, 'w', zipfile.ZIP_STORED) as zipfp:
            for (fpath, fdata) in SMALL_TEST_DATA:
                zipfp.writestr(fpath, fdata)

    def test_extract(self):
        if False:
            return 10
        with temp_cwd():
            self.make_test_file()
            with zipfile.ZipFile(TESTFN2, 'r') as zipfp:
                for (fpath, fdata) in SMALL_TEST_DATA:
                    writtenfile = zipfp.extract(fpath)
                    correctfile = os.path.join(os.getcwd(), fpath)
                    correctfile = os.path.normpath(correctfile)
                    self.assertEqual(writtenfile, correctfile)
                    with open(writtenfile, 'rb') as f:
                        self.assertEqual(fdata.encode(), f.read())
                    unlink(writtenfile)

    def _test_extract_with_target(self, target):
        if False:
            print('Hello World!')
        self.make_test_file()
        with zipfile.ZipFile(TESTFN2, 'r') as zipfp:
            for (fpath, fdata) in SMALL_TEST_DATA:
                writtenfile = zipfp.extract(fpath, target)
                correctfile = os.path.join(target, fpath)
                correctfile = os.path.normpath(correctfile)
                self.assertTrue(os.path.samefile(writtenfile, correctfile), (writtenfile, target))
                with open(writtenfile, 'rb') as f:
                    self.assertEqual(fdata.encode(), f.read())
                unlink(writtenfile)
        unlink(TESTFN2)

    def test_extract_with_target(self):
        if False:
            while True:
                i = 10
        with temp_dir() as extdir:
            self._test_extract_with_target(extdir)

    def test_extract_with_target_pathlike(self):
        if False:
            return 10
        with temp_dir() as extdir:
            self._test_extract_with_target(pathlib.Path(extdir))

    def test_extract_all(self):
        if False:
            return 10
        with temp_cwd():
            self.make_test_file()
            with zipfile.ZipFile(TESTFN2, 'r') as zipfp:
                zipfp.extractall()
                for (fpath, fdata) in SMALL_TEST_DATA:
                    outfile = os.path.join(os.getcwd(), fpath)
                    with open(outfile, 'rb') as f:
                        self.assertEqual(fdata.encode(), f.read())
                    unlink(outfile)

    def _test_extract_all_with_target(self, target):
        if False:
            i = 10
            return i + 15
        self.make_test_file()
        with zipfile.ZipFile(TESTFN2, 'r') as zipfp:
            zipfp.extractall(target)
            for (fpath, fdata) in SMALL_TEST_DATA:
                outfile = os.path.join(target, fpath)
                with open(outfile, 'rb') as f:
                    self.assertEqual(fdata.encode(), f.read())
                unlink(outfile)
        unlink(TESTFN2)

    def test_extract_all_with_target(self):
        if False:
            while True:
                i = 10
        with temp_dir() as extdir:
            self._test_extract_all_with_target(extdir)

    def test_extract_all_with_target_pathlike(self):
        if False:
            i = 10
            return i + 15
        with temp_dir() as extdir:
            self._test_extract_all_with_target(pathlib.Path(extdir))

    def check_file(self, filename, content):
        if False:
            print('Hello World!')
        self.assertTrue(os.path.isfile(filename))
        with open(filename, 'rb') as f:
            self.assertEqual(f.read(), content)

    def test_sanitize_windows_name(self):
        if False:
            return 10
        san = zipfile.ZipFile._sanitize_windows_name
        self.assertEqual(san(',,?,C:,foo,bar/z', ','), '_,C_,foo,bar/z')
        self.assertEqual(san('a\\b,c<d>e|f"g?h*i', ','), 'a\\b,c_d_e_f_g_h_i')
        self.assertEqual(san('../../foo../../ba..r', '/'), 'foo/ba..r')

    def test_extract_hackers_arcnames_common_cases(self):
        if False:
            return 10
        common_hacknames = [('../foo/bar', 'foo/bar'), ('foo/../bar', 'foo/bar'), ('foo/../../bar', 'foo/bar'), ('foo/bar/..', 'foo/bar'), ('./../foo/bar', 'foo/bar'), ('/foo/bar', 'foo/bar'), ('/foo/../bar', 'foo/bar'), ('/foo/../../bar', 'foo/bar')]
        self._test_extract_hackers_arcnames(common_hacknames)

    @unittest.skipIf(os.path.sep != '\\', 'Requires \\ as path separator.')
    def test_extract_hackers_arcnames_windows_only(self):
        if False:
            for i in range(10):
                print('nop')
        'Test combination of path fixing and windows name sanitization.'
        windows_hacknames = [('..\\foo\\bar', 'foo/bar'), ('..\\/foo\\/bar', 'foo/bar'), ('foo/\\..\\/bar', 'foo/bar'), ('foo\\/../\\bar', 'foo/bar'), ('C:foo/bar', 'foo/bar'), ('C:/foo/bar', 'foo/bar'), ('C://foo/bar', 'foo/bar'), ('C:\\foo\\bar', 'foo/bar'), ('//conky/mountpoint/foo/bar', 'foo/bar'), ('\\\\conky\\mountpoint\\foo\\bar', 'foo/bar'), ('///conky/mountpoint/foo/bar', 'conky/mountpoint/foo/bar'), ('\\\\\\conky\\mountpoint\\foo\\bar', 'conky/mountpoint/foo/bar'), ('//conky//mountpoint/foo/bar', 'conky/mountpoint/foo/bar'), ('\\\\conky\\\\mountpoint\\foo\\bar', 'conky/mountpoint/foo/bar'), ('//?/C:/foo/bar', 'foo/bar'), ('\\\\?\\C:\\foo\\bar', 'foo/bar'), ('C:/../C:/foo/bar', 'C_/foo/bar'), ('a:b\\c<d>e|f"g?h*i', 'b/c_d_e_f_g_h_i'), ('../../foo../../ba..r', 'foo/ba..r')]
        self._test_extract_hackers_arcnames(windows_hacknames)

    @unittest.skipIf(os.path.sep != '/', 'Requires / as path separator.')
    def test_extract_hackers_arcnames_posix_only(self):
        if False:
            print('Hello World!')
        posix_hacknames = [('//foo/bar', 'foo/bar'), ('../../foo../../ba..r', 'foo../ba..r'), ('foo/..\\bar', 'foo/..\\bar')]
        self._test_extract_hackers_arcnames(posix_hacknames)

    def _test_extract_hackers_arcnames(self, hacknames):
        if False:
            i = 10
            return i + 15
        for (arcname, fixedname) in hacknames:
            content = b'foobar' + arcname.encode()
            with zipfile.ZipFile(TESTFN2, 'w', zipfile.ZIP_STORED) as zipfp:
                zinfo = zipfile.ZipInfo()
                zinfo.filename = arcname
                zinfo.external_attr = 384 << 16
                zipfp.writestr(zinfo, content)
            arcname = arcname.replace(os.sep, '/')
            targetpath = os.path.join('target', 'subdir', 'subsub')
            correctfile = os.path.join(targetpath, *fixedname.split('/'))
            with zipfile.ZipFile(TESTFN2, 'r') as zipfp:
                writtenfile = zipfp.extract(arcname, targetpath)
                self.assertEqual(writtenfile, correctfile, msg='extract %r: %r != %r' % (arcname, writtenfile, correctfile))
            self.check_file(correctfile, content)
            rmtree('target')
            with zipfile.ZipFile(TESTFN2, 'r') as zipfp:
                zipfp.extractall(targetpath)
            self.check_file(correctfile, content)
            rmtree('target')
            correctfile = os.path.join(os.getcwd(), *fixedname.split('/'))
            with zipfile.ZipFile(TESTFN2, 'r') as zipfp:
                writtenfile = zipfp.extract(arcname)
                self.assertEqual(writtenfile, correctfile, msg='extract %r' % arcname)
            self.check_file(correctfile, content)
            rmtree(fixedname.split('/')[0])
            with zipfile.ZipFile(TESTFN2, 'r') as zipfp:
                zipfp.extractall()
            self.check_file(correctfile, content)
            rmtree(fixedname.split('/')[0])
            unlink(TESTFN2)

class OtherTests(unittest.TestCase):

    def test_open_via_zip_info(self):
        if False:
            for i in range(10):
                print('nop')
        with zipfile.ZipFile(TESTFN2, 'w', zipfile.ZIP_STORED) as zipfp:
            zipfp.writestr('name', 'foo')
            with self.assertWarns(UserWarning):
                zipfp.writestr('name', 'bar')
            self.assertEqual(zipfp.namelist(), ['name'] * 2)
        with zipfile.ZipFile(TESTFN2, 'r') as zipfp:
            infos = zipfp.infolist()
            data = b''
            for info in infos:
                with zipfp.open(info) as zipopen:
                    data += zipopen.read()
            self.assertIn(data, {b'foobar', b'barfoo'})
            data = b''
            for info in infos:
                data += zipfp.read(info)
            self.assertIn(data, {b'foobar', b'barfoo'})

    def test_writestr_extended_local_header_issue1202(self):
        if False:
            i = 10
            return i + 15
        with zipfile.ZipFile(TESTFN2, 'w') as orig_zip:
            for data in 'abcdefghijklmnop':
                zinfo = zipfile.ZipInfo(data)
                zinfo.flag_bits |= 8
                orig_zip.writestr(zinfo, data)

    def test_close(self):
        if False:
            return 10
        "Check that the zipfile is closed after the 'with' block."
        with zipfile.ZipFile(TESTFN2, 'w') as zipfp:
            for (fpath, fdata) in SMALL_TEST_DATA:
                zipfp.writestr(fpath, fdata)
                self.assertIsNotNone(zipfp.fp, 'zipfp is not open')
        self.assertIsNone(zipfp.fp, 'zipfp is not closed')
        with zipfile.ZipFile(TESTFN2, 'r') as zipfp:
            self.assertIsNotNone(zipfp.fp, 'zipfp is not open')
        self.assertIsNone(zipfp.fp, 'zipfp is not closed')

    def test_close_on_exception(self):
        if False:
            while True:
                i = 10
        "Check that the zipfile is closed if an exception is raised in the\n        'with' block."
        with zipfile.ZipFile(TESTFN2, 'w') as zipfp:
            for (fpath, fdata) in SMALL_TEST_DATA:
                zipfp.writestr(fpath, fdata)
        try:
            with zipfile.ZipFile(TESTFN2, 'r') as zipfp2:
                raise zipfile.BadZipFile()
        except zipfile.BadZipFile:
            self.assertIsNone(zipfp2.fp, 'zipfp is not closed')

    def test_unsupported_version(self):
        if False:
            i = 10
            return i + 15
        data = b'PK\x03\x04x\x00\x00\x00\x00\x00!p\xa1@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00xPK\x01\x02x\x03x\x00\x00\x00\x00\x00!p\xa1@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x01\x00\x00\x00\x00xPK\x05\x06\x00\x00\x00\x00\x01\x00\x01\x00/\x00\x00\x00\x1f\x00\x00\x00\x00\x00'
        self.assertRaises(NotImplementedError, zipfile.ZipFile, io.BytesIO(data), 'r')

    @requires_zlib()
    def test_read_unicode_filenames(self):
        if False:
            while True:
                i = 10
        fname = findfile('zip_cp437_header.zip')
        with zipfile.ZipFile(fname) as zipfp:
            for name in zipfp.namelist():
                zipfp.open(name).close()

    def test_write_unicode_filenames(self):
        if False:
            i = 10
            return i + 15
        with zipfile.ZipFile(TESTFN, 'w') as zf:
            zf.writestr('foo.txt', 'Test for unicode filename')
            zf.writestr('ö.txt', 'Test for unicode filename')
            self.assertIsInstance(zf.infolist()[0].filename, str)
        with zipfile.ZipFile(TESTFN, 'r') as zf:
            self.assertEqual(zf.filelist[0].filename, 'foo.txt')
            self.assertEqual(zf.filelist[1].filename, 'ö.txt')

    def test_read_after_write_unicode_filenames(self):
        if False:
            return 10
        with zipfile.ZipFile(TESTFN2, 'w') as zipfp:
            zipfp.writestr('приклад', b'sample')
            self.assertEqual(zipfp.read('приклад'), b'sample')

    def test_exclusive_create_zip_file(self):
        if False:
            return 10
        'Test exclusive creating a new zipfile.'
        unlink(TESTFN2)
        filename = 'testfile.txt'
        content = b'hello, world. this is some content.'
        with zipfile.ZipFile(TESTFN2, 'x', zipfile.ZIP_STORED) as zipfp:
            zipfp.writestr(filename, content)
        with self.assertRaises(FileExistsError):
            zipfile.ZipFile(TESTFN2, 'x', zipfile.ZIP_STORED)
        with zipfile.ZipFile(TESTFN2, 'r') as zipfp:
            self.assertEqual(zipfp.namelist(), [filename])
            self.assertEqual(zipfp.read(filename), content)

    def test_create_non_existent_file_for_append(self):
        if False:
            i = 10
            return i + 15
        if os.path.exists(TESTFN):
            os.unlink(TESTFN)
        filename = 'testfile.txt'
        content = b'hello, world. this is some content.'
        try:
            with zipfile.ZipFile(TESTFN, 'a') as zf:
                zf.writestr(filename, content)
        except OSError:
            self.fail('Could not append data to a non-existent zip file.')
        self.assertTrue(os.path.exists(TESTFN))
        with zipfile.ZipFile(TESTFN, 'r') as zf:
            self.assertEqual(zf.read(filename), content)

    def test_close_erroneous_file(self):
        if False:
            return 10
        with open(TESTFN, 'w', encoding='utf-8') as fp:
            fp.write('this is not a legal zip file\n')
        try:
            zf = zipfile.ZipFile(TESTFN)
        except zipfile.BadZipFile:
            pass

    def test_is_zip_erroneous_file(self):
        if False:
            i = 10
            return i + 15
        'Check that is_zipfile() correctly identifies non-zip files.'
        with open(TESTFN, 'w', encoding='utf-8') as fp:
            fp.write('this is not a legal zip file\n')
        self.assertFalse(zipfile.is_zipfile(TESTFN))
        self.assertFalse(zipfile.is_zipfile(pathlib.Path(TESTFN)))
        with open(TESTFN, 'rb') as fp:
            self.assertFalse(zipfile.is_zipfile(fp))
        fp = io.BytesIO()
        fp.write(b'this is not a legal zip file\n')
        self.assertFalse(zipfile.is_zipfile(fp))
        fp.seek(0, 0)
        self.assertFalse(zipfile.is_zipfile(fp))

    def test_damaged_zipfile(self):
        if False:
            while True:
                i = 10
        'Check that zipfiles with missing bytes at the end raise BadZipFile.'
        fp = io.BytesIO()
        with zipfile.ZipFile(fp, mode='w') as zipf:
            zipf.writestr('foo.txt', b'O, for a Muse of Fire!')
        zipfiledata = fp.getvalue()
        for N in range(len(zipfiledata)):
            fp = io.BytesIO(zipfiledata[:N])
            self.assertRaises(zipfile.BadZipFile, zipfile.ZipFile, fp)

    def test_is_zip_valid_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Check that is_zipfile() correctly identifies zip files.'
        with zipfile.ZipFile(TESTFN, mode='w') as zipf:
            zipf.writestr('foo.txt', b'O, for a Muse of Fire!')
        self.assertTrue(zipfile.is_zipfile(TESTFN))
        with open(TESTFN, 'rb') as fp:
            self.assertTrue(zipfile.is_zipfile(fp))
            fp.seek(0, 0)
            zip_contents = fp.read()
        fp = io.BytesIO()
        fp.write(zip_contents)
        self.assertTrue(zipfile.is_zipfile(fp))
        fp.seek(0, 0)
        self.assertTrue(zipfile.is_zipfile(fp))

    def test_non_existent_file_raises_OSError(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(OSError, zipfile.ZipFile, TESTFN)

    def test_empty_file_raises_BadZipFile(self):
        if False:
            print('Hello World!')
        f = open(TESTFN, 'w', encoding='utf-8')
        f.close()
        self.assertRaises(zipfile.BadZipFile, zipfile.ZipFile, TESTFN)
        with open(TESTFN, 'w', encoding='utf-8') as fp:
            fp.write('short file')
        self.assertRaises(zipfile.BadZipFile, zipfile.ZipFile, TESTFN)

    def test_negative_central_directory_offset_raises_BadZipFile(self):
        if False:
            return 10
        buffer = bytearray(b'PK\x05\x06' + b'\x00' * 18)
        for dirsize in (1, 2 ** 32 - 1):
            buffer[12:16] = struct.pack('<L', dirsize)
            f = io.BytesIO(buffer)
            self.assertRaises(zipfile.BadZipFile, zipfile.ZipFile, f)

    def test_closed_zip_raises_ValueError(self):
        if False:
            while True:
                i = 10
        "Verify that testzip() doesn't swallow inappropriate exceptions."
        data = io.BytesIO()
        with zipfile.ZipFile(data, mode='w') as zipf:
            zipf.writestr('foo.txt', 'O, for a Muse of Fire!')
        self.assertRaises(ValueError, zipf.read, 'foo.txt')
        self.assertRaises(ValueError, zipf.open, 'foo.txt')
        self.assertRaises(ValueError, zipf.testzip)
        self.assertRaises(ValueError, zipf.writestr, 'bogus.txt', 'bogus')
        with open(TESTFN, 'w', encoding='utf-8') as f:
            f.write('zipfile test data')
        self.assertRaises(ValueError, zipf.write, TESTFN)

    def test_bad_constructor_mode(self):
        if False:
            return 10
        'Check that bad modes passed to ZipFile constructor are caught.'
        self.assertRaises(ValueError, zipfile.ZipFile, TESTFN, 'q')

    def test_bad_open_mode(self):
        if False:
            i = 10
            return i + 15
        'Check that bad modes passed to ZipFile.open are caught.'
        with zipfile.ZipFile(TESTFN, mode='w') as zipf:
            zipf.writestr('foo.txt', 'O, for a Muse of Fire!')
        with zipfile.ZipFile(TESTFN, mode='r') as zipf:
            zipf.read('foo.txt')
            self.assertRaises(ValueError, zipf.open, 'foo.txt', 'q')
            self.assertRaises(ValueError, zipf.open, 'foo.txt', 'U')
            self.assertRaises(ValueError, zipf.open, 'foo.txt', 'rU')

    def test_read0(self):
        if False:
            for i in range(10):
                print('nop')
        "Check that calling read(0) on a ZipExtFile object returns an empty\n        string and doesn't advance file pointer."
        with zipfile.ZipFile(TESTFN, mode='w') as zipf:
            zipf.writestr('foo.txt', 'O, for a Muse of Fire!')
            with zipf.open('foo.txt') as f:
                for i in range(FIXEDTEST_SIZE):
                    self.assertEqual(f.read(0), b'')
                self.assertEqual(f.read(), b'O, for a Muse of Fire!')

    def test_open_non_existent_item(self):
        if False:
            i = 10
            return i + 15
        "Check that attempting to call open() for an item that doesn't\n        exist in the archive raises a RuntimeError."
        with zipfile.ZipFile(TESTFN, mode='w') as zipf:
            self.assertRaises(KeyError, zipf.open, 'foo.txt', 'r')

    def test_bad_compression_mode(self):
        if False:
            return 10
        'Check that bad compression methods passed to ZipFile.open are\n        caught.'
        self.assertRaises(NotImplementedError, zipfile.ZipFile, TESTFN, 'w', -1)

    def test_unsupported_compression(self):
        if False:
            while True:
                i = 10
        data = b'PK\x03\x04.\x00\x00\x00\x01\x00\xe4C\xa1@\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00x\x03\x00PK\x01\x02.\x03.\x00\x00\x00\x01\x00\xe4C\xa1@\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x01\x00\x00\x00\x00xPK\x05\x06\x00\x00\x00\x00\x01\x00\x01\x00/\x00\x00\x00!\x00\x00\x00\x00\x00'
        with zipfile.ZipFile(io.BytesIO(data), 'r') as zipf:
            self.assertRaises(NotImplementedError, zipf.open, 'x')

    def test_null_byte_in_filename(self):
        if False:
            for i in range(10):
                print('nop')
        'Check that a filename containing a null byte is properly\n        terminated.'
        with zipfile.ZipFile(TESTFN, mode='w') as zipf:
            zipf.writestr('foo.txt\x00qqq', b'O, for a Muse of Fire!')
            self.assertEqual(zipf.namelist(), ['foo.txt'])

    def test_struct_sizes(self):
        if False:
            return 10
        'Check that ZIP internal structure sizes are calculated correctly.'
        self.assertEqual(zipfile.sizeEndCentDir, 22)
        self.assertEqual(zipfile.sizeCentralDir, 46)
        self.assertEqual(zipfile.sizeEndCentDir64, 56)
        self.assertEqual(zipfile.sizeEndCentDir64Locator, 20)

    def test_comments(self):
        if False:
            return 10
        'Check that comments on the archive are handled properly.'
        with zipfile.ZipFile(TESTFN, mode='w') as zipf:
            self.assertEqual(zipf.comment, b'')
            zipf.writestr('foo.txt', 'O, for a Muse of Fire!')
        with zipfile.ZipFile(TESTFN, mode='r') as zipfr:
            self.assertEqual(zipfr.comment, b'')
        comment = b'Bravely taking to his feet, he beat a very brave retreat.'
        with zipfile.ZipFile(TESTFN, mode='w') as zipf:
            zipf.comment = comment
            zipf.writestr('foo.txt', 'O, for a Muse of Fire!')
        with zipfile.ZipFile(TESTFN, mode='r') as zipfr:
            self.assertEqual(zipf.comment, comment)
        comment2 = ''.join(['%d' % (i ** 3 % 10) for i in range((1 << 16) - 1)])
        comment2 = comment2.encode('ascii')
        with zipfile.ZipFile(TESTFN, mode='w') as zipf:
            zipf.comment = comment2
            zipf.writestr('foo.txt', 'O, for a Muse of Fire!')
        with zipfile.ZipFile(TESTFN, mode='r') as zipfr:
            self.assertEqual(zipfr.comment, comment2)
        with zipfile.ZipFile(TESTFN, mode='w') as zipf:
            with self.assertWarns(UserWarning):
                zipf.comment = comment2 + b'oops'
            zipf.writestr('foo.txt', 'O, for a Muse of Fire!')
        with zipfile.ZipFile(TESTFN, mode='r') as zipfr:
            self.assertEqual(zipfr.comment, comment2)
        with zipfile.ZipFile(TESTFN, mode='w') as zipf:
            zipf.comment = b'original comment'
            zipf.writestr('foo.txt', 'O, for a Muse of Fire!')
        with zipfile.ZipFile(TESTFN, mode='a') as zipf:
            zipf.comment = b'an updated comment'
        with zipfile.ZipFile(TESTFN, mode='r') as zipf:
            self.assertEqual(zipf.comment, b'an updated comment')
        with zipfile.ZipFile(TESTFN, mode='w') as zipf:
            zipf.comment = b"original comment that's longer"
            zipf.writestr('foo.txt', 'O, for a Muse of Fire!')
        original_zip_size = os.path.getsize(TESTFN)
        with zipfile.ZipFile(TESTFN, mode='a') as zipf:
            zipf.comment = b'shorter comment'
        self.assertTrue(original_zip_size > os.path.getsize(TESTFN))
        with zipfile.ZipFile(TESTFN, mode='r') as zipf:
            self.assertEqual(zipf.comment, b'shorter comment')

    def test_unicode_comment(self):
        if False:
            print('Hello World!')
        with zipfile.ZipFile(TESTFN, 'w', zipfile.ZIP_STORED) as zipf:
            zipf.writestr('foo.txt', 'O, for a Muse of Fire!')
            with self.assertRaises(TypeError):
                zipf.comment = 'this is an error'

    def test_change_comment_in_empty_archive(self):
        if False:
            i = 10
            return i + 15
        with zipfile.ZipFile(TESTFN, 'a', zipfile.ZIP_STORED) as zipf:
            self.assertFalse(zipf.filelist)
            zipf.comment = b'this is a comment'
        with zipfile.ZipFile(TESTFN, 'r') as zipf:
            self.assertEqual(zipf.comment, b'this is a comment')

    def test_change_comment_in_nonempty_archive(self):
        if False:
            print('Hello World!')
        with zipfile.ZipFile(TESTFN, 'w', zipfile.ZIP_STORED) as zipf:
            zipf.writestr('foo.txt', 'O, for a Muse of Fire!')
        with zipfile.ZipFile(TESTFN, 'a', zipfile.ZIP_STORED) as zipf:
            self.assertTrue(zipf.filelist)
            zipf.comment = b'this is a comment'
        with zipfile.ZipFile(TESTFN, 'r') as zipf:
            self.assertEqual(zipf.comment, b'this is a comment')

    def test_empty_zipfile(self):
        if False:
            i = 10
            return i + 15
        zipf = zipfile.ZipFile(TESTFN, mode='w')
        zipf.close()
        try:
            zipf = zipfile.ZipFile(TESTFN, mode='r')
        except zipfile.BadZipFile:
            self.fail("Unable to create empty ZIP file in 'w' mode")
        zipf = zipfile.ZipFile(TESTFN, mode='a')
        zipf.close()
        try:
            zipf = zipfile.ZipFile(TESTFN, mode='r')
        except:
            self.fail("Unable to create empty ZIP file in 'a' mode")

    def test_open_empty_file(self):
        if False:
            print('Hello World!')
        f = open(TESTFN, 'w', encoding='utf-8')
        f.close()
        self.assertRaises(zipfile.BadZipFile, zipfile.ZipFile, TESTFN, 'r')

    def test_create_zipinfo_before_1980(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(ValueError, zipfile.ZipInfo, 'seventies', (1979, 1, 1, 0, 0, 0))

    def test_create_empty_zipinfo_repr(self):
        if False:
            for i in range(10):
                print('nop')
        'Before bpo-26185, repr() on empty ZipInfo object was failing.'
        zi = zipfile.ZipInfo(filename='empty')
        self.assertEqual(repr(zi), "<ZipInfo filename='empty' file_size=0>")

    def test_create_empty_zipinfo_default_attributes(self):
        if False:
            print('Hello World!')
        'Ensure all required attributes are set.'
        zi = zipfile.ZipInfo()
        self.assertEqual(zi.orig_filename, 'NoName')
        self.assertEqual(zi.filename, 'NoName')
        self.assertEqual(zi.date_time, (1980, 1, 1, 0, 0, 0))
        self.assertEqual(zi.compress_type, zipfile.ZIP_STORED)
        self.assertEqual(zi.comment, b'')
        self.assertEqual(zi.extra, b'')
        self.assertIn(zi.create_system, (0, 3))
        self.assertEqual(zi.create_version, zipfile.DEFAULT_VERSION)
        self.assertEqual(zi.extract_version, zipfile.DEFAULT_VERSION)
        self.assertEqual(zi.reserved, 0)
        self.assertEqual(zi.flag_bits, 0)
        self.assertEqual(zi.volume, 0)
        self.assertEqual(zi.internal_attr, 0)
        self.assertEqual(zi.external_attr, 0)
        self.assertEqual(zi.file_size, 0)
        self.assertEqual(zi.compress_size, 0)

    def test_zipfile_with_short_extra_field(self):
        if False:
            i = 10
            return i + 15
        'If an extra field in the header is less than 4 bytes, skip it.'
        zipdata = b'PK\x03\x04\x14\x00\x00\x00\x00\x00\x93\x9b\xad@\x8b\x9e\xd9\xd3\x01\x00\x00\x00\x01\x00\x00\x00\x03\x00\x03\x00abc\x00\x00\x00APK\x01\x02\x14\x03\x14\x00\x00\x00\x00\x00\x93\x9b\xad@\x8b\x9e\xd9\xd3\x01\x00\x00\x00\x01\x00\x00\x00\x03\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\xa4\x81\x00\x00\x00\x00abc\x00\x00PK\x05\x06\x00\x00\x00\x00\x01\x00\x01\x003\x00\x00\x00%\x00\x00\x00\x00\x00'
        with zipfile.ZipFile(io.BytesIO(zipdata), 'r') as zipf:
            self.assertIsNone(zipf.testzip())

    def test_open_conflicting_handles(self):
        if False:
            print('Hello World!')
        msg1 = b"It's fun to charter an accountant!"
        msg2 = b'And sail the wide accountant sea'
        msg3 = b'To find, explore the funds offshore'
        with zipfile.ZipFile(TESTFN2, 'w', zipfile.ZIP_STORED) as zipf:
            with zipf.open('foo', mode='w') as w2:
                w2.write(msg1)
            with zipf.open('bar', mode='w') as w1:
                with self.assertRaises(ValueError):
                    zipf.open('handle', mode='w')
                with self.assertRaises(ValueError):
                    zipf.open('foo', mode='r')
                with self.assertRaises(ValueError):
                    zipf.writestr('str', 'abcde')
                with self.assertRaises(ValueError):
                    zipf.write(__file__, 'file')
                with self.assertRaises(ValueError):
                    zipf.close()
                w1.write(msg2)
            with zipf.open('baz', mode='w') as w2:
                w2.write(msg3)
        with zipfile.ZipFile(TESTFN2, 'r') as zipf:
            self.assertEqual(zipf.read('foo'), msg1)
            self.assertEqual(zipf.read('bar'), msg2)
            self.assertEqual(zipf.read('baz'), msg3)
            self.assertEqual(zipf.namelist(), ['foo', 'bar', 'baz'])

    def test_seek_tell(self):
        if False:
            return 10
        txt = b"Where's Bruce?"
        bloc = txt.find(b'Bruce')
        with zipfile.ZipFile(TESTFN, 'w') as zipf:
            zipf.writestr('foo.txt', txt)
        with zipfile.ZipFile(TESTFN, 'r') as zipf:
            with zipf.open('foo.txt', 'r') as fp:
                fp.seek(bloc, os.SEEK_SET)
                self.assertEqual(fp.tell(), bloc)
                fp.seek(-bloc, os.SEEK_CUR)
                self.assertEqual(fp.tell(), 0)
                fp.seek(bloc, os.SEEK_CUR)
                self.assertEqual(fp.tell(), bloc)
                self.assertEqual(fp.read(5), txt[bloc:bloc + 5])
                fp.seek(0, os.SEEK_END)
                self.assertEqual(fp.tell(), len(txt))
                fp.seek(0, os.SEEK_SET)
                self.assertEqual(fp.tell(), 0)
        data = io.BytesIO()
        with zipfile.ZipFile(data, mode='w') as zipf:
            zipf.writestr('foo.txt', txt)
        with zipfile.ZipFile(data, mode='r') as zipf:
            with zipf.open('foo.txt', 'r') as fp:
                fp.seek(bloc, os.SEEK_SET)
                self.assertEqual(fp.tell(), bloc)
                fp.seek(-bloc, os.SEEK_CUR)
                self.assertEqual(fp.tell(), 0)
                fp.seek(bloc, os.SEEK_CUR)
                self.assertEqual(fp.tell(), bloc)
                self.assertEqual(fp.read(5), txt[bloc:bloc + 5])
                fp.seek(0, os.SEEK_END)
                self.assertEqual(fp.tell(), len(txt))
                fp.seek(0, os.SEEK_SET)
                self.assertEqual(fp.tell(), 0)

    @requires_bz2()
    def test_decompress_without_3rd_party_library(self):
        if False:
            for i in range(10):
                print('nop')
        data = b'PK\x05\x06\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        zip_file = io.BytesIO(data)
        with zipfile.ZipFile(zip_file, 'w', compression=zipfile.ZIP_BZIP2) as zf:
            zf.writestr('a.txt', b'a')
        with mock.patch('zipfile.bz2', None):
            with zipfile.ZipFile(zip_file) as zf:
                self.assertRaises(RuntimeError, zf.extract, 'a.txt')

    def tearDown(self):
        if False:
            print('Hello World!')
        unlink(TESTFN)
        unlink(TESTFN2)

class AbstractBadCrcTests:

    def test_testzip_with_bad_crc(self):
        if False:
            i = 10
            return i + 15
        'Tests that files with bad CRCs return their name from testzip.'
        zipdata = self.zip_with_bad_crc
        with zipfile.ZipFile(io.BytesIO(zipdata), mode='r') as zipf:
            self.assertEqual('afile', zipf.testzip())

    def test_read_with_bad_crc(self):
        if False:
            return 10
        'Tests that files with bad CRCs raise a BadZipFile exception when read.'
        zipdata = self.zip_with_bad_crc
        with zipfile.ZipFile(io.BytesIO(zipdata), mode='r') as zipf:
            self.assertRaises(zipfile.BadZipFile, zipf.read, 'afile')
        with zipfile.ZipFile(io.BytesIO(zipdata), mode='r') as zipf:
            with zipf.open('afile', 'r') as corrupt_file:
                self.assertRaises(zipfile.BadZipFile, corrupt_file.read)
        with zipfile.ZipFile(io.BytesIO(zipdata), mode='r') as zipf:
            with zipf.open('afile', 'r') as corrupt_file:
                corrupt_file.MIN_READ_SIZE = 2
                with self.assertRaises(zipfile.BadZipFile):
                    while corrupt_file.read(2):
                        pass

class StoredBadCrcTests(AbstractBadCrcTests, unittest.TestCase):
    compression = zipfile.ZIP_STORED
    zip_with_bad_crc = b'PK\x03\x04\x14\x00\x00\x00\x00\x00 \x8b\x8a;:r\xab\xff\x0c\x00\x00\x00\x0c\x00\x00\x00\x05\x00\x00\x00afilehello,AworldPK\x01\x02\x14\x03\x14\x00\x00\x00\x00\x00 \x8b\x8a;:r\xab\xff\x0c\x00\x00\x00\x0c\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x01\x00\x00\x00\x00afilePK\x05\x06\x00\x00\x00\x00\x01\x00\x01\x003\x00\x00\x00/\x00\x00\x00\x00\x00'

@requires_zlib()
class DeflateBadCrcTests(AbstractBadCrcTests, unittest.TestCase):
    compression = zipfile.ZIP_DEFLATED
    zip_with_bad_crc = b'PK\x03\x04\x14\x00\x00\x00\x08\x00n}\x0c=FAKE\x10\x00\x00\x00n\x00\x00\x00\x05\x00\x00\x00afile\xcbH\xcd\xc9\xc9W(\xcf/\xcaI\xc9\xa0=\x13\x00PK\x01\x02\x14\x03\x14\x00\x00\x00\x08\x00n}\x0c=FAKE\x10\x00\x00\x00n\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x01\x00\x00\x00\x00afilePK\x05\x06\x00\x00\x00\x00\x01\x00\x01\x003\x00\x00\x003\x00\x00\x00\x00\x00'

@requires_bz2()
class Bzip2BadCrcTests(AbstractBadCrcTests, unittest.TestCase):
    compression = zipfile.ZIP_BZIP2
    zip_with_bad_crc = b'PK\x03\x04\x14\x03\x00\x00\x0c\x00nu\x0c=FAKE8\x00\x00\x00n\x00\x00\x00\x05\x00\x00\x00afileBZh91AY&SY\xd4\xa8\xca\x7f\x00\x00\x0f\x11\x80@\x00\x06D\x90\x80 \x00 \xa5P\xd9!\x03\x03\x13\x13\x13\x89\xa9\xa9\xc2u5:\x9f\x8b\xb9"\x9c(HjTe?\x80PK\x01\x02\x14\x03\x14\x03\x00\x00\x0c\x00nu\x0c=FAKE8\x00\x00\x00n\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00 \x80\x80\x81\x00\x00\x00\x00afilePK\x05\x06\x00\x00\x00\x00\x01\x00\x01\x003\x00\x00\x00[\x00\x00\x00\x00\x00'

@requires_lzma()
class LzmaBadCrcTests(AbstractBadCrcTests, unittest.TestCase):
    compression = zipfile.ZIP_LZMA
    zip_with_bad_crc = b'PK\x03\x04\x14\x03\x00\x00\x0e\x00nu\x0c=FAKE\x1b\x00\x00\x00n\x00\x00\x00\x05\x00\x00\x00afile\t\x04\x05\x00]\x00\x00\x00\x04\x004\x19I\xee\x8d\xe9\x17\x89:3`\tq!.8\x00PK\x01\x02\x14\x03\x14\x03\x00\x00\x0e\x00nu\x0c=FAKE\x1b\x00\x00\x00n\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00 \x80\x80\x81\x00\x00\x00\x00afilePK\x05\x06\x00\x00\x00\x00\x01\x00\x01\x003\x00\x00\x00>\x00\x00\x00\x00\x00'

class DecryptionTests(unittest.TestCase):
    """Check that ZIP decryption works. Since the library does not
    support encryption at the moment, we use a pre-generated encrypted
    ZIP file."""
    data = b'PK\x03\x04\x14\x00\x01\x00\x00\x00n\x92i.#y\xef?&\x00\x00\x00\x1a\x00\x00\x00\x08\x00\x00\x00test.txt\xfa\x10\xa0gly|\xfa-\xc5\xc0=\xf9y\x18\xe0\xa8r\xb3Z}Lg\xbc\xae\xf9|\x9b\x19\xe4\x8b\xba\xbb)\x8c\xb0\xdblPK\x01\x02\x14\x00\x14\x00\x01\x00\x00\x00n\x92i.#y\xef?&\x00\x00\x00\x1a\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x01\x00 \x00\xb6\x81\x00\x00\x00\x00test.txtPK\x05\x06\x00\x00\x00\x00\x01\x00\x01\x006\x00\x00\x00L\x00\x00\x00\x00\x00'
    data2 = b'PK\x03\x04\x14\x00\t\x00\x08\x00\xcf}38xu\xaa\xb2\x14\x00\x00\x00\x00\x02\x00\x00\x04\x00\x15\x00zeroUT\t\x00\x03\xd6\x8b\x92G\xda\x8b\x92GUx\x04\x00\xe8\x03\xe8\x03\xc7<M\xb5a\xceX\xa3Y&\x8b{oE\xd7\x9d\x8c\x98\x02\xc0PK\x07\x08xu\xaa\xb2\x14\x00\x00\x00\x00\x02\x00\x00PK\x01\x02\x17\x03\x14\x00\t\x00\x08\x00\xcf}38xu\xaa\xb2\x14\x00\x00\x00\x00\x02\x00\x00\x04\x00\r\x00\x00\x00\x00\x00\x00\x00\x00\x00\xa4\x81\x00\x00\x00\x00zeroUT\x05\x00\x03\xd6\x8b\x92GUx\x00\x00PK\x05\x06\x00\x00\x00\x00\x01\x00\x01\x00?\x00\x00\x00[\x00\x00\x00\x00\x00'
    plain = b'zipfile.py encryption test'
    plain2 = b'\x00' * 512

    def setUp(self):
        if False:
            return 10
        with open(TESTFN, 'wb') as fp:
            fp.write(self.data)
        self.zip = zipfile.ZipFile(TESTFN, 'r')
        with open(TESTFN2, 'wb') as fp:
            fp.write(self.data2)
        self.zip2 = zipfile.ZipFile(TESTFN2, 'r')

    def tearDown(self):
        if False:
            return 10
        self.zip.close()
        os.unlink(TESTFN)
        self.zip2.close()
        os.unlink(TESTFN2)

    def test_no_password(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(RuntimeError, self.zip.read, 'test.txt')
        self.assertRaises(RuntimeError, self.zip2.read, 'zero')

    def test_bad_password(self):
        if False:
            for i in range(10):
                print('nop')
        self.zip.setpassword(b'perl')
        self.assertRaises(RuntimeError, self.zip.read, 'test.txt')
        self.zip2.setpassword(b'perl')
        self.assertRaises(RuntimeError, self.zip2.read, 'zero')

    @requires_zlib()
    def test_good_password(self):
        if False:
            while True:
                i = 10
        self.zip.setpassword(b'python')
        self.assertEqual(self.zip.read('test.txt'), self.plain)
        self.zip2.setpassword(b'12345')
        self.assertEqual(self.zip2.read('zero'), self.plain2)

    def test_unicode_password(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, self.zip.setpassword, 'unicode')
        self.assertRaises(TypeError, self.zip.read, 'test.txt', 'python')
        self.assertRaises(TypeError, self.zip.open, 'test.txt', pwd='python')
        self.assertRaises(TypeError, self.zip.extract, 'test.txt', pwd='python')

    def test_seek_tell(self):
        if False:
            for i in range(10):
                print('nop')
        self.zip.setpassword(b'python')
        txt = self.plain
        test_word = b'encryption'
        bloc = txt.find(test_word)
        bloc_len = len(test_word)
        with self.zip.open('test.txt', 'r') as fp:
            fp.seek(bloc, os.SEEK_SET)
            self.assertEqual(fp.tell(), bloc)
            fp.seek(-bloc, os.SEEK_CUR)
            self.assertEqual(fp.tell(), 0)
            fp.seek(bloc, os.SEEK_CUR)
            self.assertEqual(fp.tell(), bloc)
            self.assertEqual(fp.read(bloc_len), txt[bloc:bloc + bloc_len])
            old_read_size = fp.MIN_READ_SIZE
            fp.MIN_READ_SIZE = 1
            fp._readbuffer = b''
            fp._offset = 0
            fp.seek(0, os.SEEK_SET)
            self.assertEqual(fp.tell(), 0)
            fp.seek(bloc, os.SEEK_CUR)
            self.assertEqual(fp.read(bloc_len), txt[bloc:bloc + bloc_len])
            fp.MIN_READ_SIZE = old_read_size
            fp.seek(0, os.SEEK_END)
            self.assertEqual(fp.tell(), len(txt))
            fp.seek(0, os.SEEK_SET)
            self.assertEqual(fp.tell(), 0)
            fp.read()

class AbstractTestsWithRandomBinaryFiles:

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        datacount = randint(16, 64) * 1024 + randint(1, 1024)
        cls.data = b''.join((struct.pack('<f', random() * randint(-1000, 1000)) for i in range(datacount)))

    def setUp(self):
        if False:
            return 10
        with open(TESTFN, 'wb') as fp:
            fp.write(self.data)

    def tearDown(self):
        if False:
            print('Hello World!')
        unlink(TESTFN)
        unlink(TESTFN2)

    def make_test_archive(self, f, compression):
        if False:
            while True:
                i = 10
        with zipfile.ZipFile(f, 'w', compression) as zipfp:
            zipfp.write(TESTFN, 'another.name')
            zipfp.write(TESTFN, TESTFN)

    def zip_test(self, f, compression):
        if False:
            for i in range(10):
                print('nop')
        self.make_test_archive(f, compression)
        with zipfile.ZipFile(f, 'r', compression) as zipfp:
            testdata = zipfp.read(TESTFN)
            self.assertEqual(len(testdata), len(self.data))
            self.assertEqual(testdata, self.data)
            self.assertEqual(zipfp.read('another.name'), self.data)

    def test_read(self):
        if False:
            print('Hello World!')
        for f in get_files(self):
            self.zip_test(f, self.compression)

    def zip_open_test(self, f, compression):
        if False:
            while True:
                i = 10
        self.make_test_archive(f, compression)
        with zipfile.ZipFile(f, 'r', compression) as zipfp:
            zipdata1 = []
            with zipfp.open(TESTFN) as zipopen1:
                while True:
                    read_data = zipopen1.read(256)
                    if not read_data:
                        break
                    zipdata1.append(read_data)
            zipdata2 = []
            with zipfp.open('another.name') as zipopen2:
                while True:
                    read_data = zipopen2.read(256)
                    if not read_data:
                        break
                    zipdata2.append(read_data)
            testdata1 = b''.join(zipdata1)
            self.assertEqual(len(testdata1), len(self.data))
            self.assertEqual(testdata1, self.data)
            testdata2 = b''.join(zipdata2)
            self.assertEqual(len(testdata2), len(self.data))
            self.assertEqual(testdata2, self.data)

    def test_open(self):
        if False:
            i = 10
            return i + 15
        for f in get_files(self):
            self.zip_open_test(f, self.compression)

    def zip_random_open_test(self, f, compression):
        if False:
            print('Hello World!')
        self.make_test_archive(f, compression)
        with zipfile.ZipFile(f, 'r', compression) as zipfp:
            zipdata1 = []
            with zipfp.open(TESTFN) as zipopen1:
                while True:
                    read_data = zipopen1.read(randint(1, 1024))
                    if not read_data:
                        break
                    zipdata1.append(read_data)
            testdata = b''.join(zipdata1)
            self.assertEqual(len(testdata), len(self.data))
            self.assertEqual(testdata, self.data)

    def test_random_open(self):
        if False:
            for i in range(10):
                print('nop')
        for f in get_files(self):
            self.zip_random_open_test(f, self.compression)

class StoredTestsWithRandomBinaryFiles(AbstractTestsWithRandomBinaryFiles, unittest.TestCase):
    compression = zipfile.ZIP_STORED

@requires_zlib()
class DeflateTestsWithRandomBinaryFiles(AbstractTestsWithRandomBinaryFiles, unittest.TestCase):
    compression = zipfile.ZIP_DEFLATED

@requires_bz2()
class Bzip2TestsWithRandomBinaryFiles(AbstractTestsWithRandomBinaryFiles, unittest.TestCase):
    compression = zipfile.ZIP_BZIP2

@requires_lzma()
class LzmaTestsWithRandomBinaryFiles(AbstractTestsWithRandomBinaryFiles, unittest.TestCase):
    compression = zipfile.ZIP_LZMA

class Tellable:

    def __init__(self, fp):
        if False:
            return 10
        self.fp = fp
        self.offset = 0

    def write(self, data):
        if False:
            for i in range(10):
                print('nop')
        n = self.fp.write(data)
        self.offset += n
        return n

    def tell(self):
        if False:
            while True:
                i = 10
        return self.offset

    def flush(self):
        if False:
            for i in range(10):
                print('nop')
        self.fp.flush()

class Unseekable:

    def __init__(self, fp):
        if False:
            while True:
                i = 10
        self.fp = fp

    def write(self, data):
        if False:
            for i in range(10):
                print('nop')
        return self.fp.write(data)

    def flush(self):
        if False:
            while True:
                i = 10
        self.fp.flush()

class UnseekableTests(unittest.TestCase):

    def test_writestr(self):
        if False:
            for i in range(10):
                print('nop')
        for wrapper in (lambda f: f, Tellable, Unseekable):
            with self.subTest(wrapper=wrapper):
                f = io.BytesIO()
                f.write(b'abc')
                bf = io.BufferedWriter(f)
                with zipfile.ZipFile(wrapper(bf), 'w', zipfile.ZIP_STORED) as zipfp:
                    zipfp.writestr('ones', b'111')
                    zipfp.writestr('twos', b'222')
                self.assertEqual(f.getvalue()[:5], b'abcPK')
                with zipfile.ZipFile(f, mode='r') as zipf:
                    with zipf.open('ones') as zopen:
                        self.assertEqual(zopen.read(), b'111')
                    with zipf.open('twos') as zopen:
                        self.assertEqual(zopen.read(), b'222')

    def test_write(self):
        if False:
            print('Hello World!')
        for wrapper in (lambda f: f, Tellable, Unseekable):
            with self.subTest(wrapper=wrapper):
                f = io.BytesIO()
                f.write(b'abc')
                bf = io.BufferedWriter(f)
                with zipfile.ZipFile(wrapper(bf), 'w', zipfile.ZIP_STORED) as zipfp:
                    self.addCleanup(unlink, TESTFN)
                    with open(TESTFN, 'wb') as f2:
                        f2.write(b'111')
                    zipfp.write(TESTFN, 'ones')
                    with open(TESTFN, 'wb') as f2:
                        f2.write(b'222')
                    zipfp.write(TESTFN, 'twos')
                self.assertEqual(f.getvalue()[:5], b'abcPK')
                with zipfile.ZipFile(f, mode='r') as zipf:
                    with zipf.open('ones') as zopen:
                        self.assertEqual(zopen.read(), b'111')
                    with zipf.open('twos') as zopen:
                        self.assertEqual(zopen.read(), b'222')

    def test_open_write(self):
        if False:
            i = 10
            return i + 15
        for wrapper in (lambda f: f, Tellable, Unseekable):
            with self.subTest(wrapper=wrapper):
                f = io.BytesIO()
                f.write(b'abc')
                bf = io.BufferedWriter(f)
                with zipfile.ZipFile(wrapper(bf), 'w', zipfile.ZIP_STORED) as zipf:
                    with zipf.open('ones', 'w') as zopen:
                        zopen.write(b'111')
                    with zipf.open('twos', 'w') as zopen:
                        zopen.write(b'222')
                self.assertEqual(f.getvalue()[:5], b'abcPK')
                with zipfile.ZipFile(f) as zipf:
                    self.assertEqual(zipf.read('ones'), b'111')
                    self.assertEqual(zipf.read('twos'), b'222')

@requires_zlib()
class TestsWithMultipleOpens(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        cls.data1 = b'111' + randbytes(10000)
        cls.data2 = b'222' + randbytes(10000)

    def make_test_archive(self, f):
        if False:
            print('Hello World!')
        with zipfile.ZipFile(f, 'w', zipfile.ZIP_DEFLATED) as zipfp:
            zipfp.writestr('ones', self.data1)
            zipfp.writestr('twos', self.data2)

    def test_same_file(self):
        if False:
            i = 10
            return i + 15
        for f in get_files(self):
            self.make_test_archive(f)
            with zipfile.ZipFile(f, mode='r') as zipf:
                with zipf.open('ones') as zopen1, zipf.open('ones') as zopen2:
                    data1 = zopen1.read(500)
                    data2 = zopen2.read(500)
                    data1 += zopen1.read()
                    data2 += zopen2.read()
                self.assertEqual(data1, data2)
                self.assertEqual(data1, self.data1)

    def test_different_file(self):
        if False:
            print('Hello World!')
        for f in get_files(self):
            self.make_test_archive(f)
            with zipfile.ZipFile(f, mode='r') as zipf:
                with zipf.open('ones') as zopen1, zipf.open('twos') as zopen2:
                    data1 = zopen1.read(500)
                    data2 = zopen2.read(500)
                    data1 += zopen1.read()
                    data2 += zopen2.read()
                self.assertEqual(data1, self.data1)
                self.assertEqual(data2, self.data2)

    def test_interleaved(self):
        if False:
            i = 10
            return i + 15
        for f in get_files(self):
            self.make_test_archive(f)
            with zipfile.ZipFile(f, mode='r') as zipf:
                with zipf.open('ones') as zopen1:
                    data1 = zopen1.read(500)
                    with zipf.open('twos') as zopen2:
                        data2 = zopen2.read(500)
                        data1 += zopen1.read()
                        data2 += zopen2.read()
                self.assertEqual(data1, self.data1)
                self.assertEqual(data2, self.data2)

    def test_read_after_close(self):
        if False:
            while True:
                i = 10
        for f in get_files(self):
            self.make_test_archive(f)
            with contextlib.ExitStack() as stack:
                with zipfile.ZipFile(f, 'r') as zipf:
                    zopen1 = stack.enter_context(zipf.open('ones'))
                    zopen2 = stack.enter_context(zipf.open('twos'))
                data1 = zopen1.read(500)
                data2 = zopen2.read(500)
                data1 += zopen1.read()
                data2 += zopen2.read()
            self.assertEqual(data1, self.data1)
            self.assertEqual(data2, self.data2)

    def test_read_after_write(self):
        if False:
            return 10
        for f in get_files(self):
            with zipfile.ZipFile(f, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.writestr('ones', self.data1)
                zipf.writestr('twos', self.data2)
                with zipf.open('ones') as zopen1:
                    data1 = zopen1.read(500)
            self.assertEqual(data1, self.data1[:500])
            with zipfile.ZipFile(f, 'r') as zipf:
                data1 = zipf.read('ones')
                data2 = zipf.read('twos')
            self.assertEqual(data1, self.data1)
            self.assertEqual(data2, self.data2)

    def test_write_after_read(self):
        if False:
            for i in range(10):
                print('nop')
        for f in get_files(self):
            with zipfile.ZipFile(f, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.writestr('ones', self.data1)
                with zipf.open('ones') as zopen1:
                    zopen1.read(500)
                    zipf.writestr('twos', self.data2)
            with zipfile.ZipFile(f, 'r') as zipf:
                data1 = zipf.read('ones')
                data2 = zipf.read('twos')
            self.assertEqual(data1, self.data1)
            self.assertEqual(data2, self.data2)

    def test_many_opens(self):
        if False:
            i = 10
            return i + 15
        self.make_test_archive(TESTFN2)
        with zipfile.ZipFile(TESTFN2, mode='r') as zipf:
            for x in range(100):
                zipf.read('ones')
                with zipf.open('ones') as zopen1:
                    pass
        with open(os.devnull, 'rb') as f:
            self.assertLess(f.fileno(), 100)

    def test_write_while_reading(self):
        if False:
            return 10
        with zipfile.ZipFile(TESTFN2, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr('ones', self.data1)
        with zipfile.ZipFile(TESTFN2, 'a', zipfile.ZIP_DEFLATED) as zipf:
            with zipf.open('ones', 'r') as r1:
                data1 = r1.read(500)
                with zipf.open('twos', 'w') as w1:
                    w1.write(self.data2)
                data1 += r1.read()
        self.assertEqual(data1, self.data1)
        with zipfile.ZipFile(TESTFN2) as zipf:
            self.assertEqual(zipf.read('twos'), self.data2)

    def tearDown(self):
        if False:
            return 10
        unlink(TESTFN2)

class TestWithDirectory(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        os.mkdir(TESTFN2)

    def test_extract_dir(self):
        if False:
            print('Hello World!')
        with zipfile.ZipFile(findfile('zipdir.zip')) as zipf:
            zipf.extractall(TESTFN2)
        self.assertTrue(os.path.isdir(os.path.join(TESTFN2, 'a')))
        self.assertTrue(os.path.isdir(os.path.join(TESTFN2, 'a', 'b')))
        self.assertTrue(os.path.exists(os.path.join(TESTFN2, 'a', 'b', 'c')))

    def test_bug_6050(self):
        if False:
            return 10
        os.mkdir(os.path.join(TESTFN2, 'a'))
        self.test_extract_dir()

    def test_write_dir(self):
        if False:
            print('Hello World!')
        dirpath = os.path.join(TESTFN2, 'x')
        os.mkdir(dirpath)
        mode = os.stat(dirpath).st_mode & 65535
        with zipfile.ZipFile(TESTFN, 'w') as zipf:
            zipf.write(dirpath)
            zinfo = zipf.filelist[0]
            self.assertTrue(zinfo.filename.endswith('/x/'))
            self.assertEqual(zinfo.external_attr, mode << 16 | 16)
            zipf.write(dirpath, 'y')
            zinfo = zipf.filelist[1]
            self.assertTrue(zinfo.filename, 'y/')
            self.assertEqual(zinfo.external_attr, mode << 16 | 16)
        with zipfile.ZipFile(TESTFN, 'r') as zipf:
            zinfo = zipf.filelist[0]
            self.assertTrue(zinfo.filename.endswith('/x/'))
            self.assertEqual(zinfo.external_attr, mode << 16 | 16)
            zinfo = zipf.filelist[1]
            self.assertTrue(zinfo.filename, 'y/')
            self.assertEqual(zinfo.external_attr, mode << 16 | 16)
            target = os.path.join(TESTFN2, 'target')
            os.mkdir(target)
            zipf.extractall(target)
            self.assertTrue(os.path.isdir(os.path.join(target, 'y')))
            self.assertEqual(len(os.listdir(target)), 2)

    def test_writestr_dir(self):
        if False:
            for i in range(10):
                print('nop')
        os.mkdir(os.path.join(TESTFN2, 'x'))
        with zipfile.ZipFile(TESTFN, 'w') as zipf:
            zipf.writestr('x/', b'')
            zinfo = zipf.filelist[0]
            self.assertEqual(zinfo.filename, 'x/')
            self.assertEqual(zinfo.external_attr, 16893 << 16 | 16)
        with zipfile.ZipFile(TESTFN, 'r') as zipf:
            zinfo = zipf.filelist[0]
            self.assertTrue(zinfo.filename.endswith('x/'))
            self.assertEqual(zinfo.external_attr, 16893 << 16 | 16)
            target = os.path.join(TESTFN2, 'target')
            os.mkdir(target)
            zipf.extractall(target)
            self.assertTrue(os.path.isdir(os.path.join(target, 'x')))
            self.assertEqual(os.listdir(target), ['x'])

    def tearDown(self):
        if False:
            while True:
                i = 10
        rmtree(TESTFN2)
        if os.path.exists(TESTFN):
            unlink(TESTFN)

class ZipInfoTests(unittest.TestCase):

    def test_from_file(self):
        if False:
            print('Hello World!')
        zi = zipfile.ZipInfo.from_file(__file__)
        self.assertEqual(posixpath.basename(zi.filename), 'test_zipfile.py')
        self.assertFalse(zi.is_dir())
        self.assertEqual(zi.file_size, os.path.getsize(__file__))

    def test_from_file_pathlike(self):
        if False:
            print('Hello World!')
        zi = zipfile.ZipInfo.from_file(pathlib.Path(__file__))
        self.assertEqual(posixpath.basename(zi.filename), 'test_zipfile.py')
        self.assertFalse(zi.is_dir())
        self.assertEqual(zi.file_size, os.path.getsize(__file__))

    def test_from_file_bytes(self):
        if False:
            while True:
                i = 10
        zi = zipfile.ZipInfo.from_file(os.fsencode(__file__), 'test')
        self.assertEqual(posixpath.basename(zi.filename), 'test')
        self.assertFalse(zi.is_dir())
        self.assertEqual(zi.file_size, os.path.getsize(__file__))

    def test_from_file_fileno(self):
        if False:
            print('Hello World!')
        with open(__file__, 'rb') as f:
            zi = zipfile.ZipInfo.from_file(f.fileno(), 'test')
            self.assertEqual(posixpath.basename(zi.filename), 'test')
            self.assertFalse(zi.is_dir())
            self.assertEqual(zi.file_size, os.path.getsize(__file__))

    def test_from_dir(self):
        if False:
            i = 10
            return i + 15
        dirpath = os.path.dirname(os.path.abspath(__file__))
        zi = zipfile.ZipInfo.from_file(dirpath, 'stdlib_tests')
        self.assertEqual(zi.filename, 'stdlib_tests/')
        self.assertTrue(zi.is_dir())
        self.assertEqual(zi.compress_type, zipfile.ZIP_STORED)
        self.assertEqual(zi.file_size, 0)

class CommandLineTest(unittest.TestCase):

    def zipfilecmd(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        (rc, out, err) = script_helper.assert_python_ok('-m', 'zipfile', *args, **kwargs)
        return out.replace(os.linesep.encode(), b'\n')

    def zipfilecmd_failure(self, *args):
        if False:
            i = 10
            return i + 15
        return script_helper.assert_python_failure('-m', 'zipfile', *args)

    def test_bad_use(self):
        if False:
            while True:
                i = 10
        (rc, out, err) = self.zipfilecmd_failure()
        self.assertEqual(out, b'')
        self.assertIn(b'usage', err.lower())
        self.assertIn(b'error', err.lower())
        self.assertIn(b'required', err.lower())
        (rc, out, err) = self.zipfilecmd_failure('-l', '')
        self.assertEqual(out, b'')
        self.assertNotEqual(err.strip(), b'')

    def test_test_command(self):
        if False:
            while True:
                i = 10
        zip_name = findfile('zipdir.zip')
        for opt in ('-t', '--test'):
            out = self.zipfilecmd(opt, zip_name)
            self.assertEqual(out.rstrip(), b'Done testing')
        zip_name = findfile('testtar.tar')
        (rc, out, err) = self.zipfilecmd_failure('-t', zip_name)
        self.assertEqual(out, b'')

    def test_list_command(self):
        if False:
            i = 10
            return i + 15
        zip_name = findfile('zipdir.zip')
        t = io.StringIO()
        with zipfile.ZipFile(zip_name, 'r') as tf:
            tf.printdir(t)
        expected = t.getvalue().encode('ascii', 'backslashreplace')
        for opt in ('-l', '--list'):
            out = self.zipfilecmd(opt, zip_name, PYTHONIOENCODING='ascii:backslashreplace')
            self.assertEqual(out, expected)

    @requires_zlib()
    def test_create_command(self):
        if False:
            for i in range(10):
                print('nop')
        self.addCleanup(unlink, TESTFN)
        with open(TESTFN, 'w', encoding='utf-8') as f:
            f.write('test 1')
        os.mkdir(TESTFNDIR)
        self.addCleanup(rmtree, TESTFNDIR)
        with open(os.path.join(TESTFNDIR, 'file.txt'), 'w', encoding='utf-8') as f:
            f.write('test 2')
        files = [TESTFN, TESTFNDIR]
        namelist = [TESTFN, TESTFNDIR + '/', TESTFNDIR + '/file.txt']
        for opt in ('-c', '--create'):
            try:
                out = self.zipfilecmd(opt, TESTFN2, *files)
                self.assertEqual(out, b'')
                with zipfile.ZipFile(TESTFN2) as zf:
                    self.assertEqual(zf.namelist(), namelist)
                    self.assertEqual(zf.read(namelist[0]), b'test 1')
                    self.assertEqual(zf.read(namelist[2]), b'test 2')
            finally:
                unlink(TESTFN2)

    def test_extract_command(self):
        if False:
            i = 10
            return i + 15
        zip_name = findfile('zipdir.zip')
        for opt in ('-e', '--extract'):
            with temp_dir() as extdir:
                out = self.zipfilecmd(opt, zip_name, extdir)
                self.assertEqual(out, b'')
                with zipfile.ZipFile(zip_name) as zf:
                    for zi in zf.infolist():
                        path = os.path.join(extdir, zi.filename.replace('/', os.sep))
                        if zi.is_dir():
                            self.assertTrue(os.path.isdir(path))
                        else:
                            self.assertTrue(os.path.isfile(path))
                            with open(path, 'rb') as f:
                                self.assertEqual(f.read(), zf.read(zi))

class TestExecutablePrependedZip(unittest.TestCase):
    """Test our ability to open zip files with an executable prepended."""

    def setUp(self):
        if False:
            print('Hello World!')
        self.exe_zip = findfile('exe_with_zip', subdir='ziptestdata')
        self.exe_zip64 = findfile('exe_with_z64', subdir='ziptestdata')

    def _test_zip_works(self, name):
        if False:
            while True:
                i = 10
        self.assertTrue(zipfile.is_zipfile(name), f'is_zipfile failed on {name}')
        with zipfile.ZipFile(name) as zipfp:
            for n in zipfp.namelist():
                data = zipfp.read(n)
                self.assertIn(b'FAVORITE_NUMBER', data)

    def test_read_zip_with_exe_prepended(self):
        if False:
            print('Hello World!')
        self._test_zip_works(self.exe_zip)

    def test_read_zip64_with_exe_prepended(self):
        if False:
            i = 10
            return i + 15
        self._test_zip_works(self.exe_zip64)

    @unittest.skipUnless(sys.executable, 'sys.executable required.')
    @unittest.skipUnless(os.access('/bin/bash', os.X_OK), 'Test relies on #!/bin/bash working.')
    def test_execute_zip2(self):
        if False:
            print('Hello World!')
        output = subprocess.check_output([self.exe_zip, sys.executable])
        self.assertIn(b'number in executable: 5', output)

    @unittest.skipUnless(sys.executable, 'sys.executable required.')
    @unittest.skipUnless(os.access('/bin/bash', os.X_OK), 'Test relies on #!/bin/bash working.')
    def test_execute_zip64(self):
        if False:
            print('Hello World!')
        output = subprocess.check_output([self.exe_zip64, sys.executable])
        self.assertIn(b'number in executable: 5', output)
consume = tuple

class jaraco:

    class itertools:

        class Counter:

            def __init__(self, i):
                if False:
                    print('Hello World!')
                self.count = 0
                self._orig_iter = iter(i)

            def __iter__(self):
                if False:
                    while True:
                        i = 10
                return self

            def __next__(self):
                if False:
                    return 10
                result = next(self._orig_iter)
                self.count += 1
                return result

def add_dirs(zf):
    if False:
        return 10
    '\n    Given a writable zip file zf, inject directory entries for\n    any directories implied by the presence of children.\n    '
    for name in zipfile.CompleteDirs._implied_dirs(zf.namelist()):
        zf.writestr(name, b'')
    return zf

def build_alpharep_fixture():
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a zip file with this structure:\n\n    .\n    ├── a.txt\n    ├── b\n    │   ├── c.txt\n    │   ├── d\n    │   │   └── e.txt\n    │   └── f.txt\n    └── g\n        └── h\n            └── i.txt\n\n    This fixture has the following key characteristics:\n\n    - a file at the root (a)\n    - a file two levels deep (b/d/e)\n    - multiple files in a directory (b/c, b/f)\n    - a directory containing only a directory (g/h)\n\n    "alpha" because it uses alphabet\n    "rep" because it\'s a representative example\n    '
    data = io.BytesIO()
    zf = zipfile.ZipFile(data, 'w')
    zf.writestr('a.txt', b'content of a')
    zf.writestr('b/c.txt', b'content of c')
    zf.writestr('b/d/e.txt', b'content of e')
    zf.writestr('b/f.txt', b'content of f')
    zf.writestr('g/h/i.txt', b'content of i')
    zf.filename = 'alpharep.zip'
    return zf

def pass_alpharep(meth):
    if False:
        while True:
            i = 10
    '\n    Given a method, wrap it in a for loop that invokes method\n    with each subtest.\n    '

    @functools.wraps(meth)
    def wrapper(self):
        if False:
            return 10
        for alpharep in self.zipfile_alpharep():
            meth(self, alpharep=alpharep)
    return wrapper

class TestPath(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.fixtures = contextlib.ExitStack()
        self.addCleanup(self.fixtures.close)

    def zipfile_alpharep(self):
        if False:
            for i in range(10):
                print('nop')
        with self.subTest():
            yield build_alpharep_fixture()
        with self.subTest():
            yield add_dirs(build_alpharep_fixture())

    def zipfile_ondisk(self, alpharep):
        if False:
            for i in range(10):
                print('nop')
        tmpdir = pathlib.Path(self.fixtures.enter_context(temp_dir()))
        buffer = alpharep.fp
        alpharep.close()
        path = tmpdir / alpharep.filename
        with path.open('wb') as strm:
            strm.write(buffer.getvalue())
        return path

    @pass_alpharep
    def test_iterdir_and_types(self, alpharep):
        if False:
            for i in range(10):
                print('nop')
        root = zipfile.Path(alpharep)
        assert root.is_dir()
        (a, b, g) = root.iterdir()
        assert a.is_file()
        assert b.is_dir()
        assert g.is_dir()
        (c, f, d) = b.iterdir()
        assert c.is_file() and f.is_file()
        (e,) = d.iterdir()
        assert e.is_file()
        (h,) = g.iterdir()
        (i,) = h.iterdir()
        assert i.is_file()

    @pass_alpharep
    def test_is_file_missing(self, alpharep):
        if False:
            print('Hello World!')
        root = zipfile.Path(alpharep)
        assert not root.joinpath('missing.txt').is_file()

    @pass_alpharep
    def test_iterdir_on_file(self, alpharep):
        if False:
            for i in range(10):
                print('nop')
        root = zipfile.Path(alpharep)
        (a, b, g) = root.iterdir()
        with self.assertRaises(ValueError):
            a.iterdir()

    @pass_alpharep
    def test_subdir_is_dir(self, alpharep):
        if False:
            i = 10
            return i + 15
        root = zipfile.Path(alpharep)
        assert (root / 'b').is_dir()
        assert (root / 'b/').is_dir()
        assert (root / 'g').is_dir()
        assert (root / 'g/').is_dir()

    @pass_alpharep
    def test_open(self, alpharep):
        if False:
            for i in range(10):
                print('nop')
        root = zipfile.Path(alpharep)
        (a, b, g) = root.iterdir()
        with a.open(encoding='utf-8') as strm:
            data = strm.read()
        assert data == 'content of a'

    def test_open_write(self):
        if False:
            while True:
                i = 10
        '\n        If the zipfile is open for write, it should be possible to\n        write bytes or text to it.\n        '
        zf = zipfile.Path(zipfile.ZipFile(io.BytesIO(), mode='w'))
        with zf.joinpath('file.bin').open('wb') as strm:
            strm.write(b'binary contents')
        with zf.joinpath('file.txt').open('w', encoding='utf-8') as strm:
            strm.write('text file')

    def test_open_extant_directory(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Attempting to open a directory raises IsADirectoryError.\n        '
        zf = zipfile.Path(add_dirs(build_alpharep_fixture()))
        with self.assertRaises(IsADirectoryError):
            zf.joinpath('b').open()

    @pass_alpharep
    def test_open_binary_invalid_args(self, alpharep):
        if False:
            for i in range(10):
                print('nop')
        root = zipfile.Path(alpharep)
        with self.assertRaises(ValueError):
            root.joinpath('a.txt').open('rb', encoding='utf-8')
        with self.assertRaises(ValueError):
            root.joinpath('a.txt').open('rb', 'utf-8')

    def test_open_missing_directory(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Attempting to open a missing directory raises FileNotFoundError.\n        '
        zf = zipfile.Path(add_dirs(build_alpharep_fixture()))
        with self.assertRaises(FileNotFoundError):
            zf.joinpath('z').open()

    @pass_alpharep
    def test_read(self, alpharep):
        if False:
            print('Hello World!')
        root = zipfile.Path(alpharep)
        (a, b, g) = root.iterdir()
        assert a.read_text(encoding='utf-8') == 'content of a'
        assert a.read_bytes() == b'content of a'

    @pass_alpharep
    def test_joinpath(self, alpharep):
        if False:
            while True:
                i = 10
        root = zipfile.Path(alpharep)
        a = root.joinpath('a.txt')
        assert a.is_file()
        e = root.joinpath('b').joinpath('d').joinpath('e.txt')
        assert e.read_text(encoding='utf-8') == 'content of e'

    @pass_alpharep
    def test_joinpath_multiple(self, alpharep):
        if False:
            for i in range(10):
                print('nop')
        root = zipfile.Path(alpharep)
        e = root.joinpath('b', 'd', 'e.txt')
        assert e.read_text(encoding='utf-8') == 'content of e'

    @pass_alpharep
    def test_traverse_truediv(self, alpharep):
        if False:
            while True:
                i = 10
        root = zipfile.Path(alpharep)
        a = root / 'a.txt'
        assert a.is_file()
        e = root / 'b' / 'd' / 'e.txt'
        assert e.read_text(encoding='utf-8') == 'content of e'

    @pass_alpharep
    def test_traverse_simplediv(self, alpharep):
        if False:
            while True:
                i = 10
        '\n        Disable the __future__.division when testing traversal.\n        '
        code = compile(source="zipfile.Path(alpharep) / 'a'", filename='(test)', mode='eval', dont_inherit=True)
        eval(code)

    @pass_alpharep
    def test_pathlike_construction(self, alpharep):
        if False:
            return 10
        '\n        zipfile.Path should be constructable from a path-like object\n        '
        zipfile_ondisk = self.zipfile_ondisk(alpharep)
        pathlike = pathlib.Path(str(zipfile_ondisk))
        zipfile.Path(pathlike)

    @pass_alpharep
    def test_traverse_pathlike(self, alpharep):
        if False:
            i = 10
            return i + 15
        root = zipfile.Path(alpharep)
        root / pathlib.Path('a')

    @pass_alpharep
    def test_parent(self, alpharep):
        if False:
            for i in range(10):
                print('nop')
        root = zipfile.Path(alpharep)
        assert (root / 'a').parent.at == ''
        assert (root / 'a' / 'b').parent.at == 'a/'

    @pass_alpharep
    def test_dir_parent(self, alpharep):
        if False:
            print('Hello World!')
        root = zipfile.Path(alpharep)
        assert (root / 'b').parent.at == ''
        assert (root / 'b/').parent.at == ''

    @pass_alpharep
    def test_missing_dir_parent(self, alpharep):
        if False:
            while True:
                i = 10
        root = zipfile.Path(alpharep)
        assert (root / 'missing dir/').parent.at == ''

    @pass_alpharep
    def test_mutability(self, alpharep):
        if False:
            for i in range(10):
                print('nop')
        '\n        If the underlying zipfile is changed, the Path object should\n        reflect that change.\n        '
        root = zipfile.Path(alpharep)
        (a, b, g) = root.iterdir()
        alpharep.writestr('foo.txt', 'foo')
        alpharep.writestr('bar/baz.txt', 'baz')
        assert any((child.name == 'foo.txt' for child in root.iterdir()))
        assert (root / 'foo.txt').read_text(encoding='utf-8') == 'foo'
        (baz,) = (root / 'bar').iterdir()
        assert baz.read_text(encoding='utf-8') == 'baz'
    HUGE_ZIPFILE_NUM_ENTRIES = 2 ** 13

    def huge_zipfile(self):
        if False:
            for i in range(10):
                print('nop')
        'Create a read-only zipfile with a huge number of entries entries.'
        strm = io.BytesIO()
        zf = zipfile.ZipFile(strm, 'w')
        for entry in map(str, range(self.HUGE_ZIPFILE_NUM_ENTRIES)):
            zf.writestr(entry, entry)
        zf.mode = 'r'
        return zf

    def test_joinpath_constant_time(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Ensure joinpath on items in zipfile is linear time.\n        '
        root = zipfile.Path(self.huge_zipfile())
        entries = jaraco.itertools.Counter(root.iterdir())
        for entry in entries:
            entry.joinpath('suffix')
        assert entries.count == self.HUGE_ZIPFILE_NUM_ENTRIES

    def test_implied_dirs_performance(self):
        if False:
            print('Hello World!')
        data = ['/'.join(string.ascii_lowercase + str(n)) for n in range(10000)]
        zipfile.CompleteDirs._implied_dirs(data)

    @pass_alpharep
    def test_read_does_not_close(self, alpharep):
        if False:
            i = 10
            return i + 15
        alpharep = self.zipfile_ondisk(alpharep)
        with zipfile.ZipFile(alpharep) as file:
            for rep in range(2):
                zipfile.Path(file, 'a.txt').read_text(encoding='utf-8')

    @pass_alpharep
    def test_subclass(self, alpharep):
        if False:
            for i in range(10):
                print('nop')

        class Subclass(zipfile.Path):
            pass
        root = Subclass(alpharep)
        assert isinstance(root / 'b', Subclass)

    @pass_alpharep
    def test_filename(self, alpharep):
        if False:
            i = 10
            return i + 15
        root = zipfile.Path(alpharep)
        assert root.filename == pathlib.Path('alpharep.zip')

    @pass_alpharep
    def test_root_name(self, alpharep):
        if False:
            for i in range(10):
                print('nop')
        '\n        The name of the root should be the name of the zipfile\n        '
        root = zipfile.Path(alpharep)
        assert root.name == 'alpharep.zip' == root.filename.name

    @pass_alpharep
    def test_root_parent(self, alpharep):
        if False:
            for i in range(10):
                print('nop')
        root = zipfile.Path(alpharep)
        assert root.parent == pathlib.Path('.')
        root.root.filename = 'foo/bar.zip'
        assert root.parent == pathlib.Path('foo')

    @pass_alpharep
    def test_root_unnamed(self, alpharep):
        if False:
            while True:
                i = 10
        '\n        It is an error to attempt to get the name\n        or parent of an unnamed zipfile.\n        '
        alpharep.filename = None
        root = zipfile.Path(alpharep)
        with self.assertRaises(TypeError):
            root.name
        with self.assertRaises(TypeError):
            root.parent
        sub = root / 'b'
        assert sub.name == 'b'
        assert sub.parent

    @pass_alpharep
    def test_inheritance(self, alpharep):
        if False:
            i = 10
            return i + 15
        cls = type('PathChild', (zipfile.Path,), {})
        for alpharep in self.zipfile_alpharep():
            file = cls(alpharep).joinpath('some dir').parent
            assert isinstance(file, cls)
if __name__ == '__main__':
    unittest.main()