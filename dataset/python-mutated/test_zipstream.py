"""
Tests for L{twisted.python.zipstream}
"""
import random
import struct
import zipfile
from hashlib import md5
from twisted.python import filepath, zipstream
from twisted.trial import unittest

class FileEntryMixin:
    """
    File entry classes should behave as file-like objects
    """

    def getFileEntry(self, contents):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return an appropriate zip file entry\n        '
        filename = self.mktemp()
        with zipfile.ZipFile(filename, 'w', self.compression) as z:
            z.writestr('content', contents)
        z = zipstream.ChunkingZipFile(filename, 'r')
        return z.readfile('content')

    def test_isatty(self):
        if False:
            print('Hello World!')
        '\n        zip files should not be ttys, so isatty() should be false\n        '
        with self.getFileEntry('') as fileEntry:
            self.assertFalse(fileEntry.isatty())

    def test_closed(self):
        if False:
            i = 10
            return i + 15
        '\n        The C{closed} attribute should reflect whether C{close()} has been\n        called.\n        '
        with self.getFileEntry('') as fileEntry:
            self.assertFalse(fileEntry.closed)
        self.assertTrue(fileEntry.closed)

    def test_readline(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        C{readline()} should mirror L{file.readline} and return up to a single\n        delimiter.\n        '
        with self.getFileEntry(b'hoho\nho') as fileEntry:
            self.assertEqual(fileEntry.readline(), b'hoho\n')
            self.assertEqual(fileEntry.readline(), b'ho')
            self.assertEqual(fileEntry.readline(), b'')

    def test_next(self):
        if False:
            while True:
                i = 10
        '\n        Zip file entries should implement the iterator protocol as files do.\n        '
        with self.getFileEntry(b'ho\nhoho') as fileEntry:
            self.assertEqual(fileEntry.next(), b'ho\n')
            self.assertEqual(fileEntry.next(), b'hoho')
            self.assertRaises(StopIteration, fileEntry.next)

    def test_readlines(self):
        if False:
            while True:
                i = 10
        '\n        C{readlines()} should return a list of all the lines.\n        '
        with self.getFileEntry(b'ho\nho\nho') as fileEntry:
            self.assertEqual(fileEntry.readlines(), [b'ho\n', b'ho\n', b'ho'])

    def test_iteration(self):
        if False:
            while True:
                i = 10
        '\n        C{__iter__()} and C{xreadlines()} should return C{self}.\n        '
        with self.getFileEntry('') as fileEntry:
            self.assertIs(iter(fileEntry), fileEntry)
            self.assertIs(fileEntry.xreadlines(), fileEntry)

    def test_readWhole(self):
        if False:
            print('Hello World!')
        '\n        C{.read()} should read the entire file.\n        '
        contents = b'Hello, world!'
        with self.getFileEntry(contents) as entry:
            self.assertEqual(entry.read(), contents)

    def test_readPartial(self):
        if False:
            return 10
        '\n        C{.read(num)} should read num bytes from the file.\n        '
        contents = '0123456789'
        with self.getFileEntry(contents) as entry:
            one = entry.read(4)
            two = entry.read(200)
        self.assertEqual(one, b'0123')
        self.assertEqual(two, b'456789')

    def test_tell(self):
        if False:
            return 10
        '\n        C{.tell()} should return the number of bytes that have been read so\n        far.\n        '
        contents = 'x' * 100
        with self.getFileEntry(contents) as entry:
            entry.read(2)
            self.assertEqual(entry.tell(), 2)
            entry.read(4)
            self.assertEqual(entry.tell(), 6)

class DeflatedZipFileEntryTests(FileEntryMixin, unittest.TestCase):
    """
    DeflatedZipFileEntry should be file-like
    """
    compression = zipfile.ZIP_DEFLATED

class ZipFileEntryTests(FileEntryMixin, unittest.TestCase):
    """
    ZipFileEntry should be file-like
    """
    compression = zipfile.ZIP_STORED

class ZipstreamTests(unittest.TestCase):
    """
    Tests for twisted.python.zipstream
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        '\n        Creates junk data that can be compressed and a test directory for any\n        files that will be created\n        '
        self.testdir = filepath.FilePath(self.mktemp())
        self.testdir.makedirs()
        self.unzipdir = self.testdir.child('unzipped')
        self.unzipdir.makedirs()

    def makeZipFile(self, contents, directory=''):
        if False:
            while True:
                i = 10
        '\n        Makes a zip file archive containing len(contents) files.  Contents\n        should be a list of strings, each string being the content of one file.\n        '
        zpfilename = self.testdir.child('zipfile.zip').path
        with zipfile.ZipFile(zpfilename, 'w') as zpfile:
            for (i, content) in enumerate(contents):
                filename = str(i)
                if directory:
                    filename = directory + '/' + filename
                zpfile.writestr(filename, content)
        return zpfilename

    def test_invalidMode(self):
        if False:
            i = 10
            return i + 15
        '\n        A ChunkingZipFile opened in write-mode should not allow .readfile(),\n        and raise a RuntimeError instead.\n        '
        with zipstream.ChunkingZipFile(self.mktemp(), 'w') as czf:
            self.assertRaises(RuntimeError, czf.readfile, 'something')

    def test_closedArchive(self):
        if False:
            while True:
                i = 10
        '\n        A closed ChunkingZipFile should raise a L{RuntimeError} when\n        .readfile() is invoked.\n        '
        czf = zipstream.ChunkingZipFile(self.makeZipFile(['something']), 'r')
        czf.close()
        self.assertRaises(RuntimeError, czf.readfile, 'something')

    def test_invalidHeader(self):
        if False:
            return 10
        '\n        A zipfile entry with the wrong magic number should raise BadZipFile for\n        readfile(), but that should not affect other files in the archive.\n        '
        fn = self.makeZipFile(['test contents', 'more contents'])
        with zipfile.ZipFile(fn, 'r') as zf:
            zeroOffset = zf.getinfo('0').header_offset
        with open(fn, 'r+b') as scribble:
            scribble.seek(zeroOffset, 0)
            scribble.write(b'0' * 4)
        with zipstream.ChunkingZipFile(fn) as czf:
            self.assertRaises(zipfile.BadZipFile, czf.readfile, '0')
            with czf.readfile('1') as zfe:
                self.assertEqual(zfe.read(), b'more contents')

    def test_filenameMismatch(self):
        if False:
            i = 10
            return i + 15
        '\n        A zipfile entry with a different filename than is found in the central\n        directory should raise BadZipFile.\n        '
        fn = self.makeZipFile([b'test contents', b'more contents'])
        with zipfile.ZipFile(fn, 'r') as zf:
            info = zf.getinfo('0')
            info.filename = 'not zero'
        with open(fn, 'r+b') as scribble:
            scribble.seek(info.header_offset, 0)
            scribble.write(info.FileHeader())
        with zipstream.ChunkingZipFile(fn) as czf:
            self.assertRaises(zipfile.BadZipFile, czf.readfile, '0')
            with czf.readfile('1') as zfe:
                self.assertEqual(zfe.read(), b'more contents')

    def test_unsupportedCompression(self):
        if False:
            print('Hello World!')
        '\n        A zipfile which describes an unsupported compression mechanism should\n        raise BadZipFile.\n        '
        fn = self.mktemp()
        with zipfile.ZipFile(fn, 'w') as zf:
            zi = zipfile.ZipInfo('0')
            zf.writestr(zi, 'some data')
            zi.compress_type = 1234
        with zipstream.ChunkingZipFile(fn) as czf:
            self.assertRaises(zipfile.BadZipFile, czf.readfile, '0')

    def test_extraData(self):
        if False:
            print('Hello World!')
        "\n        readfile() should skip over 'extra' data present in the zip metadata.\n        "
        fn = self.mktemp()
        with zipfile.ZipFile(fn, 'w') as zf:
            zi = zipfile.ZipInfo('0')
            extra_data = b'hello, extra'
            zi.extra = struct.pack('<hh', 42, len(extra_data)) + extra_data
            zf.writestr(zi, b'the real data')
        with zipstream.ChunkingZipFile(fn) as czf, czf.readfile('0') as zfe:
            self.assertEqual(zfe.read(), b'the real data')

    def test_unzipIterChunky(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{twisted.python.zipstream.unzipIterChunky} returns an iterator which\n        must be exhausted to completely unzip the input archive.\n        '
        numfiles = 10
        contents = ['This is test file %d!' % i for i in range(numfiles)]
        contents = [i.encode('ascii') for i in contents]
        zpfilename = self.makeZipFile(contents)
        list(zipstream.unzipIterChunky(zpfilename, self.unzipdir.path))
        self.assertEqual(set(self.unzipdir.listdir()), set(map(str, range(numfiles))))
        for child in self.unzipdir.children():
            num = int(child.basename())
            self.assertEqual(child.getContent(), contents[num])

    def test_unzipIterChunkyDirectory(self):
        if False:
            i = 10
            return i + 15
        '\n        The path to which a file is extracted by L{zipstream.unzipIterChunky}\n        is determined by joining the C{directory} argument to C{unzip} with the\n        path within the archive of the file being extracted.\n        '
        numfiles = 10
        contents = ['This is test file %d!' % i for i in range(numfiles)]
        contents = [i.encode('ascii') for i in contents]
        zpfilename = self.makeZipFile(contents, 'foo')
        list(zipstream.unzipIterChunky(zpfilename, self.unzipdir.path))
        fileContents = {str(num).encode('ascii') for num in range(numfiles)}
        self.assertEqual(set(self.unzipdir.child(b'foo').listdir()), fileContents)
        for child in self.unzipdir.child(b'foo').children():
            num = int(child.basename())
            self.assertEqual(child.getContent(), contents[num])

    def _unzipIterChunkyTest(self, compression, chunksize, lower, upper):
        if False:
            while True:
                i = 10
        '\n        unzipIterChunky should unzip the given number of bytes per iteration.\n        '
        junk = b''
        for n in range(1000):
            num = round(random.random(), 12)
            numEncoded = str(num).encode('ascii')
            junk += b' ' + numEncoded
        junkmd5 = md5(junk).hexdigest()
        tempdir = filepath.FilePath(self.mktemp())
        tempdir.makedirs()
        zfpath = tempdir.child('bigfile.zip').path
        self._makebigfile(zfpath, compression, junk)
        uziter = zipstream.unzipIterChunky(zfpath, tempdir.path, chunksize=chunksize)
        r = next(uziter)
        approx = lower < r < upper
        self.assertTrue(approx)
        for r in uziter:
            pass
        self.assertEqual(r, 0)
        with tempdir.child('zipstreamjunk').open() as f:
            newmd5 = md5(f.read()).hexdigest()
            self.assertEqual(newmd5, junkmd5)

    def test_unzipIterChunkyStored(self):
        if False:
            print('Hello World!')
        '\n        unzipIterChunky should unzip the given number of bytes per iteration on\n        a stored archive.\n        '
        self._unzipIterChunkyTest(zipfile.ZIP_STORED, 500, 35, 45)

    def test_chunkyDeflated(self):
        if False:
            return 10
        '\n        unzipIterChunky should unzip the given number of bytes per iteration on\n        a deflated archive.\n        '
        self._unzipIterChunkyTest(zipfile.ZIP_DEFLATED, 972, 23, 27)

    def _makebigfile(self, filename, compression, junk):
        if False:
            while True:
                i = 10
        '\n        Create a zip file with the given file name and compression scheme.\n        '
        with zipfile.ZipFile(filename, 'w', compression) as zf:
            for i in range(10):
                fn = 'zipstream%d' % i
                zf.writestr(fn, '')
            zf.writestr('zipstreamjunk', junk)