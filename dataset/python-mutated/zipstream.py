"""
An incremental approach to unzipping files.  This allows you to unzip a little
bit of a file at a time, which means you can report progress as a file unzips.
"""
import os.path
import struct
import zipfile
import zlib

class ChunkingZipFile(zipfile.ZipFile):
    """
    A L{zipfile.ZipFile} object which, with L{readfile}, also gives you access
    to a file-like object for each entry.
    """

    def readfile(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return file-like object for name.\n        '
        if self.mode not in ('r', 'a'):
            raise RuntimeError('read() requires mode "r" or "a"')
        if not self.fp:
            raise RuntimeError('Attempt to read ZIP archive that was already closed')
        zinfo = self.getinfo(name)
        self.fp.seek(zinfo.header_offset, 0)
        fheader = self.fp.read(zipfile.sizeFileHeader)
        if fheader[0:4] != zipfile.stringFileHeader:
            raise zipfile.BadZipFile('Bad magic number for file header')
        fheader = struct.unpack(zipfile.structFileHeader, fheader)
        fname = self.fp.read(fheader[zipfile._FH_FILENAME_LENGTH])
        if fheader[zipfile._FH_EXTRA_FIELD_LENGTH]:
            self.fp.read(fheader[zipfile._FH_EXTRA_FIELD_LENGTH])
        if zinfo.flag_bits & 2048:
            fname_str = fname.decode('utf-8')
        else:
            fname_str = fname.decode('cp437')
        if fname_str != zinfo.orig_filename:
            raise zipfile.BadZipFile('File name in directory "%s" and header "%s" differ.' % (zinfo.orig_filename, fname_str))
        if zinfo.compress_type == zipfile.ZIP_STORED:
            return ZipFileEntry(self, zinfo.compress_size)
        elif zinfo.compress_type == zipfile.ZIP_DEFLATED:
            return DeflatedZipFileEntry(self, zinfo.compress_size)
        else:
            raise zipfile.BadZipFile('Unsupported compression method %d for file %s' % (zinfo.compress_type, name))

class _FileEntry:
    """
    Abstract superclass of both compressed and uncompressed variants of
    file-like objects within a zip archive.

    @ivar chunkingZipFile: a chunking zip file.
    @type chunkingZipFile: L{ChunkingZipFile}

    @ivar length: The number of bytes within the zip file that represent this
    file.  (This is the size on disk, not the number of decompressed bytes
    which will result from reading it.)

    @ivar fp: the underlying file object (that contains pkzip data).  Do not
    touch this, please.  It will quite likely move or go away.

    @ivar closed: File-like 'closed' attribute; True before this file has been
    closed, False after.
    @type closed: L{bool}

    @ivar finished: An older, broken synonym for 'closed'.  Do not touch this,
    please.
    @type finished: L{int}
    """

    def __init__(self, chunkingZipFile, length):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a L{_FileEntry} from a L{ChunkingZipFile}.\n        '
        self.chunkingZipFile = chunkingZipFile
        self.fp = self.chunkingZipFile.fp
        self.length = length
        self.finished = 0
        self.closed = False

    def isatty(self):
        if False:
            while True:
                i = 10
        '\n        Returns false because zip files should not be ttys\n        '
        return False

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Close self (file-like object)\n        '
        self.closed = True
        self.finished = 1
        del self.fp

    def readline(self):
        if False:
            print('Hello World!')
        '\n        Read a line.\n        '
        line = b''
        for byte in iter(lambda : self.read(1), b''):
            line += byte
            if byte == b'\n':
                break
        return line

    def __next__(self):
        if False:
            while True:
                i = 10
        '\n        Implement next as file does (like readline, except raises StopIteration\n        at EOF)\n        '
        nextline = self.readline()
        if nextline:
            return nextline
        raise StopIteration()
    next = __next__

    def readlines(self):
        if False:
            return 10
        '\n        Returns a list of all the lines\n        '
        return list(self)

    def xreadlines(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns an iterator (so self)\n        '
        return self

    def __iter__(self):
        if False:
            return 10
        '\n        Returns an iterator (so self)\n        '
        return self

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            for i in range(10):
                print('nop')
        self.close()

class ZipFileEntry(_FileEntry):
    """
    File-like object used to read an uncompressed entry in a ZipFile
    """

    def __init__(self, chunkingZipFile, length):
        if False:
            return 10
        _FileEntry.__init__(self, chunkingZipFile, length)
        self.readBytes = 0

    def tell(self):
        if False:
            print('Hello World!')
        return self.readBytes

    def read(self, n=None):
        if False:
            for i in range(10):
                print('nop')
        if n is None:
            n = self.length - self.readBytes
        if n == 0 or self.finished:
            return b''
        data = self.chunkingZipFile.fp.read(min(n, self.length - self.readBytes))
        self.readBytes += len(data)
        if self.readBytes == self.length or len(data) < n:
            self.finished = 1
        return data

class DeflatedZipFileEntry(_FileEntry):
    """
    File-like object used to read a deflated entry in a ZipFile
    """

    def __init__(self, chunkingZipFile, length):
        if False:
            print('Hello World!')
        _FileEntry.__init__(self, chunkingZipFile, length)
        self.returnedBytes = 0
        self.readBytes = 0
        self.decomp = zlib.decompressobj(-15)
        self.buffer = b''

    def tell(self):
        if False:
            print('Hello World!')
        return self.returnedBytes

    def read(self, n=None):
        if False:
            print('Hello World!')
        if self.finished:
            return b''
        if n is None:
            result = [self.buffer]
            result.append(self.decomp.decompress(self.chunkingZipFile.fp.read(self.length - self.readBytes)))
            result.append(self.decomp.decompress(b'Z'))
            result.append(self.decomp.flush())
            self.buffer = b''
            self.finished = 1
            result = b''.join(result)
            self.returnedBytes += len(result)
            return result
        else:
            while len(self.buffer) < n:
                data = self.chunkingZipFile.fp.read(min(n, 1024, self.length - self.readBytes))
                self.readBytes += len(data)
                if not data:
                    result = self.buffer + self.decomp.decompress(b'Z') + self.decomp.flush()
                    self.finished = 1
                    self.buffer = b''
                    self.returnedBytes += len(result)
                    return result
                else:
                    self.buffer += self.decomp.decompress(data)
            result = self.buffer[:n]
            self.buffer = self.buffer[n:]
            self.returnedBytes += len(result)
            return result
DIR_BIT = 16

def countZipFileChunks(filename, chunksize):
    if False:
        print('Hello World!')
    '\n    Predict the number of chunks that will be extracted from the entire\n    zipfile, given chunksize blocks.\n    '
    totalchunks = 0
    zf = ChunkingZipFile(filename)
    for info in zf.infolist():
        totalchunks += countFileChunks(info, chunksize)
    return totalchunks

def countFileChunks(zipinfo, chunksize):
    if False:
        print('Hello World!')
    '\n    Count the number of chunks that will result from the given C{ZipInfo}.\n\n    @param zipinfo: a C{zipfile.ZipInfo} instance describing an entry in a zip\n    archive to be counted.\n\n    @return: the number of chunks present in the zip file.  (Even an empty file\n    counts as one chunk.)\n    @rtype: L{int}\n    '
    (count, extra) = divmod(zipinfo.file_size, chunksize)
    if extra > 0:
        count += 1
    return count or 1

def unzipIterChunky(filename, directory='.', overwrite=0, chunksize=4096):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a generator for the zipfile.  This implementation will yield after\n    every chunksize uncompressed bytes, or at the end of a file, whichever\n    comes first.\n\n    The value it yields is the number of chunks left to unzip.\n    '
    czf = ChunkingZipFile(filename, 'r')
    if not os.path.exists(directory):
        os.makedirs(directory)
    remaining = countZipFileChunks(filename, chunksize)
    names = czf.namelist()
    infos = czf.infolist()
    for (entry, info) in zip(names, infos):
        isdir = info.external_attr & DIR_BIT
        f = os.path.join(directory, entry)
        if isdir:
            if not os.path.exists(f):
                os.makedirs(f)
            remaining -= 1
            yield remaining
        else:
            fdir = os.path.split(f)[0]
            if not os.path.exists(fdir):
                os.makedirs(fdir)
            if overwrite or not os.path.exists(f):
                fp = czf.readfile(entry)
                if info.file_size == 0:
                    remaining -= 1
                    yield remaining
                with open(f, 'wb') as outfile:
                    while fp.tell() < info.file_size:
                        hunk = fp.read(chunksize)
                        outfile.write(hunk)
                        remaining -= 1
                        yield remaining
            else:
                remaining -= countFileChunks(info, chunksize)
                yield remaining