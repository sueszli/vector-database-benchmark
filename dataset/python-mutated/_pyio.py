"""
Python implementation of the io module.
"""
import os
import abc
import codecs
import errno
import stat
import sys
from _thread import allocate_lock as Lock
if sys.platform in {'win32', 'cygwin'}:
    from msvcrt import setmode as _setmode
else:
    _setmode = None
import io
from io import __all__, SEEK_SET, SEEK_CUR, SEEK_END
valid_seek_flags = {0, 1, 2}
if hasattr(os, 'SEEK_HOLE'):
    valid_seek_flags.add(os.SEEK_HOLE)
    valid_seek_flags.add(os.SEEK_DATA)
DEFAULT_BUFFER_SIZE = 8 * 1024
BlockingIOError = BlockingIOError
_IOBASE_EMITS_UNRAISABLE = hasattr(sys, 'gettotalrefcount') or sys.flags.dev_mode
_CHECK_ERRORS = _IOBASE_EMITS_UNRAISABLE

def text_encoding(encoding, stacklevel=2):
    if False:
        print('Hello World!')
    '\n    A helper function to choose the text encoding.\n\n    When encoding is not None, just return it.\n    Otherwise, return the default text encoding (i.e. "locale").\n\n    This function emits an EncodingWarning if *encoding* is None and\n    sys.flags.warn_default_encoding is true.\n\n    This can be used in APIs with an encoding=None parameter\n    that pass it to TextIOWrapper or open.\n    However, please consider using encoding="utf-8" for new APIs.\n    '
    if encoding is None:
        encoding = 'locale'
        if sys.flags.warn_default_encoding:
            import warnings
            warnings.warn("'encoding' argument not specified.", EncodingWarning, stacklevel + 1)
    return encoding

@staticmethod
def open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
    if False:
        for i in range(10):
            print('nop')
    'Open file and return a stream.  Raise OSError upon failure.\n\n    file is either a text or byte string giving the name (and the path\n    if the file isn\'t in the current working directory) of the file to\n    be opened or an integer file descriptor of the file to be\n    wrapped. (If a file descriptor is given, it is closed when the\n    returned I/O object is closed, unless closefd is set to False.)\n\n    mode is an optional string that specifies the mode in which the file is\n    opened. It defaults to \'r\' which means open for reading in text mode. Other\n    common values are \'w\' for writing (truncating the file if it already\n    exists), \'x\' for exclusive creation of a new file, and \'a\' for appending\n    (which on some Unix systems, means that all writes append to the end of the\n    file regardless of the current seek position). In text mode, if encoding is\n    not specified the encoding used is platform dependent. (For reading and\n    writing raw bytes use binary mode and leave encoding unspecified.) The\n    available modes are:\n\n    ========= ===============================================================\n    Character Meaning\n    --------- ---------------------------------------------------------------\n    \'r\'       open for reading (default)\n    \'w\'       open for writing, truncating the file first\n    \'x\'       create a new file and open it for writing\n    \'a\'       open for writing, appending to the end of the file if it exists\n    \'b\'       binary mode\n    \'t\'       text mode (default)\n    \'+\'       open a disk file for updating (reading and writing)\n    \'U\'       universal newline mode (deprecated)\n    ========= ===============================================================\n\n    The default mode is \'rt\' (open for reading text). For binary random\n    access, the mode \'w+b\' opens and truncates the file to 0 bytes, while\n    \'r+b\' opens the file without truncation. The \'x\' mode implies \'w\' and\n    raises an `FileExistsError` if the file already exists.\n\n    Python distinguishes between files opened in binary and text modes,\n    even when the underlying operating system doesn\'t. Files opened in\n    binary mode (appending \'b\' to the mode argument) return contents as\n    bytes objects without any decoding. In text mode (the default, or when\n    \'t\' is appended to the mode argument), the contents of the file are\n    returned as strings, the bytes having been first decoded using a\n    platform-dependent encoding or using the specified encoding if given.\n\n    \'U\' mode is deprecated and will raise an exception in future versions\n    of Python.  It has no effect in Python 3.  Use newline to control\n    universal newlines mode.\n\n    buffering is an optional integer used to set the buffering policy.\n    Pass 0 to switch buffering off (only allowed in binary mode), 1 to select\n    line buffering (only usable in text mode), and an integer > 1 to indicate\n    the size of a fixed-size chunk buffer.  When no buffering argument is\n    given, the default buffering policy works as follows:\n\n    * Binary files are buffered in fixed-size chunks; the size of the buffer\n      is chosen using a heuristic trying to determine the underlying device\'s\n      "block size" and falling back on `io.DEFAULT_BUFFER_SIZE`.\n      On many systems, the buffer will typically be 4096 or 8192 bytes long.\n\n    * "Interactive" text files (files for which isatty() returns True)\n      use line buffering.  Other text files use the policy described above\n      for binary files.\n\n    encoding is the str name of the encoding used to decode or encode the\n    file. This should only be used in text mode. The default encoding is\n    platform dependent, but any encoding supported by Python can be\n    passed.  See the codecs module for the list of supported encodings.\n\n    errors is an optional string that specifies how encoding errors are to\n    be handled---this argument should not be used in binary mode. Pass\n    \'strict\' to raise a ValueError exception if there is an encoding error\n    (the default of None has the same effect), or pass \'ignore\' to ignore\n    errors. (Note that ignoring encoding errors can lead to data loss.)\n    See the documentation for codecs.register for a list of the permitted\n    encoding error strings.\n\n    newline is a string controlling how universal newlines works (it only\n    applies to text mode). It can be None, \'\', \'\\n\', \'\\r\', and \'\\r\\n\'.  It works\n    as follows:\n\n    * On input, if newline is None, universal newlines mode is\n      enabled. Lines in the input can end in \'\\n\', \'\\r\', or \'\\r\\n\', and\n      these are translated into \'\\n\' before being returned to the\n      caller. If it is \'\', universal newline mode is enabled, but line\n      endings are returned to the caller untranslated. If it has any of\n      the other legal values, input lines are only terminated by the given\n      string, and the line ending is returned to the caller untranslated.\n\n    * On output, if newline is None, any \'\\n\' characters written are\n      translated to the system default line separator, os.linesep. If\n      newline is \'\', no translation takes place. If newline is any of the\n      other legal values, any \'\\n\' characters written are translated to\n      the given string.\n\n    closedfd is a bool. If closefd is False, the underlying file descriptor will\n    be kept open when the file is closed. This does not work when a file name is\n    given and must be True in that case.\n\n    The newly created file is non-inheritable.\n\n    A custom opener can be used by passing a callable as *opener*. The\n    underlying file descriptor for the file object is then obtained by calling\n    *opener* with (*file*, *flags*). *opener* must return an open file\n    descriptor (passing os.open as *opener* results in functionality similar to\n    passing None).\n\n    open() returns a file object whose type depends on the mode, and\n    through which the standard file operations such as reading and writing\n    are performed. When open() is used to open a file in a text mode (\'w\',\n    \'r\', \'wt\', \'rt\', etc.), it returns a TextIOWrapper. When used to open\n    a file in a binary mode, the returned class varies: in read binary\n    mode, it returns a BufferedReader; in write binary and append binary\n    modes, it returns a BufferedWriter, and in read/write mode, it returns\n    a BufferedRandom.\n\n    It is also possible to use a string or bytearray as a file for both\n    reading and writing. For strings StringIO can be used like a file\n    opened in a text mode, and for bytes a BytesIO can be used like a file\n    opened in a binary mode.\n    '
    if not isinstance(file, int):
        file = os.fspath(file)
    if not isinstance(file, (str, bytes, int)):
        raise TypeError('invalid file: %r' % file)
    if not isinstance(mode, str):
        raise TypeError('invalid mode: %r' % mode)
    if not isinstance(buffering, int):
        raise TypeError('invalid buffering: %r' % buffering)
    if encoding is not None and (not isinstance(encoding, str)):
        raise TypeError('invalid encoding: %r' % encoding)
    if errors is not None and (not isinstance(errors, str)):
        raise TypeError('invalid errors: %r' % errors)
    modes = set(mode)
    if modes - set('axrwb+tU') or len(mode) > len(modes):
        raise ValueError('invalid mode: %r' % mode)
    creating = 'x' in modes
    reading = 'r' in modes
    writing = 'w' in modes
    appending = 'a' in modes
    updating = '+' in modes
    text = 't' in modes
    binary = 'b' in modes
    if 'U' in modes:
        if creating or writing or appending or updating:
            raise ValueError("mode U cannot be combined with 'x', 'w', 'a', or '+'")
        import warnings
        warnings.warn("'U' mode is deprecated", DeprecationWarning, 2)
        reading = True
    if text and binary:
        raise ValueError("can't have text and binary mode at once")
    if creating + reading + writing + appending > 1:
        raise ValueError("can't have read/write/append mode at once")
    if not (creating or reading or writing or appending):
        raise ValueError('must have exactly one of read/write/append mode')
    if binary and encoding is not None:
        raise ValueError("binary mode doesn't take an encoding argument")
    if binary and errors is not None:
        raise ValueError("binary mode doesn't take an errors argument")
    if binary and newline is not None:
        raise ValueError("binary mode doesn't take a newline argument")
    if binary and buffering == 1:
        import warnings
        warnings.warn("line buffering (buffering=1) isn't supported in binary mode, the default buffer size will be used", RuntimeWarning, 2)
    raw = FileIO(file, (creating and 'x' or '') + (reading and 'r' or '') + (writing and 'w' or '') + (appending and 'a' or '') + (updating and '+' or ''), closefd, opener=opener)
    result = raw
    try:
        line_buffering = False
        if buffering == 1 or (buffering < 0 and raw.isatty()):
            buffering = -1
            line_buffering = True
        if buffering < 0:
            buffering = DEFAULT_BUFFER_SIZE
            try:
                bs = os.fstat(raw.fileno()).st_blksize
            except (OSError, AttributeError):
                pass
            else:
                if bs > 1:
                    buffering = bs
        if buffering < 0:
            raise ValueError('invalid buffering size')
        if buffering == 0:
            if binary:
                return result
            raise ValueError("can't have unbuffered text I/O")
        if updating:
            buffer = BufferedRandom(raw, buffering)
        elif creating or writing or appending:
            buffer = BufferedWriter(raw, buffering)
        elif reading:
            buffer = BufferedReader(raw, buffering)
        else:
            raise ValueError('unknown mode: %r' % mode)
        result = buffer
        if binary:
            return result
        encoding = text_encoding(encoding)
        text = TextIOWrapper(buffer, encoding, errors, newline, line_buffering)
        result = text
        text.mode = mode
        return result
    except:
        result.close()
        raise

def _open_code_with_warning(path):
    if False:
        for i in range(10):
            print('nop')
    "Opens the provided file with mode ``'rb'``. This function\n    should be used when the intent is to treat the contents as\n    executable code.\n\n    ``path`` should be an absolute path.\n\n    When supported by the runtime, this function can be hooked\n    in order to allow embedders more control over code files.\n    This functionality is not supported on the current runtime.\n    "
    import warnings
    warnings.warn('_pyio.open_code() may not be using hooks', RuntimeWarning, 2)
    return open(path, 'rb')
try:
    open_code = io.open_code
except AttributeError:
    open_code = _open_code_with_warning

def __getattr__(name):
    if False:
        i = 10
        return i + 15
    if name == 'OpenWrapper':
        import warnings
        warnings.warn('OpenWrapper is deprecated, use open instead', DeprecationWarning, stacklevel=2)
        global OpenWrapper
        OpenWrapper = open
        return OpenWrapper
    raise AttributeError(name)
try:
    UnsupportedOperation = io.UnsupportedOperation
except AttributeError:

    class UnsupportedOperation(OSError, ValueError):
        pass

class IOBase(metaclass=abc.ABCMeta):
    """The abstract base class for all I/O classes.

    This class provides dummy implementations for many methods that
    derived classes can override selectively; the default implementations
    represent a file that cannot be read, written or seeked.

    Even though IOBase does not declare read or write because
    their signatures will vary, implementations and clients should
    consider those methods part of the interface. Also, implementations
    may raise UnsupportedOperation when operations they do not support are
    called.

    The basic type used for binary data read from or written to a file is
    bytes. Other bytes-like objects are accepted as method arguments too.
    Text I/O classes work with str data.

    Note that calling any method (even inquiries) on a closed stream is
    undefined. Implementations may raise OSError in this case.

    IOBase (and its subclasses) support the iterator protocol, meaning
    that an IOBase object can be iterated over yielding the lines in a
    stream.

    IOBase also supports the :keyword:`with` statement. In this example,
    fp is closed after the suite of the with statement is complete:

    with open('spam.txt', 'r') as fp:
        fp.write('Spam and eggs!')
    """

    def _unsupported(self, name):
        if False:
            print('Hello World!')
        'Internal: raise an OSError exception for unsupported operations.'
        raise UnsupportedOperation('%s.%s() not supported' % (self.__class__.__name__, name))

    def seek(self, pos, whence=0):
        if False:
            i = 10
            return i + 15
        'Change stream position.\n\n        Change the stream position to byte offset pos. Argument pos is\n        interpreted relative to the position indicated by whence.  Values\n        for whence are ints:\n\n        * 0 -- start of stream (the default); offset should be zero or positive\n        * 1 -- current stream position; offset may be negative\n        * 2 -- end of stream; offset is usually negative\n        Some operating systems / file systems could provide additional values.\n\n        Return an int indicating the new absolute position.\n        '
        self._unsupported('seek')

    def tell(self):
        if False:
            i = 10
            return i + 15
        'Return an int indicating the current stream position.'
        return self.seek(0, 1)

    def truncate(self, pos=None):
        if False:
            print('Hello World!')
        'Truncate file to size bytes.\n\n        Size defaults to the current IO position as reported by tell().  Return\n        the new size.\n        '
        self._unsupported('truncate')

    def flush(self):
        if False:
            for i in range(10):
                print('nop')
        'Flush write buffers, if applicable.\n\n        This is not implemented for read-only and non-blocking streams.\n        '
        self._checkClosed()
    __closed = False

    def close(self):
        if False:
            while True:
                i = 10
        'Flush and close the IO object.\n\n        This method has no effect if the file is already closed.\n        '
        if not self.__closed:
            try:
                self.flush()
            finally:
                self.__closed = True

    def __del__(self):
        if False:
            while True:
                i = 10
        'Destructor.  Calls close().'
        try:
            closed = self.closed
        except AttributeError:
            return
        if closed:
            return
        if _IOBASE_EMITS_UNRAISABLE:
            self.close()
        else:
            try:
                self.close()
            except:
                pass

    def seekable(self):
        if False:
            i = 10
            return i + 15
        'Return a bool indicating whether object supports random access.\n\n        If False, seek(), tell() and truncate() will raise OSError.\n        This method may need to do a test seek().\n        '
        return False

    def _checkSeekable(self, msg=None):
        if False:
            for i in range(10):
                print('nop')
        'Internal: raise UnsupportedOperation if file is not seekable\n        '
        if not self.seekable():
            raise UnsupportedOperation('File or stream is not seekable.' if msg is None else msg)

    def readable(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a bool indicating whether object was opened for reading.\n\n        If False, read() will raise OSError.\n        '
        return False

    def _checkReadable(self, msg=None):
        if False:
            for i in range(10):
                print('nop')
        'Internal: raise UnsupportedOperation if file is not readable\n        '
        if not self.readable():
            raise UnsupportedOperation('File or stream is not readable.' if msg is None else msg)

    def writable(self):
        if False:
            i = 10
            return i + 15
        'Return a bool indicating whether object was opened for writing.\n\n        If False, write() and truncate() will raise OSError.\n        '
        return False

    def _checkWritable(self, msg=None):
        if False:
            i = 10
            return i + 15
        'Internal: raise UnsupportedOperation if file is not writable\n        '
        if not self.writable():
            raise UnsupportedOperation('File or stream is not writable.' if msg is None else msg)

    @property
    def closed(self):
        if False:
            i = 10
            return i + 15
        'closed: bool.  True iff the file has been closed.\n\n        For backwards compatibility, this is a property, not a predicate.\n        '
        return self.__closed

    def _checkClosed(self, msg=None):
        if False:
            return 10
        'Internal: raise a ValueError if file is closed\n        '
        if self.closed:
            raise ValueError('I/O operation on closed file.' if msg is None else msg)

    def __enter__(self):
        if False:
            return 10
        'Context management protocol.  Returns self (an instance of IOBase).'
        self._checkClosed()
        return self

    def __exit__(self, *args):
        if False:
            i = 10
            return i + 15
        'Context management protocol.  Calls close()'
        self.close()

    def fileno(self):
        if False:
            print('Hello World!')
        'Returns underlying file descriptor (an int) if one exists.\n\n        An OSError is raised if the IO object does not use a file descriptor.\n        '
        self._unsupported('fileno')

    def isatty(self):
        if False:
            i = 10
            return i + 15
        "Return a bool indicating whether this is an 'interactive' stream.\n\n        Return False if it can't be determined.\n        "
        self._checkClosed()
        return False

    def readline(self, size=-1):
        if False:
            i = 10
            return i + 15
        "Read and return a line of bytes from the stream.\n\n        If size is specified, at most size bytes will be read.\n        Size should be an int.\n\n        The line terminator is always b'\\n' for binary files; for text\n        files, the newlines argument to open can be used to select the line\n        terminator(s) recognized.\n        "
        if hasattr(self, 'peek'):

            def nreadahead():
                if False:
                    i = 10
                    return i + 15
                readahead = self.peek(1)
                if not readahead:
                    return 1
                n = readahead.find(b'\n') + 1 or len(readahead)
                if size >= 0:
                    n = min(n, size)
                return n
        else:

            def nreadahead():
                if False:
                    for i in range(10):
                        print('nop')
                return 1
        if size is None:
            size = -1
        else:
            try:
                size_index = size.__index__
            except AttributeError:
                raise TypeError(f'{size!r} is not an integer')
            else:
                size = size_index()
        res = bytearray()
        while size < 0 or len(res) < size:
            b = self.read(nreadahead())
            if not b:
                break
            res += b
            if res.endswith(b'\n'):
                break
        return bytes(res)

    def __iter__(self):
        if False:
            return 10
        self._checkClosed()
        return self

    def __next__(self):
        if False:
            while True:
                i = 10
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def readlines(self, hint=None):
        if False:
            i = 10
            return i + 15
        'Return a list of lines from the stream.\n\n        hint can be specified to control the number of lines read: no more\n        lines will be read if the total size (in bytes/characters) of all\n        lines so far exceeds hint.\n        '
        if hint is None or hint <= 0:
            return list(self)
        n = 0
        lines = []
        for line in self:
            lines.append(line)
            n += len(line)
            if n >= hint:
                break
        return lines

    def writelines(self, lines):
        if False:
            while True:
                i = 10
        'Write a list of lines to the stream.\n\n        Line separators are not added, so it is usual for each of the lines\n        provided to have a line separator at the end.\n        '
        self._checkClosed()
        for line in lines:
            self.write(line)
io.IOBase.register(IOBase)

class RawIOBase(IOBase):
    """Base class for raw binary I/O."""

    def read(self, size=-1):
        if False:
            while True:
                i = 10
        'Read and return up to size bytes, where size is an int.\n\n        Returns an empty bytes object on EOF, or None if the object is\n        set not to block and has no data to read.\n        '
        if size is None:
            size = -1
        if size < 0:
            return self.readall()
        b = bytearray(size.__index__())
        n = self.readinto(b)
        if n is None:
            return None
        del b[n:]
        return bytes(b)

    def readall(self):
        if False:
            return 10
        'Read until EOF, using multiple read() call.'
        res = bytearray()
        while True:
            data = self.read(DEFAULT_BUFFER_SIZE)
            if not data:
                break
            res += data
        if res:
            return bytes(res)
        else:
            return data

    def readinto(self, b):
        if False:
            while True:
                i = 10
        'Read bytes into a pre-allocated bytes-like object b.\n\n        Returns an int representing the number of bytes read (0 for EOF), or\n        None if the object is set not to block and has no data to read.\n        '
        self._unsupported('readinto')

    def write(self, b):
        if False:
            return 10
        'Write the given buffer to the IO stream.\n\n        Returns the number of bytes written, which may be less than the\n        length of b in bytes.\n        '
        self._unsupported('write')
io.RawIOBase.register(RawIOBase)
from _io import FileIO
RawIOBase.register(FileIO)

class BufferedIOBase(IOBase):
    """Base class for buffered IO objects.

    The main difference with RawIOBase is that the read() method
    supports omitting the size argument, and does not have a default
    implementation that defers to readinto().

    In addition, read(), readinto() and write() may raise
    BlockingIOError if the underlying raw stream is in non-blocking
    mode and not ready; unlike their raw counterparts, they will never
    return None.

    A typical implementation should not inherit from a RawIOBase
    implementation, but wrap one.
    """

    def read(self, size=-1):
        if False:
            print('Hello World!')
        "Read and return up to size bytes, where size is an int.\n\n        If the argument is omitted, None, or negative, reads and\n        returns all data until EOF.\n\n        If the argument is positive, and the underlying raw stream is\n        not 'interactive', multiple raw reads may be issued to satisfy\n        the byte count (unless EOF is reached first).  But for\n        interactive raw streams (XXX and for pipes?), at most one raw\n        read will be issued, and a short result does not imply that\n        EOF is imminent.\n\n        Returns an empty bytes array on EOF.\n\n        Raises BlockingIOError if the underlying raw stream has no\n        data at the moment.\n        "
        self._unsupported('read')

    def read1(self, size=-1):
        if False:
            return 10
        'Read up to size bytes with at most one read() system call,\n        where size is an int.\n        '
        self._unsupported('read1')

    def readinto(self, b):
        if False:
            print('Hello World!')
        "Read bytes into a pre-allocated bytes-like object b.\n\n        Like read(), this may issue multiple reads to the underlying raw\n        stream, unless the latter is 'interactive'.\n\n        Returns an int representing the number of bytes read (0 for EOF).\n\n        Raises BlockingIOError if the underlying raw stream has no\n        data at the moment.\n        "
        return self._readinto(b, read1=False)

    def readinto1(self, b):
        if False:
            for i in range(10):
                print('nop')
        'Read bytes into buffer *b*, using at most one system call\n\n        Returns an int representing the number of bytes read (0 for EOF).\n\n        Raises BlockingIOError if the underlying raw stream has no\n        data at the moment.\n        '
        return self._readinto(b, read1=True)

    def _readinto(self, b, read1):
        if False:
            print('Hello World!')
        if not isinstance(b, memoryview):
            b = memoryview(b)
        b = b.cast('B')
        if read1:
            data = self.read1(len(b))
        else:
            data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    def write(self, b):
        if False:
            while True:
                i = 10
        'Write the given bytes buffer to the IO stream.\n\n        Return the number of bytes written, which is always the length of b\n        in bytes.\n\n        Raises BlockingIOError if the buffer is full and the\n        underlying raw stream cannot accept more data at the moment.\n        '
        self._unsupported('write')

    def detach(self):
        if False:
            while True:
                i = 10
        '\n        Separate the underlying raw stream from the buffer and return it.\n\n        After the raw stream has been detached, the buffer is in an unusable\n        state.\n        '
        self._unsupported('detach')
io.BufferedIOBase.register(BufferedIOBase)

class _BufferedIOMixin(BufferedIOBase):
    """A mixin implementation of BufferedIOBase with an underlying raw stream.

    This passes most requests on to the underlying raw stream.  It
    does *not* provide implementations of read(), readinto() or
    write().
    """

    def __init__(self, raw):
        if False:
            return 10
        self._raw = raw

    def seek(self, pos, whence=0):
        if False:
            print('Hello World!')
        new_position = self.raw.seek(pos, whence)
        if new_position < 0:
            raise OSError('seek() returned an invalid position')
        return new_position

    def tell(self):
        if False:
            print('Hello World!')
        pos = self.raw.tell()
        if pos < 0:
            raise OSError('tell() returned an invalid position')
        return pos

    def truncate(self, pos=None):
        if False:
            print('Hello World!')
        self._checkClosed()
        self._checkWritable()
        self.flush()
        if pos is None:
            pos = self.tell()
        return self.raw.truncate(pos)

    def flush(self):
        if False:
            for i in range(10):
                print('nop')
        if self.closed:
            raise ValueError('flush on closed file')
        self.raw.flush()

    def close(self):
        if False:
            i = 10
            return i + 15
        if self.raw is not None and (not self.closed):
            try:
                self.flush()
            finally:
                self.raw.close()

    def detach(self):
        if False:
            return 10
        if self.raw is None:
            raise ValueError('raw stream already detached')
        self.flush()
        raw = self._raw
        self._raw = None
        return raw

    def seekable(self):
        if False:
            for i in range(10):
                print('nop')
        return self.raw.seekable()

    @property
    def raw(self):
        if False:
            for i in range(10):
                print('nop')
        return self._raw

    @property
    def closed(self):
        if False:
            print('Hello World!')
        return self.raw.closed

    @property
    def name(self):
        if False:
            print('Hello World!')
        return self.raw.name

    @property
    def mode(self):
        if False:
            while True:
                i = 10
        return self.raw.mode

    def __getstate__(self):
        if False:
            return 10
        raise TypeError(f'cannot pickle {self.__class__.__name__!r} object')

    def __repr__(self):
        if False:
            while True:
                i = 10
        modname = self.__class__.__module__
        clsname = self.__class__.__qualname__
        try:
            name = self.name
        except AttributeError:
            return '<{}.{}>'.format(modname, clsname)
        else:
            return '<{}.{} name={!r}>'.format(modname, clsname, name)

    def fileno(self):
        if False:
            for i in range(10):
                print('nop')
        return self.raw.fileno()

    def isatty(self):
        if False:
            i = 10
            return i + 15
        return self.raw.isatty()

class BytesIO(BufferedIOBase):
    """Buffered I/O implementation using an in-memory bytes buffer."""
    _buffer = None

    def __init__(self, initial_bytes=None):
        if False:
            i = 10
            return i + 15
        buf = bytearray()
        if initial_bytes is not None:
            buf += initial_bytes
        self._buffer = buf
        self._pos = 0

    def __getstate__(self):
        if False:
            return 10
        if self.closed:
            raise ValueError('__getstate__ on closed file')
        return self.__dict__.copy()

    def getvalue(self):
        if False:
            i = 10
            return i + 15
        'Return the bytes value (contents) of the buffer\n        '
        if self.closed:
            raise ValueError('getvalue on closed file')
        return bytes(self._buffer)

    def getbuffer(self):
        if False:
            i = 10
            return i + 15
        'Return a readable and writable view of the buffer.\n        '
        if self.closed:
            raise ValueError('getbuffer on closed file')
        return memoryview(self._buffer)

    def close(self):
        if False:
            print('Hello World!')
        if self._buffer is not None:
            self._buffer.clear()
        super().close()

    def read(self, size=-1):
        if False:
            print('Hello World!')
        if self.closed:
            raise ValueError('read from closed file')
        if size is None:
            size = -1
        else:
            try:
                size_index = size.__index__
            except AttributeError:
                raise TypeError(f'{size!r} is not an integer')
            else:
                size = size_index()
        if size < 0:
            size = len(self._buffer)
        if len(self._buffer) <= self._pos:
            return b''
        newpos = min(len(self._buffer), self._pos + size)
        b = self._buffer[self._pos:newpos]
        self._pos = newpos
        return bytes(b)

    def read1(self, size=-1):
        if False:
            while True:
                i = 10
        'This is the same as read.\n        '
        return self.read(size)

    def write(self, b):
        if False:
            i = 10
            return i + 15
        if self.closed:
            raise ValueError('write to closed file')
        if isinstance(b, str):
            raise TypeError("can't write str to binary stream")
        with memoryview(b) as view:
            n = view.nbytes
        if n == 0:
            return 0
        pos = self._pos
        if pos > len(self._buffer):
            padding = b'\x00' * (pos - len(self._buffer))
            self._buffer += padding
        self._buffer[pos:pos + n] = b
        self._pos += n
        return n

    def seek(self, pos, whence=0):
        if False:
            while True:
                i = 10
        if self.closed:
            raise ValueError('seek on closed file')
        try:
            pos_index = pos.__index__
        except AttributeError:
            raise TypeError(f'{pos!r} is not an integer')
        else:
            pos = pos_index()
        if whence == 0:
            if pos < 0:
                raise ValueError('negative seek position %r' % (pos,))
            self._pos = pos
        elif whence == 1:
            self._pos = max(0, self._pos + pos)
        elif whence == 2:
            self._pos = max(0, len(self._buffer) + pos)
        else:
            raise ValueError('unsupported whence value')
        return self._pos

    def tell(self):
        if False:
            for i in range(10):
                print('nop')
        if self.closed:
            raise ValueError('tell on closed file')
        return self._pos

    def truncate(self, pos=None):
        if False:
            while True:
                i = 10
        if self.closed:
            raise ValueError('truncate on closed file')
        if pos is None:
            pos = self._pos
        else:
            try:
                pos_index = pos.__index__
            except AttributeError:
                raise TypeError(f'{pos!r} is not an integer')
            else:
                pos = pos_index()
            if pos < 0:
                raise ValueError('negative truncate position %r' % (pos,))
        del self._buffer[pos:]
        return pos

    def readable(self):
        if False:
            return 10
        if self.closed:
            raise ValueError('I/O operation on closed file.')
        return True

    def writable(self):
        if False:
            print('Hello World!')
        if self.closed:
            raise ValueError('I/O operation on closed file.')
        return True

    def seekable(self):
        if False:
            return 10
        if self.closed:
            raise ValueError('I/O operation on closed file.')
        return True

class BufferedReader(_BufferedIOMixin):
    """BufferedReader(raw[, buffer_size])

    A buffer for a readable, sequential BaseRawIO object.

    The constructor creates a BufferedReader for the given readable raw
    stream and buffer_size. If buffer_size is omitted, DEFAULT_BUFFER_SIZE
    is used.
    """

    def __init__(self, raw, buffer_size=DEFAULT_BUFFER_SIZE):
        if False:
            return 10
        'Create a new buffered reader using the given readable raw IO object.\n        '
        if not raw.readable():
            raise OSError('"raw" argument must be readable.')
        _BufferedIOMixin.__init__(self, raw)
        if buffer_size <= 0:
            raise ValueError('invalid buffer size')
        self.buffer_size = buffer_size
        self._reset_read_buf()
        self._read_lock = Lock()

    def readable(self):
        if False:
            i = 10
            return i + 15
        return self.raw.readable()

    def _reset_read_buf(self):
        if False:
            print('Hello World!')
        self._read_buf = b''
        self._read_pos = 0

    def read(self, size=None):
        if False:
            i = 10
            return i + 15
        'Read size bytes.\n\n        Returns exactly size bytes of data unless the underlying raw IO\n        stream reaches EOF or if the call would block in non-blocking\n        mode. If size is negative, read until EOF or until read() would\n        block.\n        '
        if size is not None and size < -1:
            raise ValueError('invalid number of bytes to read')
        with self._read_lock:
            return self._read_unlocked(size)

    def _read_unlocked(self, n=None):
        if False:
            print('Hello World!')
        nodata_val = b''
        empty_values = (b'', None)
        buf = self._read_buf
        pos = self._read_pos
        if n is None or n == -1:
            self._reset_read_buf()
            if hasattr(self.raw, 'readall'):
                chunk = self.raw.readall()
                if chunk is None:
                    return buf[pos:] or None
                else:
                    return buf[pos:] + chunk
            chunks = [buf[pos:]]
            current_size = 0
            while True:
                chunk = self.raw.read()
                if chunk in empty_values:
                    nodata_val = chunk
                    break
                current_size += len(chunk)
                chunks.append(chunk)
            return b''.join(chunks) or nodata_val
        avail = len(buf) - pos
        if n <= avail:
            self._read_pos += n
            return buf[pos:pos + n]
        chunks = [buf[pos:]]
        wanted = max(self.buffer_size, n)
        while avail < n:
            chunk = self.raw.read(wanted)
            if chunk in empty_values:
                nodata_val = chunk
                break
            avail += len(chunk)
            chunks.append(chunk)
        n = min(n, avail)
        out = b''.join(chunks)
        self._read_buf = out[n:]
        self._read_pos = 0
        return out[:n] if out else nodata_val

    def peek(self, size=0):
        if False:
            return 10
        'Returns buffered bytes without advancing the position.\n\n        The argument indicates a desired minimal number of bytes; we\n        do at most one raw read to satisfy it.  We never return more\n        than self.buffer_size.\n        '
        with self._read_lock:
            return self._peek_unlocked(size)

    def _peek_unlocked(self, n=0):
        if False:
            i = 10
            return i + 15
        want = min(n, self.buffer_size)
        have = len(self._read_buf) - self._read_pos
        if have < want or have <= 0:
            to_read = self.buffer_size - have
            current = self.raw.read(to_read)
            if current:
                self._read_buf = self._read_buf[self._read_pos:] + current
                self._read_pos = 0
        return self._read_buf[self._read_pos:]

    def read1(self, size=-1):
        if False:
            print('Hello World!')
        'Reads up to size bytes, with at most one read() system call.'
        if size < 0:
            size = self.buffer_size
        if size == 0:
            return b''
        with self._read_lock:
            self._peek_unlocked(1)
            return self._read_unlocked(min(size, len(self._read_buf) - self._read_pos))

    def _readinto(self, buf, read1):
        if False:
            for i in range(10):
                print('nop')
        'Read data into *buf* with at most one system call.'
        if not isinstance(buf, memoryview):
            buf = memoryview(buf)
        if buf.nbytes == 0:
            return 0
        buf = buf.cast('B')
        written = 0
        with self._read_lock:
            while written < len(buf):
                avail = min(len(self._read_buf) - self._read_pos, len(buf))
                if avail:
                    buf[written:written + avail] = self._read_buf[self._read_pos:self._read_pos + avail]
                    self._read_pos += avail
                    written += avail
                    if written == len(buf):
                        break
                if len(buf) - written > self.buffer_size:
                    n = self.raw.readinto(buf[written:])
                    if not n:
                        break
                    written += n
                elif not (read1 and written):
                    if not self._peek_unlocked(1):
                        break
                if read1 and written:
                    break
        return written

    def tell(self):
        if False:
            return 10
        return _BufferedIOMixin.tell(self) - len(self._read_buf) + self._read_pos

    def seek(self, pos, whence=0):
        if False:
            while True:
                i = 10
        if whence not in valid_seek_flags:
            raise ValueError('invalid whence value')
        with self._read_lock:
            if whence == 1:
                pos -= len(self._read_buf) - self._read_pos
            pos = _BufferedIOMixin.seek(self, pos, whence)
            self._reset_read_buf()
            return pos

class BufferedWriter(_BufferedIOMixin):
    """A buffer for a writeable sequential RawIO object.

    The constructor creates a BufferedWriter for the given writeable raw
    stream. If the buffer_size is not given, it defaults to
    DEFAULT_BUFFER_SIZE.
    """

    def __init__(self, raw, buffer_size=DEFAULT_BUFFER_SIZE):
        if False:
            while True:
                i = 10
        if not raw.writable():
            raise OSError('"raw" argument must be writable.')
        _BufferedIOMixin.__init__(self, raw)
        if buffer_size <= 0:
            raise ValueError('invalid buffer size')
        self.buffer_size = buffer_size
        self._write_buf = bytearray()
        self._write_lock = Lock()

    def writable(self):
        if False:
            for i in range(10):
                print('nop')
        return self.raw.writable()

    def write(self, b):
        if False:
            print('Hello World!')
        if isinstance(b, str):
            raise TypeError("can't write str to binary stream")
        with self._write_lock:
            if self.closed:
                raise ValueError('write to closed file')
            if len(self._write_buf) > self.buffer_size:
                self._flush_unlocked()
            before = len(self._write_buf)
            self._write_buf.extend(b)
            written = len(self._write_buf) - before
            if len(self._write_buf) > self.buffer_size:
                try:
                    self._flush_unlocked()
                except BlockingIOError as e:
                    if len(self._write_buf) > self.buffer_size:
                        overage = len(self._write_buf) - self.buffer_size
                        written -= overage
                        self._write_buf = self._write_buf[:self.buffer_size]
                        raise BlockingIOError(e.errno, e.strerror, written)
            return written

    def truncate(self, pos=None):
        if False:
            return 10
        with self._write_lock:
            self._flush_unlocked()
            if pos is None:
                pos = self.raw.tell()
            return self.raw.truncate(pos)

    def flush(self):
        if False:
            i = 10
            return i + 15
        with self._write_lock:
            self._flush_unlocked()

    def _flush_unlocked(self):
        if False:
            print('Hello World!')
        if self.closed:
            raise ValueError('flush on closed file')
        while self._write_buf:
            try:
                n = self.raw.write(self._write_buf)
            except BlockingIOError:
                raise RuntimeError('self.raw should implement RawIOBase: it should not raise BlockingIOError')
            if n is None:
                raise BlockingIOError(errno.EAGAIN, 'write could not complete without blocking', 0)
            if n > len(self._write_buf) or n < 0:
                raise OSError('write() returned incorrect number of bytes')
            del self._write_buf[:n]

    def tell(self):
        if False:
            while True:
                i = 10
        return _BufferedIOMixin.tell(self) + len(self._write_buf)

    def seek(self, pos, whence=0):
        if False:
            for i in range(10):
                print('nop')
        if whence not in valid_seek_flags:
            raise ValueError('invalid whence value')
        with self._write_lock:
            self._flush_unlocked()
            return _BufferedIOMixin.seek(self, pos, whence)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        with self._write_lock:
            if self.raw is None or self.closed:
                return
        try:
            self.flush()
        finally:
            with self._write_lock:
                self.raw.close()

class BufferedRWPair(BufferedIOBase):
    """A buffered reader and writer object together.

    A buffered reader object and buffered writer object put together to
    form a sequential IO object that can read and write. This is typically
    used with a socket or two-way pipe.

    reader and writer are RawIOBase objects that are readable and
    writeable respectively. If the buffer_size is omitted it defaults to
    DEFAULT_BUFFER_SIZE.
    """

    def __init__(self, reader, writer, buffer_size=DEFAULT_BUFFER_SIZE):
        if False:
            print('Hello World!')
        'Constructor.\n\n        The arguments are two RawIO instances.\n        '
        if not reader.readable():
            raise OSError('"reader" argument must be readable.')
        if not writer.writable():
            raise OSError('"writer" argument must be writable.')
        self.reader = BufferedReader(reader, buffer_size)
        self.writer = BufferedWriter(writer, buffer_size)

    def read(self, size=-1):
        if False:
            return 10
        if size is None:
            size = -1
        return self.reader.read(size)

    def readinto(self, b):
        if False:
            for i in range(10):
                print('nop')
        return self.reader.readinto(b)

    def write(self, b):
        if False:
            print('Hello World!')
        return self.writer.write(b)

    def peek(self, size=0):
        if False:
            while True:
                i = 10
        return self.reader.peek(size)

    def read1(self, size=-1):
        if False:
            return 10
        return self.reader.read1(size)

    def readinto1(self, b):
        if False:
            return 10
        return self.reader.readinto1(b)

    def readable(self):
        if False:
            while True:
                i = 10
        return self.reader.readable()

    def writable(self):
        if False:
            print('Hello World!')
        return self.writer.writable()

    def flush(self):
        if False:
            while True:
                i = 10
        return self.writer.flush()

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.writer.close()
        finally:
            self.reader.close()

    def isatty(self):
        if False:
            print('Hello World!')
        return self.reader.isatty() or self.writer.isatty()

    @property
    def closed(self):
        if False:
            return 10
        return self.writer.closed

class BufferedRandom(BufferedWriter, BufferedReader):
    """A buffered interface to random access streams.

    The constructor creates a reader and writer for a seekable stream,
    raw, given in the first argument. If the buffer_size is omitted it
    defaults to DEFAULT_BUFFER_SIZE.
    """

    def __init__(self, raw, buffer_size=DEFAULT_BUFFER_SIZE):
        if False:
            print('Hello World!')
        raw._checkSeekable()
        BufferedReader.__init__(self, raw, buffer_size)
        BufferedWriter.__init__(self, raw, buffer_size)

    def seek(self, pos, whence=0):
        if False:
            i = 10
            return i + 15
        if whence not in valid_seek_flags:
            raise ValueError('invalid whence value')
        self.flush()
        if self._read_buf:
            with self._read_lock:
                self.raw.seek(self._read_pos - len(self._read_buf), 1)
        pos = self.raw.seek(pos, whence)
        with self._read_lock:
            self._reset_read_buf()
        if pos < 0:
            raise OSError('seek() returned invalid position')
        return pos

    def tell(self):
        if False:
            for i in range(10):
                print('nop')
        if self._write_buf:
            return BufferedWriter.tell(self)
        else:
            return BufferedReader.tell(self)

    def truncate(self, pos=None):
        if False:
            print('Hello World!')
        if pos is None:
            pos = self.tell()
        return BufferedWriter.truncate(self, pos)

    def read(self, size=None):
        if False:
            print('Hello World!')
        if size is None:
            size = -1
        self.flush()
        return BufferedReader.read(self, size)

    def readinto(self, b):
        if False:
            for i in range(10):
                print('nop')
        self.flush()
        return BufferedReader.readinto(self, b)

    def peek(self, size=0):
        if False:
            for i in range(10):
                print('nop')
        self.flush()
        return BufferedReader.peek(self, size)

    def read1(self, size=-1):
        if False:
            print('Hello World!')
        self.flush()
        return BufferedReader.read1(self, size)

    def readinto1(self, b):
        if False:
            i = 10
            return i + 15
        self.flush()
        return BufferedReader.readinto1(self, b)

    def write(self, b):
        if False:
            for i in range(10):
                print('nop')
        if self._read_buf:
            with self._read_lock:
                self.raw.seek(self._read_pos - len(self._read_buf), 1)
                self._reset_read_buf()
        return BufferedWriter.write(self, b)

class FileIO(RawIOBase):
    _fd = -1
    _created = False
    _readable = False
    _writable = False
    _appending = False
    _seekable = None
    _closefd = True

    def __init__(self, file, mode='r', closefd=True, opener=None):
        if False:
            print('Hello World!')
        "Open a file.  The mode can be 'r' (default), 'w', 'x' or 'a' for reading,\n        writing, exclusive creation or appending.  The file will be created if it\n        doesn't exist when opened for writing or appending; it will be truncated\n        when opened for writing.  A FileExistsError will be raised if it already\n        exists when opened for creating. Opening a file for creating implies\n        writing so this mode behaves in a similar way to 'w'. Add a '+' to the mode\n        to allow simultaneous reading and writing. A custom opener can be used by\n        passing a callable as *opener*. The underlying file descriptor for the file\n        object is then obtained by calling opener with (*name*, *flags*).\n        *opener* must return an open file descriptor (passing os.open as *opener*\n        results in functionality similar to passing None).\n        "
        if self._fd >= 0:
            try:
                if self._closefd:
                    os.close(self._fd)
            finally:
                self._fd = -1
        if isinstance(file, float):
            raise TypeError('integer argument expected, got float')
        if isinstance(file, int):
            fd = file
            if fd < 0:
                raise ValueError('negative file descriptor')
        else:
            fd = -1
        if not isinstance(mode, str):
            raise TypeError('invalid mode: %s' % (mode,))
        if not set(mode) <= set('xrwab+'):
            raise ValueError('invalid mode: %s' % (mode,))
        if sum((c in 'rwax' for c in mode)) != 1 or mode.count('+') > 1:
            raise ValueError('Must have exactly one of create/read/write/append mode and at most one plus')
        if 'x' in mode:
            self._created = True
            self._writable = True
            flags = os.O_EXCL | os.O_CREAT
        elif 'r' in mode:
            self._readable = True
            flags = 0
        elif 'w' in mode:
            self._writable = True
            flags = os.O_CREAT | os.O_TRUNC
        elif 'a' in mode:
            self._writable = True
            self._appending = True
            flags = os.O_APPEND | os.O_CREAT
        if '+' in mode:
            self._readable = True
            self._writable = True
        if self._readable and self._writable:
            flags |= os.O_RDWR
        elif self._readable:
            flags |= os.O_RDONLY
        else:
            flags |= os.O_WRONLY
        flags |= getattr(os, 'O_BINARY', 0)
        noinherit_flag = getattr(os, 'O_NOINHERIT', 0) or getattr(os, 'O_CLOEXEC', 0)
        flags |= noinherit_flag
        owned_fd = None
        try:
            if fd < 0:
                if not closefd:
                    raise ValueError('Cannot use closefd=False with file name')
                if opener is None:
                    fd = os.open(file, flags, 438)
                else:
                    fd = opener(file, flags)
                    if not isinstance(fd, int):
                        raise TypeError('expected integer from opener')
                    if fd < 0:
                        raise OSError('Negative file descriptor')
                owned_fd = fd
                if not noinherit_flag:
                    os.set_inheritable(fd, False)
            self._closefd = closefd
            fdfstat = os.fstat(fd)
            try:
                if stat.S_ISDIR(fdfstat.st_mode):
                    raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), file)
            except AttributeError:
                pass
            self._blksize = getattr(fdfstat, 'st_blksize', 0)
            if self._blksize <= 1:
                self._blksize = DEFAULT_BUFFER_SIZE
            if _setmode:
                _setmode(fd, os.O_BINARY)
            self.name = file
            if self._appending:
                try:
                    os.lseek(fd, 0, SEEK_END)
                except OSError as e:
                    if e.errno != errno.ESPIPE:
                        raise
        except:
            if owned_fd is not None:
                os.close(owned_fd)
            raise
        self._fd = fd

    def __del__(self):
        if False:
            i = 10
            return i + 15
        if self._fd >= 0 and self._closefd and (not self.closed):
            import warnings
            warnings.warn('unclosed file %r' % (self,), ResourceWarning, stacklevel=2, source=self)
            self.close()

    def __getstate__(self):
        if False:
            return 10
        raise TypeError(f'cannot pickle {self.__class__.__name__!r} object')

    def __repr__(self):
        if False:
            return 10
        class_name = '%s.%s' % (self.__class__.__module__, self.__class__.__qualname__)
        if self.closed:
            return '<%s [closed]>' % class_name
        try:
            name = self.name
        except AttributeError:
            return '<%s fd=%d mode=%r closefd=%r>' % (class_name, self._fd, self.mode, self._closefd)
        else:
            return '<%s name=%r mode=%r closefd=%r>' % (class_name, name, self.mode, self._closefd)

    def _checkReadable(self):
        if False:
            print('Hello World!')
        if not self._readable:
            raise UnsupportedOperation('File not open for reading')

    def _checkWritable(self, msg=None):
        if False:
            i = 10
            return i + 15
        if not self._writable:
            raise UnsupportedOperation('File not open for writing')

    def read(self, size=None):
        if False:
            print('Hello World!')
        'Read at most size bytes, returned as bytes.\n\n        Only makes one system call, so less data may be returned than requested\n        In non-blocking mode, returns None if no data is available.\n        Return an empty bytes object at EOF.\n        '
        self._checkClosed()
        self._checkReadable()
        if size is None or size < 0:
            return self.readall()
        try:
            return os.read(self._fd, size)
        except BlockingIOError:
            return None

    def readall(self):
        if False:
            for i in range(10):
                print('nop')
        'Read all data from the file, returned as bytes.\n\n        In non-blocking mode, returns as much as is immediately available,\n        or None if no data is available.  Return an empty bytes object at EOF.\n        '
        self._checkClosed()
        self._checkReadable()
        bufsize = DEFAULT_BUFFER_SIZE
        try:
            pos = os.lseek(self._fd, 0, SEEK_CUR)
            end = os.fstat(self._fd).st_size
            if end >= pos:
                bufsize = end - pos + 1
        except OSError:
            pass
        result = bytearray()
        while True:
            if len(result) >= bufsize:
                bufsize = len(result)
                bufsize += max(bufsize, DEFAULT_BUFFER_SIZE)
            n = bufsize - len(result)
            try:
                chunk = os.read(self._fd, n)
            except BlockingIOError:
                if result:
                    break
                return None
            if not chunk:
                break
            result += chunk
        return bytes(result)

    def readinto(self, b):
        if False:
            i = 10
            return i + 15
        'Same as RawIOBase.readinto().'
        m = memoryview(b).cast('B')
        data = self.read(len(m))
        n = len(data)
        m[:n] = data
        return n

    def write(self, b):
        if False:
            for i in range(10):
                print('nop')
        'Write bytes b to file, return number written.\n\n        Only makes one system call, so not all of the data may be written.\n        The number of bytes actually written is returned.  In non-blocking mode,\n        returns None if the write would block.\n        '
        self._checkClosed()
        self._checkWritable()
        try:
            return os.write(self._fd, b)
        except BlockingIOError:
            return None

    def seek(self, pos, whence=SEEK_SET):
        if False:
            for i in range(10):
                print('nop')
        'Move to new file position.\n\n        Argument offset is a byte count.  Optional argument whence defaults to\n        SEEK_SET or 0 (offset from start of file, offset should be >= 0); other values\n        are SEEK_CUR or 1 (move relative to current position, positive or negative),\n        and SEEK_END or 2 (move relative to end of file, usually negative, although\n        many platforms allow seeking beyond the end of a file).\n\n        Note that not all file objects are seekable.\n        '
        if isinstance(pos, float):
            raise TypeError('an integer is required')
        self._checkClosed()
        return os.lseek(self._fd, pos, whence)

    def tell(self):
        if False:
            while True:
                i = 10
        'tell() -> int.  Current file position.\n\n        Can raise OSError for non seekable files.'
        self._checkClosed()
        return os.lseek(self._fd, 0, SEEK_CUR)

    def truncate(self, size=None):
        if False:
            while True:
                i = 10
        'Truncate the file to at most size bytes.\n\n        Size defaults to the current file position, as returned by tell().\n        The current file position is changed to the value of size.\n        '
        self._checkClosed()
        self._checkWritable()
        if size is None:
            size = self.tell()
        os.ftruncate(self._fd, size)
        return size

    def close(self):
        if False:
            return 10
        'Close the file.\n\n        A closed file cannot be used for further I/O operations.  close() may be\n        called more than once without error.\n        '
        if not self.closed:
            try:
                if self._closefd:
                    os.close(self._fd)
            finally:
                super().close()

    def seekable(self):
        if False:
            while True:
                i = 10
        'True if file supports random-access.'
        self._checkClosed()
        if self._seekable is None:
            try:
                self.tell()
            except OSError:
                self._seekable = False
            else:
                self._seekable = True
        return self._seekable

    def readable(self):
        if False:
            return 10
        'True if file was opened in a read mode.'
        self._checkClosed()
        return self._readable

    def writable(self):
        if False:
            while True:
                i = 10
        'True if file was opened in a write mode.'
        self._checkClosed()
        return self._writable

    def fileno(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the underlying file descriptor (an integer).'
        self._checkClosed()
        return self._fd

    def isatty(self):
        if False:
            print('Hello World!')
        'True if the file is connected to a TTY device.'
        self._checkClosed()
        return os.isatty(self._fd)

    @property
    def closefd(self):
        if False:
            return 10
        'True if the file descriptor will be closed by close().'
        return self._closefd

    @property
    def mode(self):
        if False:
            while True:
                i = 10
        'String giving the file mode'
        if self._created:
            if self._readable:
                return 'xb+'
            else:
                return 'xb'
        elif self._appending:
            if self._readable:
                return 'ab+'
            else:
                return 'ab'
        elif self._readable:
            if self._writable:
                return 'rb+'
            else:
                return 'rb'
        else:
            return 'wb'

class TextIOBase(IOBase):
    """Base class for text I/O.

    This class provides a character and line based interface to stream
    I/O.
    """

    def read(self, size=-1):
        if False:
            return 10
        'Read at most size characters from stream, where size is an int.\n\n        Read from underlying buffer until we have size characters or we hit EOF.\n        If size is negative or omitted, read until EOF.\n\n        Returns a string.\n        '
        self._unsupported('read')

    def write(self, s):
        if False:
            print('Hello World!')
        'Write string s to stream and returning an int.'
        self._unsupported('write')

    def truncate(self, pos=None):
        if False:
            print('Hello World!')
        'Truncate size to pos, where pos is an int.'
        self._unsupported('truncate')

    def readline(self):
        if False:
            return 10
        'Read until newline or EOF.\n\n        Returns an empty string if EOF is hit immediately.\n        '
        self._unsupported('readline')

    def detach(self):
        if False:
            while True:
                i = 10
        '\n        Separate the underlying buffer from the TextIOBase and return it.\n\n        After the underlying buffer has been detached, the TextIO is in an\n        unusable state.\n        '
        self._unsupported('detach')

    @property
    def encoding(self):
        if False:
            i = 10
            return i + 15
        'Subclasses should override.'
        return None

    @property
    def newlines(self):
        if False:
            return 10
        'Line endings translated so far.\n\n        Only line endings translated during reading are considered.\n\n        Subclasses should override.\n        '
        return None

    @property
    def errors(self):
        if False:
            for i in range(10):
                print('nop')
        'Error setting of the decoder or encoder.\n\n        Subclasses should override.'
        return None
io.TextIOBase.register(TextIOBase)

class IncrementalNewlineDecoder(codecs.IncrementalDecoder):
    """Codec used when reading a file in universal newlines mode.  It wraps
    another incremental decoder, translating \\r\\n and \\r into \\n.  It also
    records the types of newlines encountered.  When used with
    translate=False, it ensures that the newline sequence is returned in
    one piece.
    """

    def __init__(self, decoder, translate, errors='strict'):
        if False:
            i = 10
            return i + 15
        codecs.IncrementalDecoder.__init__(self, errors=errors)
        self.translate = translate
        self.decoder = decoder
        self.seennl = 0
        self.pendingcr = False

    def decode(self, input, final=False):
        if False:
            i = 10
            return i + 15
        if self.decoder is None:
            output = input
        else:
            output = self.decoder.decode(input, final=final)
        if self.pendingcr and (output or final):
            output = '\r' + output
            self.pendingcr = False
        if output.endswith('\r') and (not final):
            output = output[:-1]
            self.pendingcr = True
        crlf = output.count('\r\n')
        cr = output.count('\r') - crlf
        lf = output.count('\n') - crlf
        self.seennl |= (lf and self._LF) | (cr and self._CR) | (crlf and self._CRLF)
        if self.translate:
            if crlf:
                output = output.replace('\r\n', '\n')
            if cr:
                output = output.replace('\r', '\n')
        return output

    def getstate(self):
        if False:
            for i in range(10):
                print('nop')
        if self.decoder is None:
            buf = b''
            flag = 0
        else:
            (buf, flag) = self.decoder.getstate()
        flag <<= 1
        if self.pendingcr:
            flag |= 1
        return (buf, flag)

    def setstate(self, state):
        if False:
            for i in range(10):
                print('nop')
        (buf, flag) = state
        self.pendingcr = bool(flag & 1)
        if self.decoder is not None:
            self.decoder.setstate((buf, flag >> 1))

    def reset(self):
        if False:
            while True:
                i = 10
        self.seennl = 0
        self.pendingcr = False
        if self.decoder is not None:
            self.decoder.reset()
    _LF = 1
    _CR = 2
    _CRLF = 4

    @property
    def newlines(self):
        if False:
            i = 10
            return i + 15
        return (None, '\n', '\r', ('\r', '\n'), '\r\n', ('\n', '\r\n'), ('\r', '\r\n'), ('\r', '\n', '\r\n'))[self.seennl]

class TextIOWrapper(TextIOBase):
    """Character and line based layer over a BufferedIOBase object, buffer.

    encoding gives the name of the encoding that the stream will be
    decoded or encoded with. It defaults to locale.getpreferredencoding(False).

    errors determines the strictness of encoding and decoding (see the
    codecs.register) and defaults to "strict".

    newline can be None, '', '\\n', '\\r', or '\\r\\n'.  It controls the
    handling of line endings. If it is None, universal newlines is
    enabled.  With this enabled, on input, the lines endings '\\n', '\\r',
    or '\\r\\n' are translated to '\\n' before being returned to the
    caller. Conversely, on output, '\\n' is translated to the system
    default line separator, os.linesep. If newline is any other of its
    legal values, that newline becomes the newline when the file is read
    and it is returned untranslated. On output, '\\n' is converted to the
    newline.

    If line_buffering is True, a call to flush is implied when a call to
    write contains a newline character.
    """
    _CHUNK_SIZE = 2048
    _buffer = None

    def __init__(self, buffer, encoding=None, errors=None, newline=None, line_buffering=False, write_through=False):
        if False:
            for i in range(10):
                print('nop')
        self._check_newline(newline)
        encoding = text_encoding(encoding)
        if encoding == 'locale':
            try:
                encoding = os.device_encoding(buffer.fileno()) or 'locale'
            except (AttributeError, UnsupportedOperation):
                pass
        if encoding == 'locale':
            try:
                import locale
            except ImportError:
                encoding = 'utf-8'
            else:
                encoding = locale.getpreferredencoding(False)
        if not isinstance(encoding, str):
            raise ValueError('invalid encoding: %r' % encoding)
        if not codecs.lookup(encoding)._is_text_encoding:
            msg = '%r is not a text encoding; use codecs.open() to handle arbitrary codecs'
            raise LookupError(msg % encoding)
        if errors is None:
            errors = 'strict'
        else:
            if not isinstance(errors, str):
                raise ValueError('invalid errors: %r' % errors)
            if _CHECK_ERRORS:
                codecs.lookup_error(errors)
        self._buffer = buffer
        self._decoded_chars = ''
        self._decoded_chars_used = 0
        self._snapshot = None
        self._seekable = self._telling = self.buffer.seekable()
        self._has_read1 = hasattr(self.buffer, 'read1')
        self._configure(encoding, errors, newline, line_buffering, write_through)

    def _check_newline(self, newline):
        if False:
            for i in range(10):
                print('nop')
        if newline is not None and (not isinstance(newline, str)):
            raise TypeError('illegal newline type: %r' % (type(newline),))
        if newline not in (None, '', '\n', '\r', '\r\n'):
            raise ValueError('illegal newline value: %r' % (newline,))

    def _configure(self, encoding=None, errors=None, newline=None, line_buffering=False, write_through=False):
        if False:
            i = 10
            return i + 15
        self._encoding = encoding
        self._errors = errors
        self._encoder = None
        self._decoder = None
        self._b2cratio = 0.0
        self._readuniversal = not newline
        self._readtranslate = newline is None
        self._readnl = newline
        self._writetranslate = newline != ''
        self._writenl = newline or os.linesep
        self._line_buffering = line_buffering
        self._write_through = write_through
        if self._seekable and self.writable():
            position = self.buffer.tell()
            if position != 0:
                try:
                    self._get_encoder().setstate(0)
                except LookupError:
                    pass

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        result = '<{}.{}'.format(self.__class__.__module__, self.__class__.__qualname__)
        try:
            name = self.name
        except AttributeError:
            pass
        else:
            result += ' name={0!r}'.format(name)
        try:
            mode = self.mode
        except AttributeError:
            pass
        else:
            result += ' mode={0!r}'.format(mode)
        return result + ' encoding={0!r}>'.format(self.encoding)

    @property
    def encoding(self):
        if False:
            for i in range(10):
                print('nop')
        return self._encoding

    @property
    def errors(self):
        if False:
            while True:
                i = 10
        return self._errors

    @property
    def line_buffering(self):
        if False:
            return 10
        return self._line_buffering

    @property
    def write_through(self):
        if False:
            i = 10
            return i + 15
        return self._write_through

    @property
    def buffer(self):
        if False:
            while True:
                i = 10
        return self._buffer

    def reconfigure(self, *, encoding=None, errors=None, newline=Ellipsis, line_buffering=None, write_through=None):
        if False:
            print('Hello World!')
        'Reconfigure the text stream with new parameters.\n\n        This also flushes the stream.\n        '
        if self._decoder is not None and (encoding is not None or errors is not None or newline is not Ellipsis):
            raise UnsupportedOperation('It is not possible to set the encoding or newline of stream after the first read')
        if errors is None:
            if encoding is None:
                errors = self._errors
            else:
                errors = 'strict'
        elif not isinstance(errors, str):
            raise TypeError('invalid errors: %r' % errors)
        if encoding is None:
            encoding = self._encoding
        elif not isinstance(encoding, str):
            raise TypeError('invalid encoding: %r' % encoding)
        if newline is Ellipsis:
            newline = self._readnl
        self._check_newline(newline)
        if line_buffering is None:
            line_buffering = self.line_buffering
        if write_through is None:
            write_through = self.write_through
        self.flush()
        self._configure(encoding, errors, newline, line_buffering, write_through)

    def seekable(self):
        if False:
            for i in range(10):
                print('nop')
        if self.closed:
            raise ValueError('I/O operation on closed file.')
        return self._seekable

    def readable(self):
        if False:
            i = 10
            return i + 15
        return self.buffer.readable()

    def writable(self):
        if False:
            while True:
                i = 10
        return self.buffer.writable()

    def flush(self):
        if False:
            for i in range(10):
                print('nop')
        self.buffer.flush()
        self._telling = self._seekable

    def close(self):
        if False:
            return 10
        if self.buffer is not None and (not self.closed):
            try:
                self.flush()
            finally:
                self.buffer.close()

    @property
    def closed(self):
        if False:
            print('Hello World!')
        return self.buffer.closed

    @property
    def name(self):
        if False:
            print('Hello World!')
        return self.buffer.name

    def fileno(self):
        if False:
            while True:
                i = 10
        return self.buffer.fileno()

    def isatty(self):
        if False:
            i = 10
            return i + 15
        return self.buffer.isatty()

    def write(self, s):
        if False:
            while True:
                i = 10
        'Write data, where s is a str'
        if self.closed:
            raise ValueError('write to closed file')
        if not isinstance(s, str):
            raise TypeError("can't write %s to text stream" % s.__class__.__name__)
        length = len(s)
        haslf = (self._writetranslate or self._line_buffering) and '\n' in s
        if haslf and self._writetranslate and (self._writenl != '\n'):
            s = s.replace('\n', self._writenl)
        encoder = self._encoder or self._get_encoder()
        b = encoder.encode(s)
        self.buffer.write(b)
        if self._line_buffering and (haslf or '\r' in s):
            self.flush()
        self._set_decoded_chars('')
        self._snapshot = None
        if self._decoder:
            self._decoder.reset()
        return length

    def _get_encoder(self):
        if False:
            for i in range(10):
                print('nop')
        make_encoder = codecs.getincrementalencoder(self._encoding)
        self._encoder = make_encoder(self._errors)
        return self._encoder

    def _get_decoder(self):
        if False:
            print('Hello World!')
        make_decoder = codecs.getincrementaldecoder(self._encoding)
        decoder = make_decoder(self._errors)
        if self._readuniversal:
            decoder = IncrementalNewlineDecoder(decoder, self._readtranslate)
        self._decoder = decoder
        return decoder

    def _set_decoded_chars(self, chars):
        if False:
            print('Hello World!')
        'Set the _decoded_chars buffer.'
        self._decoded_chars = chars
        self._decoded_chars_used = 0

    def _get_decoded_chars(self, n=None):
        if False:
            i = 10
            return i + 15
        'Advance into the _decoded_chars buffer.'
        offset = self._decoded_chars_used
        if n is None:
            chars = self._decoded_chars[offset:]
        else:
            chars = self._decoded_chars[offset:offset + n]
        self._decoded_chars_used += len(chars)
        return chars

    def _rewind_decoded_chars(self, n):
        if False:
            i = 10
            return i + 15
        'Rewind the _decoded_chars buffer.'
        if self._decoded_chars_used < n:
            raise AssertionError('rewind decoded_chars out of bounds')
        self._decoded_chars_used -= n

    def _read_chunk(self):
        if False:
            i = 10
            return i + 15
        '\n        Read and decode the next chunk of data from the BufferedReader.\n        '
        if self._decoder is None:
            raise ValueError('no decoder')
        if self._telling:
            (dec_buffer, dec_flags) = self._decoder.getstate()
        if self._has_read1:
            input_chunk = self.buffer.read1(self._CHUNK_SIZE)
        else:
            input_chunk = self.buffer.read(self._CHUNK_SIZE)
        eof = not input_chunk
        decoded_chars = self._decoder.decode(input_chunk, eof)
        self._set_decoded_chars(decoded_chars)
        if decoded_chars:
            self._b2cratio = len(input_chunk) / len(self._decoded_chars)
        else:
            self._b2cratio = 0.0
        if self._telling:
            self._snapshot = (dec_flags, dec_buffer + input_chunk)
        return not eof

    def _pack_cookie(self, position, dec_flags=0, bytes_to_feed=0, need_eof=False, chars_to_skip=0):
        if False:
            return 10
        return position | dec_flags << 64 | bytes_to_feed << 128 | chars_to_skip << 192 | bool(need_eof) << 256

    def _unpack_cookie(self, bigint):
        if False:
            return 10
        (rest, position) = divmod(bigint, 1 << 64)
        (rest, dec_flags) = divmod(rest, 1 << 64)
        (rest, bytes_to_feed) = divmod(rest, 1 << 64)
        (need_eof, chars_to_skip) = divmod(rest, 1 << 64)
        return (position, dec_flags, bytes_to_feed, bool(need_eof), chars_to_skip)

    def tell(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._seekable:
            raise UnsupportedOperation('underlying stream is not seekable')
        if not self._telling:
            raise OSError('telling position disabled by next() call')
        self.flush()
        position = self.buffer.tell()
        decoder = self._decoder
        if decoder is None or self._snapshot is None:
            if self._decoded_chars:
                raise AssertionError('pending decoded text')
            return position
        (dec_flags, next_input) = self._snapshot
        position -= len(next_input)
        chars_to_skip = self._decoded_chars_used
        if chars_to_skip == 0:
            return self._pack_cookie(position, dec_flags)
        saved_state = decoder.getstate()
        try:
            skip_bytes = int(self._b2cratio * chars_to_skip)
            skip_back = 1
            assert skip_bytes <= len(next_input)
            while skip_bytes > 0:
                decoder.setstate((b'', dec_flags))
                n = len(decoder.decode(next_input[:skip_bytes]))
                if n <= chars_to_skip:
                    (b, d) = decoder.getstate()
                    if not b:
                        dec_flags = d
                        chars_to_skip -= n
                        break
                    skip_bytes -= len(b)
                    skip_back = 1
                else:
                    skip_bytes -= skip_back
                    skip_back = skip_back * 2
            else:
                skip_bytes = 0
                decoder.setstate((b'', dec_flags))
            start_pos = position + skip_bytes
            start_flags = dec_flags
            if chars_to_skip == 0:
                return self._pack_cookie(start_pos, start_flags)
            bytes_fed = 0
            need_eof = False
            chars_decoded = 0
            for i in range(skip_bytes, len(next_input)):
                bytes_fed += 1
                chars_decoded += len(decoder.decode(next_input[i:i + 1]))
                (dec_buffer, dec_flags) = decoder.getstate()
                if not dec_buffer and chars_decoded <= chars_to_skip:
                    start_pos += bytes_fed
                    chars_to_skip -= chars_decoded
                    (start_flags, bytes_fed, chars_decoded) = (dec_flags, 0, 0)
                if chars_decoded >= chars_to_skip:
                    break
            else:
                chars_decoded += len(decoder.decode(b'', final=True))
                need_eof = True
                if chars_decoded < chars_to_skip:
                    raise OSError("can't reconstruct logical file position")
            return self._pack_cookie(start_pos, start_flags, bytes_fed, need_eof, chars_to_skip)
        finally:
            decoder.setstate(saved_state)

    def truncate(self, pos=None):
        if False:
            for i in range(10):
                print('nop')
        self.flush()
        if pos is None:
            pos = self.tell()
        return self.buffer.truncate(pos)

    def detach(self):
        if False:
            print('Hello World!')
        if self.buffer is None:
            raise ValueError('buffer is already detached')
        self.flush()
        buffer = self._buffer
        self._buffer = None
        return buffer

    def seek(self, cookie, whence=0):
        if False:
            i = 10
            return i + 15

        def _reset_encoder(position):
            if False:
                for i in range(10):
                    print('nop')
            'Reset the encoder (merely useful for proper BOM handling)'
            try:
                encoder = self._encoder or self._get_encoder()
            except LookupError:
                pass
            else:
                if position != 0:
                    encoder.setstate(0)
                else:
                    encoder.reset()
        if self.closed:
            raise ValueError('tell on closed file')
        if not self._seekable:
            raise UnsupportedOperation('underlying stream is not seekable')
        if whence == SEEK_CUR:
            if cookie != 0:
                raise UnsupportedOperation("can't do nonzero cur-relative seeks")
            whence = 0
            cookie = self.tell()
        elif whence == SEEK_END:
            if cookie != 0:
                raise UnsupportedOperation("can't do nonzero end-relative seeks")
            self.flush()
            position = self.buffer.seek(0, whence)
            self._set_decoded_chars('')
            self._snapshot = None
            if self._decoder:
                self._decoder.reset()
            _reset_encoder(position)
            return position
        if whence != 0:
            raise ValueError('unsupported whence (%r)' % (whence,))
        if cookie < 0:
            raise ValueError('negative seek position %r' % (cookie,))
        self.flush()
        (start_pos, dec_flags, bytes_to_feed, need_eof, chars_to_skip) = self._unpack_cookie(cookie)
        self.buffer.seek(start_pos)
        self._set_decoded_chars('')
        self._snapshot = None
        if cookie == 0 and self._decoder:
            self._decoder.reset()
        elif self._decoder or dec_flags or chars_to_skip:
            self._decoder = self._decoder or self._get_decoder()
            self._decoder.setstate((b'', dec_flags))
            self._snapshot = (dec_flags, b'')
        if chars_to_skip:
            input_chunk = self.buffer.read(bytes_to_feed)
            self._set_decoded_chars(self._decoder.decode(input_chunk, need_eof))
            self._snapshot = (dec_flags, input_chunk)
            if len(self._decoded_chars) < chars_to_skip:
                raise OSError("can't restore logical file position")
            self._decoded_chars_used = chars_to_skip
        _reset_encoder(cookie)
        return cookie

    def read(self, size=None):
        if False:
            print('Hello World!')
        self._checkReadable()
        if size is None:
            size = -1
        else:
            try:
                size_index = size.__index__
            except AttributeError:
                raise TypeError(f'{size!r} is not an integer')
            else:
                size = size_index()
        decoder = self._decoder or self._get_decoder()
        if size < 0:
            result = self._get_decoded_chars() + decoder.decode(self.buffer.read(), final=True)
            self._set_decoded_chars('')
            self._snapshot = None
            return result
        else:
            eof = False
            result = self._get_decoded_chars(size)
            while len(result) < size and (not eof):
                eof = not self._read_chunk()
                result += self._get_decoded_chars(size - len(result))
            return result

    def __next__(self):
        if False:
            while True:
                i = 10
        self._telling = False
        line = self.readline()
        if not line:
            self._snapshot = None
            self._telling = self._seekable
            raise StopIteration
        return line

    def readline(self, size=None):
        if False:
            for i in range(10):
                print('nop')
        if self.closed:
            raise ValueError('read from closed file')
        if size is None:
            size = -1
        else:
            try:
                size_index = size.__index__
            except AttributeError:
                raise TypeError(f'{size!r} is not an integer')
            else:
                size = size_index()
        line = self._get_decoded_chars()
        start = 0
        if not self._decoder:
            self._get_decoder()
        pos = endpos = None
        while True:
            if self._readtranslate:
                pos = line.find('\n', start)
                if pos >= 0:
                    endpos = pos + 1
                    break
                else:
                    start = len(line)
            elif self._readuniversal:
                nlpos = line.find('\n', start)
                crpos = line.find('\r', start)
                if crpos == -1:
                    if nlpos == -1:
                        start = len(line)
                    else:
                        endpos = nlpos + 1
                        break
                elif nlpos == -1:
                    endpos = crpos + 1
                    break
                elif nlpos < crpos:
                    endpos = nlpos + 1
                    break
                elif nlpos == crpos + 1:
                    endpos = crpos + 2
                    break
                else:
                    endpos = crpos + 1
                    break
            else:
                pos = line.find(self._readnl)
                if pos >= 0:
                    endpos = pos + len(self._readnl)
                    break
            if size >= 0 and len(line) >= size:
                endpos = size
                break
            while self._read_chunk():
                if self._decoded_chars:
                    break
            if self._decoded_chars:
                line += self._get_decoded_chars()
            else:
                self._set_decoded_chars('')
                self._snapshot = None
                return line
        if size >= 0 and endpos > size:
            endpos = size
        self._rewind_decoded_chars(len(line) - endpos)
        return line[:endpos]

    @property
    def newlines(self):
        if False:
            for i in range(10):
                print('nop')
        return self._decoder.newlines if self._decoder else None

class StringIO(TextIOWrapper):
    """Text I/O implementation using an in-memory buffer.

    The initial_value argument sets the value of object.  The newline
    argument is like the one of TextIOWrapper's constructor.
    """

    def __init__(self, initial_value='', newline='\n'):
        if False:
            return 10
        super(StringIO, self).__init__(BytesIO(), encoding='utf-8', errors='surrogatepass', newline=newline)
        if newline is None:
            self._writetranslate = False
        if initial_value is not None:
            if not isinstance(initial_value, str):
                raise TypeError('initial_value must be str or None, not {0}'.format(type(initial_value).__name__))
            self.write(initial_value)
            self.seek(0)

    def getvalue(self):
        if False:
            for i in range(10):
                print('nop')
        self.flush()
        decoder = self._decoder or self._get_decoder()
        old_state = decoder.getstate()
        decoder.reset()
        try:
            return decoder.decode(self.buffer.getvalue(), final=True)
        finally:
            decoder.setstate(old_state)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return object.__repr__(self)

    @property
    def errors(self):
        if False:
            return 10
        return None

    @property
    def encoding(self):
        if False:
            return 10
        return None

    def detach(self):
        if False:
            while True:
                i = 10
        self._unsupported('detach')