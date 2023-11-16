"""
gevent internals.
"""
from __future__ import absolute_import, print_function, division
try:
    from errno import EBADF
except ImportError:
    EBADF = 9
import io
import functools
import sys
import os
from gevent.hub import _get_hub_noargs as get_hub
from gevent._compat import integer_types
from gevent._compat import reraise
from gevent._compat import fspath
from gevent.lock import Semaphore, DummySemaphore

class cancel_wait_ex(IOError):

    def __init__(self):
        if False:
            print('Hello World!')
        IOError.__init__(self, EBADF, 'File descriptor was closed in another greenlet')

class FileObjectClosed(IOError):

    def __init__(self):
        if False:
            print('Hello World!')
        IOError.__init__(self, EBADF, 'Bad file descriptor (FileObject was closed)')

class UniversalNewlineBytesWrapper(io.TextIOWrapper):
    """
    Uses TextWrapper to decode universal newlines, but returns the
    results as bytes.

    This is for Python 2 where the 'rU' mode did that.
    """
    mode = None

    def __init__(self, fobj, line_buffering):
        if False:
            for i in range(10):
                print('nop')
        io.TextIOWrapper.__init__(self, fobj, encoding='latin-1', newline=None, line_buffering=line_buffering)

    def read(self, *args, **kwargs):
        if False:
            return 10
        result = io.TextIOWrapper.read(self, *args, **kwargs)
        return result.encode('latin-1')

    def readline(self, limit=-1):
        if False:
            for i in range(10):
                print('nop')
        result = io.TextIOWrapper.readline(self, limit)
        return result.encode('latin-1')

    def __iter__(self):
        if False:
            return 10
        return self

    def __next__(self):
        if False:
            return 10
        line = self.readline()
        if not line:
            raise StopIteration
        return line
    next = __next__

class FlushingBufferedWriter(io.BufferedWriter):

    def write(self, b):
        if False:
            return 10
        ret = io.BufferedWriter.write(self, b)
        self.flush()
        return ret

class WriteallMixin(object):

    def writeall(self, value):
        if False:
            return 10
        '\n        Similar to :meth:`socket.socket.sendall`, ensures that all the contents of\n        *value* have been written (though not necessarily flushed) before returning.\n\n        Returns the length of *value*.\n\n        .. versionadded:: 20.12.0\n        '
        write = super(WriteallMixin, self).write
        total = len(value)
        while value:
            l = len(value)
            w = write(value)
            if w == l:
                break
            value = value[w:]
        return total

class FileIO(io.FileIO):
    """A subclass that we can dynamically assign __class__ for."""
    __slots__ = ()

class WriteIsWriteallMixin(WriteallMixin):

    def write(self, value):
        if False:
            i = 10
            return i + 15
        return self.writeall(value)

class WriteallFileIO(WriteIsWriteallMixin, io.FileIO):
    pass

class OpenDescriptor(object):
    """
    Interprets the arguments to `open`. Internal use only.

    Originally based on code in the stdlib's _pyio.py (Python implementation of
    the :mod:`io` module), but modified for gevent:

    - Native strings are returned on Python 2 when neither
      'b' nor 't' are in the mode string and no encoding is specified.
    - Universal newlines work in that mode.
    - Allows externally unbuffered text IO.

    :keyword bool atomic_write: If true, then if the opened, wrapped, stream
        is unbuffered (meaning that ``write`` can produce short writes and the return
        value needs to be checked), then the implementation will be adjusted so that
        ``write`` behaves like Python 2 on a built-in file object and writes the
        entire value. Only set this on Python 2; the only intended user is
        :class:`gevent.subprocess.Popen`.
    """

    @staticmethod
    def _collapse_arg(pref_name, preferred_val, old_name, old_val, default):
        if False:
            while True:
                i = 10
        if preferred_val is not None and old_val is not None:
            raise TypeError('Cannot specify both %s=%s and %s=%s' % (pref_name, preferred_val, old_name, old_val))
        if preferred_val is None and old_val is None:
            return default
        return preferred_val if preferred_val is not None else old_val

    def __init__(self, fobj, mode='r', bufsize=None, close=None, encoding=None, errors=None, newline=None, buffering=None, closefd=None, atomic_write=False):
        if False:
            while True:
                i = 10
        closefd = self._collapse_arg('closefd', closefd, 'close', close, True)
        del close
        buffering = self._collapse_arg('buffering', buffering, 'bufsize', bufsize, -1)
        del bufsize
        if not hasattr(fobj, 'fileno'):
            if not isinstance(fobj, integer_types):
                fobj = fspath(fobj)
            if not isinstance(fobj, (str, bytes) + integer_types):
                raise TypeError('invalid file: %r' % fobj)
            if isinstance(fobj, (str, bytes)):
                closefd = True
        if not isinstance(mode, str):
            raise TypeError('invalid mode: %r' % mode)
        if not isinstance(buffering, integer_types):
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
        universal = 'U' in modes
        can_write = creating or writing or appending or updating
        if universal:
            if can_write:
                raise ValueError("mode U cannot be combined with 'x', 'w', 'a', or '+'")
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
            warnings.warn("line buffering (buffering=1) isn't supported in binary mode, the default buffer size will be used", RuntimeWarning, 4)
        self._fobj = fobj
        self.fileio_mode = (creating and 'x' or '') + (reading and 'r' or '') + (writing and 'w' or '') + (appending and 'a' or '') + (updating and '+' or '')
        self.mode = self.fileio_mode + ('t' if text else '') + ('b' if binary else '')
        self.creating = creating
        self.reading = reading
        self.writing = writing
        self.appending = appending
        self.updating = updating
        self.text = text
        self.binary = binary
        self.can_write = can_write
        self.can_read = reading or updating
        self.native = not self.text and (not self.binary) and (not encoding) and (not errors)
        self.universal = universal
        self.buffering = buffering
        self.encoding = encoding
        self.errors = errors
        self.newline = newline
        self.closefd = closefd
        self.atomic_write = atomic_write
    default_buffer_size = io.DEFAULT_BUFFER_SIZE
    _opened = None
    _opened_raw = None

    def is_fd(self):
        if False:
            return 10
        return isinstance(self._fobj, integer_types)

    def opened(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the :meth:`wrapped` file object.\n        '
        if self._opened is None:
            raw = self.opened_raw()
            try:
                self._opened = self.__wrapped(raw)
            except:
                raw.close()
                raise
        return self._opened

    def _raw_object_is_new(self, raw):
        if False:
            return 10
        return self._fobj is not raw

    def opened_raw(self):
        if False:
            while True:
                i = 10
        if self._opened_raw is None:
            self._opened_raw = self._do_open_raw()
        return self._opened_raw

    def _do_open_raw(self):
        if False:
            return 10
        if hasattr(self._fobj, 'fileno'):
            return self._fobj
        return FileIO(self._fobj, self.fileio_mode, self.closefd)

    @staticmethod
    def is_buffered(stream):
        if False:
            i = 10
            return i + 15
        return isinstance(stream, (io.BufferedIOBase, io.TextIOBase)) or (hasattr(stream, 'buffer') and stream.buffer is not None)

    @classmethod
    def buffer_size_for_stream(cls, stream):
        if False:
            while True:
                i = 10
        result = cls.default_buffer_size
        try:
            bs = os.fstat(stream.fileno()).st_blksize
        except (OSError, AttributeError):
            pass
        else:
            if bs > 1:
                result = bs
        return result

    def __buffered(self, stream, buffering):
        if False:
            for i in range(10):
                print('nop')
        if self.updating:
            Buffer = io.BufferedRandom
        elif self.creating or self.writing or self.appending:
            Buffer = io.BufferedWriter
        elif self.reading:
            Buffer = io.BufferedReader
        else:
            raise ValueError('unknown mode: %r' % self.mode)
        try:
            result = Buffer(stream, buffering)
        except AttributeError:
            result = stream
        return result

    def _make_atomic_write(self, result, raw):
        if False:
            return 10
        if result is not raw or self._raw_object_is_new(raw):
            if result.__class__ is FileIO:
                result.__class__ = WriteallFileIO
            else:
                raise NotImplementedError("Don't know how to make %s have atomic write. Please open a gevent issue with your use-case." % result)
        return result

    def __wrapped(self, raw):
        if False:
            for i in range(10):
                print('nop')
        '\n        Wraps the raw IO object (`RawIOBase` or `io.TextIOBase`) in\n        buffers, text decoding, and newline handling.\n        '
        if self.binary and isinstance(raw, io.TextIOBase):
            raise ValueError('Unable to perform binary IO on top of text IO stream')
        result = raw
        buffering = self.buffering
        line_buffering = False
        if buffering == 1 or (buffering < 0 and raw.isatty()):
            buffering = -1
            line_buffering = True
        if buffering < 0:
            buffering = self.buffer_size_for_stream(result)
        if buffering < 0:
            raise ValueError('invalid buffering size')
        if buffering != 0 and (not self.is_buffered(result)):
            result = self.__buffered(result, buffering)
        if not self.binary:
            if not isinstance(raw, io.TextIOBase):
                result = io.TextIOWrapper(result, self.encoding, self.errors, self.newline, line_buffering)
        if result is not raw or self._raw_object_is_new(raw):
            try:
                result.mode = self.mode
            except (AttributeError, TypeError):
                pass
        if self.atomic_write and (not self.is_buffered(result)) and (not isinstance(result, WriteIsWriteallMixin)):
            result = self._make_atomic_write(result, raw)
        return result

class _ClosedIO(object):
    __slots__ = ('name',)

    def __init__(self, io_obj):
        if False:
            while True:
                i = 10
        try:
            self.name = io_obj.name
        except AttributeError:
            pass

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        if name == 'name':
            raise AttributeError
        raise FileObjectClosed

    def __bool__(self):
        if False:
            print('Hello World!')
        return False
    __nonzero__ = __bool__

class FileObjectBase(object):
    """
    Internal base class to ensure a level of consistency
    between :class:`~.FileObjectPosix`, :class:`~.FileObjectThread`
    and :class:`~.FileObjectBlock`.
    """
    _delegate_methods = ('flush', 'fileno', 'writable', 'readable', 'seek', 'seekable', 'tell', 'read', 'readline', 'readlines', 'read1', 'readinto', 'write', 'writeall', 'writelines', 'truncate')
    _io = None

    def __init__(self, descriptor):
        if False:
            return 10
        self._io = descriptor.opened()
        self._close = descriptor.closefd
        self._do_delegate_methods()
    io = property(lambda s: s._io, lambda s, nv: setattr(s, '_io', nv) or s._do_delegate_methods())

    def _do_delegate_methods(self):
        if False:
            print('Hello World!')
        for meth_name in self._delegate_methods:
            meth = getattr(self._io, meth_name, None)
            implemented_by_class = hasattr(type(self), meth_name)
            if meth and (not implemented_by_class):
                setattr(self, meth_name, self._wrap_method(meth))
            elif hasattr(self, meth_name) and (not implemented_by_class):
                delattr(self, meth_name)

    def _wrap_method(self, method):
        if False:
            while True:
                i = 10
        "\n        Wrap a method we're copying into our dictionary from the underlying\n        io object to do something special or different, if necessary.\n        "
        return method

    @property
    def closed(self):
        if False:
            i = 10
            return i + 15
        'True if the file is closed'
        return isinstance(self._io, _ClosedIO)

    def close(self):
        if False:
            i = 10
            return i + 15
        if isinstance(self._io, _ClosedIO):
            return
        fobj = self._io
        self._io = _ClosedIO(self._io)
        try:
            self._do_close(fobj, self._close)
        finally:
            fobj = None
            d = self.__dict__
            for meth_name in self._delegate_methods:
                d.pop(meth_name, None)

    def _do_close(self, fobj, closefd):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        return getattr(self._io, name)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<%s at 0x%x %s_fobj=%r%s>' % (self.__class__.__name__, id(self), 'closed' if self.closed else '', self.io, self._extra_repr())

    def _extra_repr(self):
        if False:
            i = 10
            return i + 15
        return ''

    def __enter__(self):
        if False:
            print('Hello World!')
        return self

    def __exit__(self, *args):
        if False:
            i = 10
            return i + 15
        self.close()

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        line = self.readline()
        if not line:
            raise StopIteration
        return line
    next = __next__

    def __bool__(self):
        if False:
            return 10
        return True
    __nonzero__ = __bool__

class FileObjectBlock(FileObjectBase):
    """
    FileObjectBlock()

    A simple synchronous wrapper around a file object.

    Adds no concurrency or gevent compatibility.
    """

    def __init__(self, fobj, *args, **kwargs):
        if False:
            return 10
        descriptor = OpenDescriptor(fobj, *args, **kwargs)
        FileObjectBase.__init__(self, descriptor)

    def _do_close(self, fobj, closefd):
        if False:
            while True:
                i = 10
        fobj.close()

class FileObjectThread(FileObjectBase):
    """
    FileObjectThread()

    A file-like object wrapping another file-like object, performing all blocking
    operations on that object in a background thread.

    .. caution::
        Attempting to change the threadpool or lock of an existing FileObjectThread
        has undefined consequences.

    .. versionchanged:: 1.1b1
       The file object is closed using the threadpool. Note that whether or
       not this action is synchronous or asynchronous is not documented.
    """

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        '\n        :keyword bool lock: If True (the default) then all operations will\n           be performed one-by-one. Note that this does not guarantee that, if using\n           this file object from multiple threads/greenlets, operations will be performed\n           in any particular order, only that no two operations will be attempted at the\n           same time. You can also pass your own :class:`gevent.lock.Semaphore` to synchronize\n           file operations with an external resource.\n        :keyword bool closefd: If True (the default) then when this object is closed,\n           the underlying object is closed as well. If *fobj* is a path, then\n           *closefd* must be True.\n        '
        lock = kwargs.pop('lock', True)
        threadpool = kwargs.pop('threadpool', None)
        descriptor = OpenDescriptor(*args, **kwargs)
        self.threadpool = threadpool or get_hub().threadpool
        self.lock = lock
        if self.lock is True:
            self.lock = Semaphore()
        elif not self.lock:
            self.lock = DummySemaphore()
        if not hasattr(self.lock, '__enter__'):
            raise TypeError('Expected a Semaphore or boolean, got %r' % type(self.lock))
        self.__io_holder = [descriptor.opened()]
        FileObjectBase.__init__(self, descriptor)

    def _do_close(self, fobj, closefd):
        if False:
            for i in range(10):
                print('nop')
        self.__io_holder[0] = None
        try:
            with self.lock:
                self.threadpool.apply(fobj.flush)
        finally:
            if closefd:

                def close(_fobj=fobj):
                    if False:
                        for i in range(10):
                            print('nop')
                    try:
                        _fobj.close()
                    except:
                        return sys.exc_info()
                    finally:
                        _fobj = None
                del fobj
                exc_info = self.threadpool.apply(close)
                del close
                if exc_info:
                    reraise(*exc_info)

    def _do_delegate_methods(self):
        if False:
            print('Hello World!')
        FileObjectBase._do_delegate_methods(self)
        self.__io_holder[0] = self._io

    def _extra_repr(self):
        if False:
            return 10
        return ' threadpool=%r' % (self.threadpool,)

    def _wrap_method(self, method):
        if False:
            print('Hello World!')
        io_holder = self.__io_holder
        lock = self.lock
        threadpool = self.threadpool

        @functools.wraps(method)
        def thread_method(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            if io_holder[0] is None:
                raise FileObjectClosed
            with lock:
                return threadpool.apply(method, args, kwargs)
        return thread_method