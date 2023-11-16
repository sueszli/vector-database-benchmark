import collections
import errno
import fcntl
import gevent
import gevent.socket
import os
PIPE_BUF_BYTES = None
OS_PIPE_SZ = None

def _configure_buffer_sizes():
    if False:
        for i in range(10):
            print('nop')
    'Set up module globals controlling buffer sizes'
    global PIPE_BUF_BYTES
    global OS_PIPE_SZ
    PIPE_BUF_BYTES = 65536
    OS_PIPE_SZ = None
    if not hasattr(fcntl, 'F_SETPIPE_SZ'):
        import platform
        if platform.system() == 'Linux':
            fcntl.F_SETPIPE_SZ = 1031
    try:
        with open('/proc/sys/fs/pipe-max-size', 'r') as f:
            OS_PIPE_SZ = min(int(f.read()), 1024 * 1024)
            PIPE_BUF_BYTES = max(OS_PIPE_SZ, PIPE_BUF_BYTES)
    except Exception:
        pass
_configure_buffer_sizes()

def set_buf_size(fd):
    if False:
        i = 10
        return i + 15
    'Set up os pipe buffer size, if applicable'
    if OS_PIPE_SZ and hasattr(fcntl, 'F_SETPIPE_SZ'):
        fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, OS_PIPE_SZ)

def _setup_fd(fd):
    if False:
        while True:
            i = 10
    'Common set-up code for initializing a (pipe) file descriptor'
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    set_buf_size(fd)

class ByteDeque(object):
    """Data structure for delayed defragmentation of submitted bytes"""

    def __init__(self):
        if False:
            return 10
        self._dq = collections.deque()
        self.byteSz = 0

    def add(self, b):
        if False:
            return 10
        self._dq.append(b)
        self.byteSz += len(b)

    def get(self, n):
        if False:
            i = 10
            return i + 15
        assert n <= self.byteSz, 'caller responsibility to ensure enough bytes'
        if n == self.byteSz and len(self._dq) == 1 and isinstance(self._dq[0], bytes):
            self.byteSz = 0
            return self._dq.popleft()
        out = bytearray(n)
        remaining = n
        while remaining > 0:
            part = memoryview(self._dq.popleft())
            delta = remaining - len(part)
            offset = n - remaining
            if delta == 0:
                out[offset:] = part
                remaining = 0
            elif delta > 0:
                out[offset:] = part
                remaining = delta
            elif delta < 0:
                cleave = len(part) + delta
                out[offset:] = part[:cleave]
                self._dq.appendleft(part[cleave:])
                remaining = 0
            else:
                assert False
        self.byteSz -= n
        assert len(out) == n
        return bytes(out)

    def get_all(self):
        if False:
            return 10
        return self.get(self.byteSz)

class NonBlockBufferedReader(object):
    """A buffered pipe reader that adheres to the Python file protocol"""

    def __init__(self, fp):
        if False:
            return 10
        self._fp = fp
        self._fd = fp.fileno()
        self._bd = ByteDeque()
        self.got_eof = False
        _setup_fd(self._fd)

    def _read_chunk(self, sz):
        if False:
            i = 10
            return i + 15
        chunk = None
        try:
            chunk = os.read(self._fd, sz)
            self._bd.add(chunk)
        except EnvironmentError as e:
            if e.errno in [errno.EAGAIN, errno.EWOULDBLOCK]:
                assert chunk is None
                gevent.socket.wait_read(self._fd)
            else:
                raise
        self.got_eof = chunk == b''

    def read(self, size=None):
        if False:
            i = 10
            return i + 15
        if size is None:
            while not self.got_eof:
                self._read_chunk(PIPE_BUF_BYTES)
            return self._bd.get_all()
        elif size > 0:
            while True:
                if self._bd.byteSz >= size:
                    return self._bd.get(size)
                elif self._bd.byteSz <= size and self.got_eof:
                    return self._bd.get_all()
                else:
                    assert not self.got_eof
                    if size == PIPE_BUF_BYTES:
                        to_read = PIPE_BUF_BYTES - self._bd.byteSz
                        self._read_chunk(to_read)
                    else:
                        self._read_chunk(PIPE_BUF_BYTES)
        else:
            assert False

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        if self.closed:
            return
        self._fp.close()
        try:
            del self._fp
        except AttributeError:
            pass
        try:
            del self._bd
        except AttributeError:
            pass
        self._fd = -1

    def fileno(self):
        if False:
            for i in range(10):
                print('nop')
        return self._fd

    @property
    def closed(self):
        if False:
            i = 10
            return i + 15
        return self._fd == -1

class NonBlockBufferedWriter(object):
    """A buffered pipe writer that adheres to the Python file protocol"""

    def __init__(self, fp):
        if False:
            i = 10
            return i + 15
        self._fp = fp
        self._fd = fp.fileno()
        self._bd = ByteDeque()
        _setup_fd(self._fd)

    def _partial_flush(self, max_retain):
        if False:
            i = 10
            return i + 15
        byts = self._bd.get_all()
        cursor = memoryview(byts)
        flushed = False
        while len(cursor) > max_retain:
            try:
                n = os.write(self._fd, cursor)
                flushed = True
                cursor = memoryview(cursor)[n:]
            except EnvironmentError as e:
                if e.errno in [errno.EAGAIN, errno.EWOULDBLOCK]:
                    gevent.socket.wait_write(self._fd)
                else:
                    raise
        assert self._bd.byteSz == 0
        if len(cursor) > 0:
            self._bd.add(cursor)
        return flushed

    def write(self, data):
        if False:
            while True:
                i = 10
        self._bd.add(data)
        flushed = True
        while flushed and self._bd.byteSz > PIPE_BUF_BYTES:
            flushed = self._partial_flush(65535)

    def flush(self):
        if False:
            return 10
        while self._bd.byteSz > 0:
            self._partial_flush(0)

    def fileno(self):
        if False:
            for i in range(10):
                print('nop')
        return self._fd

    def close(self):
        if False:
            while True:
                i = 10
        if self.closed:
            return
        self._fp.close()
        try:
            del self._fp
        except AttributeError:
            pass
        try:
            del self._bd
        except AttributeError:
            pass
        self._fd = -1

    @property
    def closed(self):
        if False:
            for i in range(10):
                print('nop')
        return self._fd == -1