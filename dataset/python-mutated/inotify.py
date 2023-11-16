__copyright__ = '2013, Kovid Goyal <kovid at kovidgoyal.net>'
__docformat__ = 'restructuredtext en'
import sys, os, errno, select

class INotifyError(Exception):
    pass

class NoSuchDir(ValueError):
    pass

class BaseDirChanged(ValueError):
    pass

class DirTooLarge(ValueError):

    def __init__(self, bdir):
        if False:
            print('Hello World!')
        ValueError.__init__(self, f'The directory {bdir} is too large to monitor. Try increasing the value in /proc/sys/fs/inotify/max_user_watches')
_inotify = None

def load_inotify():
    if False:
        i = 10
        return i + 15
    ' Initialize the inotify ctypes wrapper '
    global _inotify
    if _inotify is None:
        if hasattr(sys, 'getwindowsversion'):
            raise INotifyError('INotify not available on windows')
        if sys.platform == 'darwin':
            raise INotifyError('INotify not available on OS X')
        import ctypes
        if not hasattr(ctypes, 'c_ssize_t'):
            raise INotifyError('You need python >= 2.7 to use inotify')
        libc = ctypes.CDLL(None, use_errno=True)
        for function in ('inotify_add_watch', 'inotify_init1', 'inotify_rm_watch'):
            if not hasattr(libc, function):
                raise INotifyError('libc is too old')
        prototype = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, use_errno=True)
        init1 = prototype(('inotify_init1', libc), ((1, 'flags', 0),))
        prototype = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32, use_errno=True)
        add_watch = prototype(('inotify_add_watch', libc), ((1, 'fd'), (1, 'pathname'), (1, 'mask')), use_errno=True)
        prototype = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int, use_errno=True)
        rm_watch = prototype(('inotify_rm_watch', libc), ((1, 'fd'), (1, 'wd')), use_errno=True)
        prototype = ctypes.CFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, use_errno=True)
        read = prototype(('read', libc), ((1, 'fd'), (1, 'buf'), (1, 'count')), use_errno=True)
        _inotify = (init1, add_watch, rm_watch, read)
    return _inotify

class INotify:
    ACCESS = 1
    MODIFY = 2
    ATTRIB = 4
    CLOSE_WRITE = 8
    CLOSE_NOWRITE = 16
    OPEN = 32
    MOVED_FROM = 64
    MOVED_TO = 128
    CREATE = 256
    DELETE = 512
    DELETE_SELF = 1024
    MOVE_SELF = 2048
    UNMOUNT = 8192
    Q_OVERFLOW = 16384
    IGNORED = 32768
    CLOSE = CLOSE_WRITE | CLOSE_NOWRITE
    MOVE = MOVED_FROM | MOVED_TO
    ONLYDIR = 16777216
    DONT_FOLLOW = 33554432
    EXCL_UNLINK = 67108864
    MASK_ADD = 536870912
    ISDIR = 1073741824
    ONESHOT = 2147483648
    ALL_EVENTS = ACCESS | MODIFY | ATTRIB | CLOSE_WRITE | CLOSE_NOWRITE | OPEN | MOVED_FROM | MOVED_TO | CREATE | DELETE | DELETE_SELF | MOVE_SELF
    CLOEXEC = 524288
    NONBLOCK = 2048

    def __init__(self, cloexec=True, nonblock=True):
        if False:
            i = 10
            return i + 15
        import ctypes, struct
        (self._init1, self._add_watch, self._rm_watch, self._read) = load_inotify()
        flags = 0
        if cloexec:
            flags |= self.CLOEXEC
        if nonblock:
            flags |= self.NONBLOCK
        self._inotify_fd = self._init1(flags)
        if self._inotify_fd == -1:
            raise INotifyError(os.strerror(ctypes.get_errno()))
        self._buf = ctypes.create_string_buffer(5120)
        self.fenc = sys.getfilesystemencoding() or 'utf-8'
        self.hdr = struct.Struct(b'iIII')
        if self.fenc == 'ascii':
            self.fenc = 'utf-8'
        self.os = os

    def handle_error(self):
        if False:
            while True:
                i = 10
        import ctypes
        eno = ctypes.get_errno()
        extra = ''
        if eno == errno.ENOSPC:
            extra = 'You may need to increase the inotify limits on your system, via /proc/sys/inotify/max_user_*'
        raise OSError(eno, self.os.strerror(eno) + extra)

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.os.close(self._inotify_fd)
        except (AttributeError, TypeError):
            pass

    def close(self):
        if False:
            print('Hello World!')
        if hasattr(self, '_inotify_fd'):
            self.os.close(self._inotify_fd)
            del self.os
            del self._add_watch
            del self._rm_watch
            del self._inotify_fd

    def __enter__(self):
        if False:
            return 10
        return self

    def __exit__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        self.close()

    def read(self, get_name=True):
        if False:
            print('Hello World!')
        import ctypes
        buf = []
        while True:
            num = self._read(self._inotify_fd, self._buf, len(self._buf))
            if num == 0:
                break
            if num < 0:
                en = ctypes.get_errno()
                if en == errno.EAGAIN:
                    break
                if en == errno.EINTR:
                    continue
                raise OSError(en, self.os.strerror(en))
            buf.append(self._buf.raw[:num])
        raw = b''.join(buf)
        pos = 0
        lraw = len(raw)
        while lraw - pos >= self.hdr.size:
            (wd, mask, cookie, name_len) = self.hdr.unpack_from(raw, pos)
            pos += self.hdr.size
            name = None
            if get_name:
                name = raw[pos:pos + name_len].rstrip(b'\x00').decode(self.fenc)
            pos += name_len
            self.process_event(wd, mask, cookie, name)

    def process_event(self, *args):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def wait(self, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        'Return True iff there are events waiting to be read. Blocks if timeout is None. Polls if timeout is 0.'
        return len((select.select([self._inotify_fd], [], []) if timeout is None else select.select([self._inotify_fd], [], [], timeout))[0]) > 0

def realpath(path):
    if False:
        return 10
    return os.path.abspath(os.path.realpath(path))

class INotifyTreeWatcher(INotify):
    is_dummy = False

    def __init__(self, basedir, ignore_event=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.basedir = realpath(basedir)
        self.watch_tree()
        self.modified = set()
        self.ignore_event = (lambda path, name: False) if ignore_event is None else ignore_event

    def watch_tree(self):
        if False:
            return 10
        self.watched_dirs = {}
        self.watched_rmap = {}
        try:
            self.add_watches(self.basedir)
        except OSError as e:
            if e.errno == errno.ENOSPC:
                raise DirTooLarge(self.basedir)

    def add_watches(self, base, top_level=True):
        if False:
            while True:
                i = 10
        ' Add watches for this directory and all its descendant directories,\n        recursively. '
        base = realpath(base)
        if not top_level and base in self.watched_dirs:
            return
        try:
            is_dir = self.add_watch(base)
        except OSError as e:
            if e.errno == errno.ENOENT:
                if top_level:
                    raise NoSuchDir(f'The dir {base} does not exist')
                return
            if e.errno == errno.EACCES:
                if top_level:
                    raise NoSuchDir(f'You do not have permission to monitor {base}')
                return
            raise
        else:
            if is_dir:
                try:
                    files = os.listdir(base)
                except OSError as e:
                    if e.errno in (errno.ENOTDIR, errno.ENOENT):
                        if top_level:
                            raise NoSuchDir(f'The dir {base} does not exist')
                        return
                    raise
                for x in files:
                    self.add_watches(os.path.join(base, x), top_level=False)
            elif top_level:
                raise NoSuchDir(f'The dir {base} does not exist')

    def add_watch(self, path):
        if False:
            return 10
        import ctypes
        bpath = path if isinstance(path, bytes) else path.encode(self.fenc)
        wd = self._add_watch(self._inotify_fd, ctypes.c_char_p(bpath), self.DONT_FOLLOW | self.ONLYDIR | self.MODIFY | self.CREATE | self.DELETE | self.MOVE_SELF | self.MOVED_FROM | self.MOVED_TO | self.ATTRIB | self.DELETE_SELF)
        if wd == -1:
            eno = ctypes.get_errno()
            if eno == errno.ENOTDIR:
                return False
            raise OSError(eno, f'Failed to add watch for: {path}: {self.os.strerror(eno)}')
        self.watched_dirs[path] = wd
        self.watched_rmap[wd] = path
        return True

    def process_event(self, wd, mask, cookie, name):
        if False:
            i = 10
            return i + 15
        if wd == -1 and mask & self.Q_OVERFLOW:
            self.watch_tree()
            self.modified.add(None)
            return
        path = self.watched_rmap.get(wd, None)
        if path is not None:
            if not self.ignore_event(path, name):
                self.modified.add(os.path.join(path, name or ''))
            if mask & self.CREATE:
                try:
                    self.add_watch(os.path.join(path, name))
                except OSError as e:
                    if e.errno == errno.ENOENT:
                        pass
                    elif e.errno == errno.ENOSPC:
                        raise DirTooLarge(self.basedir)
                    else:
                        raise
            if (mask & self.DELETE_SELF or mask & self.MOVE_SELF) and path == self.basedir:
                raise BaseDirChanged('The directory %s was moved/deleted' % path)

    def __call__(self):
        if False:
            return 10
        self.read()
        ret = self.modified
        self.modified = set()
        return ret
if __name__ == '__main__':
    w = INotifyTreeWatcher(sys.argv[-1])
    w()
    print('Monitoring', sys.argv[-1], 'press Ctrl-C to stop')
    try:
        while w.wait():
            modified = w()
            for path in modified:
                print(path or sys.argv[-1], 'changed')
        raise SystemExit('inotify flaked out')
    except KeyboardInterrupt:
        pass