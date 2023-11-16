__license__ = 'GPL v3'
__copyright__ = '2008, Kovid Goyal <kovid at kovidgoyal.net>'
'\nProvides platform independent temporary files that persist even after\nbeing closed.\n'
import tempfile, os, atexit
from calibre.constants import __version__, __appname__, filesystem_encoding, iswindows, get_windows_temp_path, ismacos

def cleanup(path):
    if False:
        while True:
            i = 10
    try:
        import os as oss
        if oss.path.exists(path):
            oss.remove(path)
    except:
        pass
_base_dir = None

def remove_dir(x):
    if False:
        print('Hello World!')
    try:
        import shutil
        shutil.rmtree(x, ignore_errors=True)
    except:
        pass

def determined_remove_dir(x):
    if False:
        return 10
    for i in range(10):
        try:
            import shutil
            shutil.rmtree(x)
            return
        except:
            import os
            if os.path.exists(x):
                import time
                time.sleep(0.1)
            else:
                return
    try:
        import shutil
        shutil.rmtree(x, ignore_errors=True)
    except:
        pass

def app_prefix(prefix):
    if False:
        for i in range(10):
            print('nop')
    if iswindows:
        return '%s_' % __appname__
    return '%s_%s_%s' % (__appname__, __version__, prefix)
_osx_cache_dir = None

def osx_cache_dir():
    if False:
        print('Hello World!')
    global _osx_cache_dir
    if _osx_cache_dir:
        return _osx_cache_dir
    if _osx_cache_dir is None:
        _osx_cache_dir = False
        import ctypes
        libc = ctypes.CDLL(None)
        buf = ctypes.create_string_buffer(512)
        l = libc.confstr(65538, ctypes.byref(buf), len(buf))
        if 0 < l < len(buf):
            try:
                q = buf.value.decode('utf-8').rstrip('\x00')
            except ValueError:
                pass
            if q and os.path.isdir(q) and os.access(q, os.R_OK | os.W_OK | os.X_OK):
                _osx_cache_dir = q
                return q

def base_dir():
    if False:
        print('Hello World!')
    global _base_dir
    if _base_dir is not None and (not os.path.exists(_base_dir)):
        _base_dir = None
    if _base_dir is None:
        td = os.environ.get('CALIBRE_WORKER_TEMP_DIR', None)
        if td is not None:
            from calibre.utils.serialize import msgpack_loads
            from polyglot.binary import from_hex_bytes
            try:
                td = msgpack_loads(from_hex_bytes(td))
            except Exception:
                td = None
        if td and os.path.exists(td):
            _base_dir = td
        else:
            base = os.environ.get('CALIBRE_TEMP_DIR', None)
            if base is not None and iswindows:
                base = os.getenv('CALIBRE_TEMP_DIR')
            prefix = app_prefix('tmp_')
            if base is None:
                if iswindows:
                    base = get_windows_temp_path()
                elif ismacos:
                    base = osx_cache_dir()
            _base_dir = tempfile.mkdtemp(prefix=prefix, dir=base)
            atexit.register(determined_remove_dir if iswindows else remove_dir, _base_dir)
        try:
            tempfile.gettempdir()
        except Exception:
            tempfile.tempdir = _base_dir
    return _base_dir

def reset_base_dir():
    if False:
        for i in range(10):
            print('nop')
    global _base_dir
    _base_dir = None
    base_dir()

def force_unicode(x):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(x, bytes):
        x = x.decode(filesystem_encoding)
    return x

def _make_file(suffix, prefix, base):
    if False:
        print('Hello World!')
    (suffix, prefix) = map(force_unicode, (suffix, prefix))
    return tempfile.mkstemp(suffix, prefix, dir=base)

def _make_dir(suffix, prefix, base):
    if False:
        print('Hello World!')
    (suffix, prefix) = map(force_unicode, (suffix, prefix))
    return tempfile.mkdtemp(suffix, prefix, base)

class PersistentTemporaryFile:
    """
    A file-like object that is a temporary file that is available even after being closed on
    all platforms. It is automatically deleted on normal program termination.
    """
    _file = None

    def __init__(self, suffix='', prefix='', dir=None, mode='w+b'):
        if False:
            for i in range(10):
                print('nop')
        if prefix is None:
            prefix = ''
        if dir is None:
            dir = base_dir()
        (fd, name) = _make_file(suffix, prefix, dir)
        self._file = os.fdopen(fd, mode)
        self._name = name
        self._fd = fd
        atexit.register(cleanup, name)

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        if name == 'name':
            return self.__dict__['_name']
        return getattr(self.__dict__['_file'], name)

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __exit__(self, *args):
        if False:
            i = 10
            return i + 15
        self.close()

    def __del__(self):
        if False:
            while True:
                i = 10
        try:
            self.close()
        except:
            pass

def PersistentTemporaryDirectory(suffix='', prefix='', dir=None):
    if False:
        while True:
            i = 10
    '\n    Return the path to a newly created temporary directory that will\n    be automatically deleted on application exit.\n    '
    if dir is None:
        dir = base_dir()
    tdir = _make_dir(suffix, prefix, dir)
    atexit.register(remove_dir, tdir)
    return tdir

class TemporaryDirectory:
    """
    A temporary directory to be used in a with statement.
    """

    def __init__(self, suffix='', prefix='', dir=None, keep=False):
        if False:
            print('Hello World!')
        self.suffix = suffix
        self.prefix = prefix
        if dir is None:
            dir = base_dir()
        self.dir = dir
        self.keep = keep

    def __enter__(self):
        if False:
            return 10
        if not hasattr(self, 'tdir'):
            self.tdir = _make_dir(self.suffix, self.prefix, self.dir)
        return self.tdir

    def __exit__(self, *args):
        if False:
            i = 10
            return i + 15
        if not self.keep and os.path.exists(self.tdir):
            remove_dir(self.tdir)

class TemporaryFile:

    def __init__(self, suffix='', prefix='', dir=None, mode='w+b'):
        if False:
            while True:
                i = 10
        if prefix is None:
            prefix = ''
        if suffix is None:
            suffix = ''
        if dir is None:
            dir = base_dir()
        (self.prefix, self.suffix, self.dir, self.mode) = (prefix, suffix, dir, mode)
        self._file = None

    def __enter__(self):
        if False:
            while True:
                i = 10
        (fd, name) = _make_file(self.suffix, self.prefix, self.dir)
        self._file = os.fdopen(fd, self.mode)
        self._name = name
        self._file.close()
        return name

    def __exit__(self, *args):
        if False:
            print('Hello World!')
        cleanup(self._name)

class SpooledTemporaryFile(tempfile.SpooledTemporaryFile):

    def __init__(self, max_size=0, suffix='', prefix='', dir=None, mode='w+b', bufsize=-1):
        if False:
            for i in range(10):
                print('nop')
        if prefix is None:
            prefix = ''
        if suffix is None:
            suffix = ''
        if dir is None:
            dir = base_dir()
        self._name = None
        tempfile.SpooledTemporaryFile.__init__(self, max_size=max_size, suffix=suffix, prefix=prefix, dir=dir, mode=mode)

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
        return self._name

    @name.setter
    def name(self, val):
        if False:
            print('Hello World!')
        self._name = val

    def readable(self):
        if False:
            while True:
                i = 10
        return self._file.readable()

    def seekable(self):
        if False:
            for i in range(10):
                print('nop')
        return self._file.seekable()

    def writable(self):
        if False:
            return 10
        return self._file.writable()

def better_mktemp(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    (fd, path) = tempfile.mkstemp(*args, **kwargs)
    os.close(fd)
    return path