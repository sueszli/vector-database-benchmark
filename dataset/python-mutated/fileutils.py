"""Virtually every Python programmer has used Python for wrangling
disk contents, and ``fileutils`` collects solutions to some of the
most commonly-found gaps in the standard library.
"""
from __future__ import print_function
import os
import re
import sys
import stat
import errno
import fnmatch
from shutil import copy2, copystat, Error
__all__ = ['mkdir_p', 'atomic_save', 'AtomicSaver', 'FilePerms', 'iter_find_files', 'copytree']
FULL_PERMS = 511
RW_PERMS = 438
_SINGLE_FULL_PERM = 7
try:
    basestring
except NameError:
    unicode = str
    basestring = (str, bytes)

def mkdir_p(path):
    if False:
        print('Hello World!')
    'Creates a directory and any parent directories that may need to\n    be created along the way, without raising errors for any existing\n    directories. This function mimics the behavior of the ``mkdir -p``\n    command available in Linux/BSD environments, but also works on\n    Windows.\n    '
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            return
        raise
    return

class FilePerms(object):
    """The :class:`FilePerms` type is used to represent standard POSIX
    filesystem permissions:

      * Read
      * Write
      * Execute

    Across three classes of user:

      * Owning (u)ser
      * Owner's (g)roup
      * Any (o)ther user

    This class assists with computing new permissions, as well as
    working with numeric octal ``777``-style and ``rwx``-style
    permissions. Currently it only considers the bottom 9 permission
    bits; it does not support sticky bits or more advanced permission
    systems.

    Args:
        user (str): A string in the 'rwx' format, omitting characters
            for which owning user's permissions are not provided.
        group (str): A string in the 'rwx' format, omitting characters
            for which owning group permissions are not provided.
        other (str): A string in the 'rwx' format, omitting characters
            for which owning other/world permissions are not provided.

    There are many ways to use :class:`FilePerms`:

    >>> FilePerms(user='rwx', group='xrw', other='wxr')  # note character order
    FilePerms(user='rwx', group='rwx', other='rwx')
    >>> int(FilePerms('r', 'r', ''))
    288
    >>> oct(288)[-3:]  # XXX Py3k
    '440'

    See also the :meth:`FilePerms.from_int` and
    :meth:`FilePerms.from_path` classmethods for useful alternative
    ways to construct :class:`FilePerms` objects.
    """

    class _FilePermProperty(object):
        _perm_chars = 'rwx'
        _perm_set = frozenset('rwx')
        _perm_val = {'r': 4, 'w': 2, 'x': 1}

        def __init__(self, attribute, offset):
            if False:
                while True:
                    i = 10
            self.attribute = attribute
            self.offset = offset

        def __get__(self, fp_obj, type_=None):
            if False:
                while True:
                    i = 10
            if fp_obj is None:
                return self
            return getattr(fp_obj, self.attribute)

        def __set__(self, fp_obj, value):
            if False:
                while True:
                    i = 10
            cur = getattr(fp_obj, self.attribute)
            if cur == value:
                return
            try:
                invalid_chars = set(str(value)) - self._perm_set
            except TypeError:
                raise TypeError('expected string, not %r' % value)
            if invalid_chars:
                raise ValueError('got invalid chars %r in permission specification %r, expected empty string or one or more of %r' % (invalid_chars, value, self._perm_chars))
            sort_key = lambda c: self._perm_val[c]
            new_value = ''.join(sorted(set(value), key=sort_key, reverse=True))
            setattr(fp_obj, self.attribute, new_value)
            self._update_integer(fp_obj, new_value)

        def _update_integer(self, fp_obj, value):
            if False:
                while True:
                    i = 10
            mode = 0
            key = 'xwr'
            for symbol in value:
                bit = 2 ** key.index(symbol)
                mode |= bit << self.offset * 3
            fp_obj._integer |= mode

    def __init__(self, user='', group='', other=''):
        if False:
            return 10
        (self._user, self._group, self._other) = ('', '', '')
        self._integer = 0
        self.user = user
        self.group = group
        self.other = other

    @classmethod
    def from_int(cls, i):
        if False:
            while True:
                i = 10
        "Create a :class:`FilePerms` object from an integer.\n\n        >>> FilePerms.from_int(0o644)  # note the leading zero-oh for octal\n        FilePerms(user='rw', group='r', other='r')\n        "
        i &= FULL_PERMS
        key = ('', 'x', 'w', 'xw', 'r', 'rx', 'rw', 'rwx')
        parts = []
        while i:
            parts.append(key[i & _SINGLE_FULL_PERM])
            i >>= 3
        parts.reverse()
        return cls(*parts)

    @classmethod
    def from_path(cls, path):
        if False:
            return 10
        "Make a new :class:`FilePerms` object based on the permissions\n        assigned to the file or directory at *path*.\n\n        Args:\n            path (str): Filesystem path of the target file.\n\n        Here's an example that holds true on most systems:\n\n        >>> import tempfile\n        >>> 'r' in FilePerms.from_path(tempfile.gettempdir()).user\n        True\n        "
        stat_res = os.stat(path)
        return cls.from_int(stat.S_IMODE(stat_res.st_mode))

    def __int__(self):
        if False:
            i = 10
            return i + 15
        return self._integer
    user = _FilePermProperty('_user', 2)
    'Stores the ``rwx``-formatted *user* permission.'
    group = _FilePermProperty('_group', 1)
    'Stores the ``rwx``-formatted *group* permission.'
    other = _FilePermProperty('_other', 0)
    'Stores the ``rwx``-formatted *other* permission.'

    def __repr__(self):
        if False:
            return 10
        cn = self.__class__.__name__
        return '%s(user=%r, group=%r, other=%r)' % (cn, self.user, self.group, self.other)
_TEXT_OPENFLAGS = os.O_RDWR | os.O_CREAT | os.O_EXCL
if hasattr(os, 'O_NOINHERIT'):
    _TEXT_OPENFLAGS |= os.O_NOINHERIT
if hasattr(os, 'O_NOFOLLOW'):
    _TEXT_OPENFLAGS |= os.O_NOFOLLOW
_BIN_OPENFLAGS = _TEXT_OPENFLAGS
if hasattr(os, 'O_BINARY'):
    _BIN_OPENFLAGS |= os.O_BINARY
try:
    import fcntl as fcntl
except ImportError:

    def set_cloexec(fd):
        if False:
            print('Hello World!')
        'Dummy set_cloexec for platforms without fcntl support'
        pass
else:

    def set_cloexec(fd):
        if False:
            while True:
                i = 10
        'Does a best-effort :func:`fcntl.fcntl` call to set a fd to be\n        automatically closed by any future child processes.\n\n        Implementation from the :mod:`tempfile` module.\n        '
        try:
            flags = fcntl.fcntl(fd, fcntl.F_GETFD, 0)
        except IOError:
            pass
        else:
            flags |= fcntl.FD_CLOEXEC
            fcntl.fcntl(fd, fcntl.F_SETFD, flags)
        return

def atomic_save(dest_path, **kwargs):
    if False:
        return 10
    'A convenient interface to the :class:`AtomicSaver` type. Example:\n\n    >>> try:\n    ...     with atomic_save("file.txt", text_mode=True) as fo:\n    ...         _ = fo.write(\'bye\')\n    ...         1/0  # will error\n    ...         fo.write(\'bye\')\n    ... except ZeroDivisionError:\n    ...     pass  # at least our file.txt didn\'t get overwritten\n\n    See the :class:`AtomicSaver` documentation for details.\n    '
    return AtomicSaver(dest_path, **kwargs)

def path_to_unicode(path):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(path, unicode):
        return path
    encoding = sys.getfilesystemencoding() or sys.getdefaultencoding()
    return path.decode(encoding)
if os.name == 'nt':
    import ctypes
    from ctypes import c_wchar_p
    from ctypes.wintypes import DWORD, LPVOID
    _ReplaceFile = ctypes.windll.kernel32.ReplaceFile
    _ReplaceFile.argtypes = [c_wchar_p, c_wchar_p, c_wchar_p, DWORD, LPVOID, LPVOID]

    def replace(src, dst):
        if False:
            return 10
        try:
            os.rename(src, dst)
            return
        except WindowsError as we:
            if we.errno == errno.EEXIST:
                pass
            else:
                raise
        src = path_to_unicode(src)
        dst = path_to_unicode(dst)
        res = _ReplaceFile(c_wchar_p(dst), c_wchar_p(src), None, 0, None, None)
        if not res:
            raise OSError('failed to replace %r with %r' % (dst, src))
        return

    def atomic_rename(src, dst, overwrite=False):
        if False:
            i = 10
            return i + 15
        'Rename *src* to *dst*, replacing *dst* if *overwrite is True'
        if overwrite:
            replace(src, dst)
        else:
            os.rename(src, dst)
        return
else:

    def replace(src, dst):
        if False:
            return 10
        return os.rename(src, dst)

    def atomic_rename(src, dst, overwrite=False):
        if False:
            for i in range(10):
                print('nop')
        'Rename *src* to *dst*, replacing *dst* if *overwrite is True'
        if overwrite:
            os.rename(src, dst)
        else:
            os.link(src, dst)
            os.unlink(src)
        return
_atomic_rename = atomic_rename
replace.__doc__ = 'Similar to :func:`os.replace` in Python 3.3+,\nthis function will atomically create or replace the file at path\n*dst* with the file at path *src*.\n\nOn Windows, this function uses the ReplaceFile API for maximum\npossible atomicity on a range of filesystems.\n'

class AtomicSaver(object):
    """``AtomicSaver`` is a configurable `context manager`_ that provides
    a writable :class:`file` which will be moved into place as long as
    no exceptions are raised within the context manager's block. These
    "part files" are created in the same directory as the destination
    path to ensure atomic move operations (i.e., no cross-filesystem
    moves occur).

    Args:
        dest_path (str): The path where the completed file will be
            written.
        overwrite (bool): Whether to overwrite the destination file if
            it exists at completion time. Defaults to ``True``.
        file_perms (int): Integer representation of file permissions
            for the newly-created file. Defaults are, when the
            destination path already exists, to copy the permissions
            from the previous file, or if the file did not exist, to
            respect the user's configured `umask`_, usually resulting
            in octal 0644 or 0664.
        text_mode (bool): Whether to open the destination file in text
            mode (i.e., ``'w'`` not ``'wb'``). Defaults to ``False`` (``wb``).
        part_file (str): Name of the temporary *part_file*. Defaults
            to *dest_path* + ``.part``. Note that this argument is
            just the filename, and not the full path of the part
            file. To guarantee atomic saves, part files are always
            created in the same directory as the destination path.
        overwrite_part (bool): Whether to overwrite the *part_file*,
            should it exist at setup time. Defaults to ``False``,
            which results in an :exc:`OSError` being raised on
            pre-existing part files. Be careful of setting this to
            ``True`` in situations when multiple threads or processes
            could be writing to the same part file.
        rm_part_on_exc (bool): Remove *part_file* on exception cases.
            Defaults to ``True``, but ``False`` can be useful for
            recovery in some cases. Note that resumption is not
            automatic and by default an :exc:`OSError` is raised if
            the *part_file* exists.

    Practically, the AtomicSaver serves a few purposes:

      * Avoiding overwriting an existing, valid file with a partially
        written one.
      * Providing a reasonable guarantee that a part file only has one
        writer at a time.
      * Optional recovery of partial data in failure cases.

    .. _context manager: https://docs.python.org/2/reference/compound_stmts.html#with
    .. _umask: https://en.wikipedia.org/wiki/Umask

    """
    _default_file_perms = RW_PERMS

    def __init__(self, dest_path, **kwargs):
        if False:
            i = 10
            return i + 15
        self.dest_path = dest_path
        self.overwrite = kwargs.pop('overwrite', True)
        self.file_perms = kwargs.pop('file_perms', None)
        self.overwrite_part = kwargs.pop('overwrite_part', False)
        self.part_filename = kwargs.pop('part_file', None)
        self.rm_part_on_exc = kwargs.pop('rm_part_on_exc', True)
        self.text_mode = kwargs.pop('text_mode', False)
        self.buffering = kwargs.pop('buffering', -1)
        if kwargs:
            raise TypeError('unexpected kwargs: %r' % (kwargs.keys(),))
        self.dest_path = os.path.abspath(self.dest_path)
        self.dest_dir = os.path.dirname(self.dest_path)
        if not self.part_filename:
            self.part_path = dest_path + '.part'
        else:
            self.part_path = os.path.join(self.dest_dir, self.part_filename)
        self.mode = 'w+' if self.text_mode else 'w+b'
        self.open_flags = _TEXT_OPENFLAGS if self.text_mode else _BIN_OPENFLAGS
        self.part_file = None

    def _open_part_file(self):
        if False:
            return 10
        do_chmod = True
        file_perms = self.file_perms
        if file_perms is None:
            try:
                stat_res = os.stat(self.dest_path)
                file_perms = stat.S_IMODE(stat_res.st_mode)
            except (OSError, IOError):
                file_perms = self._default_file_perms
                do_chmod = False
        fd = os.open(self.part_path, self.open_flags, file_perms)
        set_cloexec(fd)
        self.part_file = os.fdopen(fd, self.mode, self.buffering)
        if do_chmod:
            try:
                os.chmod(self.part_path, file_perms)
            except (OSError, IOError):
                self.part_file.close()
                raise
        return

    def setup(self):
        if False:
            while True:
                i = 10
        'Called on context manager entry (the :keyword:`with` statement),\n        the ``setup()`` method creates the temporary file in the same\n        directory as the destination file.\n\n        ``setup()`` tests for a writable directory with rename permissions\n        early, as the part file may not be written to immediately (not\n        using :func:`os.access` because of the potential issues of\n        effective vs. real privileges).\n\n        If the caller is not using the :class:`AtomicSaver` as a\n        context manager, this method should be called explicitly\n        before writing.\n        '
        if os.path.lexists(self.dest_path):
            if not self.overwrite:
                raise OSError(errno.EEXIST, 'Overwrite disabled and file already exists', self.dest_path)
        if self.overwrite_part and os.path.lexists(self.part_path):
            os.unlink(self.part_path)
        self._open_part_file()
        return

    def __enter__(self):
        if False:
            return 10
        self.setup()
        return self.part_file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            for i in range(10):
                print('nop')
        self.part_file.close()
        if exc_type:
            if self.rm_part_on_exc:
                try:
                    os.unlink(self.part_path)
                except Exception:
                    pass
            return
        try:
            atomic_rename(self.part_path, self.dest_path, overwrite=self.overwrite)
        except OSError:
            if self.rm_part_on_exc:
                try:
                    os.unlink(self.part_path)
                except Exception:
                    pass
            raise
        return

def iter_find_files(directory, patterns, ignored=None, include_dirs=False):
    if False:
        for i in range(10):
            print('nop')
    "Returns a generator that yields file paths under a *directory*,\n    matching *patterns* using `glob`_ syntax (e.g., ``*.txt``). Also\n    supports *ignored* patterns.\n\n    Args:\n        directory (str): Path that serves as the root of the\n            search. Yielded paths will include this as a prefix.\n        patterns (str or list): A single pattern or list of\n            glob-formatted patterns to find under *directory*.\n        ignored (str or list): A single pattern or list of\n            glob-formatted patterns to ignore.\n        include_dirs (bool): Whether to include directories that match\n           patterns, as well. Defaults to ``False``.\n\n    For example, finding Python files in the current directory:\n\n    >>> _CUR_DIR = os.path.dirname(os.path.abspath(__file__))\n    >>> filenames = sorted(iter_find_files(_CUR_DIR, '*.py'))\n    >>> os.path.basename(filenames[-1])\n    'urlutils.py'\n\n    Or, Python files while ignoring emacs lockfiles:\n\n    >>> filenames = iter_find_files(_CUR_DIR, '*.py', ignored='.#*')\n\n    .. _glob: https://en.wikipedia.org/wiki/Glob_%28programming%29\n\n    "
    if isinstance(patterns, basestring):
        patterns = [patterns]
    pats_re = re.compile('|'.join([fnmatch.translate(p) for p in patterns]))
    if not ignored:
        ignored = []
    elif isinstance(ignored, basestring):
        ignored = [ignored]
    ign_re = re.compile('|'.join([fnmatch.translate(p) for p in ignored]))
    for (root, dirs, files) in os.walk(directory):
        if include_dirs:
            for basename in dirs:
                if pats_re.match(basename):
                    if ignored and ign_re.match(basename):
                        continue
                    filename = os.path.join(root, basename)
                    yield filename
        for basename in files:
            if pats_re.match(basename):
                if ignored and ign_re.match(basename):
                    continue
                filename = os.path.join(root, basename)
                yield filename
    return

def copy_tree(src, dst, symlinks=False, ignore=None):
    if False:
        while True:
            i = 10
    'The ``copy_tree`` function is an exact copy of the built-in\n    :func:`shutil.copytree`, with one key difference: it will not\n    raise an exception if part of the tree already exists. It achieves\n    this by using :func:`mkdir_p`.\n\n    As of Python 3.8, you may pass :func:`shutil.copytree` the\n    `dirs_exist_ok=True` flag to achieve the same effect.\n\n    Args:\n        src (str): Path of the source directory to copy.\n        dst (str): Destination path. Existing directories accepted.\n        symlinks (bool): If ``True``, copy symlinks rather than their\n            contents.\n        ignore (callable): A callable that takes a path and directory\n            listing, returning the files within the listing to be ignored.\n\n    For more details, check out :func:`shutil.copytree` and\n    :func:`shutil.copy2`.\n\n    '
    names = os.listdir(src)
    if ignore is not None:
        ignored_names = ignore(src, names)
    else:
        ignored_names = set()
    mkdir_p(dst)
    errors = []
    for name in names:
        if name in ignored_names:
            continue
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if symlinks and os.path.islink(srcname):
                linkto = os.readlink(srcname)
                os.symlink(linkto, dstname)
            elif os.path.isdir(srcname):
                copytree(srcname, dstname, symlinks, ignore)
            else:
                copy2(srcname, dstname)
        except Error as e:
            errors.extend(e.args[0])
        except EnvironmentError as why:
            errors.append((srcname, dstname, str(why)))
    try:
        copystat(src, dst)
    except OSError as why:
        if WindowsError is not None and isinstance(why, WindowsError):
            pass
        else:
            errors.append((src, dst, str(why)))
    if errors:
        raise Error(errors)
copytree = copy_tree
try:
    file
except NameError:
    file = object

class DummyFile(file):

    def __init__(self, path, mode='r', buffering=None):
        if False:
            while True:
                i = 10
        self.name = path
        self.mode = mode
        self.closed = False
        self.errors = None
        self.isatty = False
        self.encoding = None
        self.newlines = None
        self.softspace = 0

    def close(self):
        if False:
            while True:
                i = 10
        self.closed = True

    def fileno(self):
        if False:
            print('Hello World!')
        return -1

    def flush(self):
        if False:
            i = 10
            return i + 15
        if self.closed:
            raise ValueError('I/O operation on a closed file')
        return

    def next(self):
        if False:
            print('Hello World!')
        raise StopIteration()

    def read(self, size=0):
        if False:
            while True:
                i = 10
        if self.closed:
            raise ValueError('I/O operation on a closed file')
        return ''

    def readline(self, size=0):
        if False:
            while True:
                i = 10
        if self.closed:
            raise ValueError('I/O operation on a closed file')
        return ''

    def readlines(self, size=0):
        if False:
            return 10
        if self.closed:
            raise ValueError('I/O operation on a closed file')
        return []

    def seek(self):
        if False:
            for i in range(10):
                print('nop')
        if self.closed:
            raise ValueError('I/O operation on a closed file')
        return

    def tell(self):
        if False:
            while True:
                i = 10
        if self.closed:
            raise ValueError('I/O operation on a closed file')
        return 0

    def truncate(self):
        if False:
            return 10
        if self.closed:
            raise ValueError('I/O operation on a closed file')
        return

    def write(self, string):
        if False:
            for i in range(10):
                print('nop')
        if self.closed:
            raise ValueError('I/O operation on a closed file')
        return

    def writelines(self, list_of_strings):
        if False:
            for i in range(10):
                print('nop')
        if self.closed:
            raise ValueError('I/O operation on a closed file')
        return

    def __next__(self):
        if False:
            i = 10
            return i + 15
        raise StopIteration()

    def __enter__(self):
        if False:
            return 10
        if self.closed:
            raise ValueError('I/O operation on a closed file')
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            i = 10
            return i + 15
        return
if __name__ == '__main__':
    with atomic_save('/tmp/final.txt') as f:
        f.write('rofl')
        f.write('\n')