import io
import posixpath
import zipfile
import itertools
import contextlib
import sys
import pathlib
if sys.version_info < (3, 7):
    from collections import OrderedDict
else:
    OrderedDict = dict
__all__ = ['Path']

def _parents(path):
    if False:
        while True:
            i = 10
    "\n    Given a path with elements separated by\n    posixpath.sep, generate all parents of that path.\n\n    >>> list(_parents('b/d'))\n    ['b']\n    >>> list(_parents('/b/d/'))\n    ['/b']\n    >>> list(_parents('b/d/f/'))\n    ['b/d', 'b']\n    >>> list(_parents('b'))\n    []\n    >>> list(_parents(''))\n    []\n    "
    return itertools.islice(_ancestry(path), 1, None)

def _ancestry(path):
    if False:
        return 10
    "\n    Given a path with elements separated by\n    posixpath.sep, generate all elements of that path\n\n    >>> list(_ancestry('b/d'))\n    ['b/d', 'b']\n    >>> list(_ancestry('/b/d/'))\n    ['/b/d', '/b']\n    >>> list(_ancestry('b/d/f/'))\n    ['b/d/f', 'b/d', 'b']\n    >>> list(_ancestry('b'))\n    ['b']\n    >>> list(_ancestry(''))\n    []\n    "
    path = path.rstrip(posixpath.sep)
    while path and path != posixpath.sep:
        yield path
        (path, tail) = posixpath.split(path)
_dedupe = OrderedDict.fromkeys
'Deduplicate an iterable in original order'

def _difference(minuend, subtrahend):
    if False:
        i = 10
        return i + 15
    '\n    Return items in minuend not in subtrahend, retaining order\n    with O(1) lookup.\n    '
    return itertools.filterfalse(set(subtrahend).__contains__, minuend)

class CompleteDirs(zipfile.ZipFile):
    """
    A ZipFile subclass that ensures that implied directories
    are always included in the namelist.
    """

    @staticmethod
    def _implied_dirs(names):
        if False:
            for i in range(10):
                print('nop')
        parents = itertools.chain.from_iterable(map(_parents, names))
        as_dirs = (p + posixpath.sep for p in parents)
        return _dedupe(_difference(as_dirs, names))

    def namelist(self):
        if False:
            return 10
        names = super(CompleteDirs, self).namelist()
        return names + list(self._implied_dirs(names))

    def _name_set(self):
        if False:
            return 10
        return set(self.namelist())

    def resolve_dir(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        If the name represents a directory, return that name\n        as a directory (with the trailing slash).\n        '
        names = self._name_set()
        dirname = name + '/'
        dir_match = name not in names and dirname in names
        return dirname if dir_match else name

    @classmethod
    def make(cls, source):
        if False:
            print('Hello World!')
        '\n        Given a source (filename or zipfile), return an\n        appropriate CompleteDirs subclass.\n        '
        if isinstance(source, CompleteDirs):
            return source
        if not isinstance(source, zipfile.ZipFile):
            return cls(_pathlib_compat(source))
        if 'r' not in source.mode:
            cls = CompleteDirs
        source.__class__ = cls
        return source

class FastLookup(CompleteDirs):
    """
    ZipFile subclass to ensure implicit
    dirs exist and are resolved rapidly.
    """

    def namelist(self):
        if False:
            print('Hello World!')
        with contextlib.suppress(AttributeError):
            return self.__names
        self.__names = super(FastLookup, self).namelist()
        return self.__names

    def _name_set(self):
        if False:
            print('Hello World!')
        with contextlib.suppress(AttributeError):
            return self.__lookup
        self.__lookup = super(FastLookup, self)._name_set()
        return self.__lookup

def _pathlib_compat(path):
    if False:
        i = 10
        return i + 15
    '\n    For path-like objects, convert to a filename for compatibility\n    on Python 3.6.1 and earlier.\n    '
    try:
        return path.__fspath__()
    except AttributeError:
        return str(path)

class Path:
    """
    A pathlib-compatible interface for zip files.

    Consider a zip file with this structure::

        .
        ├── a.txt
        └── b
            ├── c.txt
            └── d
                └── e.txt

    >>> data = io.BytesIO()
    >>> zf = zipfile.ZipFile(data, 'w')
    >>> zf.writestr('a.txt', 'content of a')
    >>> zf.writestr('b/c.txt', 'content of c')
    >>> zf.writestr('b/d/e.txt', 'content of e')
    >>> zf.filename = 'mem/abcde.zip'

    Path accepts the zipfile object itself or a filename

    >>> root = Path(zf)

    From there, several path operations are available.

    Directory iteration (including the zip file itself):

    >>> a, b = root.iterdir()
    >>> a
    Path('mem/abcde.zip', 'a.txt')
    >>> b
    Path('mem/abcde.zip', 'b/')

    name property:

    >>> b.name
    'b'

    join with divide operator:

    >>> c = b / 'c.txt'
    >>> c
    Path('mem/abcde.zip', 'b/c.txt')
    >>> c.name
    'c.txt'

    Read text:

    >>> c.read_text()
    'content of c'

    existence:

    >>> c.exists()
    True
    >>> (b / 'missing.txt').exists()
    False

    Coercion to string:

    >>> import os
    >>> str(c).replace(os.sep, posixpath.sep)
    'mem/abcde.zip/b/c.txt'

    At the root, ``name``, ``filename``, and ``parent``
    resolve to the zipfile. Note these attributes are not
    valid and will raise a ``ValueError`` if the zipfile
    has no filename.

    >>> root.name
    'abcde.zip'
    >>> str(root.filename).replace(os.sep, posixpath.sep)
    'mem/abcde.zip'
    >>> str(root.parent)
    'mem'
    """
    __repr = '{self.__class__.__name__}({self.root.filename!r}, {self.at!r})'

    def __init__(self, root, at=''):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a Path from a ZipFile or filename.\n\n        Note: When the source is an existing ZipFile object,\n        its type (__class__) will be mutated to a\n        specialized type. If the caller wishes to retain the\n        original type, the caller should either create a\n        separate ZipFile object or pass a filename.\n        '
        self.root = FastLookup.make(root)
        self.at = at

    def open(self, mode='r', *args, pwd=None, **kwargs):
        if False:
            return 10
        '\n        Open this entry as text or binary following the semantics\n        of ``pathlib.Path.open()`` by passing arguments through\n        to io.TextIOWrapper().\n        '
        if self.is_dir():
            raise IsADirectoryError(self)
        zip_mode = mode[0]
        if not self.exists() and zip_mode == 'r':
            raise FileNotFoundError(self)
        stream = self.root.open(self.at, zip_mode, pwd=pwd)
        if 'b' in mode:
            if args or kwargs:
                raise ValueError('encoding args invalid for binary operation')
            return stream
        return io.TextIOWrapper(stream, *args, **kwargs)

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return pathlib.Path(self.at).name or self.filename.name

    @property
    def suffix(self):
        if False:
            i = 10
            return i + 15
        return pathlib.Path(self.at).suffix or self.filename.suffix

    @property
    def suffixes(self):
        if False:
            for i in range(10):
                print('nop')
        return pathlib.Path(self.at).suffixes or self.filename.suffixes

    @property
    def stem(self):
        if False:
            for i in range(10):
                print('nop')
        return pathlib.Path(self.at).stem or self.filename.stem

    @property
    def filename(self):
        if False:
            while True:
                i = 10
        return pathlib.Path(self.root.filename).joinpath(self.at)

    def read_text(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        with self.open('r', *args, **kwargs) as strm:
            return strm.read()

    def read_bytes(self):
        if False:
            i = 10
            return i + 15
        with self.open('rb') as strm:
            return strm.read()

    def _is_child(self, path):
        if False:
            print('Hello World!')
        return posixpath.dirname(path.at.rstrip('/')) == self.at.rstrip('/')

    def _next(self, at):
        if False:
            return 10
        return self.__class__(self.root, at)

    def is_dir(self):
        if False:
            return 10
        return not self.at or self.at.endswith('/')

    def is_file(self):
        if False:
            i = 10
            return i + 15
        return self.exists() and (not self.is_dir())

    def exists(self):
        if False:
            return 10
        return self.at in self.root._name_set()

    def iterdir(self):
        if False:
            print('Hello World!')
        if not self.is_dir():
            raise ValueError("Can't listdir a file")
        subs = map(self._next, self.root.namelist())
        return filter(self._is_child, subs)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return posixpath.join(self.root.filename, self.at)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__repr.format(self=self)

    def joinpath(self, *other):
        if False:
            i = 10
            return i + 15
        next = posixpath.join(self.at, *map(_pathlib_compat, other))
        return self._next(self.root.resolve_dir(next))
    __truediv__ = joinpath

    @property
    def parent(self):
        if False:
            return 10
        if not self.at:
            return self.filename.parent
        parent_at = posixpath.dirname(self.at.rstrip('/'))
        if parent_at:
            parent_at += '/'
        return self._next(parent_at)