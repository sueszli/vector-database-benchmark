import os
import pathlib
import posixpath
from typing import Callable
from urllib.parse import urlparse
from dvc.utils import relpath
from dvc.utils.objects import cached_property

class _BasePath:

    def overlaps(self, other):
        if False:
            return 10
        if isinstance(other, (str, bytes)):
            other = self.__class__(other)
        elif self.__class__ != other.__class__:
            return False
        return self.isin_or_eq(other) or other.isin(self)

    def isin_or_eq(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self == other or self.isin(other)

class PathInfo(pathlib.PurePath, _BasePath):
    __slots__ = ()
    scheme = 'local'

    def __new__(cls, *args):
        if False:
            for i in range(10):
                print('nop')
        if cls is PathInfo:
            cls = WindowsPathInfo if os.name == 'nt' else PosixPathInfo
        return cls._from_parts(args)

    def as_posix(self):
        if False:
            print('Hello World!')
        f = self._flavour
        return self.fspath.replace(f.sep, '/')

    def __str__(self):
        if False:
            return 10
        path = self.__fspath__()
        return relpath(path)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f"{type(self).__name__}: '{self}'"

    def __fspath__(self):
        if False:
            while True:
                i = 10
        return pathlib.PurePath.__str__(self)

    @property
    def fspath(self):
        if False:
            while True:
                i = 10
        return os.fspath(self)
    url = fspath
    path = fspath

    def relpath(self, other):
        if False:
            while True:
                i = 10
        return self.__class__(relpath(self, other))

    def isin(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, (str, bytes)):
            other = self.__class__(other)
        elif self.__class__ != other.__class__:
            return False
        n = len(other._cparts)
        return len(self._cparts) > n and self._cparts[:n] == other._cparts

    def relative_to(self, other):
        if False:
            i = 10
            return i + 15
        try:
            path = super().relative_to(other)
        except ValueError:
            path = relpath(self, other)
        return self.__class__(path)

class WindowsPathInfo(PathInfo, pathlib.PureWindowsPath):
    pass

class PosixPathInfo(PathInfo, pathlib.PurePosixPath):
    pass

class _URLPathInfo(PosixPathInfo):

    def __str__(self):
        if False:
            return 10
        return self.__fspath__()
    __unicode__ = __str__

class _URLPathParents:

    def __init__(self, src):
        if False:
            for i in range(10):
                print('nop')
        self.src = src
        self._parents = self.src._path.parents

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self._parents)

    def __getitem__(self, idx):
        if False:
            return 10
        return self.src.replace(path=self._parents[idx])

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'<{self.src}.parents>'

class URLInfo(_BasePath):
    DEFAULT_PORTS = {'http': 80, 'https': 443, 'ssh': 22, 'hdfs': 0}

    def __init__(self, url):
        if False:
            return 10
        p = urlparse(url)
        assert not p.query
        assert not p.params
        assert not p.fragment
        assert p.password is None
        self._fill_parts(p.scheme, p.hostname, p.username, p.port, p.path)

    @classmethod
    def from_parts(cls, scheme=None, host=None, user=None, port=None, path='', netloc=None):
        if False:
            return 10
        assert bool(host) ^ bool(netloc)
        if netloc is not None:
            return cls(f'{scheme}://{netloc}{path}')
        obj = cls.__new__(cls)
        obj._fill_parts(scheme, host, user, port, path)
        return obj

    def _fill_parts(self, scheme, host, user, port, path):
        if False:
            return 10
        assert scheme != 'remote'
        assert isinstance(path, (str, bytes, _URLPathInfo))
        (self.scheme, self.host, self.user) = (scheme, host, user)
        self.port = int(port) if port else self.DEFAULT_PORTS.get(self.scheme)
        if isinstance(path, _URLPathInfo):
            self._spath = str(path)
            self._path = path
        else:
            if path and path[0] != '/':
                path = '/' + path
            self._spath = path

    @property
    def _base_parts(self):
        if False:
            while True:
                i = 10
        return (self.scheme, self.host, self.user, self.port)

    @property
    def parts(self):
        if False:
            while True:
                i = 10
        return self._base_parts + self._path.parts

    def replace(self, path=None):
        if False:
            i = 10
            return i + 15
        return self.from_parts(*self._base_parts, path=path)

    @cached_property
    def url(self) -> str:
        if False:
            return 10
        return f'{self.scheme}://{self.netloc}{self._spath}'

    def __str__(self):
        if False:
            return 10
        return self.url

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f"{type(self).__name__}: '{self}'"

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, (str, bytes)):
            other = self.__class__(other)
        return self.__class__ == other.__class__ and self._base_parts == other._base_parts and (self._path == other._path)

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(self.parts)

    def __div__(self, other):
        if False:
            i = 10
            return i + 15
        return self.replace(path=posixpath.join(self._spath, other))

    def joinpath(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return self.replace(path=posixpath.join(self._spath, *args))
    __truediv__ = __div__

    @property
    def path(self):
        if False:
            while True:
                i = 10
        return self._spath

    @cached_property
    def _path(self) -> '_URLPathInfo':
        if False:
            i = 10
            return i + 15
        return _URLPathInfo(self._spath)

    @property
    def name(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._path.name

    @cached_property
    def netloc(self) -> str:
        if False:
            print('Hello World!')
        netloc = self.host
        if self.user:
            netloc = self.user + '@' + netloc
        if self.port and int(self.port) != self.DEFAULT_PORTS.get(self.scheme):
            netloc += ':' + str(self.port)
        return netloc

    @property
    def bucket(self) -> str:
        if False:
            while True:
                i = 10
        return self.netloc

    @property
    def parent(self):
        if False:
            for i in range(10):
                print('nop')
        return self.replace(path=self._path.parent)

    @property
    def parents(self):
        if False:
            return 10
        return _URLPathParents(self)

    def relative_to(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, (str, bytes)):
            other = self.__class__(other)
        if self.__class__ != other.__class__:
            msg = f"'{self}' has incompatible class with '{other}'"
            raise ValueError(msg)
        if self._base_parts != other._base_parts:
            msg = f"'{self}' does not start with '{other}'"
            raise ValueError(msg)
        return self._path.relative_to(other._path)

    def isin(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, (str, bytes)):
            other = self.__class__(other)
        elif self.__class__ != other.__class__:
            return False
        return self._base_parts == other._base_parts and self._path.isin(other._path)

class CloudURLInfo(URLInfo):

    @property
    def path(self):
        if False:
            print('Hello World!')
        return self._spath.lstrip('/')

class HTTPURLInfo(URLInfo):
    __hash__: Callable[['HTTPURLInfo'], int] = URLInfo.__hash__

    def __init__(self, url):
        if False:
            while True:
                i = 10
        p = urlparse(url)
        stripped = p._replace(params=None, query=None, fragment=None)
        super().__init__(stripped.geturl())
        self.params = p.params
        self.query = p.query
        self.fragment = p.fragment

    def replace(self, path=None):
        if False:
            i = 10
            return i + 15
        return self.from_parts(*self._base_parts, params=self.params, query=self.query, fragment=self.fragment, path=path)

    @classmethod
    def from_parts(cls, scheme=None, host=None, user=None, port=None, path='', netloc=None, params=None, query=None, fragment=None):
        if False:
            while True:
                i = 10
        assert bool(host) ^ bool(netloc)
        if netloc is not None:
            return cls('{}://{}{}{}{}{}'.format(scheme, netloc, path, ';' + params if params else '', '?' + query if query else '', '#' + fragment if fragment else ''))
        obj = cls.__new__(cls)
        obj._fill_parts(scheme, host, user, port, path)
        obj.params = params
        obj.query = query
        obj.fragment = fragment
        return obj

    @property
    def _extra_parts(self):
        if False:
            return 10
        return (self.params, self.query, self.fragment)

    @property
    def parts(self):
        if False:
            return 10
        return self._base_parts + self._path.parts + self._extra_parts

    @cached_property
    def url(self) -> str:
        if False:
            return 10
        return '{}://{}{}{}{}{}'.format(self.scheme, self.netloc, self._spath, ';' + self.params if self.params else '', '?' + self.query if self.query else '', '#' + self.fragment if self.fragment else '')

    def __eq__(self, other):
        if False:
            return 10
        if isinstance(other, (str, bytes)):
            other = self.__class__(other)
        return self.__class__ == other.__class__ and self._base_parts == other._base_parts and (self._path == other._path) and (self._extra_parts == other._extra_parts)

class WebDAVURLInfo(URLInfo):

    @cached_property
    def url(self) -> str:
        if False:
            i = 10
            return i + 15
        return '{}://{}{}'.format(self.scheme.replace('webdav', 'http'), self.netloc, self._spath)