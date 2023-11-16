import typing
from shlex import shlex
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit
from starlette.concurrency import run_in_threadpool
from starlette.types import Scope

class Address(typing.NamedTuple):
    host: str
    port: int
_KeyType = typing.TypeVar('_KeyType')
_CovariantValueType = typing.TypeVar('_CovariantValueType', covariant=True)

class URL:

    def __init__(self, url: str='', scope: typing.Optional[Scope]=None, **components: typing.Any) -> None:
        if False:
            i = 10
            return i + 15
        if scope is not None:
            assert not url, 'Cannot set both "url" and "scope".'
            assert not components, 'Cannot set both "scope" and "**components".'
            scheme = scope.get('scheme', 'http')
            server = scope.get('server', None)
            path = scope.get('root_path', '') + scope['path']
            query_string = scope.get('query_string', b'')
            host_header = None
            for (key, value) in scope['headers']:
                if key == b'host':
                    host_header = value.decode('latin-1')
                    break
            if host_header is not None:
                url = f'{scheme}://{host_header}{path}'
            elif server is None:
                url = path
            else:
                (host, port) = server
                default_port = {'http': 80, 'https': 443, 'ws': 80, 'wss': 443}[scheme]
                if port == default_port:
                    url = f'{scheme}://{host}{path}'
                else:
                    url = f'{scheme}://{host}:{port}{path}'
            if query_string:
                url += '?' + query_string.decode()
        elif components:
            assert not url, 'Cannot set both "url" and "**components".'
            url = URL('').replace(**components).components.geturl()
        self._url = url

    @property
    def components(self) -> SplitResult:
        if False:
            print('Hello World!')
        if not hasattr(self, '_components'):
            self._components = urlsplit(self._url)
        return self._components

    @property
    def scheme(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.components.scheme

    @property
    def netloc(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.components.netloc

    @property
    def path(self) -> str:
        if False:
            print('Hello World!')
        return self.components.path

    @property
    def query(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.components.query

    @property
    def fragment(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.components.fragment

    @property
    def username(self) -> typing.Union[None, str]:
        if False:
            for i in range(10):
                print('nop')
        return self.components.username

    @property
    def password(self) -> typing.Union[None, str]:
        if False:
            print('Hello World!')
        return self.components.password

    @property
    def hostname(self) -> typing.Union[None, str]:
        if False:
            while True:
                i = 10
        return self.components.hostname

    @property
    def port(self) -> typing.Optional[int]:
        if False:
            print('Hello World!')
        return self.components.port

    @property
    def is_secure(self) -> bool:
        if False:
            while True:
                i = 10
        return self.scheme in ('https', 'wss')

    def replace(self, **kwargs: typing.Any) -> 'URL':
        if False:
            while True:
                i = 10
        if 'username' in kwargs or 'password' in kwargs or 'hostname' in kwargs or ('port' in kwargs):
            hostname = kwargs.pop('hostname', None)
            port = kwargs.pop('port', self.port)
            username = kwargs.pop('username', self.username)
            password = kwargs.pop('password', self.password)
            if hostname is None:
                netloc = self.netloc
                (_, _, hostname) = netloc.rpartition('@')
                if hostname[-1] != ']':
                    hostname = hostname.rsplit(':', 1)[0]
            netloc = hostname
            if port is not None:
                netloc += f':{port}'
            if username is not None:
                userpass = username
                if password is not None:
                    userpass += f':{password}'
                netloc = f'{userpass}@{netloc}'
            kwargs['netloc'] = netloc
        components = self.components._replace(**kwargs)
        return self.__class__(components.geturl())

    def include_query_params(self, **kwargs: typing.Any) -> 'URL':
        if False:
            return 10
        params = MultiDict(parse_qsl(self.query, keep_blank_values=True))
        params.update({str(key): str(value) for (key, value) in kwargs.items()})
        query = urlencode(params.multi_items())
        return self.replace(query=query)

    def replace_query_params(self, **kwargs: typing.Any) -> 'URL':
        if False:
            print('Hello World!')
        query = urlencode([(str(key), str(value)) for (key, value) in kwargs.items()])
        return self.replace(query=query)

    def remove_query_params(self, keys: typing.Union[str, typing.Sequence[str]]) -> 'URL':
        if False:
            i = 10
            return i + 15
        if isinstance(keys, str):
            keys = [keys]
        params = MultiDict(parse_qsl(self.query, keep_blank_values=True))
        for key in keys:
            params.pop(key, None)
        query = urlencode(params.multi_items())
        return self.replace(query=query)

    def __eq__(self, other: typing.Any) -> bool:
        if False:
            i = 10
            return i + 15
        return str(self) == str(other)

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return self._url

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        url = str(self)
        if self.password:
            url = str(self.replace(password='********'))
        return f'{self.__class__.__name__}({repr(url)})'

class URLPath(str):
    """
    A URL path string that may also hold an associated protocol and/or host.
    Used by the routing to return `url_path_for` matches.
    """

    def __new__(cls, path: str, protocol: str='', host: str='') -> 'URLPath':
        if False:
            for i in range(10):
                print('nop')
        assert protocol in ('http', 'websocket', '')
        return str.__new__(cls, path)

    def __init__(self, path: str, protocol: str='', host: str='') -> None:
        if False:
            return 10
        self.protocol = protocol
        self.host = host

    def make_absolute_url(self, base_url: typing.Union[str, URL]) -> URL:
        if False:
            while True:
                i = 10
        if isinstance(base_url, str):
            base_url = URL(base_url)
        if self.protocol:
            scheme = {'http': {True: 'https', False: 'http'}, 'websocket': {True: 'wss', False: 'ws'}}[self.protocol][base_url.is_secure]
        else:
            scheme = base_url.scheme
        netloc = self.host or base_url.netloc
        path = base_url.path.rstrip('/') + str(self)
        return URL(scheme=scheme, netloc=netloc, path=path)

class Secret:
    """
    Holds a string value that should not be revealed in tracebacks etc.
    You should cast the value to `str` at the point it is required.
    """

    def __init__(self, value: str):
        if False:
            i = 10
            return i + 15
        self._value = value

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        class_name = self.__class__.__name__
        return f"{class_name}('**********')"

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._value

    def __bool__(self) -> bool:
        if False:
            return 10
        return bool(self._value)

class CommaSeparatedStrings(typing.Sequence[str]):

    def __init__(self, value: typing.Union[str, typing.Sequence[str]]):
        if False:
            while True:
                i = 10
        if isinstance(value, str):
            splitter = shlex(value, posix=True)
            splitter.whitespace = ','
            splitter.whitespace_split = True
            self._items = [item.strip() for item in splitter]
        else:
            self._items = list(value)

    def __len__(self) -> int:
        if False:
            while True:
                i = 10
        return len(self._items)

    def __getitem__(self, index: typing.Union[int, slice]) -> typing.Any:
        if False:
            i = 10
            return i + 15
        return self._items[index]

    def __iter__(self) -> typing.Iterator[str]:
        if False:
            i = 10
            return i + 15
        return iter(self._items)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        class_name = self.__class__.__name__
        items = [item for item in self]
        return f'{class_name}({items!r})'

    def __str__(self) -> str:
        if False:
            return 10
        return ', '.join((repr(item) for item in self))

class ImmutableMultiDict(typing.Mapping[_KeyType, _CovariantValueType]):
    _dict: typing.Dict[_KeyType, _CovariantValueType]

    def __init__(self, *args: typing.Union['ImmutableMultiDict[_KeyType, _CovariantValueType]', typing.Mapping[_KeyType, _CovariantValueType], typing.Iterable[typing.Tuple[_KeyType, _CovariantValueType]]], **kwargs: typing.Any) -> None:
        if False:
            while True:
                i = 10
        assert len(args) < 2, 'Too many arguments.'
        value: typing.Any = args[0] if args else []
        if kwargs:
            value = ImmutableMultiDict(value).multi_items() + ImmutableMultiDict(kwargs).multi_items()
        if not value:
            _items: typing.List[typing.Tuple[typing.Any, typing.Any]] = []
        elif hasattr(value, 'multi_items'):
            value = typing.cast(ImmutableMultiDict[_KeyType, _CovariantValueType], value)
            _items = list(value.multi_items())
        elif hasattr(value, 'items'):
            value = typing.cast(typing.Mapping[_KeyType, _CovariantValueType], value)
            _items = list(value.items())
        else:
            value = typing.cast(typing.List[typing.Tuple[typing.Any, typing.Any]], value)
            _items = list(value)
        self._dict = {k: v for (k, v) in _items}
        self._list = _items

    def getlist(self, key: typing.Any) -> typing.List[_CovariantValueType]:
        if False:
            while True:
                i = 10
        return [item_value for (item_key, item_value) in self._list if item_key == key]

    def keys(self) -> typing.KeysView[_KeyType]:
        if False:
            print('Hello World!')
        return self._dict.keys()

    def values(self) -> typing.ValuesView[_CovariantValueType]:
        if False:
            for i in range(10):
                print('nop')
        return self._dict.values()

    def items(self) -> typing.ItemsView[_KeyType, _CovariantValueType]:
        if False:
            for i in range(10):
                print('nop')
        return self._dict.items()

    def multi_items(self) -> typing.List[typing.Tuple[_KeyType, _CovariantValueType]]:
        if False:
            i = 10
            return i + 15
        return list(self._list)

    def __getitem__(self, key: _KeyType) -> _CovariantValueType:
        if False:
            for i in range(10):
                print('nop')
        return self._dict[key]

    def __contains__(self, key: typing.Any) -> bool:
        if False:
            while True:
                i = 10
        return key in self._dict

    def __iter__(self) -> typing.Iterator[_KeyType]:
        if False:
            i = 10
            return i + 15
        return iter(self.keys())

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        return len(self._dict)

    def __eq__(self, other: typing.Any) -> bool:
        if False:
            print('Hello World!')
        if not isinstance(other, self.__class__):
            return False
        return sorted(self._list) == sorted(other._list)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        class_name = self.__class__.__name__
        items = self.multi_items()
        return f'{class_name}({items!r})'

class MultiDict(ImmutableMultiDict[typing.Any, typing.Any]):

    def __setitem__(self, key: typing.Any, value: typing.Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.setlist(key, [value])

    def __delitem__(self, key: typing.Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._list = [(k, v) for (k, v) in self._list if k != key]
        del self._dict[key]

    def pop(self, key: typing.Any, default: typing.Any=None) -> typing.Any:
        if False:
            while True:
                i = 10
        self._list = [(k, v) for (k, v) in self._list if k != key]
        return self._dict.pop(key, default)

    def popitem(self) -> typing.Tuple[typing.Any, typing.Any]:
        if False:
            return 10
        (key, value) = self._dict.popitem()
        self._list = [(k, v) for (k, v) in self._list if k != key]
        return (key, value)

    def poplist(self, key: typing.Any) -> typing.List[typing.Any]:
        if False:
            while True:
                i = 10
        values = [v for (k, v) in self._list if k == key]
        self.pop(key)
        return values

    def clear(self) -> None:
        if False:
            while True:
                i = 10
        self._dict.clear()
        self._list.clear()

    def setdefault(self, key: typing.Any, default: typing.Any=None) -> typing.Any:
        if False:
            print('Hello World!')
        if key not in self:
            self._dict[key] = default
            self._list.append((key, default))
        return self[key]

    def setlist(self, key: typing.Any, values: typing.List[typing.Any]) -> None:
        if False:
            i = 10
            return i + 15
        if not values:
            self.pop(key, None)
        else:
            existing_items = [(k, v) for (k, v) in self._list if k != key]
            self._list = existing_items + [(key, value) for value in values]
            self._dict[key] = values[-1]

    def append(self, key: typing.Any, value: typing.Any) -> None:
        if False:
            return 10
        self._list.append((key, value))
        self._dict[key] = value

    def update(self, *args: typing.Union['MultiDict', typing.Mapping[typing.Any, typing.Any], typing.List[typing.Tuple[typing.Any, typing.Any]]], **kwargs: typing.Any) -> None:
        if False:
            print('Hello World!')
        value = MultiDict(*args, **kwargs)
        existing_items = [(k, v) for (k, v) in self._list if k not in value.keys()]
        self._list = existing_items + value.multi_items()
        self._dict.update(value)

class QueryParams(ImmutableMultiDict[str, str]):
    """
    An immutable multidict.
    """

    def __init__(self, *args: typing.Union['ImmutableMultiDict[typing.Any, typing.Any]', typing.Mapping[typing.Any, typing.Any], typing.List[typing.Tuple[typing.Any, typing.Any]], str, bytes], **kwargs: typing.Any) -> None:
        if False:
            i = 10
            return i + 15
        assert len(args) < 2, 'Too many arguments.'
        value = args[0] if args else []
        if isinstance(value, str):
            super().__init__(parse_qsl(value, keep_blank_values=True), **kwargs)
        elif isinstance(value, bytes):
            super().__init__(parse_qsl(value.decode('latin-1'), keep_blank_values=True), **kwargs)
        else:
            super().__init__(*args, **kwargs)
        self._list = [(str(k), str(v)) for (k, v) in self._list]
        self._dict = {str(k): str(v) for (k, v) in self._dict.items()}

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return urlencode(self._list)

    def __repr__(self) -> str:
        if False:
            return 10
        class_name = self.__class__.__name__
        query_string = str(self)
        return f'{class_name}({query_string!r})'

class UploadFile:
    """
    An uploaded file included as part of the request data.
    """

    def __init__(self, file: typing.BinaryIO, *, size: typing.Optional[int]=None, filename: typing.Optional[str]=None, headers: 'typing.Optional[Headers]'=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.filename = filename
        self.file = file
        self.size = size
        self.headers = headers or Headers()

    @property
    def content_type(self) -> typing.Optional[str]:
        if False:
            i = 10
            return i + 15
        return self.headers.get('content-type', None)

    @property
    def _in_memory(self) -> bool:
        if False:
            while True:
                i = 10
        rolled_to_disk = getattr(self.file, '_rolled', True)
        return not rolled_to_disk

    async def write(self, data: bytes) -> None:
        if self.size is not None:
            self.size += len(data)
        if self._in_memory:
            self.file.write(data)
        else:
            await run_in_threadpool(self.file.write, data)

    async def read(self, size: int=-1) -> bytes:
        if self._in_memory:
            return self.file.read(size)
        return await run_in_threadpool(self.file.read, size)

    async def seek(self, offset: int) -> None:
        if self._in_memory:
            self.file.seek(offset)
        else:
            await run_in_threadpool(self.file.seek, offset)

    async def close(self) -> None:
        if self._in_memory:
            self.file.close()
        else:
            await run_in_threadpool(self.file.close)

class FormData(ImmutableMultiDict[str, typing.Union[UploadFile, str]]):
    """
    An immutable multidict, containing both file uploads and text input.
    """

    def __init__(self, *args: typing.Union['FormData', typing.Mapping[str, typing.Union[str, UploadFile]], typing.List[typing.Tuple[str, typing.Union[str, UploadFile]]]], **kwargs: typing.Union[str, UploadFile]) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)

    async def close(self) -> None:
        for (key, value) in self.multi_items():
            if isinstance(value, UploadFile):
                await value.close()

class Headers(typing.Mapping[str, str]):
    """
    An immutable, case-insensitive multidict.
    """

    def __init__(self, headers: typing.Optional[typing.Mapping[str, str]]=None, raw: typing.Optional[typing.List[typing.Tuple[bytes, bytes]]]=None, scope: typing.Optional[typing.MutableMapping[str, typing.Any]]=None) -> None:
        if False:
            i = 10
            return i + 15
        self._list: typing.List[typing.Tuple[bytes, bytes]] = []
        if headers is not None:
            assert raw is None, 'Cannot set both "headers" and "raw".'
            assert scope is None, 'Cannot set both "headers" and "scope".'
            self._list = [(key.lower().encode('latin-1'), value.encode('latin-1')) for (key, value) in headers.items()]
        elif raw is not None:
            assert scope is None, 'Cannot set both "raw" and "scope".'
            self._list = raw
        elif scope is not None:
            self._list = scope['headers'] = list(scope['headers'])

    @property
    def raw(self) -> typing.List[typing.Tuple[bytes, bytes]]:
        if False:
            for i in range(10):
                print('nop')
        return list(self._list)

    def keys(self) -> typing.List[str]:
        if False:
            while True:
                i = 10
        return [key.decode('latin-1') for (key, value) in self._list]

    def values(self) -> typing.List[str]:
        if False:
            for i in range(10):
                print('nop')
        return [value.decode('latin-1') for (key, value) in self._list]

    def items(self) -> typing.List[typing.Tuple[str, str]]:
        if False:
            for i in range(10):
                print('nop')
        return [(key.decode('latin-1'), value.decode('latin-1')) for (key, value) in self._list]

    def getlist(self, key: str) -> typing.List[str]:
        if False:
            for i in range(10):
                print('nop')
        get_header_key = key.lower().encode('latin-1')
        return [item_value.decode('latin-1') for (item_key, item_value) in self._list if item_key == get_header_key]

    def mutablecopy(self) -> 'MutableHeaders':
        if False:
            print('Hello World!')
        return MutableHeaders(raw=self._list[:])

    def __getitem__(self, key: str) -> str:
        if False:
            return 10
        get_header_key = key.lower().encode('latin-1')
        for (header_key, header_value) in self._list:
            if header_key == get_header_key:
                return header_value.decode('latin-1')
        raise KeyError(key)

    def __contains__(self, key: typing.Any) -> bool:
        if False:
            while True:
                i = 10
        get_header_key = key.lower().encode('latin-1')
        for (header_key, header_value) in self._list:
            if header_key == get_header_key:
                return True
        return False

    def __iter__(self) -> typing.Iterator[typing.Any]:
        if False:
            for i in range(10):
                print('nop')
        return iter(self.keys())

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return len(self._list)

    def __eq__(self, other: typing.Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, Headers):
            return False
        return sorted(self._list) == sorted(other._list)

    def __repr__(self) -> str:
        if False:
            return 10
        class_name = self.__class__.__name__
        as_dict = dict(self.items())
        if len(as_dict) == len(self):
            return f'{class_name}({as_dict!r})'
        return f'{class_name}(raw={self.raw!r})'

class MutableHeaders(Headers):

    def __setitem__(self, key: str, value: str) -> None:
        if False:
            return 10
        '\n        Set the header `key` to `value`, removing any duplicate entries.\n        Retains insertion order.\n        '
        set_key = key.lower().encode('latin-1')
        set_value = value.encode('latin-1')
        found_indexes: 'typing.List[int]' = []
        for (idx, (item_key, item_value)) in enumerate(self._list):
            if item_key == set_key:
                found_indexes.append(idx)
        for idx in reversed(found_indexes[1:]):
            del self._list[idx]
        if found_indexes:
            idx = found_indexes[0]
            self._list[idx] = (set_key, set_value)
        else:
            self._list.append((set_key, set_value))

    def __delitem__(self, key: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove the header `key`.\n        '
        del_key = key.lower().encode('latin-1')
        pop_indexes: 'typing.List[int]' = []
        for (idx, (item_key, item_value)) in enumerate(self._list):
            if item_key == del_key:
                pop_indexes.append(idx)
        for idx in reversed(pop_indexes):
            del self._list[idx]

    def __ior__(self, other: typing.Mapping[str, str]) -> 'MutableHeaders':
        if False:
            print('Hello World!')
        if not isinstance(other, typing.Mapping):
            raise TypeError(f'Expected a mapping but got {other.__class__.__name__}')
        self.update(other)
        return self

    def __or__(self, other: typing.Mapping[str, str]) -> 'MutableHeaders':
        if False:
            while True:
                i = 10
        if not isinstance(other, typing.Mapping):
            raise TypeError(f'Expected a mapping but got {other.__class__.__name__}')
        new = self.mutablecopy()
        new.update(other)
        return new

    @property
    def raw(self) -> typing.List[typing.Tuple[bytes, bytes]]:
        if False:
            print('Hello World!')
        return self._list

    def setdefault(self, key: str, value: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        If the header `key` does not exist, then set it to `value`.\n        Returns the header value.\n        '
        set_key = key.lower().encode('latin-1')
        set_value = value.encode('latin-1')
        for (idx, (item_key, item_value)) in enumerate(self._list):
            if item_key == set_key:
                return item_value.decode('latin-1')
        self._list.append((set_key, set_value))
        return value

    def update(self, other: typing.Mapping[str, str]) -> None:
        if False:
            i = 10
            return i + 15
        for (key, val) in other.items():
            self[key] = val

    def append(self, key: str, value: str) -> None:
        if False:
            print('Hello World!')
        '\n        Append a header, preserving any duplicate entries.\n        '
        append_key = key.lower().encode('latin-1')
        append_value = value.encode('latin-1')
        self._list.append((append_key, append_value))

    def add_vary_header(self, vary: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        existing = self.get('vary')
        if existing is not None:
            vary = ', '.join([existing, vary])
        self['vary'] = vary

class State:
    """
    An object that can be used to store arbitrary state.

    Used for `request.state` and `app.state`.
    """
    _state: typing.Dict[str, typing.Any]

    def __init__(self, state: typing.Optional[typing.Dict[str, typing.Any]]=None):
        if False:
            while True:
                i = 10
        if state is None:
            state = {}
        super().__setattr__('_state', state)

    def __setattr__(self, key: typing.Any, value: typing.Any) -> None:
        if False:
            return 10
        self._state[key] = value

    def __getattr__(self, key: typing.Any) -> typing.Any:
        if False:
            print('Hello World!')
        try:
            return self._state[key]
        except KeyError:
            message = "'{}' object has no attribute '{}'"
            raise AttributeError(message.format(self.__class__.__name__, key))

    def __delattr__(self, key: typing.Any) -> None:
        if False:
            i = 10
            return i + 15
        del self._state[key]