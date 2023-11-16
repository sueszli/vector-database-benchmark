import base64
import json
import uuid
import warnings
from typing import TYPE_CHECKING, Any, Dict, Generic, Iterable, Tuple, Type, TypeVar, Union
import cloudpickle
from prefect._internal.pydantic import HAS_PYDANTIC_V2
if HAS_PYDANTIC_V2:
    import pydantic.v1 as pydantic
else:
    import pydantic
from typing_extensions import Protocol
from prefect._internal.compatibility.deprecated import deprecated_callable, generate_deprecation_message
from prefect._internal.schemas.bases import PrefectBaseModel
if TYPE_CHECKING:
    from prefect.packaging.base import PackageManifest
T = TypeVar('T', bound='DataDocument')
D = TypeVar('D', bound=Any)
_SERIALIZERS: Dict[str, 'Serializer'] = {}

class Serializer(Protocol[D]):
    """
    Define a serializer that can encode data of type 'D' into bytes
    """

    @staticmethod
    def dumps(data: D, **kwargs: Any) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @staticmethod
    def loads(blob: bytes) -> D:
        if False:
            print('Hello World!')
        raise NotImplementedError

def register_serializer(encoding: Union[str, Tuple[str, ...]], serializer: Serializer=None):
    if False:
        while True:
            i = 10
    'Register dispatch of `func` on arguments of encoding `encoding`'

    def wrapper(serializer):
        if False:
            i = 10
            return i + 15
        if isinstance(encoding, tuple):
            for e in encoding:
                register_serializer(e, serializer)
        else:
            _SERIALIZERS[encoding] = serializer
        return serializer
    return wrapper(serializer) if serializer is not None else wrapper

def lookup_serializer(encoding: str) -> Serializer:
    if False:
        while True:
            i = 10
    'Return the serializer implementation for the given ``encoding``'
    try:
        return _SERIALIZERS[encoding]
    except KeyError:
        raise ValueError(f'Unregistered encoding {encoding!r}')

class DataDocument(PrefectBaseModel, Generic[D]):
    """
    A data document includes an encoding string and a blob of encoded data

    Subclasses can define the expected type for the blob's underlying type using the
    generic variable `D`.

    For example `DataDocument[str]` indicates that a string should be passed when
    creating the document and a string will be returned when it is decoded.
    """
    encoding: str
    blob: bytes
    _data: D
    __slots__ = ['_data']

    @classmethod
    @deprecated_callable(start_date='Sep 2022', help='Data documents should not be created. Use result persistence instead.')
    def encode(cls: Type['DataDocument'], encoding: str, data: D, **kwargs: Any) -> 'DataDocument[D]':
        if False:
            i = 10
            return i + 15
        '\n        Create a new data document\n\n        A serializer must be registered for the given `encoding`\n        '
        blob = lookup_serializer(encoding).dumps(data, **kwargs)
        inst = cls(blob=blob, encoding=encoding)
        inst._cache_data(data)
        return inst

    def decode(self) -> D:
        if False:
            for i in range(10):
                print('nop')
        "\n        Get the data from a data document\n\n        A serializer must be registered for the document's encoding\n        "
        if self.has_cached_data():
            return self._data
        data = lookup_serializer(self.encoding).loads(self.blob)
        self._cache_data(data)
        return data

    def _cache_data(self, data) -> None:
        if False:
            while True:
                i = 10
        object.__setattr__(self, '_data', data)

    def has_cached_data(self):
        if False:
            for i in range(10):
                print('nop')
        return hasattr(self, '_data')

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        if self.has_cached_data():
            return repr(self._data)
        else:
            return repr(self)

    def __repr__(self) -> str:
        if False:
            return 10
        return f'{type(self).__name__}(encoding={self.encoding!r})'

@register_serializer('json')
class DocumentJSONSerializer:
    """
    Serializes data to JSON.

    Input types must be compatible with the stdlib json library.

    Wraps the `json` library to serialize to UTF-8 bytes instead of string types.
    """

    @staticmethod
    @deprecated_callable(start_date='Sep 2022', help='Data document serializers should not be used. Use result serializers instead.')
    def dumps(data: Any) -> bytes:
        if False:
            while True:
                i = 10
        return json.dumps(data).encode()

    @staticmethod
    def loads(blob: bytes) -> Any:
        if False:
            return 10
        return json.loads(blob.decode())

@register_serializer('text')
class TextSerializer:

    @staticmethod
    @deprecated_callable(start_date='Sep 2022', help='Data document serializers should not be used. Use result serializers instead.')
    def dumps(data: str) -> bytes:
        if False:
            i = 10
            return i + 15
        return data.encode()

    @staticmethod
    def loads(blob: bytes) -> str:
        if False:
            print('Hello World!')
        return blob.decode()

@register_serializer('cloudpickle')
class DocumentPickleSerializer:
    """
    Serializes arbitrary objects using the pickle protocol.

    Wraps `cloudpickle` to encode bytes in base64 for safe transmission.
    """

    @staticmethod
    @deprecated_callable(start_date='Sep 2022', help='Data document serializers should not be used. Use result serializers instead.')
    def dumps(data: Any) -> bytes:
        if False:
            return 10
        data_bytes = cloudpickle.dumps(data)
        return base64.encodebytes(data_bytes)

    @staticmethod
    def loads(blob: bytes) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return cloudpickle.loads(base64.decodebytes(blob))

@register_serializer('package-manifest')
class PackageManifestSerializer:
    """
    Serializes a package manifest.
    """

    @staticmethod
    @deprecated_callable(start_date='Sep 2022', help='Data document serializers should not be used. Use result serializers instead.')
    def dumps(data: 'PackageManifest') -> bytes:
        if False:
            print('Hello World!')
        return data.json().encode()

    @staticmethod
    def loads(blob: bytes) -> 'PackageManifest':
        if False:
            print('Hello World!')
        from prefect.packaging.base import PackageManifest
        return PackageManifest.parse_raw(blob)

@register_serializer('result')
class ResultSerializer:
    """
    Serializes a result object
    """

    @staticmethod
    @deprecated_callable(start_date='Sep 2022', help='Data document serializers should not be used. Use result serializers instead.')
    def dumps(data: '_Result') -> bytes:
        if False:
            for i in range(10):
                print('nop')
        return data.json().encode()

    @staticmethod
    def loads(blob: bytes) -> '_Result':
        if False:
            return 10
        return _Result.parse_raw(blob)

def result_from_state_with_data_document(state, raise_on_failure: bool) -> Any:
    if False:
        for i in range(10):
            print('nop')
    warnings.warn(generate_deprecation_message('Retrieving results from states with data documents', start_date='Sep 2022'), DeprecationWarning, stacklevel=3)
    data = None
    if state.data:
        data = state.data.decode()
    from prefect.states import State
    if (state.is_failed() or state.is_crashed() or state.is_cancelled()) and raise_on_failure:
        if isinstance(data, Exception):
            raise data
        elif isinstance(data, BaseException):
            warnings.warn(f'State result is a {type(data).__name__!r} type and is not safe to re-raise, it will be returned instead.')
            return data
        elif isinstance(data, State):
            data.result(fetch=False)
        elif isinstance(data, Iterable) and all([isinstance(o, State) for o in data]):
            for state in data:
                state.result(fetch=False)
        raise TypeError(f'Unexpected result for failure state: {data!r} —— {type(data).__name__} cannot be resolved into an exception')
    return data

async def _persist_serialized_result(content: bytes, filesystem) -> DataDocument:
    key = uuid.uuid4().hex
    await filesystem.write_path(key, content)
    result = _Result(key=key, filesystem_document_id=filesystem._block_document_id)
    return DataDocument.encode('result', result)

async def _retrieve_serialized_result(document: DataDocument, client) -> bytes:
    from prefect.blocks.core import Block
    if document.encoding != 'result':
        raise TypeError(f"Got unsupported data document encoding of {document.encoding!r}. Expected 'result'.")
    result = document.decode()
    filesystem_document = await client.read_block_document(result.filesystem_document_id)
    filesystem = Block._from_block_document(filesystem_document)
    return await filesystem.read_path(result.key)

async def _retrieve_result(state, client):
    serialized_result = await _retrieve_serialized_result(state.data, client)
    return DataDocument.parse_raw(serialized_result).decode()

class _Result(pydantic.BaseModel):
    key: str
    filesystem_document_id: uuid.UUID