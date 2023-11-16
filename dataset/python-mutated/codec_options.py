"""Tools for specifying BSON codec options."""
from __future__ import annotations
import abc
import datetime
import enum
from collections.abc import MutableMapping as _MutableMapping
from typing import TYPE_CHECKING, Any, Callable, Generic, Iterable, Mapping, NamedTuple, Optional, Tuple, Type, Union, cast
from bson.binary import ALL_UUID_REPRESENTATIONS, UUID_REPRESENTATION_NAMES, UuidRepresentation
from bson.typings import _DocumentType
_RAW_BSON_DOCUMENT_MARKER = 101

def _raw_document_class(document_class: Any) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Determine if a document_class is a RawBSONDocument class.'
    marker = getattr(document_class, '_type_marker', None)
    return marker == _RAW_BSON_DOCUMENT_MARKER

class TypeEncoder(abc.ABC):
    """Base class for defining type codec classes which describe how a
    custom type can be transformed to one of the types BSON understands.

    Codec classes must implement the ``python_type`` attribute, and the
    ``transform_python`` method to support encoding.

    See :ref:`custom-type-type-codec` documentation for an example.
    """

    @abc.abstractproperty
    def python_type(self) -> Any:
        if False:
            while True:
                i = 10
        'The Python type to be converted into something serializable.'

    @abc.abstractmethod
    def transform_python(self, value: Any) -> Any:
        if False:
            while True:
                i = 10
        'Convert the given Python object into something serializable.'

class TypeDecoder(abc.ABC):
    """Base class for defining type codec classes which describe how a
    BSON type can be transformed to a custom type.

    Codec classes must implement the ``bson_type`` attribute, and the
    ``transform_bson`` method to support decoding.

    See :ref:`custom-type-type-codec` documentation for an example.
    """

    @abc.abstractproperty
    def bson_type(self) -> Any:
        if False:
            print('Hello World!')
        'The BSON type to be converted into our own type.'

    @abc.abstractmethod
    def transform_bson(self, value: Any) -> Any:
        if False:
            print('Hello World!')
        'Convert the given BSON value into our own type.'

class TypeCodec(TypeEncoder, TypeDecoder):
    """Base class for defining type codec classes which describe how a
    custom type can be transformed to/from one of the types :mod:`bson`
    can already encode/decode.

    Codec classes must implement the ``python_type`` attribute, and the
    ``transform_python`` method to support encoding, as well as the
    ``bson_type`` attribute, and the ``transform_bson`` method to support
    decoding.

    See :ref:`custom-type-type-codec` documentation for an example.
    """
_Codec = Union[TypeEncoder, TypeDecoder, TypeCodec]
_Fallback = Callable[[Any], Any]

class TypeRegistry:
    """Encapsulates type codecs used in encoding and / or decoding BSON, as
    well as the fallback encoder. Type registries cannot be modified after
    instantiation.

    ``TypeRegistry`` can be initialized with an iterable of type codecs, and
    a callable for the fallback encoder::

      >>> from bson.codec_options import TypeRegistry
      >>> type_registry = TypeRegistry([Codec1, Codec2, Codec3, ...],
      ...                              fallback_encoder)

    See :ref:`custom-type-type-registry` documentation for an example.

    :Parameters:
      - `type_codecs` (optional): iterable of type codec instances. If
        ``type_codecs`` contains multiple codecs that transform a single
        python or BSON type, the transformation specified by the type codec
        occurring last prevails. A TypeError will be raised if one or more
        type codecs modify the encoding behavior of a built-in :mod:`bson`
        type.
      - `fallback_encoder` (optional): callable that accepts a single,
        unencodable python value and transforms it into a type that
        :mod:`bson` can encode. See :ref:`fallback-encoder-callable`
        documentation for an example.
    """

    def __init__(self, type_codecs: Optional[Iterable[_Codec]]=None, fallback_encoder: Optional[_Fallback]=None) -> None:
        if False:
            while True:
                i = 10
        self.__type_codecs = list(type_codecs or [])
        self._fallback_encoder = fallback_encoder
        self._encoder_map: dict[Any, Any] = {}
        self._decoder_map: dict[Any, Any] = {}
        if self._fallback_encoder is not None:
            if not callable(fallback_encoder):
                raise TypeError('fallback_encoder %r is not a callable' % fallback_encoder)
        for codec in self.__type_codecs:
            is_valid_codec = False
            if isinstance(codec, TypeEncoder):
                self._validate_type_encoder(codec)
                is_valid_codec = True
                self._encoder_map[codec.python_type] = codec.transform_python
            if isinstance(codec, TypeDecoder):
                is_valid_codec = True
                self._decoder_map[codec.bson_type] = codec.transform_bson
            if not is_valid_codec:
                raise TypeError(f'Expected an instance of {TypeEncoder.__name__}, {TypeDecoder.__name__}, or {TypeCodec.__name__}, got {codec!r} instead')

    def _validate_type_encoder(self, codec: _Codec) -> None:
        if False:
            for i in range(10):
                print('nop')
        from bson import _BUILT_IN_TYPES
        for pytype in _BUILT_IN_TYPES:
            if issubclass(cast(TypeCodec, codec).python_type, pytype):
                err_msg = f'TypeEncoders cannot change how built-in types are encoded (encoder {codec} transforms type {pytype})'
                raise TypeError(err_msg)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return '{}(type_codecs={!r}, fallback_encoder={!r})'.format(self.__class__.__name__, self.__type_codecs, self._fallback_encoder)

    def __eq__(self, other: Any) -> Any:
        if False:
            while True:
                i = 10
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._decoder_map == other._decoder_map and self._encoder_map == other._encoder_map and (self._fallback_encoder == other._fallback_encoder)

class DatetimeConversion(int, enum.Enum):
    """Options for decoding BSON datetimes."""
    DATETIME = 1
    'Decode a BSON UTC datetime as a :class:`datetime.datetime`.\n\n    BSON UTC datetimes that cannot be represented as a\n    :class:`~datetime.datetime` will raise an :class:`OverflowError`\n    or a :class:`ValueError`.\n\n    .. versionadded 4.3\n    '
    DATETIME_CLAMP = 2
    'Decode a BSON UTC datetime as a :class:`datetime.datetime`, clamping\n    to :attr:`~datetime.datetime.min` and :attr:`~datetime.datetime.max`.\n\n    .. versionadded 4.3\n    '
    DATETIME_MS = 3
    'Decode a BSON UTC datetime as a :class:`~bson.datetime_ms.DatetimeMS`\n    object.\n\n    .. versionadded 4.3\n    '
    DATETIME_AUTO = 4
    'Decode a BSON UTC datetime as a :class:`datetime.datetime` if possible,\n    and a :class:`~bson.datetime_ms.DatetimeMS` if not.\n\n    .. versionadded 4.3\n    '

class _BaseCodecOptions(NamedTuple):
    document_class: Type[Mapping[str, Any]]
    tz_aware: bool
    uuid_representation: int
    unicode_decode_error_handler: str
    tzinfo: Optional[datetime.tzinfo]
    type_registry: TypeRegistry
    datetime_conversion: Optional[DatetimeConversion]
if TYPE_CHECKING:

    class CodecOptions(Tuple[_DocumentType], Generic[_DocumentType]):
        document_class: Type[_DocumentType]
        tz_aware: bool
        uuid_representation: int
        unicode_decode_error_handler: Optional[str]
        tzinfo: Optional[datetime.tzinfo]
        type_registry: TypeRegistry
        datetime_conversion: Optional[int]

        def __new__(cls: Type[CodecOptions[_DocumentType]], document_class: Optional[Type[_DocumentType]]=..., tz_aware: bool=..., uuid_representation: Optional[int]=..., unicode_decode_error_handler: Optional[str]=..., tzinfo: Optional[datetime.tzinfo]=..., type_registry: Optional[TypeRegistry]=..., datetime_conversion: Optional[int]=...) -> CodecOptions[_DocumentType]:
            if False:
                i = 10
                return i + 15
            ...

        def with_options(self, **kwargs: Any) -> CodecOptions[Any]:
            if False:
                i = 10
                return i + 15
            ...

        def _arguments_repr(self) -> str:
            if False:
                for i in range(10):
                    print('nop')
            ...

        def _options_dict(self) -> dict[Any, Any]:
            if False:
                print('Hello World!')
            ...

        @classmethod
        def _make(cls, obj: Iterable[Any]) -> CodecOptions[_DocumentType]:
            if False:
                for i in range(10):
                    print('nop')
            ...

        def _asdict(self) -> dict[str, Any]:
            if False:
                i = 10
                return i + 15
            ...

        def _replace(self, **kwargs: Any) -> CodecOptions[_DocumentType]:
            if False:
                return 10
            ...
        _source: str
        _fields: Tuple[str]
else:

    class CodecOptions(_BaseCodecOptions):
        """Encapsulates options used encoding and / or decoding BSON."""

        def __init__(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            "Encapsulates options used encoding and / or decoding BSON.\n\n            The `document_class` option is used to define a custom type for use\n            decoding BSON documents. Access to the underlying raw BSON bytes for\n            a document is available using the :class:`~bson.raw_bson.RawBSONDocument`\n            type::\n\n              >>> from bson.raw_bson import RawBSONDocument\n              >>> from bson.codec_options import CodecOptions\n              >>> codec_options = CodecOptions(document_class=RawBSONDocument)\n              >>> coll = db.get_collection('test', codec_options=codec_options)\n              >>> doc = coll.find_one()\n              >>> doc.raw\n              '\\x16\\x00\\x00\\x00\\x07_id\\x00[0\\x165\\x91\\x10\\xea\\x14\\xe8\\xc5\\x8b\\x93\\x00'\n\n            The document class can be any type that inherits from\n            :class:`~collections.abc.MutableMapping`::\n\n              >>> class AttributeDict(dict):\n              ...     # A dict that supports attribute access.\n              ...     def __getattr__(self, key):\n              ...         return self[key]\n              ...     def __setattr__(self, key, value):\n              ...         self[key] = value\n              ...\n              >>> codec_options = CodecOptions(document_class=AttributeDict)\n              >>> coll = db.get_collection('test', codec_options=codec_options)\n              >>> doc = coll.find_one()\n              >>> doc._id\n              ObjectId('5b3016359110ea14e8c58b93')\n\n            See :doc:`/examples/datetimes` for examples using the `tz_aware` and\n            `tzinfo` options.\n\n            See :doc:`/examples/uuid` for examples using the `uuid_representation`\n            option.\n\n            :Parameters:\n              - `document_class`: BSON documents returned in queries will be decoded\n                to an instance of this class. Must be a subclass of\n                :class:`~collections.abc.MutableMapping`. Defaults to :class:`dict`.\n              - `tz_aware`: If ``True``, BSON datetimes will be decoded to timezone\n                aware instances of :class:`~datetime.datetime`. Otherwise they will be\n                naive. Defaults to ``False``.\n              - `uuid_representation`: The BSON representation to use when encoding\n                and decoding instances of :class:`~uuid.UUID`. Defaults to\n                :data:`~bson.binary.UuidRepresentation.UNSPECIFIED`. New\n                applications should consider setting this to\n                :data:`~bson.binary.UuidRepresentation.STANDARD` for cross language\n                compatibility. See :ref:`handling-uuid-data-example` for details.\n              - `unicode_decode_error_handler`: The error handler to apply when\n                a Unicode-related error occurs during BSON decoding that would\n                otherwise raise :exc:`UnicodeDecodeError`. Valid options include\n                'strict', 'replace', 'backslashreplace', 'surrogateescape', and\n                'ignore'. Defaults to 'strict'.\n              - `tzinfo`: A :class:`~datetime.tzinfo` subclass that specifies the\n                timezone to/from which :class:`~datetime.datetime` objects should be\n                encoded/decoded.\n              - `type_registry`: Instance of :class:`TypeRegistry` used to customize\n                encoding and decoding behavior.\n              - `datetime_conversion`: Specifies how UTC datetimes should be decoded\n                within BSON. Valid options include 'datetime_ms' to return as a\n                DatetimeMS, 'datetime' to return as a datetime.datetime and\n                raising a ValueError for out-of-range values, 'datetime_auto' to\n                return DatetimeMS objects when the underlying datetime is\n                out-of-range and 'datetime_clamp' to clamp to the minimum and\n                maximum possible datetimes. Defaults to 'datetime'.\n\n            .. versionchanged:: 4.0\n               The default for `uuid_representation` was changed from\n               :const:`~bson.binary.UuidRepresentation.PYTHON_LEGACY` to\n               :const:`~bson.binary.UuidRepresentation.UNSPECIFIED`.\n\n            .. versionadded:: 3.8\n               `type_registry` attribute.\n\n            .. warning:: Care must be taken when changing\n               `unicode_decode_error_handler` from its default value ('strict').\n               The 'replace' and 'ignore' modes should not be used when documents\n               retrieved from the server will be modified in the client application\n               and stored back to the server.\n            "
            super().__init__()

        def __new__(cls: Type[CodecOptions], document_class: Optional[Type[Mapping[str, Any]]]=None, tz_aware: bool=False, uuid_representation: Optional[int]=UuidRepresentation.UNSPECIFIED, unicode_decode_error_handler: str='strict', tzinfo: Optional[datetime.tzinfo]=None, type_registry: Optional[TypeRegistry]=None, datetime_conversion: Optional[DatetimeConversion]=DatetimeConversion.DATETIME) -> CodecOptions:
            if False:
                print('Hello World!')
            doc_class = document_class or dict
            is_mapping = False
            try:
                is_mapping = issubclass(doc_class, _MutableMapping)
            except TypeError:
                if hasattr(doc_class, '__origin__'):
                    is_mapping = issubclass(doc_class.__origin__, _MutableMapping)
            if not (is_mapping or _raw_document_class(doc_class)):
                raise TypeError('document_class must be dict, bson.son.SON, bson.raw_bson.RawBSONDocument, or a subclass of collections.abc.MutableMapping')
            if not isinstance(tz_aware, bool):
                raise TypeError(f'tz_aware must be True or False, was: tz_aware={tz_aware}')
            if uuid_representation not in ALL_UUID_REPRESENTATIONS:
                raise ValueError('uuid_representation must be a value from bson.binary.UuidRepresentation')
            if not isinstance(unicode_decode_error_handler, str):
                raise ValueError('unicode_decode_error_handler must be a string')
            if tzinfo is not None:
                if not isinstance(tzinfo, datetime.tzinfo):
                    raise TypeError('tzinfo must be an instance of datetime.tzinfo')
                if not tz_aware:
                    raise ValueError('cannot specify tzinfo without also setting tz_aware=True')
            type_registry = type_registry or TypeRegistry()
            if not isinstance(type_registry, TypeRegistry):
                raise TypeError('type_registry must be an instance of TypeRegistry')
            return tuple.__new__(cls, (doc_class, tz_aware, uuid_representation, unicode_decode_error_handler, tzinfo, type_registry, datetime_conversion))

        def _arguments_repr(self) -> str:
            if False:
                return 10
            'Representation of the arguments used to create this object.'
            document_class_repr = 'dict' if self.document_class is dict else repr(self.document_class)
            uuid_rep_repr = UUID_REPRESENTATION_NAMES.get(self.uuid_representation, self.uuid_representation)
            return 'document_class={}, tz_aware={!r}, uuid_representation={}, unicode_decode_error_handler={!r}, tzinfo={!r}, type_registry={!r}, datetime_conversion={!s}'.format(document_class_repr, self.tz_aware, uuid_rep_repr, self.unicode_decode_error_handler, self.tzinfo, self.type_registry, self.datetime_conversion)

        def _options_dict(self) -> dict[str, Any]:
            if False:
                while True:
                    i = 10
            'Dictionary of the arguments used to create this object.'
            return {'document_class': self.document_class, 'tz_aware': self.tz_aware, 'uuid_representation': self.uuid_representation, 'unicode_decode_error_handler': self.unicode_decode_error_handler, 'tzinfo': self.tzinfo, 'type_registry': self.type_registry, 'datetime_conversion': self.datetime_conversion}

        def __repr__(self) -> str:
            if False:
                print('Hello World!')
            return f'{self.__class__.__name__}({self._arguments_repr()})'

        def with_options(self, **kwargs: Any) -> CodecOptions:
            if False:
                for i in range(10):
                    print('nop')
            'Make a copy of this CodecOptions, overriding some options::\n\n                >>> from bson.codec_options import DEFAULT_CODEC_OPTIONS\n                >>> DEFAULT_CODEC_OPTIONS.tz_aware\n                False\n                >>> options = DEFAULT_CODEC_OPTIONS.with_options(tz_aware=True)\n                >>> options.tz_aware\n                True\n\n            .. versionadded:: 3.5\n            '
            opts = self._options_dict()
            opts.update(kwargs)
            return CodecOptions(**opts)
DEFAULT_CODEC_OPTIONS: CodecOptions[dict[str, Any]] = CodecOptions()

def _parse_codec_options(options: Any) -> CodecOptions[Any]:
    if False:
        print('Hello World!')
    'Parse BSON codec options.'
    kwargs = {}
    for k in set(options) & {'document_class', 'tz_aware', 'uuidrepresentation', 'unicode_decode_error_handler', 'tzinfo', 'type_registry', 'datetime_conversion'}:
        if k == 'uuidrepresentation':
            kwargs['uuid_representation'] = options[k]
        else:
            kwargs[k] = options[k]
    return CodecOptions(**kwargs)