"""Tools for representing raw BSON documents.

Inserting and Retrieving RawBSONDocuments
=========================================

Example: Moving a document between different databases/collections

.. doctest::

  >>> import bson
  >>> from pymongo import MongoClient
  >>> from bson.raw_bson import RawBSONDocument
  >>> client = MongoClient(document_class=RawBSONDocument)
  >>> client.drop_database("db")
  >>> client.drop_database("replica_db")
  >>> db = client.db
  >>> result = db.test.insert_many(
  ...     [{"_id": 1, "a": 1}, {"_id": 2, "b": 1}, {"_id": 3, "c": 1}, {"_id": 4, "d": 1}]
  ... )
  >>> replica_db = client.replica_db
  >>> for doc in db.test.find():
  ...     print(f"raw document: {doc.raw}")
  ...     print(f"decoded document: {bson.decode(doc.raw)}")
  ...     result = replica_db.test.insert_one(doc)
  ...
  raw document: b'...'
  decoded document: {'_id': 1, 'a': 1}
  raw document: b'...'
  decoded document: {'_id': 2, 'b': 1}
  raw document: b'...'
  decoded document: {'_id': 3, 'c': 1}
  raw document: b'...'
  decoded document: {'_id': 4, 'd': 1}

For use cases like moving documents across different databases or writing binary
blobs to disk, using raw BSON documents provides better speed and avoids the
overhead of decoding or encoding BSON.
"""
from __future__ import annotations
from typing import Any, ItemsView, Iterator, Mapping, MutableMapping, Optional
from bson import _get_object_size, _raw_to_dict
from bson.codec_options import _RAW_BSON_DOCUMENT_MARKER, CodecOptions
from bson.codec_options import DEFAULT_CODEC_OPTIONS as DEFAULT
from bson.son import SON

def _inflate_bson(bson_bytes: bytes, codec_options: CodecOptions[RawBSONDocument], raw_array: bool=False) -> MutableMapping[str, Any]:
    if False:
        i = 10
        return i + 15
    'Inflates the top level fields of a BSON document.\n\n    :Parameters:\n      - `bson_bytes`: the BSON bytes that compose this document\n      - `codec_options`: An instance of\n        :class:`~bson.codec_options.CodecOptions` whose ``document_class``\n        must be :class:`RawBSONDocument`.\n    '
    return _raw_to_dict(bson_bytes, 4, len(bson_bytes) - 1, codec_options, SON(), raw_array=raw_array)

class RawBSONDocument(Mapping[str, Any]):
    """Representation for a MongoDB document that provides access to the raw
    BSON bytes that compose it.

    Only when a field is accessed or modified within the document does
    RawBSONDocument decode its bytes.
    """
    __slots__ = ('__raw', '__inflated_doc', '__codec_options')
    _type_marker = _RAW_BSON_DOCUMENT_MARKER
    __codec_options: CodecOptions[RawBSONDocument]

    def __init__(self, bson_bytes: bytes, codec_options: Optional[CodecOptions[RawBSONDocument]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Create a new :class:`RawBSONDocument`\n\n        :class:`RawBSONDocument` is a representation of a BSON document that\n        provides access to the underlying raw BSON bytes. Only when a field is\n        accessed or modified within the document does RawBSONDocument decode\n        its bytes.\n\n        :class:`RawBSONDocument` implements the ``Mapping`` abstract base\n        class from the standard library so it can be used like a read-only\n        ``dict``::\n\n            >>> from bson import encode\n            >>> raw_doc = RawBSONDocument(encode({'_id': 'my_doc'}))\n            >>> raw_doc.raw\n            b'...'\n            >>> raw_doc['_id']\n            'my_doc'\n\n        :Parameters:\n          - `bson_bytes`: the BSON bytes that compose this document\n          - `codec_options` (optional): An instance of\n            :class:`~bson.codec_options.CodecOptions` whose ``document_class``\n            must be :class:`RawBSONDocument`. The default is\n            :attr:`DEFAULT_RAW_BSON_OPTIONS`.\n\n        .. versionchanged:: 3.8\n          :class:`RawBSONDocument` now validates that the ``bson_bytes``\n          passed in represent a single bson document.\n\n        .. versionchanged:: 3.5\n          If a :class:`~bson.codec_options.CodecOptions` is passed in, its\n          `document_class` must be :class:`RawBSONDocument`.\n        "
        self.__raw = bson_bytes
        self.__inflated_doc: Optional[Mapping[str, Any]] = None
        if codec_options is None:
            codec_options = DEFAULT_RAW_BSON_OPTIONS
        elif not issubclass(codec_options.document_class, RawBSONDocument):
            raise TypeError(f'RawBSONDocument cannot use CodecOptions with document class {codec_options.document_class}')
        self.__codec_options = codec_options
        _get_object_size(bson_bytes, 0, len(bson_bytes))

    @property
    def raw(self) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        'The raw BSON bytes composing this document.'
        return self.__raw

    def items(self) -> ItemsView[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Lazily decode and iterate elements in this document.'
        return self.__inflated.items()

    @property
    def __inflated(self) -> Mapping[str, Any]:
        if False:
            while True:
                i = 10
        if self.__inflated_doc is None:
            self.__inflated_doc = self._inflate_bson(self.__raw, self.__codec_options)
        return self.__inflated_doc

    @staticmethod
    def _inflate_bson(bson_bytes: bytes, codec_options: CodecOptions[RawBSONDocument]) -> Mapping[str, Any]:
        if False:
            i = 10
            return i + 15
        return _inflate_bson(bson_bytes, codec_options)

    def __getitem__(self, item: str) -> Any:
        if False:
            i = 10
            return i + 15
        return self.__inflated[item]

    def __iter__(self) -> Iterator[str]:
        if False:
            return 10
        return iter(self.__inflated)

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return len(self.__inflated)

    def __eq__(self, other: Any) -> bool:
        if False:
            i = 10
            return i + 15
        if isinstance(other, RawBSONDocument):
            return self.__raw == other.raw
        return NotImplemented

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'{self.__class__.__name__}({self.raw!r}, codec_options={self.__codec_options!r})'

class _RawArrayBSONDocument(RawBSONDocument):
    """A RawBSONDocument that only expands sub-documents and arrays when accessed."""

    @staticmethod
    def _inflate_bson(bson_bytes: bytes, codec_options: CodecOptions[RawBSONDocument]) -> Mapping[str, Any]:
        if False:
            i = 10
            return i + 15
        return _inflate_bson(bson_bytes, codec_options, raw_array=True)
DEFAULT_RAW_BSON_OPTIONS: CodecOptions[RawBSONDocument] = DEFAULT.with_options(document_class=RawBSONDocument)
_RAW_ARRAY_BSON_OPTIONS: CodecOptions[_RawArrayBSONDocument] = DEFAULT.with_options(document_class=_RawArrayBSONDocument)
'The default :class:`~bson.codec_options.CodecOptions` for\n:class:`RawBSONDocument`.\n'