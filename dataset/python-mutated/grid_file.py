"""Tools for representing files stored in GridFS."""
from __future__ import annotations
import datetime
import io
import math
import os
from typing import Any, Iterable, Mapping, NoReturn, Optional
from bson.binary import Binary
from bson.int64 import Int64
from bson.objectid import ObjectId
from bson.son import SON
from gridfs.errors import CorruptGridFile, FileExists, NoFile
from pymongo import ASCENDING
from pymongo.client_session import ClientSession
from pymongo.collection import Collection
from pymongo.cursor import Cursor
from pymongo.errors import ConfigurationError, CursorNotFound, DuplicateKeyError, InvalidOperation, OperationFailure
from pymongo.read_preferences import ReadPreference
_SEEK_SET = os.SEEK_SET
_SEEK_CUR = os.SEEK_CUR
_SEEK_END = os.SEEK_END
EMPTY = b''
NEWLN = b'\n'
'Default chunk size, in bytes.'
DEFAULT_CHUNK_SIZE = 255 * 1024
_C_INDEX: SON[str, Any] = SON([('files_id', ASCENDING), ('n', ASCENDING)])
_F_INDEX: SON[str, Any] = SON([('filename', ASCENDING), ('uploadDate', ASCENDING)])

def _grid_in_property(field_name: str, docstring: str, read_only: Optional[bool]=False, closed_only: Optional[bool]=False) -> Any:
    if False:
        for i in range(10):
            print('nop')
    'Create a GridIn property.'

    def getter(self: Any) -> Any:
        if False:
            return 10
        if closed_only and (not self._closed):
            raise AttributeError('can only get %r on a closed file' % field_name)
        if field_name == 'length':
            return self._file.get(field_name, 0)
        return self._file.get(field_name, None)

    def setter(self: Any, value: Any) -> Any:
        if False:
            return 10
        if self._closed:
            self._coll.files.update_one({'_id': self._file['_id']}, {'$set': {field_name: value}})
        self._file[field_name] = value
    if read_only:
        docstring += '\n\nThis attribute is read-only.'
    elif closed_only:
        docstring = '{}\n\n{}'.format(docstring, 'This attribute is read-only and can only be read after :meth:`close` has been called.')
    if not read_only and (not closed_only):
        return property(getter, setter, doc=docstring)
    return property(getter, doc=docstring)

def _grid_out_property(field_name: str, docstring: str) -> Any:
    if False:
        while True:
            i = 10
    'Create a GridOut property.'

    def getter(self: Any) -> Any:
        if False:
            while True:
                i = 10
        self._ensure_file()
        if field_name == 'length':
            return self._file.get(field_name, 0)
        return self._file.get(field_name, None)
    docstring += '\n\nThis attribute is read-only.'
    return property(getter, doc=docstring)

def _clear_entity_type_registry(entity: Any, **kwargs: Any) -> Any:
    if False:
        i = 10
        return i + 15
    "Clear the given database/collection object's type registry."
    codecopts = entity.codec_options.with_options(type_registry=None)
    return entity.with_options(codec_options=codecopts, **kwargs)

def _disallow_transactions(session: Optional[ClientSession]) -> None:
    if False:
        for i in range(10):
            print('nop')
    if session and session.in_transaction:
        raise InvalidOperation('GridFS does not support multi-document transactions')

class GridIn:
    """Class to write data to GridFS."""

    def __init__(self, root_collection: Collection, session: Optional[ClientSession]=None, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Write a file to GridFS\n\n        Application developers should generally not need to\n        instantiate this class directly - instead see the methods\n        provided by :class:`~gridfs.GridFS`.\n\n        Raises :class:`TypeError` if `root_collection` is not an\n        instance of :class:`~pymongo.collection.Collection`.\n\n        Any of the file level options specified in the `GridFS Spec\n        <http://dochub.mongodb.org/core/gridfsspec>`_ may be passed as\n        keyword arguments. Any additional keyword arguments will be\n        set as additional fields on the file document. Valid keyword\n        arguments include:\n\n          - ``"_id"``: unique ID for this file (default:\n            :class:`~bson.objectid.ObjectId`) - this ``"_id"`` must\n            not have already been used for another file\n\n          - ``"filename"``: human name for the file\n\n          - ``"contentType"`` or ``"content_type"``: valid mime-type\n            for the file\n\n          - ``"chunkSize"`` or ``"chunk_size"``: size of each of the\n            chunks, in bytes (default: 255 kb)\n\n          - ``"encoding"``: encoding used for this file. Any :class:`str`\n            that is written to the file will be converted to :class:`bytes`.\n\n        :Parameters:\n          - `root_collection`: root collection to write to\n          - `session` (optional): a\n            :class:`~pymongo.client_session.ClientSession` to use for all\n            commands\n          - `**kwargs: Any` (optional): file level options (see above)\n\n        .. versionchanged:: 4.0\n           Removed the `disable_md5` parameter. See\n           :ref:`removed-gridfs-checksum` for details.\n\n        .. versionchanged:: 3.7\n           Added the `disable_md5` parameter.\n\n        .. versionchanged:: 3.6\n           Added ``session`` parameter.\n\n        .. versionchanged:: 3.0\n           `root_collection` must use an acknowledged\n           :attr:`~pymongo.collection.Collection.write_concern`\n        '
        if not isinstance(root_collection, Collection):
            raise TypeError('root_collection must be an instance of Collection')
        if not root_collection.write_concern.acknowledged:
            raise ConfigurationError('root_collection must use acknowledged write_concern')
        _disallow_transactions(session)
        if 'content_type' in kwargs:
            kwargs['contentType'] = kwargs.pop('content_type')
        if 'chunk_size' in kwargs:
            kwargs['chunkSize'] = kwargs.pop('chunk_size')
        coll = _clear_entity_type_registry(root_collection, read_preference=ReadPreference.PRIMARY)
        kwargs['_id'] = kwargs.get('_id', ObjectId())
        kwargs['chunkSize'] = kwargs.get('chunkSize', DEFAULT_CHUNK_SIZE)
        object.__setattr__(self, '_session', session)
        object.__setattr__(self, '_coll', coll)
        object.__setattr__(self, '_chunks', coll.chunks)
        object.__setattr__(self, '_file', kwargs)
        object.__setattr__(self, '_buffer', io.BytesIO())
        object.__setattr__(self, '_position', 0)
        object.__setattr__(self, '_chunk_number', 0)
        object.__setattr__(self, '_closed', False)
        object.__setattr__(self, '_ensured_index', False)

    def __create_index(self, collection: Collection, index_key: Any, unique: bool) -> None:
        if False:
            return 10
        doc = collection.find_one(projection={'_id': 1}, session=self._session)
        if doc is None:
            try:
                index_keys = [index_spec['key'] for index_spec in collection.list_indexes(session=self._session)]
            except OperationFailure:
                index_keys = []
            if index_key not in index_keys:
                collection.create_index(index_key.items(), unique=unique, session=self._session)

    def __ensure_indexes(self) -> None:
        if False:
            while True:
                i = 10
        if not object.__getattribute__(self, '_ensured_index'):
            _disallow_transactions(self._session)
            self.__create_index(self._coll.files, _F_INDEX, False)
            self.__create_index(self._coll.chunks, _C_INDEX, True)
            object.__setattr__(self, '_ensured_index', True)

    def abort(self) -> None:
        if False:
            i = 10
            return i + 15
        'Remove all chunks/files that may have been uploaded and close.'
        self._coll.chunks.delete_many({'files_id': self._file['_id']}, session=self._session)
        self._coll.files.delete_one({'_id': self._file['_id']}, session=self._session)
        object.__setattr__(self, '_closed', True)

    @property
    def closed(self) -> bool:
        if False:
            print('Hello World!')
        'Is this file closed?'
        return self._closed
    _id: Any = _grid_in_property('_id', "The ``'_id'`` value for this file.", read_only=True)
    filename: Optional[str] = _grid_in_property('filename', 'Name of this file.')
    name: Optional[str] = _grid_in_property('filename', 'Alias for `filename`.')
    content_type: Optional[str] = _grid_in_property('contentType', 'DEPRECATED, will be removed in PyMongo 5.0. Mime-type for this file.')
    length: int = _grid_in_property('length', 'Length (in bytes) of this file.', closed_only=True)
    chunk_size: int = _grid_in_property('chunkSize', 'Chunk size for this file.', read_only=True)
    upload_date: datetime.datetime = _grid_in_property('uploadDate', 'Date that this file was uploaded.', closed_only=True)
    md5: Optional[str] = _grid_in_property('md5', 'DEPRECATED, will be removed in PyMongo 5.0. MD5 of the contents of this file if an md5 sum was created.', closed_only=True)
    _buffer: io.BytesIO
    _closed: bool

    def __getattr__(self, name: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        if name in self._file:
            return self._file[name]
        raise AttributeError("GridIn object has no attribute '%s'" % name)

    def __setattr__(self, name: str, value: Any) -> None:
        if False:
            return 10
        if name in self.__dict__ or name in self.__class__.__dict__:
            object.__setattr__(self, name, value)
        else:
            self._file[name] = value
            if self._closed:
                self._coll.files.update_one({'_id': self._file['_id']}, {'$set': {name: value}})

    def __flush_data(self, data: Any) -> None:
        if False:
            return 10
        'Flush `data` to a chunk.'
        self.__ensure_indexes()
        if not data:
            return
        assert len(data) <= self.chunk_size
        chunk = {'files_id': self._file['_id'], 'n': self._chunk_number, 'data': Binary(data)}
        try:
            self._chunks.insert_one(chunk, session=self._session)
        except DuplicateKeyError:
            self._raise_file_exists(self._file['_id'])
        self._chunk_number += 1
        self._position += len(data)

    def __flush_buffer(self) -> None:
        if False:
            print('Hello World!')
        'Flush the buffer contents out to a chunk.'
        self.__flush_data(self._buffer.getvalue())
        self._buffer.close()
        self._buffer = io.BytesIO()

    def __flush(self) -> Any:
        if False:
            print('Hello World!')
        'Flush the file to the database.'
        try:
            self.__flush_buffer()
            self._file['length'] = Int64(self._position)
            self._file['uploadDate'] = datetime.datetime.now(tz=datetime.timezone.utc)
            return self._coll.files.insert_one(self._file, session=self._session)
        except DuplicateKeyError:
            self._raise_file_exists(self._id)

    def _raise_file_exists(self, file_id: Any) -> NoReturn:
        if False:
            print('Hello World!')
        'Raise a FileExists exception for the given file_id.'
        raise FileExists('file with _id %r already exists' % file_id)

    def close(self) -> None:
        if False:
            while True:
                i = 10
        'Flush the file and close it.\n\n        A closed file cannot be written any more. Calling\n        :meth:`close` more than once is allowed.\n        '
        if not self._closed:
            self.__flush()
            object.__setattr__(self, '_closed', True)

    def read(self, size: int=-1) -> NoReturn:
        if False:
            for i in range(10):
                print('nop')
        raise io.UnsupportedOperation('read')

    def readable(self) -> bool:
        if False:
            print('Hello World!')
        return False

    def seekable(self) -> bool:
        if False:
            i = 10
            return i + 15
        return False

    def write(self, data: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Write data to the file. There is no return value.\n\n        `data` can be either a string of bytes or a file-like object\n        (implementing :meth:`read`). If the file has an\n        :attr:`encoding` attribute, `data` can also be a\n        :class:`str` instance, which will be encoded as\n        :attr:`encoding` before being written.\n\n        Due to buffering, the data may not actually be written to the\n        database until the :meth:`close` method is called. Raises\n        :class:`ValueError` if this file is already closed. Raises\n        :class:`TypeError` if `data` is not an instance of\n        :class:`bytes`, a file-like object, or an instance of :class:`str`.\n        Unicode data is only allowed if the file has an :attr:`encoding`\n        attribute.\n\n        :Parameters:\n          - `data`: string of bytes or file-like object to be written\n            to the file\n        '
        if self._closed:
            raise ValueError('cannot write to a closed file')
        try:
            read = data.read
        except AttributeError:
            if not isinstance(data, (str, bytes)):
                raise TypeError('can only write strings or file-like objects') from None
            if isinstance(data, str):
                try:
                    data = data.encode(self.encoding)
                except AttributeError:
                    raise TypeError('must specify an encoding for file in order to write str') from None
            read = io.BytesIO(data).read
        if self._buffer.tell() > 0:
            space = self.chunk_size - self._buffer.tell()
            if space:
                try:
                    to_write = read(space)
                except BaseException:
                    self.abort()
                    raise
                self._buffer.write(to_write)
                if len(to_write) < space:
                    return
            self.__flush_buffer()
        to_write = read(self.chunk_size)
        while to_write and len(to_write) == self.chunk_size:
            self.__flush_data(to_write)
            to_write = read(self.chunk_size)
        self._buffer.write(to_write)

    def writelines(self, sequence: Iterable[Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Write a sequence of strings to the file.\n\n        Does not add separators.\n        '
        for line in sequence:
            self.write(line)

    def writeable(self) -> bool:
        if False:
            return 10
        return True

    def __enter__(self) -> GridIn:
        if False:
            while True:
                i = 10
        'Support for the context manager protocol.'
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        if False:
            return 10
        'Support for the context manager protocol.\n\n        Close the file if no exceptions occur and allow exceptions to propagate.\n        '
        if exc_type is None:
            self.close()
        else:
            object.__setattr__(self, '_closed', True)
        return False

class GridOut(io.IOBase):
    """Class to read data out of GridFS."""

    def __init__(self, root_collection: Collection, file_id: Optional[int]=None, file_document: Optional[Any]=None, session: Optional[ClientSession]=None) -> None:
        if False:
            return 10
        'Read a file from GridFS\n\n        Application developers should generally not need to\n        instantiate this class directly - instead see the methods\n        provided by :class:`~gridfs.GridFS`.\n\n        Either `file_id` or `file_document` must be specified,\n        `file_document` will be given priority if present. Raises\n        :class:`TypeError` if `root_collection` is not an instance of\n        :class:`~pymongo.collection.Collection`.\n\n        :Parameters:\n          - `root_collection`: root collection to read from\n          - `file_id` (optional): value of ``"_id"`` for the file to read\n          - `file_document` (optional): file document from\n            `root_collection.files`\n          - `session` (optional): a\n            :class:`~pymongo.client_session.ClientSession` to use for all\n            commands\n\n        .. versionchanged:: 3.8\n           For better performance and to better follow the GridFS spec,\n           :class:`GridOut` now uses a single cursor to read all the chunks in\n           the file.\n\n        .. versionchanged:: 3.6\n           Added ``session`` parameter.\n\n        .. versionchanged:: 3.0\n           Creating a GridOut does not immediately retrieve the file metadata\n           from the server. Metadata is fetched when first needed.\n        '
        if not isinstance(root_collection, Collection):
            raise TypeError('root_collection must be an instance of Collection')
        _disallow_transactions(session)
        root_collection = _clear_entity_type_registry(root_collection)
        super().__init__()
        self.__chunks = root_collection.chunks
        self.__files = root_collection.files
        self.__file_id = file_id
        self.__buffer = EMPTY
        self.__buffer_pos = 0
        self.__chunk_iter = None
        self.__position = 0
        self._file = file_document
        self._session = session
    _id: Any = _grid_out_property('_id', "The ``'_id'`` value for this file.")
    filename: str = _grid_out_property('filename', 'Name of this file.')
    name: str = _grid_out_property('filename', 'Alias for `filename`.')
    content_type: Optional[str] = _grid_out_property('contentType', 'DEPRECATED, will be removed in PyMongo 5.0. Mime-type for this file.')
    length: int = _grid_out_property('length', 'Length (in bytes) of this file.')
    chunk_size: int = _grid_out_property('chunkSize', 'Chunk size for this file.')
    upload_date: datetime.datetime = _grid_out_property('uploadDate', 'Date that this file was first uploaded.')
    aliases: Optional[list[str]] = _grid_out_property('aliases', 'DEPRECATED, will be removed in PyMongo 5.0. List of aliases for this file.')
    metadata: Optional[Mapping[str, Any]] = _grid_out_property('metadata', 'Metadata attached to this file.')
    md5: Optional[str] = _grid_out_property('md5', 'DEPRECATED, will be removed in PyMongo 5.0. MD5 of the contents of this file if an md5 sum was created.')
    _file: Any
    __chunk_iter: Any

    def _ensure_file(self) -> None:
        if False:
            print('Hello World!')
        if not self._file:
            _disallow_transactions(self._session)
            self._file = self.__files.find_one({'_id': self.__file_id}, session=self._session)
            if not self._file:
                raise NoFile(f'no file in gridfs collection {self.__files!r} with _id {self.__file_id!r}')

    def __getattr__(self, name: str) -> Any:
        if False:
            print('Hello World!')
        self._ensure_file()
        if name in self._file:
            return self._file[name]
        raise AttributeError("GridOut object has no attribute '%s'" % name)

    def readable(self) -> bool:
        if False:
            print('Hello World!')
        return True

    def readchunk(self) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        'Reads a chunk at a time. If the current position is within a\n        chunk the remainder of the chunk is returned.\n        '
        received = len(self.__buffer) - self.__buffer_pos
        chunk_data = EMPTY
        chunk_size = int(self.chunk_size)
        if received > 0:
            chunk_data = self.__buffer[self.__buffer_pos:]
        elif self.__position < int(self.length):
            chunk_number = int((received + self.__position) / chunk_size)
            if self.__chunk_iter is None:
                self.__chunk_iter = _GridOutChunkIterator(self, self.__chunks, self._session, chunk_number)
            chunk = self.__chunk_iter.next()
            chunk_data = chunk['data'][self.__position % chunk_size:]
            if not chunk_data:
                raise CorruptGridFile('truncated chunk')
        self.__position += len(chunk_data)
        self.__buffer = EMPTY
        self.__buffer_pos = 0
        return chunk_data

    def _read_size_or_line(self, size: int=-1, line: bool=False) -> bytes:
        if False:
            print('Hello World!')
        'Internal read() and readline() helper.'
        self._ensure_file()
        remainder = int(self.length) - self.__position
        if size < 0 or size > remainder:
            size = remainder
        if size == 0:
            return EMPTY
        received = 0
        data = []
        while received < size:
            needed = size - received
            if self.__buffer:
                buf = self.__buffer
                chunk_start = self.__buffer_pos
                chunk_data = memoryview(buf)[self.__buffer_pos:]
                self.__buffer = EMPTY
                self.__buffer_pos = 0
                self.__position += len(chunk_data)
            else:
                buf = self.readchunk()
                chunk_start = 0
                chunk_data = memoryview(buf)
            if line:
                pos = buf.find(NEWLN, chunk_start, chunk_start + needed) - chunk_start
                if pos >= 0:
                    size = received + pos + 1
                    needed = pos + 1
            if len(chunk_data) > needed:
                data.append(chunk_data[:needed])
                self.__buffer = buf
                self.__buffer_pos = chunk_start + needed
                self.__position -= len(self.__buffer) - self.__buffer_pos
            else:
                data.append(chunk_data)
            received += len(chunk_data)
        if size == remainder and self.__chunk_iter:
            try:
                self.__chunk_iter.next()
            except StopIteration:
                pass
        return b''.join(data)

    def read(self, size: int=-1) -> bytes:
        if False:
            i = 10
            return i + 15
        "Read at most `size` bytes from the file (less if there\n        isn't enough data).\n\n        The bytes are returned as an instance of :class:`bytes`\n        If `size` is negative or omitted all data is read.\n\n        :Parameters:\n          - `size` (optional): the number of bytes to read\n\n        .. versionchanged:: 3.8\n           This method now only checks for extra chunks after reading the\n           entire file. Previously, this method would check for extra chunks\n           on every call.\n        "
        return self._read_size_or_line(size=size)

    def readline(self, size: int=-1) -> bytes:
        if False:
            while True:
                i = 10
        'Read one line or up to `size` bytes from the file.\n\n        :Parameters:\n         - `size` (optional): the maximum number of bytes to read\n        '
        return self._read_size_or_line(size=size, line=True)

    def tell(self) -> int:
        if False:
            print('Hello World!')
        'Return the current position of this file.'
        return self.__position

    def seek(self, pos: int, whence: int=_SEEK_SET) -> int:
        if False:
            for i in range(10):
                print('nop')
        "Set the current position of this file.\n\n        :Parameters:\n         - `pos`: the position (or offset if using relative\n           positioning) to seek to\n         - `whence` (optional): where to seek\n           from. :attr:`os.SEEK_SET` (``0``) for absolute file\n           positioning, :attr:`os.SEEK_CUR` (``1``) to seek relative\n           to the current position, :attr:`os.SEEK_END` (``2``) to\n           seek relative to the file's end.\n\n        .. versionchanged:: 4.1\n           The method now returns the new position in the file, to\n           conform to the behavior of :meth:`io.IOBase.seek`.\n        "
        if whence == _SEEK_SET:
            new_pos = pos
        elif whence == _SEEK_CUR:
            new_pos = self.__position + pos
        elif whence == _SEEK_END:
            new_pos = int(self.length) + pos
        else:
            raise OSError(22, 'Invalid value for `whence`')
        if new_pos < 0:
            raise OSError(22, 'Invalid value for `pos` - must be positive')
        if new_pos == self.__position:
            return new_pos
        self.__position = new_pos
        self.__buffer = EMPTY
        self.__buffer_pos = 0
        if self.__chunk_iter:
            self.__chunk_iter.close()
            self.__chunk_iter = None
        return new_pos

    def seekable(self) -> bool:
        if False:
            print('Hello World!')
        return True

    def __iter__(self) -> GridOut:
        if False:
            return 10
        "Return an iterator over all of this file's data.\n\n        The iterator will return lines (delimited by ``b'\\n'``) of\n        :class:`bytes`. This can be useful when serving files\n        using a webserver that handles such an iterator efficiently.\n\n        .. versionchanged:: 3.8\n           The iterator now raises :class:`CorruptGridFile` when encountering\n           any truncated, missing, or extra chunk in a file. The previous\n           behavior was to only raise :class:`CorruptGridFile` on a missing\n           chunk.\n\n        .. versionchanged:: 4.0\n           The iterator now iterates over *lines* in the file, instead\n           of chunks, to conform to the base class :py:class:`io.IOBase`.\n           Use :meth:`GridOut.readchunk` to read chunk by chunk instead\n           of line by line.\n        "
        return self

    def close(self) -> None:
        if False:
            while True:
                i = 10
        'Make GridOut more generically file-like.'
        if self.__chunk_iter:
            self.__chunk_iter.close()
            self.__chunk_iter = None
        super().close()

    def write(self, value: Any) -> NoReturn:
        if False:
            for i in range(10):
                print('nop')
        raise io.UnsupportedOperation('write')

    def writelines(self, lines: Any) -> NoReturn:
        if False:
            i = 10
            return i + 15
        raise io.UnsupportedOperation('writelines')

    def writable(self) -> bool:
        if False:
            return 10
        return False

    def __enter__(self) -> GridOut:
        if False:
            for i in range(10):
                print('nop')
        'Makes it possible to use :class:`GridOut` files\n        with the context manager protocol.\n        '
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        if False:
            print('Hello World!')
        'Makes it possible to use :class:`GridOut` files\n        with the context manager protocol.\n        '
        self.close()
        return False

    def fileno(self) -> NoReturn:
        if False:
            while True:
                i = 10
        raise io.UnsupportedOperation('fileno')

    def flush(self) -> None:
        if False:
            while True:
                i = 10
        pass

    def isatty(self) -> bool:
        if False:
            print('Hello World!')
        return False

    def truncate(self, size: Optional[int]=None) -> NoReturn:
        if False:
            i = 10
            return i + 15
        raise io.UnsupportedOperation('truncate')

    def __del__(self) -> None:
        if False:
            return 10
        pass

class _GridOutChunkIterator:
    """Iterates over a file's chunks using a single cursor.

    Raises CorruptGridFile when encountering any truncated, missing, or extra
    chunk in a file.
    """

    def __init__(self, grid_out: GridOut, chunks: Collection, session: Optional[ClientSession], next_chunk: Any) -> None:
        if False:
            i = 10
            return i + 15
        self._id = grid_out._id
        self._chunk_size = int(grid_out.chunk_size)
        self._length = int(grid_out.length)
        self._chunks = chunks
        self._session = session
        self._next_chunk = next_chunk
        self._num_chunks = math.ceil(float(self._length) / self._chunk_size)
        self._cursor = None
    _cursor: Optional[Cursor]

    def expected_chunk_length(self, chunk_n: int) -> int:
        if False:
            for i in range(10):
                print('nop')
        if chunk_n < self._num_chunks - 1:
            return self._chunk_size
        return self._length - self._chunk_size * (self._num_chunks - 1)

    def __iter__(self) -> _GridOutChunkIterator:
        if False:
            for i in range(10):
                print('nop')
        return self

    def _create_cursor(self) -> None:
        if False:
            return 10
        filter = {'files_id': self._id}
        if self._next_chunk > 0:
            filter['n'] = {'$gte': self._next_chunk}
        _disallow_transactions(self._session)
        self._cursor = self._chunks.find(filter, sort=[('n', 1)], session=self._session)

    def _next_with_retry(self) -> Mapping[str, Any]:
        if False:
            i = 10
            return i + 15
        "Return the next chunk and retry once on CursorNotFound.\n\n        We retry on CursorNotFound to maintain backwards compatibility in\n        cases where two calls to read occur more than 10 minutes apart (the\n        server's default cursor timeout).\n        "
        if self._cursor is None:
            self._create_cursor()
            assert self._cursor is not None
        try:
            return self._cursor.next()
        except CursorNotFound:
            self._cursor.close()
            self._create_cursor()
            return self._cursor.next()

    def next(self) -> Mapping[str, Any]:
        if False:
            return 10
        try:
            chunk = self._next_with_retry()
        except StopIteration:
            if self._next_chunk >= self._num_chunks:
                raise
            raise CorruptGridFile('no chunk #%d' % self._next_chunk) from None
        if chunk['n'] != self._next_chunk:
            self.close()
            raise CorruptGridFile('Missing chunk: expected chunk #%d but found chunk with n=%d' % (self._next_chunk, chunk['n']))
        if chunk['n'] >= self._num_chunks:
            if len(chunk['data']):
                self.close()
                raise CorruptGridFile('Extra chunk found: expected %d chunks but found chunk with n=%d' % (self._num_chunks, chunk['n']))
        expected_length = self.expected_chunk_length(chunk['n'])
        if len(chunk['data']) != expected_length:
            self.close()
            raise CorruptGridFile('truncated chunk #%d: expected chunk length to be %d but found chunk with length %d' % (chunk['n'], expected_length, len(chunk['data'])))
        self._next_chunk += 1
        return chunk
    __next__ = next

    def close(self) -> None:
        if False:
            print('Hello World!')
        if self._cursor:
            self._cursor.close()
            self._cursor = None

class GridOutIterator:

    def __init__(self, grid_out: GridOut, chunks: Collection, session: ClientSession):
        if False:
            while True:
                i = 10
        self.__chunk_iter = _GridOutChunkIterator(grid_out, chunks, session, 0)

    def __iter__(self) -> GridOutIterator:
        if False:
            i = 10
            return i + 15
        return self

    def next(self) -> bytes:
        if False:
            while True:
                i = 10
        chunk = self.__chunk_iter.next()
        return bytes(chunk['data'])
    __next__ = next

class GridOutCursor(Cursor):
    """A cursor / iterator for returning GridOut objects as the result
    of an arbitrary query against the GridFS files collection.
    """

    def __init__(self, collection: Collection, filter: Optional[Mapping[str, Any]]=None, skip: int=0, limit: int=0, no_cursor_timeout: bool=False, sort: Optional[Any]=None, batch_size: int=0, session: Optional[ClientSession]=None) -> None:
        if False:
            while True:
                i = 10
        'Create a new cursor, similar to the normal\n        :class:`~pymongo.cursor.Cursor`.\n\n        Should not be called directly by application developers - see\n        the :class:`~gridfs.GridFS` method :meth:`~gridfs.GridFS.find` instead.\n\n        .. versionadded 2.7\n\n        .. seealso:: The MongoDB documentation on `cursors <https://dochub.mongodb.org/core/cursors>`_.\n        '
        _disallow_transactions(session)
        collection = _clear_entity_type_registry(collection)
        self.__root_collection = collection
        super().__init__(collection.files, filter, skip=skip, limit=limit, no_cursor_timeout=no_cursor_timeout, sort=sort, batch_size=batch_size, session=session)

    def next(self) -> GridOut:
        if False:
            i = 10
            return i + 15
        'Get next GridOut object from cursor.'
        _disallow_transactions(self.session)
        next_file = super().next()
        return GridOut(self.__root_collection, file_document=next_file, session=self.session)
    __next__ = next

    def add_option(self, *args: Any, **kwargs: Any) -> NoReturn:
        if False:
            while True:
                i = 10
        raise NotImplementedError('Method does not exist for GridOutCursor')

    def remove_option(self, *args: Any, **kwargs: Any) -> NoReturn:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Method does not exist for GridOutCursor')

    def _clone_base(self, session: Optional[ClientSession]) -> GridOutCursor:
        if False:
            print('Hello World!')
        'Creates an empty GridOutCursor for information to be copied into.'
        return GridOutCursor(self.__root_collection, session=session)