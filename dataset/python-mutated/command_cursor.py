"""CommandCursor class to iterate over command results."""
from __future__ import annotations
from collections import deque
from typing import TYPE_CHECKING, Any, Generic, Iterator, Mapping, NoReturn, Optional, Sequence, Union
from bson import CodecOptions, _convert_raw_document_lists_to_streams
from pymongo.cursor import _CURSOR_CLOSED_ERRORS, _ConnectionManager
from pymongo.errors import ConnectionFailure, InvalidOperation, OperationFailure
from pymongo.message import _CursorAddress, _GetMore, _OpMsg, _OpReply, _RawBatchGetMore
from pymongo.response import PinnedResponse
from pymongo.typings import _Address, _DocumentOut, _DocumentType
if TYPE_CHECKING:
    from pymongo.client_session import ClientSession
    from pymongo.collection import Collection
    from pymongo.pool import Connection

class CommandCursor(Generic[_DocumentType]):
    """A cursor / iterator over command cursors."""
    _getmore_class = _GetMore

    def __init__(self, collection: Collection[_DocumentType], cursor_info: Mapping[str, Any], address: Optional[_Address], batch_size: int=0, max_await_time_ms: Optional[int]=None, session: Optional[ClientSession]=None, explicit_session: bool=False, comment: Any=None) -> None:
        if False:
            print('Hello World!')
        'Create a new command cursor.'
        self.__sock_mgr: Any = None
        self.__collection: Collection[_DocumentType] = collection
        self.__id = cursor_info['id']
        self.__data = deque(cursor_info['firstBatch'])
        self.__postbatchresumetoken: Optional[Mapping[str, Any]] = cursor_info.get('postBatchResumeToken')
        self.__address = address
        self.__batch_size = batch_size
        self.__max_await_time_ms = max_await_time_ms
        self.__session = session
        self.__explicit_session = explicit_session
        self.__killed = self.__id == 0
        self.__comment = comment
        if self.__killed:
            self.__end_session(True)
        if 'ns' in cursor_info:
            self.__ns = cursor_info['ns']
        else:
            self.__ns = collection.full_name
        self.batch_size(batch_size)
        if not isinstance(max_await_time_ms, int) and max_await_time_ms is not None:
            raise TypeError('max_await_time_ms must be an integer or None')

    def __del__(self) -> None:
        if False:
            return 10
        self.__die()

    def __die(self, synchronous: bool=False) -> None:
        if False:
            while True:
                i = 10
        'Closes this cursor.'
        already_killed = self.__killed
        self.__killed = True
        if self.__id and (not already_killed):
            cursor_id = self.__id
            assert self.__address is not None
            address = _CursorAddress(self.__address, self.__ns)
        else:
            cursor_id = 0
            address = None
        self.__collection.database.client._cleanup_cursor(synchronous, cursor_id, address, self.__sock_mgr, self.__session, self.__explicit_session)
        if not self.__explicit_session:
            self.__session = None
        self.__sock_mgr = None

    def __end_session(self, synchronous: bool) -> None:
        if False:
            i = 10
            return i + 15
        if self.__session and (not self.__explicit_session):
            self.__session._end_session(lock=synchronous)
            self.__session = None

    def close(self) -> None:
        if False:
            print('Hello World!')
        'Explicitly close / kill this cursor.'
        self.__die(True)

    def batch_size(self, batch_size: int) -> CommandCursor[_DocumentType]:
        if False:
            return 10
        "Limits the number of documents returned in one batch. Each batch\n        requires a round trip to the server. It can be adjusted to optimize\n        performance and limit data transfer.\n\n        .. note:: batch_size can not override MongoDB's internal limits on the\n           amount of data it will return to the client in a single batch (i.e\n           if you set batch size to 1,000,000,000, MongoDB will currently only\n           return 4-16MB of results per batch).\n\n        Raises :exc:`TypeError` if `batch_size` is not an integer.\n        Raises :exc:`ValueError` if `batch_size` is less than ``0``.\n\n        :Parameters:\n          - `batch_size`: The size of each batch of results requested.\n        "
        if not isinstance(batch_size, int):
            raise TypeError('batch_size must be an integer')
        if batch_size < 0:
            raise ValueError('batch_size must be >= 0')
        self.__batch_size = batch_size == 1 and 2 or batch_size
        return self

    def _has_next(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Returns `True` if the cursor has documents remaining from the\n        previous batch.\n        '
        return len(self.__data) > 0

    @property
    def _post_batch_resume_token(self) -> Optional[Mapping[str, Any]]:
        if False:
            print('Hello World!')
        'Retrieve the postBatchResumeToken from the response to a\n        changeStream aggregate or getMore.\n        '
        return self.__postbatchresumetoken

    def _maybe_pin_connection(self, conn: Connection) -> None:
        if False:
            i = 10
            return i + 15
        client = self.__collection.database.client
        if not client._should_pin_cursor(self.__session):
            return
        if not self.__sock_mgr:
            conn.pin_cursor()
            conn_mgr = _ConnectionManager(conn, False)
            if self.__id == 0:
                conn_mgr.close()
            else:
                self.__sock_mgr = conn_mgr

    def __send_message(self, operation: _GetMore) -> None:
        if False:
            print('Hello World!')
        'Send a getmore message and handle the response.'
        client = self.__collection.database.client
        try:
            response = client._run_operation(operation, self._unpack_response, address=self.__address)
        except OperationFailure as exc:
            if exc.code in _CURSOR_CLOSED_ERRORS:
                self.__killed = True
            if exc.timeout:
                self.__die(False)
            else:
                self.close()
            raise
        except ConnectionFailure:
            self.__killed = True
            self.close()
            raise
        except Exception:
            self.close()
            raise
        if isinstance(response, PinnedResponse):
            if not self.__sock_mgr:
                self.__sock_mgr = _ConnectionManager(response.conn, response.more_to_come)
        if response.from_command:
            cursor = response.docs[0]['cursor']
            documents = cursor['nextBatch']
            self.__postbatchresumetoken = cursor.get('postBatchResumeToken')
            self.__id = cursor['id']
        else:
            documents = response.docs
            assert isinstance(response.data, _OpReply)
            self.__id = response.data.cursor_id
        if self.__id == 0:
            self.close()
        self.__data = deque(documents)

    def _unpack_response(self, response: Union[_OpReply, _OpMsg], cursor_id: Optional[int], codec_options: CodecOptions[Mapping[str, Any]], user_fields: Optional[Mapping[str, Any]]=None, legacy_response: bool=False) -> Sequence[_DocumentOut]:
        if False:
            for i in range(10):
                print('nop')
        return response.unpack_response(cursor_id, codec_options, user_fields, legacy_response)

    def _refresh(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Refreshes the cursor with more data from the server.\n\n        Returns the length of self.__data after refresh. Will exit early if\n        self.__data is already non-empty. Raises OperationFailure when the\n        cursor cannot be refreshed due to an error on the query.\n        '
        if len(self.__data) or self.__killed:
            return len(self.__data)
        if self.__id:
            (dbname, collname) = self.__ns.split('.', 1)
            read_pref = self.__collection._read_preference_for(self.session)
            self.__send_message(self._getmore_class(dbname, collname, self.__batch_size, self.__id, self.__collection.codec_options, read_pref, self.__session, self.__collection.database.client, self.__max_await_time_ms, self.__sock_mgr, False, self.__comment))
        else:
            self.__die(True)
        return len(self.__data)

    @property
    def alive(self) -> bool:
        if False:
            while True:
                i = 10
        'Does this cursor have the potential to return more data?\n\n        Even if :attr:`alive` is ``True``, :meth:`next` can raise\n        :exc:`StopIteration`. Best to use a for loop::\n\n            for doc in collection.aggregate(pipeline):\n                print(doc)\n\n        .. note:: :attr:`alive` can be True while iterating a cursor from\n          a failed server. In this case :attr:`alive` will return False after\n          :meth:`next` fails to retrieve the next batch of results from the\n          server.\n        '
        return bool(len(self.__data) or not self.__killed)

    @property
    def cursor_id(self) -> int:
        if False:
            print('Hello World!')
        'Returns the id of the cursor.'
        return self.__id

    @property
    def address(self) -> Optional[_Address]:
        if False:
            for i in range(10):
                print('nop')
        'The (host, port) of the server used, or None.\n\n        .. versionadded:: 3.0\n        '
        return self.__address

    @property
    def session(self) -> Optional[ClientSession]:
        if False:
            for i in range(10):
                print('nop')
        "The cursor's :class:`~pymongo.client_session.ClientSession`, or None.\n\n        .. versionadded:: 3.6\n        "
        if self.__explicit_session:
            return self.__session
        return None

    def __iter__(self) -> Iterator[_DocumentType]:
        if False:
            while True:
                i = 10
        return self

    def next(self) -> _DocumentType:
        if False:
            i = 10
            return i + 15
        'Advance the cursor.'
        while self.alive:
            doc = self._try_next(True)
            if doc is not None:
                return doc
        raise StopIteration
    __next__ = next

    def _try_next(self, get_more_allowed: bool) -> Optional[_DocumentType]:
        if False:
            return 10
        'Advance the cursor blocking for at most one getMore command.'
        if not len(self.__data) and (not self.__killed) and get_more_allowed:
            self._refresh()
        if len(self.__data):
            return self.__data.popleft()
        else:
            return None

    def try_next(self) -> Optional[_DocumentType]:
        if False:
            while True:
                i = 10
        'Advance the cursor without blocking indefinitely.\n\n        This method returns the next document without waiting\n        indefinitely for data.\n\n        If no document is cached locally then this method runs a single\n        getMore command. If the getMore yields any documents, the next\n        document is returned, otherwise, if the getMore returns no documents\n        (because there is no additional data) then ``None`` is returned.\n\n        :Returns:\n          The next document or ``None`` when no document is available\n          after running a single getMore or when the cursor is closed.\n\n        .. versionadded:: 4.5\n        '
        return self._try_next(get_more_allowed=True)

    def __enter__(self) -> CommandCursor[_DocumentType]:
        if False:
            i = 10
            return i + 15
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if False:
            i = 10
            return i + 15
        self.close()

class RawBatchCommandCursor(CommandCursor, Generic[_DocumentType]):
    _getmore_class = _RawBatchGetMore

    def __init__(self, collection: Collection[_DocumentType], cursor_info: Mapping[str, Any], address: Optional[_Address], batch_size: int=0, max_await_time_ms: Optional[int]=None, session: Optional[ClientSession]=None, explicit_session: bool=False, comment: Any=None) -> None:
        if False:
            print('Hello World!')
        'Create a new cursor / iterator over raw batches of BSON data.\n\n        Should not be called directly by application developers -\n        see :meth:`~pymongo.collection.Collection.aggregate_raw_batches`\n        instead.\n\n        .. seealso:: The MongoDB documentation on `cursors <https://dochub.mongodb.org/core/cursors>`_.\n        '
        assert not cursor_info.get('firstBatch')
        super().__init__(collection, cursor_info, address, batch_size, max_await_time_ms, session, explicit_session, comment)

    def _unpack_response(self, response: Union[_OpReply, _OpMsg], cursor_id: Optional[int], codec_options: CodecOptions, user_fields: Optional[Mapping[str, Any]]=None, legacy_response: bool=False) -> list[Mapping[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        raw_response = response.raw_response(cursor_id, user_fields=user_fields)
        if not legacy_response:
            _convert_raw_document_lists_to_streams(raw_response[0])
        return raw_response

    def __getitem__(self, index: int) -> NoReturn:
        if False:
            while True:
                i = 10
        raise InvalidOperation('Cannot call __getitem__ on RawBatchCursor')