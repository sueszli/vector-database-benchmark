"""Watch changes on a collection, a database, or the entire cluster."""
from __future__ import annotations
import copy
from typing import TYPE_CHECKING, Any, Generic, Mapping, Optional, Type, Union
from bson import CodecOptions, _bson_to_dict
from bson.raw_bson import RawBSONDocument
from bson.timestamp import Timestamp
from pymongo import _csot, common
from pymongo.aggregation import _AggregationCommand, _CollectionAggregationCommand, _DatabaseAggregationCommand
from pymongo.collation import validate_collation_or_none
from pymongo.command_cursor import CommandCursor
from pymongo.errors import ConnectionFailure, CursorNotFound, InvalidOperation, OperationFailure, PyMongoError
from pymongo.typings import _CollationIn, _DocumentType, _Pipeline
_RESUMABLE_GETMORE_ERRORS = frozenset([6, 7, 89, 91, 189, 262, 9001, 10107, 11600, 11602, 13435, 13436, 63, 150, 13388, 234, 133])
if TYPE_CHECKING:
    from pymongo.client_session import ClientSession
    from pymongo.collection import Collection
    from pymongo.database import Database
    from pymongo.mongo_client import MongoClient
    from pymongo.pool import Connection

def _resumable(exc: PyMongoError) -> bool:
    if False:
        i = 10
        return i + 15
    'Return True if given a resumable change stream error.'
    if isinstance(exc, (ConnectionFailure, CursorNotFound)):
        return True
    if isinstance(exc, OperationFailure):
        if exc._max_wire_version is None:
            return False
        return exc._max_wire_version >= 9 and exc.has_error_label('ResumableChangeStreamError') or (exc._max_wire_version < 9 and exc.code in _RESUMABLE_GETMORE_ERRORS)
    return False

class ChangeStream(Generic[_DocumentType]):
    """The internal abstract base class for change stream cursors.

    Should not be called directly by application developers. Use
    :meth:`pymongo.collection.Collection.watch`,
    :meth:`pymongo.database.Database.watch`, or
    :meth:`pymongo.mongo_client.MongoClient.watch` instead.

    .. versionadded:: 3.6
    .. seealso:: The MongoDB documentation on `changeStreams <https://mongodb.com/docs/manual/changeStreams/>`_.
    """

    def __init__(self, target: Union[MongoClient[_DocumentType], Database[_DocumentType], Collection[_DocumentType]], pipeline: Optional[_Pipeline], full_document: Optional[str], resume_after: Optional[Mapping[str, Any]], max_await_time_ms: Optional[int], batch_size: Optional[int], collation: Optional[_CollationIn], start_at_operation_time: Optional[Timestamp], session: Optional[ClientSession], start_after: Optional[Mapping[str, Any]], comment: Optional[Any]=None, full_document_before_change: Optional[str]=None, show_expanded_events: Optional[bool]=None) -> None:
        if False:
            i = 10
            return i + 15
        if pipeline is None:
            pipeline = []
        pipeline = common.validate_list('pipeline', pipeline)
        common.validate_string_or_none('full_document', full_document)
        validate_collation_or_none(collation)
        common.validate_non_negative_integer_or_none('batchSize', batch_size)
        self._decode_custom = False
        self._orig_codec_options: CodecOptions[_DocumentType] = target.codec_options
        if target.codec_options.type_registry._decoder_map:
            self._decode_custom = True
            self._target = target.with_options(codec_options=target.codec_options.with_options(document_class=RawBSONDocument))
        else:
            self._target = target
        self._pipeline = copy.deepcopy(pipeline)
        self._full_document = full_document
        self._full_document_before_change = full_document_before_change
        self._uses_start_after = start_after is not None
        self._uses_resume_after = resume_after is not None
        self._resume_token = copy.deepcopy(start_after or resume_after)
        self._max_await_time_ms = max_await_time_ms
        self._batch_size = batch_size
        self._collation = collation
        self._start_at_operation_time = start_at_operation_time
        self._session = session
        self._comment = comment
        self._closed = False
        self._timeout = self._target._timeout
        self._show_expanded_events = show_expanded_events
        self._cursor = self._create_cursor()

    @property
    def _aggregation_command_class(self) -> Type[_AggregationCommand]:
        if False:
            return 10
        'The aggregation command class to be used.'
        raise NotImplementedError

    @property
    def _client(self) -> MongoClient:
        if False:
            return 10
        'The client against which the aggregation commands for\n        this ChangeStream will be run.\n        '
        raise NotImplementedError

    def _change_stream_options(self) -> dict[str, Any]:
        if False:
            i = 10
            return i + 15
        'Return the options dict for the $changeStream pipeline stage.'
        options: dict[str, Any] = {}
        if self._full_document is not None:
            options['fullDocument'] = self._full_document
        if self._full_document_before_change is not None:
            options['fullDocumentBeforeChange'] = self._full_document_before_change
        resume_token = self.resume_token
        if resume_token is not None:
            if self._uses_start_after:
                options['startAfter'] = resume_token
            else:
                options['resumeAfter'] = resume_token
        if self._start_at_operation_time is not None:
            options['startAtOperationTime'] = self._start_at_operation_time
        if self._show_expanded_events:
            options['showExpandedEvents'] = self._show_expanded_events
        return options

    def _command_options(self) -> dict[str, Any]:
        if False:
            return 10
        'Return the options dict for the aggregation command.'
        options = {}
        if self._max_await_time_ms is not None:
            options['maxAwaitTimeMS'] = self._max_await_time_ms
        if self._batch_size is not None:
            options['batchSize'] = self._batch_size
        return options

    def _aggregation_pipeline(self) -> list[dict[str, Any]]:
        if False:
            return 10
        'Return the full aggregation pipeline for this ChangeStream.'
        options = self._change_stream_options()
        full_pipeline: list = [{'$changeStream': options}]
        full_pipeline.extend(self._pipeline)
        return full_pipeline

    def _process_result(self, result: Mapping[str, Any], conn: Connection) -> None:
        if False:
            return 10
        'Callback that caches the postBatchResumeToken or\n        startAtOperationTime from a changeStream aggregate command response\n        containing an empty batch of change documents.\n\n        This is implemented as a callback because we need access to the wire\n        version in order to determine whether to cache this value.\n        '
        if not result['cursor']['firstBatch']:
            if 'postBatchResumeToken' in result['cursor']:
                self._resume_token = result['cursor']['postBatchResumeToken']
            elif self._start_at_operation_time is None and self._uses_resume_after is False and (self._uses_start_after is False) and (conn.max_wire_version >= 7):
                self._start_at_operation_time = result.get('operationTime')
                if self._start_at_operation_time is None:
                    raise OperationFailure(f"Expected field 'operationTime' missing from command response : {result!r}")

    def _run_aggregation_cmd(self, session: Optional[ClientSession], explicit_session: bool) -> CommandCursor:
        if False:
            return 10
        'Run the full aggregation pipeline for this ChangeStream and return\n        the corresponding CommandCursor.\n        '
        cmd = self._aggregation_command_class(self._target, CommandCursor, self._aggregation_pipeline(), self._command_options(), explicit_session, result_processor=self._process_result, comment=self._comment)
        return self._client._retryable_read(cmd.get_cursor, self._target._read_preference_for(session), session)

    def _create_cursor(self) -> CommandCursor:
        if False:
            return 10
        with self._client._tmp_session(self._session, close=False) as s:
            return self._run_aggregation_cmd(session=s, explicit_session=self._session is not None)

    def _resume(self) -> None:
        if False:
            print('Hello World!')
        'Reestablish this change stream after a resumable error.'
        try:
            self._cursor.close()
        except PyMongoError:
            pass
        self._cursor = self._create_cursor()

    def close(self) -> None:
        if False:
            print('Hello World!')
        'Close this ChangeStream.'
        self._closed = True
        self._cursor.close()

    def __iter__(self) -> ChangeStream[_DocumentType]:
        if False:
            return 10
        return self

    @property
    def resume_token(self) -> Optional[Mapping[str, Any]]:
        if False:
            return 10
        'The cached resume token that will be used to resume after the most\n        recently returned change.\n\n        .. versionadded:: 3.9\n        '
        return copy.deepcopy(self._resume_token)

    @_csot.apply
    def next(self) -> _DocumentType:
        if False:
            for i in range(10):
                print('nop')
        "Advance the cursor.\n\n        This method blocks until the next change document is returned or an\n        unrecoverable error is raised. This method is used when iterating over\n        all changes in the cursor. For example::\n\n            try:\n                resume_token = None\n                pipeline = [{'$match': {'operationType': 'insert'}}]\n                with db.collection.watch(pipeline) as stream:\n                    for insert_change in stream:\n                        print(insert_change)\n                        resume_token = stream.resume_token\n            except pymongo.errors.PyMongoError:\n                # The ChangeStream encountered an unrecoverable error or the\n                # resume attempt failed to recreate the cursor.\n                if resume_token is None:\n                    # There is no usable resume token because there was a\n                    # failure during ChangeStream initialization.\n                    logging.error('...')\n                else:\n                    # Use the interrupted ChangeStream's resume token to create\n                    # a new ChangeStream. The new stream will continue from the\n                    # last seen insert change without missing any events.\n                    with db.collection.watch(\n                            pipeline, resume_after=resume_token) as stream:\n                        for insert_change in stream:\n                            print(insert_change)\n\n        Raises :exc:`StopIteration` if this ChangeStream is closed.\n        "
        while self.alive:
            doc = self.try_next()
            if doc is not None:
                return doc
        raise StopIteration
    __next__ = next

    @property
    def alive(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Does this cursor have the potential to return more data?\n\n        .. note:: Even if :attr:`alive` is ``True``, :meth:`next` can raise\n            :exc:`StopIteration` and :meth:`try_next` can return ``None``.\n\n        .. versionadded:: 3.8\n        '
        return not self._closed

    @_csot.apply
    def try_next(self) -> Optional[_DocumentType]:
        if False:
            for i in range(10):
                print('nop')
        'Advance the cursor without blocking indefinitely.\n\n        This method returns the next change document without waiting\n        indefinitely for the next change. For example::\n\n            with db.collection.watch() as stream:\n                while stream.alive:\n                    change = stream.try_next()\n                    # Note that the ChangeStream\'s resume token may be updated\n                    # even when no changes are returned.\n                    print("Current resume token: %r" % (stream.resume_token,))\n                    if change is not None:\n                        print("Change document: %r" % (change,))\n                        continue\n                    # We end up here when there are no recent changes.\n                    # Sleep for a while before trying again to avoid flooding\n                    # the server with getMore requests when no changes are\n                    # available.\n                    time.sleep(10)\n\n        If no change document is cached locally then this method runs a single\n        getMore command. If the getMore yields any documents, the next\n        document is returned, otherwise, if the getMore returns no documents\n        (because there have been no changes) then ``None`` is returned.\n\n        :Returns:\n          The next change document or ``None`` when no document is available\n          after running a single getMore or when the cursor is closed.\n\n        .. versionadded:: 3.8\n        '
        if not self._closed and (not self._cursor.alive):
            self._resume()
        try:
            try:
                change = self._cursor._try_next(True)
            except PyMongoError as exc:
                if not _resumable(exc):
                    raise
                self._resume()
                change = self._cursor._try_next(False)
        except PyMongoError as exc:
            if not _resumable(exc) and (not exc.timeout):
                self.close()
            raise
        except Exception:
            self.close()
            raise
        if not self._cursor.alive:
            self._closed = True
        if change is None:
            if self._cursor._post_batch_resume_token is not None:
                self._resume_token = self._cursor._post_batch_resume_token
                self._start_at_operation_time = None
            return change
        try:
            resume_token = change['_id']
        except KeyError:
            self.close()
            raise InvalidOperation('Cannot provide resume functionality when the resume token is missing.') from None
        if not self._cursor._has_next() and self._cursor._post_batch_resume_token:
            resume_token = self._cursor._post_batch_resume_token
        self._uses_start_after = False
        self._uses_resume_after = True
        self._resume_token = resume_token
        self._start_at_operation_time = None
        if self._decode_custom:
            return _bson_to_dict(change.raw, self._orig_codec_options)
        return change

    def __enter__(self) -> ChangeStream[_DocumentType]:
        if False:
            print('Hello World!')
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if False:
            return 10
        self.close()

class CollectionChangeStream(ChangeStream[_DocumentType]):
    """A change stream that watches changes on a single collection.

    Should not be called directly by application developers. Use
    helper method :meth:`pymongo.collection.Collection.watch` instead.

    .. versionadded:: 3.7
    """
    _target: Collection[_DocumentType]

    @property
    def _aggregation_command_class(self) -> Type[_CollectionAggregationCommand]:
        if False:
            print('Hello World!')
        return _CollectionAggregationCommand

    @property
    def _client(self) -> MongoClient[_DocumentType]:
        if False:
            return 10
        return self._target.database.client

class DatabaseChangeStream(ChangeStream[_DocumentType]):
    """A change stream that watches changes on all collections in a database.

    Should not be called directly by application developers. Use
    helper method :meth:`pymongo.database.Database.watch` instead.

    .. versionadded:: 3.7
    """
    _target: Database[_DocumentType]

    @property
    def _aggregation_command_class(self) -> Type[_DatabaseAggregationCommand]:
        if False:
            return 10
        return _DatabaseAggregationCommand

    @property
    def _client(self) -> MongoClient[_DocumentType]:
        if False:
            print('Hello World!')
        return self._target.client

class ClusterChangeStream(DatabaseChangeStream[_DocumentType]):
    """A change stream that watches changes on all collections in the cluster.

    Should not be called directly by application developers. Use
    helper method :meth:`pymongo.mongo_client.MongoClient.watch` instead.

    .. versionadded:: 3.7
    """

    def _change_stream_options(self) -> dict[str, Any]:
        if False:
            while True:
                i = 10
        options = super()._change_stream_options()
        options['allChangesForCluster'] = True
        return options