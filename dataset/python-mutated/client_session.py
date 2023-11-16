"""Logical sessions for ordering sequential operations.

.. versionadded:: 3.6

Causally Consistent Reads
=========================

.. code-block:: python

  with client.start_session(causal_consistency=True) as session:
      collection = client.db.collection
      collection.update_one({"_id": 1}, {"$set": {"x": 10}}, session=session)
      secondary_c = collection.with_options(read_preference=ReadPreference.SECONDARY)

      # A secondary read waits for replication of the write.
      secondary_c.find_one({"_id": 1}, session=session)

If `causal_consistency` is True (the default), read operations that use
the session are causally after previous read and write operations. Using a
causally consistent session, an application can read its own writes and is
guaranteed monotonic reads, even when reading from replica set secondaries.

.. seealso:: The MongoDB documentation on `causal-consistency <https://dochub.mongodb.org/core/causal-consistency>`_.

.. _transactions-ref:

Transactions
============

.. versionadded:: 3.7

MongoDB 4.0 adds support for transactions on replica set primaries. A
transaction is associated with a :class:`ClientSession`. To start a transaction
on a session, use :meth:`ClientSession.start_transaction` in a with-statement.
Then, execute an operation within the transaction by passing the session to the
operation:

.. code-block:: python

  orders = client.db.orders
  inventory = client.db.inventory
  with client.start_session() as session:
      with session.start_transaction():
          orders.insert_one({"sku": "abc123", "qty": 100}, session=session)
          inventory.update_one(
              {"sku": "abc123", "qty": {"$gte": 100}},
              {"$inc": {"qty": -100}},
              session=session,
          )

Upon normal completion of ``with session.start_transaction()`` block, the
transaction automatically calls :meth:`ClientSession.commit_transaction`.
If the block exits with an exception, the transaction automatically calls
:meth:`ClientSession.abort_transaction`.

In general, multi-document transactions only support read/write (CRUD)
operations on existing collections. However, MongoDB 4.4 adds support for
creating collections and indexes with some limitations, including an
insert operation that would result in the creation of a new collection.
For a complete description of all the supported and unsupported operations
see the `MongoDB server's documentation for transactions
<http://dochub.mongodb.org/core/transactions>`_.

A session may only have a single active transaction at a time, multiple
transactions on the same session can be executed in sequence.

Sharded Transactions
^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.9

PyMongo 3.9 adds support for transactions on sharded clusters running MongoDB
>=4.2. Sharded transactions have the same API as replica set transactions.
When running a transaction against a sharded cluster, the session is
pinned to the mongos server selected for the first operation in the
transaction. All subsequent operations that are part of the same transaction
are routed to the same mongos server. When the transaction is completed, by
running either commitTransaction or abortTransaction, the session is unpinned.

.. seealso:: The MongoDB documentation on `transactions <https://dochub.mongodb.org/core/transactions>`_.

.. _snapshot-reads-ref:

Snapshot Reads
==============

.. versionadded:: 3.12

MongoDB 5.0 adds support for snapshot reads. Snapshot reads are requested by
passing the ``snapshot`` option to
:meth:`~pymongo.mongo_client.MongoClient.start_session`.
If ``snapshot`` is True, all read operations that use this session read data
from the same snapshot timestamp. The server chooses the latest
majority-committed snapshot timestamp when executing the first read operation
using the session. Subsequent reads on this session read from the same
snapshot timestamp. Snapshot reads are also supported when reading from
replica set secondaries.

.. code-block:: python

  # Each read using this session reads data from the same point in time.
  with client.start_session(snapshot=True) as session:
      order = orders.find_one({"sku": "abc123"}, session=session)
      inventory = inventory.find_one({"sku": "abc123"}, session=session)

Snapshot Reads Limitations
^^^^^^^^^^^^^^^^^^^^^^^^^^

Snapshot reads sessions are incompatible with ``causal_consistency=True``.
Only the following read operations are supported in a snapshot reads session:

- :meth:`~pymongo.collection.Collection.find`
- :meth:`~pymongo.collection.Collection.find_one`
- :meth:`~pymongo.collection.Collection.aggregate`
- :meth:`~pymongo.collection.Collection.count_documents`
- :meth:`~pymongo.collection.Collection.distinct` (on unsharded collections)

Classes
=======
"""
from __future__ import annotations
import collections
import time
import uuid
from collections.abc import Mapping as _Mapping
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Mapping, MutableMapping, NoReturn, Optional, Type, TypeVar
from bson.binary import Binary
from bson.int64 import Int64
from bson.son import SON
from bson.timestamp import Timestamp
from pymongo import _csot
from pymongo.cursor import _ConnectionManager
from pymongo.errors import ConfigurationError, ConnectionFailure, InvalidOperation, OperationFailure, PyMongoError, WTimeoutError
from pymongo.helpers import _RETRYABLE_ERROR_CODES
from pymongo.read_concern import ReadConcern
from pymongo.read_preferences import ReadPreference, _ServerMode
from pymongo.server_type import SERVER_TYPE
from pymongo.write_concern import WriteConcern
if TYPE_CHECKING:
    from types import TracebackType
    from pymongo.pool import Connection
    from pymongo.server import Server
    from pymongo.typings import ClusterTime, _Address

class SessionOptions:
    """Options for a new :class:`ClientSession`.

    :Parameters:
      - `causal_consistency` (optional): If True, read operations are causally
        ordered within the session. Defaults to True when the ``snapshot``
        option is ``False``.
      - `default_transaction_options` (optional): The default
        TransactionOptions to use for transactions started on this session.
      - `snapshot` (optional): If True, then all reads performed using this
        session will read from the same snapshot. This option is incompatible
        with ``causal_consistency=True``. Defaults to ``False``.

    .. versionchanged:: 3.12
       Added the ``snapshot`` parameter.
    """

    def __init__(self, causal_consistency: Optional[bool]=None, default_transaction_options: Optional[TransactionOptions]=None, snapshot: Optional[bool]=False) -> None:
        if False:
            while True:
                i = 10
        if snapshot:
            if causal_consistency:
                raise ConfigurationError('snapshot reads do not support causal_consistency=True')
            causal_consistency = False
        elif causal_consistency is None:
            causal_consistency = True
        self._causal_consistency = causal_consistency
        if default_transaction_options is not None:
            if not isinstance(default_transaction_options, TransactionOptions):
                raise TypeError('default_transaction_options must be an instance of pymongo.client_session.TransactionOptions, not: {!r}'.format(default_transaction_options))
        self._default_transaction_options = default_transaction_options
        self._snapshot = snapshot

    @property
    def causal_consistency(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Whether causal consistency is configured.'
        return self._causal_consistency

    @property
    def default_transaction_options(self) -> Optional[TransactionOptions]:
        if False:
            while True:
                i = 10
        'The default TransactionOptions to use for transactions started on\n        this session.\n\n        .. versionadded:: 3.7\n        '
        return self._default_transaction_options

    @property
    def snapshot(self) -> Optional[bool]:
        if False:
            return 10
        'Whether snapshot reads are configured.\n\n        .. versionadded:: 3.12\n        '
        return self._snapshot

class TransactionOptions:
    """Options for :meth:`ClientSession.start_transaction`.

    :Parameters:
      - `read_concern` (optional): The
        :class:`~pymongo.read_concern.ReadConcern` to use for this transaction.
        If ``None`` (the default) the :attr:`read_preference` of
        the :class:`MongoClient` is used.
      - `write_concern` (optional): The
        :class:`~pymongo.write_concern.WriteConcern` to use for this
        transaction. If ``None`` (the default) the :attr:`read_preference` of
        the :class:`MongoClient` is used.
      - `read_preference` (optional): The read preference to use. If
        ``None`` (the default) the :attr:`read_preference` of this
        :class:`MongoClient` is used. See :mod:`~pymongo.read_preferences`
        for options. Transactions which read must use
        :attr:`~pymongo.read_preferences.ReadPreference.PRIMARY`.
      - `max_commit_time_ms` (optional): The maximum amount of time to allow a
        single commitTransaction command to run. This option is an alias for
        maxTimeMS option on the commitTransaction command. If ``None`` (the
        default) maxTimeMS is not used.

    .. versionchanged:: 3.9
       Added the ``max_commit_time_ms`` option.

    .. versionadded:: 3.7
    """

    def __init__(self, read_concern: Optional[ReadConcern]=None, write_concern: Optional[WriteConcern]=None, read_preference: Optional[_ServerMode]=None, max_commit_time_ms: Optional[int]=None) -> None:
        if False:
            i = 10
            return i + 15
        self._read_concern = read_concern
        self._write_concern = write_concern
        self._read_preference = read_preference
        self._max_commit_time_ms = max_commit_time_ms
        if read_concern is not None:
            if not isinstance(read_concern, ReadConcern):
                raise TypeError(f'read_concern must be an instance of pymongo.read_concern.ReadConcern, not: {read_concern!r}')
        if write_concern is not None:
            if not isinstance(write_concern, WriteConcern):
                raise TypeError(f'write_concern must be an instance of pymongo.write_concern.WriteConcern, not: {write_concern!r}')
            if not write_concern.acknowledged:
                raise ConfigurationError(f'transactions do not support unacknowledged write concern: {write_concern!r}')
        if read_preference is not None:
            if not isinstance(read_preference, _ServerMode):
                raise TypeError(f'{read_preference!r} is not valid for read_preference. See pymongo.read_preferences for valid options.')
        if max_commit_time_ms is not None:
            if not isinstance(max_commit_time_ms, int):
                raise TypeError('max_commit_time_ms must be an integer or None')

    @property
    def read_concern(self) -> Optional[ReadConcern]:
        if False:
            i = 10
            return i + 15
        "This transaction's :class:`~pymongo.read_concern.ReadConcern`."
        return self._read_concern

    @property
    def write_concern(self) -> Optional[WriteConcern]:
        if False:
            while True:
                i = 10
        "This transaction's :class:`~pymongo.write_concern.WriteConcern`."
        return self._write_concern

    @property
    def read_preference(self) -> Optional[_ServerMode]:
        if False:
            while True:
                i = 10
        "This transaction's :class:`~pymongo.read_preferences.ReadPreference`."
        return self._read_preference

    @property
    def max_commit_time_ms(self) -> Optional[int]:
        if False:
            while True:
                i = 10
        'The maxTimeMS to use when running a commitTransaction command.\n\n        .. versionadded:: 3.9\n        '
        return self._max_commit_time_ms

def _validate_session_write_concern(session: Optional[ClientSession], write_concern: Optional[WriteConcern]) -> Optional[ClientSession]:
    if False:
        i = 10
        return i + 15
    "Validate that an explicit session is not used with an unack'ed write.\n\n    Returns the session to use for the next operation.\n    "
    if session:
        if write_concern is not None and (not write_concern.acknowledged):
            if session._implicit:
                return None
            else:
                raise ConfigurationError(f'Explicit sessions are incompatible with unacknowledged write concern: {write_concern!r}')
    return session

class _TransactionContext:
    """Internal transaction context manager for start_transaction."""

    def __init__(self, session: ClientSession):
        if False:
            print('Hello World!')
        self.__session = session

    def __enter__(self) -> _TransactionContext:
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        if False:
            return 10
        if self.__session.in_transaction:
            if exc_val is None:
                self.__session.commit_transaction()
            else:
                self.__session.abort_transaction()

class _TxnState:
    NONE = 1
    STARTING = 2
    IN_PROGRESS = 3
    COMMITTED = 4
    COMMITTED_EMPTY = 5
    ABORTED = 6

class _Transaction:
    """Internal class to hold transaction information in a ClientSession."""

    def __init__(self, opts: Optional[TransactionOptions], client: MongoClient):
        if False:
            for i in range(10):
                print('nop')
        self.opts = opts
        self.state = _TxnState.NONE
        self.sharded = False
        self.pinned_address: Optional[_Address] = None
        self.conn_mgr: Optional[_ConnectionManager] = None
        self.recovery_token = None
        self.attempt = 0
        self.client = client

    def active(self) -> bool:
        if False:
            print('Hello World!')
        return self.state in (_TxnState.STARTING, _TxnState.IN_PROGRESS)

    def starting(self) -> bool:
        if False:
            return 10
        return self.state == _TxnState.STARTING

    @property
    def pinned_conn(self) -> Optional[Connection]:
        if False:
            i = 10
            return i + 15
        if self.active() and self.conn_mgr:
            return self.conn_mgr.conn
        return None

    def pin(self, server: Server, conn: Connection) -> None:
        if False:
            print('Hello World!')
        self.sharded = True
        self.pinned_address = server.description.address
        if server.description.server_type == SERVER_TYPE.LoadBalancer:
            conn.pin_txn()
            self.conn_mgr = _ConnectionManager(conn, False)

    def unpin(self) -> None:
        if False:
            print('Hello World!')
        self.pinned_address = None
        if self.conn_mgr:
            self.conn_mgr.close()
        self.conn_mgr = None

    def reset(self) -> None:
        if False:
            i = 10
            return i + 15
        self.unpin()
        self.state = _TxnState.NONE
        self.sharded = False
        self.recovery_token = None
        self.attempt = 0

    def __del__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.conn_mgr:
            self.client._close_cursor_soon(0, None, self.conn_mgr)
            self.conn_mgr = None

def _reraise_with_unknown_commit(exc: Any) -> NoReturn:
    if False:
        return 10
    'Re-raise an exception with the UnknownTransactionCommitResult label.'
    exc._add_error_label('UnknownTransactionCommitResult')
    raise

def _max_time_expired_error(exc: PyMongoError) -> bool:
    if False:
        print('Hello World!')
    'Return true if exc is a MaxTimeMSExpired error.'
    return isinstance(exc, OperationFailure) and exc.code == 50
_UNKNOWN_COMMIT_ERROR_CODES: frozenset = _RETRYABLE_ERROR_CODES | frozenset([64, 50])
_WITH_TRANSACTION_RETRY_TIME_LIMIT = 120

def _within_time_limit(start_time: float) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Are we within the with_transaction retry limit?'
    return time.monotonic() - start_time < _WITH_TRANSACTION_RETRY_TIME_LIMIT
_T = TypeVar('_T')
if TYPE_CHECKING:
    from pymongo.mongo_client import MongoClient

class ClientSession:
    """A session for ordering sequential operations.

    :class:`ClientSession` instances are **not thread-safe or fork-safe**.
    They can only be used by one thread or process at a time. A single
    :class:`ClientSession` cannot be used to run multiple operations
    concurrently.

    Should not be initialized directly by application developers - to create a
    :class:`ClientSession`, call
    :meth:`~pymongo.mongo_client.MongoClient.start_session`.
    """

    def __init__(self, client: MongoClient, server_session: Any, options: SessionOptions, implicit: bool) -> None:
        if False:
            while True:
                i = 10
        self._client: MongoClient = client
        self._server_session = server_session
        self._options = options
        self._cluster_time: Optional[Mapping[str, Any]] = None
        self._operation_time: Optional[Timestamp] = None
        self._snapshot_time = None
        self._implicit = implicit
        self._transaction = _Transaction(None, client)

    def end_session(self) -> None:
        if False:
            while True:
                i = 10
        'Finish this session. If a transaction has started, abort it.\n\n        It is an error to use the session after the session has ended.\n        '
        self._end_session(lock=True)

    def _end_session(self, lock: bool) -> None:
        if False:
            i = 10
            return i + 15
        if self._server_session is not None:
            try:
                if self.in_transaction:
                    self.abort_transaction()
                self._unpin()
            finally:
                self._client._return_server_session(self._server_session, lock)
                self._server_session = None

    def _check_ended(self) -> None:
        if False:
            i = 10
            return i + 15
        if self._server_session is None:
            raise InvalidOperation('Cannot use ended session')

    def __enter__(self) -> ClientSession:
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if False:
            i = 10
            return i + 15
        self._end_session(lock=True)

    @property
    def client(self) -> MongoClient:
        if False:
            while True:
                i = 10
        'The :class:`~pymongo.mongo_client.MongoClient` this session was\n        created from.\n        '
        return self._client

    @property
    def options(self) -> SessionOptions:
        if False:
            return 10
        'The :class:`SessionOptions` this session was created with.'
        return self._options

    @property
    def session_id(self) -> Mapping[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'A BSON document, the opaque server session identifier.'
        self._check_ended()
        return self._server_session.session_id

    @property
    def cluster_time(self) -> Optional[ClusterTime]:
        if False:
            i = 10
            return i + 15
        'The cluster time returned by the last operation executed\n        in this session.\n        '
        return self._cluster_time

    @property
    def operation_time(self) -> Optional[Timestamp]:
        if False:
            print('Hello World!')
        'The operation time returned by the last operation executed\n        in this session.\n        '
        return self._operation_time

    def _inherit_option(self, name: str, val: _T) -> _T:
        if False:
            i = 10
            return i + 15
        'Return the inherited TransactionOption value.'
        if val:
            return val
        txn_opts = self.options.default_transaction_options
        parent_val = txn_opts and getattr(txn_opts, name)
        if parent_val:
            return parent_val
        return getattr(self.client, name)

    def with_transaction(self, callback: Callable[[ClientSession], _T], read_concern: Optional[ReadConcern]=None, write_concern: Optional[WriteConcern]=None, read_preference: Optional[_ServerMode]=None, max_commit_time_ms: Optional[int]=None) -> _T:
        if False:
            for i in range(10):
                print('nop')
        'Execute a callback in a transaction.\n\n        This method starts a transaction on this session, executes ``callback``\n        once, and then commits the transaction. For example::\n\n          def callback(session):\n              orders = session.client.db.orders\n              inventory = session.client.db.inventory\n              orders.insert_one({"sku": "abc123", "qty": 100}, session=session)\n              inventory.update_one({"sku": "abc123", "qty": {"$gte": 100}},\n                                   {"$inc": {"qty": -100}}, session=session)\n\n          with client.start_session() as session:\n              session.with_transaction(callback)\n\n        To pass arbitrary arguments to the ``callback``, wrap your callable\n        with a ``lambda`` like this::\n\n          def callback(session, custom_arg, custom_kwarg=None):\n              # Transaction operations...\n\n          with client.start_session() as session:\n              session.with_transaction(\n                  lambda s: callback(s, "custom_arg", custom_kwarg=1))\n\n        In the event of an exception, ``with_transaction`` may retry the commit\n        or the entire transaction, therefore ``callback`` may be invoked\n        multiple times by a single call to ``with_transaction``. Developers\n        should be mindful of this possibility when writing a ``callback`` that\n        modifies application state or has any other side-effects.\n        Note that even when the ``callback`` is invoked multiple times,\n        ``with_transaction`` ensures that the transaction will be committed\n        at-most-once on the server.\n\n        The ``callback`` should not attempt to start new transactions, but\n        should simply run operations meant to be contained within a\n        transaction. The ``callback`` should also not commit the transaction;\n        this is handled automatically by ``with_transaction``. If the\n        ``callback`` does commit or abort the transaction without error,\n        however, ``with_transaction`` will return without taking further\n        action.\n\n        :class:`ClientSession` instances are **not thread-safe or fork-safe**.\n        Consequently, the ``callback`` must not attempt to execute multiple\n        operations concurrently.\n\n        When ``callback`` raises an exception, ``with_transaction``\n        automatically aborts the current transaction. When ``callback`` or\n        :meth:`~ClientSession.commit_transaction` raises an exception that\n        includes the ``"TransientTransactionError"`` error label,\n        ``with_transaction`` starts a new transaction and re-executes\n        the ``callback``.\n\n        When :meth:`~ClientSession.commit_transaction` raises an exception with\n        the ``"UnknownTransactionCommitResult"`` error label,\n        ``with_transaction`` retries the commit until the result of the\n        transaction is known.\n\n        This method will cease retrying after 120 seconds has elapsed. This\n        timeout is not configurable and any exception raised by the\n        ``callback`` or by :meth:`ClientSession.commit_transaction` after the\n        timeout is reached will be re-raised. Applications that desire a\n        different timeout duration should not use this method.\n\n        :Parameters:\n          - `callback`: The callable ``callback`` to run inside a transaction.\n            The callable must accept a single argument, this session. Note,\n            under certain error conditions the callback may be run multiple\n            times.\n          - `read_concern` (optional): The\n            :class:`~pymongo.read_concern.ReadConcern` to use for this\n            transaction.\n          - `write_concern` (optional): The\n            :class:`~pymongo.write_concern.WriteConcern` to use for this\n            transaction.\n          - `read_preference` (optional): The read preference to use for this\n            transaction. If ``None`` (the default) the :attr:`read_preference`\n            of this :class:`Database` is used. See\n            :mod:`~pymongo.read_preferences` for options.\n\n        :Returns:\n          The return value of the ``callback``.\n\n        .. versionadded:: 3.9\n        '
        start_time = time.monotonic()
        while True:
            self.start_transaction(read_concern, write_concern, read_preference, max_commit_time_ms)
            try:
                ret = callback(self)
            except Exception as exc:
                if self.in_transaction:
                    self.abort_transaction()
                if isinstance(exc, PyMongoError) and exc.has_error_label('TransientTransactionError') and _within_time_limit(start_time):
                    continue
                raise
            if not self.in_transaction:
                return ret
            while True:
                try:
                    self.commit_transaction()
                except PyMongoError as exc:
                    if exc.has_error_label('UnknownTransactionCommitResult') and _within_time_limit(start_time) and (not _max_time_expired_error(exc)):
                        continue
                    if exc.has_error_label('TransientTransactionError') and _within_time_limit(start_time):
                        break
                    raise
                return ret

    def start_transaction(self, read_concern: Optional[ReadConcern]=None, write_concern: Optional[WriteConcern]=None, read_preference: Optional[_ServerMode]=None, max_commit_time_ms: Optional[int]=None) -> ContextManager:
        if False:
            print('Hello World!')
        'Start a multi-statement transaction.\n\n        Takes the same arguments as :class:`TransactionOptions`.\n\n        .. versionchanged:: 3.9\n           Added the ``max_commit_time_ms`` option.\n\n        .. versionadded:: 3.7\n        '
        self._check_ended()
        if self.options.snapshot:
            raise InvalidOperation('Transactions are not supported in snapshot sessions')
        if self.in_transaction:
            raise InvalidOperation('Transaction already in progress')
        read_concern = self._inherit_option('read_concern', read_concern)
        write_concern = self._inherit_option('write_concern', write_concern)
        read_preference = self._inherit_option('read_preference', read_preference)
        if max_commit_time_ms is None:
            opts = self.options.default_transaction_options
            if opts:
                max_commit_time_ms = opts.max_commit_time_ms
        self._transaction.opts = TransactionOptions(read_concern, write_concern, read_preference, max_commit_time_ms)
        self._transaction.reset()
        self._transaction.state = _TxnState.STARTING
        self._start_retryable_write()
        return _TransactionContext(self)

    def commit_transaction(self) -> None:
        if False:
            i = 10
            return i + 15
        'Commit a multi-statement transaction.\n\n        .. versionadded:: 3.7\n        '
        self._check_ended()
        state = self._transaction.state
        if state is _TxnState.NONE:
            raise InvalidOperation('No transaction started')
        elif state in (_TxnState.STARTING, _TxnState.COMMITTED_EMPTY):
            self._transaction.state = _TxnState.COMMITTED_EMPTY
            return
        elif state is _TxnState.ABORTED:
            raise InvalidOperation('Cannot call commitTransaction after calling abortTransaction')
        elif state is _TxnState.COMMITTED:
            self._transaction.state = _TxnState.IN_PROGRESS
        try:
            self._finish_transaction_with_retry('commitTransaction')
        except ConnectionFailure as exc:
            exc._remove_error_label('TransientTransactionError')
            _reraise_with_unknown_commit(exc)
        except WTimeoutError as exc:
            _reraise_with_unknown_commit(exc)
        except OperationFailure as exc:
            if exc.code not in _UNKNOWN_COMMIT_ERROR_CODES:
                raise
            _reraise_with_unknown_commit(exc)
        finally:
            self._transaction.state = _TxnState.COMMITTED

    def abort_transaction(self) -> None:
        if False:
            i = 10
            return i + 15
        'Abort a multi-statement transaction.\n\n        .. versionadded:: 3.7\n        '
        self._check_ended()
        state = self._transaction.state
        if state is _TxnState.NONE:
            raise InvalidOperation('No transaction started')
        elif state is _TxnState.STARTING:
            self._transaction.state = _TxnState.ABORTED
            return
        elif state is _TxnState.ABORTED:
            raise InvalidOperation('Cannot call abortTransaction twice')
        elif state in (_TxnState.COMMITTED, _TxnState.COMMITTED_EMPTY):
            raise InvalidOperation('Cannot call abortTransaction after calling commitTransaction')
        try:
            self._finish_transaction_with_retry('abortTransaction')
        except (OperationFailure, ConnectionFailure):
            pass
        finally:
            self._transaction.state = _TxnState.ABORTED
            self._unpin()

    def _finish_transaction_with_retry(self, command_name: str) -> dict[str, Any]:
        if False:
            return 10
        'Run commit or abort with one retry after any retryable error.\n\n        :Parameters:\n          - `command_name`: Either "commitTransaction" or "abortTransaction".\n        '

        def func(_session: Optional[ClientSession], conn: Connection, _retryable: bool) -> dict[str, Any]:
            if False:
                print('Hello World!')
            return self._finish_transaction(conn, command_name)
        return self._client._retry_internal(func, self, None, retryable=True)

    def _finish_transaction(self, conn: Connection, command_name: str) -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        self._transaction.attempt += 1
        opts = self._transaction.opts
        assert opts
        wc = opts.write_concern
        cmd = SON([(command_name, 1)])
        if command_name == 'commitTransaction':
            if opts.max_commit_time_ms and _csot.get_timeout() is None:
                cmd['maxTimeMS'] = opts.max_commit_time_ms
            if self._transaction.attempt > 1:
                assert wc
                wc_doc = wc.document
                wc_doc['w'] = 'majority'
                wc_doc.setdefault('wtimeout', 10000)
                wc = WriteConcern(**wc_doc)
        if self._transaction.recovery_token:
            cmd['recoveryToken'] = self._transaction.recovery_token
        return self._client.admin._command(conn, cmd, session=self, write_concern=wc, parse_write_concern_error=True)

    def _advance_cluster_time(self, cluster_time: Optional[Mapping[str, Any]]) -> None:
        if False:
            i = 10
            return i + 15
        'Internal cluster time helper.'
        if self._cluster_time is None:
            self._cluster_time = cluster_time
        elif cluster_time is not None:
            if cluster_time['clusterTime'] > self._cluster_time['clusterTime']:
                self._cluster_time = cluster_time

    def advance_cluster_time(self, cluster_time: Mapping[str, Any]) -> None:
        if False:
            print('Hello World!')
        'Update the cluster time for this session.\n\n        :Parameters:\n          - `cluster_time`: The\n            :data:`~pymongo.client_session.ClientSession.cluster_time` from\n            another `ClientSession` instance.\n        '
        if not isinstance(cluster_time, _Mapping):
            raise TypeError('cluster_time must be a subclass of collections.Mapping')
        if not isinstance(cluster_time.get('clusterTime'), Timestamp):
            raise ValueError('Invalid cluster_time')
        self._advance_cluster_time(cluster_time)

    def _advance_operation_time(self, operation_time: Optional[Timestamp]) -> None:
        if False:
            return 10
        'Internal operation time helper.'
        if self._operation_time is None:
            self._operation_time = operation_time
        elif operation_time is not None:
            if operation_time > self._operation_time:
                self._operation_time = operation_time

    def advance_operation_time(self, operation_time: Timestamp) -> None:
        if False:
            while True:
                i = 10
        'Update the operation time for this session.\n\n        :Parameters:\n          - `operation_time`: The\n            :data:`~pymongo.client_session.ClientSession.operation_time` from\n            another `ClientSession` instance.\n        '
        if not isinstance(operation_time, Timestamp):
            raise TypeError('operation_time must be an instance of bson.timestamp.Timestamp')
        self._advance_operation_time(operation_time)

    def _process_response(self, reply: Mapping[str, Any]) -> None:
        if False:
            print('Hello World!')
        'Process a response to a command that was run with this session.'
        self._advance_cluster_time(reply.get('$clusterTime'))
        self._advance_operation_time(reply.get('operationTime'))
        if self._options.snapshot and self._snapshot_time is None:
            if 'cursor' in reply:
                ct = reply['cursor'].get('atClusterTime')
            else:
                ct = reply.get('atClusterTime')
            self._snapshot_time = ct
        if self.in_transaction and self._transaction.sharded:
            recovery_token = reply.get('recoveryToken')
            if recovery_token:
                self._transaction.recovery_token = recovery_token

    @property
    def has_ended(self) -> bool:
        if False:
            print('Hello World!')
        'True if this session is finished.'
        return self._server_session is None

    @property
    def in_transaction(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'True if this session has an active multi-statement transaction.\n\n        .. versionadded:: 3.10\n        '
        return self._transaction.active()

    @property
    def _starting_transaction(self) -> bool:
        if False:
            while True:
                i = 10
        'True if this session is starting a multi-statement transaction.'
        return self._transaction.starting()

    @property
    def _pinned_address(self) -> Optional[_Address]:
        if False:
            print('Hello World!')
        'The mongos address this transaction was created on.'
        if self._transaction.active():
            return self._transaction.pinned_address
        return None

    @property
    def _pinned_connection(self) -> Optional[Connection]:
        if False:
            return 10
        'The connection this transaction was started on.'
        return self._transaction.pinned_conn

    def _pin(self, server: Server, conn: Connection) -> None:
        if False:
            while True:
                i = 10
        'Pin this session to the given Server or to the given connection.'
        self._transaction.pin(server, conn)

    def _unpin(self) -> None:
        if False:
            i = 10
            return i + 15
        'Unpin this session from any pinned Server.'
        self._transaction.unpin()

    def _txn_read_preference(self) -> Optional[_ServerMode]:
        if False:
            return 10
        'Return read preference of this transaction or None.'
        if self.in_transaction:
            assert self._transaction.opts
            return self._transaction.opts.read_preference
        return None

    def _materialize(self) -> None:
        if False:
            return 10
        if isinstance(self._server_session, _EmptyServerSession):
            old = self._server_session
            self._server_session = self._client._topology.get_server_session()
            if old.started_retryable_write:
                self._server_session.inc_transaction_id()

    def _apply_to(self, command: MutableMapping[str, Any], is_retryable: bool, read_preference: _ServerMode, conn: Connection) -> None:
        if False:
            print('Hello World!')
        self._check_ended()
        self._materialize()
        if self.options.snapshot:
            self._update_read_concern(command, conn)
        self._server_session.last_use = time.monotonic()
        command['lsid'] = self._server_session.session_id
        if is_retryable:
            command['txnNumber'] = self._server_session.transaction_id
            return
        if self.in_transaction:
            if read_preference != ReadPreference.PRIMARY:
                raise InvalidOperation(f'read preference in a transaction must be primary, not: {read_preference!r}')
            if self._transaction.state == _TxnState.STARTING:
                self._transaction.state = _TxnState.IN_PROGRESS
                command['startTransaction'] = True
                assert self._transaction.opts
                if self._transaction.opts.read_concern:
                    rc = self._transaction.opts.read_concern.document
                    if rc:
                        command['readConcern'] = rc
                self._update_read_concern(command, conn)
            command['txnNumber'] = self._server_session.transaction_id
            command['autocommit'] = False

    def _start_retryable_write(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._check_ended()
        self._server_session.inc_transaction_id()

    def _update_read_concern(self, cmd: MutableMapping[str, Any], conn: Connection) -> None:
        if False:
            i = 10
            return i + 15
        if self.options.causal_consistency and self.operation_time is not None:
            cmd.setdefault('readConcern', {})['afterClusterTime'] = self.operation_time
        if self.options.snapshot:
            if conn.max_wire_version < 13:
                raise ConfigurationError('Snapshot reads require MongoDB 5.0 or later')
            rc = cmd.setdefault('readConcern', {})
            rc['level'] = 'snapshot'
            if self._snapshot_time is not None:
                rc['atClusterTime'] = self._snapshot_time

    def __copy__(self) -> NoReturn:
        if False:
            while True:
                i = 10
        raise TypeError('A ClientSession cannot be copied, create a new session instead')

class _EmptyServerSession:
    __slots__ = ('dirty', 'started_retryable_write')

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.dirty = False
        self.started_retryable_write = False

    def mark_dirty(self) -> None:
        if False:
            i = 10
            return i + 15
        self.dirty = True

    def inc_transaction_id(self) -> None:
        if False:
            i = 10
            return i + 15
        self.started_retryable_write = True

class _ServerSession:

    def __init__(self, generation: int):
        if False:
            for i in range(10):
                print('nop')
        self.session_id = {'id': Binary(uuid.uuid4().bytes, 4)}
        self.last_use = time.monotonic()
        self._transaction_id = 0
        self.dirty = False
        self.generation = generation

    def mark_dirty(self) -> None:
        if False:
            while True:
                i = 10
        'Mark this session as dirty.\n\n        A server session is marked dirty when a command fails with a network\n        error. Dirty sessions are later discarded from the server session pool.\n        '
        self.dirty = True

    def timed_out(self, session_timeout_minutes: float) -> bool:
        if False:
            for i in range(10):
                print('nop')
        idle_seconds = time.monotonic() - self.last_use
        return idle_seconds > (session_timeout_minutes - 1) * 60

    @property
    def transaction_id(self) -> Int64:
        if False:
            i = 10
            return i + 15
        'Positive 64-bit integer.'
        return Int64(self._transaction_id)

    def inc_transaction_id(self) -> None:
        if False:
            return 10
        self._transaction_id += 1

class _ServerSessionPool(collections.deque):
    """Pool of _ServerSession objects.

    This class is not thread-safe, access it while holding the Topology lock.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self.generation = 0

    def reset(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.generation += 1
        self.clear()

    def pop_all(self) -> list[_ServerSession]:
        if False:
            print('Hello World!')
        ids = []
        while self:
            ids.append(self.pop().session_id)
        return ids

    def get_server_session(self, session_timeout_minutes: float) -> _ServerSession:
        if False:
            return 10
        self._clear_stale(session_timeout_minutes)
        while self:
            s = self.popleft()
            if not s.timed_out(session_timeout_minutes):
                return s
        return _ServerSession(self.generation)

    def return_server_session(self, server_session: _ServerSession, session_timeout_minutes: Optional[float]) -> None:
        if False:
            while True:
                i = 10
        if session_timeout_minutes is not None:
            self._clear_stale(session_timeout_minutes)
            if server_session.timed_out(session_timeout_minutes):
                return
        self.return_server_session_no_lock(server_session)

    def return_server_session_no_lock(self, server_session: _ServerSession) -> None:
        if False:
            return 10
        if server_session.generation == self.generation and (not server_session.dirty):
            self.appendleft(server_session)

    def _clear_stale(self, session_timeout_minutes: float) -> None:
        if False:
            print('Hello World!')
        while self:
            if self[-1].timed_out(session_timeout_minutes):
                self.pop()
            else:
                break