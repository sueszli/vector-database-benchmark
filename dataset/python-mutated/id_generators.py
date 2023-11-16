import abc
import heapq
import logging
import threading
from collections import OrderedDict
from contextlib import contextmanager
from types import TracebackType
from typing import TYPE_CHECKING, AsyncContextManager, ContextManager, Dict, Generator, Generic, Iterable, List, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, cast
import attr
from sortedcontainers import SortedList, SortedSet
from synapse.metrics.background_process_metrics import run_as_background_process
from synapse.storage.database import DatabasePool, LoggingDatabaseConnection, LoggingTransaction
from synapse.storage.types import Cursor
from synapse.storage.util.sequence import PostgresSequenceGenerator
if TYPE_CHECKING:
    from synapse.notifier import ReplicationNotifier
logger = logging.getLogger(__name__)
T = TypeVar('T')

class IdGenerator:

    def __init__(self, db_conn: LoggingDatabaseConnection, table: str, column: str):
        if False:
            for i in range(10):
                print('nop')
        self._lock = threading.Lock()
        self._next_id = _load_current_id(db_conn, table, column)

    def get_next(self) -> int:
        if False:
            print('Hello World!')
        with self._lock:
            self._next_id += 1
            return self._next_id

def _load_current_id(db_conn: LoggingDatabaseConnection, table: str, column: str, step: int=1) -> int:
    if False:
        for i in range(10):
            print('nop')
    cur = db_conn.cursor(txn_name='_load_current_id')
    if step == 1:
        cur.execute('SELECT MAX(%s) FROM %s' % (column, table))
    else:
        cur.execute('SELECT MIN(%s) FROM %s' % (column, table))
    result = cur.fetchone()
    assert result is not None
    (val,) = result
    cur.close()
    current_id = int(val) if val else step
    res = (max if step > 0 else min)(current_id, step)
    logger.info('Initialising stream generator for %s(%s): %i', table, column, res)
    return res

class AbstractStreamIdGenerator(metaclass=abc.ABCMeta):
    """Generates or tracks stream IDs for a stream that may have multiple writers.

    Each stream ID represents a write transaction, whose completion is tracked
    so that the "current" stream ID of the stream can be determined.

    Stream IDs are monotonically increasing or decreasing integers representing write
    transactions. The "current" stream ID is the stream ID such that all transactions
    with equal or smaller stream IDs have completed. Since transactions may complete out
    of order, this is not the same as the stream ID of the last completed transaction.

    Completed transactions include both committed transactions and transactions that
    have been rolled back.
    """

    @abc.abstractmethod
    def advance(self, instance_name: str, new_id: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Advance the position of the named writer to the given ID, if greater\n        than existing entry.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def get_current_token(self) -> int:
        if False:
            print('Hello World!')
        'Returns the maximum stream id such that all stream ids less than or\n        equal to it have been successfully persisted.\n\n        Returns:\n            The maximum stream id.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def get_current_token_for_writer(self, instance_name: str) -> int:
        if False:
            i = 10
            return i + 15
        'Returns the position of the given writer.\n\n        For streams with single writers this is equivalent to `get_current_token`.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def get_minimal_local_current_token(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Tries to return a minimal current token for the local instance,\n        i.e. for writers this would be the last successful write.\n\n        If local instance is not a writer (or has written yet) then falls back\n        to returning the normal "current token".\n        '

    @abc.abstractmethod
    def get_next(self) -> AsyncContextManager[int]:
        if False:
            return 10
        '\n        Usage:\n            async with stream_id_gen.get_next() as stream_id:\n                # ... persist event ...\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def get_next_mult(self, n: int) -> AsyncContextManager[Sequence[int]]:
        if False:
            i = 10
            return i + 15
        '\n        Usage:\n            async with stream_id_gen.get_next(n) as stream_ids:\n                # ... persist events ...\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def get_next_txn(self, txn: LoggingTransaction) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Usage:\n            stream_id_gen.get_next_txn(txn)\n            # ... persist events ...\n        '
        raise NotImplementedError()

class StreamIdGenerator(AbstractStreamIdGenerator):
    """Generates and tracks stream IDs for a stream with a single writer.

    This class must only be used when the current Synapse process is the sole
    writer for a stream.

    Args:
        db_conn(connection):  A database connection to use to fetch the
            initial value of the generator from.
        table(str): A database table to read the initial value of the id
            generator from.
        column(str): The column of the database table to read the initial
            value from the id generator from.
        extra_tables(list): List of pairs of database tables and columns to
            use to source the initial value of the generator from. The value
            with the largest magnitude is used.
        step(int): which direction the stream ids grow in. +1 to grow
            upwards, -1 to grow downwards.

    Usage:
        async with stream_id_gen.get_next() as stream_id:
            # ... persist event ...
    """

    def __init__(self, db_conn: LoggingDatabaseConnection, notifier: 'ReplicationNotifier', table: str, column: str, extra_tables: Iterable[Tuple[str, str]]=(), step: int=1, is_writer: bool=True) -> None:
        if False:
            while True:
                i = 10
        assert step != 0
        self._lock = threading.Lock()
        self._step: int = step
        self._current: int = _load_current_id(db_conn, table, column, step)
        self._is_writer = is_writer
        for (table, column) in extra_tables:
            self._current = (max if step > 0 else min)(self._current, _load_current_id(db_conn, table, column, step))
        self._unfinished_ids: OrderedDict[int, int] = OrderedDict()
        self._notifier = notifier

    def advance(self, instance_name: str, new_id: int) -> None:
        if False:
            print('Hello World!')
        if self._is_writer:
            raise Exception('Replication is not supported by writer StreamIdGenerator')
        self._current = (max if self._step > 0 else min)(self._current, new_id)

    def get_next(self) -> AsyncContextManager[int]:
        if False:
            i = 10
            return i + 15
        with self._lock:
            self._current += self._step
            next_id = self._current
            self._unfinished_ids[next_id] = next_id

        @contextmanager
        def manager() -> Generator[int, None, None]:
            if False:
                print('Hello World!')
            try:
                yield next_id
            finally:
                with self._lock:
                    self._unfinished_ids.pop(next_id)
                self._notifier.notify_replication()
        return _AsyncCtxManagerWrapper(manager())

    def get_next_mult(self, n: int) -> AsyncContextManager[Sequence[int]]:
        if False:
            return 10
        with self._lock:
            next_ids = range(self._current + self._step, self._current + self._step * (n + 1), self._step)
            self._current += n * self._step
            for next_id in next_ids:
                self._unfinished_ids[next_id] = next_id

        @contextmanager
        def manager() -> Generator[Sequence[int], None, None]:
            if False:
                i = 10
                return i + 15
            try:
                yield next_ids
            finally:
                with self._lock:
                    for next_id in next_ids:
                        self._unfinished_ids.pop(next_id)
                self._notifier.notify_replication()
        return _AsyncCtxManagerWrapper(manager())

    def get_next_txn(self, txn: LoggingTransaction) -> int:
        if False:
            return 10
        '\n        Retrieve the next stream ID from within a database transaction.\n\n        Clean-up functions will be called when the transaction finishes.\n\n        Args:\n            txn: The database transaction object.\n\n        Returns:\n            The next stream ID.\n        '
        if not self._is_writer:
            raise Exception('Tried to allocate stream ID on non-writer')
        with self._lock:
            self._current += self._step
            next_id = self._current
            self._unfinished_ids[next_id] = next_id

        def clear_unfinished_id(id_to_clear: int) -> None:
            if False:
                return 10
            'A function to mark processing this ID as finished'
            with self._lock:
                self._unfinished_ids.pop(id_to_clear)
        txn.call_after(clear_unfinished_id, next_id)
        txn.call_on_exception(clear_unfinished_id, next_id)
        return next_id

    def get_current_token(self) -> int:
        if False:
            return 10
        if not self._is_writer:
            return self._current
        with self._lock:
            if self._unfinished_ids:
                return next(iter(self._unfinished_ids)) - self._step
            return self._current

    def get_current_token_for_writer(self, instance_name: str) -> int:
        if False:
            i = 10
            return i + 15
        return self.get_current_token()

    def get_minimal_local_current_token(self) -> int:
        if False:
            return 10
        return self.get_current_token()

class MultiWriterIdGenerator(AbstractStreamIdGenerator):
    """Generates and tracks stream IDs for a stream with multiple writers.

    Uses a Postgres sequence to coordinate ID assignment, but positions of other
    writers will only get updated when `advance` is called (by replication).

    Note: Only works with Postgres.

    Args:
        db_conn
        db
        stream_name: A name for the stream, for use in the `stream_positions`
            table. (Does not need to be the same as the replication stream name)
        instance_name: The name of this instance.
        tables: List of tables associated with the stream. Tuple of table
            name, column name that stores the writer's instance name, and
            column name that stores the stream ID.
        sequence_name: The name of the postgres sequence used to generate new
            IDs.
        writers: A list of known writers to use to populate current positions
            on startup. Can be empty if nothing uses `get_current_token` or
            `get_positions` (e.g. caches stream).
        positive: Whether the IDs are positive (true) or negative (false).
            When using negative IDs we go backwards from -1 to -2, -3, etc.
    """

    def __init__(self, db_conn: LoggingDatabaseConnection, db: DatabasePool, notifier: 'ReplicationNotifier', stream_name: str, instance_name: str, tables: List[Tuple[str, str, str]], sequence_name: str, writers: List[str], positive: bool=True) -> None:
        if False:
            return 10
        self._db = db
        self._notifier = notifier
        self._stream_name = stream_name
        self._instance_name = instance_name
        self._positive = positive
        self._writers = writers
        self._return_factor = 1 if positive else -1
        self._lock = threading.Lock()
        self._current_positions: Dict[str, int] = {}
        self._unfinished_ids: SortedSet[int] = SortedSet()
        self._in_flight_fetches: SortedList[int] = SortedList()
        self._finished_ids: Set[int] = set()
        self._persisted_upto_position = min(self._current_positions.values()) if self._current_positions else 1
        self._known_persisted_positions: List[int] = []
        self._max_seen_allocated_stream_id = 1
        self._max_position_of_local_instance = self._max_seen_allocated_stream_id
        self._sequence_gen = PostgresSequenceGenerator(sequence_name)
        for (table, _, id_column) in tables:
            self._sequence_gen.check_consistency(db_conn, table=table, id_column=id_column, stream_name=stream_name, positive=positive)
        self._load_current_ids(db_conn, tables)
        self._max_seen_allocated_stream_id = max(self._current_positions.values(), default=1)
        self._max_seen_allocated_stream_id = max(self._max_seen_allocated_stream_id, self._persisted_upto_position)
        self._max_position_of_local_instance = self._max_seen_allocated_stream_id
        if not writers:
            self._current_positions[self._instance_name] = self._persisted_upto_position

    def _load_current_ids(self, db_conn: LoggingDatabaseConnection, tables: List[Tuple[str, str, str]]) -> None:
        if False:
            print('Hello World!')
        cur = db_conn.cursor(txn_name='_load_current_ids')
        if self._writers:
            sql = '\n                DELETE FROM stream_positions\n                WHERE\n                    stream_name = ?\n                    AND instance_name != ALL(?)\n            '
            cur.execute(sql, (self._stream_name, self._writers))
            sql = '\n                SELECT instance_name, stream_id FROM stream_positions\n                WHERE stream_name = ?\n            '
            cur.execute(sql, (self._stream_name,))
            self._current_positions = {instance: stream_id * self._return_factor for (instance, stream_id) in cur if instance in self._writers}
        min_stream_id = min(self._current_positions.values(), default=None)
        if min_stream_id is None:
            max_stream_id = 1
            for (table, _, id_column) in tables:
                sql = '\n                    SELECT GREATEST(COALESCE(%(agg)s(%(id)s), 1), 1)\n                    FROM %(table)s\n                ' % {'id': id_column, 'table': table, 'agg': 'MAX' if self._positive else '-MIN'}
                cur.execute(sql)
                result = cur.fetchone()
                assert result is not None
                (stream_id,) = result
                max_stream_id = max(max_stream_id, stream_id)
            self._persisted_upto_position = max_stream_id
        else:
            self._persisted_upto_position = min_stream_id
            rows: List[Tuple[str, int]] = []
            for (table, instance_column, id_column) in tables:
                sql = '\n                    SELECT %(instance)s, %(id)s FROM %(table)s\n                    WHERE ? %(cmp)s %(id)s\n                ' % {'id': id_column, 'table': table, 'instance': instance_column, 'cmp': '<=' if self._positive else '>='}
                cur.execute(sql, (min_stream_id * self._return_factor,))
                rows.extend(cast(Iterable[Tuple[str, int]], cur))

            def sort_by_stream_id_key_func(row: Tuple[str, int]) -> int:
                if False:
                    for i in range(10):
                        print('nop')
                (instance, stream_id) = row
                return stream_id
            rows.sort(key=sort_by_stream_id_key_func)
            with self._lock:
                for (instance, stream_id) in rows:
                    stream_id = self._return_factor * stream_id
                    self._add_persisted_position(stream_id)
                    if instance == self._instance_name:
                        self._current_positions[instance] = stream_id
        if self._writers:
            for writer in self._writers:
                self._current_positions.setdefault(writer, self._persisted_upto_position)
        cur.close()

    def _load_next_id_txn(self, txn: Cursor) -> int:
        if False:
            i = 10
            return i + 15
        stream_ids = self._load_next_mult_id_txn(txn, 1)
        return stream_ids[0]

    def _load_next_mult_id_txn(self, txn: Cursor, n: int) -> List[int]:
        if False:
            return 10
        with self._lock:
            current_max = self._max_seen_allocated_stream_id
            self._in_flight_fetches.add(current_max)
        try:
            stream_ids = self._sequence_gen.get_next_mult_txn(txn, n)
            with self._lock:
                self._unfinished_ids.update(stream_ids)
                self._max_seen_allocated_stream_id = max(self._max_seen_allocated_stream_id, self._unfinished_ids[-1])
        finally:
            with self._lock:
                self._in_flight_fetches.remove(current_max)
        return stream_ids

    def get_next(self) -> AsyncContextManager[int]:
        if False:
            while True:
                i = 10
        if self._writers and self._instance_name not in self._writers:
            raise Exception('Tried to allocate stream ID on non-writer')
        return cast(AsyncContextManager[int], _MultiWriterCtxManager(self, self._notifier))

    def get_next_mult(self, n: int) -> AsyncContextManager[List[int]]:
        if False:
            while True:
                i = 10
        if self._writers and self._instance_name not in self._writers:
            raise Exception('Tried to allocate stream ID on non-writer')
        return cast(AsyncContextManager[List[int]], _MultiWriterCtxManager(self, self._notifier, n))

    def get_next_txn(self, txn: LoggingTransaction) -> int:
        if False:
            print('Hello World!')
        '\n        Usage:\n\n            stream_id = stream_id_gen.get_next_txn(txn)\n            # ... persist event ...\n        '
        if self._writers and self._instance_name not in self._writers:
            raise Exception('Tried to allocate stream ID on non-writer')
        next_id = self._load_next_id_txn(txn)
        txn.call_after(self._mark_ids_as_finished, [next_id])
        txn.call_on_exception(self._mark_ids_as_finished, [next_id])
        txn.call_after(self._notifier.notify_replication)
        if self._writers:
            txn.call_after(run_as_background_process, 'MultiWriterIdGenerator._update_table', self._db.runInteraction, 'MultiWriterIdGenerator._update_table', self._update_stream_positions_table_txn)
        return self._return_factor * next_id

    def get_next_mult_txn(self, txn: LoggingTransaction, n: int) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Usage:\n\n            stream_id = stream_id_gen.get_next_txn(txn)\n            # ... persist event ...\n        '
        if self._writers and self._instance_name not in self._writers:
            raise Exception('Tried to allocate stream ID on non-writer')
        next_ids = self._load_next_mult_id_txn(txn, n)
        txn.call_after(self._mark_ids_as_finished, next_ids)
        txn.call_on_exception(self._mark_ids_as_finished, next_ids)
        txn.call_after(self._notifier.notify_replication)
        if self._writers:
            txn.call_after(run_as_background_process, 'MultiWriterIdGenerator._update_table', self._db.runInteraction, 'MultiWriterIdGenerator._update_table', self._update_stream_positions_table_txn)
        return [self._return_factor * next_id for next_id in next_ids]

    def _mark_ids_as_finished(self, next_ids: List[int]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'These IDs have finished being processed so we should advance the\n        current position if possible.\n        '
        with self._lock:
            self._unfinished_ids.difference_update(next_ids)
            self._finished_ids.update(next_ids)
            new_cur: Optional[int] = None
            if self._unfinished_ids or self._in_flight_fetches:
                if self._unfinished_ids and self._in_flight_fetches:
                    min_unfinished = min(self._unfinished_ids[0], self._in_flight_fetches[0] + 1)
                elif self._in_flight_fetches:
                    min_unfinished = self._in_flight_fetches[0] + 1
                else:
                    min_unfinished = self._unfinished_ids[0]
                finished = set()
                for s in self._finished_ids:
                    if s < min_unfinished:
                        if new_cur is None or new_cur < s:
                            new_cur = s
                    else:
                        finished.add(s)
                self._finished_ids = finished
            else:
                new_cur = max(self._finished_ids)
                self._finished_ids.clear()
            if new_cur:
                curr = self._current_positions.get(self._instance_name, 0)
                self._current_positions[self._instance_name] = max(curr, new_cur)
                self._max_position_of_local_instance = max(curr, new_cur, self._max_position_of_local_instance)
            for next_id in next_ids:
                self._add_persisted_position(next_id)

    def get_current_token(self) -> int:
        if False:
            while True:
                i = 10
        return self.get_persisted_upto_position()

    def get_current_token_for_writer(self, instance_name: str) -> int:
        if False:
            while True:
                i = 10
        with self._lock:
            if self._instance_name == instance_name:
                return self._return_factor * self._max_position_of_local_instance
            pos = self._current_positions.get(instance_name, self._persisted_upto_position)
            pos = max(pos, self._persisted_upto_position)
            return self._return_factor * pos

    def get_minimal_local_current_token(self) -> int:
        if False:
            i = 10
            return i + 15
        with self._lock:
            return self._return_factor * self._current_positions.get(self._instance_name, self._persisted_upto_position)

    def get_positions(self) -> Dict[str, int]:
        if False:
            i = 10
            return i + 15
        "Get a copy of the current positon map.\n\n        Note that this won't necessarily include all configured writers if some\n        writers haven't written anything yet.\n        "
        with self._lock:
            return {name: self._return_factor * i for (name, i) in self._current_positions.items()}

    def advance(self, instance_name: str, new_id: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        new_id *= self._return_factor
        with self._lock:
            self._current_positions[instance_name] = max(new_id, self._current_positions.get(instance_name, 0))
            self._max_seen_allocated_stream_id = max(self._max_seen_allocated_stream_id, new_id)
            self._add_persisted_position(new_id)

    def get_persisted_upto_position(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        "Get the max position where all previous positions have been\n        persisted.\n\n        Note: In the worst case scenario this will be equal to the minimum\n        position across writers. This means that the returned position here can\n        lag if one writer doesn't write very often.\n        "
        with self._lock:
            return self._return_factor * self._persisted_upto_position

    def _add_persisted_position(self, new_id: int) -> None:
        if False:
            while True:
                i = 10
        'Record that we have persisted a position.\n\n        This is used to keep the `_current_positions` up to date.\n        '
        assert self._lock.locked()
        heapq.heappush(self._known_persisted_positions, new_id)
        our_current_position = self._current_positions.get(self._instance_name, 0)
        min_curr = min((token for (name, token) in self._current_positions.items() if name != self._instance_name), default=our_current_position)
        if our_current_position and (self._unfinished_ids or self._in_flight_fetches):
            min_curr = min(min_curr, our_current_position)
        self._persisted_upto_position = max(min_curr, self._persisted_upto_position)
        self._max_position_of_local_instance = max(self._max_position_of_local_instance, self._persisted_upto_position)
        if not self._unfinished_ids and (not self._in_flight_fetches):
            self._max_position_of_local_instance = max(self._max_seen_allocated_stream_id, self._max_position_of_local_instance)
        while self._known_persisted_positions:
            if self._known_persisted_positions[0] <= self._persisted_upto_position:
                heapq.heappop(self._known_persisted_positions)
            elif self._known_persisted_positions[0] == self._persisted_upto_position + 1:
                heapq.heappop(self._known_persisted_positions)
                self._persisted_upto_position += 1
            else:
                break

    def _update_stream_positions_table_txn(self, txn: Cursor) -> None:
        if False:
            while True:
                i = 10
        'Update the `stream_positions` table with newly persisted position.'
        if not self._writers:
            return
        sql = '\n            INSERT INTO stream_positions (stream_name, instance_name, stream_id)\n            VALUES (?, ?, ?)\n            ON CONFLICT (stream_name, instance_name)\n            DO UPDATE SET\n                stream_id = %(agg)s(stream_positions.stream_id, EXCLUDED.stream_id)\n        ' % {'agg': 'GREATEST' if self._positive else 'LEAST'}
        pos = (self.get_current_token_for_writer(self._instance_name),)
        txn.execute(sql, (self._stream_name, self._instance_name, pos))

@attr.s(frozen=True, auto_attribs=True)
class _AsyncCtxManagerWrapper(Generic[T]):
    """Helper class to convert a plain context manager to an async one.

    This is mainly useful if you have a plain context manager but the interface
    requires an async one.
    """
    inner: ContextManager[T]

    async def __aenter__(self) -> T:
        return self.inner.__enter__()

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]) -> Optional[bool]:
        return self.inner.__exit__(exc_type, exc, tb)

@attr.s(slots=True, auto_attribs=True)
class _MultiWriterCtxManager:
    """Async context manager returned by MultiWriterIdGenerator"""
    id_gen: MultiWriterIdGenerator
    notifier: 'ReplicationNotifier'
    multiple_ids: Optional[int] = None
    stream_ids: List[int] = attr.Factory(list)

    async def __aenter__(self) -> Union[int, List[int]]:
        self.stream_ids = await self.id_gen._db.runInteraction('_load_next_mult_id', self.id_gen._load_next_mult_id_txn, self.multiple_ids or 1, db_autocommit=True)
        if self.multiple_ids is None:
            return self.stream_ids[0] * self.id_gen._return_factor
        else:
            return [i * self.id_gen._return_factor for i in self.stream_ids]

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]) -> bool:
        self.id_gen._mark_ids_as_finished(self.stream_ids)
        self.notifier.notify_replication()
        if exc_type is not None:
            return False
        if self.id_gen._writers:
            await self.id_gen._db.runInteraction('MultiWriterIdGenerator._update_table', self.id_gen._update_stream_positions_table_txn, db_autocommit=True)
        return False