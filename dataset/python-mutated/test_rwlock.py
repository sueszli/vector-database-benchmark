from typing import AsyncContextManager, Callable, Sequence, Tuple
from twisted.internet import defer
from twisted.internet.defer import CancelledError, Deferred
from synapse.util.async_helpers import ReadWriteLock
from tests import unittest

class ReadWriteLockTestCase(unittest.TestCase):

    def _start_reader_or_writer(self, read_or_write: Callable[[str], AsyncContextManager], key: str, return_value: str) -> Tuple['Deferred[str]', 'Deferred[None]', 'Deferred[None]']:
        if False:
            print('Hello World!')
        'Starts a reader or writer which acquires the lock, blocks, then completes.\n\n        Args:\n            read_or_write: A function returning a context manager for a lock.\n                Either a bound `ReadWriteLock.read` or `ReadWriteLock.write`.\n            key: The key to read or write.\n            return_value: A string that the reader or writer will resolve with when\n                done.\n\n        Returns:\n            A tuple of three `Deferred`s:\n             * A cancellable `Deferred` for the entire read or write operation that\n               resolves with `return_value` on successful completion.\n             * A `Deferred` that resolves once the reader or writer acquires the lock.\n             * A `Deferred` that blocks the reader or writer. Must be resolved by the\n               caller to allow the reader or writer to release the lock and complete.\n        '
        acquired_d: 'Deferred[None]' = Deferred()
        unblock_d: 'Deferred[None]' = Deferred()

        async def reader_or_writer() -> str:
            async with read_or_write(key):
                acquired_d.callback(None)
                await unblock_d
            return return_value
        d = defer.ensureDeferred(reader_or_writer())
        return (d, acquired_d, unblock_d)

    def _start_blocking_reader(self, rwlock: ReadWriteLock, key: str, return_value: str) -> Tuple['Deferred[str]', 'Deferred[None]', 'Deferred[None]']:
        if False:
            print('Hello World!')
        'Starts a reader which acquires the lock, blocks, then releases the lock.\n\n        See the docstring for `_start_reader_or_writer` for details about the arguments\n        and return values.\n        '
        return self._start_reader_or_writer(rwlock.read, key, return_value)

    def _start_blocking_writer(self, rwlock: ReadWriteLock, key: str, return_value: str) -> Tuple['Deferred[str]', 'Deferred[None]', 'Deferred[None]']:
        if False:
            for i in range(10):
                print('nop')
        'Starts a writer which acquires the lock, blocks, then releases the lock.\n\n        See the docstring for `_start_reader_or_writer` for details about the arguments\n        and return values.\n        '
        return self._start_reader_or_writer(rwlock.write, key, return_value)

    def _start_nonblocking_reader(self, rwlock: ReadWriteLock, key: str, return_value: str) -> Tuple['Deferred[str]', 'Deferred[None]']:
        if False:
            for i in range(10):
                print('nop')
        'Starts a reader which acquires the lock, then releases it immediately.\n\n        See the docstring for `_start_reader_or_writer` for details about the arguments.\n\n        Returns:\n            A tuple of two `Deferred`s:\n             * A cancellable `Deferred` for the entire read operation that resolves with\n               `return_value` on successful completion.\n             * A `Deferred` that resolves once the reader acquires the lock.\n        '
        (d, acquired_d, unblock_d) = self._start_reader_or_writer(rwlock.read, key, return_value)
        unblock_d.callback(None)
        return (d, acquired_d)

    def _start_nonblocking_writer(self, rwlock: ReadWriteLock, key: str, return_value: str) -> Tuple['Deferred[str]', 'Deferred[None]']:
        if False:
            print('Hello World!')
        'Starts a writer which acquires the lock, then releases it immediately.\n\n        See the docstring for `_start_reader_or_writer` for details about the arguments.\n\n        Returns:\n            A tuple of two `Deferred`s:\n             * A cancellable `Deferred` for the entire write operation that resolves\n               with `return_value` on successful completion.\n             * A `Deferred` that resolves once the writer acquires the lock.\n        '
        (d, acquired_d, unblock_d) = self._start_reader_or_writer(rwlock.write, key, return_value)
        unblock_d.callback(None)
        return (d, acquired_d)

    def _assert_first_n_resolved(self, deferreds: Sequence['defer.Deferred[None]'], n: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Assert that exactly the first n `Deferred`s in the given list are resolved.\n\n        Args:\n            deferreds: The list of `Deferred`s to be checked.\n            n: The number of `Deferred`s at the start of `deferreds` that should be\n                resolved.\n        '
        for (i, d) in enumerate(deferreds[:n]):
            self.assertTrue(d.called, msg='deferred %d was unexpectedly unresolved' % i)
        for (i, d) in enumerate(deferreds[n:]):
            self.assertFalse(d.called, msg='deferred %d was unexpectedly resolved' % (i + n))

    def test_rwlock(self) -> None:
        if False:
            while True:
                i = 10
        rwlock = ReadWriteLock()
        key = 'key'
        ds = [self._start_blocking_reader(rwlock, key, '0'), self._start_blocking_reader(rwlock, key, '1'), self._start_blocking_writer(rwlock, key, '2'), self._start_blocking_writer(rwlock, key, '3'), self._start_blocking_reader(rwlock, key, '4'), self._start_blocking_reader(rwlock, key, '5'), self._start_blocking_writer(rwlock, key, '6')]
        acquired_ds = [acquired_d for (_, acquired_d, _) in ds]
        release_ds = [release_d for (_, _, release_d) in ds]
        self._assert_first_n_resolved(acquired_ds, 2)
        self._assert_first_n_resolved(acquired_ds, 2)
        release_ds[0].callback(None)
        self._assert_first_n_resolved(acquired_ds, 2)
        self._assert_first_n_resolved(acquired_ds, 2)
        release_ds[1].callback(None)
        self._assert_first_n_resolved(acquired_ds, 3)
        self._assert_first_n_resolved(acquired_ds, 3)
        release_ds[2].callback(None)
        self._assert_first_n_resolved(acquired_ds, 4)
        self._assert_first_n_resolved(acquired_ds, 4)
        release_ds[3].callback(None)
        self._assert_first_n_resolved(acquired_ds, 6)
        self._assert_first_n_resolved(acquired_ds, 6)
        release_ds[5].callback(None)
        self._assert_first_n_resolved(acquired_ds, 6)
        self._assert_first_n_resolved(acquired_ds, 6)
        release_ds[4].callback(None)
        self._assert_first_n_resolved(acquired_ds, 7)
        release_ds[6].callback(None)
        (_, acquired_d) = self._start_nonblocking_writer(rwlock, key, 'last writer')
        self.assertTrue(acquired_d.called)
        (_, acquired_d) = self._start_nonblocking_reader(rwlock, key, 'last reader')
        self.assertTrue(acquired_d.called)

    def test_lock_handoff_to_nonblocking_writer(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test a writer handing the lock to another writer that completes instantly.'
        rwlock = ReadWriteLock()
        key = 'key'
        (d1, _, unblock) = self._start_blocking_writer(rwlock, key, 'write 1 completed')
        (d2, _) = self._start_nonblocking_writer(rwlock, key, 'write 2 completed')
        self.assertFalse(d1.called)
        self.assertFalse(d2.called)
        unblock.callback(None)
        self.assertTrue(d1.called)
        self.assertTrue(d2.called)
        (d3, _) = self._start_nonblocking_writer(rwlock, key, 'write 3 completed')
        self.assertTrue(d3.called)

    def test_cancellation_while_holding_read_lock(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test cancellation while holding a read lock.\n\n        A waiting writer should be given the lock when the reader holding the lock is\n        cancelled.\n        '
        rwlock = ReadWriteLock()
        key = 'key'
        (reader_d, _, _) = self._start_blocking_reader(rwlock, key, 'read completed')
        (writer_d, _) = self._start_nonblocking_writer(rwlock, key, 'write completed')
        self.assertFalse(writer_d.called)
        reader_d.cancel()
        self.failureResultOf(reader_d, CancelledError)
        self.assertTrue(writer_d.called, 'Writer is stuck waiting for a cancelled reader')
        self.assertEqual('write completed', self.successResultOf(writer_d))

    def test_cancellation_while_holding_write_lock(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test cancellation while holding a write lock.\n\n        A waiting reader should be given the lock when the writer holding the lock is\n        cancelled.\n        '
        rwlock = ReadWriteLock()
        key = 'key'
        (writer_d, _, _) = self._start_blocking_writer(rwlock, key, 'write completed')
        (reader_d, _) = self._start_nonblocking_reader(rwlock, key, 'read completed')
        self.assertFalse(reader_d.called)
        writer_d.cancel()
        self.failureResultOf(writer_d, CancelledError)
        self.assertTrue(reader_d.called, 'Reader is stuck waiting for a cancelled writer')
        self.assertEqual('read completed', self.successResultOf(reader_d))

    def test_cancellation_while_waiting_for_read_lock(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test cancellation while waiting for a read lock.\n\n        Tests that cancelling a waiting reader:\n         * does not cancel the writer it is waiting on\n         * does not cancel the next writer waiting on it\n         * does not allow the next writer to acquire the lock before an earlier writer\n           has finished\n         * does not keep the next writer waiting indefinitely\n\n        These correspond to the asserts with explicit messages.\n        '
        rwlock = ReadWriteLock()
        key = 'key'
        (writer1_d, _, unblock_writer1) = self._start_blocking_writer(rwlock, key, 'write 1 completed')
        (reader_d, _) = self._start_nonblocking_reader(rwlock, key, 'read completed')
        self.assertFalse(reader_d.called)
        (writer2_d, _) = self._start_nonblocking_writer(rwlock, key, 'write 2 completed')
        self.assertFalse(writer2_d.called)
        reader_d.cancel()
        self.failureResultOf(reader_d, CancelledError)
        self.assertFalse(writer1_d.called, 'First writer was unexpectedly cancelled')
        self.assertFalse(writer2_d.called, 'Second writer was unexpectedly cancelled or given the lock before the first writer finished')
        unblock_writer1.callback(None)
        self.assertEqual('write 1 completed', self.successResultOf(writer1_d))
        self.assertTrue(writer2_d.called, 'Second writer is stuck waiting for a cancelled reader')
        self.assertEqual('write 2 completed', self.successResultOf(writer2_d))

    def test_cancellation_while_waiting_for_write_lock(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test cancellation while waiting for a write lock.\n\n        Tests that cancelling a waiting writer:\n         * does not cancel the reader or writer it is waiting on\n         * does not cancel the next writer waiting on it\n         * does not allow the next writer to acquire the lock before an earlier reader\n           and writer have finished\n         * does not keep the next writer waiting indefinitely\n\n        These correspond to the asserts with explicit messages.\n        '
        rwlock = ReadWriteLock()
        key = 'key'
        (reader_d, _, unblock_reader) = self._start_blocking_reader(rwlock, key, 'read completed')
        (writer1_d, _, unblock_writer1) = self._start_blocking_writer(rwlock, key, 'write 1 completed')
        (writer2_d, _) = self._start_nonblocking_writer(rwlock, key, 'write 2 completed')
        self.assertFalse(writer2_d.called)
        (writer3_d, _) = self._start_nonblocking_writer(rwlock, key, 'write 3 completed')
        self.assertFalse(writer3_d.called)
        writer2_d.cancel()
        self.assertNoResult(writer2_d)
        self.assertFalse(reader_d.called, 'Reader was unexpectedly cancelled')
        self.assertFalse(writer1_d.called, 'First writer was unexpectedly cancelled')
        self.assertFalse(writer3_d.called, 'Third writer was unexpectedly cancelled or given the lock before the first writer finished')
        unblock_reader.callback(None)
        self.assertEqual('read completed', self.successResultOf(reader_d))
        self.assertNoResult(writer2_d)
        self.assertFalse(writer3_d.called, 'Third writer was unexpectedly given the lock before the first writer finished')
        unblock_writer1.callback(None)
        self.assertEqual('write 1 completed', self.successResultOf(writer1_d))
        self.failureResultOf(writer2_d, CancelledError)
        self.assertTrue(writer3_d.called, 'Third writer is stuck waiting for a cancelled writer')
        self.assertEqual('write 3 completed', self.successResultOf(writer3_d))