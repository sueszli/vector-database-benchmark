from twisted.internet import defer
from twisted.test.proto_helpers import MemoryReactor
from synapse.server import HomeServer
from synapse.util import Clock
from tests import unittest
from tests.replication._base import BaseMultiWorkerStreamTestCase

class WorkerLockTestCase(unittest.HomeserverTestCase):

    def prepare(self, reactor: MemoryReactor, clock: Clock, homeserver: HomeServer) -> None:
        if False:
            return 10
        self.worker_lock_handler = self.hs.get_worker_locks_handler()

    def test_wait_for_lock_locally(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test waiting for a lock on a single worker'
        lock1 = self.worker_lock_handler.acquire_lock('name', 'key')
        self.get_success(lock1.__aenter__())
        lock2 = self.worker_lock_handler.acquire_lock('name', 'key')
        d2 = defer.ensureDeferred(lock2.__aenter__())
        self.assertNoResult(d2)
        self.get_success(lock1.__aexit__(None, None, None))
        self.get_success(d2)
        self.get_success(lock2.__aexit__(None, None, None))

class WorkerLockWorkersTestCase(BaseMultiWorkerStreamTestCase):

    def prepare(self, reactor: MemoryReactor, clock: Clock, homeserver: HomeServer) -> None:
        if False:
            while True:
                i = 10
        self.main_worker_lock_handler = self.hs.get_worker_locks_handler()

    def test_wait_for_lock_worker(self) -> None:
        if False:
            print('Hello World!')
        'Test waiting for a lock on another worker'
        worker = self.make_worker_hs('synapse.app.generic_worker', extra_config={'redis': {'enabled': True}})
        worker_lock_handler = worker.get_worker_locks_handler()
        lock1 = self.main_worker_lock_handler.acquire_lock('name', 'key')
        self.get_success(lock1.__aenter__())
        lock2 = worker_lock_handler.acquire_lock('name', 'key')
        d2 = defer.ensureDeferred(lock2.__aenter__())
        self.assertNoResult(d2)
        self.get_success(lock1.__aexit__(None, None, None))
        self.get_success(d2)
        self.get_success(lock2.__aexit__(None, None, None))