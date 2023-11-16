from __future__ import absolute_import
import unittest2
import uuid
from oslo_config import cfg
from st2common.services import coordination
import st2tests.config as tests_config

class SynchronizationTest(unittest2.TestCase):
    coordinator = None

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        super(SynchronizationTest, cls).setUpClass()
        tests_config.parse_args(coordinator_noop=False)
        cls.coordinator = coordination.get_coordinator(use_cache=False)

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        coordination.coordinator_teardown(cls.coordinator)
        coordination.COORDINATOR = None
        super(SynchronizationTest, cls).tearDownClass()

    def test_service_configured(self):
        if False:
            for i in range(10):
                print('nop')
        cfg.CONF.set_override(name='url', override=None, group='coordination')
        self.assertEqual(coordination.get_driver_name(), None)
        cfg.CONF.set_override(name='url', override='kazoo://127.0.0.1:2181', group='coordination')
        self.assertTrue(coordination.configured())
        self.assertEqual(coordination.get_driver_name(), 'kazoo')
        cfg.CONF.set_override(name='url', override='file:///tmp', group='coordination')
        self.assertFalse(coordination.configured())
        self.assertEqual(coordination.get_driver_name(), 'file')
        cfg.CONF.set_override(name='url', override='zake://', group='coordination')
        self.assertFalse(coordination.configured())
        self.assertEqual(coordination.get_driver_name(), 'zake')
        cfg.CONF.set_override(name='url', override='redis://foo:bar@127.0.0.1', group='coordination')
        self.assertTrue(coordination.configured())
        self.assertEqual(coordination.get_driver_name(), 'redis')

    def test_lock(self):
        if False:
            print('Hello World!')
        name = uuid.uuid4().hex
        lock = self.coordinator.get_lock(name)
        self.assertTrue(lock.acquire())
        lock.release()

    def test_multiple_acquire(self):
        if False:
            for i in range(10):
                print('nop')
        name = uuid.uuid4().hex
        lock1 = self.coordinator.get_lock(name)
        self.assertTrue(lock1.acquire())
        lock2 = self.coordinator.get_lock(name)
        self.assertFalse(lock2.acquire(blocking=False))
        lock1.release()
        self.assertTrue(lock2.acquire())
        lock2.release()

    def test_lock_expiry_on_session_close(self):
        if False:
            return 10
        name = uuid.uuid4().hex
        lock1 = self.coordinator.get_lock(name)
        self.assertTrue(lock1.acquire())
        lock2 = self.coordinator.get_lock(name)
        self.assertFalse(lock2.acquire(blocking=False))
        self.coordinator.stop()
        self.coordinator.start()
        lock3 = self.coordinator.get_lock(name)
        self.assertTrue(lock3.acquire())
        lock3.release()