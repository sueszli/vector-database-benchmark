import datetime
from huey.api import Huey
from huey.api import MemoryHuey
from huey.exceptions import TaskException
from huey.storage import BlackHoleStorage
from huey.tests.base import BaseTestCase

class TestImmediate(BaseTestCase):

    def get_huey(self):
        if False:
            while True:
                i = 10
        return MemoryHuey(immediate=True, utc=False)

    def test_immediate(self):
        if False:
            print('Hello World!')

        @self.huey.task()
        def task_a(n):
            if False:
                for i in range(10):
                    print('nop')
            return n + 1
        r = task_a(3)
        self.assertEqual(len(self.huey), 0)
        self.assertEqual(self.huey.result_count(), 1)
        self.assertEqual(r.get(), 4)
        self.assertEqual(self.huey.result_count(), 0)
        r_err = task_a(None)
        self.assertRaises(TaskException, r_err.get)

    def test_immediate_pipeline(self):
        if False:
            i = 10
            return i + 15

        @self.huey.task()
        def add(a, b):
            if False:
                i = 10
                return i + 15
            return a + b
        p = add.s(3, 4).then(add, 5).then(add, 6).then(add, 7)
        result_group = self.huey.enqueue(p)
        self.assertEqual(result_group(), [7, 12, 18, 25])

    def test_immediate_scheduling(self):
        if False:
            return 10

        @self.huey.task()
        def task_a(n):
            if False:
                print('Hello World!')
            return n + 1
        r = task_a.schedule((3,), delay=10)
        self.assertEqual(len(self.huey), 0)
        self.assertEqual(self.huey.result_count(), 0)
        self.assertEqual(self.huey.scheduled_count(), 1)
        self.assertTrue(r.get() is None)

    def test_immediate_reschedule(self):
        if False:
            print('Hello World!')
        state = []

        @self.huey.task(context=True)
        def task_s(task=None):
            if False:
                for i in range(10):
                    print('nop')
            state.append(task.id)
            return 1
        r = task_s.schedule(delay=60)
        self.assertEqual(len(self.huey), 0)
        self.assertTrue(r() is None)
        r2 = r.reschedule()
        self.assertTrue(r.id != r2.id)
        self.assertEqual(state, [r2.id])
        self.assertEqual(r2(), 1)
        self.assertEqual(len(self.huey), 0)
        self.assertTrue(r.is_revoked())
        self.assertEqual(self.huey.result_count(), 1)
        self.assertEqual(self.huey.scheduled_count(), 1)

    def test_immediate_revoke_restore(self):
        if False:
            while True:
                i = 10

        @self.huey.task()
        def task_a(n):
            if False:
                i = 10
                return i + 15
            return n + 1
        task_a.revoke()
        r = task_a(3)
        self.assertEqual(len(self.huey), 0)
        self.assertTrue(r.get() is None)
        self.assertTrue(task_a.restore())
        r = task_a(4)
        self.assertEqual(r.get(), 5)

    def test_swap_immediate(self):
        if False:
            return 10

        @self.huey.task()
        def task_a(n):
            if False:
                for i in range(10):
                    print('nop')
            return n + 1
        r = task_a(1)
        self.assertEqual(r.get(), 2)
        self.huey.immediate = False
        r = task_a(2)
        self.assertEqual(len(self.huey), 1)
        self.assertEqual(self.huey.result_count(), 0)
        task = self.huey.dequeue()
        self.assertEqual(self.huey.execute(task), 3)
        self.assertEqual(r.get(), 3)
        self.huey.immediate = True
        r = task_a(3)
        self.assertEqual(r.get(), 4)
        self.assertEqual(len(self.huey), 0)
        self.assertEqual(self.huey.result_count(), 0)

    def test_map(self):
        if False:
            i = 10
            return i + 15

        @self.huey.task()
        def task_a(n):
            if False:
                for i in range(10):
                    print('nop')
            return n + 1
        result_group = task_a.map(range(8))
        self.assertEqual(result_group(), [1, 2, 3, 4, 5, 6, 7, 8])

class NoUseException(Exception):
    pass

class NoUseStorage(BlackHoleStorage):

    def enqueue(self, data, priority=None):
        if False:
            while True:
                i = 10
        raise NoUseException()

    def dequeue(self):
        if False:
            for i in range(10):
                print('nop')
        raise NoUseException()

    def add_to_schedule(self, data, ts, utc):
        if False:
            i = 10
            return i + 15
        raise NoUseException()

    def read_schedule(self, ts):
        if False:
            for i in range(10):
                print('nop')
        raise NoUseException()

    def put_data(self, key, value):
        if False:
            while True:
                i = 10
        raise NoUseException()

    def peek_data(self, key):
        if False:
            print('Hello World!')
        raise NoUseException()

    def pop_data(self, key):
        if False:
            print('Hello World!')
        raise NoUseException()

    def has_data_for_key(self, key):
        if False:
            i = 10
            return i + 15
        raise NoUseException()

    def put_if_empty(self, key, value):
        if False:
            print('Hello World!')
        raise NoUseException()

class NoUseHuey(Huey):

    def get_storage(self, **storage_kwargs):
        if False:
            return 10
        return NoUseStorage()

class TestImmediateMemoryStorage(BaseTestCase):

    def get_huey(self):
        if False:
            print('Hello World!')
        return NoUseHuey(utc=False)

    def test_immediate_storage(self):
        if False:
            while True:
                i = 10

        @self.huey.task()
        def task_a(n):
            if False:
                return 10
            return n + 1
        self.huey.immediate = True
        res = task_a(2)
        self.assertEqual(res(), 3)
        task_a.revoke()
        res = task_a(3)
        self.assertTrue(res() is None)
        self.assertTrue(task_a.restore())
        res = task_a(4)
        self.assertEqual(res(), 5)
        eta = datetime.datetime.now() + datetime.timedelta(seconds=60)
        res = task_a.schedule((5,), eta=eta)
        self.assertTrue(res() is None)
        minus_1 = eta - datetime.timedelta(seconds=1)
        self.assertEqual(self.huey.read_schedule(minus_1), [])
        tasks = self.huey.read_schedule(eta)
        self.assertEqual([t.id for t in tasks], [res.id])
        self.assertTrue(res() is None)
        self.huey.immediate = False
        self.assertRaises(NoUseException, task_a, 1)
        self.huey.immediate = True
        res = task_a(10)
        self.assertEqual(res(), 11)

    def test_immediate_real_storage(self):
        if False:
            return 10
        self.huey.immediate_use_memory = False

        @self.huey.task()
        def task_a(n):
            if False:
                return 10
            return n + 1
        self.huey.immediate = True
        self.assertRaises(NoUseException, task_a, 1)
        self.huey.immediate = False
        self.assertRaises(NoUseException, task_a, 2)