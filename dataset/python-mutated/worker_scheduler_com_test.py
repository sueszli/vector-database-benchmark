import os
import time
import threading
import tempfile
import shutil
import contextlib
import luigi
from luigi.scheduler import Scheduler
from luigi.worker import Worker
from helpers import LuigiTestCase

class WorkerSchedulerCommunicationTest(LuigiTestCase):
    """
    Tests related to communication between Worker and Scheduler that is based on the ping polling.

    See https://github.com/spotify/luigi/pull/1993
    """

    def run(self, result=None):
        if False:
            return 10
        self.sch = Scheduler()
        with Worker(scheduler=self.sch, worker_id='X', ping_interval=1, max_reschedules=0) as w:
            self.w = w
            self.sw = self.sch._state.get_worker(self.w._id)
            super(WorkerSchedulerCommunicationTest, self).run(result)

    def wrapper_task(test_self):
        if False:
            return 10
        tmp = tempfile.mkdtemp()

        class MyTask(luigi.Task):
            n = luigi.IntParameter()
            delay = 3

            def output(self):
                if False:
                    while True:
                        i = 10
                basename = '%s_%s.txt' % (self.__class__.__name__, self.n)
                return luigi.LocalTarget(os.path.join(tmp, basename))

            def run(self):
                if False:
                    print('Hello World!')
                time.sleep(self.delay)
                with self.output().open('w') as f:
                    f.write('content\n')

        class Wrapper(MyTask):
            delay = 0

            def requires(self):
                if False:
                    print('Hello World!')
                return [MyTask(n=n) for n in range(self.n)]
        return (Wrapper, tmp)

    def test_message_handling(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(10):
            self.sw.add_rpc_message('foo', i=i)
        self.assertEqual(10, len(self.sw.rpc_messages))
        self.assertEqual(9, self.sw.rpc_messages[-1]['kwargs']['i'])
        msgs = self.sw.fetch_rpc_messages()
        self.assertEqual(0, len(self.sw.rpc_messages))
        self.assertEqual(9, msgs[-1]['kwargs']['i'])

    def test_ping_content(self):
        if False:
            print('Hello World!')
        for i in range(10):
            self.sw.add_rpc_message('foo', i=i)
        res = self.sch.ping(worker=self.w._id)
        self.assertIn('rpc_messages', res)
        msgs = res['rpc_messages']
        self.assertEqual(10, len(msgs))
        self.assertEqual('foo', msgs[-1]['name'])
        self.assertEqual(9, msgs[-1]['kwargs']['i'])
        self.assertEqual(0, len(self.sw.rpc_messages))

    @contextlib.contextmanager
    def run_wrapper(self, n):
        if False:
            for i in range(10):
                print('nop')
        (Wrapper, tmp) = self.wrapper_task()
        wrapper = Wrapper(n=n)
        self.assertTrue(self.w.add(wrapper))
        self.assertEqual(1, self.w.worker_processes)
        t = threading.Thread(target=self.w.run)
        t.start()
        yield (wrapper, t)
        self.assertFalse(t.is_alive())
        shutil.rmtree(tmp)

    def test_dispatch_valid_message(self):
        if False:
            return 10
        with self.run_wrapper(3) as (wrapper, t):
            t.join(1)
            self.sch.set_worker_processes(self.w._id, 2)
            t.join(3)
            self.assertEqual(2, self.w.worker_processes)
            t.join(3)
            self.assertTrue(all((task.complete() for task in wrapper.requires())))
            self.assertTrue(wrapper.complete())

    def test_dispatch_invalid_message(self):
        if False:
            return 10
        with self.run_wrapper(2) as (wrapper, t):
            t.join(1)
            self.sw.add_rpc_message('set_worker_processes_not_there', n=2)
            t.join(3)
            self.assertEqual(1, self.w.worker_processes)
            t.join(3)
            self.assertTrue(all((task.complete() for task in wrapper.requires())))
            self.assertTrue(wrapper.complete())

    def test_dispatch_unregistered_message(self):
        if False:
            i = 10
            return i + 15
        set_worker_processes_orig = self.w.set_worker_processes

        def set_worker_processes_replacement(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return set_worker_processes_orig(*args, **kwargs)
        self.w.set_worker_processes = set_worker_processes_replacement
        self.assertFalse(getattr(self.w.set_worker_processes, 'is_rpc_message_callback', False))
        with self.run_wrapper(2) as (wrapper, t):
            t.join(1)
            self.sw.add_rpc_message('set_worker_processes', n=2)
            t.join(3)
            self.assertEqual(1, self.w.worker_processes)
            t.join(3)
            self.assertTrue(all((task.complete() for task in wrapper.requires())))
            self.assertTrue(wrapper.complete())