import email.parser
import functools
import logging
import os
import shutil
import signal
import tempfile
import threading
import time
import psutil
from helpers import unittest, with_config, skipOnTravisAndGithubActions, LuigiTestCase, temporary_unloaded_module
import luigi.notifications
import luigi.task_register
import luigi.worker
import mock
from luigi import ExternalTask, RemoteScheduler, Task, Event
from luigi.mock import MockTarget, MockFileSystem
from luigi.scheduler import Scheduler
from luigi.worker import Worker
from luigi.rpc import RPCError
from luigi.cmdline import luigi_run
luigi.notifications.DEBUG = True

class DummyTask(Task):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(DummyTask, self).__init__(*args, **kwargs)
        self.has_run = False

    def complete(self):
        if False:
            i = 10
            return i + 15
        return self.has_run

    def run(self):
        if False:
            i = 10
            return i + 15
        logging.debug('%s - setting has_run', self)
        self.has_run = True

class DynamicDummyTask(Task):
    p = luigi.Parameter()
    sleep = luigi.FloatParameter(default=0.5, significant=False)

    def output(self):
        if False:
            return 10
        return luigi.LocalTarget(self.p)

    def run(self):
        if False:
            print('Hello World!')
        with self.output().open('w') as f:
            f.write('Done!')
        time.sleep(self.sleep)

class DynamicDummyTaskWithNamespace(DynamicDummyTask):
    task_namespace = 'banana'

class DynamicRequires(Task):
    p = luigi.Parameter()
    use_banana_task = luigi.BoolParameter(default=False)

    def output(self):
        if False:
            print('Hello World!')
        return luigi.LocalTarget(os.path.join(self.p, 'parent'))

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        if self.use_banana_task:
            task_cls = DynamicDummyTaskWithNamespace
        else:
            task_cls = DynamicDummyTask
        dummy_targets = (yield [task_cls(os.path.join(self.p, str(i))) for i in range(5)])
        dummy_targets += (yield [task_cls(os.path.join(self.p, str(i))) for i in range(5, 7)])
        with self.output().open('w') as f:
            for (i, d) in enumerate(dummy_targets):
                for line in d.open('r'):
                    print('%d: %s' % (i, line.strip()), file=f)

class DynamicRequiresWrapped(Task):
    p = luigi.Parameter()

    def output(self):
        if False:
            i = 10
            return i + 15
        return luigi.LocalTarget(os.path.join(self.p, 'parent'))

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        reqs = [DynamicDummyTask(p=os.path.join(self.p, '%s.txt' % i), sleep=0.0) for i in range(10)]
        yield luigi.DynamicRequirements(reqs)

        def custom_complete(complete_fn):
            if False:
                while True:
                    i = 10
            if not complete_fn(reqs[0]):
                return False
            paths = [task.output().path for task in reqs]
            basenames = os.listdir(os.path.dirname(paths[0]))
            self._custom_complete_called = True
            self._custom_complete_result = all((os.path.basename(path) in basenames for path in paths))
            return self._custom_complete_result
        yield luigi.DynamicRequirements(reqs, custom_complete)
        with self.output().open('w') as f:
            f.write('Done!')

class DynamicRequiresOtherModule(Task):
    p = luigi.Parameter()

    def output(self):
        if False:
            while True:
                i = 10
        return luigi.LocalTarget(os.path.join(self.p, 'baz'))

    def run(self):
        if False:
            while True:
                i = 10
        import other_module
        other_target_foo = (yield other_module.OtherModuleTask(os.path.join(self.p, 'foo')))
        other_target_bar = (yield other_module.OtherModuleTask(os.path.join(self.p, 'bar')))
        with self.output().open('w') as f:
            f.write('Done!')

class DummyErrorTask(Task):
    retry_index = 0

    def run(self):
        if False:
            i = 10
            return i + 15
        self.retry_index += 1
        raise Exception('Retry index is %s for %s' % (self.retry_index, self.task_family))

class WorkerTest(LuigiTestCase):

    def run(self, result=None):
        if False:
            while True:
                i = 10
        self.sch = Scheduler(retry_delay=100, remove_delay=1000, worker_disconnect_delay=10, stable_done_cooldown_secs=0)
        self.time = time.time
        with Worker(scheduler=self.sch, worker_id='X') as w, Worker(scheduler=self.sch, worker_id='Y') as w2:
            self.w = w
            self.w2 = w2
            super(WorkerTest, self).run(result)
        if time.time != self.time:
            time.time = self.time

    def setTime(self, t):
        if False:
            print('Hello World!')
        time.time = lambda : t

    def test_dep(self):
        if False:
            for i in range(10):
                print('nop')

        class A(Task):

            def run(self):
                if False:
                    while True:
                        i = 10
                self.has_run = True

            def complete(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.has_run
        a = A()

        class B(Task):

            def requires(self):
                if False:
                    for i in range(10):
                        print('nop')
                return a

            def run(self):
                if False:
                    print('Hello World!')
                self.has_run = True

            def complete(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.has_run
        b = B()
        a.has_run = False
        b.has_run = False
        self.assertTrue(self.w.add(b))
        self.assertTrue(self.w.run())
        self.assertTrue(a.has_run)
        self.assertTrue(b.has_run)

    def test_external_dep(self):
        if False:
            return 10

        class A(ExternalTask):

            def complete(self):
                if False:
                    while True:
                        i = 10
                return False
        a = A()

        class B(Task):

            def requires(self):
                if False:
                    print('Hello World!')
                return a

            def run(self):
                if False:
                    while True:
                        i = 10
                self.has_run = True

            def complete(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.has_run
        b = B()
        a.has_run = False
        b.has_run = False
        self.assertTrue(self.w.add(b))
        self.assertTrue(self.w.run())
        self.assertFalse(a.has_run)
        self.assertFalse(b.has_run)

    def test_externalized_dep(self):
        if False:
            i = 10
            return i + 15

        class A(Task):
            has_run = False

            def run(self):
                if False:
                    while True:
                        i = 10
                self.has_run = True

            def complete(self):
                if False:
                    i = 10
                    return i + 15
                return self.has_run
        a = A()

        class B(A):

            def requires(self):
                if False:
                    i = 10
                    return i + 15
                return luigi.task.externalize(a)
        b = B()
        self.assertTrue(self.w.add(b))
        self.assertTrue(self.w.run())
        self.assertFalse(a.has_run)
        self.assertFalse(b.has_run)

    def test_legacy_externalized_dep(self):
        if False:
            print('Hello World!')

        class A(Task):
            has_run = False

            def run(self):
                if False:
                    i = 10
                    return i + 15
                self.has_run = True

            def complete(self):
                if False:
                    print('Hello World!')
                return self.has_run
        a = A()
        a.run = NotImplemented

        class B(A):

            def requires(self):
                if False:
                    while True:
                        i = 10
                return a
        b = B()
        self.assertTrue(self.w.add(b))
        self.assertTrue(self.w.run())
        self.assertFalse(a.has_run)
        self.assertFalse(b.has_run)

    def test_type_error_in_tracking_run_deprecated(self):
        if False:
            while True:
                i = 10

        class A(Task):
            num_runs = 0

            def complete(self):
                if False:
                    for i in range(10):
                        print('nop')
                return False

            def run(self, tracking_url_callback=None):
                if False:
                    print('Hello World!')
                self.num_runs += 1
                raise TypeError('bad type')
        a = A()
        self.assertTrue(self.w.add(a))
        self.assertFalse(self.w.run())
        self.assertEqual(1, a.num_runs)

    def test_tracking_url(self):
        if False:
            i = 10
            return i + 15
        tracking_url = 'http://test_url.com/'

        class A(Task):
            has_run = False

            def complete(self):
                if False:
                    return 10
                return self.has_run

            def run(self):
                if False:
                    while True:
                        i = 10
                self.set_tracking_url(tracking_url)
                self.has_run = True
        a = A()
        self.assertTrue(self.w.add(a))
        self.assertTrue(self.w.run())
        tasks = self.sch.task_list('DONE', '')
        self.assertEqual(1, len(tasks))
        self.assertEqual(tracking_url, tasks[a.task_id]['tracking_url'])

    def test_fail(self):
        if False:
            return 10

        class CustomException(BaseException):

            def __init__(self, msg):
                if False:
                    return 10
                self.msg = msg

        class A(Task):

            def run(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.has_run = True
                raise CustomException('bad things')

            def complete(self):
                if False:
                    while True:
                        i = 10
                return self.has_run
        a = A()

        class B(Task):

            def requires(self):
                if False:
                    return 10
                return a

            def run(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.has_run = True

            def complete(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.has_run
        b = B()
        a.has_run = False
        b.has_run = False
        self.assertTrue(self.w.add(b))
        self.assertFalse(self.w.run())
        self.assertTrue(a.has_run)
        self.assertFalse(b.has_run)

    def test_unknown_dep(self):
        if False:
            for i in range(10):
                print('nop')

        class A(ExternalTask):

            def complete(self):
                if False:
                    return 10
                return False

        class C(Task):

            def complete(self):
                if False:
                    while True:
                        i = 10
                return True

        def get_b(dep):
            if False:
                return 10

            class B(Task):

                def requires(self):
                    if False:
                        i = 10
                        return i + 15
                    return dep

                def run(self):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.has_run = True

                def complete(self):
                    if False:
                        print('Hello World!')
                    return False
            b = B()
            b.has_run = False
            return b
        b_a = get_b(A())
        b_c = get_b(C())
        self.assertTrue(self.w.add(b_a))
        self.assertTrue(self.w2.add(b_c))
        self.assertFalse(self.w.run())
        self.assertFalse(b_a.has_run)

    def test_unfulfilled_dep(self):
        if False:
            return 10

        class A(Task):

            def complete(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.done

            def run(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.done = True

        def get_b(a):
            if False:
                print('Hello World!')

            class B(A):

                def requires(self):
                    if False:
                        return 10
                    return a
            b = B()
            b.done = False
            a.done = True
            return b
        a = A()
        b = get_b(a)
        self.assertTrue(self.w.add(b))
        a.done = False
        self.w.run()
        self.assertTrue(a.complete())
        self.assertTrue(b.complete())

    def test_check_unfulfilled_deps_config(self):
        if False:
            for i in range(10):
                print('nop')

        class A(Task):
            i = luigi.IntParameter()

            def __init__(self, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                super(A, self).__init__(*args, **kwargs)
                self.complete_count = 0
                self.has_run = False

            def complete(self):
                if False:
                    return 10
                self.complete_count += 1
                return self.has_run

            def run(self):
                if False:
                    print('Hello World!')
                self.has_run = True

        class B(A):

            def requires(self):
                if False:
                    print('Hello World!')
                return A(i=self.i)
        with Worker(scheduler=self.sch, worker_id='1') as w:
            w._config.check_unfulfilled_deps = True
            a1 = A(i=1)
            b1 = B(i=1)
            self.assertTrue(w.add(b1))
            self.assertEqual(a1.complete_count, 1)
            self.assertEqual(b1.complete_count, 1)
            w.run()
            self.assertTrue(a1.complete())
            self.assertTrue(b1.complete())
            self.assertEqual(a1.complete_count, 3)
            self.assertEqual(b1.complete_count, 2)
        with Worker(scheduler=self.sch, worker_id='2') as w:
            w._config.check_unfulfilled_deps = False
            a2 = A(i=2)
            b2 = B(i=2)
            self.assertTrue(w.add(b2))
            self.assertEqual(a2.complete_count, 1)
            self.assertEqual(b2.complete_count, 1)
            w.run()
            self.assertTrue(a2.complete())
            self.assertTrue(b2.complete())
            self.assertEqual(a2.complete_count, 2)
            self.assertEqual(b2.complete_count, 2)

    def test_cache_task_completion_config(self):
        if False:
            return 10

        class A(Task):
            i = luigi.IntParameter()

            def __init__(self, *args, **kwargs):
                if False:
                    return 10
                super(A, self).__init__(*args, **kwargs)
                self.complete_count = 0
                self.has_run = False

            def complete(self):
                if False:
                    return 10
                self.complete_count += 1
                return self.has_run

            def run(self):
                if False:
                    i = 10
                    return i + 15
                self.has_run = True

        class B(A):

            def run(self):
                if False:
                    return 10
                yield A(i=self.i + 0)
                yield A(i=self.i + 1)
                yield A(i=self.i + 2)
                self.has_run = True
        with Worker(scheduler=self.sch, worker_id='2', cache_task_completion=True) as w:
            b0 = B(i=0)
            a0 = A(i=0)
            a1 = A(i=1)
            a2 = A(i=2)
            self.assertTrue(w.add(b0))
            self.assertEqual(b0.complete_count, 1)
            self.assertEqual(a0.complete_count, 0)
            self.assertEqual(a1.complete_count, 0)
            self.assertEqual(a2.complete_count, 0)
            w.run()
            self.assertEqual(b0.complete_count, 1)
            self.assertEqual(a0.complete_count, 2)
            self.assertEqual(a1.complete_count, 2)
            self.assertEqual(a2.complete_count, 2)
        with Worker(scheduler=self.sch, worker_id='2', cache_task_completion=False) as w:
            b10 = B(i=10)
            a10 = A(i=10)
            a11 = A(i=11)
            a12 = A(i=12)
            self.assertTrue(w.add(b10))
            self.assertEqual(b10.complete_count, 1)
            self.assertEqual(a10.complete_count, 0)
            self.assertEqual(a11.complete_count, 0)
            self.assertEqual(a12.complete_count, 0)
            w.run()
            self.assertEqual(b10.complete_count, 1)
            self.assertEqual(a10.complete_count, 5)
            self.assertEqual(a11.complete_count, 4)
            self.assertEqual(a12.complete_count, 3)
        with Worker(scheduler=self.sch, worker_id='2', check_complete_on_run=True) as w:
            b20 = B(i=20)
            a20 = A(i=20)
            a21 = A(i=21)
            a22 = A(i=22)
            self.assertTrue(w.add(b20))
            self.assertEqual(b20.complete_count, 1)
            self.assertEqual(a20.complete_count, 0)
            self.assertEqual(a21.complete_count, 0)
            self.assertEqual(a22.complete_count, 0)
            w.run()
            self.assertEqual(b20.complete_count, 2)
            self.assertEqual(a20.complete_count, 6)
            self.assertEqual(a21.complete_count, 5)
            self.assertEqual(a22.complete_count, 4)

    def test_gets_missed_work(self):
        if False:
            for i in range(10):
                print('nop')

        class A(Task):
            done = False

            def complete(self):
                if False:
                    i = 10
                    return i + 15
                return self.done

            def run(self):
                if False:
                    i = 10
                    return i + 15
                self.done = True
        a = A()
        self.assertTrue(self.w.add(a))
        self.assertEqual(a.task_id, self.sch.get_work(worker='X')['task_id'])
        self.assertTrue(self.w.run())
        self.assertTrue(a.complete())

    def test_avoid_infinite_reschedule(self):
        if False:
            while True:
                i = 10

        class A(Task):

            def complete(self):
                if False:
                    print('Hello World!')
                return False

        class B(Task):

            def complete(self):
                if False:
                    return 10
                return False

            def requires(self):
                if False:
                    print('Hello World!')
                return A()
        self.assertTrue(self.w.add(B()))
        self.assertFalse(self.w.run())

    def test_fails_registering_signal(self):
        if False:
            i = 10
            return i + 15
        with mock.patch('luigi.worker.signal', spec=['signal']):
            Worker()

    def test_allow_reschedule_with_many_missing_deps(self):
        if False:
            i = 10
            return i + 15

        class A(Task):
            """ Task that must run twice to succeed """
            i = luigi.IntParameter()
            runs = 0

            def complete(self):
                if False:
                    print('Hello World!')
                return self.runs >= 2

            def run(self):
                if False:
                    return 10
                self.runs += 1

        class B(Task):
            done = False

            def requires(self):
                if False:
                    for i in range(10):
                        print('nop')
                return map(A, range(20))

            def complete(self):
                if False:
                    print('Hello World!')
                return self.done

            def run(self):
                if False:
                    return 10
                self.done = True
        b = B()
        w = Worker(scheduler=self.sch, worker_id='X', max_reschedules=1)
        self.assertTrue(w.add(b))
        self.assertFalse(w.run())
        self.assertTrue(b.complete())
        self.assertTrue(all((a.complete() for a in b.deps())))

    def test_interleaved_workers(self):
        if False:
            return 10

        class A(DummyTask):
            pass
        a = A()

        class B(DummyTask):

            def requires(self):
                if False:
                    for i in range(10):
                        print('nop')
                return a
        ExternalB = luigi.task.externalize(B)
        b = B()
        eb = ExternalB()
        self.assertEqual(str(eb), 'B()')
        sch = Scheduler(retry_delay=100, remove_delay=1000, worker_disconnect_delay=10)
        with Worker(scheduler=sch, worker_id='X') as w, Worker(scheduler=sch, worker_id='Y') as w2:
            self.assertTrue(w.add(b))
            self.assertTrue(w2.add(eb))
            logging.debug('RUNNING BROKEN WORKER')
            self.assertTrue(w2.run())
            self.assertFalse(a.complete())
            self.assertFalse(b.complete())
            logging.debug('RUNNING FUNCTIONAL WORKER')
            self.assertTrue(w.run())
            self.assertTrue(a.complete())
            self.assertTrue(b.complete())

    def test_interleaved_workers2(self):
        if False:
            while True:
                i = 10

        class B(DummyTask):
            pass
        ExternalB = luigi.task.externalize(B)
        b = B()
        eb = ExternalB()
        self.assertEqual(str(eb), 'B()')
        sch = Scheduler(retry_delay=100, remove_delay=1000, worker_disconnect_delay=10)
        with Worker(scheduler=sch, worker_id='X') as w, Worker(scheduler=sch, worker_id='Y') as w2:
            self.assertTrue(w2.add(eb))
            self.assertTrue(w.add(b))
            self.assertTrue(w2.run())
            self.assertFalse(b.complete())
            self.assertTrue(w.run())
            self.assertTrue(b.complete())

    def test_interleaved_workers3(self):
        if False:
            print('Hello World!')

        class A(DummyTask):

            def run(self):
                if False:
                    while True:
                        i = 10
                logging.debug('running A')
                time.sleep(0.1)
                super(A, self).run()
        a = A()

        class B(DummyTask):

            def requires(self):
                if False:
                    i = 10
                    return i + 15
                return a

            def run(self):
                if False:
                    while True:
                        i = 10
                logging.debug('running B')
                super(B, self).run()
        b = B()
        sch = Scheduler(retry_delay=100, remove_delay=1000, worker_disconnect_delay=10)
        with Worker(scheduler=sch, worker_id='X', keep_alive=True, count_uniques=True) as w:
            with Worker(scheduler=sch, worker_id='Y', keep_alive=True, count_uniques=True, wait_interval=0.1, wait_jitter=0.05) as w2:
                self.assertTrue(w.add(a))
                self.assertTrue(w2.add(b))
                threading.Thread(target=w.run).start()
                self.assertTrue(w2.run())
                self.assertTrue(a.complete())
                self.assertTrue(b.complete())

    def test_die_for_non_unique_pending(self):
        if False:
            while True:
                i = 10

        class A(DummyTask):

            def run(self):
                if False:
                    i = 10
                    return i + 15
                logging.debug('running A')
                time.sleep(0.1)
                super(A, self).run()
        a = A()

        class B(DummyTask):

            def requires(self):
                if False:
                    print('Hello World!')
                return a

            def run(self):
                if False:
                    for i in range(10):
                        print('nop')
                logging.debug('running B')
                super(B, self).run()
        b = B()
        sch = Scheduler(retry_delay=100, remove_delay=1000, worker_disconnect_delay=10)
        with Worker(scheduler=sch, worker_id='X', keep_alive=True, count_uniques=True) as w:
            with Worker(scheduler=sch, worker_id='Y', keep_alive=True, count_uniques=True, wait_interval=0.1, wait_jitter=0.05) as w2:
                self.assertTrue(w.add(b))
                self.assertTrue(w2.add(b))
                self.assertEqual(w._get_work()[0], a.task_id)
                self.assertTrue(w2.run())
                self.assertFalse(a.complete())
                self.assertFalse(b.complete())

    def test_complete_exception(self):
        if False:
            i = 10
            return i + 15
        'Tests that a task is still scheduled if its sister task crashes in the complete() method'

        class A(DummyTask):

            def complete(self):
                if False:
                    i = 10
                    return i + 15
                raise Exception('doh')
        a = A()

        class C(DummyTask):
            pass
        c = C()

        class B(DummyTask):

            def requires(self):
                if False:
                    i = 10
                    return i + 15
                return (a, c)
        b = B()
        sch = Scheduler(retry_delay=100, remove_delay=1000, worker_disconnect_delay=10)
        with Worker(scheduler=sch, worker_id='foo') as w:
            self.assertFalse(w.add(b))
            self.assertTrue(w.run())
            self.assertFalse(b.has_run)
            self.assertTrue(c.has_run)
            self.assertFalse(a.has_run)

    def test_requires_exception(self):
        if False:
            while True:
                i = 10

        class A(DummyTask):

            def requires(self):
                if False:
                    print('Hello World!')
                raise Exception('doh')
        a = A()

        class D(DummyTask):
            pass
        d = D()

        class C(DummyTask):

            def requires(self):
                if False:
                    i = 10
                    return i + 15
                return d
        c = C()

        class B(DummyTask):

            def requires(self):
                if False:
                    for i in range(10):
                        print('nop')
                return (c, a)
        b = B()
        sch = Scheduler(retry_delay=100, remove_delay=1000, worker_disconnect_delay=10)
        with Worker(scheduler=sch, worker_id='foo') as w:
            self.assertFalse(w.add(b))
            self.assertTrue(w.run())
            self.assertFalse(b.has_run)
            self.assertTrue(c.has_run)
            self.assertTrue(d.has_run)
            self.assertFalse(a.has_run)

    def test_run_csv_batch_job(self):
        if False:
            while True:
                i = 10
        completed = set()

        class CsvBatchJob(luigi.Task):
            values = luigi.parameter.Parameter(batch_method=','.join)
            has_run = False

            def run(self):
                if False:
                    return 10
                completed.update(self.values.split(','))
                self.has_run = True

            def complete(self):
                if False:
                    print('Hello World!')
                return all((value in completed for value in self.values.split(',')))
        tasks = [CsvBatchJob(str(i)) for i in range(10)]
        for task in tasks:
            self.assertTrue(self.w.add(task))
        self.assertTrue(self.w.run())
        for task in tasks:
            self.assertTrue(task.complete())
            self.assertFalse(task.has_run)

    def test_run_max_batch_job(self):
        if False:
            i = 10
            return i + 15
        completed = set()

        class MaxBatchJob(luigi.Task):
            value = luigi.IntParameter(batch_method=max)
            has_run = False

            def run(self):
                if False:
                    return 10
                completed.add(self.value)
                self.has_run = True

            def complete(self):
                if False:
                    while True:
                        i = 10
                return any((self.value <= ran for ran in completed))
        tasks = [MaxBatchJob(i) for i in range(10)]
        for task in tasks:
            self.assertTrue(self.w.add(task))
        self.assertTrue(self.w.run())
        for task in tasks:
            self.assertTrue(task.complete())
            self.assertFalse(task.has_run and task.value < 9)

    def test_run_batch_job_unbatched(self):
        if False:
            for i in range(10):
                print('nop')
        completed = set()

        class MaxNonBatchJob(luigi.Task):
            value = luigi.IntParameter(batch_method=max)
            has_run = False
            batchable = False

            def run(self):
                if False:
                    i = 10
                    return i + 15
                completed.add(self.value)
                self.has_run = True

            def complete(self):
                if False:
                    i = 10
                    return i + 15
                return self.value in completed
        tasks = [MaxNonBatchJob((i,)) for i in range(10)]
        for task in tasks:
            self.assertTrue(self.w.add(task))
        self.assertTrue(self.w.run())
        for task in tasks:
            self.assertTrue(task.complete())
            self.assertTrue(task.has_run)

    def test_run_batch_job_limit_batch_size(self):
        if False:
            print('Hello World!')
        completed = set()
        runs = []

        class CsvLimitedBatchJob(luigi.Task):
            value = luigi.parameter.Parameter(batch_method=','.join)
            has_run = False
            max_batch_size = 4

            def run(self):
                if False:
                    while True:
                        i = 10
                completed.update(self.value.split(','))
                runs.append(self)

            def complete(self):
                if False:
                    for i in range(10):
                        print('nop')
                return all((value in completed for value in self.value.split(',')))
        tasks = [CsvLimitedBatchJob(str(i)) for i in range(11)]
        for task in tasks:
            self.assertTrue(self.w.add(task))
        self.assertTrue(self.w.run())
        for task in tasks:
            self.assertTrue(task.complete())
        self.assertEqual(3, len(runs))

    def test_fail_max_batch_job(self):
        if False:
            for i in range(10):
                print('nop')

        class MaxBatchFailJob(luigi.Task):
            value = luigi.IntParameter(batch_method=max)
            has_run = False

            def run(self):
                if False:
                    print('Hello World!')
                self.has_run = True
                assert False

            def complete(self):
                if False:
                    for i in range(10):
                        print('nop')
                return False
        tasks = [MaxBatchFailJob(i) for i in range(10)]
        for task in tasks:
            self.assertTrue(self.w.add(task))
        self.assertFalse(self.w.run())
        for task in tasks:
            self.assertFalse(task.has_run and task.value < 9)
        self.assertEqual({task.task_id for task in tasks}, set(self.sch.task_list('FAILED', '')))

    def test_gracefully_handle_batch_method_failure(self):
        if False:
            print('Hello World!')

        class BadBatchMethodTask(DummyTask):
            priority = 10
            batch_int_param = luigi.IntParameter(batch_method=int.__add__)
        bad_tasks = [BadBatchMethodTask(i) for i in range(5)]
        good_tasks = [DummyTask()]
        all_tasks = good_tasks + bad_tasks
        self.assertFalse(any((task.complete() for task in all_tasks)))
        worker = Worker(scheduler=Scheduler(retry_count=1), keep_alive=True)
        for task in all_tasks:
            self.assertTrue(worker.add(task))
        self.assertFalse(worker.run())
        self.assertFalse(any((task.complete() for task in bad_tasks)))
        self.assertTrue(all((task.complete() for task in good_tasks)))

    def test_post_error_message_for_failed_batch_methods(self):
        if False:
            print('Hello World!')

        class BadBatchMethodTask(DummyTask):
            batch_int_param = luigi.IntParameter(batch_method=int.__add__)
        tasks = [BadBatchMethodTask(1), BadBatchMethodTask(2)]
        for task in tasks:
            self.assertTrue(self.w.add(task))
        self.assertFalse(self.w.run())
        failed_ids = set(self.sch.task_list('FAILED', ''))
        self.assertEqual({task.task_id for task in tasks}, failed_ids)
        self.assertTrue(all((self.sch.fetch_error(task_id)['error'] for task_id in failed_ids)))

class WorkerKeepAliveTests(LuigiTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.sch = Scheduler()
        super(WorkerKeepAliveTests, self).setUp()

    def _worker_keep_alive_test(self, first_should_live, second_should_live, task_status=None, **worker_args):
        if False:
            for i in range(10):
                print('nop')
        worker_args.update({'scheduler': self.sch, 'worker_processes': 0, 'wait_interval': 0.01, 'wait_jitter': 0.0})
        w1 = Worker(worker_id='w1', **worker_args)
        w2 = Worker(worker_id='w2', **worker_args)
        with w1 as worker1, w2 as worker2:
            worker1.add(DummyTask())
            t1 = threading.Thread(target=worker1.run)
            t1.start()
            worker2.add(DummyTask())
            t2 = threading.Thread(target=worker2.run)
            t2.start()
            if task_status:
                self.sch.add_task(worker='DummyWorker', task_id=DummyTask().task_id, status=task_status)
            time.sleep(0.1)
            try:
                self.assertEqual(first_should_live, t1.is_alive())
                self.assertEqual(second_should_live, t2.is_alive())
            finally:
                self.sch.add_task(worker='DummyWorker', task_id=DummyTask().task_id, status='DONE')
                t1.join()
                t2.join()

    def test_no_keep_alive(self):
        if False:
            for i in range(10):
                print('nop')
        self._worker_keep_alive_test(first_should_live=False, second_should_live=False)

    def test_keep_alive(self):
        if False:
            for i in range(10):
                print('nop')
        self._worker_keep_alive_test(first_should_live=True, second_should_live=True, keep_alive=True)

    def test_keep_alive_count_uniques(self):
        if False:
            i = 10
            return i + 15
        self._worker_keep_alive_test(first_should_live=False, second_should_live=False, keep_alive=True, count_uniques=True)

    def test_keep_alive_count_last_scheduled(self):
        if False:
            return 10
        self._worker_keep_alive_test(first_should_live=False, second_should_live=True, keep_alive=True, count_last_scheduled=True)

    def test_keep_alive_through_failure(self):
        if False:
            for i in range(10):
                print('nop')
        self._worker_keep_alive_test(first_should_live=True, second_should_live=True, keep_alive=True, task_status='FAILED')

    def test_do_not_keep_alive_through_disable(self):
        if False:
            i = 10
            return i + 15
        self._worker_keep_alive_test(first_should_live=False, second_should_live=False, keep_alive=True, task_status='DISABLED')

class WorkerInterruptedTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.sch = Scheduler(retry_delay=100, remove_delay=1000, worker_disconnect_delay=10)
    requiring_sigusr = unittest.skipUnless(hasattr(signal, 'SIGUSR1'), 'signal.SIGUSR1 not found on this system')

    def _test_stop_getting_new_work(self, worker):
        if False:
            print('Hello World!')
        d = DummyTask()
        with worker:
            worker.add(d)
            self.assertFalse(d.complete())
            worker.handle_interrupt(signal.SIGUSR1, None)
            worker.run()
            self.assertFalse(d.complete())

    @requiring_sigusr
    def test_stop_getting_new_work(self):
        if False:
            while True:
                i = 10
        self._test_stop_getting_new_work(Worker(scheduler=self.sch))

    @requiring_sigusr
    def test_stop_getting_new_work_assistant(self):
        if False:
            while True:
                i = 10
        self._test_stop_getting_new_work(Worker(scheduler=self.sch, keep_alive=False, assistant=True))

    @requiring_sigusr
    def test_stop_getting_new_work_assistant_keep_alive(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_stop_getting_new_work(Worker(scheduler=self.sch, keep_alive=True, assistant=True))

    def test_existence_of_disabling_option(self):
        if False:
            print('Hello World!')
        Worker(no_install_shutdown_handler=True)

    @with_config({'worker': {'no_install_shutdown_handler': 'True'}})
    def test_can_run_luigi_in_thread(self):
        if False:
            print('Hello World!')

        class A(DummyTask):
            pass
        task = A()
        t = threading.Thread(target=lambda : luigi.build([task], local_scheduler=True))
        t.start()
        t.join()
        self.assertTrue(task.complete())

class WorkerDisabledTest(LuigiTestCase):

    def make_sch(self):
        if False:
            i = 10
            return i + 15
        return Scheduler(retry_delay=100, remove_delay=1000, worker_disconnect_delay=10)

    def _test_stop_getting_new_work_build(self, sch, worker):
        if False:
            while True:
                i = 10
        '\n        I got motivated to create this test case when I saw that the\n        execution_summary crashed after my first attempted solution.\n        '

        class KillWorkerTask(luigi.Task):
            did_actually_run = False

            def run(self):
                if False:
                    print('Hello World!')
                sch.disable_worker('my_worker_id')
                KillWorkerTask.did_actually_run = True

        class Factory:

            def create_local_scheduler(self, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                return sch

            def create_worker(self, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                return worker
        luigi.build([KillWorkerTask()], worker_scheduler_factory=Factory(), local_scheduler=True)
        self.assertTrue(KillWorkerTask.did_actually_run)

    def _test_stop_getting_new_work_manual(self, sch, worker):
        if False:
            print('Hello World!')
        d = DummyTask()
        with worker:
            worker.add(d)
            self.assertFalse(d.complete())
            sch.disable_worker('my_worker_id')
            worker.run()
            self.assertFalse(d.complete())

    def _test_stop_getting_new_work(self, **worker_kwargs):
        if False:
            return 10
        worker_kwargs['worker_id'] = 'my_worker_id'
        sch = self.make_sch()
        worker_kwargs['scheduler'] = sch
        self._test_stop_getting_new_work_manual(sch, Worker(**worker_kwargs))
        sch = self.make_sch()
        worker_kwargs['scheduler'] = sch
        self._test_stop_getting_new_work_build(sch, Worker(**worker_kwargs))

    def test_stop_getting_new_work_keep_alive(self):
        if False:
            print('Hello World!')
        self._test_stop_getting_new_work(keep_alive=True, assistant=False)

    def test_stop_getting_new_work_assistant(self):
        if False:
            return 10
        self._test_stop_getting_new_work(keep_alive=False, assistant=True)

    def test_stop_getting_new_work_assistant_keep_alive(self):
        if False:
            i = 10
            return i + 15
        self._test_stop_getting_new_work(keep_alive=True, assistant=True)

class DynamicDependenciesTest(unittest.TestCase):
    n_workers = 1
    timeout = float('inf')

    def setUp(self):
        if False:
            while True:
                i = 10
        self.p = tempfile.mkdtemp()

    def tearDown(self):
        if False:
            while True:
                i = 10
        shutil.rmtree(self.p)

    def test_dynamic_dependencies(self, use_banana_task=False):
        if False:
            i = 10
            return i + 15
        t0 = time.time()
        t = DynamicRequires(p=self.p, use_banana_task=use_banana_task)
        luigi.build([t], local_scheduler=True, workers=self.n_workers)
        self.assertTrue(t.complete())
        with t.output().open('r') as f:
            for i in range(7):
                self.assertEqual(f.readline().strip(), '%d: Done!' % i)
        self.assertTrue(time.time() - t0 < self.timeout)

    def test_dynamic_dependencies_with_namespace(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_dynamic_dependencies(use_banana_task=True)

    def test_dynamic_dependencies_other_module(self):
        if False:
            while True:
                i = 10
        t = DynamicRequiresOtherModule(p=self.p)
        luigi.build([t], local_scheduler=True, workers=self.n_workers)
        self.assertTrue(t.complete())

    def test_wrapped_dynamic_requirements(self):
        if False:
            print('Hello World!')
        t = DynamicRequiresWrapped(p=self.p)
        luigi.build([t], local_scheduler=True, workers=1)
        self.assertTrue(t.complete())
        self.assertTrue(getattr(t, '_custom_complete_called', False))
        self.assertTrue(getattr(t, '_custom_complete_result', False))

class DynamicDependenciesWithMultipleWorkersTest(DynamicDependenciesTest):
    n_workers = 100
    timeout = 3.0

class WorkerPingThreadTests(unittest.TestCase):

    def test_ping_retry(self):
        if False:
            print('Hello World!')
        ' Worker ping fails once. Ping continues to try to connect to scheduler\n\n        Kind of ugly since it uses actual timing with sleep to test the thread\n        '
        sch = Scheduler(retry_delay=100, remove_delay=1000, worker_disconnect_delay=10)
        self._total_pings = 0

        def fail_ping(worker):
            if False:
                i = 10
                return i + 15
            self._total_pings += 1
            raise Exception('Some random exception')
        sch.ping = fail_ping
        with Worker(scheduler=sch, worker_id='foo', ping_interval=0.01):
            time.sleep(0.1)
        self.assertTrue(self._total_pings > 1, msg="Didn't retry pings (%d pings performed)" % (self._total_pings,))

    def test_ping_thread_shutdown(self):
        if False:
            return 10
        with Worker(ping_interval=0.01) as w:
            self.assertTrue(w._keep_alive_thread.is_alive())
        self.assertFalse(w._keep_alive_thread.is_alive())

def email_patch(test_func, email_config=None):
    if False:
        return 10
    EMAIL_CONFIG = {'email': {'receiver': 'not-a-real-email-address-for-test-only', 'force_send': 'true'}}
    if email_config is not None:
        EMAIL_CONFIG.update(email_config)
    emails = []

    def mock_send_email(sender, recipients, msg):
        if False:
            while True:
                i = 10
        emails.append(msg)

    @with_config(EMAIL_CONFIG)
    @functools.wraps(test_func)
    @mock.patch('smtplib.SMTP')
    def run_test(self, smtp):
        if False:
            i = 10
            return i + 15
        smtp().sendmail.side_effect = mock_send_email
        test_func(self, emails)
    return run_test

def custom_email_patch(config):
    if False:
        while True:
            i = 10
    return functools.partial(email_patch, email_config=config)

class WorkerEmailTest(LuigiTestCase):

    def run(self, result=None):
        if False:
            return 10
        super(WorkerEmailTest, self).setUp()
        sch = Scheduler(retry_delay=100, remove_delay=1000, worker_disconnect_delay=10)
        with Worker(scheduler=sch, worker_id='foo') as self.worker:
            super(WorkerEmailTest, self).run(result)

    @email_patch
    def test_connection_error(self, emails):
        if False:
            i = 10
            return i + 15
        sch = RemoteScheduler('http://tld.invalid:1337', connect_timeout=1)
        sch._rpc_retry_wait = 1

        class A(DummyTask):
            pass
        a = A()
        self.assertEqual(emails, [])
        with Worker(scheduler=sch) as worker:
            try:
                worker.add(a)
            except RPCError as e:
                self.assertTrue(str(e).find('Errors (3 attempts)') != -1)
                self.assertNotEqual(emails, [])
                self.assertTrue(emails[0].find('Luigi: Framework error while scheduling %s' % (a,)) != -1)
            else:
                self.fail()

    @email_patch
    def test_complete_error(self, emails):
        if False:
            for i in range(10):
                print('nop')

        class A(DummyTask):

            def complete(self):
                if False:
                    return 10
                raise Exception('b0rk')
        a = A()
        self.assertEqual(emails, [])
        self.worker.add(a)
        self.assertTrue(emails[0].find('Luigi: %s failed scheduling' % (a,)) != -1)
        self.worker.run()
        self.assertTrue(emails[0].find('Luigi: %s failed scheduling' % (a,)) != -1)
        self.assertFalse(a.has_run)

    @with_config({'batch_email': {'email_interval': '0'}, 'worker': {'send_failure_email': 'False'}})
    @email_patch
    def test_complete_error_email_batch(self, emails):
        if False:
            for i in range(10):
                print('nop')

        class A(DummyTask):

            def complete(self):
                if False:
                    i = 10
                    return i + 15
                raise Exception('b0rk')
        scheduler = Scheduler(batch_emails=True)
        worker = Worker(scheduler)
        a = A()
        self.assertEqual(emails, [])
        worker.add(a)
        self.assertEqual(emails, [])
        worker.run()
        self.assertEqual(emails, [])
        self.assertFalse(a.has_run)
        scheduler.prune()
        self.assertTrue('1 scheduling failure' in emails[0])

    @with_config({'batch_email': {'email_interval': '0'}, 'worker': {'send_failure_email': 'False'}})
    @email_patch
    def test_complete_error_email_batch_to_owner(self, emails):
        if False:
            return 10

        class A(DummyTask):
            owner_email = 'a_owner@test.com'

            def complete(self):
                if False:
                    i = 10
                    return i + 15
                raise Exception('b0rk')
        scheduler = Scheduler(batch_emails=True)
        worker = Worker(scheduler)
        a = A()
        self.assertEqual(emails, [])
        worker.add(a)
        self.assertEqual(emails, [])
        worker.run()
        self.assertEqual(emails, [])
        self.assertFalse(a.has_run)
        scheduler.prune()
        self.assertTrue(any(('1 scheduling failure' in email and 'a_owner@test.com' in email for email in emails)))

    @email_patch
    def test_announce_scheduling_failure_unexpected_error(self, emails):
        if False:
            print('Hello World!')

        class A(DummyTask):
            owner_email = 'a_owner@test.com'

            def complete(self):
                if False:
                    return 10
                pass
        scheduler = Scheduler(batch_emails=True)
        worker = Worker(scheduler)
        a = A()
        with mock.patch.object(worker._scheduler, 'announce_scheduling_failure', side_effect=Exception('Unexpected')), self.assertRaises(Exception):
            worker.add(a)
        self.assertTrue(len(emails) == 2)
        self.assertTrue('Luigi: Framework error while scheduling' in emails[1])
        self.assertTrue('a_owner@test.com' in emails[1])

    @email_patch
    def test_requires_error(self, emails):
        if False:
            i = 10
            return i + 15

        class A(DummyTask):

            def requires(self):
                if False:
                    for i in range(10):
                        print('nop')
                raise Exception('b0rk')
        a = A()
        self.assertEqual(emails, [])
        self.worker.add(a)
        self.assertTrue(emails[0].find('Luigi: %s failed scheduling' % (a,)) != -1)
        self.worker.run()
        self.assertFalse(a.has_run)

    @with_config({'batch_email': {'email_interval': '0'}, 'worker': {'send_failure_email': 'False'}})
    @email_patch
    def test_requires_error_email_batch(self, emails):
        if False:
            i = 10
            return i + 15

        class A(DummyTask):

            def requires(self):
                if False:
                    for i in range(10):
                        print('nop')
                raise Exception('b0rk')
        scheduler = Scheduler(batch_emails=True)
        worker = Worker(scheduler)
        a = A()
        self.assertEqual(emails, [])
        worker.add(a)
        self.assertEqual(emails, [])
        worker.run()
        self.assertFalse(a.has_run)
        scheduler.prune()
        self.assertTrue('1 scheduling failure' in emails[0])

    @email_patch
    def test_complete_return_value(self, emails):
        if False:
            while True:
                i = 10

        class A(DummyTask):

            def complete(self):
                if False:
                    while True:
                        i = 10
                pass
        a = A()
        self.assertEqual(emails, [])
        self.worker.add(a)
        self.assertTrue(emails[0].find('Luigi: %s failed scheduling' % (a,)) != -1)
        self.worker.run()
        self.assertTrue(emails[0].find('Luigi: %s failed scheduling' % (a,)) != -1)
        self.assertFalse(a.has_run)

    @with_config({'batch_email': {'email_interval': '0'}, 'worker': {'send_failure_email': 'False'}})
    @email_patch
    def test_complete_return_value_email_batch(self, emails):
        if False:
            print('Hello World!')

        class A(DummyTask):

            def complete(self):
                if False:
                    while True:
                        i = 10
                pass
        scheduler = Scheduler(batch_emails=True)
        worker = Worker(scheduler)
        a = A()
        self.assertEqual(emails, [])
        worker.add(a)
        self.assertEqual(emails, [])
        self.worker.run()
        self.assertEqual(emails, [])
        self.assertFalse(a.has_run)
        scheduler.prune()
        self.assertTrue('1 scheduling failure' in emails[0])

    @email_patch
    def test_run_error(self, emails):
        if False:
            return 10

        class A(luigi.Task):

            def run(self):
                if False:
                    print('Hello World!')
                raise Exception('b0rk')
        a = A()
        luigi.build([a], workers=1, local_scheduler=True)
        self.assertEqual(1, len(emails))
        self.assertTrue(emails[0].find('Luigi: %s FAILED' % (a,)) != -1)

    @email_patch
    def test_run_error_long_traceback(self, emails):
        if False:
            while True:
                i = 10

        class A(luigi.Task):

            def run(self):
                if False:
                    i = 10
                    return i + 15
                raise Exception('b0rk' * 10500)
        a = A()
        luigi.build([a], workers=1, local_scheduler=True)
        self.assertTrue(len(emails[0]) < 10000)
        self.assertTrue(emails[0].find('Traceback exceeds max length and has been truncated'))

    @with_config({'batch_email': {'email_interval': '0'}, 'worker': {'send_failure_email': 'False'}})
    @email_patch
    def test_run_error_email_batch(self, emails):
        if False:
            print('Hello World!')

        class A(luigi.Task):
            owner_email = ['a@test.com', 'b@test.com']

            def run(self):
                if False:
                    print('Hello World!')
                raise Exception('b0rk')
        scheduler = Scheduler(batch_emails=True)
        worker = Worker(scheduler)
        worker.add(A())
        worker.run()
        scheduler.prune()
        self.assertEqual(3, len(emails))
        self.assertTrue(any(('a@test.com' in email for email in emails)))
        self.assertTrue(any(('b@test.com' in email for email in emails)))

    @with_config({'batch_email': {'email_interval': '0'}, 'worker': {'send_failure_email': 'False'}})
    @email_patch
    def test_run_error_batch_email_string(self, emails):
        if False:
            while True:
                i = 10

        class A(luigi.Task):
            owner_email = 'a@test.com'

            def run(self):
                if False:
                    print('Hello World!')
                raise Exception('b0rk')
        scheduler = Scheduler(batch_emails=True)
        worker = Worker(scheduler)
        worker.add(A())
        worker.run()
        scheduler.prune()
        self.assertEqual(2, len(emails))
        self.assertTrue(any(('a@test.com' in email for email in emails)))

    @with_config({'worker': {'send_failure_email': 'False'}})
    @email_patch
    def test_run_error_no_email(self, emails):
        if False:
            i = 10
            return i + 15

        class A(luigi.Task):

            def run(self):
                if False:
                    i = 10
                    return i + 15
                raise Exception('b0rk')
        luigi.build([A()], workers=1, local_scheduler=True)
        self.assertFalse(emails)

    @staticmethod
    def read_email(email_msg):
        if False:
            return 10
        (subject_obj, body_obj) = email.parser.Parser().parsestr(email_msg).walk()
        return (str(subject_obj['Subject']), str(body_obj.get_payload(decode=True)))

    @email_patch
    def test_task_process_dies_with_email(self, emails):
        if False:
            while True:
                i = 10
        a = SendSignalTask(signal.SIGKILL)
        luigi.build([a], workers=2, local_scheduler=True)
        self.assertEqual(1, len(emails))
        (subject, body) = self.read_email(emails[0])
        self.assertIn('Luigi: {} FAILED'.format(a), subject)
        self.assertIn('died unexpectedly with exit code -9', body)

    @with_config({'worker': {'send_failure_email': 'False'}})
    @email_patch
    def test_task_process_dies_no_email(self, emails):
        if False:
            while True:
                i = 10
        luigi.build([SendSignalTask(signal.SIGKILL)], workers=2, local_scheduler=True)
        self.assertEqual([], emails)

    @email_patch
    def test_task_times_out(self, emails):
        if False:
            while True:
                i = 10

        class A(luigi.Task):
            worker_timeout = 0.0001

            def run(self):
                if False:
                    return 10
                time.sleep(5)
        a = A()
        luigi.build([a], workers=2, local_scheduler=True)
        self.assertEqual(1, len(emails))
        (subject, body) = self.read_email(emails[0])
        self.assertIn('Luigi: %s FAILED' % (a,), subject)
        self.assertIn('timed out after 0.0001 seconds and was terminated.', body)

    @with_config({'worker': {'send_failure_email': 'False'}})
    @email_patch
    def test_task_times_out_no_email(self, emails):
        if False:
            return 10

        class A(luigi.Task):
            worker_timeout = 0.0001

            def run(self):
                if False:
                    while True:
                        i = 10
                time.sleep(5)
        luigi.build([A()], workers=2, local_scheduler=True)
        self.assertEqual([], emails)

    @with_config(dict(worker=dict(retry_external_tasks='true')))
    @email_patch
    def test_external_task_retries(self, emails):
        if False:
            print('Hello World!')
        '\n        Test that we do not send error emails on the failures of external tasks\n        '

        class A(luigi.ExternalTask):
            pass
        a = A()
        luigi.build([a], workers=2, local_scheduler=True)
        self.assertEqual(emails, [])

    @email_patch
    def test_no_error(self, emails):
        if False:
            return 10

        class A(DummyTask):
            pass
        a = A()
        self.assertEqual(emails, [])
        self.worker.add(a)
        self.assertEqual(emails, [])
        self.worker.run()
        self.assertEqual(emails, [])
        self.assertTrue(a.complete())

    @custom_email_patch({'email': {'receiver': 'not-a-real-email-address-for-test-only', 'format': 'none'}})
    def test_disable_emails(self, emails):
        if False:
            i = 10
            return i + 15

        class A(luigi.Task):

            def complete(self):
                if False:
                    while True:
                        i = 10
                raise Exception('b0rk')
        self.worker.add(A())
        self.assertEqual(emails, [])

class RaiseSystemExit(luigi.Task):

    def run(self):
        if False:
            return 10
        raise SystemExit('System exit!!')

class SendSignalTask(luigi.Task):
    signal = luigi.IntParameter()

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        os.kill(os.getpid(), self.signal)

class HangTheWorkerTask(luigi.Task):
    worker_timeout = luigi.IntParameter(default=None)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        while True:
            pass

    def complete(self):
        if False:
            while True:
                i = 10
        return False

class MultipleWorkersTest(unittest.TestCase):

    @unittest.skip('Always skip. There are many intermittent failures')
    def test_multiple_workers(self):
        if False:
            i = 10
            return i + 15

        class MyDynamicTask(luigi.Task):
            x = luigi.Parameter()

            def run(self):
                if False:
                    while True:
                        i = 10
                time.sleep(0.1)
        t0 = time.time()
        luigi.build([MyDynamicTask(i) for i in range(100)], workers=100, local_scheduler=True)
        self.assertTrue(time.time() < t0 + 5.0)

    def test_zero_workers(self):
        if False:
            for i in range(10):
                print('nop')
        d = DummyTask()
        luigi.build([d], workers=0, local_scheduler=True)
        self.assertFalse(d.complete())

    def test_system_exit(self):
        if False:
            for i in range(10):
                print('nop')
        luigi.build([RaiseSystemExit()], workers=2, local_scheduler=True)

    def test_term_worker(self):
        if False:
            for i in range(10):
                print('nop')
        luigi.build([SendSignalTask(signal.SIGTERM)], workers=2, local_scheduler=True)

    def test_kill_worker(self):
        if False:
            i = 10
            return i + 15
        luigi.build([SendSignalTask(signal.SIGKILL)], workers=2, local_scheduler=True)

    def test_purge_multiple_workers(self):
        if False:
            return 10
        w = Worker(worker_processes=2, wait_interval=0.01)
        t1 = SendSignalTask(signal.SIGTERM)
        t2 = SendSignalTask(signal.SIGKILL)
        w.add(t1)
        w.add(t2)
        w._run_task(t1.task_id)
        w._run_task(t2.task_id)
        time.sleep(1.0)
        w._handle_next_task()
        w._handle_next_task()
        w._handle_next_task()

    def test_stop_worker_kills_subprocesses(self):
        if False:
            print('Hello World!')
        with Worker(worker_processes=2) as w:
            hung_task = HangTheWorkerTask()
            w.add(hung_task)
            w._run_task(hung_task.task_id)
            pids = [p.pid for p in w._running_tasks.values()]
            self.assertEqual(1, len(pids))
            pid = pids[0]

            def is_running():
                if False:
                    while True:
                        i = 10
                return pid in {p.pid for p in psutil.Process().children()}
            self.assertTrue(is_running())
        self.assertFalse(is_running())

    @mock.patch('luigi.worker.time')
    def test_no_process_leak_from_repeatedly_running_same_task(self, worker_time):
        if False:
            i = 10
            return i + 15
        with Worker(worker_processes=2) as w:
            hung_task = HangTheWorkerTask()
            w.add(hung_task)
            w._run_task(hung_task.task_id)
            children = set(psutil.Process().children())
            for _ in range(10):
                worker_time.sleep.reset_mock()
                w._run_task(hung_task.task_id)
                worker_time.sleep.assert_called_once_with(mock.ANY)
            self.assertEqual(children, set(psutil.Process().children()))

    def test_time_out_hung_worker(self):
        if False:
            return 10
        luigi.build([HangTheWorkerTask(0.1)], workers=2, local_scheduler=True)

    def test_time_out_hung_single_worker(self):
        if False:
            print('Hello World!')
        luigi.build([HangTheWorkerTask(0.1)], workers=1, local_scheduler=True)

    @skipOnTravisAndGithubActions('https://travis-ci.org/spotify/luigi/jobs/72953986')
    @mock.patch('luigi.worker.time')
    def test_purge_hung_worker_default_timeout_time(self, mock_time):
        if False:
            return 10
        w = Worker(worker_processes=2, wait_interval=0.01, timeout=5)
        mock_time.time.return_value = 0
        task = HangTheWorkerTask()
        w.add(task)
        w._run_task(task.task_id)
        mock_time.time.return_value = 5
        w._handle_next_task()
        self.assertEqual(1, len(w._running_tasks))
        mock_time.time.return_value = 6
        w._handle_next_task()
        self.assertEqual(0, len(w._running_tasks))

    @skipOnTravisAndGithubActions('https://travis-ci.org/spotify/luigi/jobs/76645264')
    @mock.patch('luigi.worker.time')
    def test_purge_hung_worker_override_timeout_time(self, mock_time):
        if False:
            print('Hello World!')
        w = Worker(worker_processes=2, wait_interval=0.01, timeout=5)
        mock_time.time.return_value = 0
        task = HangTheWorkerTask(worker_timeout=10)
        w.add(task)
        w._run_task(task.task_id)
        mock_time.time.return_value = 10
        w._handle_next_task()
        self.assertEqual(1, len(w._running_tasks))
        mock_time.time.return_value = 11
        w._handle_next_task()
        self.assertEqual(0, len(w._running_tasks))

class Dummy2Task(Task):
    p = luigi.Parameter()

    def output(self):
        if False:
            print('Hello World!')
        return MockTarget(self.p)

    def run(self):
        if False:
            i = 10
            return i + 15
        f = self.output().open('w')
        f.write('test')
        f.close()

class AssistantTest(unittest.TestCase):

    def run(self, result=None):
        if False:
            i = 10
            return i + 15
        self.sch = Scheduler(retry_delay=100, remove_delay=1000, worker_disconnect_delay=10)
        self.assistant = Worker(scheduler=self.sch, worker_id='Y', assistant=True)
        with Worker(scheduler=self.sch, worker_id='X') as w:
            self.w = w
            super(AssistantTest, self).run(result)

    def test_get_work(self):
        if False:
            return 10
        d = Dummy2Task('123')
        self.w.add(d)
        self.assertFalse(d.complete())
        self.assistant.run()
        self.assertTrue(d.complete())

    def test_bad_job_type(self):
        if False:
            print('Hello World!')

        class Dummy3Task(Dummy2Task):
            task_family = 'UnknownTaskFamily'
        d = Dummy3Task('123')
        self.w.add(d)
        self.assertFalse(d.complete())
        self.assertFalse(self.assistant.run())
        self.assertFalse(d.complete())
        self.assertEqual(list(self.sch.task_list('FAILED', '').keys()), [d.task_id])

    def test_unimported_job_type(self):
        if False:
            for i in range(10):
                print('nop')
        MODULE_CONTENTS = b'\nimport luigi\n\n\nclass UnimportedTask(luigi.Task):\n    def complete(self):\n        return False\n'
        reg = luigi.task_register.Register._get_reg()

        class UnimportedTask(luigi.Task):
            task_module = None
        luigi.task_register.Register._set_reg(reg)
        task = UnimportedTask()
        self.w.add(task)
        self.assertFalse(self.assistant.run())
        self.assertEqual(list(self.sch.task_list('FAILED', '').keys()), [task.task_id])
        with temporary_unloaded_module(MODULE_CONTENTS) as task.task_module:
            self.w.add(task)
            self.assertTrue(self.assistant.run())
            self.assertEqual(list(self.sch.task_list('DONE', '').keys()), [task.task_id])

    def test_unimported_job_sends_failure_message(self):
        if False:
            while True:
                i = 10

        class NotInAssistantTask(luigi.Task):
            task_family = 'Unknown'
            task_module = None
        task = NotInAssistantTask()
        self.w.add(task)
        self.assertFalse(self.assistant.run())
        self.assertEqual(list(self.sch.task_list('FAILED', '').keys()), [task.task_id])
        self.assertTrue(self.sch.fetch_error(task.task_id)['error'])

class ForkBombTask(luigi.Task):
    depth = luigi.IntParameter()
    breadth = luigi.IntParameter()
    p = luigi.Parameter(default=(0,))

    def output(self):
        if False:
            return 10
        return MockTarget('.'.join(map(str, self.p)))

    def run(self):
        if False:
            return 10
        with self.output().open('w') as f:
            f.write('Done!')

    def requires(self):
        if False:
            print('Hello World!')
        if len(self.p) < self.depth:
            for i in range(self.breadth):
                yield ForkBombTask(self.depth, self.breadth, self.p + (i,))

class TaskLimitTest(unittest.TestCase):

    def tearDown(self):
        if False:
            while True:
                i = 10
        MockFileSystem().remove('')

    @with_config({'worker': {'task_limit': '6'}})
    def test_task_limit_exceeded(self):
        if False:
            print('Hello World!')
        w = Worker()
        t = ForkBombTask(3, 2)
        w.add(t)
        w.run()
        self.assertFalse(t.complete())
        leaf_tasks = [ForkBombTask(3, 2, branch) for branch in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]]
        self.assertEqual(3, sum((t.complete() for t in leaf_tasks)), "should have gracefully completed as much as possible even though the single last leaf didn't get scheduled")

    @with_config({'worker': {'task_limit': '7'}})
    def test_task_limit_not_exceeded(self):
        if False:
            return 10
        w = Worker()
        t = ForkBombTask(3, 2)
        w.add(t)
        w.run()
        self.assertTrue(t.complete())

    def test_no_task_limit(self):
        if False:
            return 10
        w = Worker()
        t = ForkBombTask(4, 2)
        w.add(t)
        w.run()
        self.assertTrue(t.complete())

class WorkerConfigurationTest(unittest.TestCase):

    def test_asserts_for_worker(self):
        if False:
            print('Hello World!')
        "\n        Test that Worker() asserts that it's sanely configured\n        "
        Worker(wait_interval=1)
        self.assertRaises(AssertionError, Worker, wait_interval=0)

class WorkerWaitJitterTest(unittest.TestCase):

    @with_config({'worker': {'wait_jitter': '10.0'}})
    @mock.patch('random.uniform')
    @mock.patch('time.sleep')
    def test_wait_jitter(self, mock_sleep, mock_random):
        if False:
            for i in range(10):
                print('nop')
        ' verify configured jitter amount '
        mock_random.return_value = 1.0
        w = Worker()
        x = w._sleeper()
        next(x)
        mock_random.assert_called_with(0, 10.0)
        mock_sleep.assert_called_with(2.0)
        mock_random.return_value = 2.0
        next(x)
        mock_random.assert_called_with(0, 10.0)
        mock_sleep.assert_called_with(3.0)

    @mock.patch('random.uniform')
    @mock.patch('time.sleep')
    def test_wait_jitter_default(self, mock_sleep, mock_random):
        if False:
            i = 10
            return i + 15
        ' verify default jitter is as expected '
        mock_random.return_value = 1.0
        w = Worker()
        x = w._sleeper()
        next(x)
        mock_random.assert_called_with(0, 5.0)
        mock_sleep.assert_called_with(2.0)
        mock_random.return_value = 3.3
        next(x)
        mock_random.assert_called_with(0, 5.0)
        mock_sleep.assert_called_with(4.3)

class KeyboardInterruptBehaviorTest(LuigiTestCase):

    def test_propagation_when_executing(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Ensure that keyboard interrupts causes luigi to quit when you are\n        executing tasks.\n\n        TODO: Add a test that tests the multiprocessing (--worker >1) case\n        '

        class KeyboardInterruptTask(luigi.Task):

            def run(self):
                if False:
                    for i in range(10):
                        print('nop')
                raise KeyboardInterrupt()
        cmd = 'KeyboardInterruptTask --local-scheduler --no-lock'.split(' ')
        self.assertRaises(KeyboardInterrupt, luigi_run, cmd)

    def test_propagation_when_scheduling(self):
        if False:
            print('Hello World!')
        '\n        Test that KeyboardInterrupt causes luigi to quit while scheduling.\n        '

        class KeyboardInterruptTask(luigi.Task):

            def complete(self):
                if False:
                    print('Hello World!')
                raise KeyboardInterrupt()

        class ExternalKeyboardInterruptTask(luigi.ExternalTask):

            def complete(self):
                if False:
                    while True:
                        i = 10
                raise KeyboardInterrupt()
        self.assertRaises(KeyboardInterrupt, luigi_run, ['KeyboardInterruptTask', '--local-scheduler', '--no-lock'])
        self.assertRaises(KeyboardInterrupt, luigi_run, ['ExternalKeyboardInterruptTask', '--local-scheduler', '--no-lock'])

class WorkerPurgeEventHandlerTest(unittest.TestCase):

    @mock.patch('luigi.worker.ContextManagedTaskProcess')
    def test_process_killed_handler(self, task_proc):
        if False:
            print('Hello World!')
        result = []

        @HangTheWorkerTask.event_handler(Event.PROCESS_FAILURE)
        def store_task(t, error_msg):
            if False:
                for i in range(10):
                    print('nop')
            self.assertTrue(error_msg)
            result.append(t)
        w = Worker()
        task = HangTheWorkerTask()
        task_process = mock.MagicMock(is_alive=lambda : False, exitcode=-14, task=task)
        task_proc.return_value = task_process
        w.add(task)
        w._run_task(task.task_id)
        w._handle_next_task()
        self.assertEqual(result, [task])

    @mock.patch('luigi.worker.time')
    def test_timeout_handler(self, mock_time):
        if False:
            while True:
                i = 10
        result = []

        @HangTheWorkerTask.event_handler(Event.TIMEOUT)
        def store_task(t, error_msg):
            if False:
                print('Hello World!')
            self.assertTrue(error_msg)
            result.append(t)
        w = Worker(worker_processes=2, wait_interval=0.01, timeout=5)
        mock_time.time.return_value = 0
        task = HangTheWorkerTask(worker_timeout=1)
        w.add(task)
        w._run_task(task.task_id)
        mock_time.time.return_value = 3
        w._handle_next_task()
        self.assertEqual(result, [task])

    @mock.patch('luigi.worker.time')
    def test_timeout_handler_single_worker(self, mock_time):
        if False:
            for i in range(10):
                print('nop')
        result = []

        @HangTheWorkerTask.event_handler(Event.TIMEOUT)
        def store_task(t, error_msg):
            if False:
                return 10
            self.assertTrue(error_msg)
            result.append(t)
        w = Worker(wait_interval=0.01, timeout=5)
        mock_time.time.return_value = 0
        task = HangTheWorkerTask(worker_timeout=1)
        w.add(task)
        w._run_task(task.task_id)
        mock_time.time.return_value = 3
        w._handle_next_task()
        self.assertEqual(result, [task])

class PerTaskRetryPolicyBehaviorTest(LuigiTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(PerTaskRetryPolicyBehaviorTest, self).setUp()
        self.per_task_retry_count = 3
        self.default_retry_count = 1
        self.sch = Scheduler(retry_delay=0.1, retry_count=self.default_retry_count, prune_on_get_work=True)

    def test_with_all_disabled_with_single_worker(self):
        if False:
            print('Hello World!')
        '\n            With this test, a case which has a task (TestWrapperTask), requires two another tasks (TestErrorTask1,TestErrorTask1) which both is failed, is\n            tested.\n\n            Task TestErrorTask1 has default retry_count which is 1, but Task TestErrorTask2 has retry_count at task level as 2.\n\n            This test is running on single worker\n        '

        class TestErrorTask1(DummyErrorTask):
            pass
        e1 = TestErrorTask1()

        class TestErrorTask2(DummyErrorTask):
            retry_count = self.per_task_retry_count
        e2 = TestErrorTask2()

        class TestWrapperTask(luigi.WrapperTask):

            def requires(self):
                if False:
                    i = 10
                    return i + 15
                return [e2, e1]
        wt = TestWrapperTask()
        with Worker(scheduler=self.sch, worker_id='X', keep_alive=True, wait_interval=0.1, wait_jitter=0.05) as w1:
            self.assertTrue(w1.add(wt))
            self.assertFalse(w1.run())
            self.assertEqual([wt.task_id], list(self.sch.task_list('PENDING', 'UPSTREAM_DISABLED').keys()))
            self.assertEqual(sorted([e1.task_id, e2.task_id]), sorted(self.sch.task_list('DISABLED', '').keys()))
            self.assertEqual(0, self.sch._state.get_task(wt.task_id).num_failures())
            self.assertEqual(self.per_task_retry_count, self.sch._state.get_task(e2.task_id).num_failures())
            self.assertEqual(self.default_retry_count, self.sch._state.get_task(e1.task_id).num_failures())

    def test_with_all_disabled_with_multiple_worker(self):
        if False:
            i = 10
            return i + 15
        '\n            With this test, a case which has a task (TestWrapperTask), requires two another tasks (TestErrorTask1,TestErrorTask1) which both is failed, is\n            tested.\n\n            Task TestErrorTask1 has default retry_count which is 1, but Task TestErrorTask2 has retry_count at task level as 2.\n\n            This test is running on multiple worker\n        '

        class TestErrorTask1(DummyErrorTask):
            pass
        e1 = TestErrorTask1()

        class TestErrorTask2(DummyErrorTask):
            retry_count = self.per_task_retry_count
        e2 = TestErrorTask2()

        class TestWrapperTask(luigi.WrapperTask):

            def requires(self):
                if False:
                    print('Hello World!')
                return [e2, e1]
        wt = TestWrapperTask()
        with Worker(scheduler=self.sch, worker_id='X', keep_alive=True, wait_interval=0.1, wait_jitter=0.05) as w1:
            with Worker(scheduler=self.sch, worker_id='Y', keep_alive=True, wait_interval=0.1, wait_jitter=0.05) as w2:
                with Worker(scheduler=self.sch, worker_id='Z', keep_alive=True, wait_interval=0.1, wait_jitter=0.05) as w3:
                    self.assertTrue(w1.add(wt))
                    self.assertTrue(w2.add(e2))
                    self.assertTrue(w3.add(e1))
                    self.assertFalse(w3.run())
                    self.assertFalse(w2.run())
                    self.assertTrue(w1.run())
                    self.assertEqual([wt.task_id], list(self.sch.task_list('PENDING', 'UPSTREAM_DISABLED').keys()))
                    self.assertEqual(sorted([e1.task_id, e2.task_id]), sorted(self.sch.task_list('DISABLED', '').keys()))
                    self.assertEqual(0, self.sch._state.get_task(wt.task_id).num_failures())
                    self.assertEqual(self.per_task_retry_count, self.sch._state.get_task(e2.task_id).num_failures())
                    self.assertEqual(self.default_retry_count, self.sch._state.get_task(e1.task_id).num_failures())

    def test_with_includes_success_with_single_worker(self):
        if False:
            for i in range(10):
                print('nop')
        '\n            With this test, a case which has a task (TestWrapperTask), requires one (TestErrorTask1) FAILED and one (TestSuccessTask1) SUCCESS, is tested.\n\n            Task TestSuccessTask1 will be DONE successfully, but Task TestErrorTask1 will be failed and it has retry_count at task level as 2.\n\n            This test is running on single worker\n        '

        class TestSuccessTask1(DummyTask):
            pass
        s1 = TestSuccessTask1()

        class TestErrorTask1(DummyErrorTask):
            retry_count = self.per_task_retry_count
        e1 = TestErrorTask1()

        class TestWrapperTask(luigi.WrapperTask):

            def requires(self):
                if False:
                    return 10
                return [e1, s1]
        wt = TestWrapperTask()
        with Worker(scheduler=self.sch, worker_id='X', keep_alive=True, wait_interval=0.1, wait_jitter=0.05) as w1:
            self.assertTrue(w1.add(wt))
            self.assertFalse(w1.run())
            self.assertEqual([wt.task_id], list(self.sch.task_list('PENDING', 'UPSTREAM_DISABLED').keys()))
            self.assertEqual([e1.task_id], list(self.sch.task_list('DISABLED', '').keys()))
            self.assertEqual([s1.task_id], list(self.sch.task_list('DONE', '').keys()))
            self.assertEqual(0, self.sch._state.get_task(wt.task_id).num_failures())
            self.assertEqual(self.per_task_retry_count, self.sch._state.get_task(e1.task_id).num_failures())
            self.assertEqual(0, self.sch._state.get_task(s1.task_id).num_failures())

    def test_with_includes_success_with_multiple_worker(self):
        if False:
            while True:
                i = 10
        '\n            With this test, a case which has a task (TestWrapperTask), requires one (TestErrorTask1) FAILED and one (TestSuccessTask1) SUCCESS, is tested.\n\n            Task TestSuccessTask1 will be DONE successfully, but Task TestErrorTask1 will be failed and it has retry_count at task level as 2.\n\n            This test is running on multiple worker\n        '

        class TestSuccessTask1(DummyTask):
            pass
        s1 = TestSuccessTask1()

        class TestErrorTask1(DummyErrorTask):
            retry_count = self.per_task_retry_count
        e1 = TestErrorTask1()

        class TestWrapperTask(luigi.WrapperTask):

            def requires(self):
                if False:
                    return 10
                return [e1, s1]
        wt = TestWrapperTask()
        with Worker(scheduler=self.sch, worker_id='X', keep_alive=True, wait_interval=0.1, wait_jitter=0.05) as w1:
            with Worker(scheduler=self.sch, worker_id='Y', keep_alive=True, wait_interval=0.1, wait_jitter=0.05) as w2:
                with Worker(scheduler=self.sch, worker_id='Z', keep_alive=True, wait_interval=0.1, wait_jitter=0.05) as w3:
                    self.assertTrue(w1.add(wt))
                    self.assertTrue(w2.add(e1))
                    self.assertTrue(w3.add(s1))
                    self.assertTrue(w3.run())
                    self.assertFalse(w2.run())
                    self.assertTrue(w1.run())
                    self.assertEqual([wt.task_id], list(self.sch.task_list('PENDING', 'UPSTREAM_DISABLED').keys()))
                    self.assertEqual([e1.task_id], list(self.sch.task_list('DISABLED', '').keys()))
                    self.assertEqual([s1.task_id], list(self.sch.task_list('DONE', '').keys()))
                    self.assertEqual(0, self.sch._state.get_task(wt.task_id).num_failures())
                    self.assertEqual(self.per_task_retry_count, self.sch._state.get_task(e1.task_id).num_failures())
                    self.assertEqual(0, self.sch._state.get_task(s1.task_id).num_failures())

    def test_with_dynamic_dependencies_with_single_worker(self):
        if False:
            i = 10
            return i + 15
        '\n            With this test, a case includes dependency tasks(TestErrorTask1,TestErrorTask2) which both are failed.\n\n            Task TestErrorTask1 has default retry_count which is 1, but Task TestErrorTask2 has retry_count at task level as 2.\n\n            This test is running on single worker\n        '

        class TestErrorTask1(DummyErrorTask):
            pass
        e1 = TestErrorTask1()

        class TestErrorTask2(DummyErrorTask):
            retry_count = self.per_task_retry_count
        e2 = TestErrorTask2()

        class TestSuccessTask1(DummyTask):
            pass
        s1 = TestSuccessTask1()

        class TestWrapperTask(DummyTask):

            def requires(self):
                if False:
                    return 10
                return [s1]

            def run(self):
                if False:
                    i = 10
                    return i + 15
                super(TestWrapperTask, self).run()
                yield (e2, e1)
        wt = TestWrapperTask()
        with Worker(scheduler=self.sch, worker_id='X', keep_alive=True, wait_interval=0.1, wait_jitter=0.05) as w1:
            self.assertTrue(w1.add(wt))
            self.assertFalse(w1.run())
            self.assertEqual([wt.task_id], list(self.sch.task_list('PENDING', 'UPSTREAM_DISABLED').keys()))
            self.assertEqual(sorted([e1.task_id, e2.task_id]), sorted(self.sch.task_list('DISABLED', '').keys()))
            self.assertEqual(0, self.sch._state.get_task(wt.task_id).num_failures())
            self.assertEqual(0, self.sch._state.get_task(s1.task_id).num_failures())
            self.assertEqual(self.per_task_retry_count, self.sch._state.get_task(e2.task_id).num_failures())
            self.assertEqual(self.default_retry_count, self.sch._state.get_task(e1.task_id).num_failures())

    def test_with_dynamic_dependencies_with_multiple_workers(self):
        if False:
            while True:
                i = 10
        '\n            With this test, a case includes dependency tasks(TestErrorTask1,TestErrorTask2) which both are failed.\n\n            Task TestErrorTask1 has default retry_count which is 1, but Task TestErrorTask2 has retry_count at task level as 2.\n\n            This test is running on multiple worker\n        '

        class TestErrorTask1(DummyErrorTask):
            pass
        e1 = TestErrorTask1()

        class TestErrorTask2(DummyErrorTask):
            retry_count = self.per_task_retry_count
        e2 = TestErrorTask2()

        class TestSuccessTask1(DummyTask):
            pass
        s1 = TestSuccessTask1()

        class TestWrapperTask(DummyTask):

            def requires(self):
                if False:
                    print('Hello World!')
                return [s1]

            def run(self):
                if False:
                    return 10
                super(TestWrapperTask, self).run()
                yield (e2, e1)
        wt = TestWrapperTask()
        with Worker(scheduler=self.sch, worker_id='X', keep_alive=True, wait_interval=0.1, wait_jitter=0.05) as w1:
            with Worker(scheduler=self.sch, worker_id='Y', keep_alive=True, wait_interval=0.1, wait_jitter=0.05) as w2:
                self.assertTrue(w1.add(wt))
                self.assertTrue(w2.add(s1))
                self.assertTrue(w2.run())
                self.assertFalse(w1.run())
                self.assertEqual([wt.task_id], list(self.sch.task_list('PENDING', 'UPSTREAM_DISABLED').keys()))
                self.assertEqual(sorted([e1.task_id, e2.task_id]), sorted(self.sch.task_list('DISABLED', '').keys()))
                self.assertEqual(0, self.sch._state.get_task(wt.task_id).num_failures())
                self.assertEqual(0, self.sch._state.get_task(s1.task_id).num_failures())
                self.assertEqual(self.per_task_retry_count, self.sch._state.get_task(e2.task_id).num_failures())
                self.assertEqual(self.default_retry_count, self.sch._state.get_task(e1.task_id).num_failures())

    def test_per_task_disable_persist_with_single_worker(self):
        if False:
            i = 10
            return i + 15
        '\n        Ensure that `Task.disable_window` impacts the task retrying policy:\n        - with the scheduler retry policy (disable_window=3), task fails twice and gets disabled\n        - with the task retry policy (disable_window=0.5) task never gets into the DISABLED state\n        '

        class TwoErrorsThenSuccessTask(Task):
            """
            The task is failing two times and then succeeds, waiting 1s before each try
            """
            retry_index = 0
            disable_window = None

            def run(self):
                if False:
                    for i in range(10):
                        print('nop')
                time.sleep(1)
                self.retry_index += 1
                if self.retry_index < 3:
                    raise Exception('Retry index is %s for %s' % (self.retry_index, self.task_family))
        t = TwoErrorsThenSuccessTask()
        sch = Scheduler(retry_delay=0.1, retry_count=2, prune_on_get_work=True, disable_window=2)
        with Worker(scheduler=sch, worker_id='X', keep_alive=True, wait_interval=0.1, wait_jitter=0.05) as w:
            self.assertTrue(w.add(t))
            self.assertFalse(w.run())
            self.assertEqual(2, t.retry_index)
            self.assertEqual([t.task_id], list(sch.task_list('DISABLED').keys()))
            self.assertEqual(2, sch._state.get_task(t.task_id).num_failures())
        t = TwoErrorsThenSuccessTask()
        t.retry_index = 0
        t.disable_window = 0.5
        sch = Scheduler(retry_delay=0.1, retry_count=2, prune_on_get_work=True, disable_window=2)
        with Worker(scheduler=sch, worker_id='X', keep_alive=True, wait_interval=0.1, wait_jitter=0.05) as w:
            self.assertTrue(w.add(t))
            self.assertFalse(w.run())
            self.assertEqual(3, t.retry_index)
            self.assertEqual([t.task_id], list(sch.task_list('DONE').keys()))
            self.assertEqual(1, len(sch._state.get_task(t.task_id).failures))