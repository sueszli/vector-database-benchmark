from helpers import LuigiTestCase
from luigi.scheduler import Scheduler
from luigi.worker import Worker
import luigi
import threading

class WorkerKeepAliveUpstreamTest(LuigiTestCase):
    """
    Tests related to how the worker stays alive after upstream status changes.

    See https://github.com/spotify/luigi/pull/1789
    """

    def run(self, result=None):
        if False:
            print('Hello World!')
        '\n        Common setup code. Due to the contextmanager cant use normal setup\n        '
        self.sch = Scheduler(retry_delay=1e-08, retry_count=2)
        with Worker(scheduler=self.sch, worker_id='X', keep_alive=True, wait_interval=0.1, wait_jitter=0) as w:
            self.w = w
            super(WorkerKeepAliveUpstreamTest, self).run(result)

    def test_alive_while_has_failure(self):
        if False:
            print('Hello World!')
        '\n        One dependency disables and one fails\n        '

        class Disabler(luigi.Task):
            pass

        class Failer(luigi.Task):
            did_run = False

            def run(self):
                if False:
                    print('Hello World!')
                self.did_run = True

        class Wrapper(luigi.WrapperTask):

            def requires(self):
                if False:
                    while True:
                        i = 10
                return (Disabler(), Failer())
        self.w.add(Wrapper())
        disabler = Disabler().task_id
        failer = Failer().task_id
        self.sch.add_task(disabler, 'FAILED', worker='X')
        self.sch.prune()
        self.sch.add_task(disabler, 'FAILED', worker='X')
        self.sch.add_task(failer, 'FAILED', worker='X')
        try:
            t = threading.Thread(target=self.w.run)
            t.start()
            t.join(timeout=1)
            self.assertTrue(t.is_alive())
            self.assertFalse(Failer.did_run)
        finally:
            self.sch.prune()
            t.join(timeout=1)
            assert not t.is_alive()

    def test_alive_while_has_success(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        One dependency disables and one succeeds\n        '

        class Disabler(luigi.Task):
            pass

        class Succeeder(luigi.Task):
            did_run = False

            def run(self):
                if False:
                    return 10
                self.did_run = True

        class Wrapper(luigi.WrapperTask):

            def requires(self):
                if False:
                    i = 10
                    return i + 15
                return (Disabler(), Succeeder())
        self.w.add(Wrapper())
        disabler = Disabler().task_id
        succeeder = Succeeder().task_id
        self.sch.add_task(disabler, 'FAILED', worker='X')
        self.sch.prune()
        self.sch.add_task(disabler, 'FAILED', worker='X')
        self.sch.add_task(succeeder, 'DONE', worker='X')
        try:
            t = threading.Thread(target=self.w.run)
            t.start()
            t.join(timeout=1)
            self.assertFalse(t.is_alive())
            self.assertFalse(Succeeder.did_run)
        finally:
            self.sch.prune()
            t.join(timeout=1)
            assert not t.is_alive()