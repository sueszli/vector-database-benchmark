from helpers import LuigiTestCase, RunOnceTask
import luigi
import luigi.scheduler
import luigi.worker
FORWARDED_ATTRIBUTES = set(luigi.worker.TaskProcess.forward_reporter_attributes.values())

class NonYieldingTask(RunOnceTask):
    accepts_messages = True

    def gather_forwarded_attributes(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a set of names of attributes that are forwarded by the TaskProcess and that are not\n        *None*. The tests in this file check if and which attributes are present at different times,\n        e.g. while running, or before and after a dynamic dependency was yielded.\n        '
        attrs = set()
        for attr in FORWARDED_ATTRIBUTES:
            if getattr(self, attr, None) is not None:
                attrs.add(attr)
        return attrs

    def run(self):
        if False:
            print('Hello World!')
        self.attributes_while_running = self.gather_forwarded_attributes()
        RunOnceTask.run(self)

class YieldingTask(NonYieldingTask):

    def run(self):
        if False:
            i = 10
            return i + 15
        self.attributes_before_yield = self.gather_forwarded_attributes()
        yield RunOnceTask()
        self.attributes_after_yield = self.gather_forwarded_attributes()
        RunOnceTask.run(self)

class TaskForwardedAttributesTest(LuigiTestCase):

    def run_task(self, task):
        if False:
            i = 10
            return i + 15
        sch = luigi.scheduler.Scheduler()
        with luigi.worker.Worker(scheduler=sch) as w:
            w.add(task)
            w.run()
        return task

    def test_non_yielding_task(self):
        if False:
            for i in range(10):
                print('nop')
        task = self.run_task(NonYieldingTask())
        self.assertEqual(task.attributes_while_running, FORWARDED_ATTRIBUTES)

    def test_yielding_task(self):
        if False:
            for i in range(10):
                print('nop')
        task = self.run_task(YieldingTask())
        self.assertEqual(task.attributes_before_yield, FORWARDED_ATTRIBUTES)
        self.assertEqual(task.attributes_after_yield, FORWARDED_ATTRIBUTES)