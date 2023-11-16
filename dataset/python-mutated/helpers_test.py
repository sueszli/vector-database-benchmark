import luigi
import luigi.date_interval
import luigi.interface
import luigi.notifications
from helpers import LuigiTestCase, RunOnceTask

class LuigiTestCaseTest(LuigiTestCase):

    def test_1(self):
        if False:
            return 10

        class MyClass(luigi.Task):
            pass
        self.assertTrue(self.run_locally(['MyClass']))

    def test_2(self):
        if False:
            while True:
                i = 10

        class MyClass(luigi.Task):
            pass
        self.assertTrue(self.run_locally(['MyClass']))

class RunOnceTaskTest(LuigiTestCase):

    def test_complete_behavior(self):
        if False:
            i = 10
            return i + 15
        '\n        Verify that RunOnceTask works as expected.\n\n        This task will fail if it is a normal ``luigi.Task``, because\n        RequiringTask will not run (missing dependency at runtime).\n        '

        class MyTask(RunOnceTask):
            pass

        class RequiringTask(luigi.Task):
            counter = 0

            def requires(self):
                if False:
                    return 10
                yield MyTask()

            def run(self):
                if False:
                    while True:
                        i = 10
                RequiringTask.counter += 1
        self.run_locally(['RequiringTask'])
        self.assertEqual(1, RequiringTask.counter)