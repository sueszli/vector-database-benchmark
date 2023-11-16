from helpers import LuigiTestCase, temporary_unloaded_module
import luigi
from luigi.worker import Worker
import multiprocessing

class ContextManagedTaskProcessTest(LuigiTestCase):

    def _test_context_manager(self, force_multiprocessing):
        if False:
            return 10
        CONTEXT_MANAGER_MODULE = b'\nclass MyContextManager:\n    def __init__(self, task_process):\n        self.task = task_process.task\n    def __enter__(self):\n        assert not self.task.run_event.is_set(), "the task should not have run yet"\n        self.task.enter_event.set()\n        return self\n    def __exit__(self, exc_type=None, exc_value=None, traceback=None):\n        assert self.task.run_event.is_set(), "the task should have run"\n        self.task.exit_event.set()\n'

        class DummyEventRecordingTask(luigi.Task):

            def __init__(self, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                self.enter_event = multiprocessing.Event()
                self.exit_event = multiprocessing.Event()
                self.run_event = multiprocessing.Event()
                super(DummyEventRecordingTask, self).__init__(*args, **kwargs)

            def run(self):
                if False:
                    for i in range(10):
                        print('nop')
                assert self.enter_event.is_set(), 'the context manager should have been entered'
                assert not self.exit_event.is_set(), 'the context manager should not have been exited yet'
                assert not self.run_event.is_set(), 'the task should not have run yet'
                self.run_event.set()

            def complete(self):
                if False:
                    i = 10
                    return i + 15
                return self.run_event.is_set()
        with temporary_unloaded_module(CONTEXT_MANAGER_MODULE) as module_name:
            t = DummyEventRecordingTask()
            w = Worker(task_process_context=module_name + '.MyContextManager', force_multiprocessing=force_multiprocessing)
            w.add(t)
            self.assertTrue(w.run())
            self.assertTrue(t.complete())
            self.assertTrue(t.enter_event.is_set())
            self.assertTrue(t.exit_event.is_set())

    def test_context_manager_without_multiprocessing(self):
        if False:
            while True:
                i = 10
        self._test_context_manager(False)

    def test_context_manager_with_multiprocessing(self):
        if False:
            return 10
        self._test_context_manager(True)