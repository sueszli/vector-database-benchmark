"""Pass manager test cases."""
from test.python.passmanager import PassManagerTestCase
from logging import getLogger
from qiskit.passmanager import GenericPass
from qiskit.passmanager import PassManagerState, WorkflowStatus, PropertySet
from qiskit.passmanager.compilation_status import RunState

class TestGenericPass(PassManagerTestCase):
    """Tests for the GenericPass subclass."""

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.state = PassManagerState(workflow_status=WorkflowStatus(), property_set=PropertySet())

    def test_run_task(self):
        if False:
            print('Hello World!')
        'Test case: Simple successful task execution.'

        class Task(GenericPass):

            def run(self, passmanager_ir):
                if False:
                    print('Hello World!')
                return passmanager_ir
        task = Task()
        data = 'test_data'
        expected = ['Pass: Task - (\\d*\\.)?\\d+ \\(ms\\)']
        with self.assertLogContains(expected):
            task.execute(passmanager_ir=data, state=self.state)
        self.assertEqual(self.state.workflow_status.count, 1)
        self.assertIn(task, self.state.workflow_status.completed_passes)
        self.assertEqual(self.state.workflow_status.previous_run, RunState.SUCCESS)

    def test_failure_task(self):
        if False:
            while True:
                i = 10
        'Test case: Log is created regardless of success.'

        class TestError(Exception):
            pass

        class RaiseError(GenericPass):

            def run(self, passmanager_ir):
                if False:
                    i = 10
                    return i + 15
                raise TestError()
        task = RaiseError()
        data = 'test_data'
        expected = ['Pass: RaiseError - (\\d*\\.)?\\d+ \\(ms\\)']
        with self.assertLogContains(expected):
            with self.assertRaises(TestError):
                task.execute(passmanager_ir=data, state=self.state)
        self.assertEqual(self.state.workflow_status.count, 0)
        self.assertNotIn(task, self.state.workflow_status.completed_passes)
        self.assertEqual(self.state.workflow_status.previous_run, RunState.FAIL)

    def test_requires(self):
        if False:
            while True:
                i = 10
        'Test case: Dependency tasks are run in advance to user provided task.'

        class TaskA(GenericPass):

            def run(self, passmanager_ir):
                if False:
                    for i in range(10):
                        print('nop')
                return passmanager_ir

        class TaskB(GenericPass):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.requires = [TaskA()]

            def run(self, passmanager_ir):
                if False:
                    for i in range(10):
                        print('nop')
                return passmanager_ir
        task = TaskB()
        data = 'test_data'
        expected = ['Pass: TaskA - (\\d*\\.)?\\d+ \\(ms\\)', 'Pass: TaskB - (\\d*\\.)?\\d+ \\(ms\\)']
        with self.assertLogContains(expected):
            task.execute(passmanager_ir=data, state=self.state)
        self.assertEqual(self.state.workflow_status.count, 2)

    def test_requires_in_list(self):
        if False:
            print('Hello World!')
        'Test case: Dependency tasks are not executed multiple times.'

        class TaskA(GenericPass):

            def run(self, passmanager_ir):
                if False:
                    while True:
                        i = 10
                return passmanager_ir

        class TaskB(GenericPass):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.requires = [TaskA()]

            def run(self, passmanager_ir):
                if False:
                    i = 10
                    return i + 15
                return passmanager_ir
        task = TaskB()
        data = 'test_data'
        expected = ['Pass: TaskB - (\\d*\\.)?\\d+ \\(ms\\)']
        self.state.workflow_status.completed_passes.add(task.requires[0])
        with self.assertLogContains(expected):
            task.execute(passmanager_ir=data, state=self.state)
        self.assertEqual(self.state.workflow_status.count, 1)

    def test_run_with_callable(self):
        if False:
            i = 10
            return i + 15
        'Test case: Callable is called after generic pass is run.'

        def test_callable(task, passmanager_ir, property_set, running_time, count):
            if False:
                for i in range(10):
                    print('nop')
            logger = getLogger()
            logger.info('%s is running on %s', task.name(), passmanager_ir)

        class Task(GenericPass):

            def run(self, passmanager_ir):
                if False:
                    i = 10
                    return i + 15
                return passmanager_ir
        task = Task()
        data = 'test_data'
        expected = ['Pass: Task - (\\d*\\.)?\\d+ \\(ms\\)', 'Task is running on test_data']
        with self.assertLogContains(expected):
            task.execute(passmanager_ir=data, state=self.state, callback=test_callable)