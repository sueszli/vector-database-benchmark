from helpers import unittest
import luigi
from luigi import Event, Task, build
from luigi.mock import MockTarget, MockFileSystem
from luigi.task import flatten
from mock import patch

class DummyException(Exception):
    pass

class EmptyTask(Task):
    fail = luigi.BoolParameter()

    def run(self):
        if False:
            return 10
        self.trigger_event(Event.PROGRESS, self, {'foo': 'bar'})
        if self.fail:
            raise DummyException()

class TaskWithBrokenDependency(Task):

    def requires(self):
        if False:
            i = 10
            return i + 15
        raise DummyException()

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class TaskWithCallback(Task):

    def run(self):
        if False:
            while True:
                i = 10
        print('Triggering event')
        self.trigger_event('foo event')

class TestEventCallbacks(unittest.TestCase):

    def test_start_handler(self):
        if False:
            return 10
        saved_tasks = []

        @EmptyTask.event_handler(Event.START)
        def save_task(task):
            if False:
                print('Hello World!')
            print('Saving task...')
            saved_tasks.append(task)
        t = EmptyTask(True)
        build([t], local_scheduler=True)
        self.assertEqual(saved_tasks, [t])

    def _run_empty_task(self, fail):
        if False:
            print('Hello World!')
        progresses = []
        progresses_data = []
        successes = []
        failures = []
        exceptions = []

        @EmptyTask.event_handler(Event.SUCCESS)
        def success(task):
            if False:
                return 10
            successes.append(task)

        @EmptyTask.event_handler(Event.FAILURE)
        def failure(task, exception):
            if False:
                return 10
            failures.append(task)
            exceptions.append(exception)

        @EmptyTask.event_handler(Event.PROGRESS)
        def progress(task, data):
            if False:
                for i in range(10):
                    print('nop')
            progresses.append(task)
            progresses_data.append(data)
        t = EmptyTask(fail)
        build([t], local_scheduler=True)
        return (t, progresses, progresses_data, successes, failures, exceptions)

    def test_success(self):
        if False:
            while True:
                i = 10
        (t, progresses, progresses_data, successes, failures, exceptions) = self._run_empty_task(False)
        self.assertEqual(progresses, [t])
        self.assertEqual(progresses_data, [{'foo': 'bar'}])
        self.assertEqual(successes, [t])
        self.assertEqual(failures, [])
        self.assertEqual(exceptions, [])

    def test_failure(self):
        if False:
            i = 10
            return i + 15
        (t, progresses, progresses_data, successes, failures, exceptions) = self._run_empty_task(True)
        self.assertEqual(progresses, [t])
        self.assertEqual(progresses_data, [{'foo': 'bar'}])
        self.assertEqual(successes, [])
        self.assertEqual(failures, [t])
        self.assertEqual(len(exceptions), 1)
        self.assertTrue(isinstance(exceptions[0], DummyException))

    def test_broken_dependency(self):
        if False:
            return 10
        failures = []
        exceptions = []

        @TaskWithBrokenDependency.event_handler(Event.BROKEN_TASK)
        def failure(task, exception):
            if False:
                return 10
            failures.append(task)
            exceptions.append(exception)
        t = TaskWithBrokenDependency()
        build([t], local_scheduler=True)
        self.assertEqual(failures, [t])
        self.assertEqual(len(exceptions), 1)
        self.assertTrue(isinstance(exceptions[0], DummyException))

    def test_custom_handler(self):
        if False:
            for i in range(10):
                print('nop')
        dummies = []

        @TaskWithCallback.event_handler('foo event')
        def story_dummy():
            if False:
                while True:
                    i = 10
            dummies.append('foo')
        t = TaskWithCallback()
        build([t], local_scheduler=True)
        self.assertEqual(dummies[0], 'foo')

    def _run_processing_time_handler(self, fail):
        if False:
            for i in range(10):
                print('nop')
        result = []

        @EmptyTask.event_handler(Event.PROCESSING_TIME)
        def save_task(task, processing_time):
            if False:
                print('Hello World!')
            result.append((task, processing_time))
        times = [43.0, 1.0]
        t = EmptyTask(fail)
        with patch('luigi.worker.time') as mock:
            mock.time = times.pop
            build([t], local_scheduler=True)
        return (t, result)

    def test_processing_time_handler_success(self):
        if False:
            print('Hello World!')
        (t, result) = self._run_processing_time_handler(False)
        self.assertEqual(len(result), 1)
        (task, time) = result[0]
        self.assertTrue(task is t)
        self.assertEqual(time, 42.0)

    def test_processing_time_handler_failure(self):
        if False:
            for i in range(10):
                print('nop')
        (t, result) = self._run_processing_time_handler(True)
        self.assertEqual(result, [])

def eval_contents(f):
    if False:
        return 10
    with f.open('r') as i:
        return eval(i.read())

class ConsistentMockOutput:
    """
    Computes output location and contents from the task and its parameters. Rids us of writing ad-hoc boilerplate output() et al.
    """
    param = luigi.IntParameter(default=1)

    def output(self):
        if False:
            i = 10
            return i + 15
        return MockTarget('/%s/%u' % (self.__class__.__name__, self.param))

    def produce_output(self):
        if False:
            return 10
        with self.output().open('w') as o:
            o.write(repr([self.task_id] + sorted([eval_contents(i) for i in flatten(self.input())])))

class HappyTestFriend(ConsistentMockOutput, luigi.Task):
    """
    Does trivial "work", outputting the list of inputs. Results in a convenient lispy comparable.
    """

    def run(self):
        if False:
            i = 10
            return i + 15
        self.produce_output()

class D(ConsistentMockOutput, luigi.ExternalTask):
    pass

class C(HappyTestFriend):

    def requires(self):
        if False:
            print('Hello World!')
        return [D(self.param), D(self.param + 1)]

class B(HappyTestFriend):

    def requires(self):
        if False:
            return 10
        return C(self.param)

class A(HappyTestFriend):
    task_namespace = 'event_callbacks'

    def requires(self):
        if False:
            print('Hello World!')
        return [B(1), B(2)]

class TestDependencyEvents(unittest.TestCase):

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        MockFileSystem().remove('')

    def _run_test(self, task, expected_events):
        if False:
            i = 10
            return i + 15
        actual_events = {}

        @luigi.Task.event_handler(Event.DEPENDENCY_DISCOVERED)
        def callback_dependency_discovered(*args):
            if False:
                while True:
                    i = 10
            actual_events.setdefault(Event.DEPENDENCY_DISCOVERED, set()).add(tuple(map(lambda t: t.task_id, args)))

        @luigi.Task.event_handler(Event.DEPENDENCY_MISSING)
        def callback_dependency_missing(*args):
            if False:
                while True:
                    i = 10
            actual_events.setdefault(Event.DEPENDENCY_MISSING, set()).add(tuple(map(lambda t: t.task_id, args)))

        @luigi.Task.event_handler(Event.DEPENDENCY_PRESENT)
        def callback_dependency_present(*args):
            if False:
                for i in range(10):
                    print('nop')
            actual_events.setdefault(Event.DEPENDENCY_PRESENT, set()).add(tuple(map(lambda t: t.task_id, args)))
        build([task], local_scheduler=True)
        self.assertEqual(actual_events, expected_events)

    def test_incomplete_dag(self):
        if False:
            for i in range(10):
                print('nop')
        for param in range(1, 3):
            D(param).produce_output()
        self._run_test(A(), {'event.core.dependency.discovered': {(A(param=1).task_id, B(param=1).task_id), (A(param=1).task_id, B(param=2).task_id), (B(param=1).task_id, C(param=1).task_id), (B(param=2).task_id, C(param=2).task_id), (C(param=1).task_id, D(param=1).task_id), (C(param=1).task_id, D(param=2).task_id), (C(param=2).task_id, D(param=2).task_id), (C(param=2).task_id, D(param=3).task_id)}, 'event.core.dependency.missing': {(D(param=3).task_id,)}, 'event.core.dependency.present': {(D(param=1).task_id,), (D(param=2).task_id,)}})
        self.assertFalse(A().output().exists())

    def test_complete_dag(self):
        if False:
            for i in range(10):
                print('nop')
        for param in range(1, 4):
            D(param).produce_output()
        self._run_test(A(), {'event.core.dependency.discovered': {(A(param=1).task_id, B(param=1).task_id), (A(param=1).task_id, B(param=2).task_id), (B(param=1).task_id, C(param=1).task_id), (B(param=2).task_id, C(param=2).task_id), (C(param=1).task_id, D(param=1).task_id), (C(param=1).task_id, D(param=2).task_id), (C(param=2).task_id, D(param=2).task_id), (C(param=2).task_id, D(param=3).task_id)}, 'event.core.dependency.present': {(D(param=1).task_id,), (D(param=2).task_id,), (D(param=3).task_id,)}})
        self.assertEqual(eval_contents(A().output()), [A(param=1).task_id, [B(param=1).task_id, [C(param=1).task_id, [D(param=1).task_id], [D(param=2).task_id]]], [B(param=2).task_id, [C(param=2).task_id, [D(param=2).task_id], [D(param=3).task_id]]]])