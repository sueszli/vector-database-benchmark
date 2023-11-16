"""Unit tests for scripts/concurrent_task_utils.py."""
from __future__ import annotations
import builtins
import threading
import time
from core.tests import test_utils
from scripts import concurrent_task_utils
from typing import Callable, List

def test_function(unused_arg: str) -> Callable[[], None]:
    if False:
        return 10

    def task_func() -> None:
        if False:
            for i in range(10):
                print('nop')
        pass
    return task_func

class ConcurrentTaskUtilsTests(test_utils.GenericTestBase):
    """Test for concurrent_task_utils.py flie."""

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        super().setUp()
        self.semaphore = threading.Semaphore(1)
        self.task_stdout: List[str] = []

        def mock_print(*args: str) -> None:
            if False:
                for i in range(10):
                    print('nop')
            'Mock for print. Append the values to print to\n            task_stdout list.\n\n            Args:\n                *args: list(*). Variable length argument list of values to print\n                    in the same line of output.\n            '
            self.task_stdout.append(' '.join((str(arg) for arg in args)))
        self.print_swap = self.swap(builtins, 'print', mock_print)

class TaskResultTests(ConcurrentTaskUtilsTests):
    """Tests for TaskResult class."""

    def test_all_messages_with_success_message(self) -> None:
        if False:
            print('Hello World!')
        output_object = concurrent_task_utils.TaskResult('Test', False, [], [])
        self.assertEqual(output_object.trimmed_messages, [])
        self.assertEqual(output_object.get_report(), ['SUCCESS  Test check passed'])
        self.assertFalse(output_object.failed)
        self.assertEqual(output_object.name, 'Test')

    def test_all_messages_with_failed_message(self) -> None:
        if False:
            i = 10
            return i + 15
        output_object = concurrent_task_utils.TaskResult('Test', True, [], [])
        self.assertEqual(output_object.trimmed_messages, [])
        self.assertEqual(output_object.get_report(), ['FAILED  Test check failed'])
        self.assertTrue(output_object.failed)
        self.assertEqual(output_object.name, 'Test')

class CreateTaskTests(ConcurrentTaskUtilsTests):
    """Tests for create_task method."""

    def test_create_task_with_success(self) -> None:
        if False:
            while True:
                i = 10
        task = concurrent_task_utils.create_task(test_function, True, self.semaphore)
        self.assertTrue(isinstance(task, concurrent_task_utils.TaskThread))

class TaskThreadTests(ConcurrentTaskUtilsTests):
    """Tests for TaskThread class."""

    def test_task_thread_with_success(self) -> None:
        if False:
            return 10
        task = concurrent_task_utils.TaskThread(test_function('unused_arg'), False, self.semaphore, name='test', report_enabled=True)
        self.semaphore.acquire()
        task.start_time = time.time()
        with self.print_swap:
            task.start()
            task.join()
        expected_output = [s for s in self.task_stdout if 'FINISHED' in s]
        self.assertTrue(len(expected_output) == 1)

    def test_task_thread_with_exception(self) -> None:
        if False:
            while True:
                i = 10
        task = concurrent_task_utils.TaskThread(test_function, True, self.semaphore, name='test', report_enabled=True)
        self.semaphore.acquire()
        task.start_time = time.time()
        with self.print_swap:
            task.start()
            task.join()
        self.assertIn("test_function() missing 1 required positional argument: 'unused_arg'", self.task_stdout)

    def test_task_thread_with_verbose_mode_enabled(self) -> None:
        if False:
            while True:
                i = 10

        class HelperTests:

            def test_show(self) -> concurrent_task_utils.TaskResult:
                if False:
                    while True:
                        i = 10
                return concurrent_task_utils.TaskResult('name', True, [], [])

            def test_perform_all_check(self) -> List[concurrent_task_utils.TaskResult]:
                if False:
                    i = 10
                    return i + 15
                return [self.test_show()]

        def test_func() -> HelperTests:
            if False:
                print('Hello World!')
            return HelperTests()
        task = concurrent_task_utils.TaskThread(test_func().test_perform_all_check, True, self.semaphore, name='test', report_enabled=True)
        self.semaphore.acquire()
        task.start_time = time.time()
        with self.print_swap:
            task.start()
            task.join()
        self.assertRegex(self.task_stdout[0], '\\d+:\\d+:\\d+ Report from name check\\n-+\\nFAILED  name check failed')

    def test_task_thread_with_task_report_disabled(self) -> None:
        if False:
            i = 10
            return i + 15

        class HelperTests:

            def test_show(self) -> concurrent_task_utils.TaskResult:
                if False:
                    i = 10
                    return i + 15
                return concurrent_task_utils.TaskResult('', False, [], ['msg'])

            def test_perform_all_check(self) -> List[concurrent_task_utils.TaskResult]:
                if False:
                    for i in range(10):
                        print('nop')
                return [self.test_show()]

        def test_func() -> HelperTests:
            if False:
                print('Hello World!')
            return HelperTests()
        task = concurrent_task_utils.TaskThread(test_func().test_perform_all_check, True, self.semaphore, name='test', report_enabled=False)
        self.semaphore.acquire()
        task.start_time = time.time()
        with self.print_swap:
            task.start()
            task.join()
        expected_output = [s for s in self.task_stdout if 'FINISHED' in s]
        self.assertTrue(len(expected_output) == 1)

class ExecuteTasksTests(ConcurrentTaskUtilsTests):
    """Tests for execute_tasks method."""

    def test_execute_task_with_single_task(self) -> None:
        if False:
            while True:
                i = 10
        task = concurrent_task_utils.create_task(test_function('unused_arg'), False, self.semaphore, name='test')
        with self.print_swap:
            concurrent_task_utils.execute_tasks([task], self.semaphore)
        expected_output = [s for s in self.task_stdout if 'FINISHED' in s]
        self.assertTrue(len(expected_output) == 1)

    def test_execute_task_with_multiple_task(self) -> None:
        if False:
            while True:
                i = 10
        task_list = []
        for _ in range(6):
            task = concurrent_task_utils.create_task(test_function('unused_arg'), False, self.semaphore)
            task_list.append(task)
        with self.print_swap:
            concurrent_task_utils.execute_tasks(task_list, self.semaphore)
        expected_output = [s for s in self.task_stdout if 'FINISHED' in s]
        self.assertTrue(len(expected_output) == 6)

    def test_execute_task_with_exception(self) -> None:
        if False:
            while True:
                i = 10
        task_list = []
        for _ in range(6):
            task = concurrent_task_utils.create_task(test_function, True, self.semaphore)
            task_list.append(task)
        with self.print_swap:
            concurrent_task_utils.execute_tasks(task_list, self.semaphore)
        self.assertIn("test_function() missing 1 required positional argument: 'unused_arg'", self.task_stdout)