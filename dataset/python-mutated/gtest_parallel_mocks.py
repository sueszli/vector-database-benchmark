import collections
import threading
import time

class LoggerMock(object):

    def __init__(self, test_lib):
        if False:
            while True:
                i = 10
        self.test_lib = test_lib
        self.runtimes = collections.defaultdict(list)
        self.exit_codes = collections.defaultdict(list)
        self.last_execution_times = collections.defaultdict(list)
        self.execution_numbers = collections.defaultdict(list)

    def log_exit(self, task):
        if False:
            print('Hello World!')
        self.runtimes[task.test_id].append(task.runtime_ms)
        self.exit_codes[task.test_id].append(task.exit_code)
        self.last_execution_times[task.test_id].append(task.last_execution_time)
        self.execution_numbers[task.test_id].append(task.execution_number)

    def assertRecorded(self, test_id, expected, retries):
        if False:
            for i in range(10):
                print('nop')
        self.test_lib.assertIn(test_id, self.runtimes)
        self.test_lib.assertListEqual(expected['runtime_ms'][:retries], self.runtimes[test_id])
        self.test_lib.assertListEqual(expected['exit_code'][:retries], self.exit_codes[test_id])
        self.test_lib.assertListEqual(expected['last_execution_time'][:retries], self.last_execution_times[test_id])
        self.test_lib.assertListEqual(expected['execution_number'][:retries], self.execution_numbers[test_id])

class TestTimesMock(object):

    def __init__(self, test_lib, test_data=None):
        if False:
            return 10
        self.test_lib = test_lib
        self.test_data = test_data or {}
        self.last_execution_times = collections.defaultdict(list)

    def record_test_time(self, test_binary, test_name, last_execution_time):
        if False:
            while True:
                i = 10
        test_id = (test_binary, test_name)
        self.last_execution_times[test_id].append(last_execution_time)

    def get_test_time(self, test_binary, test_name):
        if False:
            for i in range(10):
                print('nop')
        (test_group, test) = test_name.split('.')
        return self.test_data.get(test_binary, {}).get(test_group, {}).get(test, None)

    def assertRecorded(self, test_id, expected, retries):
        if False:
            print('Hello World!')
        self.test_lib.assertIn(test_id, self.last_execution_times)
        self.test_lib.assertListEqual(expected['last_execution_time'][:retries], self.last_execution_times[test_id])

class TestResultsMock(object):

    def __init__(self, test_lib):
        if False:
            return 10
        self.results = []
        self.test_lib = test_lib

    def log(self, test_name, runtime_ms, actual_result):
        if False:
            print('Hello World!')
        self.results.append((test_name, runtime_ms, actual_result))

    def assertRecorded(self, test_id, expected, retries):
        if False:
            for i in range(10):
                print('nop')
        test_results = [(test_id[1], runtime_ms, 'PASS' if exit_code == 0 else 'FAIL') for (runtime_ms, exit_code) in zip(expected['runtime_ms'][:retries], expected['exit_code'][:retries])]
        for test_result in test_results:
            self.test_lib.assertIn(test_result, self.results)

class TaskManagerMock(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.running_groups = []
        self.check_lock = threading.Lock()
        self.had_running_parallel_groups = False
        self.total_tasks_run = 0

    def run_task(self, task):
        if False:
            while True:
                i = 10
        test_group = task.test_name.split('.')[0]
        with self.check_lock:
            self.total_tasks_run += 1
            if test_group in self.running_groups:
                self.had_running_parallel_groups = True
            self.running_groups.append(test_group)
        time.sleep(0.001)
        with self.check_lock:
            self.running_groups.remove(test_group)

class TaskMockFactory(object):

    def __init__(self, test_data):
        if False:
            for i in range(10):
                print('nop')
        self.data = test_data
        self.passed = []
        self.failed = []

    def get_task(self, test_id, execution_number=0):
        if False:
            print('Hello World!')
        task = TaskMock(test_id, execution_number, self.data[test_id])
        if task.exit_code == 0:
            self.passed.append(task)
        else:
            self.failed.append(task)
        return task

    def __call__(self, test_binary, test_name, test_command, execution_number, last_execution_time, output_dir):
        if False:
            return 10
        return self.get_task((test_binary, test_name), execution_number)

class TaskMock(object):

    def __init__(self, test_id, execution_number, test_data):
        if False:
            print('Hello World!')
        self.test_id = test_id
        self.execution_number = execution_number
        self.runtime_ms = test_data['runtime_ms'][execution_number]
        self.exit_code = test_data['exit_code'][execution_number]
        self.last_execution_time = test_data['last_execution_time'][execution_number]
        if 'log_file' in test_data:
            self.log_file = test_data['log_file'][execution_number]
        else:
            self.log_file = None
        self.test_command = None
        self.output_dir = None
        self.test_binary = test_id[0]
        self.test_name = test_id[1]
        self.task_id = (test_id[0], test_id[1], execution_number)

    def run(self):
        if False:
            while True:
                i = 10
        pass

class SubprocessMock(object):

    def __init__(self, test_data=None):
        if False:
            print('Hello World!')
        self._test_data = test_data
        self.last_invocation = None

    def __call__(self, command, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.last_invocation = command
        binary = command[0]
        test_list = []
        tests_for_binary = sorted(self._test_data.get(binary, {}).items())
        for (test_group, tests) in tests_for_binary:
            test_list.append(test_group + '.')
            for test in sorted(tests):
                test_list.append('  ' + test)
        return '\n'.join(test_list)