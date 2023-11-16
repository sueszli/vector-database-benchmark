import errno
from functools import total_ordering
import gzip
import io
import json
import multiprocessing
import optparse
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
if sys.version_info.major >= 3:
    long = int
    import _pickle as cPickle
    import _thread as thread
else:
    import cPickle
    import thread
from pickle import HIGHEST_PROTOCOL as PICKLE_HIGHEST_PROTOCOL
if sys.platform == 'win32':
    import msvcrt
else:
    import fcntl

class SigintHandler(object):

    class ProcessWasInterrupted(Exception):
        pass
    sigint_returncodes = {-signal.SIGINT, -1073741510}

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.__lock = threading.Lock()
        self.__processes = set()
        self.__got_sigint = False
        signal.signal(signal.SIGINT, lambda signal_num, frame: self.interrupt())

    def __on_sigint(self):
        if False:
            print('Hello World!')
        self.__got_sigint = True
        while self.__processes:
            try:
                self.__processes.pop().terminate()
            except OSError:
                pass

    def interrupt(self):
        if False:
            print('Hello World!')
        with self.__lock:
            self.__on_sigint()

    def got_sigint(self):
        if False:
            while True:
                i = 10
        with self.__lock:
            return self.__got_sigint

    def wait(self, p):
        if False:
            for i in range(10):
                print('nop')
        with self.__lock:
            if self.__got_sigint:
                p.terminate()
            self.__processes.add(p)
        code = p.wait()
        with self.__lock:
            self.__processes.discard(p)
            if code in self.sigint_returncodes:
                self.__on_sigint()
            if self.__got_sigint:
                raise self.ProcessWasInterrupted
        return code
sigint_handler = SigintHandler()

def term_width(out):
    if False:
        print('Hello World!')
    if not out.isatty():
        return None
    try:
        p = subprocess.Popen(['stty', 'size'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out, err) = p.communicate()
        if p.returncode != 0 or err:
            return None
        return int(out.split()[1])
    except (IndexError, OSError, ValueError):
        return None

class Outputter(object):

    def __init__(self, out_file):
        if False:
            for i in range(10):
                print('nop')
        self.__out_file = out_file
        self.__previous_line_was_transient = False
        self.__width = term_width(out_file)

    def transient_line(self, msg):
        if False:
            for i in range(10):
                print('nop')
        if self.__width is None:
            self.__out_file.write(msg + '\n')
        else:
            self.__out_file.write('\r' + msg[:self.__width].ljust(self.__width))
            self.__previous_line_was_transient = True

    def flush_transient_output(self):
        if False:
            print('Hello World!')
        if self.__previous_line_was_transient:
            self.__out_file.write('\n')
            self.__previous_line_was_transient = False

    def permanent_line(self, msg):
        if False:
            while True:
                i = 10
        self.flush_transient_output()
        self.__out_file.write(msg + '\n')

def get_save_file_path():
    if False:
        i = 10
        return i + 15
    'Return path to file for saving transient data.'
    if sys.platform == 'win32':
        default_cache_path = os.path.join(os.path.expanduser('~'), 'AppData', 'Local')
        cache_path = os.environ.get('LOCALAPPDATA', default_cache_path)
    else:
        default_cache_path = os.path.join(os.path.expanduser('~'), '.cache')
        cache_path = os.environ.get('XDG_CACHE_HOME', default_cache_path)
    if os.path.isdir(cache_path):
        return os.path.join(cache_path, 'gtest-parallel')
    else:
        sys.stderr.write('Directory {} does not exist'.format(cache_path))
        return os.path.join(os.path.expanduser('~'), '.gtest-parallel-times')

@total_ordering
class Task(object):
    """Stores information about a task (single execution of a test).

  This class stores information about the test to be executed (gtest binary and
  test name), and its result (log file, exit code and runtime).
  Each task is uniquely identified by the gtest binary, the test name and an
  execution number that increases each time the test is executed.
  Additionaly we store the last execution time, so that next time the test is
  executed, the slowest tests are run first.
  """

    def __init__(self, test_binary, test_name, test_command, execution_number, last_execution_time, output_dir):
        if False:
            return 10
        self.test_name = test_name
        self.output_dir = output_dir
        self.test_binary = test_binary
        self.test_command = test_command
        self.execution_number = execution_number
        self.last_execution_time = last_execution_time
        self.exit_code = None
        self.runtime_ms = None
        self.test_id = (test_binary, test_name)
        self.task_id = (test_binary, test_name, self.execution_number)
        self.log_file = Task._logname(self.output_dir, self.test_binary, test_name, self.execution_number)

    def __sorting_key(self):
        if False:
            while True:
                i = 10
        return (1 if self.last_execution_time is None else 0, self.last_execution_time)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.__sorting_key() == other.__sorting_key()

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return not self == other

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        return self.__sorting_key() < other.__sorting_key()

    @staticmethod
    def _normalize(string):
        if False:
            for i in range(10):
                print('nop')
        return re.sub('[^A-Za-z0-9]', '_', string)

    @staticmethod
    def _logname(output_dir, test_binary, test_name, execution_number):
        if False:
            return 10
        if output_dir is None:
            (log_handle, log_name) = tempfile.mkstemp(prefix='gtest_parallel_', suffix='.log')
            os.close(log_handle)
            return log_name
        log_name = '%s-%s-%d.log' % (Task._normalize(os.path.basename(test_binary)), Task._normalize(test_name), execution_number)
        return os.path.join(output_dir, log_name)

    def run(self):
        if False:
            i = 10
            return i + 15
        begin = time.time()
        with open(self.log_file, 'w') as log:
            task = subprocess.Popen(self.test_command, stdout=log, stderr=log)
            try:
                self.exit_code = sigint_handler.wait(task)
            except sigint_handler.ProcessWasInterrupted:
                thread.exit()
        self.runtime_ms = int(1000 * (time.time() - begin))
        self.last_execution_time = None if self.exit_code else self.runtime_ms

class TaskManager(object):
    """Executes the tasks and stores the passed, failed and interrupted tasks.

  When a task is run, this class keeps track if it passed, failed or was
  interrupted. After a task finishes it calls the relevant functions of the
  Logger, TestResults and TestTimes classes, and in case of failure, retries the
  test as specified by the --retry_failed flag.
  """

    def __init__(self, times, logger, test_results, task_factory, times_to_retry, initial_execution_number):
        if False:
            i = 10
            return i + 15
        self.times = times
        self.logger = logger
        self.test_results = test_results
        self.task_factory = task_factory
        self.times_to_retry = times_to_retry
        self.initial_execution_number = initial_execution_number
        self.global_exit_code = 0
        self.passed = []
        self.failed = []
        self.started = {}
        self.execution_number = {}
        self.lock = threading.Lock()

    def __get_next_execution_number(self, test_id):
        if False:
            while True:
                i = 10
        with self.lock:
            next_execution_number = self.execution_number.setdefault(test_id, self.initial_execution_number)
            self.execution_number[test_id] += 1
        return next_execution_number

    def __register_start(self, task):
        if False:
            while True:
                i = 10
        with self.lock:
            self.started[task.task_id] = task

    def __register_exit(self, task):
        if False:
            print('Hello World!')
        self.logger.log_exit(task)
        self.times.record_test_time(task.test_binary, task.test_name, task.last_execution_time)
        if self.test_results:
            self.test_results.log(task.test_name, task.runtime_ms, 'PASS' if task.exit_code == 0 else 'FAIL')
        with self.lock:
            self.started.pop(task.task_id)
            if task.exit_code == 0:
                self.passed.append(task)
            else:
                self.failed.append(task)

    def run_task(self, task):
        if False:
            return 10
        for try_number in range(self.times_to_retry + 1):
            self.__register_start(task)
            task.run()
            self.__register_exit(task)
            if task.exit_code == 0:
                break
            if try_number < self.times_to_retry:
                execution_number = self.__get_next_execution_number(task.test_id)
                task = self.task_factory(task.test_binary, task.test_name, task.test_command, execution_number, task.last_execution_time, task.output_dir)
        with self.lock:
            if task.exit_code != 0:
                self.global_exit_code = task.exit_code

class FilterFormat(object):

    def __init__(self, output_dir):
        if False:
            return 10
        if sys.stdout.isatty():
            if isinstance(sys.stdout, io.TextIOWrapper):
                sys.stdout = io.TextIOWrapper(sys.stdout.detach(), line_buffering=True, write_through=True, newline='\n')
            else:
                sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
        self.output_dir = output_dir
        self.total_tasks = 0
        self.finished_tasks = 0
        self.out = Outputter(sys.stdout)
        self.stdout_lock = threading.Lock()

    def move_to(self, destination_dir, tasks):
        if False:
            for i in range(10):
                print('nop')
        if self.output_dir is None:
            return
        destination_dir = os.path.join(self.output_dir, destination_dir)
        os.makedirs(destination_dir)
        for task in tasks:
            shutil.move(task.log_file, destination_dir)

    def print_tests(self, message, tasks, print_try_number):
        if False:
            return 10
        self.out.permanent_line('%s (%s/%s):' % (message, len(tasks), self.total_tasks))
        for task in sorted(tasks):
            runtime_ms = 'Interrupted'
            if task.runtime_ms is not None:
                runtime_ms = '%d ms' % task.runtime_ms
            self.out.permanent_line('%11s: %s %s%s' % (runtime_ms, task.test_binary, task.test_name, ' (try #%d)' % task.execution_number if print_try_number else ''))

    def log_exit(self, task):
        if False:
            for i in range(10):
                print('nop')
        with self.stdout_lock:
            self.finished_tasks += 1
            self.out.transient_line('[%d/%d] %s (%d ms)' % (self.finished_tasks, self.total_tasks, task.test_name, task.runtime_ms))
            if task.exit_code != 0:
                with open(task.log_file) as f:
                    for line in f.readlines():
                        self.out.permanent_line(line.rstrip())
                self.out.permanent_line('[%d/%d] %s returned/aborted with exit code %d (%d ms)' % (self.finished_tasks, self.total_tasks, task.test_name, task.exit_code, task.runtime_ms))
        if self.output_dir is None:
            num_tries = 100
            for i in range(num_tries):
                try:
                    os.remove(task.log_file)
                except OSError as e:
                    if e.errno is not errno.ENOENT:
                        if i is num_tries - 1:
                            self.out.permanent_line('Could not remove temporary log file: ' + str(e))
                        else:
                            time.sleep(0.1)
                        continue
                break

    def log_tasks(self, total_tasks):
        if False:
            print('Hello World!')
        self.total_tasks += total_tasks
        self.out.transient_line('[0/%d] Running tests...' % self.total_tasks)

    def summarize(self, passed_tasks, failed_tasks, interrupted_tasks):
        if False:
            while True:
                i = 10
        stats = {}

        def add_stats(stats, task, idx):
            if False:
                i = 10
                return i + 15
            task_key = (task.test_binary, task.test_name)
            if not task_key in stats:
                stats[task_key] = [0, 0, 0, task_key]
            stats[task_key][idx] += 1
        for task in passed_tasks:
            add_stats(stats, task, 0)
        for task in failed_tasks:
            add_stats(stats, task, 1)
        for task in interrupted_tasks:
            add_stats(stats, task, 2)
        self.out.permanent_line('SUMMARY:')
        for task_key in sorted(stats, key=stats.__getitem__):
            (num_passed, num_failed, num_interrupted, _) = stats[task_key]
            (test_binary, task_name) = task_key
            total_runs = num_passed + num_failed + num_interrupted
            if num_passed == total_runs:
                continue
            self.out.permanent_line('  %s %s passed %d / %d times%s.' % (test_binary, task_name, num_passed, total_runs, '' if num_interrupted == 0 else ' (%d interrupted)' % num_interrupted))

    def flush(self):
        if False:
            i = 10
            return i + 15
        self.out.flush_transient_output()

class CollectTestResults(object):

    def __init__(self, json_dump_filepath):
        if False:
            for i in range(10):
                print('nop')
        self.test_results_lock = threading.Lock()
        self.json_dump_file = open(json_dump_filepath, 'w')
        self.test_results = {'interrupted': False, 'path_delimiter': '.', 'version': 3, 'seconds_since_epoch': int(time.time()), 'num_failures_by_type': {'PASS': 0, 'FAIL': 0}, 'tests': {}}

    def log(self, test, runtime_ms, actual_result):
        if False:
            while True:
                i = 10
        with self.test_results_lock:
            self.test_results['num_failures_by_type'][actual_result] += 1
            results = self.test_results['tests']
            for name in test.split('.'):
                results = results.setdefault(name, {})
            if results:
                results['actual'] += ' ' + actual_result
                results['times'].append(runtime_ms)
            else:
                results['actual'] = actual_result
                results['times'] = [runtime_ms]
                results['time'] = runtime_ms
                results['expected'] = 'PASS'

    def dump_to_file_and_close(self):
        if False:
            i = 10
            return i + 15
        json.dump(self.test_results, self.json_dump_file)
        self.json_dump_file.close()

class TestTimes(object):

    class LockedFile(object):

        def __init__(self, filename, mode):
            if False:
                i = 10
                return i + 15
            self._filename = filename
            self._mode = mode
            self._fo = None

        def __enter__(self):
            if False:
                i = 10
                return i + 15
            self._fo = open(self._filename, self._mode)
            self._fo.seek(0)
            try:
                if sys.platform == 'win32':
                    msvcrt.locking(self._fo.fileno(), msvcrt.LK_LOCK, 1)
                else:
                    fcntl.flock(self._fo.fileno(), fcntl.LOCK_EX)
            except IOError:
                self._fo.close()
                raise
            return self._fo

        def __exit__(self, exc_type, exc_value, traceback):
            if False:
                i = 10
                return i + 15
            self._fo.flush()
            try:
                if sys.platform == 'win32':
                    self._fo.seek(0)
                    msvcrt.locking(self._fo.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(self._fo.fileno(), fcntl.LOCK_UN)
            finally:
                self._fo.close()
            return exc_value is None

    def __init__(self, save_file):
        if False:
            return 10
        'Create new object seeded with saved test times from the given file.'
        self.__times = {}
        self.__lock = threading.Lock()
        try:
            with TestTimes.LockedFile(save_file, 'rb') as fd:
                times = TestTimes.__read_test_times_file(fd)
        except IOError:
            return
        if type(times) is not dict:
            return
        for ((test_binary, test_name), runtime) in times.items():
            if type(test_binary) is not str or type(test_name) is not str or type(runtime) not in {int, long, type(None)}:
                return
        self.__times = times

    def get_test_time(self, binary, testname):
        if False:
            return 10
        "Return the last duration for the given test as an integer number of\n    milliseconds, or None if the test failed or if there's no record for it."
        return self.__times.get((binary, testname), None)

    def record_test_time(self, binary, testname, runtime_ms):
        if False:
            print('Hello World!')
        'Record that the given test ran in the specified number of\n    milliseconds. If the test failed, runtime_ms should be None.'
        with self.__lock:
            self.__times[binary, testname] = runtime_ms

    def write_to_file(self, save_file):
        if False:
            i = 10
            return i + 15
        'Write all the times to file.'
        try:
            with TestTimes.LockedFile(save_file, 'a+b') as fd:
                times = TestTimes.__read_test_times_file(fd)
                if times is None:
                    times = self.__times
                else:
                    times.update(self.__times)
                fd.seek(0)
                fd.truncate()
                with gzip.GzipFile(fileobj=fd, mode='wb') as gzf:
                    cPickle.dump(times, gzf, PICKLE_HIGHEST_PROTOCOL)
        except IOError:
            pass

    @staticmethod
    def __read_test_times_file(fd):
        if False:
            i = 10
            return i + 15
        try:
            with gzip.GzipFile(fileobj=fd, mode='rb') as gzf:
                times = cPickle.load(gzf)
        except Exception:
            return None
        else:
            return times

def find_tests(binaries, additional_args, options, times):
    if False:
        print('Hello World!')
    test_count = 0
    tasks = []
    for test_binary in binaries:
        command = [test_binary]
        if options.gtest_also_run_disabled_tests:
            command += ['--gtest_also_run_disabled_tests']
        list_command = command + ['--gtest_list_tests']
        if options.gtest_filter != '':
            list_command += ['--gtest_filter=' + options.gtest_filter]
        try:
            test_list = subprocess.check_output(list_command, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            sys.exit('%s: %s\n%s' % (test_binary, str(e), e.output))
        try:
            test_list = test_list.split('\n')
        except TypeError:
            test_list = test_list.decode(sys.stdout.encoding).split('\n')
        command += additional_args + ['--gtest_color=' + options.gtest_color]
        test_group = ''
        for line in test_list:
            if not line.strip():
                continue
            if line[0] != ' ':
                test_group = line.split('#')[0].strip()
                continue
            line = line.split('#')[0].strip()
            if not line:
                continue
            test_name = test_group + line
            if not options.gtest_also_run_disabled_tests and 'DISABLED_' in test_name:
                continue
            last_execution_time = times.get_test_time(test_binary, test_name)
            if options.failed and last_execution_time is not None:
                continue
            test_command = command + ['--gtest_filter=' + test_name]
            if (test_count - options.shard_index) % options.shard_count == 0:
                for execution_number in range(options.repeat):
                    tasks.append(Task(test_binary, test_name, test_command, execution_number + 1, last_execution_time, options.output_dir))
            test_count += 1
    return sorted(tasks, reverse=True)

def execute_tasks(tasks, pool_size, task_manager, timeout, serialize_test_cases):
    if False:
        while True:
            i = 10

    class WorkerFn(object):

        def __init__(self, tasks, running_groups):
            if False:
                for i in range(10):
                    print('nop')
            self.tasks = tasks
            self.running_groups = running_groups
            self.task_lock = threading.Lock()

        def __call__(self):
            if False:
                return 10
            while True:
                with self.task_lock:
                    for task_id in range(len(self.tasks)):
                        task = self.tasks[task_id]
                        if self.running_groups is not None:
                            test_group = task.test_name.split('.')[0]
                            if test_group in self.running_groups:
                                continue
                            else:
                                self.running_groups.add(test_group)
                        del self.tasks[task_id]
                        break
                    else:
                        return
                task_manager.run_task(task)
                if self.running_groups is not None:
                    with self.task_lock:
                        self.running_groups.remove(test_group)

    def start_daemon(func):
        if False:
            print('Hello World!')
        t = threading.Thread(target=func)
        t.daemon = True
        t.start()
        return t
    try:
        if timeout:
            timeout.start()
        running_groups = set() if serialize_test_cases else None
        worker_fn = WorkerFn(tasks, running_groups)
        workers = [start_daemon(worker_fn) for _ in range(pool_size)]
        for worker in workers:
            worker.join()
    finally:
        if timeout:
            timeout.cancel()

def default_options_parser():
    if False:
        return 10
    parser = optparse.OptionParser(usage='usage: %prog [options] binary [binary ...] -- [additional args]')
    parser.add_option('-d', '--output_dir', type='string', default=None, help='Output directory for test logs. Logs will be available under gtest-parallel-logs/, so --output_dir=/tmp will results in all logs being available under /tmp/gtest-parallel-logs/.')
    parser.add_option('-r', '--repeat', type='int', default=1, help='Number of times to execute all the tests.')
    parser.add_option('--retry_failed', type='int', default=0, help='Number of times to repeat failed tests.')
    parser.add_option('--failed', action='store_true', default=False, help='run only failed and new tests')
    parser.add_option('-w', '--workers', type='int', default=multiprocessing.cpu_count(), help='number of workers to spawn')
    parser.add_option('--gtest_color', type='string', default='yes', help='color output')
    parser.add_option('--gtest_filter', type='string', default='', help='test filter')
    parser.add_option('--gtest_also_run_disabled_tests', action='store_true', default=False, help='run disabled tests too')
    parser.add_option('--print_test_times', action='store_true', default=False, help='list the run time of each test at the end of execution')
    parser.add_option('--shard_count', type='int', default=1, help='total number of shards (for sharding test execution between multiple machines)')
    parser.add_option('--shard_index', type='int', default=0, help='zero-indexed number identifying this shard (for sharding test execution between multiple machines)')
    parser.add_option('--dump_json_test_results', type='string', default=None, help='Saves the results of the tests as a JSON machine-readable file. The format of the file is specified at https://www.chromium.org/developers/the-json-test-results-format')
    parser.add_option('--timeout', type='int', default=None, help='Interrupt all remaining processes after the given time (in seconds).')
    parser.add_option('--serialize_test_cases', action='store_true', default=False, help='Do not run tests from the same test case in parallel.')
    return parser

def main():
    if False:
        for i in range(10):
            print('nop')
    additional_args = []
    for i in range(len(sys.argv)):
        if sys.argv[i] == '--':
            additional_args = sys.argv[i + 1:]
            sys.argv = sys.argv[:i]
            break
    parser = default_options_parser()
    (options, binaries) = parser.parse_args()
    if options.output_dir is not None and (not os.path.isdir(options.output_dir)):
        parser.error('--output_dir value must be an existing directory, current value is "%s"' % options.output_dir)
    if options.output_dir:
        options.output_dir = os.path.join(options.output_dir, 'gtest-parallel-logs')
    if binaries == []:
        parser.print_usage()
        sys.exit(1)
    if options.shard_count < 1:
        parser.error('Invalid number of shards: %d. Must be at least 1.' % options.shard_count)
    if not 0 <= options.shard_index < options.shard_count:
        parser.error('Invalid shard index: %d. Must be between 0 and %d (less than the number of shards).' % (options.shard_index, options.shard_count - 1))
    unique_binaries = set((os.path.basename(binary) for binary in binaries))
    assert len(unique_binaries) == len(binaries), 'All test binaries must have an unique basename.'
    if options.output_dir:
        if os.path.isdir(options.output_dir):
            shutil.rmtree(options.output_dir)
        try:
            os.makedirs(options.output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST or not os.path.isdir(options.output_dir):
                raise e
    timeout = None
    if options.timeout is not None:
        timeout = threading.Timer(options.timeout, sigint_handler.interrupt)
    test_results = None
    if options.dump_json_test_results is not None:
        test_results = CollectTestResults(options.dump_json_test_results)
    save_file = get_save_file_path()
    times = TestTimes(save_file)
    logger = FilterFormat(options.output_dir)
    task_manager = TaskManager(times, logger, test_results, Task, options.retry_failed, options.repeat + 1)
    tasks = find_tests(binaries, additional_args, options, times)
    logger.log_tasks(len(tasks))
    execute_tasks(tasks, options.workers, task_manager, timeout, options.serialize_test_cases)
    print_try_number = options.retry_failed > 0 or options.repeat > 1
    if task_manager.passed:
        logger.move_to('passed', task_manager.passed)
        if options.print_test_times:
            logger.print_tests('PASSED TESTS', task_manager.passed, print_try_number)
    if task_manager.failed:
        logger.print_tests('FAILED TESTS', task_manager.failed, print_try_number)
        logger.move_to('failed', task_manager.failed)
    if task_manager.started:
        logger.print_tests('INTERRUPTED TESTS', task_manager.started.values(), print_try_number)
        logger.move_to('interrupted', task_manager.started.values())
    if options.repeat > 1 and (task_manager.failed or task_manager.started):
        logger.summarize(task_manager.passed, task_manager.failed, task_manager.started.values())
    logger.flush()
    times.write_to_file(save_file)
    if test_results:
        test_results.dump_to_file_and_close()
    if sigint_handler.got_sigint():
        return -signal.SIGINT
    return task_manager.global_exit_code
if __name__ == '__main__':
    sys.exit(main())