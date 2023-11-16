import collections
import faulthandler
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
from typing import NamedTuple, NoReturn, Literal, Any
from test import support
from test.support import os_helper
from test.libregrtest.cmdline import Namespace
from test.libregrtest.main import Regrtest
from test.libregrtest.runtest import runtest, is_failed, TestResult, Interrupted, Timeout, ChildError, PROGRESS_MIN_TIME
from test.libregrtest.setup import setup_tests
from test.libregrtest.utils import format_duration, print_warning
PROGRESS_UPDATE = 30.0
assert PROGRESS_UPDATE >= PROGRESS_MIN_TIME
MAIN_PROCESS_TIMEOUT = 5 * 60.0
assert MAIN_PROCESS_TIMEOUT >= PROGRESS_UPDATE
JOIN_TIMEOUT = 30.0
USE_PROCESS_GROUP = hasattr(os, 'setsid') and hasattr(os, 'killpg')

def get_cinderjit_xargs():
    if False:
        print('Hello World!')
    args = []
    for (k, v) in sys._xoptions.items():
        if not k.startswith('jit'):
            continue
        elif v is True:
            args.extend(['-X', k])
        else:
            args.extend(['-X', f'{k}={v}'])
    return args

def must_stop(result: TestResult, ns: Namespace) -> bool:
    if False:
        while True:
            i = 10
    if isinstance(result, Interrupted):
        return True
    if ns.failfast and is_failed(result, ns):
        return True
    return False

def parse_worker_args(worker_args) -> tuple[Namespace, str]:
    if False:
        return 10
    (ns_dict, test_name) = json.loads(worker_args)
    ns = Namespace(**ns_dict)
    return (ns, test_name)

def run_test_in_subprocess(testname: str, ns: Namespace) -> subprocess.Popen:
    if False:
        return 10
    ns_dict = vars(ns)
    worker_args = (ns_dict, testname)
    worker_args = json.dumps(worker_args)
    cmd = [sys.executable, *get_cinderjit_xargs(), *support.args_from_interpreter_flags(), '-u', '-m', 'test.regrtest', '--worker-args', worker_args]
    kw = {}
    if USE_PROCESS_GROUP:
        kw['start_new_session'] = True
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, close_fds=os.name != 'nt', cwd=os_helper.SAVEDCWD, **kw)

def run_tests_worker(ns: Namespace, test_name: str) -> NoReturn:
    if False:
        while True:
            i = 10
    setup_tests(ns)
    result = runtest(ns, test_name)
    print()
    print(json.dumps(result, cls=EncodeTestResult), flush=True)
    sys.exit(0)

class MultiprocessIterator:
    """A thread-safe iterator over tests for multiprocess mode."""

    def __init__(self, tests_iter):
        if False:
            for i in range(10):
                print('nop')
        self.lock = threading.Lock()
        self.tests_iter = tests_iter

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __next__(self):
        if False:
            i = 10
            return i + 15
        with self.lock:
            if self.tests_iter is None:
                raise StopIteration
            return next(self.tests_iter)

    def stop(self):
        if False:
            i = 10
            return i + 15
        with self.lock:
            self.tests_iter = None

class MultiprocessResult(NamedTuple):
    result: TestResult
    stdout: str
    stderr: str
    error_msg: str
ExcStr = str
QueueOutput = tuple[Literal[False], MultiprocessResult] | tuple[Literal[True], ExcStr]

class ExitThread(Exception):
    pass

class TestWorkerProcess(threading.Thread):

    def __init__(self, worker_id: int, runner: 'MultiprocessTestRunner') -> None:
        if False:
            return 10
        super().__init__()
        self.worker_id = worker_id
        self.pending = runner.pending
        self.output = runner.output
        self.ns = runner.ns
        self.timeout = runner.worker_timeout
        self.regrtest = runner.regrtest
        self.current_test_name = None
        self.start_time = None
        self._popen = None
        self._killed = False
        self._stopped = False

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        info = [f'TestWorkerProcess #{self.worker_id}']
        if self.is_alive():
            info.append('running')
        else:
            info.append('stopped')
        test = self.current_test_name
        if test:
            info.append(f'test={test}')
        popen = self._popen
        if popen is not None:
            dt = time.monotonic() - self.start_time
            info.extend((f'pid={self._popen.pid}', f'time={format_duration(dt)}'))
        return '<%s>' % ' '.join(info)

    def _kill(self) -> None:
        if False:
            return 10
        popen = self._popen
        if popen is None:
            return
        if self._killed:
            return
        self._killed = True
        if USE_PROCESS_GROUP:
            what = f'{self} process group'
        else:
            what = f'{self}'
        print(f'Kill {what}', file=sys.stderr, flush=True)
        try:
            if USE_PROCESS_GROUP:
                os.killpg(popen.pid, signal.SIGKILL)
            else:
                popen.kill()
        except ProcessLookupError:
            pass
        except OSError as exc:
            print_warning(f'Failed to kill {what}: {exc!r}')

    def stop(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._stopped = True
        self._kill()

    def mp_result_error(self, test_result: TestResult, stdout: str='', stderr: str='', err_msg=None) -> MultiprocessResult:
        if False:
            print('Hello World!')
        test_result.duration_sec = time.monotonic() - self.start_time
        return MultiprocessResult(test_result, stdout, stderr, err_msg)

    def _run_process(self, test_name: str) -> tuple[int, str, str]:
        if False:
            for i in range(10):
                print('nop')
        self.start_time = time.monotonic()
        self.current_test_name = test_name
        try:
            popen = run_test_in_subprocess(test_name, self.ns)
            self._killed = False
            self._popen = popen
        except:
            self.current_test_name = None
            raise
        try:
            if self._stopped:
                self._kill()
                raise ExitThread
            try:
                (stdout, stderr) = popen.communicate(timeout=self.timeout)
                retcode = popen.returncode
                assert retcode is not None
            except subprocess.TimeoutExpired:
                if self._stopped:
                    raise ExitThread
                self._kill()
                retcode = None
                stdout = stderr = ''
            except OSError:
                if self._stopped:
                    raise ExitThread
                raise
            else:
                stdout = stdout.strip()
                stderr = stderr.rstrip()
            return (retcode, stdout, stderr)
        except:
            self._kill()
            raise
        finally:
            self._wait_completed()
            self._popen = None
            self.current_test_name = None

    def _runtest(self, test_name: str) -> MultiprocessResult:
        if False:
            while True:
                i = 10
        (retcode, stdout, stderr) = self._run_process(test_name)
        if retcode is None:
            return self.mp_result_error(Timeout(test_name), stdout, stderr)
        err_msg = None
        if retcode != 0:
            err_msg = 'Exit code %s' % retcode
        else:
            (stdout, _, result) = stdout.rpartition('\n')
            stdout = stdout.rstrip()
            if not result:
                err_msg = 'Failed to parse worker stdout'
            else:
                try:
                    result = json.loads(result, object_hook=decode_test_result)
                except Exception as exc:
                    err_msg = 'Failed to parse worker JSON: %s' % exc
        if err_msg is not None:
            return self.mp_result_error(ChildError(test_name), stdout, stderr, err_msg)
        return MultiprocessResult(result, stdout, stderr, err_msg)

    def run(self) -> None:
        if False:
            print('Hello World!')
        while not self._stopped:
            try:
                try:
                    test_name = next(self.pending)
                except StopIteration:
                    break
                mp_result = self._runtest(test_name)
                self.output.put((False, mp_result))
                if must_stop(mp_result.result, self.ns):
                    break
            except ExitThread:
                break
            except BaseException:
                self.output.put((True, traceback.format_exc()))
                break

    def _wait_completed(self) -> None:
        if False:
            print('Hello World!')
        popen = self._popen
        popen.stdout.close()
        popen.stderr.close()
        try:
            popen.wait(JOIN_TIMEOUT)
        except (subprocess.TimeoutExpired, OSError) as exc:
            print_warning(f'Failed to wait for {self} completion (timeout={format_duration(JOIN_TIMEOUT)}): {exc!r}')

    def wait_stopped(self, start_time: float) -> None:
        if False:
            return 10
        while True:
            self.join(1.0)
            if not self.is_alive():
                break
            dt = time.monotonic() - start_time
            self.regrtest.log(f'Waiting for {self} thread for {format_duration(dt)}')
            if dt > JOIN_TIMEOUT:
                print_warning(f'Failed to join {self} in {format_duration(dt)}')
                break

def get_running(workers: list[TestWorkerProcess]) -> list[TestWorkerProcess]:
    if False:
        for i in range(10):
            print('nop')
    running = []
    for worker in workers:
        current_test_name = worker.current_test_name
        if not current_test_name:
            continue
        dt = time.monotonic() - worker.start_time
        if dt >= PROGRESS_MIN_TIME:
            text = '%s (%s)' % (current_test_name, format_duration(dt))
            running.append(text)
    return running

class MultiprocessTestRunner:

    def __init__(self, regrtest: Regrtest) -> None:
        if False:
            print('Hello World!')
        self.regrtest = regrtest
        self.log = self.regrtest.log
        self.ns = regrtest.ns
        self.output: queue.Queue[QueueOutput] = queue.Queue()
        self.pending = MultiprocessIterator(self.regrtest.tests)
        if self.ns.timeout is not None:
            self.worker_timeout = min(self.ns.timeout * 1.5, self.ns.timeout + 5 * 60)
        else:
            self.worker_timeout = None
        self.workers = None

    def start_workers(self) -> None:
        if False:
            while True:
                i = 10
        self.workers = [TestWorkerProcess(index, self) for index in range(1, self.ns.use_mp + 1)]
        msg = f'Run tests in parallel using {len(self.workers)} child processes'
        if self.ns.timeout:
            msg += ' (timeout: %s, worker timeout: %s)' % (format_duration(self.ns.timeout), format_duration(self.worker_timeout))
        self.log(msg)
        for worker in self.workers:
            worker.start()

    def stop_workers(self) -> None:
        if False:
            i = 10
            return i + 15
        start_time = time.monotonic()
        for worker in self.workers:
            worker.stop()
        for worker in self.workers:
            worker.wait_stopped(start_time)

    def _get_result(self) -> QueueOutput | None:
        if False:
            return 10
        use_faulthandler = self.ns.timeout is not None
        timeout = PROGRESS_UPDATE
        while any((worker.is_alive() for worker in self.workers)):
            if use_faulthandler:
                faulthandler.dump_traceback_later(MAIN_PROCESS_TIMEOUT, exit=True)
            try:
                return self.output.get(timeout=timeout)
            except queue.Empty:
                pass
            running = get_running(self.workers)
            if running and (not self.ns.pgo):
                self.log('running: %s' % ', '.join(running))
        try:
            return self.output.get(timeout=0)
        except queue.Empty:
            return None

    def display_result(self, mp_result: MultiprocessResult) -> None:
        if False:
            i = 10
            return i + 15
        result = mp_result.result
        text = str(result)
        if mp_result.error_msg is not None:
            text += ' (%s)' % mp_result.error_msg
        elif result.duration_sec >= PROGRESS_MIN_TIME and (not self.ns.pgo):
            text += ' (%s)' % format_duration(result.duration_sec)
        running = get_running(self.workers)
        if running and (not self.ns.pgo):
            text += ' -- running: %s' % ', '.join(running)
        self.regrtest.display_progress(self.test_index, text)

    def _process_result(self, item: QueueOutput) -> bool:
        if False:
            i = 10
            return i + 15
        'Returns True if test runner must stop.'
        if item[0]:
            format_exc = item[1]
            print_warning(f'regrtest worker thread failed: {format_exc}')
            return True
        self.test_index += 1
        mp_result = item[1]
        self.regrtest.accumulate_result(mp_result.result)
        self.display_result(mp_result)
        if mp_result.stdout:
            print(mp_result.stdout, flush=True)
        if mp_result.stderr and (not self.ns.pgo):
            print(mp_result.stderr, file=sys.stderr, flush=True)
        if must_stop(mp_result.result, self.ns):
            return True
        return False

    def run_tests(self) -> None:
        if False:
            while True:
                i = 10
        self.start_workers()
        self.test_index = 0
        try:
            while True:
                item = self._get_result()
                if item is None:
                    break
                stop = self._process_result(item)
                if stop:
                    break
        except KeyboardInterrupt:
            print()
            self.regrtest.interrupted = True
        finally:
            if self.ns.timeout is not None:
                faulthandler.cancel_dump_traceback_later()
            self.pending.stop()
            self.stop_workers()

def run_tests_multiprocess(regrtest: Regrtest) -> None:
    if False:
        for i in range(10):
            print('nop')
    MultiprocessTestRunner(regrtest).run_tests()

class EncodeTestResult(json.JSONEncoder):
    """Encode a TestResult (sub)class object into a JSON dict."""

    def default(self, o: Any) -> dict[str, Any]:
        if False:
            i = 10
            return i + 15
        if isinstance(o, TestResult):
            result = vars(o)
            result['__test_result__'] = o.__class__.__name__
            return result
        return super().default(o)

def decode_test_result(d: dict[str, Any]) -> TestResult | dict[str, Any]:
    if False:
        print('Hello World!')
    'Decode a TestResult (sub)class object from a JSON dict.'
    if '__test_result__' not in d:
        return d
    cls_name = d.pop('__test_result__')
    for cls in get_all_test_result_classes():
        if cls.__name__ == cls_name:
            return cls(**d)

def get_all_test_result_classes() -> set[type[TestResult]]:
    if False:
        while True:
            i = 10
    prev_count = 0
    classes = {TestResult}
    while len(classes) > prev_count:
        prev_count = len(classes)
        to_add = []
        for cls in classes:
            to_add.extend(cls.__subclasses__())
        classes.update(to_add)
    return classes