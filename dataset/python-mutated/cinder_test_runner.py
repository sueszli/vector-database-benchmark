import argparse
import functools
import gc
import json
import multiprocessing
import os
import os.path
import pathlib
import pickle
import queue
import resource
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import types
import unittest
from dataclasses import dataclass
from test import support
from test.support import os_helper
from test.libregrtest.cmdline import Namespace
from test.libregrtest.main import Regrtest
from test.libregrtest.runtest import NOTTESTS, STDTESTS, findtests, runtest, ChildError, Interrupted, Failed, Passed, ResourceDenied, Skipped, TestResult
from test.libregrtest.runtest_mp import get_cinderjit_xargs
from test.libregrtest.setup import setup_tests
from typing import Dict, Iterable, IO, List, Optional, Set, Tuple
MAX_WORKERS = 64
WORKER_PATH = os.path.abspath(__file__)
WORKER_RESPAWN_INTERVAL = 10
CINDER_RUNNER_LOG_DIR = '/tmp/cinder_test_runner_logs'
TESTS_TO_SERIALIZE = {'test_ftplib', 'test_epoll', 'test_urllib2_localnet', 'test_docxmlrpc', 'test_filecmp', 'test_jitlist'}
RR_RECORD_BASE_CMD = ['fdb', '--caller-to-log', 'cinder-jit-test-runner', 'replay', 'record']

@dataclass
class ActiveTest:
    worker_pid: int
    start_time: float
    worker_test_log: str
    rr_trace_dir: Optional[str]

class ReplayInfo:

    def __init__(self, pid: int, test_log: str, rr_trace_dir: Optional[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.pid = pid
        self.test_log = test_log
        self.rr_trace_dir = rr_trace_dir
        self.failed: List[str] = []
        self.crashed: Optional[str] = None

    def _build_replay_cmd(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        args = [sys.executable, *get_cinderjit_xargs(), sys.argv[0], 'replay', self.test_log]
        if sys.argv[1:]:
            args.append('--')
            args.extend(sys.argv[1:])
        return shlex.join(args)

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        if not (self.failed or self.crashed):
            return f'No failures in worker {self.pid}'
        msg = f'In worker {self.pid},'
        if self.failed:
            msg += ' ' + ', '.join(self.failed) + ' failed'
        if self.crashed:
            if self.failed:
                msg += ' and'
            msg += f' {self.crashed} crashed'
        cmd = self._build_replay_cmd()
        msg += f".\n Replay using '{cmd}'"
        if self.rr_trace_dir is not None:
            msg += f'\n Replay recording with: fdb replay debug {self.rr_trace_dir}'
        return msg

    def should_share(self):
        if False:
            while True:
                i = 10
        return self.rr_trace_dir is not None and self.has_broken_tests()

    def has_broken_tests(self):
        if False:
            print('Hello World!')
        return self.failed or self.crashed

    def broken_tests(self):
        if False:
            return 10
        tests = self.failed.copy()
        if self.crashed:
            tests.append(self.crashed)
        return tests

class Message:
    pass

class RunTest(Message):

    def __init__(self, test_name: str) -> None:
        if False:
            while True:
                i = 10
        self.test_name = test_name

class TestStarted(Message):

    def __init__(self, worker_pid: int, test_name: str, test_log: str, rr_trace_dir: Optional[str]) -> None:
        if False:
            return 10
        self.worker_pid = worker_pid
        self.test_name = test_name
        self.test_log = test_log
        self.rr_trace_dir = rr_trace_dir

class TestComplete(Message):

    def __init__(self, test_name: str, result) -> None:
        if False:
            while True:
                i = 10
        self.test_name = test_name
        self.result = result

class ShutdownWorker(Message):
    pass

class WorkerDone(Message):
    pass

class MessagePipe:

    def __init__(self, read_fd: int, write_fd: int) -> None:
        if False:
            return 10
        self.infile = os.fdopen(read_fd, 'rb')
        self.outfile = os.fdopen(write_fd, 'wb')

    def __enter__(self):
        if False:
            return 10
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            print('Hello World!')
        self.close()
        return False

    def close(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.infile.close()
        self.outfile.close()

    def recv(self) -> None:
        if False:
            while True:
                i = 10
        return pickle.load(self.infile)

    def send(self, message) -> None:
        if False:
            while True:
                i = 10
        pickle.dump(message, self.outfile)
        self.outfile.flush()

class TestLog:

    def __init__(self, pid: int=-1, path: str=None) -> None:
        if False:
            i = 10
            return i + 15
        self.pid = pid
        self.test_order: List[str] = []
        if path is None:
            self.path = tempfile.NamedTemporaryFile(delete=False).name
        else:
            self.path = path
            self._deserialize()

    def add_test(self, test_name: str) -> None:
        if False:
            print('Hello World!')
        self.test_order.append(test_name)
        self._serialize()

    def _serialize(self) -> None:
        if False:
            i = 10
            return i + 15
        data = {'pid': self.pid, 'test_order': self.test_order}
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tf:
            json.dump(data, tf)
            shutil.move(tf.name, self.path)

    def _deserialize(self) -> None:
        if False:
            return 10
        with open(self.path) as f:
            data = json.load(f)
            self.pid = data['pid']
            self.test_order = data['test_order']

class WorkSender:

    def __init__(self, pipe: MessagePipe, popen: subprocess.Popen, rr_trace_dir: Optional[str]) -> None:
        if False:
            print('Hello World!')
        self.pipe = pipe
        self.popen = popen
        self.ncompleted = 0
        self.rr_trace_dir = rr_trace_dir
        self.test_log = TestLog(popen.pid)

    @property
    def pid(self) -> int:
        if False:
            print('Hello World!')
        return self.popen.pid

    def send(self, msg: Message) -> None:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(msg, RunTest):
            self.test_log.add_test(msg.test_name)
        self.pipe.send(msg)

    def recv(self) -> Message:
        if False:
            print('Hello World!')
        msg = self.pipe.recv()
        if isinstance(msg, TestComplete):
            self.ncompleted += 1
        return msg

    def shutdown(self) -> int:
        if False:
            i = 10
            return i + 15
        self.pipe.send(ShutdownWorker())
        return self.wait()

    def wait(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        assert self.popen is not None
        r = self.popen.wait()
        if r != 0 and self.rr_trace_dir is not None:
            print(f'Worker with PID {self.pid} ended with exit code {r}.\nReplay recording with: fdb replay debug {self.rr_trace_dir}')
        self.pipe.close()

class WorkReceiver:

    def __init__(self, pipe: MessagePipe) -> None:
        if False:
            while True:
                i = 10
        self.pipe = pipe

    def run(self, ns: types.SimpleNamespace) -> None:
        if False:
            i = 10
            return i + 15
        'Read commands from pipe and execute them'
        t = Regrtest()
        t.ns = ns
        t.set_temp_dir()
        test_cwd = t.create_temp_dir()
        setup_tests(t.ns)
        with os_helper.temp_cwd(test_cwd, quiet=True):
            msg = self.pipe.recv()
            while not isinstance(msg, ShutdownWorker):
                if isinstance(msg, RunTest):
                    result = runtest(ns, msg.test_name)
                    self.pipe.send(TestComplete(msg.test_name, result))
                msg = self.pipe.recv()
            self.pipe.send(WorkerDone())

def start_worker(ns: types.SimpleNamespace, worker_timeout: int, use_rr: bool) -> WorkSender:
    if False:
        while True:
            i = 10
    'Start a worker process we can use to run tests'
    (d_r, w_w) = os.pipe()
    os.set_inheritable(d_r, True)
    os.set_inheritable(w_w, True)
    (w_r, d_w) = os.pipe()
    os.set_inheritable(w_r, True)
    os.set_inheritable(d_w, True)
    pipe = MessagePipe(d_r, d_w)
    ns_dict = vars(ns)
    worker_ns = json.dumps(ns_dict)
    cmd = [sys.executable, *get_cinderjit_xargs(), WORKER_PATH, 'worker', str(w_r), str(w_w), worker_ns]
    env = dict(os.environ)
    env['PYTHONREGRTEST_UNICODE_GUARD'] = '0'
    rr_trace_dir = None
    if use_rr:
        rr_trace_dir = tempfile.mkdtemp(prefix='rr-', dir=CINDER_RUNNER_LOG_DIR)
        rr_trace_dir += '/d'
        cmd = RR_RECORD_BASE_CMD + [f'--recording-dir={rr_trace_dir}'] + cmd
    if worker_timeout != 0:
        cmd = ['timeout', '--foreground', f'{worker_timeout}s'] + cmd
    popen = subprocess.Popen(cmd, pass_fds=(w_r, w_w), cwd=os_helper.SAVEDCWD, env=env)
    os.close(w_r)
    os.close(w_w)
    return WorkSender(pipe, popen, rr_trace_dir)

def manage_worker(ns: types.SimpleNamespace, testq: queue.Queue, resultq: queue.Queue, worker_timeout: int, worker_respawn_interval: int, use_rr: bool) -> None:
    if False:
        i = 10
        return i + 15
    'Spawn and manage a subprocess to execute tests.\n\n    This handles spawning worker processes that crash and periodically restarting workers\n    in order to avoid consuming too much memory.\n    '
    worker = start_worker(ns, worker_timeout, use_rr)
    result = None
    while not isinstance(result, WorkerDone):
        msg = testq.get()
        if isinstance(msg, RunTest):
            resultq.put(TestStarted(worker.pid, msg.test_name, worker.test_log.path, worker.rr_trace_dir))
        try:
            worker.send(msg)
            result = worker.recv()
        except (BrokenPipeError, EOFError):
            if isinstance(msg, ShutdownWorker):
                resultq.put(WorkerDone())
                break
            elif isinstance(msg, RunTest):
                test_result = ChildError(msg.test_name, 0.0)
                result = TestComplete(msg.test_name, test_result)
                resultq.put(result)
                worker.wait()
                worker = start_worker(ns, worker_timeout, use_rr)
        else:
            resultq.put(result)
            if worker.ncompleted == worker_respawn_interval:
                worker.shutdown()
                worker = start_worker(ns, worker_timeout, use_rr)
    worker.wait()

def print_running_tests(tests: Dict[str, ActiveTest]) -> None:
    if False:
        while True:
            i = 10
    if len(tests) == 0:
        print('No tests running')
        return
    now = time.time()
    msg_parts = ['Running tests:']
    for k in sorted(tests.keys()):
        elapsed = int(now - tests[k].start_time)
        msg_parts.append(f'{k} ({elapsed}s, pid {tests[k].worker_pid})')
    print(' '.join(msg_parts))

def log_err(msg: str) -> None:
    if False:
        while True:
            i = 10
    sys.stderr.write(msg)
    sys.stderr.flush()

def _setupCinderIgnoredTests(ns: Namespace, use_rr: bool) -> Tuple[List[str], Set[str]]:
    if False:
        print('Hello World!')
    skip_list_files = ['devserver_skip_tests.txt', 'cinder_skip_test.txt']
    if support.check_sanitizer(address=True):
        skip_list_files.append('asan_skip_tests.txt')
    if use_rr:
        skip_list_files.append('rr_skip_tests.txt')
    if sysconfig.get_config_var('ENABLE_CINDERX') != 1:
        skip_list_files.append('no_cinderx_skip_tests.txt')
    try:
        import cinderjit
        skip_list_files.append('cinder_jit_ignore_tests.txt')
    except ImportError:
        pass
    if ns.huntrleaks:
        skip_list_files.append('refleak_skip_tests.txt')
    if ns.ignore_tests is None:
        ns.ignore_tests = []
    stdtest_set = set(STDTESTS)
    nottests = NOTTESTS.copy()
    for skip_file in skip_list_files:
        with open(os.path.join(os.path.dirname(__file__), skip_file)) as fp:
            for line in fp:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if len({'.', '*'} & set(line)):
                    ns.ignore_tests.append(line)
                else:
                    stdtest_set.discard(line)
                    nottests.add(line)
    return (list(stdtest_set), nottests)

class MultiWorkerCinderRegrtest(Regrtest):

    def __init__(self, logfile: IO, log_to_scuba: bool, worker_timeout: int, worker_respawn_interval: int, success_on_test_errors: bool, use_rr: bool, recording_metadata_path: str, no_retry_on_test_errors: bool):
        if False:
            return 10
        Regrtest.__init__(self)
        self._cinder_regr_runner_logfile = logfile
        self._log_to_scuba = log_to_scuba
        self._success_on_test_errors = success_on_test_errors
        self._worker_timeout = worker_timeout
        self._worker_respawn_interval = worker_respawn_interval
        self._use_rr = use_rr
        self._recording_metadata_path = recording_metadata_path
        self._no_retry_on_test_errors = no_retry_on_test_errors

    def run_tests(self) -> List[ReplayInfo]:
        if False:
            return 10
        self.test_count = '/{}'.format(len(self.selected))
        self._ntests_done = 0
        self.test_count_width = len(self.test_count) - 1
        replay_infos: List[ReplayInfo] = []
        self._run_tests_with_n_workers([t for t in self.selected if t not in TESTS_TO_SERIALIZE], self.num_workers, replay_infos)
        if not self.interrupted:
            print('Running serial tests')
            self._run_tests_with_n_workers([t for t in self.selected if t in TESTS_TO_SERIALIZE], 1, replay_infos)
        return replay_infos

    def _run_tests_with_n_workers(self, tests: Iterable[str], n_workers: int, replay_infos: List[ReplayInfo]) -> None:
        if False:
            return 10
        resultq = queue.Queue()
        testq = queue.Queue()
        ntests_remaining = 0
        for test in tests:
            ntests_remaining += 1
            testq.put(RunTest(test))
        threads = []
        for i in range(n_workers):
            testq.put(ShutdownWorker())
            t = threading.Thread(target=manage_worker, args=(self.ns, testq, resultq, self._worker_timeout, self._worker_respawn_interval, self._use_rr))
            t.start()
            threads.append(t)
        try:
            active_tests: Dict[str, ActiveTest] = {}
            worker_infos: Dict[int, ReplayInfo] = {}
            while ntests_remaining:
                try:
                    msg = resultq.get(timeout=10)
                except queue.Empty:
                    print_running_tests(active_tests)
                    continue
                if isinstance(msg, TestStarted):
                    active_tests[msg.test_name] = ActiveTest(msg.worker_pid, time.time(), msg.test_log, msg.rr_trace_dir)
                    print(f"Running test '{msg.test_name}' on worker {msg.worker_pid}", file=self._cinder_regr_runner_logfile)
                    self._cinder_regr_runner_logfile.flush()
                elif isinstance(msg, TestComplete):
                    ntests_remaining -= 1
                    self._ntests_done += 1
                    result = msg.result
                    if not isinstance(result, (Passed, ResourceDenied, Skipped)):
                        worker_pid = active_tests[msg.test_name].worker_pid
                        rr_trace_dir = active_tests[msg.test_name].rr_trace_dir
                        err = f'TEST ERROR: {msg.result} in pid {worker_pid}'
                        if rr_trace_dir:
                            err += f' Replay recording with: fdb replay debug {rr_trace_dir}'
                        log_err(f'{err}\n')
                        if worker_pid not in worker_infos:
                            log = active_tests[msg.test_name].worker_test_log
                            worker_infos[worker_pid] = ReplayInfo(worker_pid, log, rr_trace_dir)
                        replay_info = worker_infos[worker_pid]
                        if isinstance(result, ChildError):
                            replay_info.crashed = msg.test_name
                        else:
                            replay_info.failed.append(msg.test_name)
                    self.accumulate_result(msg.result)
                    self.display_progress(self._ntests_done, msg.test_name)
                    del active_tests[msg.test_name]
        except KeyboardInterrupt:
            replay_infos.extend(worker_infos.values())
            self._show_replay_infos(replay_infos)
            self._save_recording_metadata(replay_infos)
            os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)
        replay_infos.extend(worker_infos.values())
        for t in threads:
            t.join()

    def _show_replay_infos(self, replay_infos: List[ReplayInfo]) -> None:
        if False:
            while True:
                i = 10
        if not replay_infos:
            return
        info = ['', 'You can replay failed tests using the commands below:']
        seen = 0
        for ri in replay_infos:
            info.append('    ' + str(ri))
            seen += 1
            if seen == 10:
                info.append('NOTE: Only showing 10 instances')
                break
        info.append('\n')
        log_err('\n'.join(info))

    def _save_recording_metadata(self, replay_infos: List[ReplayInfo]) -> None:
        if False:
            i = 10
            return i + 15
        if not self._recording_metadata_path:
            return
        recordings = [{'tests': info.broken_tests(), 'recording_path': info.rr_trace_dir} for info in replay_infos if info.should_share()]
        metadata = {'recordings': recordings}
        os.makedirs(os.path.dirname(self._recording_metadata_path), exist_ok=True)
        with open(self._recording_metadata_path, 'w') as f:
            json.dump(metadata, f)

    def _main(self, tests, kwargs):
        if False:
            while True:
                i = 10
        self.ns.fail_env_changed = True
        setup_tests(self.ns)
        test_filters = _setupCinderIgnoredTests(self.ns, self._use_rr)
        cinderx_dir = os.path.dirname(os.path.dirname(__file__))
        self.ns.testdir = cinderx_dir
        sys.path.append(cinderx_dir)
        if tests is None:
            self._selectDefaultCinderTests(test_filters, cinderx_dir)
        else:
            self.find_tests(tests)
        replay_infos = self.run_tests()
        self.display_result()
        self._show_replay_infos(replay_infos)
        self._save_recording_metadata(replay_infos)
        if self._log_to_scuba:
            print('Logging data to Scuba...')
            self._writeResultsToScuba()
            print('done.')
        if not self._no_retry_on_test_errors and self.ns.verbose2 and self.bad:
            self.rerun_failed_tests()
        self.finalize()
        if not self._success_on_test_errors:
            if self.bad:
                sys.exit(2)
            if self.interrupted:
                sys.exit(130)
            if self.ns.fail_env_changed and self.environment_changed:
                sys.exit(3)
        sys.exit(0)

    def _selectDefaultCinderTests(self, test_filters: Tuple[List[str], Set[str]], cinderx_dir: str) -> None:
        if False:
            print('Hello World!')
        (stdtest, nottests) = test_filters
        tests = ['test.' + t for t in findtests(None, stdtest, nottests)]
        cinderx_tests = findtests(os.path.join(cinderx_dir, 'test_cinderx'), list(), nottests)
        tests.extend(('test_cinderx.' + t for t in cinderx_tests))
        self.selected = tests

    def _writeResultsToScuba(self) -> None:
        if False:
            while True:
                i = 10
        template = {'int': {'time': int(time.time())}, 'normal': {'test_name': None, 'status': None, 'host': socket.gethostname()}, 'normvector': {'env': [k if v is True else f'{k}={v}' for (k, v) in list(sys._xoptions.items())]}}
        data_to_log = []
        template['normal']['status'] = 'good'
        for good_name in self.good:
            template['normal']['test_name'] = good_name
            data_to_log.append(json.dumps(template))
        template['normal']['status'] = 'bad'
        for bad_name in self.bad:
            template['normal']['test_name'] = bad_name
            data_to_log.append(json.dumps(template) + '\n')
        if not len(data_to_log):
            return
        with subprocess.Popen(['scribe_cat', 'perfpipe_cinder_test_status'], stdin=subprocess.PIPE) as sc_proc:
            sc_proc.stdin.write(('\n'.join(data_to_log) + '\n').encode('utf-8'))
            sc_proc.stdin.close()
            sc_proc.wait()

def _patched_runtest_inner2(ns: Namespace, tests_name: str) -> bool:
    if False:
        while True:
            i = 10
    import test.libregrtest.runtest as runtest
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromName(tests_name, None)
    for error in loader.errors:
        print(error, file=sys.stderr)
    if loader.errors:
        raise Exception('errors while loading tests')
    if ns.huntrleaks:
        from test.libregrtest.refleak import dash_R
    test_runner = functools.partial(support.run_unittest, tests)
    try:
        with runtest.save_env(ns, tests_name):
            if ns.huntrleaks:
                refleak = dash_R(ns, tests_name, test_runner)
            else:
                test_runner()
                refleak = False
    finally:
        runtest.cleanup_test_droppings(tests_name, ns.verbose)
    support.gc_collect()
    if gc.garbage:
        support.environment_altered = True
        runtest.print_warning(f'{test_name} created {len(gc.garbage)} uncollectable object(s).')
        runtest.FOUND_GARBAGE.extend(gc.garbage)
        gc.garbage.clear()
    support.reap_children()
    return refleak

class UserSelectedCinderRegrtest(Regrtest):

    def __init__(self):
        if False:
            return 10
        Regrtest.__init__(self)

    def _main(self, tests, kwargs):
        if False:
            while True:
                i = 10
        import test.libregrtest.runtest as runtest
        runtest._runtest_inner2 = _patched_runtest_inner2
        cinderx_dir = os.path.dirname(os.path.dirname(__file__))
        sys.path.append(cinderx_dir)
        self.ns.fail_env_changed = True
        setup_tests(self.ns)
        _setupCinderIgnoredTests(self.ns, False)
        if not self.ns.verbose and (not self.ns.huntrleaks):
            from unittest import TextTestResult
            old_init = TextTestResult.__init__

            def force_dots_output(self, *args, **kwargs):
                if False:
                    print('Hello World!')
                old_init(self, *args, **kwargs)
                self.dots = True
            TextTestResult.__init__ = force_dots_output
        for t in tests:
            self.accumulate_result(runtest.runtest(self.ns, t))
        self.display_result()
        self.finalize()
        if self.bad:
            sys.exit(2)
        if self.interrupted:
            sys.exit(130)
        if self.ns.fail_env_changed and self.environment_changed:
            sys.exit(3)
        sys.exit(0)

def worker_main(args):
    if False:
        for i in range(10):
            print('nop')
    ns_dict = json.loads(args.ns)
    ns = types.SimpleNamespace(**ns_dict)
    with MessagePipe(args.cmd_fd, args.result_fd) as pipe:
        WorkReceiver(pipe).run(ns)

def user_selected_main(args):
    if False:
        i = 10
        return i + 15
    test_runner = UserSelectedCinderRegrtest()
    sys.argv[1:] = args.rest[1:]
    test_runner.main(args.test)

def dispatcher_main(args):
    if False:
        print('Hello World!')
    pathlib.Path(CINDER_RUNNER_LOG_DIR).mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(delete=False, mode='w+t', dir=CINDER_RUNNER_LOG_DIR) as logfile:
            print(f'Using scheduling log file {logfile.name}')
            test_runner = MultiWorkerCinderRegrtest(logfile, args.log_to_scuba, args.worker_timeout, args.worker_respawn_interval, args.success_on_test_errors, args.use_rr, args.recording_metadata_path, args.no_retry_on_test_errors)
            test_runner.num_workers = args.num_workers
            print(f'Spawning {test_runner.num_workers} workers')
            sys.argv[1:] = args.rest[1:]
            test_runner.main(args.test)
    finally:
        if args.use_rr:
            print(f'Consider cleaning out RR data with: rm -rf {CINDER_RUNNER_LOG_DIR}/rr-*')

def replay_main(args):
    if False:
        for i in range(10):
            print('nop')
    print(f'Replaying tests from {args.test_log}')
    test_log = TestLog(path=args.test_log)
    sys.argv[1:] = args.rest
    Regrtest().main(test_log.test_order)
if __name__ == '__main__':
    try:
        sys.executable = os.path.realpath(sys.executable)
    except OSError:
        pass
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    worker_parser = subparsers.add_parser('worker')
    worker_parser.add_argument('cmd_fd', type=int, help='Readable fd to receive commands to execute')
    worker_parser.add_argument('result_fd', type=int, help='Writeable fd to write test results')
    worker_parser.add_argument('ns', help='Serialized namespace')
    worker_parser.set_defaults(func=worker_main)
    dispatcher_parser = subparsers.add_parser('dispatcher')
    dispatcher_parser.add_argument('--num-workers', type=int, default=min(multiprocessing.cpu_count(), MAX_WORKERS), help='Number of parallel test runners to use')
    dispatcher_parser.add_argument('--log-to-scuba', action='store_true', help='Log results to Scuba table (intended for Sandcastle jobs)')
    dispatcher_parser.add_argument('--worker-timeout', type=int, help='Timeout for worker jobs (in seconds)', default=20 * 60)
    dispatcher_parser.add_argument('--worker-respawn-interval', type=int, help='Number of jobs to run in a worker before respawning.', default=WORKER_RESPAWN_INTERVAL)
    dispatcher_parser.add_argument('--use-rr', action='store_true', help='Run worker processes in RR')
    dispatcher_parser.add_argument('--recording-metadata-path', type=str, help='Path of recording metadata output file')
    dispatcher_parser.add_argument('--success-on-test-errors', action='store_true', help='Return with exit code 0 even if tests fail')
    dispatcher_parser.add_argument('--no-retry-on-test-errors', action='store_true', help='Do not retry tests which fail with verbose output')
    dispatcher_parser.add_argument('-t', '--test', action='append', help='The name of a test to run (e.g. `test_math`). Can be supplied multiple times.')
    dispatcher_parser.add_argument('rest', nargs=argparse.REMAINDER)
    dispatcher_parser.set_defaults(func=dispatcher_main)
    user_selected_parser = subparsers.add_parser('test')
    user_selected_parser.add_argument('-t', '--test', action='append', required=True, help='The name of a test to run (e.g. `test_math`). Can be supplied multiple times.')
    user_selected_parser.add_argument('rest', nargs=argparse.REMAINDER)
    user_selected_parser.set_defaults(func=user_selected_main)
    replay_parser = subparsers.add_parser('replay')
    replay_parser.add_argument('test_log', type=str, help='Replay the test run from the specified file in one worker')
    replay_parser.add_argument('rest', nargs=argparse.REMAINDER)
    replay_parser.set_defaults(func=replay_main)
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.error('too few arguments')