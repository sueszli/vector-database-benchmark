"""Base class for a subprocess run for tests."""
import re
import time
import warnings
import dataclasses
import pytest
import pytestqt.wait_signal
from qutebrowser.qt.core import pyqtSlot, pyqtSignal, QProcess, QObject, QElapsedTimer, QProcessEnvironment
from qutebrowser.qt.test import QSignalSpy
from helpers import testutils
from qutebrowser.utils import utils as quteutils

class InvalidLine(Exception):
    """Raised when the process prints a line which is not parsable."""

class ProcessExited(Exception):
    """Raised when the child process did exit."""

class WaitForTimeout(Exception):
    """Raised when wait_for didn't get the expected message."""

class BlacklistedMessageError(Exception):
    """Raised when ensure_not_logged found a message."""

@dataclasses.dataclass
class Line:
    """Container for a line of data the process emits.

    Attributes:
        data: The raw data passed to the constructor.
        waited_for: If Process.wait_for was used on this line already.
    """
    data: str
    waited_for: bool = False

def _render_log(data, *, verbose, threshold=100):
    if False:
        i = 10
        return i + 15
    'Shorten the given log without -v and convert to a string.'
    data = [str(d) for d in data]
    is_exception = any(('Traceback (most recent call last):' in line or 'Uncaught exception' in line for line in data))
    if len(data) > threshold and (not verbose) and (not is_exception) and (not testutils.ON_CI):
        msg = '[{} lines suppressed, use -v to show]'.format(len(data) - threshold)
        data = [msg] + data[-threshold:]
    if testutils.ON_CI:
        data = [testutils.gha_group_begin('Log')] + data + [testutils.gha_group_end()]
    return '\n'.join(data)

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    if False:
        while True:
            i = 10
    'Add qutebrowser/server sections to captured output if a test failed.'
    outcome = (yield)
    if call.when not in ['call', 'teardown']:
        return
    report = outcome.get_result()
    if report.passed:
        return
    quteproc_log = getattr(item, '_quteproc_log', None)
    server_logs = getattr(item, '_server_logs', [])
    if not hasattr(report.longrepr, 'addsection'):
        return
    if item.config.getoption('--capture') == 'no':
        return
    verbose = item.config.getoption('--verbose')
    if quteproc_log is not None:
        report.longrepr.addsection('qutebrowser output', _render_log(quteproc_log, verbose=verbose))
    for (name, content) in server_logs:
        report.longrepr.addsection(f'{name} output', _render_log(content, verbose=verbose))

class Process(QObject):
    """Abstraction over a running test subprocess process.

    Reads the log from its stdout and parses it.

    Attributes:
        _invalid: A list of lines which could not be parsed.
        _data: A list of parsed lines.
        _started: Whether the process was ever started.
        proc: The QProcess for the underlying process.
        exit_expected: Whether the process is expected to quit.
        request: The request object for the current test.

    Signals:
        ready: Emitted when the server finished starting up.
        new_data: Emitted when a new line was parsed.
    """
    ready = pyqtSignal()
    new_data = pyqtSignal(object)
    KEYS = ['data']

    def __init__(self, request, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self.request = request
        self.captured_log = []
        self._started = False
        self._invalid = []
        self._data = []
        self.proc = QProcess()
        self.proc.setReadChannel(QProcess.ProcessChannel.StandardError)
        self.exit_expected = None

    def _log(self, line):
        if False:
            return 10
        'Add the given line to the captured log output.'
        if self.request.config.getoption('--capture') == 'no':
            print(line)
        self.captured_log.append(line)

    def log_summary(self, text):
        if False:
            return 10
        'Log the given line as summary/title.'
        text = '\n{line} {text} {line}\n'.format(line='=' * 30, text=text)
        self._log(text)

    def _parse_line(self, line):
        if False:
            for i in range(10):
                print('nop')
        'Parse the given line from the log.\n\n        Return:\n            A self.ParseResult member.\n        '
        raise NotImplementedError

    def _executable_args(self):
        if False:
            while True:
                i = 10
        'Get the executable and necessary arguments as a tuple.'
        raise NotImplementedError

    def _default_args(self):
        if False:
            return 10
        'Get the default arguments to use if none were passed to start().'
        raise NotImplementedError

    def _get_data(self):
        if False:
            return 10
        'Get the parsed data for this test.\n\n        Also waits for 0.5s to make sure any new data is received.\n\n        Subprocesses are expected to alias this to a public method with a\n        better name.\n        '
        self.proc.waitForReadyRead(500)
        self.read_log()
        return self._data

    def _wait_signal(self, signal, timeout=5000, raising=True):
        if False:
            for i in range(10):
                print('nop')
        'Wait for a signal to be emitted.\n\n        Should be used in a contextmanager.\n        '
        blocker = pytestqt.wait_signal.SignalBlocker(timeout=timeout, raising=raising)
        blocker.connect(signal)
        return blocker

    @pyqtSlot()
    def read_log(self):
        if False:
            print('Hello World!')
        "Read the log from the process' stdout."
        if not hasattr(self, 'proc'):
            return
        while self.proc.canReadLine():
            line = self.proc.readLine()
            line = bytes(line).decode('utf-8', errors='ignore').rstrip('\r\n')
            try:
                parsed = self._parse_line(line)
            except InvalidLine:
                self._invalid.append(line)
                self._log('INVALID: {}'.format(line))
                continue
            if parsed is None:
                if self._invalid:
                    self._log('IGNORED: {}'.format(line))
            else:
                self._data.append(parsed)
                self.new_data.emit(parsed)

    def start(self, args=None, *, env=None):
        if False:
            i = 10
            return i + 15
        'Start the process and wait until it started.'
        self._start(args, env=env)
        self._started = True
        verbose = self.request.config.getoption('--verbose')
        timeout = 60 if testutils.ON_CI else 20
        for _ in range(timeout):
            with self._wait_signal(self.ready, timeout=1000, raising=False) as blocker:
                pass
            if not self.is_running():
                if self.exit_expected:
                    return
                raise ProcessExited('\n' + _render_log(self.captured_log, verbose=verbose))
            if blocker.signal_triggered:
                self._after_start()
                return
        raise WaitForTimeout('Timed out while waiting for process start.\n' + _render_log(self.captured_log, verbose=verbose))

    def _start(self, args, env):
        if False:
            while True:
                i = 10
        'Actually start the process.'
        (executable, exec_args) = self._executable_args()
        if args is None:
            args = self._default_args()
        procenv = QProcessEnvironment.systemEnvironment()
        if env is not None:
            for (k, v) in env.items():
                procenv.insert(k, v)
        self.proc.readyRead.connect(self.read_log)
        self.proc.setProcessEnvironment(procenv)
        self.proc.start(executable, exec_args + args)
        ok = self.proc.waitForStarted()
        assert ok
        assert self.is_running()

    def _after_start(self):
        if False:
            for i in range(10):
                print('nop')
        'Do things which should be done immediately after starting.'

    def before_test(self):
        if False:
            for i in range(10):
                print('nop')
        'Restart process before a test if it exited before.'
        self._invalid = []
        if not self.is_running():
            self.start()

    def after_test(self):
        if False:
            i = 10
            return i + 15
        'Clean up data after each test.\n\n        Also checks self._invalid so the test counts as failed if there were\n        unexpected output lines earlier.\n        '
        __tracebackhide__ = lambda e: e.errisinstance(ProcessExited)
        self.captured_log = []
        if self._invalid:
            time.sleep(1)
            self.terminate()
            self.clear_data()
            raise InvalidLine('\n' + '\n'.join(self._invalid))
        self.clear_data()
        if not self.is_running() and (not self.exit_expected) and self._started:
            raise ProcessExited
        self.exit_expected = False

    def clear_data(self):
        if False:
            return 10
        'Clear the collected data.'
        self._data.clear()

    def terminate(self):
        if False:
            print('Hello World!')
        'Clean up and shut down the process.'
        if not self.is_running():
            return
        if quteutils.is_windows:
            self.proc.kill()
        else:
            self.proc.terminate()
        ok = self.proc.waitForFinished(5000)
        if not ok:
            cmdline = ' '.join([self.proc.program()] + self.proc.arguments())
            warnings.warn(f'Test process {cmdline} with PID {self.proc.processId()} failed to terminate!')
            self.proc.kill()
            self.proc.waitForFinished()

    def is_running(self):
        if False:
            while True:
                i = 10
        'Check if the process is currently running.'
        return self.proc.state() == QProcess.ProcessState.Running

    def _match_data(self, value, expected):
        if False:
            for i in range(10):
                print('nop')
        'Helper for wait_for to match a given value.\n\n        The behavior of this method is slightly different depending on the\n        types of the filtered values:\n\n        - If expected is None, the filter always matches.\n        - If the value is a string or bytes object and the expected value is\n          too, the pattern is treated as a glob pattern (with only * active).\n        - If the value is a string or bytes object and the expected value is a\n          compiled regex, it is used for matching.\n        - If the value is any other type, == is used.\n\n        Return:\n            A bool\n        '
        regex_type = type(re.compile(''))
        if expected is None:
            return True
        elif isinstance(expected, regex_type):
            return expected.search(value)
        elif isinstance(value, (bytes, str)):
            return testutils.pattern_match(pattern=expected, value=value)
        else:
            return value == expected

    def _wait_for_existing(self, override_waited_for, after, **kwargs):
        if False:
            print('Hello World!')
        'Check if there are any line in the history for wait_for.\n\n        Return: either the found line or None.\n        '
        for line in self._data:
            matches = []
            for (key, expected) in kwargs.items():
                value = getattr(line, key)
                matches.append(self._match_data(value, expected))
            if after is None:
                too_early = False
            else:
                too_early = (line.timestamp, line.msecs) < (after.timestamp, after.msecs)
            if all(matches) and (not line.waited_for or override_waited_for) and (not too_early):
                line.waited_for = True
                self._log('\n----> Already found {!r} in the log: {}'.format(kwargs.get('message', 'line'), line))
                return line
        return None

    def _wait_for_new(self, timeout, do_skip, **kwargs):
        if False:
            while True:
                i = 10
        "Wait for a log message which doesn't exist yet.\n\n        Called via wait_for.\n        "
        __tracebackhide__ = lambda e: e.errisinstance(WaitForTimeout)
        message = kwargs.get('message', None)
        if message is not None:
            elided = quteutils.elide(repr(message), 100)
            self._log('\n----> Waiting for {} in the log'.format(elided))
        spy = QSignalSpy(self.new_data)
        elapsed_timer = QElapsedTimer()
        elapsed_timer.start()
        while True:
            self._maybe_skip()
            got_signal = spy.wait(timeout)
            if not got_signal or elapsed_timer.hasExpired(timeout):
                msg = 'Timed out after {}ms waiting for {!r}.'.format(timeout, kwargs)
                if do_skip:
                    pytest.skip(msg)
                else:
                    raise WaitForTimeout(msg)
            match = self._wait_for_match(spy, kwargs)
            if match is not None:
                if message is not None:
                    self._log('----> found it')
                return match
        raise quteutils.Unreachable

    def _wait_for_match(self, spy, kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Try matching the kwargs with the given QSignalSpy.'
        for args in spy:
            assert len(args) == 1
            line = args[0]
            matches = []
            for (key, expected) in kwargs.items():
                value = getattr(line, key)
                matches.append(self._match_data(value, expected))
            if all(matches):
                line.waited_for = True
                return line
        return None

    def _maybe_skip(self):
        if False:
            i = 10
            return i + 15
        "Can be overridden by subclasses to skip on certain log lines.\n\n        We can't run pytest.skip directly while parsing the log, as that would\n        lead to a pytest.skip.Exception error in a virtual Qt method, which\n        means pytest-qt fails the test.\n\n        Instead, we check for skip messages periodically in\n        QuteProc._maybe_skip, and call _maybe_skip after every parsed message\n        in wait_for (where it's most likely that new messages arrive).\n        "

    def wait_for(self, timeout=None, *, override_waited_for=False, do_skip=False, divisor=1, after=None, **kwargs):
        if False:
            while True:
                i = 10
        "Wait until a given value is found in the data.\n\n        Keyword arguments to this function get interpreted as attributes of the\n        searched data. Every given argument is treated as a pattern which\n        the attribute has to match against.\n\n        Args:\n            timeout: How long to wait for the message.\n            override_waited_for: If set, gets triggered by previous messages\n                                 again.\n            do_skip: If set, call pytest.skip on a timeout.\n            divisor: A factor to decrease the timeout by.\n            after: If it's an existing line, ensure it's after the given one.\n\n        Return:\n            The matched line.\n        "
        __tracebackhide__ = lambda e: e.errisinstance(WaitForTimeout)
        if timeout is None:
            if do_skip:
                timeout = 2000
            elif testutils.ON_CI:
                timeout = 15000
            else:
                timeout = 5000
        timeout //= divisor
        if not kwargs:
            raise TypeError('No keyword arguments given!')
        for key in kwargs:
            assert key in self.KEYS
        existing = self._wait_for_existing(override_waited_for, after, **kwargs)
        if existing is not None:
            return existing
        else:
            return self._wait_for_new(timeout=timeout, do_skip=do_skip, **kwargs)

    def ensure_not_logged(self, delay=500, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Make sure the data matching the given arguments is not logged.\n\n        If nothing is found in the log, we wait for delay ms to make sure\n        nothing arrives.\n        '
        __tracebackhide__ = lambda e: e.errisinstance(BlacklistedMessageError)
        try:
            line = self.wait_for(timeout=delay, override_waited_for=True, **kwargs)
        except WaitForTimeout:
            return
        else:
            raise BlacklistedMessageError(line)

    def wait_for_quit(self):
        if False:
            print('Hello World!')
        'Wait until the process has quit.'
        self.exit_expected = True
        with self._wait_signal(self.proc.finished, timeout=15000):
            pass
        assert not self.is_running()