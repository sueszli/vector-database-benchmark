"""Test testprocess.Process."""
import sys
import time
import contextlib
import datetime
import pytest
from qutebrowser.qt.core import QProcess
from end2end.fixtures import testprocess
pytestmark = [pytest.mark.not_frozen]

@contextlib.contextmanager
def stopwatch(min_ms=None, max_ms=None):
    if False:
        while True:
            i = 10
    if min_ms is None and max_ms is None:
        raise ValueError('Using stopwatch with both min_ms/max_ms None does nothing.')
    start = datetime.datetime.now()
    yield
    stop = datetime.datetime.now()
    delta_ms = (stop - start).total_seconds() * 1000
    if min_ms is not None:
        assert delta_ms >= min_ms
    if max_ms is not None:
        assert delta_ms <= max_ms

class PythonProcess(testprocess.Process):
    """A testprocess which runs the given Python code."""

    def __init__(self, request):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(request)
        self.proc.setReadChannel(QProcess.ProcessChannel.StandardOutput)
        self.code = None

    def _parse_line(self, line):
        if False:
            for i in range(10):
                print('nop')
        print('LINE: {}'.format(line))
        if line.strip() == 'ready':
            self.ready.emit()
        return testprocess.Line(line)

    def _executable_args(self):
        if False:
            return 10
        code = ['import sys, time', 'print("ready")', 'sys.stdout.flush()', self.code, 'sys.stdout.flush()', 'time.sleep(20)']
        return (sys.executable, ['-c', ';'.join(code)])

    def _default_args(self):
        if False:
            print('Hello World!')
        return []

class QuitPythonProcess(PythonProcess):
    """A testprocess which quits immediately."""

    def _executable_args(self):
        if False:
            i = 10
            return i + 15
        code = ['import sys', 'print("ready")', 'sys.exit(0)']
        return (sys.executable, ['-c', ';'.join(code)])

class NoReadyPythonProcess(PythonProcess):
    """A testprocess which never emits 'ready' and quits."""

    def _executable_args(self):
        if False:
            for i in range(10):
                print('nop')
        code = ['import sys', 'sys.exit(0)']
        return (sys.executable, ['-c', ';'.join(code)])

@pytest.fixture
def pyproc(request):
    if False:
        return 10
    proc = PythonProcess(request)
    yield proc
    proc.terminate()

@pytest.fixture
def quit_pyproc(request):
    if False:
        i = 10
        return i + 15
    proc = QuitPythonProcess(request)
    yield proc
    proc.terminate()

@pytest.fixture
def noready_pyproc(request):
    if False:
        return 10
    proc = NoReadyPythonProcess(request)
    yield proc
    proc.terminate()

def test_no_ready_python_process(noready_pyproc):
    if False:
        i = 10
        return i + 15
    'When a process quits immediately, waiting for start should interrupt.'
    with pytest.raises(testprocess.ProcessExited):
        with stopwatch(max_ms=5000):
            noready_pyproc.start()

def test_quitting_process(qtbot, quit_pyproc):
    if False:
        print('Hello World!')
    with qtbot.wait_signal(quit_pyproc.proc.finished):
        quit_pyproc.start()
    with pytest.raises(testprocess.ProcessExited):
        quit_pyproc.after_test()

def test_quitting_process_expected(qtbot, quit_pyproc):
    if False:
        return 10
    quit_pyproc.exit_expected = True
    with qtbot.wait_signal(quit_pyproc.proc.finished):
        quit_pyproc.start()
    quit_pyproc.after_test()

def test_process_never_started(qtbot, quit_pyproc):
    if False:
        i = 10
        return i + 15
    'Calling after_test without start should not fail.'
    quit_pyproc.after_test()

def test_wait_signal_raising(request, qtbot):
    if False:
        return 10
    'testprocess._wait_signal should raise by default.'
    proc = testprocess.Process(request)
    with pytest.raises(qtbot.TimeoutError):
        with proc._wait_signal(proc.proc.started, timeout=0):
            pass

def test_custom_environment(pyproc):
    if False:
        print('Hello World!')
    pyproc.code = 'import os; print(os.environ["CUSTOM_ENV"])'
    pyproc.start(env={'CUSTOM_ENV': 'blah'})
    pyproc.wait_for(data='blah')

@pytest.mark.posix
def test_custom_environment_system_env(monkeypatch, pyproc):
    if False:
        for i in range(10):
            print('nop')
    'When env=... is given, the system environment should be present.'
    monkeypatch.setenv('QUTE_TEST_ENV', 'blubb')
    pyproc.code = 'import os; print(os.environ["QUTE_TEST_ENV"])'
    pyproc.start(env={})
    pyproc.wait_for(data='blubb')

class TestWaitFor:

    def test_successful(self, pyproc):
        if False:
            while True:
                i = 10
        'Using wait_for with the expected text.'
        pyproc.code = "time.sleep(0.5); print('foobar')"
        with stopwatch(min_ms=500):
            pyproc.start()
            pyproc.wait_for(data='foobar')

    def test_other_text(self, pyproc):
        if False:
            i = 10
            return i + 15
        'Test wait_for when getting some unrelated text.'
        pyproc.code = "time.sleep(0.1); print('blahblah')"
        pyproc.start()
        with pytest.raises(testprocess.WaitForTimeout):
            pyproc.wait_for(data='foobar', timeout=500)

    def test_no_text(self, pyproc):
        if False:
            print('Hello World!')
        'Test wait_for when getting no text at all.'
        pyproc.code = 'pass'
        pyproc.start()
        with pytest.raises(testprocess.WaitForTimeout):
            pyproc.wait_for(data='foobar', timeout=100)

    @pytest.mark.parametrize('message', ['foobar', 'literal [x]'])
    def test_existing_message(self, message, pyproc):
        if False:
            while True:
                i = 10
        'Test with a message which already passed when waiting.'
        pyproc.code = "print('{}')".format(message)
        pyproc.start()
        time.sleep(0.5)
        pyproc.wait_for(data=message)

    def test_existing_message_previous_test(self, pyproc):
        if False:
            return 10
        'Make sure the message of a previous test gets ignored.'
        pyproc.code = "print('foobar')"
        pyproc.start()
        line = pyproc.wait_for(data='foobar')
        line.waited_for = False
        pyproc.after_test()
        with pytest.raises(testprocess.WaitForTimeout):
            pyproc.wait_for(data='foobar', timeout=100)

    def test_existing_message_already_waited(self, pyproc):
        if False:
            return 10
        "Make sure an existing message doesn't stop waiting twice.\n\n        wait_for checks existing messages (see above), but we don't want it to\n        automatically proceed if we already *did* use wait_for on one of the\n        existing messages, as that makes it likely it's not what we actually\n        want.\n        "
        pyproc.code = "time.sleep(0.1); print('foobar')"
        pyproc.start()
        pyproc.wait_for(data='foobar')
        with pytest.raises(testprocess.WaitForTimeout):
            pyproc.wait_for(data='foobar', timeout=100)

    def test_no_kwargs(self, pyproc):
        if False:
            while True:
                i = 10
        "Using wait_for without kwargs should raise an exception.\n\n        Otherwise it'd match automatically because of the all(matches).\n        "
        with pytest.raises(TypeError):
            pyproc.wait_for()

    def test_do_skip(self, pyproc):
        if False:
            return 10
        'Test wait_for when getting no text at all, with do_skip.'
        pyproc.code = 'pass'
        pyproc.start()
        with pytest.raises(pytest.skip.Exception):
            pyproc.wait_for(data='foobar', timeout=100, do_skip=True)

class TestEnsureNotLogged:

    @pytest.mark.parametrize('message, pattern', [('blacklisted', 'blacklisted'), ('bl[a]cklisted', 'bl[a]cklisted'), ('blacklisted', 'black*')])
    def test_existing_message(self, pyproc, message, pattern):
        if False:
            return 10
        pyproc.code = "print('{}')".format(message)
        pyproc.start()
        with stopwatch(max_ms=1000):
            with pytest.raises(testprocess.BlacklistedMessageError):
                pyproc.ensure_not_logged(data=pattern, delay=2000)

    def test_late_message(self, pyproc):
        if False:
            i = 10
            return i + 15
        pyproc.code = "time.sleep(0.5); print('blacklisted')"
        pyproc.start()
        with pytest.raises(testprocess.BlacklistedMessageError):
            pyproc.ensure_not_logged(data='blacklisted', delay=5000)

    def test_no_matching_message(self, pyproc):
        if False:
            i = 10
            return i + 15
        pyproc.code = "print('blacklisted... nope!')"
        pyproc.start()
        pyproc.ensure_not_logged(data='blacklisted', delay=100)

    def test_wait_for_and_blacklist(self, pyproc):
        if False:
            for i in range(10):
                print('nop')
        pyproc.code = "print('blacklisted')"
        pyproc.start()
        pyproc.wait_for(data='blacklisted')
        with pytest.raises(testprocess.BlacklistedMessageError):
            pyproc.ensure_not_logged(data='blacklisted', delay=0)