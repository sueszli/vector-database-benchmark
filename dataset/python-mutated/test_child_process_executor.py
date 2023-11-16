import multiprocessing
import os
import time
import pytest
from dagster._core.executor.child_process_executor import ChildProcessCommand, ChildProcessCrashException, ChildProcessDoneEvent, ChildProcessEvent, ChildProcessStartEvent, ChildProcessSystemErrorEvent, execute_child_process_command
from dagster._utils import segfault

class DoubleAStringChildProcessCommand(ChildProcessCommand):

    def __init__(self, a_str):
        if False:
            print('Hello World!')
        self.a_str = a_str

    def execute(self):
        if False:
            return 10
        yield (self.a_str + self.a_str)

class AnError(Exception):
    pass

class ThrowAnErrorCommand(ChildProcessCommand):

    def execute(self):
        if False:
            while True:
                i = 10
        raise AnError('Oh noes!')

class CrashyCommand(ChildProcessCommand):

    def execute(self):
        if False:
            i = 10
            return i + 15
        os._exit(1)

class SegfaultCommand(ChildProcessCommand):

    def execute(self):
        if False:
            i = 10
            return i + 15
        segfault()

class LongRunningCommand(ChildProcessCommand):

    def execute(self):
        if False:
            print('Hello World!')
        time.sleep(0.5)
        yield 1

def test_basic_child_process_command():
    if False:
        while True:
            i = 10
    events = list(filter(lambda x: x and (not isinstance(x, ChildProcessEvent)), execute_child_process_command(multiprocessing, DoubleAStringChildProcessCommand('aa'))))
    assert events == ['aaaa']

def test_basic_child_process_command_with_process_events():
    if False:
        while True:
            i = 10
    events = list(filter(lambda x: x, execute_child_process_command(multiprocessing, DoubleAStringChildProcessCommand('aa'))))
    assert len(events) == 3
    assert isinstance(events[0], ChildProcessStartEvent)
    child_pid = events[0].pid
    assert child_pid != os.getpid()
    assert events[1] == 'aaaa'
    assert isinstance(events[2], ChildProcessDoneEvent)
    assert events[2].pid == child_pid

def test_child_process_uncaught_exception():
    if False:
        print('Hello World!')
    results = list(filter(lambda x: x and isinstance(x, ChildProcessSystemErrorEvent), execute_child_process_command(multiprocessing, ThrowAnErrorCommand())))
    assert len(results) == 1
    assert 'AnError' in str(results[0].error_info.message)

def test_child_process_crashy_process():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ChildProcessCrashException) as exc:
        list(execute_child_process_command(multiprocessing, CrashyCommand()))
    assert exc.value.exit_code == 1

@pytest.mark.skipif(os.name == 'nt', reason='Segfault not being caught on Windows: See issue #2791')
def test_child_process_segfault():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ChildProcessCrashException) as exc:
        list(execute_child_process_command(multiprocessing, SegfaultCommand()))
    assert exc.value.exit_code == -11

@pytest.mark.skip('too long')
def test_long_running_command():
    if False:
        print('Hello World!')
    list(execute_child_process_command(multiprocessing, LongRunningCommand()))