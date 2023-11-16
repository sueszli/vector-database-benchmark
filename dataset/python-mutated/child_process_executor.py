"""Facilities for running arbitrary commands in child processes."""
import os
import queue
import sys
from abc import ABC, abstractmethod
from multiprocessing import Queue
from multiprocessing.context import BaseContext as MultiprocessingBaseContext
from typing import TYPE_CHECKING, Iterator, NamedTuple, Optional, Union
from typing_extensions import Literal
import dagster._check as check
from dagster._core.errors import DagsterExecutionInterruptedError
from dagster._utils.error import SerializableErrorInfo, serializable_error_info_from_exc_info
from dagster._utils.interrupts import capture_interrupts
if TYPE_CHECKING:
    from dagster._core.events import DagsterEvent

class ChildProcessEvent:
    pass

class ChildProcessStartEvent(NamedTuple('ChildProcessStartEvent', [('pid', int)]), ChildProcessEvent):
    pass

class ChildProcessDoneEvent(NamedTuple('ChildProcessDoneEvent', [('pid', int)]), ChildProcessEvent):
    pass

class ChildProcessSystemErrorEvent(NamedTuple('ChildProcessSystemErrorEvent', [('pid', int), ('error_info', SerializableErrorInfo)]), ChildProcessEvent):
    pass

class ChildProcessCommand(ABC):
    """Inherit from this class in order to use this library.

    The object must be picklable; instantiate it and pass it to _execute_command_in_child_process.
    """

    @abstractmethod
    def execute(self) -> Iterator[Union[ChildProcessEvent, 'DagsterEvent']]:
        if False:
            while True:
                i = 10
        'This method is invoked in the child process.\n\n        Yields a sequence of events to be handled by _execute_command_in_child_process.\n        '

class ChildProcessCrashException(Exception):
    """Thrown when the child process crashes."""

    def __init__(self, exit_code=None):
        if False:
            i = 10
            return i + 15
        self.exit_code = exit_code
        super().__init__()

def _execute_command_in_child_process(event_queue: Queue, command: ChildProcessCommand):
    if False:
        return 10
    'Wraps the execution of a ChildProcessCommand.\n\n    Handles errors and communicates across a queue with the parent process.\n    '
    check.inst_param(command, 'command', ChildProcessCommand)
    with capture_interrupts():
        pid = os.getpid()
        event_queue.put(ChildProcessStartEvent(pid=pid))
        try:
            for step_event in command.execute():
                event_queue.put(step_event)
            event_queue.put(ChildProcessDoneEvent(pid=pid))
        except (Exception, KeyboardInterrupt, DagsterExecutionInterruptedError):
            event_queue.put(ChildProcessSystemErrorEvent(pid=pid, error_info=serializable_error_info_from_exc_info(sys.exc_info())))
TICK = 20.0 * 1.0 / 1000.0
'The minimum interval at which to check for child process liveness -- default 20ms.'
PROCESS_DEAD_AND_QUEUE_EMPTY = 'PROCESS_DEAD_AND_QUEUE_EMPTY'
'Sentinel value.'

def _poll_for_event(process, event_queue) -> Optional[Union['DagsterEvent', Literal['PROCESS_DEAD_AND_QUEUE_EMPTY']]]:
    if False:
        for i in range(10):
            print('nop')
    try:
        return event_queue.get(block=True, timeout=TICK)
    except queue.Empty:
        if not process.is_alive():
            try:
                return event_queue.get(block=False)
            except queue.Empty:
                return PROCESS_DEAD_AND_QUEUE_EMPTY
    return None

def execute_child_process_command(multiprocessing_ctx: MultiprocessingBaseContext, command: ChildProcessCommand) -> Iterator[Optional['DagsterEvent']]:
    if False:
        for i in range(10):
            print('nop')
    'Execute a ChildProcessCommand in a new process.\n\n    This function starts a new process whose execution target is a ChildProcessCommand wrapped by\n    _execute_command_in_child_process; polls the queue for events yielded by the child process\n    until the process dies and the queue is empty.\n\n    This function yields a complex set of objects to enable having multiple child process\n    executions in flight:\n        * None - nothing has happened, yielded to enable cooperative multitasking other iterators\n\n        * ChildProcessEvent - Family of objects that communicates state changes in the child process\n\n        * KeyboardInterrupt - Yielded in the case that an interrupt was recieved while\n            polling the child process. Yielded instead of raised to allow forwarding of the\n            interrupt to the child and completion of the iterator for this child and\n            any others that may be executing\n\n        * The actual values yielded by the child process command\n\n    Args:\n        multiprocessing_ctx: The multiprocessing context to execute in (spawn, forkserver, fork)\n        command (ChildProcessCommand): The command to execute in the child process.\n\n    Warning: if the child process is in an infinite loop, this will\n    also infinitely loop.\n    '
    check.inst_param(command, 'command', ChildProcessCommand)
    event_queue = multiprocessing_ctx.Queue()
    try:
        process = multiprocessing_ctx.Process(target=_execute_command_in_child_process, args=(event_queue, command))
        process.start()
        completed_properly = False
        while not completed_properly:
            event = _poll_for_event(process, event_queue)
            if event == PROCESS_DEAD_AND_QUEUE_EMPTY:
                break
            yield event
            if isinstance(event, (ChildProcessDoneEvent, ChildProcessSystemErrorEvent)):
                completed_properly = True
        if not completed_properly:
            raise ChildProcessCrashException(exit_code=process.exitcode)
        process.join()
    finally:
        event_queue.close()