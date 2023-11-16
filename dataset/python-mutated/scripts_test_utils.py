"""Common utilities for test classes."""
from __future__ import annotations
import io
import signal
import psutil
from typing import List, Optional, Tuple

class PopenStub:
    """Stubs the API of psutil.Popen() to make unit tests less expensive.

    Starting a new process for every unit test is intrinsically more expensive
    than checking an object's attributes, and for some developers it isn't even
    possible for them to kill a spawned process due to a lack of permission on
    their operating system.

    We used to spawn real processes for tests, and observed the following:
        With actual processes: Runs 78 tests in 50.7 seconds
        With PopenStub:        Runs 97 tests in 32.3 seconds

    Thus, using this stub gives us a ~4.62x speed boost per-test.

    Attributes:
        pid: int. The ID of the process.
        stdout: bytes. The text written to standard output by the process.
        stderr: bytes. The text written to error output by the process.
        poll_count: int. The number of times poll() has been called.
        signals_received: list(int). List of received signals (as ints) in order
            of receipt.
        terminate_count: int. Number of times terminate() has been called.
        kill_count: int. Number of times kill() has been called.
        alive: bool. Whether the process should be considered to be alive.
        reject_signal: bool. Whether to raise OSError in send_signal().
        reject_terminate: bool. Whether to raise OSError in terminate().
        reject_kill: bool. Whether to raise OSError in kill().
        unresponsive: bool. Whether the process will end normally.
        returncode: int. The return code of the process.
    """

    def __init__(self, pid: int=1, name: str='process', stdout: bytes=b'', stderr: bytes=b'', reject_signal: bool=False, reject_terminate: bool=False, reject_kill: bool=False, alive: bool=True, unresponsive: bool=False, return_code: int=0, child_procs: Optional[List[PopenStub]]=None) -> None:
        if False:
            print('Hello World!')
        'Initializes a new PopenStub instance.\n\n        Args:\n            pid: int. The ID of the process.\n            name: str. The name of the process.\n            stdout: bytes. The text written to standard output by the process.\n            stderr: bytes. The text written to error output by the process.\n            return_code: int. The return code of the process.\n            reject_signal: bool. Whether to raise OSError in send_signal().\n            reject_terminate: bool. Whether to raise OSError in terminate().\n            reject_kill: bool. Whether to raise OSError in kill().\n            alive: bool. Whether the process should be considered to be alive.\n            unresponsive: bool. Whether the process will end normally.\n            child_procs: list(PopenStub)|None. Processes "owned" by the stub, or\n                None if there aren\'t any.\n        '
        self.pid = pid
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.poll_count = 0
        self.signals_received: List[int] = []
        self.terminate_count = 0
        self.kill_count = 0
        self.alive = alive
        self.reject_signal = reject_signal
        self.reject_terminate = reject_terminate
        self.reject_kill = reject_kill
        self.unresponsive = unresponsive
        self._name = name
        self._child_procs = tuple(child_procs) if child_procs else ()
        self._return_code = return_code

    @property
    def returncode(self) -> int:
        if False:
            i = 10
            return i + 15
        'Returns the return code of the process.\n\n        Returns:\n            int. The return code of the process.\n        '
        return self._return_code

    @returncode.setter
    def returncode(self, return_code: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Assigns a return code to the process.\n\n        Args:\n            return_code: int. The return code to assign to the process.\n        '
        self._return_code = return_code

    def is_running(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Returns whether the process is running.\n\n        Returns:\n            bool. The value of self.alive, which mocks whether the process is\n            still alive.\n        '
        return self.alive

    def name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Returns the name of the process.\n\n        Returns:\n            str. The name of the process.\n        '
        return self._name

    def children(self, recursive: bool=False) -> List[PopenStub]:
        if False:
            for i in range(10):
                print('nop')
        'Returns the children spawned by this process.\n\n        Args:\n            recursive: bool. Whether to also return non-direct decendants from\n                self (i.e. children of children).\n\n        Returns:\n            list(PopenStub). A list of the child processes.\n        '
        children = []
        for child in self._child_procs:
            children.append(child)
            if recursive:
                children.extend(child.children(recursive=True))
        return children

    def terminate(self) -> None:
        if False:
            while True:
                i = 10
        'Increment terminate_count.\n\n        Mocks the process being terminated.\n        '
        self.terminate_count += 1
        if self.reject_terminate:
            raise OSError('rejected')
        if self.unresponsive:
            return
        self._exit(return_code=1)

    def kill(self) -> None:
        if False:
            while True:
                i = 10
        'Increment kill_count.\n\n        NOTE: kill() does not respect self.unresponsive.\n\n        Mocks the process being killed.\n        '
        self.kill_count += 1
        if self.reject_kill:
            raise OSError('rejected')
        self._exit(return_code=1)

    def send_signal(self, signal_number: int) -> None:
        if False:
            while True:
                i = 10
        'Append signal to self.signals_received.\n\n        Mocks receiving a process signal. If a SIGINT signal is received (e.g.\n        from ctrl-C) and self.unresponsive is True, then we call self._exit().\n\n        Args:\n            signal_number: int. The number of the received signal.\n\n        Raises:\n            OSError. The SIGINT signal rejected.\n        '
        self.signals_received.append(signal_number)
        if self.reject_signal:
            raise OSError('rejected')
        if signal_number == signal.SIGINT and (not self.unresponsive):
            self._exit(return_code=1)

    def poll(self) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        'Increment poll_count.\n\n        Mocks checking whether the process is still alive.\n\n        Returns:\n            int|None. The return code of the process if it has ended, otherwise\n            None.\n        '
        self.poll_count += 1
        return None if self.alive else self._return_code

    def wait(self, timeout: Optional[int]=None) -> None:
        if False:
            while True:
                i = 10
        'Wait for the process completion.\n\n        Mocks the process waiting for completion before it continues execution.\n        No time is actually spent waiting, however, since the lifetime of the\n        program is completely defined by the initialization params.\n\n        Args:\n            timeout: int|None. Time to wait before raising an exception, or None\n                to wait indefinitely.\n\n        Raises:\n            RuntimeError. The PopenStub has entered an infinite loop.\n        '
        if not self.alive:
            return
        if not self.unresponsive:
            self._exit()
        elif timeout is not None:
            raise psutil.TimeoutExpired(timeout)
        else:
            raise RuntimeError('PopenStub has entered an infinite loop')

    def communicate(self, input: bytes=b'') -> Tuple[bytes, bytes]:
        if False:
            while True:
                i = 10
        "Mocks an interaction with the process.\n\n        Args:\n            input: bytes. Input string to write to the process's stdin.\n\n        Returns:\n            tuple(bytes, bytes). The stdout and stderr of the process,\n            respectively.\n\n        Raises:\n            RuntimeError. The PopenStub has entered an infinite loop.\n        "
        if not self.alive:
            return (self.stdout.getvalue(), self.stderr.getvalue())
        if not self.unresponsive:
            self.stdin.write(input)
            self._exit()
            return (self.stdout.getvalue(), self.stderr.getvalue())
        else:
            raise RuntimeError('PopenStub has entered an infinite loop')

    def _exit(self, return_code: Optional[int]=None) -> None:
        if False:
            while True:
                i = 10
        'Simulates the end of the process.\n\n        Args:\n            return_code: int|None. The return code of the program. If None, the\n                return code assigned at initialization is used instead.\n        '
        self.alive = False
        if return_code is not None:
            self._return_code = return_code