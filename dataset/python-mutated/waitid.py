import errno
import math
import os
import sys
from typing import TYPE_CHECKING
from .. import _core, _subprocess
from .._sync import CapacityLimiter, Event
from .._threads import to_thread_run_sync
assert sys.platform != 'win32' and sys.platform != 'darwin' or not TYPE_CHECKING
try:
    from os import waitid

    def sync_wait_reapable(pid: int) -> None:
        if False:
            i = 10
            return i + 15
        waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
except ImportError:
    import cffi
    waitid_ffi = cffi.FFI()
    waitid_ffi.cdef('\ntypedef struct siginfo_s {\n    int si_signo;\n    int si_errno;\n    int si_code;\n    int si_pid;\n    int si_uid;\n    int si_status;\n    int pad[26];\n} siginfo_t;\nint waitid(int idtype, int id, siginfo_t* result, int options);\n')
    waitid_cffi = waitid_ffi.dlopen(None).waitid

    def sync_wait_reapable(pid: int) -> None:
        if False:
            return 10
        P_PID = 1
        WEXITED = 4
        if sys.platform == 'darwin':
            WNOWAIT = 32
        else:
            WNOWAIT = 16777216
        result = waitid_ffi.new('siginfo_t *')
        while waitid_cffi(P_PID, pid, result, WEXITED | WNOWAIT) < 0:
            got_errno = waitid_ffi.errno
            if got_errno == errno.EINTR:
                continue
            raise OSError(got_errno, os.strerror(got_errno))
waitid_limiter = CapacityLimiter(math.inf)

async def _waitid_system_task(pid: int, event: Event) -> None:
    """Spawn a thread that waits for ``pid`` to exit, then wake any tasks
    that were waiting on it.
    """
    try:
        await to_thread_run_sync(sync_wait_reapable, pid, abandon_on_cancel=True, limiter=waitid_limiter)
    except OSError:
        pass
    finally:
        event.set()

async def wait_child_exiting(process: '_subprocess.Process') -> None:
    if process._wait_for_exit_data is None:
        process._wait_for_exit_data = event = Event()
        _core.spawn_system_task(_waitid_system_task, process.pid, event)
    assert isinstance(process._wait_for_exit_data, Event)
    await process._wait_for_exit_data.wait()