from __future__ import annotations
import select
import sys
from typing import TYPE_CHECKING
from .. import _core, _subprocess
assert sys.platform != 'win32' and sys.platform != 'linux' or not TYPE_CHECKING

async def wait_child_exiting(process: _subprocess.Process) -> None:
    kqueue = _core.current_kqueue()
    try:
        from select import KQ_NOTE_EXIT
    except ImportError:
        KQ_NOTE_EXIT = 2147483648

    def make_event(flags: int) -> select.kevent:
        if False:
            return 10
        return select.kevent(process.pid, filter=select.KQ_FILTER_PROC, flags=flags, fflags=KQ_NOTE_EXIT)
    try:
        kqueue.control([make_event(select.KQ_EV_ADD | select.KQ_EV_ONESHOT)], 0)
    except ProcessLookupError:
        return

    def abort(_: _core.RaiseCancelT) -> _core.Abort:
        if False:
            for i in range(10):
                print('nop')
        kqueue.control([make_event(select.KQ_EV_DELETE)], 0)
        return _core.Abort.SUCCEEDED
    await _core.wait_kevent(process.pid, select.KQ_FILTER_PROC, abort)