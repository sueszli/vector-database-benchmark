"""Routines common to all posix systems."""
import glob
import os
import signal
import sys
import time
from ._common import MACOS
from ._common import TimeoutExpired
from ._common import memoize
from ._common import sdiskusage
from ._common import usage_percent
from ._compat import PY3
from ._compat import ChildProcessError
from ._compat import FileNotFoundError
from ._compat import InterruptedError
from ._compat import PermissionError
from ._compat import ProcessLookupError
from ._compat import unicode
if MACOS:
    from . import _psutil_osx
if PY3:
    import enum
else:
    enum = None
__all__ = ['pid_exists', 'wait_pid', 'disk_usage', 'get_terminal_map']

def pid_exists(pid):
    if False:
        for i in range(10):
            print('nop')
    'Check whether pid exists in the current process table.'
    if pid == 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True
if enum is not None and hasattr(signal, 'Signals'):
    Negsignal = enum.IntEnum('Negsignal', dict([(x.name, -x.value) for x in signal.Signals]))

    def negsig_to_enum(num):
        if False:
            print('Hello World!')
        'Convert a negative signal value to an enum.'
        try:
            return Negsignal(num)
        except ValueError:
            return num
else:

    def negsig_to_enum(num):
        if False:
            while True:
                i = 10
        return num

def wait_pid(pid, timeout=None, proc_name=None, _waitpid=os.waitpid, _timer=getattr(time, 'monotonic', time.time), _min=min, _sleep=time.sleep, _pid_exists=pid_exists):
    if False:
        print('Hello World!')
    'Wait for a process PID to terminate.\n\n    If the process terminated normally by calling exit(3) or _exit(2),\n    or by returning from main(), the return value is the positive integer\n    passed to *exit().\n\n    If it was terminated by a signal it returns the negated value of the\n    signal which caused the termination (e.g. -SIGTERM).\n\n    If PID is not a children of os.getpid() (current process) just\n    wait until the process disappears and return None.\n\n    If PID does not exist at all return None immediately.\n\n    If *timeout* != None and process is still alive raise TimeoutExpired.\n    timeout=0 is also possible (either return immediately or raise).\n    '
    if pid <= 0:
        msg = "can't wait for PID 0"
        raise ValueError(msg)
    interval = 0.0001
    flags = 0
    if timeout is not None:
        flags |= os.WNOHANG
        stop_at = _timer() + timeout

    def sleep(interval):
        if False:
            print('Hello World!')
        if timeout is not None:
            if _timer() >= stop_at:
                raise TimeoutExpired(timeout, pid=pid, name=proc_name)
        _sleep(interval)
        return _min(interval * 2, 0.04)
    while True:
        try:
            (retpid, status) = os.waitpid(pid, flags)
        except InterruptedError:
            interval = sleep(interval)
        except ChildProcessError:
            while _pid_exists(pid):
                interval = sleep(interval)
            return
        else:
            if retpid == 0:
                interval = sleep(interval)
                continue
            elif os.WIFEXITED(status):
                return os.WEXITSTATUS(status)
            elif os.WIFSIGNALED(status):
                return negsig_to_enum(-os.WTERMSIG(status))
            else:
                raise ValueError('unknown process exit status %r' % status)

def disk_usage(path):
    if False:
        for i in range(10):
            print('nop')
    'Return disk usage associated with path.\n    Note: UNIX usually reserves 5% disk space which is not accessible\n    by user. In this function "total" and "used" values reflect the\n    total and used disk space whereas "free" and "percent" represent\n    the "free" and "used percent" user disk space.\n    '
    if PY3:
        st = os.statvfs(path)
    else:
        try:
            st = os.statvfs(path)
        except UnicodeEncodeError:
            if isinstance(path, unicode):
                try:
                    path = path.encode(sys.getfilesystemencoding())
                except UnicodeEncodeError:
                    pass
                st = os.statvfs(path)
            else:
                raise
    total = st.f_blocks * st.f_frsize
    avail_to_root = st.f_bfree * st.f_frsize
    avail_to_user = st.f_bavail * st.f_frsize
    used = total - avail_to_root
    if MACOS:
        used = _psutil_osx.disk_usage_used(path, used)
    total_user = used + avail_to_user
    usage_percent_user = usage_percent(used, total_user, round_=1)
    return sdiskusage(total=total, used=used, free=avail_to_user, percent=usage_percent_user)

@memoize
def get_terminal_map():
    if False:
        while True:
            i = 10
    'Get a map of device-id -> path as a dict.\n    Used by Process.terminal().\n    '
    ret = {}
    ls = glob.glob('/dev/tty*') + glob.glob('/dev/pts/*')
    for name in ls:
        assert name not in ret, name
        try:
            ret[os.stat(name).st_rdev] = name
        except FileNotFoundError:
            pass
    return ret