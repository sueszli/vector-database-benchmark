import _thread
import contextlib
import functools
import sys
import threading
import time
from test import support

def threading_setup():
    if False:
        i = 10
        return i + 15
    return (_thread._count(), threading._dangling.copy())

def threading_cleanup(*original_values):
    if False:
        i = 10
        return i + 15
    _MAX_COUNT = 100
    for count in range(_MAX_COUNT):
        values = (_thread._count(), threading._dangling)
        if values == original_values:
            break
        if not count:
            support.environment_altered = True
            dangling_threads = values[1]
            support.print_warning(f'threading_cleanup() failed to cleanup {values[0] - original_values[0]} threads (count: {values[0]}, dangling: {len(dangling_threads)})')
            for thread in dangling_threads:
                support.print_warning(f'Dangling thread: {thread!r}')
            dangling_threads = None
        values = None
        time.sleep(0.01)
        support.gc_collect()

def reap_threads(func):
    if False:
        print('Hello World!')
    'Use this function when threads are being used.  This will\n    ensure that the threads are cleaned up even when the test fails.\n    '

    @functools.wraps(func)
    def decorator(*args):
        if False:
            print('Hello World!')
        key = threading_setup()
        try:
            return func(*args)
        finally:
            threading_cleanup(*key)
    return decorator

@contextlib.contextmanager
def wait_threads_exit(timeout=None):
    if False:
        i = 10
        return i + 15
    "\n    bpo-31234: Context manager to wait until all threads created in the with\n    statement exit.\n\n    Use _thread.count() to check if threads exited. Indirectly, wait until\n    threads exit the internal t_bootstrap() C function of the _thread module.\n\n    threading_setup() and threading_cleanup() are designed to emit a warning\n    if a test leaves running threads in the background. This context manager\n    is designed to cleanup threads started by the _thread.start_new_thread()\n    which doesn't allow to wait for thread exit, whereas thread.Thread has a\n    join() method.\n    "
    if timeout is None:
        timeout = support.SHORT_TIMEOUT
    old_count = _thread._count()
    try:
        yield
    finally:
        start_time = time.monotonic()
        deadline = start_time + timeout
        while True:
            count = _thread._count()
            if count <= old_count:
                break
            if time.monotonic() > deadline:
                dt = time.monotonic() - start_time
                msg = f'wait_threads() failed to cleanup {count - old_count} threads after {dt:.1f} seconds (count: {count}, old count: {old_count})'
                raise AssertionError(msg)
            time.sleep(0.01)
            support.gc_collect()

def join_thread(thread, timeout=None):
    if False:
        while True:
            i = 10
    'Join a thread. Raise an AssertionError if the thread is still alive\n    after timeout seconds.\n    '
    if timeout is None:
        timeout = support.SHORT_TIMEOUT
    thread.join(timeout)
    if thread.is_alive():
        msg = f'failed to join the thread in {timeout:.1f} seconds'
        raise AssertionError(msg)

@contextlib.contextmanager
def start_threads(threads, unlock=None):
    if False:
        i = 10
        return i + 15
    import faulthandler
    threads = list(threads)
    started = []
    try:
        try:
            for t in threads:
                t.start()
                started.append(t)
        except:
            if support.verbose:
                print("Can't start %d threads, only %d threads started" % (len(threads), len(started)))
            raise
        yield
    finally:
        try:
            if unlock:
                unlock()
            endtime = time.monotonic()
            for timeout in range(1, 16):
                endtime += 60
                for t in started:
                    t.join(max(endtime - time.monotonic(), 0.01))
                started = [t for t in started if t.is_alive()]
                if not started:
                    break
                if support.verbose:
                    print('Unable to join %d threads during a period of %d minutes' % (len(started), timeout))
        finally:
            started = [t for t in started if t.is_alive()]
            if started:
                faulthandler.dump_traceback(sys.stdout)
                raise AssertionError('Unable to join %d threads' % len(started))

class catch_threading_exception:
    """
    Context manager catching threading.Thread exception using
    threading.excepthook.

    Attributes set when an exception is caught:

    * exc_type
    * exc_value
    * exc_traceback
    * thread

    See threading.excepthook() documentation for these attributes.

    These attributes are deleted at the context manager exit.

    Usage:

        with threading_helper.catch_threading_exception() as cm:
            # code spawning a thread which raises an exception
            ...

            # check the thread exception, use cm attributes:
            # exc_type, exc_value, exc_traceback, thread
            ...

        # exc_type, exc_value, exc_traceback, thread attributes of cm no longer
        # exists at this point
        # (to avoid reference cycles)
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.exc_type = None
        self.exc_value = None
        self.exc_traceback = None
        self.thread = None
        self._old_hook = None

    def _hook(self, args):
        if False:
            i = 10
            return i + 15
        self.exc_type = args.exc_type
        self.exc_value = args.exc_value
        self.exc_traceback = args.exc_traceback
        self.thread = args.thread

    def __enter__(self):
        if False:
            return 10
        self._old_hook = threading.excepthook
        threading.excepthook = self._hook
        return self

    def __exit__(self, *exc_info):
        if False:
            for i in range(10):
                print('nop')
        threading.excepthook = self._old_hook
        del self.exc_type
        del self.exc_value
        del self.exc_traceback
        del self.thread