import errno
import os
import sys
import gevent
import gevent.monkey
gevent.monkey.patch_all()
pid = None
awaiting_child = []

def handle_sigchld(*_args):
    if False:
        return 10
    gevent.sleep()
    awaiting_child.pop()
    raise TypeError('This should be ignored but printed')
import signal
if hasattr(signal, 'SIGCHLD'):
    if sys.version_info[:2] >= (3, 8) and os.environ.get('PYTHONDEVMODE'):
        print('Ran 1 tests in 0.0s (skipped=1)')
        sys.exit(0)
    assert signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL
    signal.signal(signal.SIGCHLD, handle_sigchld)
    handler = signal.getsignal(signal.SIGCHLD)
    assert signal.getsignal(signal.SIGCHLD) is handle_sigchld, handler
    if hasattr(os, 'forkpty'):

        def forkpty():
            if False:
                return 10
            return os.forkpty()[0]
        funcs = (os.fork, forkpty)
    else:
        funcs = (os.fork,)
    for func in funcs:
        awaiting_child = [True]
        pid = func()
        if not pid:
            gevent.sleep(0.3)
            sys.exit(0)
        else:
            timeout = gevent.Timeout(1)
            try:
                while awaiting_child:
                    gevent.sleep(0.01)
                (wpid, status) = os.waitpid(-1, os.WNOHANG)
                if wpid != pid:
                    raise AssertionError('Failed to wait on a child pid forked with a function', wpid, pid, func)
                try:
                    (wpid, status) = os.waitpid(-1, os.WNOHANG)
                    raise AssertionError('Should not be able to wait again')
                except OSError as e:
                    assert e.errno == errno.ECHILD
            except gevent.Timeout as t:
                if timeout is not t:
                    raise
                raise AssertionError('Failed to wait using', func)
            finally:
                timeout.close()
    print('Ran 1 tests in 0.0s')
    sys.exit(0)
else:
    print('No SIGCHLD, not testing')
    print('Ran 1 tests in 0.0s (skipped=1)')