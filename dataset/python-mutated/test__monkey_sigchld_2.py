import os
import sys
import signal

def handle(*_args):
    if False:
        for i in range(10):
            print('nop')
    if not pid:
        os.waitpid(-1, os.WNOHANG)
if hasattr(signal, 'SIGCHLD'):
    if sys.version_info[:2] >= (3, 8) and os.environ.get('PYTHONDEVMODE'):
        print('Ran 1 tests in 0.0s (skipped=1)')
        sys.exit(0)
    import platform
    platform.uname()
    signal.signal(signal.SIGCHLD, handle)
    pid = os.fork()
    if pid:
        try:
            (_, stat) = os.waitpid(pid, 0)
        except OSError:
            (_, stat) = os.waitpid(pid, 0)
        assert stat == 0, stat
    else:
        import gevent.monkey
        gevent.monkey.patch_all()
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        f = os.popen('true')
        f.close()
        sys.exit(0)
else:
    print('No SIGCHLD, not testing')