from __future__ import print_function
import gevent.monkey
gevent.monkey.patch_all()
from gevent import get_hub
import os
import sys
import signal
import subprocess

def _waitpid(p):
    if False:
        return 10
    try:
        (_, stat) = os.waitpid(p, 0)
    except OSError:
        (_, stat) = os.waitpid(p, 0)
    assert stat == 0, stat
if hasattr(signal, 'SIGCHLD'):
    if sys.version_info[:2] >= (3, 8) and os.environ.get('PYTHONDEVMODE'):
        print('Ran 1 tests in 0.0s (skipped=1)')
        sys.exit(0)
    get_hub().loop.install_sigchld()
    pid = os.fork()
    if pid:
        _waitpid(pid)
    else:
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        popen = subprocess.Popen([sys.executable, '-c', 'import sys'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        popen.stderr.read()
        popen.stdout.read()
        popen.wait()
        popen.stderr.close()
        popen.stdout.close()
        sys.exit(0)
else:
    print('No SIGCHLD, not testing')
    print('Ran 1 tests in 0.0s (skipped=1)')