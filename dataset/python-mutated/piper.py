"""
Utilities for handling subprocesses.

Mostly necessary only because of http://bugs.python.org/issue1652.

"""
import copy
import errno
import gevent
import gevent.socket
import subprocess
from subprocess import PIPE
from wal_e import pipebuf
assert PIPE

class PopenShim(object):

    def __init__(self, sleep_time=1, max_tries=None):
        if False:
            print('Hello World!')
        self.sleep_time = sleep_time
        self.max_tries = max_tries

    def __call__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        "Call Popen, but be persistent in the face of ENOMEM.\n\n        The utility of this is that on systems with overcommit off,\n        the momentary spike in committed virtual memory from fork()\n        can be large, but is cleared soon thereafter because\n        'subprocess' uses an 'exec' system call.  Without retrying,\n        the the backup process would lose all its progress\n        immediately with no recourse, which is undesirable.\n\n        Because the ENOMEM error happens on fork() before any\n        meaningful work can be done, one thinks this retry would be\n        safe, and without side effects.  Because fork is being\n        called through 'subprocess' and not directly here, this\n        program has to rely on the semantics of the exceptions\n        raised from 'subprocess' to avoid retries in unrelated\n        scenarios, which could be dangerous.\n\n        "
        tries = 0
        while True:
            try:
                proc = subprocess.Popen(*args, **kwargs)
            except OSError as e:
                if e.errno == errno.ENOMEM:
                    should_retry = self.max_tries is not None and tries >= self.max_tries
                    if should_retry:
                        raise
                    gevent.sleep(self.sleep_time)
                    tries += 1
                    continue
                raise
            else:
                break
        return proc
popen_sp = PopenShim()

def popen_nonblock(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Create a process in the same way as popen_sp, but patch the file\n    descriptors so they can be accessed from Python/gevent\n    in a non-blocking manner.\n    '
    proc = popen_sp(*args, **kwargs)
    if proc.stdin:
        proc.stdin = pipebuf.NonBlockBufferedWriter(proc.stdin)
    if proc.stdout:
        proc.stdout = pipebuf.NonBlockBufferedReader(proc.stdout)
    if proc.stderr:
        proc.stderr = pipebuf.NonBlockBufferedReader(proc.stderr)
    return proc

def pipe(*args):
    if False:
        print('Hello World!')
    '\n    Takes as parameters several dicts, each with the same\n    parameters passed to popen.\n\n    Runs the various processes in a pipeline, connecting\n    the stdout of every process except the last with the\n    stdin of the next process.\n\n    Adapted from http://www.enricozini.org/2009/debian/python-pipes/\n\n    '
    if len(args) < 2:
        raise ValueError('pipe needs at least 2 processes')
    for i in args[:-1]:
        i['stdout'] = subprocess.PIPE
    popens = [popen_sp(**args[0])]
    for i in range(1, len(args)):
        args[i]['stdin'] = popens[i - 1].stdout
        popens.append(popen_sp(**args[i]))
        popens[i - 1].stdout.close()
    return popens

def pipe_wait(popens):
    if False:
        print('Hello World!')
    '\n    Given an array of Popen objects returned by the\n    pipe method, wait for all processes to terminate\n    and return the array with their return values.\n\n    Taken from http://www.enricozini.org/2009/debian/python-pipes/\n\n    '
    popens = copy.copy(popens)
    results = [0] * len(popens)
    while popens:
        last = popens.pop(-1)
        results[len(popens)] = last.wait()
    return results