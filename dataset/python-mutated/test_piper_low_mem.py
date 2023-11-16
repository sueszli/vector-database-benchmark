import errno
import os
import subprocess
import pytest
from fast_wait import fast_wait
from wal_e import piper
assert fast_wait

def invoke_program():
    if False:
        for i in range(10):
            print('nop')
    with open(os.devnull, 'w') as devnull:
        proc = piper.popen_sp(['python', '--version'], stdout=devnull, stderr=devnull)
        if proc:
            proc.wait()

def test_normal():
    if False:
        while True:
            i = 10
    invoke_program()

class OomTimes(object):

    def __init__(self, real, n):
        if False:
            while True:
                i = 10
        self.real = real
        self.n = n

    def __call__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        if self.n == 0:
            return self.real(*args, **kwargs)
        else:
            self.n -= 1
            e = OSError('faked oom')
            e.errno = errno.ENOMEM
            raise e

def pytest_generate_tests(metafunc):
    if False:
        return 10
    if 'oomtimes' in metafunc.funcargnames:
        scenarios = [OomTimes(subprocess.Popen, n) for n in [0, 1, 2, 10]]
        metafunc.parametrize('oomtimes', scenarios)

def test_low_mem(oomtimes, monkeypatch):
    if False:
        return 10
    monkeypatch.setattr(subprocess, 'Popen', oomtimes)
    invoke_program()

def test_advanced_shim(oomtimes, monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.setattr(subprocess, 'Popen', oomtimes)
    old_n = oomtimes.n

    def reset():
        if False:
            while True:
                i = 10
        oomtimes.n = old_n

    def invoke(max_tries):
        if False:
            return 10
        with open(os.devnull, 'w') as devnull:
            popen = piper.PopenShim(sleep_time=0, max_tries=max_tries)
            proc = popen(['python', '--version'], stdout=devnull, stderr=devnull)
            if proc:
                proc.wait()
    if oomtimes.n >= 1:
        with pytest.raises(OSError) as e:
            invoke(oomtimes.n - 1)
        assert e.value.errno == errno.ENOMEM
    else:
        invoke(oomtimes.n - 1)
    reset()
    invoke(oomtimes.n)
    reset()
    invoke(oomtimes.n + 1)
    reset()