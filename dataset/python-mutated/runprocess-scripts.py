from __future__ import absolute_import
from __future__ import print_function
import os
import select
import signal
import subprocess
import sys
import time
import psutil

def invoke_script(function, *args):
    if False:
        print('Hello World!')
    cmd = [sys.executable, __file__, function] + list(args)
    if os.name == 'nt':
        DETACHED_PROCESS = 8
        subprocess.Popen(cmd, shell=False, stdin=None, stdout=None, stderr=None, close_fds=True, creationflags=DETACHED_PROCESS)
    else:
        subprocess.Popen(cmd, shell=False, stdin=None, stdout=None, stderr=None, close_fds=True)

def write_pidfile(pidfile):
    if False:
        return 10
    pidfile_tmp = pidfile + '~'
    f = open(pidfile_tmp, 'w')
    f.write(str(os.getpid()))
    f.close()
    os.rename(pidfile_tmp, pidfile)

def sleep_forever():
    if False:
        return 10
    signal.alarm(110)
    while True:
        time.sleep(10)
script_fns = {}

def script(fn):
    if False:
        print('Hello World!')
    script_fns[fn.__name__] = fn
    return fn

@script
def write_pidfile_and_sleep():
    if False:
        i = 10
        return i + 15
    pidfile = sys.argv[2]
    write_pidfile(pidfile)
    sleep_forever()

@script
def spawn_child():
    if False:
        print('Hello World!')
    (parent_pidfile, child_pidfile) = sys.argv[2:]
    invoke_script('write_pidfile_and_sleep', child_pidfile)
    write_pidfile(parent_pidfile)
    sleep_forever()

@script
def wait_for_pid_death_and_write_pidfile_and_sleep():
    if False:
        while True:
            i = 10
    wait_pid = int(sys.argv[2])
    pidfile = sys.argv[3]
    while psutil.pid_exists(wait_pid):
        time.sleep(0.01)
    write_pidfile(pidfile)
    sleep_forever()

@script
def double_fork():
    if False:
        print('Hello World!')
    if os.name == 'posix':
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    (parent_pidfile, child_pidfile) = sys.argv[2:]
    parent_pid = os.getpid()
    invoke_script('wait_for_pid_death_and_write_pidfile_and_sleep', str(parent_pid), child_pidfile)
    write_pidfile(parent_pidfile)
    sys.exit(0)

@script
def assert_stdin_closed():
    if False:
        for i in range(10):
            print('nop')
    bail_at = time.time() + 10
    while True:
        (r, _, __) = select.select([0], [], [], 0.01)
        if r == [0]:
            return
        if time.time() > bail_at:
            assert False
if not hasattr(signal, 'alarm'):
    signal.alarm = lambda t: None
signal.alarm(110)
script_fns[sys.argv[1]]()