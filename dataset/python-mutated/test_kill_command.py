import argparse
import multiprocessing
import os
import subprocess
import signal
import sys
import signal
import time
import pytest
from nni.tools.nnictl.command_utils import kill_command, _check_pid_running
pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='Windows has confirmation upon process killing.')

def process_normal():
    if False:
        return 10
    time.sleep(360)

def process_kill_slow(kill_time=2):
    if False:
        i = 10
        return i + 15

    def handler_stop_signals(signum, frame):
        if False:
            print('Hello World!')
        time.sleep(kill_time)
        sys.exit(0)
    signal.signal(signal.SIGINT, handler_stop_signals)
    signal.signal(signal.SIGTERM, handler_stop_signals)
    time.sleep(360)

def process_patiently_kill():
    if False:
        print('Hello World!')
    process = subprocess.Popen([sys.executable, __file__, '--mode', 'kill_very_slow'])
    time.sleep(1)
    kill_command(process.pid)

@pytest.mark.skip(reason='The test has too many failures.')
def test_kill_process():
    if False:
        while True:
            i = 10
    process = multiprocessing.Process(target=process_normal)
    process.start()
    time.sleep(0.5)
    start_time = time.time()
    kill_command(process.pid)
    end_time = time.time()
    assert not _check_pid_running(process.pid)
    assert end_time - start_time < 2

@pytest.mark.skip(reason='The test has too many failures.')
def test_kill_process_slow_no_patience():
    if False:
        while True:
            i = 10
    process = subprocess.Popen([sys.executable, __file__, '--mode', 'kill_slow'])
    time.sleep(1)
    start_time = time.time()
    kill_command(process.pid, timeout=1)
    end_time = time.time()
    if sys.platform == 'linux':
        assert end_time - start_time < 2
        assert process.poll() is None
        assert _check_pid_running(process.pid)
    else:
        assert end_time - start_time < 2
    for _ in range(20):
        time.sleep(1)
        if not _check_pid_running(process.pid):
            return

@pytest.mark.skip(reason='The test has too many failures.')
def test_kill_process_slow_patiently():
    if False:
        i = 10
        return i + 15
    process = subprocess.Popen([sys.executable, __file__, '--mode', 'kill_slow'])
    time.sleep(1)
    start_time = time.time()
    kill_command(process.pid, timeout=3)
    end_time = time.time()
    assert end_time - start_time < 5

@pytest.mark.skip(reason='The test has too many failures.')
def test_kill_process_interrupted():
    if False:
        return 10
    process = multiprocessing.Process(target=process_patiently_kill)
    process.start()
    time.sleep(3)
    os.kill(process.pid, signal.SIGINT)
    assert process.is_alive()
    time.sleep(0.5)
    os.kill(process.pid, signal.SIGINT)
    time.sleep(0.5)
    assert not process.is_alive()
    if sys.platform == 'linux':
        assert process.exitcode != 0

def start_new_process_group(cmd):
    if False:
        print('Hello World!')
    if sys.platform == 'win32':
        return subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        return subprocess.Popen(cmd, preexec_fn=os.setpgrp)

def main():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['kill_slow', 'kill_very_slow'])
    args = parser.parse_args()
    if args.mode == 'kill_slow':
        process_kill_slow()
    elif args.mode == 'kill_very_slow':
        process_kill_slow(15)
    else:
        pass
if __name__ == '__main__':
    main()