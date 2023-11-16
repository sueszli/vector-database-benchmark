import os
import signal
import time
from dramatiq.brokers.stub import StubBroker
from .common import skip_on_windows
broker = StubBroker()

def remove(filename):
    if False:
        while True:
            i = 10
    try:
        os.remove(filename)
    except OSError:
        pass

@skip_on_windows
def test_cli_scrubs_stale_pid_files(start_cli):
    if False:
        return 10
    try:
        filename = 'test_scrub.pid'
        with open(filename, 'w') as f:
            f.write('999999')
        proc = start_cli('tests.test_pidfile:broker', extra_args=['--pid-file', filename])
        time.sleep(1)
        with open(filename, 'r') as f:
            pid = int(f.read())
        assert pid == proc.pid
        proc.terminate()
        proc.wait()
        assert proc.returncode == 0
        assert not os.path.exists(filename)
    finally:
        remove(filename)

def test_cli_aborts_when_pidfile_contains_garbage(start_cli):
    if False:
        return 10
    try:
        filename = 'test_garbage.pid'
        with open(filename, 'w') as f:
            f.write('important!')
        proc = start_cli('tests.test_pidfile:broker', extra_args=['--pid-file', filename])
        proc.wait()
        assert proc.returncode == 4
    finally:
        remove(filename)

@skip_on_windows
def test_cli_with_pidfile_can_be_reloaded(start_cli):
    if False:
        for i in range(10):
            print('nop')
    try:
        filename = 'test_reload.pid'
        proc = start_cli('tests.test_pidfile:broker', extra_args=['--pid-file', filename])
        time.sleep(1)
        proc.send_signal(signal.SIGHUP)
        time.sleep(5)
        proc.terminate()
        proc.wait()
        assert proc.returncode == 0
    finally:
        remove(filename)