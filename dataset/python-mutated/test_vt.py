import os
import signal
import pytest
import salt.utils.vt as vt

@pytest.mark.skip_on_windows(reason="salt.utils.vt.Terminal doesn't have _spawn.")
def test_isalive_no_child():
    if False:
        for i in range(10):
            print('nop')
    term = vt.Terminal('sleep 100', shell=True, stream_stdout=False, stream_stderr=False)
    aliveness = term.isalive()
    assert term.exitstatus is None
    assert aliveness is True
    os.kill(term.pid, signal.SIGKILL)
    os.waitpid(term.pid, 0)
    aliveness = term.isalive()
    assert term.exitstatus == 0
    assert aliveness is False

@pytest.mark.parametrize('test_cmd', ['echo', 'ls'])
@pytest.mark.skip_on_windows()
def test_log_sanitize(test_cmd, caplog):
    if False:
        return 10
    '\n    test when log_sanitize is passed in\n    we do not see the password in either\n    standard out or standard error logs\n    '
    password = '123456'
    cmd = [test_cmd, password]
    term = vt.Terminal(cmd, log_stdout=True, log_stderr=True, log_sanitize=password, stream_stdout=False, stream_stderr=False)
    ret = term.recv()
    assert password not in caplog.text
    assert '******' in caplog.text