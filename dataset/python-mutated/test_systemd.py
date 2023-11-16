from contextlib import contextmanager
import os
from unittest import mock
import pytest
from gunicorn import systemd

@contextmanager
def check_environ(unset=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    A context manager that asserts post-conditions of ``listen_fds`` at exit.\n\n    This helper is used to ease checking of the test post-conditions for the\n    systemd socket activation tests that parametrize the call argument.\n    '
    with mock.patch.dict(os.environ):
        old_fds = os.environ.get('LISTEN_FDS', None)
        old_pid = os.environ.get('LISTEN_PID', None)
        yield
        if unset:
            assert 'LISTEN_FDS' not in os.environ, 'LISTEN_FDS should have been unset'
            assert 'LISTEN_PID' not in os.environ, 'LISTEN_PID should have been unset'
        else:
            new_fds = os.environ.get('LISTEN_FDS', None)
            new_pid = os.environ.get('LISTEN_PID', None)
            assert new_fds == old_fds, 'LISTEN_FDS should not have been changed'
            assert new_pid == old_pid, 'LISTEN_PID should not have been changed'

@pytest.mark.parametrize('unset', [True, False])
def test_listen_fds_ignores_wrong_pid(unset):
    if False:
        return 10
    with mock.patch.dict(os.environ):
        os.environ['LISTEN_FDS'] = str(5)
        os.environ['LISTEN_PID'] = str(1)
        with check_environ(False):
            assert systemd.listen_fds(unset) == 0, 'should ignore listen fds not intended for this pid'

@pytest.mark.parametrize('unset', [True, False])
def test_listen_fds_returns_count(unset):
    if False:
        return 10
    with mock.patch.dict(os.environ):
        os.environ['LISTEN_FDS'] = str(5)
        os.environ['LISTEN_PID'] = str(os.getpid())
        with check_environ(unset):
            assert systemd.listen_fds(unset) == 5, 'should return the correct count of fds'