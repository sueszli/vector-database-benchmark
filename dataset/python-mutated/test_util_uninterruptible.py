"""Unit tests for ``uninterruptible`` decorator."""
from __future__ import print_function, absolute_import
import os
import signal
import pytest
from workflow.util import uninterruptible

class Target(object):
    """Object to be manipulated by :func:`fakewrite`."""

    def __init__(self, kill=True, finished=False):
        if False:
            i = 10
            return i + 15
        'Create new `Target`.'
        self.kill = kill
        self.finished = finished
        self.handled = False

    def handler(self, signum, frame):
        if False:
            i = 10
            return i + 15
        'Alternate signal handler.'
        self.handled = True

@uninterruptible
def fakewrite(target):
    if False:
        while True:
            i = 10
    'Mock writer.\n\n    Sets ``target.finished`` if it completes.\n\n    Args:\n        target (Target): Object to set status on\n    '
    if target.kill:
        target.kill = False
        os.kill(os.getpid(), signal.SIGTERM)
    target.finished = True

@pytest.fixture(scope='function')
def target():
    if False:
        print('Hello World!')
    'Create a `Target`.'
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    yield Target()
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

def test_normal(target):
    if False:
        i = 10
        return i + 15
    'Normal writing operator'
    target.kill = False
    fakewrite(target)
    assert target.finished

def test_sigterm_signal(target):
    if False:
        while True:
            i = 10
    'Process is killed, but call completes'
    with pytest.raises(SystemExit):
        fakewrite(target)
    assert target.finished
    assert not target.kill

def test_old_signal_handler(target):
    if False:
        return 10
    'Kill with different signal handler registered'
    signal.signal(signal.SIGTERM, target.handler)
    fakewrite(target)
    assert target.finished
    assert target.handled
    assert not target.kill

def test_old_signal_handler_restore(target):
    if False:
        i = 10
        return i + 15
    'Restore previous signal handler after write'
    signal.signal(signal.SIGTERM, target.handler)
    target.kill = False
    fakewrite(target)
    assert target.finished
    assert signal.getsignal(signal.SIGTERM) == target.handler
if __name__ == '__main__':
    pytest.main([__file__])