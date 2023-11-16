"""Unit tests to verify the functionality of the ansible-forked pytest plugin."""
from __future__ import annotations
import os
import pytest
import signal
import sys
import warnings
warnings.warn('This verifies that warnings generated during test collection are reported.')

@pytest.mark.xfail
def test_kill_xfail():
    if False:
        while True:
            i = 10
    os.kill(os.getpid(), signal.SIGKILL)

def test_kill():
    if False:
        print('Hello World!')
    os.kill(os.getpid(), signal.SIGKILL)

@pytest.mark.xfail
def test_exception_xfail():
    if False:
        i = 10
        return i + 15
    sys.stdout.write('This stdout message should be hidden due to xfail.')
    sys.stderr.write('This stderr message should be hidden due to xfail.')
    raise Exception('This error is expected, but should be hidden due to xfail.')

def test_exception():
    if False:
        i = 10
        return i + 15
    sys.stdout.write("This stdout message should be reported since we're throwing an exception.")
    sys.stderr.write("This stderr message should be reported since we're throwing an exception.")
    raise Exception('This error is expected and should be visible.')

def test_warning():
    if False:
        print('Hello World!')
    warnings.warn('This verifies that warnings generated at test time are reported.')

def test_passed():
    if False:
        i = 10
        return i + 15
    pass