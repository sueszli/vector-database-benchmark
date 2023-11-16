from __future__ import absolute_import, print_function, division
import sys
import functools
import unittest
from . import sysinfo
from . import six

class FlakyAssertionError(AssertionError):
    """Re-raised so that we know it's a known-flaky test."""

class FlakyTest(unittest.SkipTest):
    """
    A unittest exception that causes the test to be skipped when raised.

    Use this carefully, it is a code smell and indicates an undebugged problem.
    """

class FlakyTestRaceCondition(FlakyTest):
    """
    Use this when the flaky test is definitely caused by a race condition.
    """

class FlakyTestTimeout(FlakyTest):
    """
    Use this when the flaky test is definitely caused by an
    unexpected timeout.
    """

class FlakyTestCrashes(FlakyTest):
    """
    Use this when the test sometimes crashes.
    """

def reraiseFlakyTestRaceCondition():
    if False:
        for i in range(10):
            print('nop')
    six.reraise(FlakyAssertionError, FlakyAssertionError(sys.exc_info()[1]), sys.exc_info()[2])
reraiseFlakyTestTimeout = reraiseFlakyTestRaceCondition
reraiseFlakyTestRaceConditionLibuv = reraiseFlakyTestRaceCondition
reraiseFlakyTestTimeoutLibuv = reraiseFlakyTestRaceCondition
if sysinfo.RUNNING_ON_CI or (sysinfo.PYPY and sysinfo.WIN):

    def reraiseFlakyTestRaceCondition():
        if False:
            print('Hello World!')
        msg = str(sys.exc_info()[1])
        six.reraise(FlakyTestRaceCondition, FlakyTestRaceCondition(msg), sys.exc_info()[2])

    def reraiseFlakyTestTimeout():
        if False:
            i = 10
            return i + 15
        msg = str(sys.exc_info()[1])
        six.reraise(FlakyTestTimeout, FlakyTestTimeout(msg), sys.exc_info()[2])
    if sysinfo.LIBUV:
        reraiseFlakyTestRaceConditionLibuv = reraiseFlakyTestRaceCondition
        reraiseFlakyTestTimeoutLibuv = reraiseFlakyTestTimeout

def reraises_flaky_timeout(exc_kind=AssertionError, _func=reraiseFlakyTestTimeout):
    if False:
        while True:
            i = 10

    def wrapper(f):
        if False:
            print('Hello World!')

        @functools.wraps(f)
        def m(*args):
            if False:
                print('Hello World!')
            try:
                f(*args)
            except exc_kind:
                _func()
        return m
    return wrapper

def reraises_flaky_race_condition(exc_kind=AssertionError):
    if False:
        i = 10
        return i + 15
    return reraises_flaky_timeout(exc_kind, _func=reraiseFlakyTestRaceCondition)