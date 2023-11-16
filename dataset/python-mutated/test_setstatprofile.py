import sys
import time
from typing import Any
import pytest
from ..util import busy_wait, flaky_in_ci
from .util import parametrize_setstatprofile

class CallCounter:

    def __init__(self) -> None:
        if False:
            return 10
        self.count = 0

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        self.count += 1

@flaky_in_ci
@parametrize_setstatprofile
def test_100ms(setstatprofile):
    if False:
        i = 10
        return i + 15
    counter = CallCounter()
    setstatprofile(counter, 0.1)
    busy_wait(1.0)
    setstatprofile(None)
    assert 8 < counter.count < 12

@flaky_in_ci
@parametrize_setstatprofile
def test_10ms(setstatprofile):
    if False:
        i = 10
        return i + 15
    counter = CallCounter()
    setstatprofile(counter, 0.01)
    busy_wait(1.0)
    setstatprofile(None)
    assert 70 <= counter.count <= 130

@parametrize_setstatprofile
def test_internal_object_compatibility(setstatprofile):
    if False:
        print('Hello World!')
    setstatprofile(CallCounter(), 1000000.0)
    profile_state = sys.getprofile()
    print(repr(profile_state))
    print(str(profile_state))
    print(profile_state)
    print(type(profile_state))
    print(type(profile_state).__name__)
    setstatprofile(None)