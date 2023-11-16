from typing import Any
from .util import parametrize_setstatprofile

class CallCounter:

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.count = 0

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if False:
            print('Hello World!')
        self.count += 1

@parametrize_setstatprofile
def test_increment(setstatprofile):
    if False:
        i = 10
        return i + 15
    time = 0.0

    def fake_time():
        if False:
            while True:
                i = 10
        return time

    def fake_sleep(duration):
        if False:
            return 10
        nonlocal time
        time += duration
    counter = CallCounter()
    setstatprofile(counter, timer_func=fake_time)
    for _ in range(100):
        fake_sleep(1.0)
    setstatprofile(None)
    assert counter.count == 100