import asyncio
import contextlib
import functools
import random
from typing import TYPE_CHECKING
from unittest import mock
from pyinstrument import stack_sampler
if TYPE_CHECKING:
    from trio.testing import MockClock

class FakeClock:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.time = random.random() * 1000000.0

    def get_time(self):
        if False:
            i = 10
            return i + 15
        return self.time

    def sleep(self, duration):
        if False:
            return 10
        self.time += duration

@contextlib.contextmanager
def fake_time(fake_clock=None):
    if False:
        for i in range(10):
            print('nop')
    fake_clock = fake_clock or FakeClock()
    stack_sampler.get_stack_sampler().timer_func = fake_clock.get_time
    try:
        with mock.patch('time.sleep', new=fake_clock.sleep):
            yield fake_clock
    finally:
        stack_sampler.get_stack_sampler().timer_func = None

class FakeClockAsyncio:

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.time = random.random() * 1000000.0

    def get_time(self):
        if False:
            print('Hello World!')
        return self.time

    def sleep(self, duration):
        if False:
            i = 10
            return i + 15
        self.time += duration

    def _virtual_select(self, orig_select, timeout):
        if False:
            return 10
        self.time += timeout
        return orig_select(0)

@contextlib.contextmanager
def fake_time_asyncio(loop=None):
    if False:
        while True:
            i = 10
    loop = loop or asyncio.get_running_loop()
    fake_clock = FakeClockAsyncio()
    with mock.patch.object(loop._selector, 'select', new=functools.partial(fake_clock._virtual_select, loop._selector.select)), mock.patch.object(loop, 'time', new=fake_clock.get_time), fake_time(fake_clock):
        yield fake_clock

class FakeClockTrio:

    def __init__(self, clock: 'MockClock') -> None:
        if False:
            i = 10
            return i + 15
        self.trio_clock = clock

    def get_time(self):
        if False:
            return 10
        return self.trio_clock.current_time()

    def sleep(self, duration):
        if False:
            while True:
                i = 10
        self.trio_clock.jump(duration)

@contextlib.contextmanager
def fake_time_trio():
    if False:
        for i in range(10):
            print('nop')
    from trio.testing import MockClock
    trio_clock = MockClock(autojump_threshold=0)
    fake_clock = FakeClockTrio(trio_clock)
    with fake_time(fake_clock):
        yield fake_clock