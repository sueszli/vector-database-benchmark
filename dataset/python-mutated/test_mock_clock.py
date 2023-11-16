import time
from math import inf
import pytest
from trio import sleep
from ... import _core
from .. import wait_all_tasks_blocked
from .._mock_clock import MockClock
from .tutil import slow

def test_mock_clock() -> None:
    if False:
        print('Hello World!')
    REAL_NOW = 123.0
    c = MockClock()
    c._real_clock = lambda : REAL_NOW
    repr(c)
    assert c.rate == 0
    assert c.current_time() == 0
    c.jump(1.2)
    assert c.current_time() == 1.2
    with pytest.raises(ValueError):
        c.jump(-1)
    assert c.current_time() == 1.2
    assert c.deadline_to_sleep_time(1.1) == 0
    assert c.deadline_to_sleep_time(1.2) == 0
    assert c.deadline_to_sleep_time(1.3) > 999999
    with pytest.raises(ValueError):
        c.rate = -1
    assert c.rate == 0
    c.rate = 2
    assert c.current_time() == 1.2
    REAL_NOW += 1
    assert c.current_time() == 3.2
    assert c.deadline_to_sleep_time(3.1) == 0
    assert c.deadline_to_sleep_time(3.2) == 0
    assert c.deadline_to_sleep_time(4.2) == 0.5
    c.rate = 0.5
    assert c.current_time() == 3.2
    assert c.deadline_to_sleep_time(3.1) == 0
    assert c.deadline_to_sleep_time(3.2) == 0
    assert c.deadline_to_sleep_time(4.2) == 2.0
    c.jump(0.8)
    assert c.current_time() == 4.0
    REAL_NOW += 1
    assert c.current_time() == 4.5
    c2 = MockClock(rate=3)
    assert c2.rate == 3
    assert c2.current_time() < 10

async def test_mock_clock_autojump(mock_clock: MockClock) -> None:
    assert mock_clock.autojump_threshold == inf
    mock_clock.autojump_threshold = 0
    assert mock_clock.autojump_threshold == 0
    real_start = time.perf_counter()
    virtual_start = _core.current_time()
    for i in range(10):
        print(f'sleeping {10 * i} seconds')
        await sleep(10 * i)
        print('woke up!')
        assert virtual_start + 10 * i == _core.current_time()
        virtual_start = _core.current_time()
    real_duration = time.perf_counter() - real_start
    print(f'Slept {10 * sum(range(10))} seconds in {real_duration} seconds')
    assert real_duration < 1
    mock_clock.autojump_threshold = 0.02
    t = _core.current_time()
    await wait_all_tasks_blocked()
    assert t == _core.current_time()
    await wait_all_tasks_blocked(0.01)
    assert t == _core.current_time()
    mock_clock.autojump_threshold = 10000
    await wait_all_tasks_blocked()
    mock_clock.autojump_threshold = 0
    await sleep(100000)

async def test_mock_clock_autojump_interference(mock_clock: MockClock) -> None:
    mock_clock.autojump_threshold = 0.02
    mock_clock2 = MockClock()
    mock_clock2.autojump_threshold = 0.01
    await wait_all_tasks_blocked(0.015)
    await sleep(100000)

def test_mock_clock_autojump_preset() -> None:
    if False:
        print('Hello World!')
    mock_clock = MockClock(autojump_threshold=0.1)
    mock_clock.autojump_threshold = 0.01
    real_start = time.perf_counter()
    _core.run(sleep, 10000, clock=mock_clock)
    assert time.perf_counter() - real_start < 1

async def test_mock_clock_autojump_0_and_wait_all_tasks_blocked_0(mock_clock: MockClock) -> None:
    mock_clock.autojump_threshold = 0
    record = []

    async def sleeper() -> None:
        await sleep(100)
        record.append('yawn')

    async def waiter() -> None:
        await wait_all_tasks_blocked()
        record.append('waiter woke')
        await sleep(1000)
        record.append('waiter done')
    async with _core.open_nursery() as nursery:
        nursery.start_soon(sleeper)
        nursery.start_soon(waiter)
    assert record == ['waiter woke', 'yawn', 'waiter done']

@slow
async def test_mock_clock_autojump_0_and_wait_all_tasks_blocked_nonzero(mock_clock: MockClock) -> None:
    mock_clock.autojump_threshold = 0
    record = []

    async def sleeper() -> None:
        await sleep(100)
        record.append('yawn')

    async def waiter() -> None:
        await wait_all_tasks_blocked(1)
        record.append('waiter done')
    async with _core.open_nursery() as nursery:
        nursery.start_soon(sleeper)
        nursery.start_soon(waiter)
    assert record == ['waiter done', 'yawn']