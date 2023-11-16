import os
import time
import anyio
import pytest
import prefect
SLEEP_TIME = 4 if os.environ.get('CI') else 2

@pytest.mark.timeout(method='thread')
def test_sync_flow_timeout():
    if False:
        while True:
            i = 10

    @prefect.flow(timeout_seconds=0.1)
    def sleep_flow():
        if False:
            while True:
                i = 10
        time.sleep(SLEEP_TIME)
    t0 = time.monotonic()
    state = sleep_flow(return_state=True)
    t1 = time.monotonic()
    runtime = t1 - t0
    assert runtime < SLEEP_TIME, f'Flow should exit early; ran for {runtime}s'
    assert state.is_failed()
    with pytest.raises(TimeoutError):
        state.result()

async def test_async_flow_timeout():

    @prefect.flow(timeout_seconds=0.1)
    async def sleep_flow():
        await anyio.sleep(SLEEP_TIME)
    t0 = time.monotonic()
    state = await sleep_flow(return_state=True)
    t1 = time.monotonic()
    runtime = t1 - t0
    assert runtime < SLEEP_TIME, f'Flow should exit early; ran for {runtime}s'
    assert state.is_failed()
    with pytest.raises(TimeoutError):
        await state.result()

@pytest.mark.timeout(method='thread')
def test_sync_flow_timeout_in_sync_flow():
    if False:
        i = 10
        return i + 15

    @prefect.flow(timeout_seconds=0.1)
    def sleep_flow():
        if False:
            return 10
        time.sleep(SLEEP_TIME)

    @prefect.flow
    def parent_flow():
        if False:
            while True:
                i = 10
        t0 = time.monotonic()
        state = sleep_flow(return_state=True)
        t1 = time.monotonic()
        return (t1 - t0, state)
    (runtime, flow_state) = parent_flow()
    assert runtime < SLEEP_TIME, f'Flow should exit early; ran for {runtime}s'
    assert flow_state.is_failed()
    with pytest.raises(TimeoutError):
        flow_state.result()

async def test_sync_flow_timeout_in_async_flow():

    @prefect.flow(timeout_seconds=0.1)
    def sleep_flow():
        if False:
            i = 10
            return i + 15
        for _ in range(SLEEP_TIME * 10):
            time.sleep(0.1)

    @prefect.flow
    async def parent_flow():
        t0 = time.monotonic()
        state = sleep_flow(return_state=True)
        t1 = time.monotonic()
        return (t1 - t0, state)
    (runtime, flow_state) = await parent_flow()
    assert runtime < SLEEP_TIME, f'Flow should exit early; ran for {runtime}s'
    assert flow_state.is_failed()
    with pytest.raises(TimeoutError):
        await flow_state.result()

def test_async_flow_timeout_in_sync_flow():
    if False:
        while True:
            i = 10

    @prefect.flow(timeout_seconds=0.1)
    async def sleep_flow():
        await anyio.sleep(SLEEP_TIME)

    @prefect.flow
    def parent_flow():
        if False:
            i = 10
            return i + 15
        t0 = time.monotonic()
        state = sleep_flow(return_state=True)
        t1 = time.monotonic()
        return (t1 - t0, state)
    (runtime, flow_state) = parent_flow()
    assert runtime < SLEEP_TIME, f'Flow should exit early; ran for {runtime}s'
    assert flow_state.is_failed()
    with pytest.raises(TimeoutError):
        flow_state.result()

async def test_async_flow_timeout_in_async_flow():

    @prefect.flow(timeout_seconds=0.1)
    async def sleep_flow():
        await anyio.sleep(SLEEP_TIME)

    @prefect.flow
    async def parent_flow():
        t0 = time.monotonic()
        state = await sleep_flow(return_state=True)
        t1 = time.monotonic()
        return (t1 - t0, state)
    (runtime, flow_state) = await parent_flow()
    assert runtime < SLEEP_TIME, f'Flow should exit early; ran for {runtime}s'
    assert flow_state.is_failed()
    with pytest.raises(TimeoutError):
        await flow_state.result()