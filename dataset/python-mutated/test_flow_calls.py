import asyncio
import inspect
import anyio
import prefect

@prefect.flow
def identity_flow(x):
    if False:
        print('Hello World!')
    return x

@prefect.flow
async def aidentity_flow(x):
    return x

def test_async_flow_called_with_asyncio():
    if False:
        print('Hello World!')
    coro = aidentity_flow(1)
    assert inspect.isawaitable(coro)
    assert asyncio.run(coro) == 1

def test_async_flow_called_with_anyio():
    if False:
        while True:
            i = 10
    assert anyio.run(aidentity_flow, 1) == 1

async def test_async_flow_called_with_running_loop():
    coro = aidentity_flow(1)
    assert inspect.isawaitable(coro)
    assert await coro == 1

def test_sync_flow_called():
    if False:
        print('Hello World!')
    assert identity_flow(1) == 1

async def test_sync_flow_called_with_running_loop():
    assert identity_flow(1) == 1