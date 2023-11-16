from concurrent.futures import Future, wait
import anyio
from litestar.testing import create_test_client

def test_with_portal() -> None:
    if False:
        for i in range(10):
            print('nop')
    'This example shows how to manage asynchronous tasks using a portal.\n\n    The test function itself is not async. Asynchronous functions are executed and awaited using the portal.\n    '

    async def get_float(value: float) -> float:
        await anyio.sleep(value)
        return value
    with create_test_client(route_handlers=[]) as test_client, test_client.portal() as portal:
        future: Future[float] = portal.start_task_soon(get_float, 0.25)
        assert portal.call(get_float, 0.1) == 0.1
        wait([future])
        assert future.done()
        assert future.result() == 0.25