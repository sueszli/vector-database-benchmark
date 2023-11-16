from unittest.mock import AsyncMock, patch
from tribler.core.components.restapi.rest.rest_endpoint import RESTEndpoint
from tribler.core.utilities.async_group.async_group import AsyncGroup

async def test_shutdown():

    async def coro():
        ...
    root_endpoint = RESTEndpoint()
    root_endpoint.async_group.add_task(coro())
    child_endpoints = [RESTEndpoint(), RESTEndpoint()]
    for (index, child_endpoint) in enumerate(child_endpoints):
        root_endpoint.add_endpoint(prefix=f'/{index}', endpoint=child_endpoint)
        child_endpoint.async_group.add_task(coro())

    def total_coro_count():
        if False:
            for i in range(10):
                print('nop')
        count = 0
        for endpoint in child_endpoints + [root_endpoint]:
            count += len(endpoint.async_group.futures)
        return count
    assert total_coro_count() == 3
    await root_endpoint.shutdown()
    assert total_coro_count() == 0

@patch.object(AsyncGroup, 'cancel', new_callable=AsyncMock)
async def test_multiple_shutdown_calls(async_group_cancel: AsyncMock):
    endpoint = RESTEndpoint()
    await endpoint.shutdown()
    await endpoint.shutdown()
    async_group_cancel.assert_called_once()