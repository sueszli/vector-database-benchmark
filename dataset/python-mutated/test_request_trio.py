import json
from azure.core.pipeline.transport import TrioRequestsTransport
from utils import HTTP_REQUESTS
from azure.core.pipeline._tools import is_rest
import pytest

@pytest.mark.trio
@pytest.mark.parametrize('http_request', HTTP_REQUESTS)
async def test_async_gen_data(port, http_request):

    class AsyncGen:

        def __init__(self):
            if False:
                return 10
            self._range = iter([b'azerty'])

        def __aiter__(self):
            if False:
                for i in range(10):
                    print('nop')
            return self

        async def __anext__(self):
            try:
                return next(self._range)
            except StopIteration:
                raise StopAsyncIteration
    async with TrioRequestsTransport() as transport:
        req = http_request('GET', 'http://localhost:{}/basic/anything'.format(port), data=AsyncGen())
        response = await transport.send(req)
        if is_rest(http_request):
            assert is_rest(response)
        assert json.loads(response.text())['data'] == 'azerty'

@pytest.mark.trio
@pytest.mark.parametrize('http_request', HTTP_REQUESTS)
async def test_send_data(port, http_request):
    async with TrioRequestsTransport() as transport:
        req = http_request('PUT', 'http://localhost:{}/basic/anything'.format(port), data=b'azerty')
        response = await transport.send(req)
        if is_rest(http_request):
            assert is_rest(response)
        assert json.loads(response.text())['data'] == 'azerty'