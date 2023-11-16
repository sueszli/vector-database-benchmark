import asyncio
import time
try:
    from unittest import mock
except ImportError:
    import mock
import pytest
from azure.core import AsyncPipelineClient
from azure.core.polling import *
from azure.core.exceptions import ServiceResponseError

@pytest.fixture
def client():
    if False:
        return 10
    return AsyncPipelineClient('https://baseurl')

@pytest.mark.asyncio
async def test_no_polling(client):
    no_polling = AsyncNoPolling()
    initial_response = 'initial response'

    def deserialization_cb(response):
        if False:
            i = 10
            return i + 15
        assert response == initial_response
        return 'Treated: ' + response
    no_polling.initialize(client, initial_response, deserialization_cb)
    await no_polling.run()
    assert no_polling.status() == 'succeeded'
    assert no_polling.finished()
    assert no_polling.resource() == 'Treated: ' + initial_response
    continuation_token = no_polling.get_continuation_token()
    assert isinstance(continuation_token, str)
    no_polling_revived_args = NoPolling.from_continuation_token(continuation_token, deserialization_callback=deserialization_cb, client=client)
    no_polling_revived = NoPolling()
    no_polling_revived.initialize(*no_polling_revived_args)
    assert no_polling_revived.status() == 'succeeded'
    assert no_polling_revived.finished()
    assert no_polling_revived.resource() == 'Treated: ' + initial_response

class PollingTwoSteps(AsyncPollingMethod):
    """An empty poller that returns the deserialized initial response."""

    def __init__(self, sleep=0):
        if False:
            print('Hello World!')
        self._initial_response = None
        self._deserialization_callback = None
        self._sleep = sleep

    def initialize(self, _, initial_response, deserialization_callback):
        if False:
            i = 10
            return i + 15
        self._initial_response = initial_response
        self._deserialization_callback = deserialization_callback
        self._finished = False

    async def run(self):
        """Empty run, no polling."""
        self._finished = True
        await asyncio.sleep(self._sleep)

    def status(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the current status as a string.\n        :rtype: str\n        '
        return 'succeeded' if self._finished else 'running'

    def finished(self):
        if False:
            return 10
        'Is this polling finished?\n        :rtype: bool\n        '
        return self._finished

    def resource(self):
        if False:
            for i in range(10):
                print('nop')
        return self._deserialization_callback(self._initial_response)

    def get_continuation_token(self):
        if False:
            while True:
                i = 10
        return self._initial_response

    @classmethod
    def from_continuation_token(cls, continuation_token, **kwargs):
        if False:
            print('Hello World!')
        initial_response = continuation_token
        deserialization_callback = kwargs['deserialization_callback']
        return (None, initial_response, deserialization_callback)

@pytest.mark.asyncio
async def test_poller(client):
    initial_response = 'Initial response'

    def deserialization_callback(response):
        if False:
            print('Hello World!')
        assert response == initial_response
        return 'Treated: ' + response
    method = AsyncNoPolling()
    raw_poller = AsyncLROPoller(client, initial_response, deserialization_callback, method)
    poller = asyncio.ensure_future(raw_poller.result())
    done_cb = mock.MagicMock()
    poller.add_done_callback(done_cb)
    result = await poller
    assert poller.done()
    assert result == 'Treated: ' + initial_response
    assert raw_poller.status() == 'succeeded'
    assert raw_poller.polling_method() is method
    done_cb.assert_called_once_with(poller)
    method = PollingTwoSteps(sleep=1)
    raw_poller = AsyncLROPoller(client, initial_response, deserialization_callback, method)
    poller = asyncio.ensure_future(raw_poller.result())
    done_cb = mock.MagicMock()
    done_cb2 = mock.MagicMock()
    poller.add_done_callback(done_cb)
    poller.remove_done_callback(done_cb2)
    result = await poller
    assert result == 'Treated: ' + initial_response
    assert raw_poller.status() == 'succeeded'
    done_cb.assert_called_once_with(poller)
    done_cb2.assert_not_called()
    cont_token = raw_poller.continuation_token()
    method = PollingTwoSteps(sleep=1)
    new_poller = AsyncLROPoller.from_continuation_token(continuation_token=cont_token, client=client, initial_response=initial_response, deserialization_callback=deserialization_callback, polling_method=method)
    result = await new_poller.result()
    assert result == 'Treated: ' + initial_response
    assert new_poller.status() == 'succeeded'

@pytest.mark.asyncio
async def test_broken_poller(client):

    class NoPollingError(PollingTwoSteps):

        async def run(self):
            raise ValueError('Something bad happened')
    initial_response = 'Initial response'

    def deserialization_callback(response):
        if False:
            while True:
                i = 10
        return 'Treated: ' + response
    method = NoPollingError()
    poller = AsyncLROPoller(client, initial_response, deserialization_callback, method)
    with pytest.raises(ValueError) as excinfo:
        await poller.result()
    assert 'Something bad happened' in str(excinfo.value)

@pytest.mark.asyncio
async def test_async_poller_error_continuation(client):

    class NoPollingError(PollingTwoSteps):

        async def run(self):
            raise ServiceResponseError('Something bad happened')
    initial_response = 'Initial response'

    def deserialization_callback(response):
        if False:
            while True:
                i = 10
        return 'Treated: ' + response
    method = NoPollingError()
    poller = AsyncLROPoller(client, initial_response, deserialization_callback, method)
    with pytest.raises(ServiceResponseError) as excinfo:
        await poller.result()
    assert 'Something bad happened' in str(excinfo.value)
    assert excinfo.value.continuation_token == 'Initial response'