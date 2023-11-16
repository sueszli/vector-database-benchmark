import time
try:
    from unittest import mock
except ImportError:
    import mock
import pytest
from azure.core import PipelineClient
from azure.core.exceptions import ServiceResponseError
from azure.core.polling import *
from azure.core.polling.base_polling import LROBasePolling, LocationPolling

@pytest.fixture
def client():
    if False:
        print('Hello World!')
    return PipelineClient('https://baseurl')

def test_abc_polling():
    if False:
        i = 10
        return i + 15
    abc_polling = PollingMethod()
    with pytest.raises(NotImplementedError):
        abc_polling.initialize(None, None, None)
    with pytest.raises(NotImplementedError):
        abc_polling.run()
    with pytest.raises(NotImplementedError):
        abc_polling.status()
    with pytest.raises(NotImplementedError):
        abc_polling.finished()
    with pytest.raises(NotImplementedError):
        abc_polling.resource()
    with pytest.raises(TypeError):
        abc_polling.get_continuation_token()
    with pytest.raises(TypeError):
        abc_polling.from_continuation_token('token')

def test_no_polling(client):
    if False:
        print('Hello World!')
    no_polling = NoPolling()
    initial_response = 'initial response'

    def deserialization_cb(response):
        if False:
            print('Hello World!')
        assert response == initial_response
        return 'Treated: ' + response
    no_polling.initialize(client, initial_response, deserialization_cb)
    no_polling.run()
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

def test_polling_with_path_format_arguments(client):
    if False:
        while True:
            i = 10
    method = LROBasePolling(timeout=0, path_format_arguments={'host': 'host:3000', 'accountName': 'local'})
    client._base_url = 'http://{accountName}{host}'
    method._operation = LocationPolling()
    method._operation._location_url = '/results/1'
    method._client = client
    assert 'http://localhost:3000/results/1' == method._client.format_url(method._operation.get_polling_url(), **method._path_format_arguments)

class PollingTwoSteps(PollingMethod):
    """An empty poller that returns the deserialized initial response."""

    def __init__(self, sleep=0):
        if False:
            while True:
                i = 10
        self._initial_response = None
        self._deserialization_callback = None
        self._sleep = sleep

    def initialize(self, _, initial_response, deserialization_callback):
        if False:
            for i in range(10):
                print('nop')
        self._initial_response = initial_response
        self._deserialization_callback = deserialization_callback
        self._finished = False

    def run(self):
        if False:
            return 10
        'Empty run, no polling.'
        self._finished = True
        time.sleep(self._sleep)

    def status(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the current status as a string.\n        :rtype: str\n        '
        return 'succeeded' if self._finished else 'running'

    def finished(self):
        if False:
            i = 10
            return i + 15
        'Is this polling finished?\n        :rtype: bool\n        '
        return self._finished

    def resource(self):
        if False:
            print('Hello World!')
        return self._deserialization_callback(self._initial_response)

    def get_continuation_token(self):
        if False:
            for i in range(10):
                print('nop')
        return self._initial_response

    @classmethod
    def from_continuation_token(cls, continuation_token, **kwargs):
        if False:
            print('Hello World!')
        initial_response = continuation_token
        deserialization_callback = kwargs['deserialization_callback']
        return (None, initial_response, deserialization_callback)

def test_poller(client):
    if False:
        i = 10
        return i + 15
    initial_response = 'Initial response'

    def deserialization_callback(response):
        if False:
            for i in range(10):
                print('nop')
        assert response == initial_response
        return 'Treated: ' + response
    method = NoPolling()
    poller = LROPoller(client, initial_response, deserialization_callback, method)
    done_cb = mock.MagicMock()
    poller.add_done_callback(done_cb)
    result = poller.result()
    assert poller.done()
    assert result == 'Treated: ' + initial_response
    assert poller.status() == 'succeeded'
    assert poller.polling_method() is method
    done_cb.assert_called_once_with(method)
    method = PollingTwoSteps(sleep=1)
    poller = LROPoller(client, initial_response, deserialization_callback, method)
    done_cb = mock.MagicMock()
    done_cb2 = mock.MagicMock()
    poller.add_done_callback(done_cb)
    poller.remove_done_callback(done_cb2)
    result = poller.result()
    assert result == 'Treated: ' + initial_response
    assert poller.status() == 'succeeded'
    done_cb.assert_called_once_with(method)
    done_cb2.assert_not_called()
    with pytest.raises(ValueError) as excinfo:
        poller.remove_done_callback(done_cb)
    assert 'Process is complete' in str(excinfo.value)
    cont_token = poller.continuation_token()
    method = PollingTwoSteps(sleep=1)
    new_poller = LROPoller.from_continuation_token(continuation_token=cont_token, client=client, initial_response=initial_response, deserialization_callback=deserialization_callback, polling_method=method)
    result = new_poller.result()
    assert result == 'Treated: ' + initial_response
    assert new_poller.status() == 'succeeded'

def test_broken_poller(client):
    if False:
        while True:
            i = 10

    class NoPollingError(PollingTwoSteps):

        def run(self):
            if False:
                return 10
            raise ValueError('Something bad happened')
    initial_response = 'Initial response'

    def deserialization_callback(response):
        if False:
            return 10
        return 'Treated: ' + response
    method = NoPollingError()
    poller = LROPoller(client, initial_response, deserialization_callback, method)
    with pytest.raises(ValueError) as excinfo:
        poller.result()
    assert 'Something bad happened' in str(excinfo.value)

def test_poller_error_continuation(client):
    if False:
        i = 10
        return i + 15

    class NoPollingError(PollingTwoSteps):

        def run(self):
            if False:
                while True:
                    i = 10
            raise ServiceResponseError('Something bad happened')
    initial_response = 'Initial response'

    def deserialization_callback(response):
        if False:
            print('Hello World!')
        return 'Treated: ' + response
    method = NoPollingError()
    poller = LROPoller(client, initial_response, deserialization_callback, method)
    with pytest.raises(ServiceResponseError) as excinfo:
        poller.result()
    assert 'Something bad happened' in str(excinfo.value)
    assert excinfo.value.continuation_token == 'Initial response'