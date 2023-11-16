from typing import Any
from twisted.web.server import Request
from synapse.http.additional_resource import AdditionalResource
from synapse.http.server import respond_with_json
from synapse.http.site import SynapseRequest
from synapse.types import JsonDict
from tests.server import FakeSite, make_request
from tests.unittest import HomeserverTestCase

class _AsyncTestCustomEndpoint:

    def __init__(self, config: JsonDict, module_api: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    async def handle_request(self, request: Request) -> None:
        assert isinstance(request, SynapseRequest)
        respond_with_json(request, 200, {'some_key': 'some_value_async'})

class _SyncTestCustomEndpoint:

    def __init__(self, config: JsonDict, module_api: Any) -> None:
        if False:
            return 10
        pass

    async def handle_request(self, request: Request) -> None:
        assert isinstance(request, SynapseRequest)
        respond_with_json(request, 200, {'some_key': 'some_value_sync'})

class AdditionalResourceTests(HomeserverTestCase):
    """Very basic tests that `AdditionalResource` works correctly with sync
    and async handlers.
    """

    def test_async(self) -> None:
        if False:
            i = 10
            return i + 15
        handler = _AsyncTestCustomEndpoint({}, None).handle_request
        resource = AdditionalResource(self.hs, handler)
        channel = make_request(self.reactor, FakeSite(resource, self.reactor), 'GET', '/')
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body, {'some_key': 'some_value_async'})

    def test_sync(self) -> None:
        if False:
            return 10
        handler = _SyncTestCustomEndpoint({}, None).handle_request
        resource = AdditionalResource(self.hs, handler)
        channel = make_request(self.reactor, FakeSite(resource, self.reactor), 'GET', '/')
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body, {'some_key': 'some_value_sync'})