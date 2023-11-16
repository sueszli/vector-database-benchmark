import json
from http import HTTPStatus
from io import BytesIO
from typing import Tuple, Union
from unittest.mock import Mock
from synapse.api.errors import Codes, SynapseError
from synapse.http.servlet import RestServlet, parse_json_object_from_request, parse_json_value_from_request
from synapse.http.site import SynapseRequest
from synapse.rest.client._base import client_patterns
from synapse.server import HomeServer
from synapse.types import JsonDict
from synapse.util.cancellation import cancellable
from tests import unittest
from tests.http.server._base import test_disconnect

def make_request(content: Union[bytes, JsonDict]) -> Mock:
    if False:
        for i in range(10):
            print('nop')
    'Make an object that acts enough like a request.'
    request = Mock(spec=['method', 'uri', 'content'])
    if isinstance(content, dict):
        content = json.dumps(content).encode('utf8')
    request.method = bytes('STUB_METHOD', 'ascii')
    request.uri = bytes('/test_stub_uri', 'ascii')
    request.content = BytesIO(content)
    return request

class TestServletUtils(unittest.TestCase):

    def test_parse_json_value(self) -> None:
        if False:
            i = 10
            return i + 15
        'Basic tests for parse_json_value_from_request.'
        obj = {'foo': 1}
        result1 = parse_json_value_from_request(make_request(obj))
        self.assertEqual(result1, obj)
        result2 = parse_json_value_from_request(make_request(b'["foo"]'))
        self.assertEqual(result2, ['foo'])
        with self.assertRaises(SynapseError):
            parse_json_value_from_request(make_request(b''))
        result3 = parse_json_value_from_request(make_request(b''), allow_empty_body=True)
        self.assertIsNone(result3)
        with self.assertRaises(SynapseError):
            parse_json_value_from_request(make_request(b'\xff\x00'))
        with self.assertRaises(SynapseError):
            parse_json_value_from_request(make_request(b'foo'))
        with self.assertRaises(SynapseError):
            parse_json_value_from_request(make_request(b'{"foo": Infinity}'))

    def test_parse_json_object(self) -> None:
        if False:
            print('Hello World!')
        'Basic tests for parse_json_object_from_request.'
        result = parse_json_object_from_request(make_request(b''), allow_empty_body=True)
        self.assertEqual(result, {})
        with self.assertRaises(SynapseError):
            parse_json_object_from_request(make_request(b'["foo"]'))

class CancellableRestServlet(RestServlet):
    """A `RestServlet` with a mix of cancellable and uncancellable handlers."""
    PATTERNS = client_patterns('/sleep$')

    def __init__(self, hs: HomeServer):
        if False:
            print('Hello World!')
        super().__init__()
        self.clock = hs.get_clock()

    @cancellable
    async def on_GET(self, request: SynapseRequest) -> Tuple[int, JsonDict]:
        await self.clock.sleep(1.0)
        return (HTTPStatus.OK, {'result': True})

    async def on_POST(self, request: SynapseRequest) -> Tuple[int, JsonDict]:
        await self.clock.sleep(1.0)
        return (HTTPStatus.OK, {'result': True})

class TestRestServletCancellation(unittest.HomeserverTestCase):
    """Tests for `RestServlet` cancellation."""
    servlets = [lambda hs, http_server: CancellableRestServlet(hs).register(http_server)]

    def test_cancellable_disconnect(self) -> None:
        if False:
            return 10
        'Test that handlers with the `@cancellable` flag can be cancelled.'
        channel = self.make_request('GET', '/sleep', await_result=False)
        test_disconnect(self.reactor, channel, expect_cancellation=True, expected_body={'error': 'Request cancelled', 'errcode': Codes.UNKNOWN})

    def test_uncancellable_disconnect(self) -> None:
        if False:
            print('Hello World!')
        'Test that handlers without the `@cancellable` flag cannot be cancelled.'
        channel = self.make_request('POST', '/sleep', await_result=False)
        test_disconnect(self.reactor, channel, expect_cancellation=False, expected_body={'result': True})