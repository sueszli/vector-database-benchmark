"""The tests for the emulated Hue component."""
from http import HTTPStatus
import json
import unittest
from unittest.mock import patch
from aiohttp import web
import defusedxml.ElementTree as ET
import pytest
from homeassistant import setup
from homeassistant.components import emulated_hue
from homeassistant.components.emulated_hue import upnp
from homeassistant.const import CONTENT_TYPE_JSON
from homeassistant.core import HomeAssistant
from tests.common import get_test_instance_port
BRIDGE_SERVER_PORT = get_test_instance_port()

class MockTransport:
    """Mock asyncio transport."""

    def __init__(self):
        if False:
            return 10
        'Create a place to store the sends.'
        self.sends = []

    def sendto(self, response, addr):
        if False:
            print('Hello World!')
        'Mock sendto.'
        self.sends.append((response, addr))

@pytest.fixture
def aiohttp_client(event_loop, aiohttp_client, socket_enabled):
    if False:
        for i in range(10):
            print('nop')
    'Return aiohttp_client and allow opening sockets.'
    return aiohttp_client

@pytest.fixture
def hue_client(aiohttp_client):
    if False:
        for i in range(10):
            print('nop')
    'Return a hue API client.'
    app = web.Application()
    with unittest.mock.patch('homeassistant.components.emulated_hue.web.Application', return_value=app):

        async def client():
            """Return an authenticated client."""
            return await aiohttp_client(app)
        yield client

async def setup_hue(hass):
    """Set up the emulated_hue integration."""
    with patch('homeassistant.components.emulated_hue.async_create_upnp_datagram_endpoint'):
        assert await setup.async_setup_component(hass, emulated_hue.DOMAIN, {emulated_hue.DOMAIN: {emulated_hue.CONF_LISTEN_PORT: BRIDGE_SERVER_PORT}})
        await hass.async_block_till_done()

def test_upnp_discovery_basic() -> None:
    if False:
        return 10
    'Tests the UPnP basic discovery response.'
    upnp_responder_protocol = upnp.UPNPResponderProtocol(None, None, '192.0.2.42', 8080)
    mock_transport = MockTransport()
    upnp_responder_protocol.transport = mock_transport
    'Original request emitted by the Hue Bridge v1 app.'
    request = 'M-SEARCH * HTTP/1.1\nHOST:239.255.255.250:1900\nST:ssdp:all\nMan:"ssdp:discover"\nMX:3\n\n'
    encoded_request = request.replace('\n', '\r\n').encode('utf-8')
    upnp_responder_protocol.datagram_received(encoded_request, 1234)
    expected_response = 'HTTP/1.1 200 OK\nCACHE-CONTROL: max-age=60\nEXT:\nLOCATION: http://192.0.2.42:8080/description.xml\nSERVER: FreeRTOS/6.0.5, UPnP/1.0, IpBridge/1.16.0\nhue-bridgeid: 001788FFFE23BFC2\nST: urn:schemas-upnp-org:device:basic:1\nUSN: uuid:2f402f80-da50-11e1-9b23-001788255acc\n\n'
    expected_send = expected_response.replace('\n', '\r\n').encode('utf-8')
    assert mock_transport.sends == [(expected_send, 1234)]

def test_upnp_discovery_rootdevice() -> None:
    if False:
        print('Hello World!')
    'Tests the UPnP rootdevice discovery response.'
    upnp_responder_protocol = upnp.UPNPResponderProtocol(None, None, '192.0.2.42', 8080)
    mock_transport = MockTransport()
    upnp_responder_protocol.transport = mock_transport
    'Original request emitted by Busch-Jaeger free@home SysAP.'
    request = 'M-SEARCH * HTTP/1.1\nHOST: 239.255.255.250:1900\nMAN: "ssdp:discover"\nMX: 40\nST: upnp:rootdevice\n\n'
    encoded_request = request.replace('\n', '\r\n').encode('utf-8')
    upnp_responder_protocol.datagram_received(encoded_request, 1234)
    expected_response = 'HTTP/1.1 200 OK\nCACHE-CONTROL: max-age=60\nEXT:\nLOCATION: http://192.0.2.42:8080/description.xml\nSERVER: FreeRTOS/6.0.5, UPnP/1.0, IpBridge/1.16.0\nhue-bridgeid: 001788FFFE23BFC2\nST: upnp:rootdevice\nUSN: uuid:2f402f80-da50-11e1-9b23-001788255acc::upnp:rootdevice\n\n'
    expected_send = expected_response.replace('\n', '\r\n').encode('utf-8')
    assert mock_transport.sends == [(expected_send, 1234)]

def test_upnp_no_response() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Tests the UPnP does not response on an invalid request.'
    upnp_responder_protocol = upnp.UPNPResponderProtocol(None, None, '192.0.2.42', 8080)
    mock_transport = MockTransport()
    upnp_responder_protocol.transport = mock_transport
    'Original request emitted by the Hue Bridge v1 app.'
    request = 'INVALID * HTTP/1.1\nHOST:239.255.255.250:1900\nST:ssdp:all\nMan:"ssdp:discover"\nMX:3\n\n'
    encoded_request = request.replace('\n', '\r\n').encode('utf-8')
    upnp_responder_protocol.datagram_received(encoded_request, 1234)
    assert mock_transport.sends == []

async def test_description_xml(hass: HomeAssistant, hue_client) -> None:
    """Test the description."""
    await setup_hue(hass)
    client = await hue_client()
    result = await client.get('/description.xml', timeout=5)
    assert result.status == HTTPStatus.OK
    assert 'text/xml' in result.headers['content-type']
    try:
        root = ET.fromstring(await result.text())
        ns = {'s': 'urn:schemas-upnp-org:device-1-0'}
        assert root.find('./s:device/s:serialNumber', ns).text == '001788FFFE23BFC2'
    except Exception:
        pytest.fail('description.xml is not valid XML!')

async def test_create_username(hass: HomeAssistant, hue_client) -> None:
    """Test the creation of an username."""
    await setup_hue(hass)
    client = await hue_client()
    request_json = {'devicetype': 'my_device'}
    result = await client.post('/api', data=json.dumps(request_json), timeout=5)
    assert result.status == HTTPStatus.OK
    assert CONTENT_TYPE_JSON in result.headers['content-type']
    resp_json = await result.json()
    success_json = resp_json[0]
    assert 'success' in success_json
    assert 'username' in success_json['success']

async def test_unauthorized_view(hass: HomeAssistant, hue_client) -> None:
    """Test unauthorized view."""
    await setup_hue(hass)
    client = await hue_client()
    request_json = {'devicetype': 'my_device'}
    result = await client.get('/api/unauthorized', data=json.dumps(request_json), timeout=5)
    assert result.status == HTTPStatus.OK
    assert CONTENT_TYPE_JSON in result.headers['content-type']
    resp_json = await result.json()
    assert len(resp_json) == 1
    success_json = resp_json[0]
    assert len(success_json) == 1
    assert 'error' in success_json
    error_json = success_json['error']
    assert len(error_json) == 3
    assert '/' in error_json['address']
    assert 'unauthorized user' in error_json['description']
    assert '1' in error_json['type']

async def test_valid_username_request(hass: HomeAssistant, hue_client) -> None:
    """Test request with a valid username."""
    await setup_hue(hass)
    client = await hue_client()
    request_json = {'invalid_key': 'my_device'}
    result = await client.post('/api', data=json.dumps(request_json), timeout=5)
    assert result.status == HTTPStatus.BAD_REQUEST