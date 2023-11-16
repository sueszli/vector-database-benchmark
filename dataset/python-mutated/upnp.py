"""Support UPNP discovery method that mimics Hue hubs."""
from __future__ import annotations
import asyncio
from contextlib import suppress
import logging
import socket
from typing import cast
from aiohttp import web
from homeassistant import core
from homeassistant.components.http import HomeAssistantView
from .config import Config
from .const import HUE_SERIAL_NUMBER, HUE_UUID
_LOGGER = logging.getLogger(__name__)
BROADCAST_PORT = 1900
BROADCAST_ADDR = '239.255.255.250'

class DescriptionXmlView(HomeAssistantView):
    """Handles requests for the description.xml file."""
    url = '/description.xml'
    name = 'description:xml'
    requires_auth = False

    def __init__(self, config: Config) -> None:
        if False:
            return 10
        'Initialize the instance of the view.'
        self.config = config

    @core.callback
    def get(self, request: web.Request) -> web.Response:
        if False:
            while True:
                i = 10
        'Handle a GET request.'
        resp_text = f'<?xml version="1.0" encoding="UTF-8" ?>\n<root xmlns="urn:schemas-upnp-org:device-1-0">\n<specVersion>\n<major>1</major>\n<minor>0</minor>\n</specVersion>\n<URLBase>http://{self.config.advertise_ip}:{self.config.advertise_port}/</URLBase>\n<device>\n<deviceType>urn:schemas-upnp-org:device:Basic:1</deviceType>\n<friendlyName>Home Assistant Bridge ({self.config.advertise_ip})</friendlyName>\n<manufacturer>Royal Philips Electronics</manufacturer>\n<manufacturerURL>http://www.philips.com</manufacturerURL>\n<modelDescription>Philips hue Personal Wireless Lighting</modelDescription>\n<modelName>Philips hue bridge 2015</modelName>\n<modelNumber>BSB002</modelNumber>\n<modelURL>http://www.meethue.com</modelURL>\n<serialNumber>{HUE_SERIAL_NUMBER}</serialNumber>\n<UDN>uuid:{HUE_UUID}</UDN>\n</device>\n</root>\n'
        return web.Response(text=resp_text, content_type='text/xml')

class UPNPResponderProtocol(asyncio.Protocol):
    """Handle responding to UPNP/SSDP discovery requests."""

    def __init__(self, loop: asyncio.AbstractEventLoop, ssdp_socket: socket.socket, advertise_ip: str, advertise_port: int) -> None:
        if False:
            while True:
                i = 10
        'Initialize the class.'
        self.transport: asyncio.DatagramTransport | None = None
        self._loop = loop
        self._sock = ssdp_socket
        self.advertise_ip = advertise_ip
        self.advertise_port = advertise_port
        self._upnp_root_response = self._prepare_response('upnp:rootdevice', f'uuid:{HUE_UUID}::upnp:rootdevice')
        self._upnp_device_response = self._prepare_response('urn:schemas-upnp-org:device:basic:1', f'uuid:{HUE_UUID}')

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        if False:
            return 10
        'Set the transport.'
        self.transport = cast(asyncio.DatagramTransport, transport)

    def connection_lost(self, exc: Exception | None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Handle connection lost.'

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        if False:
            print('Hello World!')
        'Respond to msearch packets.'
        decoded_data = data.decode('utf-8', errors='ignore')
        if 'M-SEARCH' not in decoded_data:
            return
        _LOGGER.debug('UPNP Responder M-SEARCH method received: %s', data)
        response = self._handle_request(decoded_data)
        _LOGGER.debug('UPNP Responder responding with: %s', response)
        assert self.transport is not None
        self.transport.sendto(response, addr)

    def error_received(self, exc: Exception) -> None:
        if False:
            print('Hello World!')
        'Log UPNP errors.'
        _LOGGER.error('UPNP Error received: %s', exc)

    def close(self) -> None:
        if False:
            while True:
                i = 10
        'Stop the server.'
        _LOGGER.info('UPNP responder shutting down')
        if self.transport:
            self.transport.close()
        self._loop.remove_writer(self._sock.fileno())
        self._loop.remove_reader(self._sock.fileno())
        self._sock.close()

    def _handle_request(self, decoded_data: str) -> bytes:
        if False:
            i = 10
            return i + 15
        if 'upnp:rootdevice' in decoded_data:
            return self._upnp_root_response
        return self._upnp_device_response

    def _prepare_response(self, search_target: str, unique_service_name: str) -> bytes:
        if False:
            while True:
                i = 10
        response = f'HTTP/1.1 200 OK\nCACHE-CONTROL: max-age=60\nEXT:\nLOCATION: http://{self.advertise_ip}:{self.advertise_port}/description.xml\nSERVER: FreeRTOS/6.0.5, UPnP/1.0, IpBridge/1.16.0\nhue-bridgeid: {HUE_SERIAL_NUMBER}\nST: {search_target}\nUSN: {unique_service_name}\n\n'
        return response.replace('\n', '\r\n').encode('utf-8')

async def async_create_upnp_datagram_endpoint(host_ip_addr: str, upnp_bind_multicast: bool, advertise_ip: str, advertise_port: int) -> UPNPResponderProtocol:
    """Create the UPNP socket and protocol."""
    ssdp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ssdp_socket.setblocking(False)
    ssdp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    with suppress(AttributeError):
        ssdp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    ssdp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    ssdp_socket.setsockopt(socket.SOL_IP, socket.IP_MULTICAST_IF, socket.inet_aton(host_ip_addr))
    ssdp_socket.setsockopt(socket.SOL_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(BROADCAST_ADDR) + socket.inet_aton(host_ip_addr))
    ssdp_socket.bind(('' if upnp_bind_multicast else host_ip_addr, BROADCAST_PORT))
    loop = asyncio.get_event_loop()
    transport_protocol = await loop.create_datagram_endpoint(lambda : UPNPResponderProtocol(loop, ssdp_socket, advertise_ip, advertise_port), sock=ssdp_socket)
    return transport_protocol[1]