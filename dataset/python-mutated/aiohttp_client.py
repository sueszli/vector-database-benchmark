"""Helper for aiohttp webclient stuff."""
from __future__ import annotations
import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from ssl import SSLContext
import sys
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
import aiohttp
from aiohttp import web
from aiohttp.hdrs import CONTENT_TYPE, USER_AGENT
from aiohttp.web_exceptions import HTTPBadGateway, HTTPGatewayTimeout
from homeassistant import config_entries
from homeassistant.const import APPLICATION_NAME, EVENT_HOMEASSISTANT_CLOSE, __version__
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.loader import bind_hass
from homeassistant.util import ssl as ssl_util
from homeassistant.util.json import json_loads
from .frame import warn_use
from .json import json_dumps
if TYPE_CHECKING:
    from aiohttp.typedefs import JSONDecoder
DATA_CONNECTOR = 'aiohttp_connector'
DATA_CLIENTSESSION = 'aiohttp_clientsession'
SERVER_SOFTWARE = '{0}/{1} aiohttp/{2} Python/{3[0]}.{3[1]}'.format(APPLICATION_NAME, __version__, aiohttp.__version__, sys.version_info)
ENABLE_CLEANUP_CLOSED = not (3, 11, 1) <= sys.version_info < (3, 11, 4)
WARN_CLOSE_MSG = 'closes the Home Assistant aiohttp session'
MAXIMUM_CONNECTIONS = 4096
MAXIMUM_CONNECTIONS_PER_HOST = 100

class HassClientResponse(aiohttp.ClientResponse):
    """aiohttp.ClientResponse with a json method that uses json_loads by default."""

    async def json(self, *args: Any, loads: JSONDecoder=json_loads, **kwargs: Any) -> Any:
        """Send a json request and parse the json response."""
        return await super().json(*args, loads=loads, **kwargs)

@callback
@bind_hass
def async_get_clientsession(hass: HomeAssistant, verify_ssl: bool=True, family: int=0) -> aiohttp.ClientSession:
    if False:
        print('Hello World!')
    'Return default aiohttp ClientSession.\n\n    This method must be run in the event loop.\n    '
    session_key = _make_key(verify_ssl, family)
    if DATA_CLIENTSESSION not in hass.data:
        sessions: dict[tuple[bool, int], aiohttp.ClientSession] = {}
        hass.data[DATA_CLIENTSESSION] = sessions
    else:
        sessions = hass.data[DATA_CLIENTSESSION]
    if session_key not in sessions:
        session = _async_create_clientsession(hass, verify_ssl, auto_cleanup_method=_async_register_default_clientsession_shutdown, family=family)
        sessions[session_key] = session
    else:
        session = sessions[session_key]
    return session

@callback
@bind_hass
def async_create_clientsession(hass: HomeAssistant, verify_ssl: bool=True, auto_cleanup: bool=True, family: int=0, **kwargs: Any) -> aiohttp.ClientSession:
    if False:
        return 10
    'Create a new ClientSession with kwargs, i.e. for cookies.\n\n    If auto_cleanup is False, you need to call detach() after the session\n    returned is no longer used. Default is True, the session will be\n    automatically detached on homeassistant_stop or when being created\n    in config entry setup, the config entry is unloaded.\n\n    This method must be run in the event loop.\n    '
    auto_cleanup_method = None
    if auto_cleanup:
        auto_cleanup_method = _async_register_clientsession_shutdown
    clientsession = _async_create_clientsession(hass, verify_ssl, auto_cleanup_method=auto_cleanup_method, family=family, **kwargs)
    return clientsession

@callback
def _async_create_clientsession(hass: HomeAssistant, verify_ssl: bool=True, auto_cleanup_method: Callable[[HomeAssistant, aiohttp.ClientSession], None] | None=None, family: int=0, **kwargs: Any) -> aiohttp.ClientSession:
    if False:
        while True:
            i = 10
    'Create a new ClientSession with kwargs, i.e. for cookies.'
    clientsession = aiohttp.ClientSession(connector=_async_get_connector(hass, verify_ssl, family), json_serialize=json_dumps, response_class=HassClientResponse, **kwargs)
    clientsession._default_headers = MappingProxyType({USER_AGENT: SERVER_SOFTWARE})
    clientsession.close = warn_use(clientsession.close, WARN_CLOSE_MSG)
    if auto_cleanup_method:
        auto_cleanup_method(hass, clientsession)
    return clientsession

@bind_hass
async def async_aiohttp_proxy_web(hass: HomeAssistant, request: web.BaseRequest, web_coro: Awaitable[aiohttp.ClientResponse], buffer_size: int=102400, timeout: int=10) -> web.StreamResponse | None:
    """Stream websession request to aiohttp web response."""
    try:
        async with asyncio.timeout(timeout):
            req = await web_coro
    except asyncio.CancelledError:
        return None
    except asyncio.TimeoutError as err:
        raise HTTPGatewayTimeout() from err
    except aiohttp.ClientError as err:
        raise HTTPBadGateway() from err
    try:
        return await async_aiohttp_proxy_stream(hass, request, req.content, req.headers.get(CONTENT_TYPE))
    finally:
        req.close()

@bind_hass
async def async_aiohttp_proxy_stream(hass: HomeAssistant, request: web.BaseRequest, stream: aiohttp.StreamReader, content_type: str | None, buffer_size: int=102400, timeout: int=10) -> web.StreamResponse:
    """Stream a stream to aiohttp web response."""
    response = web.StreamResponse()
    if content_type is not None:
        response.content_type = content_type
    await response.prepare(request)
    with suppress(asyncio.TimeoutError, aiohttp.ClientError):
        while hass.is_running:
            async with asyncio.timeout(timeout):
                data = await stream.read(buffer_size)
            if not data:
                break
            await response.write(data)
    return response

@callback
def _async_register_clientsession_shutdown(hass: HomeAssistant, clientsession: aiohttp.ClientSession) -> None:
    if False:
        i = 10
        return i + 15
    'Register ClientSession close on Home Assistant shutdown or config entry unload.\n\n    This method must be run in the event loop.\n    '

    @callback
    def _async_close_websession(*_: Any) -> None:
        if False:
            print('Hello World!')
        'Close websession.'
        clientsession.detach()
    unsub = hass.bus.async_listen_once(EVENT_HOMEASSISTANT_CLOSE, _async_close_websession)
    if not (config_entry := config_entries.current_entry.get()):
        return
    config_entry.async_on_unload(unsub)
    config_entry.async_on_unload(_async_close_websession)

@callback
def _async_register_default_clientsession_shutdown(hass: HomeAssistant, clientsession: aiohttp.ClientSession) -> None:
    if False:
        i = 10
        return i + 15
    'Register default ClientSession close on Home Assistant shutdown.\n\n    This method must be run in the event loop.\n    '

    @callback
    def _async_close_websession(event: Event) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Close websession.'
        clientsession.detach()
    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_CLOSE, _async_close_websession)

@callback
def _make_key(verify_ssl: bool=True, family: int=0) -> tuple[bool, int]:
    if False:
        for i in range(10):
            print('nop')
    'Make a key for connector or session pool.'
    return (verify_ssl, family)

@callback
def _async_get_connector(hass: HomeAssistant, verify_ssl: bool=True, family: int=0) -> aiohttp.BaseConnector:
    if False:
        return 10
    'Return the connector pool for aiohttp.\n\n    This method must be run in the event loop.\n    '
    connector_key = _make_key(verify_ssl, family)
    if DATA_CONNECTOR not in hass.data:
        connectors: dict[tuple[bool, int], aiohttp.BaseConnector] = {}
        hass.data[DATA_CONNECTOR] = connectors
    else:
        connectors = hass.data[DATA_CONNECTOR]
    if connector_key in connectors:
        return connectors[connector_key]
    if verify_ssl:
        ssl_context: bool | SSLContext = ssl_util.get_default_context()
    else:
        ssl_context = ssl_util.get_default_no_verify_context()
    connector = aiohttp.TCPConnector(family=family, enable_cleanup_closed=ENABLE_CLEANUP_CLOSED, ssl=ssl_context, limit=MAXIMUM_CONNECTIONS, limit_per_host=MAXIMUM_CONNECTIONS_PER_HOST)
    connectors[connector_key] = connector

    async def _async_close_connector(event: Event) -> None:
        """Close connector pool."""
        await connector.close()
    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_CLOSE, _async_close_connector)
    return connector