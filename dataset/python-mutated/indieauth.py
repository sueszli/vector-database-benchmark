"""Helpers to resolve client ID/secret."""
from __future__ import annotations
import asyncio
from html.parser import HTMLParser
from ipaddress import ip_address
import logging
from urllib.parse import ParseResult, urljoin, urlparse
import aiohttp
import aiohttp.client_exceptions
from homeassistant.core import HomeAssistant
from homeassistant.util.network import is_local
_LOGGER = logging.getLogger(__name__)

async def verify_redirect_uri(hass: HomeAssistant, client_id: str, redirect_uri: str) -> bool:
    """Verify that the client and redirect uri match."""
    try:
        client_id_parts = _parse_client_id(client_id)
    except ValueError:
        return False
    redirect_parts = _parse_url(redirect_uri)
    is_valid = client_id_parts.scheme == redirect_parts.scheme and client_id_parts.netloc == redirect_parts.netloc
    if is_valid:
        return True
    if client_id == 'https://home-assistant.io/iOS' and redirect_uri == 'homeassistant://auth-callback':
        return True
    if client_id == 'https://home-assistant.io/android' and redirect_uri in ('homeassistant://auth-callback', 'https://wear.googleapis.com/3p_auth/io.homeassistant.companion.android', 'https://wear.googleapis-cn.com/3p_auth/io.homeassistant.companion.android'):
        return True
    redirect_uris = await fetch_redirect_uris(hass, client_id)
    return redirect_uri in redirect_uris

class LinkTagParser(HTMLParser):
    """Parser to find link tags."""

    def __init__(self, rel: str) -> None:
        if False:
            while True:
                i = 10
        'Initialize a link tag parser.'
        super().__init__()
        self.rel = rel
        self.found: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if False:
            print('Hello World!')
        'Handle finding a start tag.'
        if tag != 'link':
            return
        attributes: dict[str, str | None] = dict(attrs)
        if attributes.get('rel') == self.rel:
            self.found.append(attributes.get('href'))

async def fetch_redirect_uris(hass: HomeAssistant, url: str) -> list[str]:
    """Find link tag with redirect_uri values.

    IndieAuth 4.2.2

    The client SHOULD publish one or more <link> tags or Link HTTP headers with
    a rel attribute of redirect_uri at the client_id URL.

    We limit to the first 10kB of the page.

    We do not implement extracting redirect uris from headers.
    """
    parser = LinkTagParser('redirect_uri')
    chunks = 0
    try:
        async with aiohttp.ClientSession() as session, session.get(url, timeout=5) as resp:
            async for data in resp.content.iter_chunked(1024):
                parser.feed(data.decode())
                chunks += 1
                if chunks == 10:
                    break
    except asyncio.TimeoutError:
        _LOGGER.error('Timeout while looking up redirect_uri %s', url)
    except aiohttp.client_exceptions.ClientSSLError:
        _LOGGER.error('SSL error while looking up redirect_uri %s', url)
    except aiohttp.client_exceptions.ClientOSError as ex:
        _LOGGER.error('OS error while looking up redirect_uri %s: %s', url, ex.strerror)
    except aiohttp.client_exceptions.ClientConnectionError:
        _LOGGER.error('Low level connection error while looking up redirect_uri %s', url)
    except aiohttp.client_exceptions.ClientError:
        _LOGGER.error('Unknown error while looking up redirect_uri %s', url)
    return [urljoin(url, found) for found in parser.found]

def verify_client_id(client_id: str) -> bool:
    if False:
        i = 10
        return i + 15
    'Verify that the client id is valid.'
    try:
        _parse_client_id(client_id)
        return True
    except ValueError:
        return False

def _parse_url(url: str) -> ParseResult:
    if False:
        i = 10
        return i + 15
    'Parse a url in parts and canonicalize according to IndieAuth.'
    parts = urlparse(url)
    parts = parts._replace(netloc=parts.netloc.lower())
    if parts.path == '':
        parts = parts._replace(path='/')
    return parts

def _parse_client_id(client_id: str) -> ParseResult:
    if False:
        i = 10
        return i + 15
    'Test if client id is a valid URL according to IndieAuth section 3.2.\n\n    https://indieauth.spec.indieweb.org/#client-identifier\n    '
    parts = _parse_url(client_id)
    if parts.scheme not in ('http', 'https'):
        raise ValueError()
    if any((segment in ('.', '..') for segment in parts.path.split('/'))):
        raise ValueError('Client ID cannot contain single-dot or double-dot path segments')
    if parts.fragment != '':
        raise ValueError('Client ID cannot contain a fragment')
    if parts.username is not None:
        raise ValueError('Client ID cannot contain username')
    if parts.password is not None:
        raise ValueError('Client ID cannot contain password')
    try:
        parts.port
    except ValueError as ex:
        raise ValueError('Client ID contains invalid port') from ex
    address = None
    try:
        netloc = parts.netloc
        if netloc[0] == '[' and netloc[-1] == ']':
            netloc = netloc[1:-1]
        address = ip_address(netloc)
    except ValueError:
        pass
    if address is None or is_local(address):
        return parts
    raise ValueError('Hostname should be a domain name or local IP address')