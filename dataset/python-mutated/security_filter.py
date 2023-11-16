"""Middleware to add some basic security filtering to requests."""
from __future__ import annotations
from collections.abc import Awaitable, Callable
import logging
import re
from typing import Final
from urllib.parse import unquote
from aiohttp.web import Application, HTTPBadRequest, Request, StreamResponse, middleware
from homeassistant.core import callback
_LOGGER = logging.getLogger(__name__)
FILTERS: Final = re.compile('(?:proc/self/environ|(<|%3C).*script.*(>|%3E)|(\\.\\.//?)+|[a-zA-Z0-9_]=/([a-z0-9_.]//?)+|union.*select.*\\(|union.*all.*select.*|concat.*\\()', flags=re.IGNORECASE)
UNSAFE_URL_BYTES = ['\t', '\r', '\n']

@callback
def setup_security_filter(app: Application) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Create security filter middleware for the app.'

    def _recursive_unquote(value: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Handle values that are encoded multiple times.'
        if (unquoted := unquote(value)) != value:
            unquoted = _recursive_unquote(unquoted)
        return unquoted

    @middleware
    async def security_filter_middleware(request: Request, handler: Callable[[Request], Awaitable[StreamResponse]]) -> StreamResponse:
        """Process request and block commonly known exploit attempts."""
        for unsafe_byte in UNSAFE_URL_BYTES:
            if unsafe_byte in request.path:
                _LOGGER.warning('Filtered a request with an unsafe byte in path: %s', request.raw_path)
                raise HTTPBadRequest
            if unsafe_byte in request.query_string:
                _LOGGER.warning('Filtered a request with unsafe byte query string: %s', request.raw_path)
                raise HTTPBadRequest
        if FILTERS.search(_recursive_unquote(request.path)):
            _LOGGER.warning('Filtered a potential harmful request to: %s', request.raw_path)
            raise HTTPBadRequest
        if FILTERS.search(_recursive_unquote(request.query_string)):
            _LOGGER.warning('Filtered a request with a potential harmful query string: %s', request.raw_path)
            raise HTTPBadRequest
        return await handler(request)
    app.middlewares.append(security_filter_middleware)