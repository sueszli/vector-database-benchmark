import logging
import random
import time
from io import BytesIO
from typing import Callable, Dict, Optional, Tuple
import attr
from twisted.internet import defer
from twisted.internet.interfaces import IReactorTime
from twisted.web.client import RedirectAgent
from twisted.web.http import stringToDatetime
from twisted.web.http_headers import Headers
from twisted.web.iweb import IAgent, IResponse
from synapse.http.client import BodyExceededMaxSize, read_body_with_max_size
from synapse.logging.context import make_deferred_yieldable
from synapse.util import Clock, json_decoder
from synapse.util.caches.ttlcache import TTLCache
from synapse.util.metrics import Measure
WELL_KNOWN_DEFAULT_CACHE_PERIOD = 24 * 3600
WELL_KNOWN_DEFAULT_CACHE_PERIOD_JITTER = 0.1
WELL_KNOWN_INVALID_CACHE_PERIOD = 1 * 3600
WELL_KNOWN_DOWN_CACHE_PERIOD = 2 * 60
WELL_KNOWN_REMEMBER_DOMAIN_HAD_VALID = 2 * 3600
WELL_KNOWN_MAX_CACHE_PERIOD = 48 * 3600
WELL_KNOWN_MIN_CACHE_PERIOD = 5 * 60
WELL_KNOWN_MAX_SIZE = 50 * 1024
WELL_KNOWN_GRACE_PERIOD_FACTOR = 0.2
WELL_KNOWN_RETRY_ATTEMPTS = 3
logger = logging.getLogger(__name__)
_well_known_cache: TTLCache[bytes, Optional[bytes]] = TTLCache('well-known')
_had_valid_well_known_cache: TTLCache[bytes, bool] = TTLCache('had-valid-well-known')

@attr.s(slots=True, frozen=True, auto_attribs=True)
class WellKnownLookupResult:
    delegated_server: Optional[bytes]

class WellKnownResolver:
    """Handles well-known lookups for matrix servers."""

    def __init__(self, reactor: IReactorTime, agent: IAgent, user_agent: bytes, well_known_cache: Optional[TTLCache[bytes, Optional[bytes]]]=None, had_well_known_cache: Optional[TTLCache[bytes, bool]]=None):
        if False:
            return 10
        self._reactor = reactor
        self._clock = Clock(reactor)
        if well_known_cache is None:
            well_known_cache = _well_known_cache
        if had_well_known_cache is None:
            had_well_known_cache = _had_valid_well_known_cache
        self._well_known_cache = well_known_cache
        self._had_valid_well_known_cache = had_well_known_cache
        self._well_known_agent = RedirectAgent(agent)
        self.user_agent = user_agent

    async def get_well_known(self, server_name: bytes) -> WellKnownLookupResult:
        """Attempt to fetch and parse a .well-known file for the given server

        Args:
            server_name: name of the server, from the requested url

        Returns:
            The result of the lookup
        """
        try:
            (prev_result, expiry, ttl) = self._well_known_cache.get_with_expiry(server_name)
            now = self._clock.time()
            if now < expiry - WELL_KNOWN_GRACE_PERIOD_FACTOR * ttl:
                return WellKnownLookupResult(delegated_server=prev_result)
        except KeyError:
            prev_result = None
        try:
            with Measure(self._clock, 'get_well_known'):
                result: Optional[bytes]
                cache_period: float
                (result, cache_period) = await self._fetch_well_known(server_name)
        except _FetchWellKnownFailure as e:
            if prev_result and e.temporary:
                return WellKnownLookupResult(delegated_server=prev_result)
            result = None
            if self._had_valid_well_known_cache.get(server_name, False):
                cache_period = WELL_KNOWN_DOWN_CACHE_PERIOD
            else:
                cache_period = WELL_KNOWN_INVALID_CACHE_PERIOD
            cache_period *= random.uniform(1 - WELL_KNOWN_DEFAULT_CACHE_PERIOD_JITTER, 1 + WELL_KNOWN_DEFAULT_CACHE_PERIOD_JITTER)
        if cache_period > 0:
            self._well_known_cache.set(server_name, result, cache_period)
        return WellKnownLookupResult(delegated_server=result)

    async def _fetch_well_known(self, server_name: bytes) -> Tuple[bytes, float]:
        """Actually fetch and parse a .well-known, without checking the cache

        Args:
            server_name: name of the server, from the requested url

        Raises:
            _FetchWellKnownFailure if we fail to lookup a result

        Returns:
            The lookup result and cache period.
        """
        had_valid_well_known = self._had_valid_well_known_cache.get(server_name, False)
        (response, body) = await self._make_well_known_request(server_name, retry=had_valid_well_known)
        try:
            if response.code != 200:
                raise Exception('Non-200 response %s' % (response.code,))
            parsed_body = json_decoder.decode(body.decode('utf-8'))
            logger.info('Response from .well-known: %s', parsed_body)
            result = parsed_body['m.server'].encode('ascii')
        except defer.CancelledError:
            raise
        except Exception as e:
            logger.info('Error parsing well-known for %s: %s', server_name, e)
            raise _FetchWellKnownFailure(temporary=False)
        cache_period = _cache_period_from_headers(response.headers, time_now=self._reactor.seconds)
        if cache_period is None:
            cache_period = WELL_KNOWN_DEFAULT_CACHE_PERIOD
            cache_period *= random.uniform(1 - WELL_KNOWN_DEFAULT_CACHE_PERIOD_JITTER, 1 + WELL_KNOWN_DEFAULT_CACHE_PERIOD_JITTER)
        else:
            cache_period = min(cache_period, WELL_KNOWN_MAX_CACHE_PERIOD)
            cache_period = max(cache_period, WELL_KNOWN_MIN_CACHE_PERIOD)
        self._had_valid_well_known_cache.set(server_name, bool(result), cache_period + WELL_KNOWN_REMEMBER_DOMAIN_HAD_VALID)
        return (result, cache_period)

    async def _make_well_known_request(self, server_name: bytes, retry: bool) -> Tuple[IResponse, bytes]:
        """Make the well known request.

        This will retry the request if requested and it fails (with unable
        to connect or receives a 5xx error).

        Args:
            server_name: name of the server, from the requested url
            retry: Whether to retry the request if it fails.

        Raises:
            _FetchWellKnownFailure if we fail to lookup a result

        Returns:
            Returns the response object and body. Response may be a non-200 response.
        """
        uri = b'https://%s/.well-known/matrix/server' % (server_name,)
        uri_str = uri.decode('ascii')
        headers = {b'User-Agent': [self.user_agent]}
        i = 0
        while True:
            i += 1
            logger.info('Fetching %s', uri_str)
            try:
                response = await make_deferred_yieldable(self._well_known_agent.request(b'GET', uri, headers=Headers(headers)))
                body_stream = BytesIO()
                await make_deferred_yieldable(read_body_with_max_size(response, body_stream, WELL_KNOWN_MAX_SIZE))
                body = body_stream.getvalue()
                if 500 <= response.code < 600:
                    raise Exception('Non-200 response %s' % (response.code,))
                return (response, body)
            except defer.CancelledError:
                raise
            except BodyExceededMaxSize:
                logger.warning('Requested .well-known file for %s is too large > %r bytes', server_name.decode('ascii'), WELL_KNOWN_MAX_SIZE)
                raise _FetchWellKnownFailure(temporary=True)
            except Exception as e:
                if not retry or i >= WELL_KNOWN_RETRY_ATTEMPTS:
                    logger.info('Error fetching %s: %s', uri_str, e)
                    raise _FetchWellKnownFailure(temporary=True)
                logger.info('Error fetching %s: %s. Retrying', uri_str, e)
            await self._clock.sleep(0.5)

def _cache_period_from_headers(headers: Headers, time_now: Callable[[], float]=time.time) -> Optional[float]:
    if False:
        while True:
            i = 10
    cache_controls = _parse_cache_control(headers)
    if b'no-store' in cache_controls:
        return 0
    if b'max-age' in cache_controls:
        max_age = cache_controls[b'max-age']
        if max_age:
            try:
                return int(max_age)
            except ValueError:
                pass
    expires = headers.getRawHeaders(b'expires')
    if expires is not None:
        try:
            expires_date = stringToDatetime(expires[-1])
            return expires_date - time_now()
        except ValueError:
            return 0
    return None

def _parse_cache_control(headers: Headers) -> Dict[bytes, Optional[bytes]]:
    if False:
        print('Hello World!')
    cache_controls = {}
    cache_control_headers = headers.getRawHeaders(b'cache-control') or []
    for hdr in cache_control_headers:
        for directive in hdr.split(b','):
            splits = [x.strip() for x in directive.split(b'=', 1)]
            k = splits[0].lower()
            v = splits[1] if len(splits) > 1 else None
            cache_controls[k] = v
    return cache_controls

@attr.s(slots=True)
class _FetchWellKnownFailure(Exception):
    temporary: bool = attr.ib()