from functools import partial
from urllib.parse import urlencode
from geopy.adapters import AdapterHTTPError
from geopy.exc import GeocoderQuotaExceeded
from geopy.geocoders.base import DEFAULT_SENTINEL, Geocoder
from geopy.location import Location
from geopy.util import logger
__all__ = ('LiveAddress',)

class LiveAddress(Geocoder):
    """Geocoder using the LiveAddress API provided by SmartyStreets.

    Documentation at:
        https://smartystreets.com/docs/cloud/us-street-api
    """
    geocode_path = '/street-address'

    def __init__(self, auth_id, auth_token, *, timeout=DEFAULT_SENTINEL, proxies=DEFAULT_SENTINEL, user_agent=None, ssl_context=DEFAULT_SENTINEL, adapter_factory=None):
        if False:
            while True:
                i = 10
        '\n\n        :param str auth_id: Valid `Auth ID` from SmartyStreets.\n\n        :param str auth_token: Valid `Auth Token` from SmartyStreets.\n\n        :param int timeout:\n            See :attr:`geopy.geocoders.options.default_timeout`.\n\n        :param dict proxies:\n            See :attr:`geopy.geocoders.options.default_proxies`.\n\n        :param str user_agent:\n            See :attr:`geopy.geocoders.options.default_user_agent`.\n\n        :type ssl_context: :class:`ssl.SSLContext`\n        :param ssl_context:\n            See :attr:`geopy.geocoders.options.default_ssl_context`.\n\n        :param callable adapter_factory:\n            See :attr:`geopy.geocoders.options.default_adapter_factory`.\n\n            .. versionadded:: 2.0\n        '
        super().__init__(scheme='https', timeout=timeout, proxies=proxies, user_agent=user_agent, ssl_context=ssl_context, adapter_factory=adapter_factory)
        self.auth_id = auth_id
        self.auth_token = auth_token
        domain = 'api.smartystreets.com'
        self.api = '%s://%s%s' % (self.scheme, domain, self.geocode_path)

    def geocode(self, query, *, exactly_one=True, timeout=DEFAULT_SENTINEL, candidates=1):
        if False:
            while True:
                i = 10
        "\n        Return a location point by address.\n\n        :param str query: The address or query you wish to geocode.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder's initialization.\n\n        :param int candidates: An integer between 1 and 10 indicating the max\n            number of candidate addresses to return if a valid address\n            could be found.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        "
        if not 1 <= candidates <= 10:
            raise ValueError('candidates must be between 1 and 10')
        query = {'auth-id': self.auth_id, 'auth-token': self.auth_token, 'street': query, 'candidates': candidates}
        url = '{url}?{query}'.format(url=self.api, query=urlencode(query))
        logger.debug('%s.geocode: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def _geocoder_exception_handler(self, error):
        if False:
            i = 10
            return i + 15
        search = 'no active subscriptions found'
        if isinstance(error, AdapterHTTPError):
            if search in str(error).lower():
                raise GeocoderQuotaExceeded(str(error)) from error
            if search in (error.text or '').lower():
                raise GeocoderQuotaExceeded(error.text) from error

    def _parse_json(self, response, exactly_one=True):
        if False:
            print('Hello World!')
        '\n        Parse responses as JSON objects.\n        '
        if not len(response):
            return None
        if exactly_one:
            return self._format_structured_address(response[0])
        else:
            return [self._format_structured_address(c) for c in response]

    def _format_structured_address(self, address):
        if False:
            i = 10
            return i + 15
        '\n        Pretty-print address and return lat, lon tuple.\n        '
        latitude = address['metadata'].get('latitude')
        longitude = address['metadata'].get('longitude')
        return Location(', '.join((address['delivery_line_1'], address['last_line'])), (latitude, longitude) if latitude and longitude else None, address)