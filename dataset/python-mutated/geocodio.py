import collections.abc
import json
from functools import partial
from urllib.parse import urlencode
from geopy.adapters import AdapterHTTPError
from geopy.exc import GeocoderQueryError, GeocoderQuotaExceeded
from geopy.geocoders.base import DEFAULT_SENTINEL, NONE_RESULT, Geocoder
from geopy.location import Location
from geopy.util import logger
__all__ = ('Geocodio',)

class Geocodio(Geocoder):
    """Geocoder using the Geocod.io API.

    Documentation at:
        https://www.geocod.io/docs/

    Pricing details:
        https://www.geocod.io/pricing/

    .. versionadded:: 2.2
    """
    structured_query_params = {'street', 'city', 'state', 'postal_code', 'country'}
    domain = 'api.geocod.io'
    geocode_path = '/v1.6/geocode'
    reverse_path = '/v1.6/reverse'

    def __init__(self, api_key, *, scheme=None, timeout=DEFAULT_SENTINEL, proxies=DEFAULT_SENTINEL, user_agent=None, ssl_context=DEFAULT_SENTINEL, adapter_factory=None, domain=None):
        if False:
            print('Hello World!')
        '\n        :param str api_key:\n            A valid Geocod.io API key. (https://dash.geocod.io/apikey/create)\n\n        :param str scheme:\n            See :attr:`geopy.geocoders.options.default_scheme`.\n\n        :param int timeout:\n            See :attr:`geopy.geocoders.options.default_timeout`.\n\n        :param dict proxies:\n            See :attr:`geopy.geocoders.options.default_proxies`.\n\n        :param str user_agent:\n            See :attr:`geopy.geocoders.options.default_user_agent`.\n\n        :type ssl_context: :class:`ssl.SSLContext`\n        :param ssl_context:\n            See :attr:`geopy.geocoders.options.default_ssl_context`.\n\n        :param callable adapter_factory:\n            See :attr:`geopy.geocoders.options.default_adapter_factory`.\n\n        :param str domain: base api domain\n\n            .. versionadded:: 2.4\n        '
        super().__init__(scheme=scheme, timeout=timeout, proxies=proxies, user_agent=user_agent, ssl_context=ssl_context, adapter_factory=adapter_factory)
        self.api_key = api_key
        if domain:
            self.domain = domain

    def geocode(self, query, *, limit=None, exactly_one=True, timeout=DEFAULT_SENTINEL):
        if False:
            while True:
                i = 10
        "\n        Return a location point by address.\n\n        :param query: The address, query or a structured query\n            you wish to geocode.\n\n            For a structured query, provide a dictionary whose keys\n            are one of: `street`, `city`, `state`, `postal_code` or `country`.\n        :type query: dict or str\n\n        :param int limit: The maximum number of matches to return. This will be reset\n            to 1 if ``exactly_one`` is ``True``.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder's initialization.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        "
        if isinstance(query, collections.abc.Mapping):
            params = {key: val for (key, val) in query.items() if key in self.structured_query_params}
        else:
            params = {'q': query}
        params['api_key'] = self.api_key
        if limit:
            params['limit'] = limit
        if exactly_one:
            params['limit'] = 1
        api = '%s://%s%s' % (self.scheme, self.domain, self.geocode_path)
        url = '?'.join((api, urlencode(params)))
        logger.debug('%s.geocode: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def reverse(self, query, *, exactly_one=True, timeout=DEFAULT_SENTINEL, limit=None):
        if False:
            print('Hello World!')
        "Return an address by location point.\n\n        :param str query: The coordinates for which you wish to obtain the\n            closest human-readable addresses\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder's initialization.\n\n        :param int limit: The maximum number of matches to return. This will be reset\n            to 1 if ``exactly_one`` is ``True``.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        "
        params = {'q': self._coerce_point_to_string(query), 'api_key': self.api_key}
        if exactly_one:
            limit = 1
        if limit is not None:
            params['limit'] = limit
        api = '%s://%s%s' % (self.scheme, self.domain, self.reverse_path)
        url = '?'.join((api, urlencode(params)))
        logger.debug('%s.reverse: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def _parse_json(self, page, exactly_one=True):
        if False:
            i = 10
            return i + 15
        'Returns location, (latitude, longitude) from json feed.'
        places = page.get('results', [])
        if not places:
            return None

        def parse_place(place):
            if False:
                for i in range(10):
                    print('nop')
            'Get the location, lat, lng from a single json place.'
            location = place.get('formatted_address')
            latitude = place['location']['lat']
            longitude = place['location']['lng']
            return Location(location, (latitude, longitude), place)
        if exactly_one:
            return parse_place(places[0])
        else:
            return [parse_place(place) for place in places]

    def _geocoder_exception_handler(self, error):
        if False:
            i = 10
            return i + 15
        'Custom exception handling for invalid queries and exceeded quotas.\n\n        Geocod.io returns a ``422`` status code for invalid queries, which is not mapped\n        in :const:`~geopy.geocoders.base.ERROR_CODE_MAP`. The service also returns a\n        ``403`` status code for exceeded quotas instead of the ``429`` code mapped in\n        :const:`~geopy.geocoders.base.ERROR_CODE_MAP`\n        '
        if not isinstance(error, AdapterHTTPError):
            return
        if error.status_code is None or error.text is None:
            return
        if error.status_code == 422:
            error_message = self._get_error_message(error)
            if 'could not geocode address' in error_message.lower() and 'postal code or city required' in error_message.lower():
                return NONE_RESULT
            raise GeocoderQueryError(error_message) from error
        if error.status_code == 403:
            error_message = self._get_error_message(error)
            quota_exceeded_snippet = "You can't make this request as it is above your daily maximum."
            if quota_exceeded_snippet in error_message:
                raise GeocoderQuotaExceeded(error_message) from error

    def _get_error_message(self, error):
        if False:
            return 10
        "Try to extract an error message from the 'error' property of a JSON response.\n        "
        try:
            error_message = json.loads(error.text).get('error')
        except ValueError:
            error_message = None
        return error_message or error.text