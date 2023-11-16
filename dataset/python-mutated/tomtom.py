from functools import partial
from urllib.parse import quote, urlencode
from geopy.adapters import AdapterHTTPError
from geopy.exc import GeocoderQuotaExceeded
from geopy.geocoders.base import DEFAULT_SENTINEL, Geocoder
from geopy.location import Location
from geopy.util import logger
__all__ = ('TomTom',)

class TomTom(Geocoder):
    """TomTom geocoder.

    Documentation at:
        https://developer.tomtom.com/search-api/search-api-documentation
    """
    geocode_path = '/search/2/geocode/%(query)s.json'
    reverse_path = '/search/2/reverseGeocode/%(position)s.json'

    def __init__(self, api_key, *, scheme=None, timeout=DEFAULT_SENTINEL, proxies=DEFAULT_SENTINEL, user_agent=None, ssl_context=DEFAULT_SENTINEL, adapter_factory=None, domain='api.tomtom.com'):
        if False:
            return 10
        '\n        :param str api_key: TomTom API key.\n\n        :param str scheme:\n            See :attr:`geopy.geocoders.options.default_scheme`.\n\n        :param int timeout:\n            See :attr:`geopy.geocoders.options.default_timeout`.\n\n        :param dict proxies:\n            See :attr:`geopy.geocoders.options.default_proxies`.\n\n        :param str user_agent:\n            See :attr:`geopy.geocoders.options.default_user_agent`.\n\n        :type ssl_context: :class:`ssl.SSLContext`\n        :param ssl_context:\n            See :attr:`geopy.geocoders.options.default_ssl_context`.\n\n        :param callable adapter_factory:\n            See :attr:`geopy.geocoders.options.default_adapter_factory`.\n\n            .. versionadded:: 2.0\n\n        :param str domain: Domain where the target TomTom service\n            is hosted.\n        '
        super().__init__(scheme=scheme, timeout=timeout, proxies=proxies, user_agent=user_agent, ssl_context=ssl_context, adapter_factory=adapter_factory)
        self.api_key = api_key
        self.api = '%s://%s%s' % (self.scheme, domain, self.geocode_path)
        self.api_reverse = '%s://%s%s' % (self.scheme, domain, self.reverse_path)

    def geocode(self, query, *, exactly_one=True, timeout=DEFAULT_SENTINEL, limit=None, typeahead=False, language=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a location point by address.\n\n        :param str query: The address or query you wish to geocode.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder\'s initialization.\n\n        :param int limit: Maximum amount of results to return from the service.\n            Unless exactly_one is set to False, limit will always be 1.\n\n        :param bool typeahead: If the "typeahead" flag is set, the query\n            will be interpreted as a partial input and the search will\n            enter predictive mode.\n\n        :param str language: Language in which search results should be\n            returned. When data in specified language is not\n            available for a specific field, default language is used.\n            List of supported languages (case-insensitive):\n            https://developer.tomtom.com/online-search/online-search-documentation/supported-languages\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        '
        params = self._geocode_params(query)
        params['typeahead'] = self._boolean_value(typeahead)
        if limit:
            params['limit'] = str(int(limit))
        if exactly_one:
            params['limit'] = '1'
        if language:
            params['language'] = language
        quoted_query = quote(query.encode('utf-8'))
        url = '?'.join((self.api % dict(query=quoted_query), urlencode(params)))
        logger.debug('%s.geocode: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def reverse(self, query, *, exactly_one=True, timeout=DEFAULT_SENTINEL, language=None):
        if False:
            while True:
                i = 10
        '\n        Return an address by location point.\n\n        :param query: The coordinates for which you wish to obtain the\n            closest human-readable addresses.\n        :type query: :class:`geopy.point.Point`, list or tuple of ``(latitude,\n            longitude)``, or string as ``"%(latitude)s, %(longitude)s"``.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder\'s initialization.\n\n        :param str language: Language in which search results should be\n            returned. When data in specified language is not\n            available for a specific field, default language is used.\n            List of supported languages (case-insensitive):\n            https://developer.tomtom.com/online-search/online-search-documentation/supported-languages\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        '
        position = self._coerce_point_to_string(query)
        params = self._reverse_params(position)
        if language:
            params['language'] = language
        quoted_position = quote(position.encode('utf-8'))
        url = '?'.join((self.api_reverse % dict(position=quoted_position), urlencode(params)))
        logger.debug('%s.reverse: %s', self.__class__.__name__, url)
        callback = partial(self._parse_reverse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def _boolean_value(self, bool_value):
        if False:
            for i in range(10):
                print('nop')
        return 'true' if bool_value else 'false'

    def _geocode_params(self, formatted_query):
        if False:
            print('Hello World!')
        return {'key': self.api_key}

    def _reverse_params(self, position):
        if False:
            return 10
        return {'key': self.api_key}

    def _parse_json(self, resources, exactly_one):
        if False:
            while True:
                i = 10
        if not resources or not resources['results']:
            return None
        if exactly_one:
            return self._parse_search_result(resources['results'][0])
        else:
            return [self._parse_search_result(result) for result in resources['results']]

    def _parse_search_result(self, result):
        if False:
            print('Hello World!')
        latitude = result['position']['lat']
        longitude = result['position']['lon']
        return Location(result['address']['freeformAddress'], (latitude, longitude), result)

    def _parse_reverse_json(self, resources, exactly_one):
        if False:
            return 10
        if not resources or not resources['addresses']:
            return None
        if exactly_one:
            return self._parse_reverse_result(resources['addresses'][0])
        else:
            return [self._parse_reverse_result(result) for result in resources['addresses']]

    def _parse_reverse_result(self, result):
        if False:
            print('Hello World!')
        (latitude, longitude) = result['position'].split(',')
        return Location(result['address']['freeformAddress'], (latitude, longitude), result)

    def _geocoder_exception_handler(self, error):
        if False:
            i = 10
            return i + 15
        if not isinstance(error, AdapterHTTPError):
            return
        if error.status_code is None or error.text is None:
            return
        if error.status_code >= 400 and 'Developer Over Qps' in error.text:
            raise GeocoderQuotaExceeded('Developer Over Qps') from error