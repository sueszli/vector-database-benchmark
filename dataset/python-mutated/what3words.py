import re
from functools import partial
from urllib.parse import urlencode
from geopy import exc
from geopy.geocoders.base import DEFAULT_SENTINEL, Geocoder
from geopy.location import Location
from geopy.util import logger
__all__ = ('What3Words', 'What3WordsV3')
_MULTIPLE_WORD_RE = re.compile('[^\\W\\d\\_]+\\.{1,1}[^\\W\\d\\_]+\\.{1,1}[^\\W\\d\\_]+$', re.U)

def _check_query(query):
    if False:
        while True:
            i = 10
    '\n    Check query validity with regex\n    '
    if not _MULTIPLE_WORD_RE.match(query):
        return False
    else:
        return True

class What3Words(Geocoder):
    """What3Words geocoder using the legacy V2 API.

    Documentation at:
        https://docs.what3words.com/api/v2/

    .. attention::
        Consider using :class:`.What3WordsV3` instead.
    """
    geocode_path = '/v2/forward'
    reverse_path = '/v2/reverse'

    def __init__(self, api_key, *, timeout=DEFAULT_SENTINEL, proxies=DEFAULT_SENTINEL, user_agent=None, ssl_context=DEFAULT_SENTINEL, adapter_factory=None, domain='api.what3words.com'):
        if False:
            while True:
                i = 10
        '\n\n        :param str api_key: Key provided by What3Words\n            (https://accounts.what3words.com/register).\n\n        :param int timeout:\n            See :attr:`geopy.geocoders.options.default_timeout`.\n\n        :param dict proxies:\n            See :attr:`geopy.geocoders.options.default_proxies`.\n\n        :param str user_agent:\n            See :attr:`geopy.geocoders.options.default_user_agent`.\n\n        :type ssl_context: :class:`ssl.SSLContext`\n        :param ssl_context:\n            See :attr:`geopy.geocoders.options.default_ssl_context`.\n\n        :param callable adapter_factory:\n            See :attr:`geopy.geocoders.options.default_adapter_factory`.\n\n            .. versionadded:: 2.0\n\n        :param str domain: base api domain\n\n            .. versionadded:: 2.4\n        '
        super().__init__(scheme='https', timeout=timeout, proxies=proxies, user_agent=user_agent, ssl_context=ssl_context, adapter_factory=adapter_factory)
        self.api_key = api_key
        self.geocode_api = '%s://%s%s' % (self.scheme, domain, self.geocode_path)
        self.reverse_api = '%s://%s%s' % (self.scheme, domain, self.reverse_path)

    def geocode(self, query, *, lang='en', exactly_one=True, timeout=DEFAULT_SENTINEL):
        if False:
            while True:
                i = 10
        "\n        Return a location point for a `3 words` query. If the `3 words` address\n        doesn't exist, a :class:`geopy.exc.GeocoderQueryError` exception will be\n        thrown.\n\n        :param str query: The 3-word address you wish to geocode.\n\n        :param str lang: two character language code as supported by\n            the API (https://docs.what3words.com/api/v2/#lang).\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available. Due to the address scheme there is always exactly one\n            result for each `3 words` address, so this parameter is rather\n            useless for this geocoder.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder's initialization.\n\n        :rtype: :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        "
        if not _check_query(query):
            raise exc.GeocoderQueryError("Search string must be 'word.word.word'")
        params = {'addr': query, 'lang': lang.lower(), 'key': self.api_key}
        url = '?'.join((self.geocode_api, urlencode(params)))
        logger.debug('%s.geocode: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def _parse_json(self, resources, exactly_one=True):
        if False:
            while True:
                i = 10
        '\n        Parse type, words, latitude, and longitude and language from a\n        JSON response.\n        '
        code = resources['status'].get('code')
        if code:
            exc_msg = 'Error returned by What3Words: %s' % resources['status']['message']
            if code == 401:
                raise exc.GeocoderAuthenticationFailure(exc_msg)
            raise exc.GeocoderQueryError(exc_msg)

        def parse_resource(resource):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Parse record.\n            '
            if 'geometry' in resource:
                words = resource['words']
                position = resource['geometry']
                (latitude, longitude) = (position['lat'], position['lng'])
                if latitude and longitude:
                    latitude = float(latitude)
                    longitude = float(longitude)
                return Location(words, (latitude, longitude), resource)
            else:
                raise exc.GeocoderParseError('Error parsing result.')
        location = parse_resource(resources)
        if exactly_one:
            return location
        else:
            return [location]

    def reverse(self, query, *, lang='en', exactly_one=True, timeout=DEFAULT_SENTINEL):
        if False:
            while True:
                i = 10
        '\n        Return a `3 words` address by location point. Each point on surface has\n        a `3 words` address, so there\'s always a non-empty response.\n\n        :param query: The coordinates for which you wish to obtain the 3 word\n            address.\n        :type query: :class:`geopy.point.Point`, list or tuple of ``(latitude,\n            longitude)``, or string as ``"%(latitude)s, %(longitude)s"``.\n\n        :param str lang: two character language code as supported by the\n            API (https://docs.what3words.com/api/v2/#lang).\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available. Due to the address scheme there is always exactly one\n            result for each `3 words` address, so this parameter is rather\n            useless for this geocoder.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder\'s initialization.\n\n        :rtype: :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n\n        '
        lang = lang.lower()
        params = {'coords': self._coerce_point_to_string(query), 'lang': lang.lower(), 'key': self.api_key}
        url = '?'.join((self.reverse_api, urlencode(params)))
        logger.debug('%s.reverse: %s', self.__class__.__name__, url)
        callback = partial(self._parse_reverse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def _parse_reverse_json(self, resources, exactly_one=True):
        if False:
            print('Hello World!')
        '\n        Parses a location from a single-result reverse API call.\n        '
        return self._parse_json(resources, exactly_one)

class What3WordsV3(Geocoder):
    """What3Words geocoder using the V3 API.

    Documentation at:
        https://developer.what3words.com/public-api/docs

    .. versionadded:: 2.2
    """
    geocode_path = '/v3/convert-to-coordinates'
    reverse_path = '/v3/convert-to-3wa'

    def __init__(self, api_key, *, timeout=DEFAULT_SENTINEL, proxies=DEFAULT_SENTINEL, user_agent=None, ssl_context=DEFAULT_SENTINEL, adapter_factory=None, domain='api.what3words.com'):
        if False:
            while True:
                i = 10
        '\n\n        :param str api_key: Key provided by What3Words\n            (https://accounts.what3words.com/register).\n\n        :param int timeout:\n            See :attr:`geopy.geocoders.options.default_timeout`.\n\n        :param dict proxies:\n            See :attr:`geopy.geocoders.options.default_proxies`.\n\n        :param str user_agent:\n            See :attr:`geopy.geocoders.options.default_user_agent`.\n\n        :type ssl_context: :class:`ssl.SSLContext`\n        :param ssl_context:\n            See :attr:`geopy.geocoders.options.default_ssl_context`.\n\n        :param callable adapter_factory:\n            See :attr:`geopy.geocoders.options.default_adapter_factory`.\n\n        :param str domain: base api domain\n\n            .. versionadded:: 2.4\n        '
        super().__init__(scheme='https', timeout=timeout, proxies=proxies, user_agent=user_agent, ssl_context=ssl_context, adapter_factory=adapter_factory)
        self.api_key = api_key
        self.geocode_api = '%s://%s%s' % (self.scheme, domain, self.geocode_path)
        self.reverse_api = '%s://%s%s' % (self.scheme, domain, self.reverse_path)

    def geocode(self, query, *, exactly_one=True, timeout=DEFAULT_SENTINEL):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return a location point for a `3 words` query. If the `3 words` address\n        doesn't exist, a :class:`geopy.exc.GeocoderQueryError` exception will be\n        thrown.\n\n        :param str query: The 3-word address you wish to geocode.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available. Due to the address scheme there is always exactly one\n            result for each `3 words` address, so this parameter is rather\n            useless for this geocoder.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder's initialization.\n\n        :rtype: :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        "
        if not _check_query(query):
            raise exc.GeocoderQueryError("Search string must be 'word.word.word'")
        params = {'words': query, 'key': self.api_key}
        url = '?'.join((self.geocode_api, urlencode(params)))
        logger.debug('%s.geocode: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def _parse_json(self, resources, exactly_one=True):
        if False:
            print('Hello World!')
        '\n        Parse type, words, latitude, and longitude and language from a\n        JSON response.\n        '
        error = resources.get('error')
        if error is not None:
            exc_msg = 'Error returned by What3Words: %s' % resources['error']['message']
            exc_code = error.get('code')
            if exc_code in ['MissingKey', 'InvalidKey']:
                raise exc.GeocoderAuthenticationFailure(exc_msg)
            raise exc.GeocoderQueryError(exc_msg)

        def parse_resource(resource):
            if False:
                while True:
                    i = 10
            '\n            Parse record.\n            '
            if 'coordinates' in resource:
                words = resource['words']
                position = resource['coordinates']
                (latitude, longitude) = (position['lat'], position['lng'])
                if latitude and longitude:
                    latitude = float(latitude)
                    longitude = float(longitude)
                return Location(words, (latitude, longitude), resource)
            else:
                raise exc.GeocoderParseError('Error parsing result.')
        location = parse_resource(resources)
        if exactly_one:
            return location
        else:
            return [location]

    def reverse(self, query, *, lang='en', exactly_one=True, timeout=DEFAULT_SENTINEL):
        if False:
            return 10
        '\n        Return a `3 words` address by location point. Each point on surface has\n        a `3 words` address, so there\'s always a non-empty response.\n\n        :param query: The coordinates for which you wish to obtain the 3 word\n            address.\n        :type query: :class:`geopy.point.Point`, list or tuple of ``(latitude,\n            longitude)``, or string as ``"%(latitude)s, %(longitude)s"``.\n\n        :param str lang: two character language code as supported by the\n            API (https://developer.what3words.com/public-api/docs#available-languages).\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available. Due to the address scheme there is always exactly one\n            result for each `3 words` address, so this parameter is rather\n            useless for this geocoder.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder\'s initialization.\n\n        :rtype: :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n\n        '
        lang = lang.lower()
        params = {'coordinates': self._coerce_point_to_string(query), 'language': lang.lower(), 'key': self.api_key}
        url = '?'.join((self.reverse_api, urlencode(params)))
        logger.debug('%s.reverse: %s', self.__class__.__name__, url)
        callback = partial(self._parse_reverse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def _parse_reverse_json(self, resources, exactly_one=True):
        if False:
            print('Hello World!')
        '\n        Parses a location from a single-result reverse API call.\n        '
        return self._parse_json(resources, exactly_one)