from functools import partial
from urllib.parse import urlencode
from geopy.exc import GeocoderAuthenticationFailure, GeocoderQueryError, GeocoderQuotaExceeded, GeocoderServiceError, GeocoderUnavailable
from geopy.geocoders.base import DEFAULT_SENTINEL, Geocoder
from geopy.location import Location
from geopy.util import logger
__all__ = ('Geokeo',)

class Geokeo(Geocoder):
    """Geocoder using the geokeo API.

    Documentation at:
        https://geokeo.com/documentation.php

    .. versionadded:: 2.4
    """
    geocode_path = '/geocode/v1/search.php'
    reverse_path = '/geocode/v1/reverse.php'

    def __init__(self, api_key, *, domain='geokeo.com', scheme=None, timeout=DEFAULT_SENTINEL, proxies=DEFAULT_SENTINEL, user_agent=None, ssl_context=DEFAULT_SENTINEL, adapter_factory=None):
        if False:
            while True:
                i = 10
        '\n\n        :param str api_key: The API key required by Geokeo.com\n            to perform geocoding requests. You can get your key here:\n            https://geokeo.com/\n\n        :param str domain: Domain where the target Geokeo service\n            is hosted.\n\n        :param str scheme:\n            See :attr:`geopy.geocoders.options.default_scheme`.\n\n        :param int timeout:\n            See :attr:`geopy.geocoders.options.default_timeout`.\n\n        :param dict proxies:\n            See :attr:`geopy.geocoders.options.default_proxies`.\n\n        :param str user_agent:\n            See :attr:`geopy.geocoders.options.default_user_agent`.\n\n        :type ssl_context: :class:`ssl.SSLContext`\n        :param ssl_context:\n            See :attr:`geopy.geocoders.options.default_ssl_context`.\n\n        :param callable adapter_factory:\n            See :attr:`geopy.geocoders.options.default_adapter_factory`.\n\n        '
        super().__init__(scheme=scheme, timeout=timeout, proxies=proxies, user_agent=user_agent, ssl_context=ssl_context, adapter_factory=adapter_factory)
        self.api_key = api_key
        self.domain = domain.strip('/')
        self.api = '%s://%s%s' % (self.scheme, self.domain, self.geocode_path)
        self.reverse_api = '%s://%s%s' % (self.scheme, self.domain, self.reverse_path)

    def geocode(self, query, *, country=None, exactly_one=True, timeout=DEFAULT_SENTINEL):
        if False:
            return 10
        "\n        Return a location point by address.\n\n        :param str query: The address or query you wish to geocode.\n\n        :param str country: Restricts the results to the specified\n            country. The country code is a 2 character code as\n            defined by the ISO 3166-1 Alpha 2 standard (e.g. ``us``).\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder's initialization.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        "
        params = {'api': self.api_key, 'q': query}
        if country:
            params['country'] = country
        url = '?'.join((self.api, urlencode(params)))
        logger.debug('%s.geocode: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def reverse(self, query, *, exactly_one=True, timeout=DEFAULT_SENTINEL):
        if False:
            return 10
        '\n        Return an address by location point.\n\n        :param query: The coordinates for which you wish to obtain the\n            closest human-readable addresses.\n        :type query: :class:`geopy.point.Point`, list or tuple of ``(latitude,\n            longitude)``, or string as ``"%(latitude)s, %(longitude)s"``.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder\'s initialization.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        '
        try:
            (lat, lng) = self._coerce_point_to_string(query).split(',')
        except ValueError:
            raise ValueError('Must be a coordinate pair or Point')
        params = {'api': self.api_key, 'lat': lat, 'lng': lng}
        url = '?'.join((self.reverse_api, urlencode(params)))
        logger.debug('%s.reverse: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def _parse_json(self, page, exactly_one=True):
        if False:
            i = 10
            return i + 15
        places = page.get('results', [])
        self._check_status(page)
        if not places:
            return None

        def parse_place(place):
            if False:
                return 10
            'Get the location, lat, lng from a single json place.'
            location = place.get('formatted_address')
            latitude = place['geometry']['location']['lat']
            longitude = place['geometry']['location']['lng']
            return Location(location, (latitude, longitude), place)
        if exactly_one:
            return parse_place(places[0])
        else:
            return [parse_place(place) for place in places]

    def _check_status(self, page):
        if False:
            for i in range(10):
                print('nop')
        status = (page.get('status') or '').upper()
        if status == 'OK':
            return
        if status == 'ZERO_RESULTS':
            return
        if status == 'INVALID_REQUEST':
            raise GeocoderQueryError('Invalid request parameters')
        elif status == 'ACCESS_DENIED':
            raise GeocoderAuthenticationFailure('Access denied')
        elif status == 'OVER_QUERY_LIMIT':
            raise GeocoderQuotaExceeded('Over query limit')
        elif status == 'INTERNAL_SERVER_ERROR':
            raise GeocoderUnavailable('Internal server error')
        else:
            raise GeocoderServiceError('Unknown error')