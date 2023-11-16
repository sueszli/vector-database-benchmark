import warnings
from functools import partial
from urllib.parse import urlencode
from geopy.geocoders.base import DEFAULT_SENTINEL, Geocoder
from geopy.location import Location
from geopy.util import logger
__all__ = ('Pelias',)

class Pelias(Geocoder):
    """Pelias geocoder.

    Documentation at:
        https://github.com/pelias/documentation

    See also :class:`geopy.geocoders.GeocodeEarth` which is a Pelias-based
    service provided by the developers of Pelias itself.
    """
    geocode_path = '/v1/search'
    reverse_path = '/v1/reverse'

    def __init__(self, domain, api_key=None, *, timeout=DEFAULT_SENTINEL, proxies=DEFAULT_SENTINEL, user_agent=None, scheme=None, ssl_context=DEFAULT_SENTINEL, adapter_factory=None):
        if False:
            return 10
        '\n        :param str domain: Specify a domain for Pelias API.\n\n        :param str api_key: Pelias API key, optional.\n\n        :param int timeout:\n            See :attr:`geopy.geocoders.options.default_timeout`.\n\n        :param dict proxies:\n            See :attr:`geopy.geocoders.options.default_proxies`.\n\n        :param str user_agent:\n            See :attr:`geopy.geocoders.options.default_user_agent`.\n\n        :param str scheme:\n            See :attr:`geopy.geocoders.options.default_scheme`.\n\n        :type ssl_context: :class:`ssl.SSLContext`\n        :param ssl_context:\n            See :attr:`geopy.geocoders.options.default_ssl_context`.\n\n        :param callable adapter_factory:\n            See :attr:`geopy.geocoders.options.default_adapter_factory`.\n\n            .. versionadded:: 2.0\n\n        '
        super().__init__(scheme=scheme, timeout=timeout, proxies=proxies, user_agent=user_agent, ssl_context=ssl_context, adapter_factory=adapter_factory)
        self.api_key = api_key
        self.domain = domain.strip('/')
        self.geocode_api = '%s://%s%s' % (self.scheme, self.domain, self.geocode_path)
        self.reverse_api = '%s://%s%s' % (self.scheme, self.domain, self.reverse_path)

    def geocode(self, query, *, exactly_one=True, timeout=DEFAULT_SENTINEL, boundary_rect=None, countries=None, country_bias=None, language=None):
        if False:
            i = 10
            return i + 15
        '\n        Return a location point by address.\n\n        :param str query: The address or query you wish to geocode.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder\'s initialization.\n\n        :type boundary_rect: list or tuple of 2 items of :class:`geopy.point.Point`\n            or ``(latitude, longitude)`` or ``"%(latitude)s, %(longitude)s"``.\n        :param boundary_rect: Coordinates to restrict search within.\n            Example: ``[Point(22, 180), Point(-22, -180)]``.\n\n        :param list countries: A list of country codes specified in\n            `ISO 3166-1 alpha-2 or alpha-3\n            <https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3>`_\n            format, e.g. ``[\'USA\', \'CAN\']``.\n            This is a hard filter.\n\n            .. versionadded:: 2.3\n\n        :param str country_bias: Bias results to this country (ISO alpha-3).\n\n            .. deprecated:: 2.3\n                Use ``countries`` instead. This option behaves the same way,\n                i.e. it\'s not a soft filter as the name suggests.\n                This parameter is scheduled for removal in geopy 3.0.\n\n        :param str language: Preferred language in which to return results.\n            Either uses standard\n            `RFC2616 <http://www.ietf.org/rfc/rfc2616.txt>`_\n            accept-language string or a simple comma-separated\n            list of language codes.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        '
        params = {'text': query}
        if self.api_key:
            params.update({'api_key': self.api_key})
        if boundary_rect:
            (lon1, lat1, lon2, lat2) = self._format_bounding_box(boundary_rect, '%(lon1)s,%(lat1)s,%(lon2)s,%(lat2)s').split(',')
            params['boundary.rect.min_lon'] = lon1
            params['boundary.rect.min_lat'] = lat1
            params['boundary.rect.max_lon'] = lon2
            params['boundary.rect.max_lat'] = lat2
        if country_bias:
            warnings.warn("`country_bias` is deprecated, because it's not a soft filter as the name suggests. Pass a list to the `countries` option instead, which behaves the same way. In geopy 3 the `country_bias` option will be removed.", DeprecationWarning, stacklevel=2)
            params['boundary.country'] = country_bias
        if countries:
            params['boundary.country'] = ','.join(countries)
        if language:
            params['lang'] = language
        url = '?'.join((self.geocode_api, urlencode(params)))
        logger.debug('%s.geocode_api: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def reverse(self, query, *, exactly_one=True, timeout=DEFAULT_SENTINEL, language=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return an address by location point.\n\n        :param query: The coordinates for which you wish to obtain the\n            closest human-readable addresses.\n        :type query: :class:`geopy.point.Point`, list or tuple of ``(latitude,\n            longitude)``, or string as ``"%(latitude)s, %(longitude)s"``.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder\'s initialization.\n\n        :param str language: Preferred language in which to return results.\n            Either uses standard\n            `RFC2616 <http://www.ietf.org/rfc/rfc2616.txt>`_\n            accept-language string or a simple comma-separated\n            list of language codes.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        '
        try:
            (lat, lon) = self._coerce_point_to_string(query).split(',')
        except ValueError:
            raise ValueError('Must be a coordinate pair or Point')
        params = {'point.lat': lat, 'point.lon': lon}
        if language:
            params['lang'] = language
        if self.api_key:
            params.update({'api_key': self.api_key})
        url = '?'.join((self.reverse_api, urlencode(params)))
        logger.debug('%s.reverse: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def _parse_code(self, feature):
        if False:
            for i in range(10):
                print('nop')
        latitude = feature.get('geometry', {}).get('coordinates', [])[1]
        longitude = feature.get('geometry', {}).get('coordinates', [])[0]
        placename = feature.get('properties', {}).get('name')
        return Location(placename, (latitude, longitude), feature)

    def _parse_json(self, response, exactly_one):
        if False:
            print('Hello World!')
        if response is None:
            return None
        features = response['features']
        if not len(features):
            return None
        if exactly_one:
            return self._parse_code(features[0])
        else:
            return [self._parse_code(feature) for feature in features]