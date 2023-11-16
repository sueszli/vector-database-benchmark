from functools import partial
from urllib.parse import urlencode
from geopy.geocoders.base import DEFAULT_SENTINEL, Geocoder
from geopy.location import Location
from geopy.util import logger
__all__ = ('BANFrance',)

class BANFrance(Geocoder):
    """Geocoder using the Base Adresse Nationale France API.

    Documentation at:
        https://adresse.data.gouv.fr/api
    """
    geocode_path = '/search'
    reverse_path = '/reverse'

    def __init__(self, *, domain='api-adresse.data.gouv.fr', scheme=None, timeout=DEFAULT_SENTINEL, proxies=DEFAULT_SENTINEL, user_agent=None, ssl_context=DEFAULT_SENTINEL, adapter_factory=None):
        if False:
            i = 10
            return i + 15
        "\n\n        :param str domain: Currently it is ``'api-adresse.data.gouv.fr'``, can\n            be changed for testing purposes.\n\n        :param str scheme:\n            See :attr:`geopy.geocoders.options.default_scheme`.\n\n        :param int timeout:\n            See :attr:`geopy.geocoders.options.default_timeout`.\n\n        :param dict proxies:\n            See :attr:`geopy.geocoders.options.default_proxies`.\n\n        :param str user_agent:\n            See :attr:`geopy.geocoders.options.default_user_agent`.\n\n        :type ssl_context: :class:`ssl.SSLContext`\n        :param ssl_context:\n            See :attr:`geopy.geocoders.options.default_ssl_context`.\n\n        :param callable adapter_factory:\n            See :attr:`geopy.geocoders.options.default_adapter_factory`.\n\n            .. versionadded:: 2.0\n\n        "
        super().__init__(scheme=scheme, timeout=timeout, proxies=proxies, user_agent=user_agent, ssl_context=ssl_context, adapter_factory=adapter_factory)
        self.domain = domain.strip('/')
        self.geocode_api = '%s://%s%s' % (self.scheme, self.domain, self.geocode_path)
        self.reverse_api = '%s://%s%s' % (self.scheme, self.domain, self.reverse_path)

    def geocode(self, query, *, limit=None, exactly_one=True, timeout=DEFAULT_SENTINEL):
        if False:
            print('Hello World!')
        "\n        Return a location point by address.\n\n        :param str query: The address or query you wish to geocode.\n\n        :param int limit: Defines the maximum number of items in the\n            response structure. If not provided and there are multiple\n            results the BAN API will return 5 results by default.\n            This will be reset to one if ``exactly_one`` is True.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder's initialization.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n\n        "
        params = {'q': query}
        if limit is not None:
            params['limit'] = limit
        url = '?'.join((self.geocode_api, urlencode(params)))
        logger.debug('%s.geocode: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def reverse(self, query, *, exactly_one=True, timeout=DEFAULT_SENTINEL):
        if False:
            print('Hello World!')
        '\n        Return an address by location point.\n\n        :param query: The coordinates for which you wish to obtain the\n            closest human-readable addresses.\n        :type query: :class:`geopy.point.Point`, list or tuple of ``(latitude,\n            longitude)``, or string as ``"%(latitude)s, %(longitude)s"``.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder\'s initialization.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n\n        '
        try:
            (lat, lon) = self._coerce_point_to_string(query).split(',')
        except ValueError:
            raise ValueError('Must be a coordinate pair or Point')
        params = {'lat': lat, 'lon': lon}
        url = '?'.join((self.reverse_api, urlencode(params)))
        logger.debug('%s.reverse: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def _parse_feature(self, feature):
        if False:
            print('Hello World!')
        latitude = feature.get('geometry', {}).get('coordinates', [])[1]
        longitude = feature.get('geometry', {}).get('coordinates', [])[0]
        placename = feature.get('properties', {}).get('label')
        return Location(placename, (latitude, longitude), feature)

    def _parse_json(self, response, exactly_one):
        if False:
            i = 10
            return i + 15
        if response is None or 'features' not in response:
            return None
        features = response['features']
        if not len(features):
            return None
        if exactly_one:
            return self._parse_feature(features[0])
        else:
            return [self._parse_feature(feature) for feature in features]