from functools import partial
from urllib.parse import urlencode
from geopy.geocoders.base import DEFAULT_SENTINEL, Geocoder
from geopy.location import Location
from geopy.util import logger
__all__ = ('MapQuest',)

class MapQuest(Geocoder):
    """Geocoder using the MapQuest API based on Licensed data.

    Documentation at:
        https://developer.mapquest.com/documentation/geocoding-api/

    MapQuest provides two Geocoding APIs:

    - :class:`geopy.geocoders.OpenMapQuest` Nominatim-alike API
      which is based on Open data from OpenStreetMap.
    - :class:`geopy.geocoders.MapQuest` (this class) MapQuest's own API
      which is based on Licensed data.
    """
    geocode_path = '/geocoding/v1/address'
    reverse_path = '/geocoding/v1/reverse'

    def __init__(self, api_key, *, scheme=None, timeout=DEFAULT_SENTINEL, proxies=DEFAULT_SENTINEL, user_agent=None, ssl_context=DEFAULT_SENTINEL, adapter_factory=None, domain='www.mapquestapi.com'):
        if False:
            i = 10
            return i + 15
        '\n        :param str api_key: The API key required by Mapquest to perform\n            geocoding requests. API keys are managed through MapQuest\'s "Manage Keys"\n            page (https://developer.mapquest.com/user/me/apps).\n\n        :param str scheme:\n            See :attr:`geopy.geocoders.options.default_scheme`.\n\n        :param int timeout:\n            See :attr:`geopy.geocoders.options.default_timeout`.\n\n        :param dict proxies:\n            See :attr:`geopy.geocoders.options.default_proxies`.\n\n        :param str user_agent:\n            See :attr:`geopy.geocoders.options.default_user_agent`.\n\n        :type ssl_context: :class:`ssl.SSLContext`\n        :param ssl_context:\n            See :attr:`geopy.geocoders.options.default_ssl_context`.\n\n        :param callable adapter_factory:\n            See :attr:`geopy.geocoders.options.default_adapter_factory`.\n\n            .. versionadded:: 2.0\n\n        :param str domain: base api domain for mapquest\n        '
        super().__init__(scheme=scheme, timeout=timeout, proxies=proxies, user_agent=user_agent, ssl_context=ssl_context, adapter_factory=adapter_factory)
        self.api_key = api_key
        self.domain = domain.strip('/')
        self.geocode_api = '%s://%s%s' % (self.scheme, self.domain, self.geocode_path)
        self.reverse_api = '%s://%s%s' % (self.scheme, self.domain, self.reverse_path)

    def _parse_json(self, json, exactly_one=True):
        if False:
            print('Hello World!')
        'Returns location, (latitude, longitude) from json feed.'
        features = json['results'][0]['locations']
        if features == []:
            return None

        def parse_location(feature):
            if False:
                while True:
                    i = 10
            addr_keys = ['street', 'adminArea6', 'adminArea5', 'adminArea4', 'adminArea3', 'adminArea2', 'adminArea1', 'postalCode']
            location = [feature[k] for k in addr_keys if feature.get(k)]
            return ', '.join(location)

        def parse_feature(feature):
            if False:
                return 10
            location = parse_location(feature)
            longitude = feature['latLng']['lng']
            latitude = feature['latLng']['lat']
            return Location(location, (latitude, longitude), feature)
        if exactly_one:
            return parse_feature(features[0])
        else:
            return [parse_feature(feature) for feature in features]

    def geocode(self, query, *, exactly_one=True, timeout=DEFAULT_SENTINEL, limit=None, bounds=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a location point by address.\n\n        :param str query: The address or query you wish to geocode.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder\'s initialization.\n\n        :param int limit: Limit the maximum number of items in the\n            response. This will be reset to one if ``exactly_one`` is True.\n\n        :param bounds: The bounding box of the viewport within which\n            to bias geocode results more prominently.\n            Example: ``[Point(22, 180), Point(-22, -180)]``.\n        :type bounds: list or tuple of 2 items of :class:`geopy.point.Point` or\n            ``(latitude, longitude)`` or ``"%(latitude)s, %(longitude)s"``.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        '
        params = {}
        params['key'] = self.api_key
        params['location'] = query
        if limit is not None:
            params['maxResults'] = limit
        if exactly_one:
            params['maxResults'] = 1
        if bounds:
            params['boundingBox'] = self._format_bounding_box(bounds, '%(lat2)s,%(lon1)s,%(lat1)s,%(lon2)s')
        url = '?'.join((self.geocode_api, urlencode(params)))
        logger.debug('%s.geocode: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def reverse(self, query, *, exactly_one=True, timeout=DEFAULT_SENTINEL):
        if False:
            i = 10
            return i + 15
        '\n        Return an address by location point.\n\n        :param query: The coordinates for which you wish to obtain the\n            closest human-readable addresses.\n        :type query: :class:`geopy.point.Point`, list or tuple of ``(latitude,\n            longitude)``, or string as ``"%(latitude)s, %(longitude)s"``.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder\'s initialization.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        '
        params = {}
        params['key'] = self.api_key
        point = self._coerce_point_to_string(query, '%(lat)s,%(lon)s')
        params['location'] = point
        url = '?'.join((self.reverse_api, urlencode(params)))
        logger.debug('%s.reverse: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)