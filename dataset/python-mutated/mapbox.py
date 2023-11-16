from functools import partial
from urllib.parse import quote, urlencode
from geopy.geocoders.base import DEFAULT_SENTINEL, Geocoder
from geopy.location import Location
from geopy.point import Point
from geopy.util import logger
__all__ = ('MapBox',)

class MapBox(Geocoder):
    """Geocoder using the Mapbox API.

    Documentation at:
        https://www.mapbox.com/api-documentation/
    """
    api_path = '/geocoding/v5/mapbox.places/%(query)s.json/'

    def __init__(self, api_key, *, scheme=None, timeout=DEFAULT_SENTINEL, proxies=DEFAULT_SENTINEL, user_agent=None, ssl_context=DEFAULT_SENTINEL, adapter_factory=None, domain='api.mapbox.com', referer=None):
        if False:
            while True:
                i = 10
        "\n        :param str api_key: The API key required by Mapbox to perform\n            geocoding requests. API keys are managed through Mapox's account\n            page (https://www.mapbox.com/account/access-tokens).\n\n        :param str scheme:\n            See :attr:`geopy.geocoders.options.default_scheme`.\n\n        :param int timeout:\n            See :attr:`geopy.geocoders.options.default_timeout`.\n\n        :param dict proxies:\n            See :attr:`geopy.geocoders.options.default_proxies`.\n\n        :param str user_agent:\n            See :attr:`geopy.geocoders.options.default_user_agent`.\n\n        :type ssl_context: :class:`ssl.SSLContext`\n        :param ssl_context:\n            See :attr:`geopy.geocoders.options.default_ssl_context`.\n\n        :param callable adapter_factory:\n            See :attr:`geopy.geocoders.options.default_adapter_factory`.\n\n            .. versionadded:: 2.0\n\n        :param str domain: base api domain for mapbox\n\n        :param str referer: The URL used to satisfy the URL restriction of\n            mapbox tokens.\n\n            .. versionadded:: 2.3\n        "
        super().__init__(scheme=scheme, timeout=timeout, proxies=proxies, user_agent=user_agent, ssl_context=ssl_context, adapter_factory=adapter_factory)
        self.api_key = api_key
        self.domain = domain.strip('/')
        self.api = '%s://%s%s' % (self.scheme, self.domain, self.api_path)
        if referer:
            self.headers['Referer'] = referer

    def _parse_json(self, json, exactly_one=True):
        if False:
            while True:
                i = 10
        'Returns location, (latitude, longitude) from json feed.'
        features = json['features']
        if features == []:
            return None

        def parse_feature(feature):
            if False:
                i = 10
                return i + 15
            location = feature['place_name']
            longitude = feature['geometry']['coordinates'][0]
            latitude = feature['geometry']['coordinates'][1]
            return Location(location, (latitude, longitude), feature)
        if exactly_one:
            return parse_feature(features[0])
        else:
            return [parse_feature(feature) for feature in features]

    def geocode(self, query, *, exactly_one=True, timeout=DEFAULT_SENTINEL, proximity=None, country=None, language=None, bbox=None):
        if False:
            while True:
                i = 10
        '\n        Return a location point by address.\n\n        :param str query: The address or query you wish to geocode.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder\'s initialization.\n\n        :param proximity: A coordinate to bias local results based on a provided\n            location.\n        :type proximity: :class:`geopy.point.Point`, list or tuple of ``(latitude,\n            longitude)``, or string as ``"%(latitude)s, %(longitude)s"``.\n\n        :param country: Country to filter result in form of\n            ISO 3166-1 alpha-2 country code (e.g. ``FR``).\n            Might be a Python list of strings.\n\n        :type country: str or list\n\n        :param str language: This parameter controls the language of the text supplied in\n            responses, and also affects result scoring, with results matching the user’s\n            query in the requested language being preferred over results that match in\n            another language. You can pass two letters country codes (ISO 639-1).\n\n            .. versionadded:: 2.3\n\n        :param bbox: The bounding box of the viewport within which\n            to bias geocode results more prominently.\n            Example: ``[Point(22, 180), Point(-22, -180)]``.\n        :type bbox: list or tuple of 2 items of :class:`geopy.point.Point` or\n            ``(latitude, longitude)`` or ``"%(latitude)s, %(longitude)s"``.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        '
        params = {}
        params['access_token'] = self.api_key
        if bbox:
            params['bbox'] = self._format_bounding_box(bbox, '%(lon1)s,%(lat1)s,%(lon2)s,%(lat2)s')
        if not country:
            country = []
        if isinstance(country, str):
            country = [country]
        if country:
            params['country'] = ','.join(country)
        if proximity:
            p = Point(proximity)
            params['proximity'] = '%s,%s' % (p.longitude, p.latitude)
        if language:
            params['language'] = language
        quoted_query = quote(query.encode('utf-8'))
        url = '?'.join((self.api % dict(query=quoted_query), urlencode(params)))
        logger.debug('%s.geocode: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def reverse(self, query, *, exactly_one=True, timeout=DEFAULT_SENTINEL):
        if False:
            return 10
        '\n        Return an address by location point.\n\n        :param query: The coordinates for which you wish to obtain the\n            closest human-readable addresses.\n        :type query: :class:`geopy.point.Point`, list or tuple of ``(latitude,\n            longitude)``, or string as ``"%(latitude)s, %(longitude)s"``.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder\'s initialization.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        '
        params = {}
        params['access_token'] = self.api_key
        point = self._coerce_point_to_string(query, '%(lon)s,%(lat)s')
        quoted_query = quote(point.encode('utf-8'))
        url = '?'.join((self.api % dict(query=quoted_query), urlencode(params)))
        logger.debug('%s.reverse: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)