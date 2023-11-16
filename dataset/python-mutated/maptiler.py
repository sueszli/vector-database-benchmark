from functools import partial
from urllib.parse import quote, urlencode
from geopy.geocoders.base import DEFAULT_SENTINEL, Geocoder
from geopy.location import Location
from geopy.point import Point
from geopy.util import logger
__all__ = ('MapTiler',)

class MapTiler(Geocoder):
    """Geocoder using the MapTiler API.

    Documentation at:
        https://cloud.maptiler.com/geocoding/ (requires sign-up)
    """
    api_path = '/geocoding/%(query)s.json'

    def __init__(self, api_key, *, scheme=None, timeout=DEFAULT_SENTINEL, proxies=DEFAULT_SENTINEL, user_agent=None, ssl_context=DEFAULT_SENTINEL, adapter_factory=None, domain='api.maptiler.com'):
        if False:
            print('Hello World!')
        "\n        :param str api_key: The API key required by Maptiler to perform\n            geocoding requests. API keys are managed through Maptiler's account\n            page (https://cloud.maptiler.com/account/keys).\n\n        :param str scheme:\n            See :attr:`geopy.geocoders.options.default_scheme`.\n\n        :param int timeout:\n            See :attr:`geopy.geocoders.options.default_timeout`.\n\n        :param dict proxies:\n            See :attr:`geopy.geocoders.options.default_proxies`.\n\n        :param str user_agent:\n            See :attr:`geopy.geocoders.options.default_user_agent`.\n\n        :type ssl_context: :class:`ssl.SSLContext`\n        :param ssl_context:\n            See :attr:`geopy.geocoders.options.default_ssl_context`.\n\n        :param callable adapter_factory:\n            See :attr:`geopy.geocoders.options.default_adapter_factory`.\n\n            .. versionadded:: 2.0\n\n        :param str domain: base api domain for Maptiler\n        "
        super().__init__(scheme=scheme, timeout=timeout, proxies=proxies, user_agent=user_agent, ssl_context=ssl_context, adapter_factory=adapter_factory)
        self.api_key = api_key
        self.domain = domain.strip('/')
        self.api = '%s://%s%s' % (self.scheme, self.domain, self.api_path)

    def _parse_json(self, json, exactly_one=True):
        if False:
            return 10
        features = json['features']
        if not features:
            return None

        def parse_feature(feature):
            if False:
                print('Hello World!')
            location = feature['place_name']
            longitude = feature['center'][0]
            latitude = feature['center'][1]
            return Location(location, (latitude, longitude), feature)
        if exactly_one:
            return parse_feature(features[0])
        else:
            return [parse_feature(feature) for feature in features]

    def geocode(self, query, *, exactly_one=True, timeout=DEFAULT_SENTINEL, proximity=None, language=None, bbox=None):
        if False:
            i = 10
            return i + 15
        '\n        Return a location point by address.\n\n        :param str query: The address or query you wish to geocode.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder\'s initialization.\n\n        :param proximity: A coordinate to bias local results based on a provided\n            location.\n        :type proximity: :class:`geopy.point.Point`, list or tuple of ``(latitude,\n            longitude)``, or string as ``"%(latitude)s, %(longitude)s"``.\n\n        :param language: Prefer results in specific languages. Accepts\n            a single string like ``"en"`` or a list like ``["de", "en"]``.\n        :type language: str or list\n\n        :param bbox: The bounding box of the viewport within which\n            to bias geocode results more prominently.\n            Example: ``[Point(22, 180), Point(-22, -180)]``.\n        :type bbox: list or tuple of 2 items of :class:`geopy.point.Point` or\n            ``(latitude, longitude)`` or ``"%(latitude)s, %(longitude)s"``.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        '
        params = {'key': self.api_key}
        query = query
        if bbox:
            params['bbox'] = self._format_bounding_box(bbox, '%(lon1)s,%(lat1)s,%(lon2)s,%(lat2)s')
        if isinstance(language, str):
            language = [language]
        if language:
            params['language'] = ','.join(language)
        if proximity:
            p = Point(proximity)
            params['proximity'] = '%s,%s' % (p.longitude, p.latitude)
        quoted_query = quote(query.encode('utf-8'))
        url = '?'.join((self.api % dict(query=quoted_query), urlencode(params)))
        logger.debug('%s.geocode: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def reverse(self, query, *, exactly_one=True, timeout=DEFAULT_SENTINEL, language=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return an address by location point.\n\n        :param query: The coordinates for which you wish to obtain the\n            closest human-readable addresses.\n        :type query: :class:`geopy.point.Point`, list or tuple of ``(latitude,\n            longitude)``, or string as ``"%(latitude)s, %(longitude)s"``.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder\'s initialization.\n\n        :param language: Prefer results in specific languages. Accepts\n            a single string like ``"en"`` or a list like ``["de", "en"]``.\n        :type language: str or list\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        '
        params = {'key': self.api_key}
        if isinstance(language, str):
            language = [language]
        if language:
            params['language'] = ','.join(language)
        point = self._coerce_point_to_string(query, '%(lon)s,%(lat)s')
        quoted_query = quote(point.encode('utf-8'))
        url = '?'.join((self.api % dict(query=quoted_query), urlencode(params)))
        logger.debug('%s.reverse: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)