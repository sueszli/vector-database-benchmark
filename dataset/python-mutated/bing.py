import collections.abc
from functools import partial
from urllib.parse import quote, urlencode
from geopy.exc import GeocoderAuthenticationFailure, GeocoderInsufficientPrivileges, GeocoderRateLimited, GeocoderServiceError, GeocoderUnavailable
from geopy.geocoders.base import DEFAULT_SENTINEL, Geocoder
from geopy.location import Location
from geopy.util import join_filter, logger
__all__ = ('Bing',)

class Bing(Geocoder):
    """Geocoder using the Bing Maps Locations API.

    Documentation at:
        https://msdn.microsoft.com/en-us/library/ff701715.aspx
    """
    structured_query_params = {'addressLine', 'locality', 'adminDistrict', 'countryRegion', 'postalCode'}
    geocode_path = '/REST/v1/Locations'
    reverse_path = '/REST/v1/Locations/%(point)s'

    def __init__(self, api_key, *, scheme=None, timeout=DEFAULT_SENTINEL, proxies=DEFAULT_SENTINEL, user_agent=None, ssl_context=DEFAULT_SENTINEL, adapter_factory=None, domain='dev.virtualearth.net'):
        if False:
            return 10
        '\n\n        :param str api_key: Should be a valid Bing Maps API key\n            (https://www.microsoft.com/en-us/maps/create-a-bing-maps-key).\n\n        :param str scheme:\n            See :attr:`geopy.geocoders.options.default_scheme`.\n\n        :param int timeout:\n            See :attr:`geopy.geocoders.options.default_timeout`.\n\n        :param dict proxies:\n            See :attr:`geopy.geocoders.options.default_proxies`.\n\n        :param str user_agent:\n            See :attr:`geopy.geocoders.options.default_user_agent`.\n\n        :type ssl_context: :class:`ssl.SSLContext`\n        :param ssl_context:\n            See :attr:`geopy.geocoders.options.default_ssl_context`.\n\n        :param callable adapter_factory:\n            See :attr:`geopy.geocoders.options.default_adapter_factory`.\n\n            .. versionadded:: 2.0\n\n        :param str domain: base api domain\n\n            .. versionadded:: 2.4\n        '
        super().__init__(scheme=scheme, timeout=timeout, proxies=proxies, user_agent=user_agent, ssl_context=ssl_context, adapter_factory=adapter_factory)
        self.api_key = api_key
        self.geocode_api = '%s://%s%s' % (self.scheme, domain, self.geocode_path)
        self.reverse_api = '%s://%s%s' % (self.scheme, domain, self.reverse_path)

    def geocode(self, query, *, exactly_one=True, user_location=None, timeout=DEFAULT_SENTINEL, culture=None, include_neighborhood=None, include_country_code=False):
        if False:
            i = 10
            return i + 15
        "\n        Return a location point by address.\n\n        :param query: The address or query you wish to geocode.\n\n            For a structured query, provide a dictionary whose keys\n            are one of: `addressLine`, `locality` (city),\n            `adminDistrict` (state), `countryRegion`, or `postalCode`.\n        :type query: str or dict\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param user_location: Prioritize results closer to\n            this location.\n        :type user_location: :class:`geopy.point.Point`\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder's initialization.\n\n        :param str culture: Affects the language of the response,\n            must be a two-letter country code.\n\n        :param bool include_neighborhood: Sets whether to include the\n            neighborhood field in the response.\n\n        :param bool include_country_code: Sets whether to include the\n            two-letter ISO code of the country in the response (field name\n            'countryRegionIso2').\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        "
        if isinstance(query, collections.abc.Mapping):
            params = {key: val for (key, val) in query.items() if key in self.structured_query_params}
            params['key'] = self.api_key
        else:
            params = {'query': query, 'key': self.api_key}
        if user_location:
            params['userLocation'] = ','.join((str(user_location.latitude), str(user_location.longitude)))
        if exactly_one:
            params['maxResults'] = 1
        if culture:
            params['culture'] = culture
        if include_neighborhood is not None:
            params['includeNeighborhood'] = include_neighborhood
        if include_country_code:
            params['include'] = 'ciso2'
        url = '?'.join((self.geocode_api, urlencode(params)))
        logger.debug('%s.geocode: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def reverse(self, query, *, exactly_one=True, timeout=DEFAULT_SENTINEL, culture=None, include_country_code=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return an address by location point.\n\n        :param query: The coordinates for which you wish to obtain the\n            closest human-readable addresses.\n        :type query: :class:`geopy.point.Point`, list or tuple of ``(latitude,\n            longitude)``, or string as ``"%(latitude)s, %(longitude)s"``.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder\'s initialization.\n\n        :param str culture: Affects the language of the response,\n            must be a two-letter country code.\n\n        :param bool include_country_code: Sets whether to include the\n            two-letter ISO code of the country in the response (field name\n            \'countryRegionIso2\').\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        '
        point = self._coerce_point_to_string(query)
        params = {'key': self.api_key}
        if culture:
            params['culture'] = culture
        if include_country_code:
            params['include'] = 'ciso2'
        quoted_point = quote(point.encode('utf-8'))
        url = '?'.join((self.reverse_api % dict(point=quoted_point), urlencode(params)))
        logger.debug('%s.reverse: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def _parse_json(self, doc, exactly_one=True):
        if False:
            while True:
                i = 10
        '\n        Parse a location name, latitude, and longitude from an JSON response.\n        '
        status_code = doc.get('statusCode', 200)
        if status_code != 200:
            err = doc.get('errorDetails', '')
            if status_code == 401:
                raise GeocoderAuthenticationFailure(err)
            elif status_code == 403:
                raise GeocoderInsufficientPrivileges(err)
            elif status_code == 429:
                raise GeocoderRateLimited(err)
            elif status_code == 503:
                raise GeocoderUnavailable(err)
            else:
                raise GeocoderServiceError(err)
        resources = doc['resourceSets'][0]['resources']
        if resources is None or not len(resources):
            return None

        def parse_resource(resource):
            if False:
                return 10
            '\n            Parse each return object.\n            '
            stripchars = ', \n'
            addr = resource['address']
            address = addr.get('addressLine', '').strip(stripchars)
            city = addr.get('locality', '').strip(stripchars)
            state = addr.get('adminDistrict', '').strip(stripchars)
            zipcode = addr.get('postalCode', '').strip(stripchars)
            country = addr.get('countryRegion', '').strip(stripchars)
            city_state = join_filter(', ', [city, state])
            place = join_filter(' ', [city_state, zipcode])
            location = join_filter(', ', [address, place, country])
            latitude = resource['point']['coordinates'][0] or None
            longitude = resource['point']['coordinates'][1] or None
            if latitude and longitude:
                latitude = float(latitude)
                longitude = float(longitude)
            return Location(location, (latitude, longitude), resource)
        if exactly_one:
            return parse_resource(resources[0])
        else:
            return [parse_resource(resource) for resource in resources]