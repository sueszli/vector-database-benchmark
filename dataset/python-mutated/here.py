import collections.abc
import json
import warnings
from functools import partial
from urllib.parse import urlencode
from geopy.adapters import AdapterHTTPError
from geopy.exc import ConfigurationError, GeocoderAuthenticationFailure, GeocoderInsufficientPrivileges, GeocoderQueryError, GeocoderRateLimited, GeocoderServiceError, GeocoderUnavailable
from geopy.geocoders.base import DEFAULT_SENTINEL, ERROR_CODE_MAP, Geocoder
from geopy.location import Location
from geopy.util import join_filter, logger
__all__ = ('Here', 'HereV7')

class Here(Geocoder):
    """Geocoder using the HERE Geocoder API.

    Documentation at:
        https://developer.here.com/documentation/geocoder/

    .. attention::
        This class uses a v6 API which is in maintenance mode.
        Consider using the newer :class:`.HereV7` class.
    """
    structured_query_params = {'city', 'county', 'district', 'country', 'state', 'street', 'housenumber', 'postalcode'}
    geocode_path = '/6.2/geocode.json'
    reverse_path = '/6.2/reversegeocode.json'

    def __init__(self, *, app_id=None, app_code=None, apikey=None, scheme=None, timeout=DEFAULT_SENTINEL, proxies=DEFAULT_SENTINEL, user_agent=None, ssl_context=DEFAULT_SENTINEL, adapter_factory=None):
        if False:
            while True:
                i = 10
        '\n\n        :param str app_id: Should be a valid HERE Maps APP ID. Will eventually\n            be replaced with APIKEY.\n            See https://developer.here.com/authenticationpage.\n\n            .. attention::\n                App ID and App Code are being replaced by API Keys and OAuth 2.0\n                by HERE. Consider getting an ``apikey`` instead of using\n                ``app_id`` and ``app_code``.\n\n        :param str app_code: Should be a valid HERE Maps APP CODE. Will\n            eventually be replaced with APIKEY.\n            See https://developer.here.com/authenticationpage.\n\n            .. attention::\n                App ID and App Code are being replaced by API Keys and OAuth 2.0\n                by HERE. Consider getting an ``apikey`` instead of using\n                ``app_id`` and ``app_code``.\n\n        :param str apikey: Should be a valid HERE Maps APIKEY. These keys were\n            introduced in December 2019 and will eventually replace the legacy\n            APP CODE/APP ID pairs which are already no longer available for new\n            accounts (but still work for old accounts).\n            More authentication details are available at\n            https://developer.here.com/blog/announcing-two-new-authentication-types.\n            See https://developer.here.com/authenticationpage.\n\n        :param str scheme:\n            See :attr:`geopy.geocoders.options.default_scheme`.\n\n        :param int timeout:\n            See :attr:`geopy.geocoders.options.default_timeout`.\n\n        :param dict proxies:\n            See :attr:`geopy.geocoders.options.default_proxies`.\n\n        :param str user_agent:\n            See :attr:`geopy.geocoders.options.default_user_agent`.\n\n        :type ssl_context: :class:`ssl.SSLContext`\n        :param ssl_context:\n            See :attr:`geopy.geocoders.options.default_ssl_context`.\n\n        :param callable adapter_factory:\n            See :attr:`geopy.geocoders.options.default_adapter_factory`.\n\n            .. versionadded:: 2.0\n        '
        super().__init__(scheme=scheme, timeout=timeout, proxies=proxies, user_agent=user_agent, ssl_context=ssl_context, adapter_factory=adapter_factory)
        is_apikey = bool(apikey)
        is_app_code = app_id and app_code
        if not is_apikey and (not is_app_code):
            raise ConfigurationError('HERE geocoder requires authentication, either `apikey` or `app_id`+`app_code` must be set')
        if is_app_code:
            warnings.warn('Since December 2019 HERE provides two new authentication methods `API Key` and `OAuth 2.0`. `app_id`+`app_code` is deprecated and might eventually be phased out. Consider switching to `apikey`, which geopy supports. See https://developer.here.com/blog/announcing-two-new-authentication-types', UserWarning, stacklevel=2)
        self.app_id = app_id
        self.app_code = app_code
        self.apikey = apikey
        domain = 'ls.hereapi.com' if is_apikey else 'api.here.com'
        self.api = '%s://geocoder.%s%s' % (self.scheme, domain, self.geocode_path)
        self.reverse_api = '%s://reverse.geocoder.%s%s' % (self.scheme, domain, self.reverse_path)

    def geocode(self, query, *, bbox=None, mapview=None, exactly_one=True, maxresults=None, pageinformation=None, language=None, additional_data=False, timeout=DEFAULT_SENTINEL):
        if False:
            return 10
        '\n        Return a location point by address.\n\n        This implementation supports only a subset of all available parameters.\n        A list of all parameters of the pure REST API is available here:\n        https://developer.here.com/documentation/geocoder/topics/resource-geocode.html\n\n        :param query: The address or query you wish to geocode.\n\n            For a structured query, provide a dictionary whose keys\n            are one of: `city`, `county`, `district`, `country`, `state`,\n            `street`, `housenumber`, or `postalcode`.\n        :type query: str or dict\n\n        :param bbox: A type of spatial filter, limits the search for any other attributes\n            in the request. Specified by two coordinate (lat/lon)\n            pairs -- corners of the box. `The bbox search is currently similar\n            to mapview but it is not extended` (cited from the REST API docs).\n            Relevant global results are also returned.\n            Example: ``[Point(22, 180), Point(-22, -180)]``.\n        :type bbox: list or tuple of 2 items of :class:`geopy.point.Point` or\n            ``(latitude, longitude)`` or ``"%(latitude)s, %(longitude)s"``.\n\n        :param mapview: The app\'s viewport, given as two coordinate pairs, specified\n            by two lat/lon pairs -- corners of the bounding box,\n            respectively. Matches from within the set map view plus an extended area\n            are ranked highest. Relevant global results are also returned.\n            Example: ``[Point(22, 180), Point(-22, -180)]``.\n        :type mapview: list or tuple of 2 items of :class:`geopy.point.Point` or\n            ``(latitude, longitude)`` or ``"%(latitude)s, %(longitude)s"``.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int maxresults: Defines the maximum number of items in the\n            response structure. If not provided and there are multiple results\n            the HERE API will return 10 results by default. This will be reset\n            to one if ``exactly_one`` is True.\n\n        :param int pageinformation: A key which identifies the page to be returned\n            when the response is separated into multiple pages. Only useful when\n            ``maxresults`` is also provided.\n\n        :param str language: Affects the language of the response,\n            must be a RFC 4647 language code, e.g. \'en-US\'.\n\n        :param str additional_data: A string with key-value pairs as described on\n            https://developer.here.com/documentation/geocoder/topics/resource-params-additional.html.\n            These will be added as one query parameter to the URL.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder\'s initialization.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        '
        if isinstance(query, collections.abc.Mapping):
            params = {key: val for (key, val) in query.items() if key in self.structured_query_params}
        else:
            params = {'searchtext': query}
        if bbox:
            params['bbox'] = self._format_bounding_box(bbox, '%(lat2)s,%(lon1)s;%(lat1)s,%(lon2)s')
        if mapview:
            params['mapview'] = self._format_bounding_box(mapview, '%(lat2)s,%(lon1)s;%(lat1)s,%(lon2)s')
        if pageinformation:
            params['pageinformation'] = pageinformation
        if maxresults:
            params['maxresults'] = maxresults
        if exactly_one:
            params['maxresults'] = 1
        if language:
            params['language'] = language
        if additional_data:
            params['additionaldata'] = additional_data
        if self.apikey:
            params['apiKey'] = self.apikey
        else:
            params['app_id'] = self.app_id
            params['app_code'] = self.app_code
        url = '?'.join((self.api, urlencode(params)))
        logger.debug('%s.geocode: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def reverse(self, query, *, radius=None, exactly_one=True, maxresults=None, pageinformation=None, language=None, mode='retrieveAddresses', timeout=DEFAULT_SENTINEL):
        if False:
            print('Hello World!')
        '\n        Return an address by location point.\n\n        This implementation supports only a subset of all available parameters.\n        A list of all parameters of the pure REST API is available here:\n        https://developer.here.com/documentation/geocoder/topics/resource-reverse-geocode.html\n\n        :param query: The coordinates for which you wish to obtain the\n            closest human-readable addresses.\n        :type query: :class:`geopy.point.Point`, list or tuple of ``(latitude,\n            longitude)``, or string as ``"%(latitude)s, %(longitude)s"``.\n\n        :param float radius: Proximity radius in meters.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int maxresults: Defines the maximum number of items in the\n            response structure. If not provided and there are multiple results\n            the HERE API will return 10 results by default. This will be reset\n            to one if ``exactly_one`` is True.\n\n        :param int pageinformation: A key which identifies the page to be returned\n            when the response is separated into multiple pages. Only useful when\n            ``maxresults`` is also provided.\n\n        :param str language: Affects the language of the response,\n            must be a RFC 4647 language code, e.g. \'en-US\'.\n\n        :param str mode: Affects the type of returned response items, must be\n            one of: \'retrieveAddresses\' (default), \'retrieveAreas\', \'retrieveLandmarks\',\n            \'retrieveAll\', or \'trackPosition\'. See online documentation for more\n            information.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder\'s initialization.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        '
        point = self._coerce_point_to_string(query)
        params = {'mode': mode, 'prox': point}
        if radius is not None:
            params['prox'] = '%s,%s' % (params['prox'], float(radius))
        if pageinformation:
            params['pageinformation'] = pageinformation
        if maxresults:
            params['maxresults'] = maxresults
        if exactly_one:
            params['maxresults'] = 1
        if language:
            params['language'] = language
        if self.apikey:
            params['apiKey'] = self.apikey
        else:
            params['app_id'] = self.app_id
            params['app_code'] = self.app_code
        url = '%s?%s' % (self.reverse_api, urlencode(params))
        logger.debug('%s.reverse: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def _parse_json(self, doc, exactly_one=True):
        if False:
            for i in range(10):
                print('nop')
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
        try:
            resources = doc['Response']['View'][0]['Result']
        except IndexError:
            resources = None
        if not resources:
            return None

        def parse_resource(resource):
            if False:
                return 10
            '\n            Parse each return object.\n            '
            stripchars = ', \n'
            addr = resource['Location']['Address']
            address = addr.get('Label', '').strip(stripchars)
            city = addr.get('City', '').strip(stripchars)
            state = addr.get('State', '').strip(stripchars)
            zipcode = addr.get('PostalCode', '').strip(stripchars)
            country = addr.get('Country', '').strip(stripchars)
            city_state = join_filter(', ', [city, state])
            place = join_filter(' ', [city_state, zipcode])
            location = join_filter(', ', [address, place, country])
            display_pos = resource['Location']['DisplayPosition']
            latitude = float(display_pos['Latitude'])
            longitude = float(display_pos['Longitude'])
            return Location(location, (latitude, longitude), resource)
        if exactly_one:
            return parse_resource(resources[0])
        else:
            return [parse_resource(resource) for resource in resources]

class HereV7(Geocoder):
    """Geocoder using the HERE Geocoding & Search v7 API.

    Documentation at:
        https://developer.here.com/documentation/geocoding-search-api/

    Terms of Service at:
        https://legal.here.com/en-gb/terms

    .. versionadded:: 2.2
    """
    structured_query_params = {'country', 'state', 'county', 'city', 'district', 'street', 'houseNumber', 'postalCode'}
    geocode_path = '/v1/geocode'
    reverse_path = '/v1/revgeocode'

    def __init__(self, apikey, *, scheme=None, timeout=DEFAULT_SENTINEL, proxies=DEFAULT_SENTINEL, user_agent=None, ssl_context=DEFAULT_SENTINEL, adapter_factory=None, domain='search.hereapi.com'):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        :param str apikey: Should be a valid HERE Maps apikey.\n            A project can be created at\n            https://developer.here.com/projects.\n\n        :param str scheme:\n            See :attr:`geopy.geocoders.options.default_scheme`.\n\n        :param int timeout:\n            See :attr:`geopy.geocoders.options.default_timeout`.\n\n        :param dict proxies:\n            See :attr:`geopy.geocoders.options.default_proxies`.\n\n        :param str user_agent:\n            See :attr:`geopy.geocoders.options.default_user_agent`.\n\n        :type ssl_context: :class:`ssl.SSLContext`\n        :param ssl_context:\n            See :attr:`geopy.geocoders.options.default_ssl_context`.\n\n        :param callable adapter_factory:\n            See :attr:`geopy.geocoders.options.default_adapter_factory`.\n\n        :param str domain: base api domain\n\n            .. versionadded:: 2.4\n        '
        super().__init__(scheme=scheme, timeout=timeout, proxies=proxies, user_agent=user_agent, ssl_context=ssl_context, adapter_factory=adapter_factory)
        self.apikey = apikey
        self.api = '%s://geocode.%s%s' % (self.scheme, domain, self.geocode_path)
        self.reverse_api = '%s://revgeocode.%s%s' % (self.scheme, domain, self.reverse_path)

    def geocode(self, query=None, *, components=None, at=None, countries=None, language=None, limit=None, exactly_one=True, timeout=DEFAULT_SENTINEL):
        if False:
            while True:
                i = 10
        '\n        Return a location point by address.\n\n        :param str query: The address or query you wish to geocode. Optional,\n            if ``components`` param is set.\n\n        :param dict components: A structured query. Can be used along with\n            the free-text ``query``. Should be a dictionary whose keys\n            are one of:\n            `country`, `state`, `county`, `city`, `district`, `street`,\n            `houseNumber`, `postalCode`.\n\n        :param at: The center of the search context.\n        :type at: :class:`geopy.point.Point`, list or tuple of ``(latitude,\n            longitude)``, or string as ``"%(latitude)s, %(longitude)s"``.\n\n        :param list countries: A list of country codes specified in\n            `ISO 3166-1 alpha-3 <https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3>`_\n            format, e.g. ``[\'USA\', \'CAN\']``.\n            This is a hard filter.\n\n        :param str language: Affects the language of the response,\n            must be a BCP 47 compliant language code, e.g. ``en-US``.\n\n        :param int limit: Defines the maximum number of items in the\n            response structure. If not provided and there are multiple results\n            the HERE API will return 20 results by default. This will be reset\n            to one if ``exactly_one`` is True.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder\'s initialization.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        '
        params = {'apiKey': self.apikey}
        if query:
            params['q'] = query
        if components:
            parts = ['{}={}'.format(key, val) for (key, val) in components.items() if key in self.structured_query_params]
            if not parts:
                raise GeocoderQueryError('`components` dict must not be empty')
            for pair in parts:
                if ';' in pair:
                    raise GeocoderQueryError("';' must not be used in values of the structured query. Offending pair: {!r}".format(pair))
            params['qq'] = ';'.join(parts)
        if at:
            point = self._coerce_point_to_string(at, output_format='%(lat)s,%(lon)s')
            params['at'] = point
        if countries:
            params['in'] = 'countryCode:' + ','.join(countries)
        if language:
            params['lang'] = language
        if limit:
            params['limit'] = limit
        if exactly_one:
            params['limit'] = 1
        url = '?'.join((self.api, urlencode(params)))
        logger.debug('%s.geocode: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def reverse(self, query, *, language=None, limit=None, exactly_one=True, timeout=DEFAULT_SENTINEL):
        if False:
            while True:
                i = 10
        '\n        Return an address by location point.\n\n        :param query: The coordinates for which you wish to obtain the\n            closest human-readable addresses.\n        :type query: :class:`geopy.point.Point`, list or tuple of ``(latitude,\n            longitude)``, or string as ``"%(latitude)s, %(longitude)s"``.\n\n        :param str language: Affects the language of the response,\n            must be a BCP 47 compliant language code, e.g. ``en-US``.\n\n        :param int limit: Maximum number of results to be returned.\n            This will be reset to one if ``exactly_one`` is True.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder\'s initialization.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        '
        params = {'at': self._coerce_point_to_string(query, output_format='%(lat)s,%(lon)s'), 'apiKey': self.apikey}
        if language:
            params['lang'] = language
        if limit:
            params['limit'] = limit
        if exactly_one:
            params['limit'] = 1
        url = '%s?%s' % (self.reverse_api, urlencode(params))
        logger.debug('%s.reverse: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def _parse_json(self, doc, exactly_one=True):
        if False:
            i = 10
            return i + 15
        resources = doc['items']
        if not resources:
            return None

        def parse_resource(resource):
            if False:
                return 10
            '\n            Parse each return object.\n            '
            location = resource['title']
            position = resource['position']
            (latitude, longitude) = (position['lat'], position['lng'])
            return Location(location, (latitude, longitude), resource)
        if exactly_one:
            return parse_resource(resources[0])
        else:
            return [parse_resource(resource) for resource in resources]

    def _geocoder_exception_handler(self, error):
        if False:
            print('Hello World!')
        if not isinstance(error, AdapterHTTPError):
            return
        if error.status_code is None or error.text is None:
            return
        try:
            body = json.loads(error.text)
        except ValueError:
            message = error.text
        else:
            message = body.get('title') or body.get('error_description') or error.text
        exc_cls = ERROR_CODE_MAP.get(error.status_code, GeocoderServiceError)
        raise exc_cls(message) from error