import collections.abc
from functools import partial
from urllib.parse import urlencode
from geopy.geocoders.base import DEFAULT_SENTINEL, Geocoder
from geopy.location import Location
from geopy.util import join_filter, logger
__all__ = ('Geolake',)

class Geolake(Geocoder):
    """Geocoder using the Geolake API.

    Documentation at:
        https://geolake.com/docs/api

    Terms of Service at:
        https://geolake.com/terms-of-use
    """
    structured_query_params = {'country', 'state', 'city', 'zipcode', 'street', 'address', 'houseNumber', 'subNumber'}
    api_path = '/v1/geocode'

    def __init__(self, api_key, *, domain='api.geolake.com', scheme=None, timeout=DEFAULT_SENTINEL, proxies=DEFAULT_SENTINEL, user_agent=None, ssl_context=DEFAULT_SENTINEL, adapter_factory=None):
        if False:
            print('Hello World!')
        "\n\n        :param str api_key: The API key required by Geolake\n            to perform geocoding requests. You can get your key here:\n            https://geolake.com/\n\n        :param str domain: Currently it is ``'api.geolake.com'``, can\n            be changed for testing purposes.\n\n        :param str scheme:\n            See :attr:`geopy.geocoders.options.default_scheme`.\n\n        :param int timeout:\n            See :attr:`geopy.geocoders.options.default_timeout`.\n\n        :param dict proxies:\n            See :attr:`geopy.geocoders.options.default_proxies`.\n\n        :param str user_agent:\n            See :attr:`geopy.geocoders.options.default_user_agent`.\n\n        :type ssl_context: :class:`ssl.SSLContext`\n        :param ssl_context:\n            See :attr:`geopy.geocoders.options.default_ssl_context`.\n\n        :param callable adapter_factory:\n            See :attr:`geopy.geocoders.options.default_adapter_factory`.\n\n            .. versionadded:: 2.0\n\n        "
        super().__init__(scheme=scheme, timeout=timeout, proxies=proxies, user_agent=user_agent, ssl_context=ssl_context, adapter_factory=adapter_factory)
        self.api_key = api_key
        self.domain = domain.strip('/')
        self.api = '%s://%s%s' % (self.scheme, self.domain, self.api_path)

    def geocode(self, query, *, country_codes=None, exactly_one=True, timeout=DEFAULT_SENTINEL):
        if False:
            print('Hello World!')
        "\n        Return a location point by address.\n\n        :param query: The address or query you wish to geocode.\n\n            For a structured query, provide a dictionary whose keys\n            are one of: `country`, `state`, `city`, `zipcode`, `street`, `address`,\n            `houseNumber` or `subNumber`.\n        :type query: str or dict\n\n        :param country_codes: Provides the geocoder with a list\n            of country codes that the query may reside in. This value will\n            limit the geocoder to the supplied countries. The country code\n            is a 2 character code as defined by the ISO-3166-1 alpha-2\n            standard (e.g. ``FR``). Multiple countries can be specified with\n            a Python list.\n\n        :type country_codes: str or list\n\n        :param bool exactly_one: Return one result or a list of one result.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder's initialization.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n\n        "
        if isinstance(query, collections.abc.Mapping):
            params = {key: val for (key, val) in query.items() if key in self.structured_query_params}
            params['api_key'] = self.api_key
        else:
            params = {'api_key': self.api_key, 'q': query}
        if not country_codes:
            country_codes = []
        if isinstance(country_codes, str):
            country_codes = [country_codes]
        if country_codes:
            params['countryCodes'] = ','.join(country_codes)
        url = '?'.join((self.api, urlencode(params)))
        logger.debug('%s.geocode: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def _parse_json(self, page, exactly_one):
        if False:
            return 10
        'Returns location, (latitude, longitude) from json feed.'
        if not page.get('success'):
            return None
        latitude = page['latitude']
        longitude = page['longitude']
        address = self._get_address(page)
        result = Location(address, (latitude, longitude), page)
        if exactly_one:
            return result
        else:
            return [result]

    def _get_address(self, page):
        if False:
            while True:
                i = 10
        '\n        Returns address string from page dictionary\n        :param page: dict\n        :return: str\n        '
        place = page.get('place')
        address_city = place.get('city')
        address_country_code = place.get('countryCode')
        address = join_filter(', ', [address_city, address_country_code])
        return address