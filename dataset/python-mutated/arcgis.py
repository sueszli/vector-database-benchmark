import json
from functools import partial
from time import time
from urllib.parse import urlencode
from geopy.exc import ConfigurationError, GeocoderAuthenticationFailure, GeocoderServiceError
from geopy.geocoders.base import DEFAULT_SENTINEL, Geocoder, _synchronized
from geopy.location import Location
from geopy.util import logger
__all__ = ('ArcGIS',)
DEFAULT_WKID = 4326

class ArcGIS(Geocoder):
    """Geocoder using the ERSI ArcGIS API.

    Documentation at:
        https://developers.arcgis.com/rest/geocode/api-reference/overview-world-geocoding-service.htm
    """
    _TOKEN_EXPIRED = 498
    auth_path = '/sharing/generateToken'
    geocode_path = '/arcgis/rest/services/World/GeocodeServer/findAddressCandidates'
    reverse_path = '/arcgis/rest/services/World/GeocodeServer/reverseGeocode'

    def __init__(self, username=None, password=None, *, referer=None, token_lifetime=60, scheme=None, timeout=DEFAULT_SENTINEL, proxies=DEFAULT_SENTINEL, user_agent=None, ssl_context=DEFAULT_SENTINEL, adapter_factory=None, auth_domain='www.arcgis.com', domain='geocode.arcgis.com'):
        if False:
            for i in range(10):
                print('nop')
        "\n\n        :param str username: ArcGIS username. Required if authenticated\n            mode is desired.\n\n        :param str password: ArcGIS password. Required if authenticated\n            mode is desired.\n\n        :param str referer: Required if authenticated mode is desired.\n            `Referer` HTTP header to send with each request,\n            e.g., ``'http://www.example.com'``. This is tied to an issued token,\n            so fielding queries for multiple referrers should be handled by\n            having multiple ArcGIS geocoder instances.\n\n        :param int token_lifetime: Desired lifetime, in minutes, of an\n            ArcGIS-issued token.\n\n        :param str scheme:\n            See :attr:`geopy.geocoders.options.default_scheme`.\n            If authenticated mode is in use, it must be ``'https'``.\n\n        :param int timeout:\n            See :attr:`geopy.geocoders.options.default_timeout`.\n\n        :param dict proxies:\n            See :attr:`geopy.geocoders.options.default_proxies`.\n\n        :param str user_agent:\n            See :attr:`geopy.geocoders.options.default_user_agent`.\n\n        :type ssl_context: :class:`ssl.SSLContext`\n        :param ssl_context:\n            See :attr:`geopy.geocoders.options.default_ssl_context`.\n\n        :param callable adapter_factory:\n            See :attr:`geopy.geocoders.options.default_adapter_factory`.\n\n            .. versionadded:: 2.0\n\n        :param str auth_domain: Domain where the target ArcGIS auth service\n            is hosted. Used only in authenticated mode (i.e. username,\n            password and referer are set).\n\n        :param str domain: Domain where the target ArcGIS service\n            is hosted.\n        "
        super().__init__(scheme=scheme, timeout=timeout, proxies=proxies, user_agent=user_agent, ssl_context=ssl_context, adapter_factory=adapter_factory)
        if username or password or referer:
            if not (username and password and referer):
                raise ConfigurationError('Authenticated mode requires username, password, and referer')
            if self.scheme != 'https':
                raise ConfigurationError("Authenticated mode requires scheme of 'https'")
        self.username = username
        self.password = password
        self.referer = referer
        self.auth_domain = auth_domain.strip('/')
        self.auth_api = '%s://%s%s' % (self.scheme, self.auth_domain, self.auth_path)
        self.token_lifetime = token_lifetime * 60
        self.domain = domain.strip('/')
        self.api = '%s://%s%s' % (self.scheme, self.domain, self.geocode_path)
        self.reverse_api = '%s://%s%s' % (self.scheme, self.domain, self.reverse_path)
        self.token = None
        self.token_expiry = None

    def geocode(self, query, *, exactly_one=True, timeout=DEFAULT_SENTINEL, out_fields=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a location point by address.\n\n        :param str query: The address or query you wish to geocode.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder\'s initialization.\n\n        :param out_fields: A list of output fields to be returned in the\n            attributes field of the raw data. This can be either a python\n            list/tuple of fields or a comma-separated string. See\n            https://developers.arcgis.com/rest/geocode/api-reference/geocoding-service-output.htm\n            for a list of supported output fields. If you want to return all\n            supported output fields, set ``out_fields="*"``.\n        :type out_fields: str or iterable\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        '
        params = {'singleLine': query, 'f': 'json'}
        if exactly_one:
            params['maxLocations'] = 1
        if out_fields is not None:
            if isinstance(out_fields, str):
                params['outFields'] = out_fields
            else:
                params['outFields'] = ','.join(out_fields)
        url = '?'.join((self.api, urlencode(params)))
        logger.debug('%s.geocode: %s', self.__class__.__name__, url)
        callback = partial(self._parse_geocode, exactly_one=exactly_one)
        return self._authenticated_call_geocoder(url, callback, timeout=timeout)

    def _parse_geocode(self, response, exactly_one):
        if False:
            for i in range(10):
                print('nop')
        if 'error' in response:
            raise GeocoderServiceError(str(response['error']))
        if not len(response['candidates']):
            return None
        geocoded = []
        for resource in response['candidates']:
            geometry = resource['location']
            geocoded.append(Location(resource['address'], (geometry['y'], geometry['x']), resource))
        if exactly_one:
            return geocoded[0]
        return geocoded

    def reverse(self, query, *, exactly_one=True, timeout=DEFAULT_SENTINEL, distance=None):
        if False:
            return 10
        '\n        Return an address by location point.\n\n        :param query: The coordinates for which you wish to obtain the\n            closest human-readable addresses.\n        :type query: :class:`geopy.point.Point`, list or tuple of ``(latitude,\n            longitude)``, or string as ``"%(latitude)s, %(longitude)s"``.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder\'s initialization.\n\n        :param int distance: Distance from the query location, in meters,\n            within which to search. ArcGIS has a default of 100 meters, if not\n            specified.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        '
        location = self._coerce_point_to_string(query, '%(lon)s,%(lat)s')
        wkid = DEFAULT_WKID
        params = {'location': location, 'f': 'json', 'outSR': wkid}
        if distance is not None:
            params['distance'] = distance
        url = '?'.join((self.reverse_api, urlencode(params)))
        logger.debug('%s.reverse: %s', self.__class__.__name__, url)
        callback = partial(self._parse_reverse, exactly_one=exactly_one)
        return self._authenticated_call_geocoder(url, callback, timeout=timeout)

    def _parse_reverse(self, response, exactly_one):
        if False:
            for i in range(10):
                print('nop')
        if not len(response):
            return None
        if 'error' in response:
            if response['error']['code'] == 400:
                try:
                    if 'Unable to find' in response['error']['details'][0]:
                        return None
                except (KeyError, IndexError):
                    pass
            raise GeocoderServiceError(str(response['error']))
        if response['address'].get('Address'):
            address = '%(Address)s, %(City)s, %(Region)s %(Postal)s, %(CountryCode)s' % response['address']
        else:
            address = response['address']['LongLabel']
        location = Location(address, (response['location']['y'], response['location']['x']), response['address'])
        if exactly_one:
            return location
        else:
            return [location]

    def _authenticated_call_geocoder(self, url, parse_callback, *, timeout=DEFAULT_SENTINEL):
        if False:
            i = 10
            return i + 15
        if not self.username:
            return self._call_geocoder(url, parse_callback, timeout=timeout)

        def query_callback():
            if False:
                print('Hello World!')
            call_url = '&'.join((url, urlencode({'token': self.token})))
            headers = {'Referer': self.referer}
            return self._call_geocoder(call_url, partial(maybe_reauthenticate_callback, from_token=self.token), timeout=timeout, headers=headers)

        def maybe_reauthenticate_callback(response, *, from_token):
            if False:
                return 10
            if 'error' in response:
                if response['error']['code'] == self._TOKEN_EXPIRED:
                    return self._refresh_authentication_token(query_retry_callback, timeout=timeout, from_token=from_token)
            return parse_callback(response)

        def query_retry_callback():
            if False:
                print('Hello World!')
            call_url = '&'.join((url, urlencode({'token': self.token})))
            headers = {'Referer': self.referer}
            return self._call_geocoder(call_url, parse_callback, timeout=timeout, headers=headers)
        if self.token is None or int(time()) > self.token_expiry:
            return self._refresh_authentication_token(query_callback, timeout=timeout, from_token=self.token)
        else:
            return query_callback()

    @_synchronized
    def _refresh_authentication_token(self, callback_success, *, timeout, from_token):
        if False:
            print('Hello World!')
        if from_token != self.token:
            return callback_success()
        token_request_arguments = {'username': self.username, 'password': self.password, 'referer': self.referer, 'expiration': self.token_lifetime, 'f': 'json'}
        url = '?'.join((self.auth_api, urlencode(token_request_arguments)))
        logger.debug('%s._refresh_authentication_token: %s', self.__class__.__name__, url)

        def cb(response):
            if False:
                print('Hello World!')
            if 'token' not in response:
                raise GeocoderAuthenticationFailure('Missing token in auth request.Request URL: %s; response JSON: %s' % (url, json.dumps(response)))
            self.token = response['token']
            self.token_expiry = int(time()) + self.token_lifetime
            return callback_success()
        return self._call_geocoder(url, cb, timeout=timeout)