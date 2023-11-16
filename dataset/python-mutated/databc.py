from functools import partial
from urllib.parse import urlencode
from geopy.exc import GeocoderQueryError
from geopy.geocoders.base import DEFAULT_SENTINEL, Geocoder
from geopy.location import Location
from geopy.util import logger
__all__ = ('DataBC',)

class DataBC(Geocoder):
    """Geocoder using the Physical Address Geocoder from DataBC.

    Documentation at:
        https://github.com/bcgov/ols-geocoder/blob/gh-pages/geocoder-developer-guide.md
    """
    geocode_path = '/addresses.geojson'

    def __init__(self, *, scheme=None, timeout=DEFAULT_SENTINEL, proxies=DEFAULT_SENTINEL, user_agent=None, ssl_context=DEFAULT_SENTINEL, adapter_factory=None, domain='geocoder.api.gov.bc.ca'):
        if False:
            i = 10
            return i + 15
        '\n\n        :param str scheme:\n            See :attr:`geopy.geocoders.options.default_scheme`.\n\n        :param int timeout:\n            See :attr:`geopy.geocoders.options.default_timeout`.\n\n        :param dict proxies:\n            See :attr:`geopy.geocoders.options.default_proxies`.\n\n        :param str user_agent:\n            See :attr:`geopy.geocoders.options.default_user_agent`.\n\n        :type ssl_context: :class:`ssl.SSLContext`\n        :param ssl_context:\n            See :attr:`geopy.geocoders.options.default_ssl_context`.\n\n        :param callable adapter_factory:\n            See :attr:`geopy.geocoders.options.default_adapter_factory`.\n\n            .. versionadded:: 2.0\n\n        :param str domain: base api domain\n\n            .. versionadded:: 2.4\n        '
        super().__init__(scheme=scheme, timeout=timeout, proxies=proxies, user_agent=user_agent, ssl_context=ssl_context, adapter_factory=adapter_factory)
        self.api = '%s://%s%s' % (self.scheme, domain, self.geocode_path)

    def geocode(self, query, *, max_results=25, set_back=0, location_descriptor='any', exactly_one=True, timeout=DEFAULT_SENTINEL):
        if False:
            print('Hello World!')
        "\n        Return a location point by address.\n\n        :param str query: The address or query you wish to geocode.\n\n        :param int max_results: The maximum number of resutls to request.\n\n        :param float set_back: The distance to move the accessPoint away\n            from the curb (in meters) and towards the interior of the parcel.\n            location_descriptor must be set to accessPoint for set_back to\n            take effect.\n\n        :param str location_descriptor: The type of point requested. It\n            can be any, accessPoint, frontDoorPoint, parcelPoint,\n            rooftopPoint and routingPoint.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder's initialization.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n        "
        params = {'addressString': query}
        if set_back != 0:
            params['setBack'] = set_back
        if location_descriptor not in ['any', 'accessPoint', 'frontDoorPoint', 'parcelPoint', 'rooftopPoint', 'routingPoint']:
            raise GeocoderQueryError('You did not provided a location_descriptor the webservice can consume. It should be any, accessPoint, frontDoorPoint, parcelPoint, rooftopPoint or routingPoint.')
        params['locationDescriptor'] = location_descriptor
        if exactly_one:
            max_results = 1
        params['maxResults'] = max_results
        url = '?'.join((self.api, urlencode(params)))
        logger.debug('%s.geocode: %s', self.__class__.__name__, url)
        callback = partial(self._parse_json, exactly_one=exactly_one)
        return self._call_geocoder(url, callback, timeout=timeout)

    def _parse_json(self, response, exactly_one):
        if False:
            print('Hello World!')
        if not len(response['features']):
            return None
        geocoded = []
        for feature in response['features']:
            geocoded.append(self._parse_feature(feature))
        if exactly_one:
            return geocoded[0]
        return geocoded

    def _parse_feature(self, feature):
        if False:
            print('Hello World!')
        properties = feature['properties']
        coordinates = feature['geometry']['coordinates']
        return Location(properties['fullAddress'], (coordinates[1], coordinates[0]), properties)