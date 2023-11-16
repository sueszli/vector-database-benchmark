import warnings
import xml.etree.ElementTree as ET
from functools import partial
from urllib.parse import urlencode
from geopy.exc import GeocoderQueryError
from geopy.geocoders.base import DEFAULT_SENTINEL, Geocoder
from geopy.location import Location
from geopy.util import logger
__all__ = ('IGNFrance',)

class IGNFrance(Geocoder):
    """Geocoder using the IGN France GeoCoder OpenLS API.

    Documentation at:
        https://geoservices.ign.fr/services-web-essentiels
    """
    xml_request = '<?xml version="1.0" encoding="UTF-8"?>\n    <XLS version="1.2"\n        xmlns="http://www.opengis.net/xls"\n        xmlns:gml="http://www.opengis.net/gml"\n        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n        xsi:schemaLocation="http://www.opengis.net/xls\n        http://schemas.opengis.net/ols/1.2/olsAll.xsd">\n        <RequestHeader srsName="epsg:4326"/>\n        <Request methodName="{method_name}"\n                 maximumResponses="{maximum_responses}"\n                 requestID=""\n                 version="1.2">\n            {sub_request}\n        </Request>\n    </XLS>'
    api_path = '/essentiels/geoportail/ols'

    def __init__(self, api_key=None, *, username=None, password=None, referer=None, domain='wxs.ign.fr', scheme=None, timeout=DEFAULT_SENTINEL, proxies=DEFAULT_SENTINEL, user_agent=None, ssl_context=DEFAULT_SENTINEL, adapter_factory=None):
        if False:
            return 10
        "\n\n        :param str api_key: Not used.\n\n            .. deprecated:: 2.3\n                IGNFrance geocoding methods no longer accept or require\n                authentication, see `<https://geoservices.ign.fr/actualites/2021-10-04-evolution-des-modalites-dacces-aux-services-web>`_.\n                This parameter is scheduled for removal in geopy 3.0.\n\n        :param str username: Not used.\n\n            .. deprecated:: 2.3\n                See the `api_key` deprecation note.\n\n        :param str password: Not used.\n\n            .. deprecated:: 2.3\n                See the `api_key` deprecation note.\n\n        :param str referer: Not used.\n\n            .. deprecated:: 2.3\n                See the `api_key` deprecation note.\n\n        :param str domain: Currently it is ``'wxs.ign.fr'``, can\n            be changed for testing purposes for developer API\n            e.g ``'gpp3-wxs.ign.fr'`` at the moment.\n\n        :param str scheme:\n            See :attr:`geopy.geocoders.options.default_scheme`.\n\n        :param int timeout:\n            See :attr:`geopy.geocoders.options.default_timeout`.\n\n        :param dict proxies:\n            See :attr:`geopy.geocoders.options.default_proxies`.\n\n        :param str user_agent:\n            See :attr:`geopy.geocoders.options.default_user_agent`.\n\n        :type ssl_context: :class:`ssl.SSLContext`\n        :param ssl_context:\n            See :attr:`geopy.geocoders.options.default_ssl_context`.\n\n        :param callable adapter_factory:\n            See :attr:`geopy.geocoders.options.default_adapter_factory`.\n\n            .. versionadded:: 2.0\n        "
        super().__init__(scheme=scheme, timeout=timeout, proxies=proxies, user_agent=user_agent, ssl_context=ssl_context, adapter_factory=adapter_factory)
        if api_key or username or password or referer:
            warnings.warn('IGNFrance no longer accepts or requires authentication, so api_key, username, password and referer are not used anymore. These arguments should be removed. In geopy 3 these options will be removed, causing an error instead of this warning.', DeprecationWarning, stacklevel=2)
        self.domain = domain.strip('/')
        api_path = self.api_path
        self.api = '%s://%s%s' % (self.scheme, self.domain, api_path)

    def geocode(self, query, *, query_type='StreetAddress', maximum_responses=25, is_freeform=False, filtering=None, exactly_one=True, timeout=DEFAULT_SENTINEL):
        if False:
            print('Hello World!')
        "\n        Return a location point by address.\n\n        :param str query: The query string to be geocoded.\n\n        :param str query_type: The type to provide for geocoding. It can be\n            `PositionOfInterest`, `StreetAddress` or `CadastralParcel`.\n            `StreetAddress` is the default choice if none provided.\n\n        :param int maximum_responses: The maximum number of responses\n            to ask to the API in the query body.\n\n        :param str is_freeform: Set if return is structured with\n            freeform structure or a more structured returned.\n            By default, value is False.\n\n        :param str filtering: Provide string that help setting geocoder\n            filter. It contains an XML string. See examples in documentation\n            and ignfrance.py file in directory tests.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder's initialization.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n\n        "
        if query_type not in ['PositionOfInterest', 'StreetAddress', 'CadastralParcel']:
            raise GeocoderQueryError("You did not provided a query_type the\n            webservice can consume. It should be PositionOfInterest,\n            'StreetAddress or CadastralParcel")
        if query_type == 'CadastralParcel' and len(query.strip()) != 14:
            raise GeocoderQueryError('You must send a string of fourteen\n                characters long to match the cadastre required code')
        sub_request = '\n                <GeocodeRequest returnFreeForm="{is_freeform}">\n                    <Address countryCode="{query_type}">\n                        <freeFormAddress>{query}</freeFormAddress>\n                        {filtering}\n                    </Address>\n                </GeocodeRequest>\n        '
        xml_request = self.xml_request.format(method_name='LocationUtilityService', sub_request=sub_request, maximum_responses=maximum_responses)
        if is_freeform:
            is_freeform = 'true'
        else:
            is_freeform = 'false'
        if filtering is None:
            filtering = ''
        request_string = xml_request.format(is_freeform=is_freeform, query=query, query_type=query_type, filtering=filtering)
        params = {'xls': request_string}
        url = '?'.join((self.api, urlencode(params)))
        logger.debug('%s.geocode: %s', self.__class__.__name__, url)
        callback = partial(self._parse_xml, is_freeform=is_freeform, exactly_one=exactly_one)
        return self._request_raw_content(url, callback, timeout=timeout)

    def reverse(self, query, *, reverse_geocode_preference=('StreetAddress',), maximum_responses=25, filtering='', exactly_one=True, timeout=DEFAULT_SENTINEL):
        if False:
            print('Hello World!')
        '\n        Return an address by location point.\n\n        :param query: The coordinates for which you wish to obtain the\n            closest human-readable addresses.\n        :type query: :class:`geopy.point.Point`, list or tuple of ``(latitude,\n            longitude)``, or string as ``"%(latitude)s, %(longitude)s"``.\n\n        :param list reverse_geocode_preference: Enable to set expected results\n            type. It can be `StreetAddress` or `PositionOfInterest`.\n            Default is set to `StreetAddress`.\n\n        :param int maximum_responses: The maximum number of responses\n            to ask to the API in the query body.\n\n        :param str filtering: Provide string that help setting geocoder\n            filter. It contains an XML string. See examples in documentation\n            and ignfrance.py file in directory tests.\n\n        :param bool exactly_one: Return one result or a list of results, if\n            available.\n\n        :param int timeout: Time, in seconds, to wait for the geocoding service\n            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`\n            exception. Set this only if you wish to override, on this call\n            only, the value set during the geocoder\'s initialization.\n\n        :rtype: ``None``, :class:`geopy.location.Location` or a list of them, if\n            ``exactly_one=False``.\n\n        '
        sub_request = '\n            <ReverseGeocodeRequest>\n                {reverse_geocode_preference}\n                <Position>\n                  <gml:Point>\n                    <gml:pos>{query}</gml:pos>\n                  </gml:Point>\n                  {filtering}\n                </Position>\n            </ReverseGeocodeRequest>\n        '
        xml_request = self.xml_request.format(method_name='ReverseGeocodeRequest', sub_request=sub_request, maximum_responses=maximum_responses)
        for pref in reverse_geocode_preference:
            if pref not in ('StreetAddress', 'PositionOfInterest'):
                raise GeocoderQueryError('`reverse_geocode_preference` must contain one or more of: StreetAddress, PositionOfInterest')
        point = self._coerce_point_to_string(query, '%(lat)s %(lon)s')
        reverse_geocode_preference = '\n'.join(('<ReverseGeocodePreference>%s</ReverseGeocodePreference>' % pref for pref in reverse_geocode_preference))
        request_string = xml_request.format(maximum_responses=maximum_responses, query=point, reverse_geocode_preference=reverse_geocode_preference, filtering=filtering)
        url = '?'.join((self.api, urlencode({'xls': request_string})))
        logger.debug('%s.reverse: %s', self.__class__.__name__, url)
        callback = partial(self._parse_xml, exactly_one=exactly_one, is_reverse=True, is_freeform='false')
        return self._request_raw_content(url, callback, timeout=timeout)

    def _parse_xml(self, page, is_reverse=False, is_freeform=False, exactly_one=True):
        if False:
            i = 10
            return i + 15
        '\n        Returns location, (latitude, longitude) from XML feed\n        and transform to json\n        '
        tree = ET.fromstring(page.encode('utf-8'))

        def remove_namespace(doc, namespace):
            if False:
                i = 10
                return i + 15
            'Remove namespace in the document in place.'
            ns = '{%s}' % namespace
            nsl = len(ns)
            for elem in doc.iter():
                if elem.tag.startswith(ns):
                    elem.tag = elem.tag[nsl:]
        remove_namespace(tree, 'http://www.opengis.net/gml')
        remove_namespace(tree, 'http://www.opengis.net/xls')
        remove_namespace(tree, 'http://www.opengis.net/xlsext')
        places = self._xml_to_json_places(tree, is_reverse=is_reverse)
        if not places:
            return None
        if exactly_one:
            return self._parse_place(places[0], is_freeform=is_freeform)
        else:
            return [self._parse_place(place, is_freeform=is_freeform) for place in places]

    def _xml_to_json_places(self, tree, is_reverse=False):
        if False:
            return 10
        '\n        Transform the xml ElementTree due to XML webservice return to json\n        '
        select_multi = 'GeocodedAddress' if not is_reverse else 'ReverseGeocodedLocation'
        adresses = tree.findall('.//' + select_multi)
        places = []
        sel_pl = './/Address/Place[@type="{}"]'
        for adr in adresses:
            el = {}
            el['pos'] = adr.find('./Point/pos')
            el['street'] = adr.find('.//Address/StreetAddress/Street')
            el['freeformaddress'] = adr.find('.//Address/freeFormAddress')
            el['municipality'] = adr.find(sel_pl.format('Municipality'))
            el['numero'] = adr.find(sel_pl.format('Numero'))
            el['feuille'] = adr.find(sel_pl.format('Feuille'))
            el['section'] = adr.find(sel_pl.format('Section'))
            el['departement'] = adr.find(sel_pl.format('Departement'))
            el['commune_absorbee'] = adr.find(sel_pl.format('CommuneAbsorbee'))
            el['commune'] = adr.find(sel_pl.format('Commune'))
            el['insee'] = adr.find(sel_pl.format('INSEE'))
            el['qualite'] = adr.find(sel_pl.format('Qualite'))
            el['territoire'] = adr.find(sel_pl.format('Territoire'))
            el['id'] = adr.find(sel_pl.format('ID'))
            el['id_tr'] = adr.find(sel_pl.format('ID_TR'))
            el['bbox'] = adr.find(sel_pl.format('Bbox'))
            el['nature'] = adr.find(sel_pl.format('Nature'))
            el['postal_code'] = adr.find('.//Address/PostalCode')
            el['extended_geocode_match_code'] = adr.find('.//ExtendedGeocodeMatchCode')
            place = {}

            def testContentAttrib(selector, key):
                if False:
                    i = 10
                    return i + 15
                '\n                Helper to select by attribute and if not attribute,\n                value set to empty string\n                '
                return selector.attrib.get(key, None) if selector is not None else None
            place['accuracy'] = testContentAttrib(adr.find('.//GeocodeMatchCode'), 'accuracy')
            place['match_type'] = testContentAttrib(adr.find('.//GeocodeMatchCode'), 'matchType')
            place['building'] = testContentAttrib(adr.find('.//Address/StreetAddress/Building'), 'number')
            place['search_centre_distance'] = testContentAttrib(adr.find('.//SearchCentreDistance'), 'value')
            for (key, value) in iter(el.items()):
                if value is not None:
                    place[key] = value.text
                else:
                    place[key] = None
            if place['pos']:
                (lat, lng) = place['pos'].split(' ')
                place['lat'] = lat.strip()
                place['lng'] = lng.strip()
            else:
                place['lat'] = place['lng'] = None
            place.pop('pos', None)
            places.append(place)
        return places

    def _request_raw_content(self, url, callback, *, timeout):
        if False:
            for i in range(10):
                print('nop')
        '\n        Send the request to get raw content.\n        '
        return self._call_geocoder(url, callback, timeout=timeout, is_json=False)

    def _parse_place(self, place, is_freeform=None):
        if False:
            return 10
        '\n        Get the location, lat, lng and place from a single json place.\n        '
        if is_freeform == 'true':
            location = place.get('freeformaddress')
        elif place.get('numero'):
            location = place.get('street')
        else:
            location = '%s %s' % (place.get('postal_code', ''), place.get('commune', ''))
            if place.get('street'):
                location = '%s, %s' % (place.get('street', ''), location)
            if place.get('building'):
                location = '%s %s' % (place.get('building', ''), location)
        return Location(location, (place.get('lat'), place.get('lng')), place)