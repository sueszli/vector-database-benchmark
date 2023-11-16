"""Performs requests to the Google Places API."""
import warnings
from googlemaps import convert
PLACES_FIND_FIELDS_BASIC = {'business_status', 'formatted_address', 'geometry', 'geometry/location', 'geometry/location/lat', 'geometry/location/lng', 'geometry/viewport', 'geometry/viewport/northeast', 'geometry/viewport/northeast/lat', 'geometry/viewport/northeast/lng', 'geometry/viewport/southwest', 'geometry/viewport/southwest/lat', 'geometry/viewport/southwest/lng', 'icon', 'name', 'permanently_closed', 'photos', 'place_id', 'plus_code', 'types'}
PLACES_FIND_FIELDS_CONTACT = {'opening_hours'}
PLACES_FIND_FIELDS_ATMOSPHERE = {'price_level', 'rating', 'user_ratings_total'}
PLACES_FIND_FIELDS = PLACES_FIND_FIELDS_BASIC ^ PLACES_FIND_FIELDS_CONTACT ^ PLACES_FIND_FIELDS_ATMOSPHERE
PLACES_DETAIL_FIELDS_BASIC = {'address_component', 'adr_address', 'business_status', 'formatted_address', 'geometry', 'geometry/location', 'geometry/location/lat', 'geometry/location/lng', 'geometry/viewport', 'geometry/viewport/northeast', 'geometry/viewport/northeast/lat', 'geometry/viewport/northeast/lng', 'geometry/viewport/southwest', 'geometry/viewport/southwest/lat', 'geometry/viewport/southwest/lng', 'icon', 'name', 'permanently_closed', 'photo', 'place_id', 'plus_code', 'type', 'url', 'utc_offset', 'vicinity', 'wheelchair_accessible_entrance'}
PLACES_DETAIL_FIELDS_CONTACT = {'formatted_phone_number', 'international_phone_number', 'opening_hours', 'current_opening_hours', 'secondary_opening_hours', 'website'}
PLACES_DETAIL_FIELDS_ATMOSPHERE = {'curbside_pickup', 'delivery', 'dine_in', 'editorial_summary', 'price_level', 'rating', 'reservable', 'review', 'reviews', 'serves_beer', 'serves_breakfast', 'serves_brunch', 'serves_dinner', 'serves_lunch', 'serves_vegetarian_food', 'serves_wine', 'takeout', 'user_ratings_total'}
PLACES_DETAIL_FIELDS = PLACES_DETAIL_FIELDS_BASIC ^ PLACES_DETAIL_FIELDS_CONTACT ^ PLACES_DETAIL_FIELDS_ATMOSPHERE
DEPRECATED_FIELDS = {'permanently_closed', 'review'}
DEPRECATED_FIELDS_MESSAGE = 'Fields, %s, are deprecated. Read more at https://developers.google.com/maps/deprecations.'

def find_place(client, input, input_type, fields=None, location_bias=None, language=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    A Find Place request takes a text input, and returns a place.\n    The text input can be any kind of Places data, for example,\n    a name, address, or phone number.\n\n    :param input: The text input specifying which place to search for (for\n                  example, a name, address, or phone number).\n    :type input: string\n\n    :param input_type: The type of input. This can be one of either 'textquery'\n                  or 'phonenumber'.\n    :type input_type: string\n\n    :param fields: The fields specifying the types of place data to return. For full details see:\n                   https://developers.google.com/places/web-service/search#FindPlaceRequests\n    :type fields: list\n\n    :param location_bias: Prefer results in a specified area, by specifying\n                          either a radius plus lat/lng, or two lat/lng pairs\n                          representing the points of a rectangle. See:\n                          https://developers.google.com/places/web-service/search#FindPlaceRequests\n    :type location_bias: string\n\n    :param language: The language in which to return results.\n    :type language: string\n\n    :rtype: result dict with the following keys:\n            status: status code\n            candidates: list of places\n    "
    params = {'input': input, 'inputtype': input_type}
    if input_type != 'textquery' and input_type != 'phonenumber':
        raise ValueError("Valid values for the `input_type` param for `find_place` are 'textquery' or 'phonenumber', the given value is invalid: '%s'" % input_type)
    if fields:
        deprecated_fields = set(fields) & DEPRECATED_FIELDS
        if deprecated_fields:
            warnings.warn(DEPRECATED_FIELDS_MESSAGE % str(list(deprecated_fields)), DeprecationWarning)
        invalid_fields = set(fields) - PLACES_FIND_FIELDS
        if invalid_fields:
            raise ValueError("Valid values for the `fields` param for `find_place` are '%s', these given field(s) are invalid: '%s'" % ("', '".join(PLACES_FIND_FIELDS), "', '".join(invalid_fields)))
        params['fields'] = convert.join_list(',', fields)
    if location_bias:
        valid = ['ipbias', 'point', 'circle', 'rectangle']
        if location_bias.split(':')[0] not in valid:
            raise ValueError('location_bias should be prefixed with one of: %s' % valid)
        params['locationbias'] = location_bias
    if language:
        params['language'] = language
    return client._request('/maps/api/place/findplacefromtext/json', params)

def places(client, query=None, location=None, radius=None, language=None, min_price=None, max_price=None, open_now=False, type=None, region=None, page_token=None):
    if False:
        i = 10
        return i + 15
    '\n    Places search.\n\n    :param query: The text string on which to search, for example: "restaurant".\n    :type query: string\n\n    :param location: The latitude/longitude value for which you wish to obtain the\n        closest, human-readable address.\n    :type location: string, dict, list, or tuple\n\n    :param radius: Distance in meters within which to bias results.\n    :type radius: int\n\n    :param language: The language in which to return results.\n    :type language: string\n\n    :param min_price: Restricts results to only those places with no less than\n        this price level. Valid values are in the range from 0 (most affordable)\n        to 4 (most expensive).\n    :type min_price: int\n\n    :param max_price: Restricts results to only those places with no greater\n        than this price level. Valid values are in the range from 0 (most\n        affordable) to 4 (most expensive).\n    :type max_price: int\n\n    :param open_now: Return only those places that are open for business at\n        the time the query is sent.\n    :type open_now: bool\n\n    :param type: Restricts the results to places matching the specified type.\n        The full list of supported types is available here:\n        https://developers.google.com/places/supported_types\n    :type type: string\n\n    :param region: The region code, optional parameter.\n        See more @ https://developers.google.com/places/web-service/search\n    :type region: string\n\n    :param page_token: Token from a previous search that when provided will\n        returns the next page of results for the same search.\n    :type page_token: string\n\n    :rtype: result dict with the following keys:\n        results: list of places\n        html_attributions: set of attributions which must be displayed\n        next_page_token: token for retrieving the next page of results\n    '
    return _places(client, 'text', query=query, location=location, radius=radius, language=language, min_price=min_price, max_price=max_price, open_now=open_now, type=type, region=region, page_token=page_token)

def places_nearby(client, location=None, radius=None, keyword=None, language=None, min_price=None, max_price=None, name=None, open_now=False, rank_by=None, type=None, page_token=None):
    if False:
        print('Hello World!')
    '\n    Performs nearby search for places.\n\n    :param location: The latitude/longitude value for which you wish to obtain the\n                     closest, human-readable address.\n    :type location: string, dict, list, or tuple\n\n    :param radius: Distance in meters within which to bias results.\n    :type radius: int\n\n    :param region: The region code, optional parameter.\n        See more @ https://developers.google.com/places/web-service/search\n    :type region: string\n\n    :param keyword: A term to be matched against all content that Google has\n                    indexed for this place.\n    :type keyword: string\n\n    :param language: The language in which to return results.\n    :type language: string\n\n    :param min_price: Restricts results to only those places with no less than\n                      this price level. Valid values are in the range from 0\n                      (most affordable) to 4 (most expensive).\n    :type min_price: int\n\n    :param max_price: Restricts results to only those places with no greater\n                      than this price level. Valid values are in the range\n                      from 0 (most affordable) to 4 (most expensive).\n    :type max_price: int\n\n    :param name: One or more terms to be matched against the names of places.\n    :type name: string or list of strings\n\n    :param open_now: Return only those places that are open for business at\n                     the time the query is sent.\n    :type open_now: bool\n\n    :param rank_by: Specifies the order in which results are listed.\n                    Possible values are: prominence (default), distance\n    :type rank_by: string\n\n    :param type: Restricts the results to places matching the specified type.\n        The full list of supported types is available here:\n        https://developers.google.com/places/supported_types\n    :type type: string\n\n    :param page_token: Token from a previous search that when provided will\n                       returns the next page of results for the same search.\n    :type page_token: string\n\n    :rtype: result dict with the following keys:\n            status: status code\n            results: list of places\n            html_attributions: set of attributions which must be displayed\n            next_page_token: token for retrieving the next page of results\n\n    '
    if not location and (not page_token):
        raise ValueError('either a location or page_token arg is required')
    if rank_by == 'distance':
        if not (keyword or name or type):
            raise ValueError('either a keyword, name, or type arg is required when rank_by is set to distance')
        elif radius is not None:
            raise ValueError('radius cannot be specified when rank_by is set to distance')
    return _places(client, 'nearby', location=location, radius=radius, keyword=keyword, language=language, min_price=min_price, max_price=max_price, name=name, open_now=open_now, rank_by=rank_by, type=type, page_token=page_token)

def _places(client, url_part, query=None, location=None, radius=None, keyword=None, language=None, min_price=0, max_price=4, name=None, open_now=False, rank_by=None, type=None, region=None, page_token=None):
    if False:
        print('Hello World!')
    "\n    Internal handler for ``places`` and ``places_nearby``.\n    See each method's docs for arg details.\n    "
    params = {'minprice': min_price, 'maxprice': max_price}
    if query:
        params['query'] = query
    if location:
        params['location'] = convert.latlng(location)
    if radius:
        params['radius'] = radius
    if keyword:
        params['keyword'] = keyword
    if language:
        params['language'] = language
    if name:
        params['name'] = convert.join_list(' ', name)
    if open_now:
        params['opennow'] = 'true'
    if rank_by:
        params['rankby'] = rank_by
    if type:
        params['type'] = type
    if region:
        params['region'] = region
    if page_token:
        params['pagetoken'] = page_token
    url = '/maps/api/place/%ssearch/json' % url_part
    return client._request(url, params)

def place(client, place_id, session_token=None, fields=None, language=None, reviews_no_translations=False, reviews_sort='most_relevant'):
    if False:
        while True:
            i = 10
    '\n    Comprehensive details for an individual place.\n\n    :param place_id: A textual identifier that uniquely identifies a place,\n        returned from a Places search.\n    :type place_id: string\n\n    :param session_token: A random string which identifies an autocomplete\n                          session for billing purposes.\n    :type session_token: string\n\n    :param fields: The fields specifying the types of place data to return,\n                   separated by a comma. For full details see:\n                   https://cloud.google.com/maps-platform/user-guide/product-changes/#places\n    :type input: list\n\n    :param language: The language in which to return results.\n    :type language: string\n\n    :param reviews_no_translations: Specify reviews_no_translations=True to disable translation of reviews; reviews_no_translations=False (default) enables translation of reviews.\n    :type reviews_no_translations: bool\n\n    :param reviews_sort: The sorting method to use when returning reviews.\n                         Can be set to most_relevant (default) or newest.\n    :type reviews_sort: string\n\n    :rtype: result dict with the following keys:\n        result: dict containing place details\n        html_attributions: set of attributions which must be displayed\n    '
    params = {'placeid': place_id}
    if fields:
        deprecated_fields = set(fields) & DEPRECATED_FIELDS
        if deprecated_fields:
            warnings.warn(DEPRECATED_FIELDS_MESSAGE % str(list(deprecated_fields)), DeprecationWarning)
        invalid_fields = set(fields) - PLACES_DETAIL_FIELDS
        if invalid_fields:
            raise ValueError("Valid values for the `fields` param for `place` are '%s', these given field(s) are invalid: '%s'" % ("', '".join(PLACES_DETAIL_FIELDS), "', '".join(invalid_fields)))
        params['fields'] = convert.join_list(',', fields)
    if language:
        params['language'] = language
    if session_token:
        params['sessiontoken'] = session_token
    if reviews_no_translations:
        params['reviews_no_translations'] = 'true'
    if reviews_sort:
        params['reviews_sort'] = reviews_sort
    return client._request('/maps/api/place/details/json', params)

def places_photo(client, photo_reference, max_width=None, max_height=None):
    if False:
        while True:
            i = 10
    "\n    Downloads a photo from the Places API.\n\n    :param photo_reference: A string identifier that uniquely identifies a\n        photo, as provided by either a Places search or Places detail request.\n    :type photo_reference: string\n\n    :param max_width: Specifies the maximum desired width, in pixels.\n    :type max_width: int\n\n    :param max_height: Specifies the maximum desired height, in pixels.\n    :type max_height: int\n\n    :rtype: iterator containing the raw image data, which typically can be\n        used to save an image file locally. For example:\n\n    .. code-block:: python\n\n        f = open(local_filename, 'wb')\n        for chunk in client.places_photo(photo_reference, max_width=100):\n            if chunk:\n                f.write(chunk)\n        f.close()\n    "
    if not (max_width or max_height):
        raise ValueError('a max_width or max_height arg is required')
    params = {'photoreference': photo_reference}
    if max_width:
        params['maxwidth'] = max_width
    if max_height:
        params['maxheight'] = max_height
    response = client._request('/maps/api/place/photo', params, extract_body=lambda response: response, requests_kwargs={'stream': True})
    return response.iter_content()

def places_autocomplete(client, input_text, session_token=None, offset=None, origin=None, location=None, radius=None, language=None, types=None, components=None, strict_bounds=False):
    if False:
        while True:
            i = 10
    "\n    Returns Place predictions given a textual search string and optional\n    geographic bounds.\n\n    :param input_text: The text string on which to search.\n    :type input_text: string\n\n    :param session_token: A random string which identifies an autocomplete\n                          session for billing purposes.\n    :type session_token: string\n\n    :param offset: The position, in the input term, of the last character\n                   that the service uses to match predictions. For example,\n                   if the input is 'Google' and the offset is 3, the\n                   service will match on 'Goo'.\n    :type offset: int\n\n    :param origin: The origin point from which to calculate straight-line distance\n                    to the destination (returned as distance_meters).\n                    If this value is omitted, straight-line distance will\n                    not be returned.\n    :type origin: string, dict, list, or tuple\n\n    :param location: The latitude/longitude value for which you wish to obtain the\n                     closest, human-readable address.\n    :type location: string, dict, list, or tuple\n\n    :param radius: Distance in meters within which to bias results.\n    :type radius: int\n\n    :param language: The language in which to return results.\n    :type language: string\n\n    :param types: Restricts the results to places matching the specified type.\n        The full list of supported types is available here:\n        https://developers.google.com/places/web-service/autocomplete#place_types\n    :type types: string\n\n    :param components: A component filter for which you wish to obtain a geocode.\n        Currently, you can use components to filter by up to 5 countries for\n        example: ``{'country': ['US', 'AU']}``\n    :type components: dict\n\n    :param strict_bounds: Returns only those places that are strictly within\n        the region defined by location and radius.\n    :type strict_bounds: bool\n\n    :rtype: list of predictions\n\n    "
    return _autocomplete(client, '', input_text, session_token=session_token, offset=offset, origin=origin, location=location, radius=radius, language=language, types=types, components=components, strict_bounds=strict_bounds)

def places_autocomplete_query(client, input_text, offset=None, location=None, radius=None, language=None):
    if False:
        i = 10
        return i + 15
    '\n    Returns Place predictions given a textual search query, such as\n    "pizza near New York", and optional geographic bounds.\n\n    :param input_text: The text query on which to search.\n    :type input_text: string\n\n    :param offset: The position, in the input term, of the last character\n        that the service uses to match predictions. For example, if the input\n        is \'Google\' and the offset is 3, the service will match on \'Goo\'.\n    :type offset: int\n\n    :param location: The latitude/longitude value for which you wish to obtain the\n        closest, human-readable address.\n    :type location: string, dict, list, or tuple\n\n    :param radius: Distance in meters within which to bias results.\n    :type radius: number\n\n    :param language: The language in which to return results.\n    :type language: string\n\n    :rtype: list of predictions\n    '
    return _autocomplete(client, 'query', input_text, offset=offset, location=location, radius=radius, language=language)

def _autocomplete(client, url_part, input_text, session_token=None, offset=None, origin=None, location=None, radius=None, language=None, types=None, components=None, strict_bounds=False):
    if False:
        i = 10
        return i + 15
    "\n    Internal handler for ``autocomplete`` and ``autocomplete_query``.\n    See each method's docs for arg details.\n    "
    params = {'input': input_text}
    if session_token:
        params['sessiontoken'] = session_token
    if offset:
        params['offset'] = offset
    if origin:
        params['origin'] = convert.latlng(origin)
    if location:
        params['location'] = convert.latlng(location)
    if radius:
        params['radius'] = radius
    if language:
        params['language'] = language
    if types:
        params['types'] = types
    if components:
        if len(components) != 1 or list(components.keys())[0] != 'country':
            raise ValueError('Only country components are supported')
        params['components'] = convert.components(components)
    if strict_bounds:
        params['strictbounds'] = 'true'
    url = '/maps/api/place/%sautocomplete/json' % url_part
    return client._request(url, params).get('predictions', [])