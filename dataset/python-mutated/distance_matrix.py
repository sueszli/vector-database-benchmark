"""Performs requests to the Google Maps Distance Matrix API."""
from googlemaps import convert

def distance_matrix(client, origins, destinations, mode=None, language=None, avoid=None, units=None, departure_time=None, arrival_time=None, transit_mode=None, transit_routing_preference=None, traffic_model=None, region=None):
    if False:
        print('Hello World!')
    ' Gets travel distance and time for a matrix of origins and destinations.\n\n    :param origins: One or more addresses, Place IDs, and/or latitude/longitude\n        values, from which to calculate distance and time. Each Place ID string\n        must be prepended with \'place_id:\'. If you pass an address as a string,\n        the service will geocode the string and convert it to a\n        latitude/longitude coordinate to calculate directions.\n    :type origins: a single location, or a list of locations, where a\n        location is a string, dict, list, or tuple\n\n    :param destinations: One or more addresses, Place IDs, and/or lat/lng values\n        , to which to calculate distance and time. Each Place ID string must be\n        prepended with \'place_id:\'. If you pass an address as a string, the\n        service will geocode the string and convert it to a latitude/longitude\n        coordinate to calculate directions.\n    :type destinations: a single location, or a list of locations, where a\n        location is a string, dict, list, or tuple\n\n    :param mode: Specifies the mode of transport to use when calculating\n        directions. Valid values are "driving", "walking", "transit" or\n        "bicycling".\n    :type mode: string\n\n    :param language: The language in which to return results.\n    :type language: string\n\n    :param avoid: Indicates that the calculated route(s) should avoid the\n        indicated features. Valid values are "tolls", "highways" or "ferries".\n    :type avoid: string\n\n    :param units: Specifies the unit system to use when displaying results.\n        Valid values are "metric" or "imperial".\n    :type units: string\n\n    :param departure_time: Specifies the desired time of departure.\n    :type departure_time: int or datetime.datetime\n\n    :param arrival_time: Specifies the desired time of arrival for transit\n        directions. Note: you can\'t specify both departure_time and\n        arrival_time.\n    :type arrival_time: int or datetime.datetime\n\n    :param transit_mode: Specifies one or more preferred modes of transit.\n        This parameter may only be specified for requests where the mode is\n        transit. Valid values are "bus", "subway", "train", "tram", "rail".\n        "rail" is equivalent to ["train", "tram", "subway"].\n    :type transit_mode: string or list of strings\n\n    :param transit_routing_preference: Specifies preferences for transit\n        requests. Valid values are "less_walking" or "fewer_transfers".\n    :type transit_routing_preference: string\n\n    :param traffic_model: Specifies the predictive travel time model to use.\n        Valid values are "best_guess" or "optimistic" or "pessimistic".\n        The traffic_model parameter may only be specified for requests where\n        the travel mode is driving, and where the request includes a\n        departure_time.\n\n    :param region: Specifies the prefered region the geocoder should search\n        first, but it will not restrict the results to only this region. Valid\n        values are a ccTLD code.\n    :type region: string\n\n    :rtype: matrix of distances. Results are returned in rows, each row\n        containing one origin paired with each destination.\n    '
    params = {'origins': convert.location_list(origins), 'destinations': convert.location_list(destinations)}
    if mode:
        if mode not in ['driving', 'walking', 'bicycling', 'transit']:
            raise ValueError('Invalid travel mode.')
        params['mode'] = mode
    if language:
        params['language'] = language
    if avoid:
        if avoid not in ['tolls', 'highways', 'ferries']:
            raise ValueError('Invalid route restriction.')
        params['avoid'] = avoid
    if units:
        params['units'] = units
    if departure_time:
        params['departure_time'] = convert.time(departure_time)
    if arrival_time:
        params['arrival_time'] = convert.time(arrival_time)
    if departure_time and arrival_time:
        raise ValueError('Should not specify both departure_time andarrival_time.')
    if transit_mode:
        params['transit_mode'] = convert.join_list('|', transit_mode)
    if transit_routing_preference:
        params['transit_routing_preference'] = transit_routing_preference
    if traffic_model:
        params['traffic_model'] = traffic_model
    if region:
        params['region'] = region
    return client._request('/maps/api/distancematrix/json', params)