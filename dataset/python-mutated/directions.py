"""Performs requests to the Google Maps Directions API."""
from googlemaps import convert

def directions(client, origin, destination, mode=None, waypoints=None, alternatives=False, avoid=None, language=None, units=None, region=None, departure_time=None, arrival_time=None, optimize_waypoints=False, transit_mode=None, transit_routing_preference=None, traffic_model=None):
    if False:
        return 10
    'Get directions between an origin point and a destination point.\n\n    :param origin: The address or latitude/longitude value from which you wish\n        to calculate directions.\n    :type origin: string, dict, list, or tuple\n\n    :param destination: The address or latitude/longitude value from which\n        you wish to calculate directions. You can use a place_id as destination\n        by putting \'place_id:\' as a prefix in the passing parameter.\n    :type destination: string, dict, list, or tuple\n\n    :param mode: Specifies the mode of transport to use when calculating\n        directions. One of "driving", "walking", "bicycling" or "transit"\n    :type mode: string\n\n    :param waypoints: Specifies an array of waypoints. Waypoints alter a\n        route by routing it through the specified location(s). To influence\n        route without adding stop prefix the waypoint with `via`, similar to\n        `waypoints = ["via:San Francisco", "via:Mountain View"]`.\n    :type waypoints: a single location, or a list of locations, where a\n        location is a string, dict, list, or tuple\n\n    :param alternatives: If True, more than one route may be returned in the\n        response.\n    :type alternatives: bool\n\n    :param avoid: Indicates that the calculated route(s) should avoid the\n        indicated features.\n    :type avoid: list or string\n\n    :param language: The language in which to return results.\n    :type language: string\n\n    :param units: Specifies the unit system to use when displaying results.\n        "metric" or "imperial"\n    :type units: string\n\n    :param region: The region code, specified as a ccTLD ("top-level domain"\n        two-character value.\n    :type region: string\n\n    :param departure_time: Specifies the desired time of departure.\n    :type departure_time: int or datetime.datetime\n\n    :param arrival_time: Specifies the desired time of arrival for transit\n        directions. Note: you can\'t specify both departure_time and\n        arrival_time.\n    :type arrival_time: int or datetime.datetime\n\n    :param optimize_waypoints: Optimize the provided route by rearranging the\n        waypoints in a more efficient order.\n    :type optimize_waypoints: bool\n\n    :param transit_mode: Specifies one or more preferred modes of transit.\n        This parameter may only be specified for requests where the mode is\n        transit. Valid values are "bus", "subway", "train", "tram", "rail".\n        "rail" is equivalent to ["train", "tram", "subway"].\n    :type transit_mode: string or list of strings\n\n    :param transit_routing_preference: Specifies preferences for transit\n        requests. Valid values are "less_walking" or "fewer_transfers"\n    :type transit_routing_preference: string\n\n    :param traffic_model: Specifies the predictive travel time model to use.\n        Valid values are "best_guess" or "optimistic" or "pessimistic".\n        The traffic_model parameter may only be specified for requests where\n        the travel mode is driving, and where the request includes a\n        departure_time.\n    :type units: string\n\n    :rtype: list of routes\n    '
    params = {'origin': convert.latlng(origin), 'destination': convert.latlng(destination)}
    if mode:
        if mode not in ['driving', 'walking', 'bicycling', 'transit']:
            raise ValueError('Invalid travel mode.')
        params['mode'] = mode
    if waypoints:
        waypoints = convert.location_list(waypoints)
        if optimize_waypoints:
            waypoints = 'optimize:true|' + waypoints
        params['waypoints'] = waypoints
    if alternatives:
        params['alternatives'] = 'true'
    if avoid:
        params['avoid'] = convert.join_list('|', avoid)
    if language:
        params['language'] = language
    if units:
        params['units'] = units
    if region:
        params['region'] = region
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
    return client._request('/maps/api/directions/json', params).get('routes', [])