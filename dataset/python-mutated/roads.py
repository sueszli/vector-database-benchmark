"""Performs requests to the Google Maps Roads API."""
import googlemaps
from googlemaps import convert
_ROADS_BASE_URL = 'https://roads.googleapis.com'

def snap_to_roads(client, path, interpolate=False):
    if False:
        return 10
    'Snaps a path to the most likely roads travelled.\n\n    Takes up to 100 GPS points collected along a route, and returns a similar\n    set of data with the points snapped to the most likely roads the vehicle\n    was traveling along.\n\n    :param path: The path to be snapped.\n    :type path: a single location, or a list of locations, where a\n        location is a string, dict, list, or tuple\n\n    :param interpolate: Whether to interpolate a path to include all points\n        forming the full road-geometry. When true, additional interpolated\n        points will also be returned, resulting in a path that smoothly follows\n        the geometry of the road, even around corners and through tunnels.\n        Interpolated paths may contain more points than the original path.\n    :type interpolate: bool\n\n    :rtype: A list of snapped points.\n    '
    params = {'path': convert.location_list(path)}
    if interpolate:
        params['interpolate'] = 'true'
    return client._request('/v1/snapToRoads', params, base_url=_ROADS_BASE_URL, accepts_clientid=False, extract_body=_roads_extract).get('snappedPoints', [])

def nearest_roads(client, points):
    if False:
        while True:
            i = 10
    'Find the closest road segments for each point\n\n    Takes up to 100 independent coordinates, and returns the closest road\n    segment for each point. The points passed do not need to be part of a\n    continuous path.\n\n    :param points: The points for which the nearest road segments are to be\n        located.\n    :type points: a single location, or a list of locations, where a\n        location is a string, dict, list, or tuple\n\n    :rtype: A list of snapped points.\n    '
    params = {'points': convert.location_list(points)}
    return client._request('/v1/nearestRoads', params, base_url=_ROADS_BASE_URL, accepts_clientid=False, extract_body=_roads_extract).get('snappedPoints', [])

def speed_limits(client, place_ids):
    if False:
        return 10
    'Returns the posted speed limit (in km/h) for given road segments.\n\n    :param place_ids: The Place ID of the road segment. Place IDs are returned\n        by the snap_to_roads function. You can pass up to 100 Place IDs.\n    :type place_ids: str or list\n\n    :rtype: list of speed limits.\n    '
    params = [('placeId', place_id) for place_id in convert.as_list(place_ids)]
    return client._request('/v1/speedLimits', params, base_url=_ROADS_BASE_URL, accepts_clientid=False, extract_body=_roads_extract).get('speedLimits', [])

def snapped_speed_limits(client, path):
    if False:
        i = 10
        return i + 15
    'Returns the posted speed limit (in km/h) for given road segments.\n\n    The provided points will first be snapped to the most likely roads the\n    vehicle was traveling along.\n\n    :param path: The path of points to be snapped.\n    :type path: a single location, or a list of locations, where a\n        location is a string, dict, list, or tuple\n\n    :rtype: dict with a list of speed limits and a list of the snapped points.\n    '
    params = {'path': convert.location_list(path)}
    return client._request('/v1/speedLimits', params, base_url=_ROADS_BASE_URL, accepts_clientid=False, extract_body=_roads_extract)

def _roads_extract(resp):
    if False:
        while True:
            i = 10
    'Extracts a result from a Roads API HTTP response.'
    try:
        j = resp.json()
    except:
        if resp.status_code != 200:
            raise googlemaps.exceptions.HTTPError(resp.status_code)
        raise googlemaps.exceptions.ApiError('UNKNOWN_ERROR', 'Received a malformed response.')
    if 'error' in j:
        error = j['error']
        status = error['status']
        if status == 'RESOURCE_EXHAUSTED':
            raise googlemaps.exceptions._OverQueryLimit(status, error.get('message'))
        raise googlemaps.exceptions.ApiError(status, error.get('message'))
    if resp.status_code != 200:
        raise googlemaps.exceptions.HTTPError(resp.status_code)
    return j