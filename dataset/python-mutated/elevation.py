"""Performs requests to the Google Maps Elevation API."""
from googlemaps import convert

def elevation(client, locations):
    if False:
        for i in range(10):
            print('nop')
    '\n    Provides elevation data for locations provided on the surface of the\n    earth, including depth locations on the ocean floor (which return negative\n    values)\n\n    :param locations: List of latitude/longitude values from which you wish\n        to calculate elevation data.\n    :type locations: a single location, or a list of locations, where a\n        location is a string, dict, list, or tuple\n\n    :rtype: list of elevation data responses\n    '
    params = {'locations': convert.shortest_path(locations)}
    return client._request('/maps/api/elevation/json', params).get('results', [])

def elevation_along_path(client, path, samples):
    if False:
        return 10
    '\n    Provides elevation data sampled along a path on the surface of the earth.\n\n    :param path: An encoded polyline string, or a list of latitude/longitude\n        values from which you wish to calculate elevation data.\n    :type path: string, dict, list, or tuple\n\n    :param samples: The number of sample points along a path for which to\n        return elevation data.\n    :type samples: int\n\n    :rtype: list of elevation data responses\n    '
    if type(path) is str:
        path = 'enc:%s' % path
    else:
        path = convert.shortest_path(path)
    params = {'path': path, 'samples': samples}
    return client._request('/maps/api/elevation/json', params).get('results', [])