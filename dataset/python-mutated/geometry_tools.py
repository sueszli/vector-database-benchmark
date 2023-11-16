"""SHERIFS
Seismic Hazard and Earthquake Rates In Fault Systems

Version 1.2

@author: Thomas Chartier
"""
import numpy as np
from math import pi, cos, radians, sin, asin, sqrt, atan2, degrees

def reproject(latitude, longitude):
    if False:
        print('Hello World!')
    'Returns the x & y coordinates in meters using a sinusoidal projection'
    earth_radius = 6371009
    lat_dist = pi * earth_radius / 180.0
    y = [lat * lat_dist for lat in latitude]
    x = [long * lat_dist * cos(radians(lat)) for (lat, long) in zip(latitude, longitude)]
    return (x, y)

def points_aligned(a, b, c):
    if False:
        while True:
            i = 10
    crossproduct = (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])
    epsilon = 10000000.0
    if abs(crossproduct) > epsilon:
        return False
    dotproduct = (c[0] - a[0]) * (b[0] - a[0]) + (c[1] - a[1]) * (b[1] - a[1])
    if dotproduct < 0:
        return False
    squaredlengthba = (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1])
    if dotproduct > squaredlengthba:
        return False
    return True

def point_at(lon, lat, azimuth, distance):
    if False:
        return 10
    '\n    Modified from OQ hazardlib\n    Perform a forward geodetic transformation: find a point lying at a given\n    distance from a given one on a great circle arc defined by azimuth.\n    :param float lon, lat:\n        Coordinates of a reference point, in decimal degrees.\n    :param azimuth:\n        An azimuth of a great circle arc of interest measured in a reference\n        point in decimal degrees.\n    :param distance:\n        Distance to target point in km.\n    :returns:\n        Tuple of two float numbers: longitude and latitude of a target point\n        in decimal degrees respectively.\n    Implements the same approach as :func:`npoints_towards`.\n    '
    (lon, lat) = (np.radians(lon), np.radians(lat))
    tc = np.radians(360 - azimuth)
    EARTH_RADIUS = 6371.0
    sin_dists = np.sin(distance / EARTH_RADIUS)
    cos_dists = np.cos(distance / EARTH_RADIUS)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lats = sin_lat * cos_dists + cos_lat * sin_dists * np.cos(tc)
    lats = np.degrees(np.arcsin(sin_lats))
    dlon = np.arctan2(np.sin(tc) * sin_dists * cos_lat, cos_dists - sin_lat * sin_lats)
    lons = np.mod(lon - dlon + np.pi, 2 * np.pi) - np.pi
    lons = np.degrees(lons)
    return (lons, lats)

def PolyArea(x, y):
    if False:
        print('Hello World!')
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def area_of_polygon(x, y):
    if False:
        i = 10
        return i + 15
    'Calculates the area of an arbitrary polygon given its verticies'
    x_vertices = []
    y_vertices = []
    inn = []
    for i in range(len(x)):
        if i == 0:
            if points_aligned([x[-1], y[-1]], [x[0], y[0]], [x[1], y[1]]) == False:
                x_vertices.append(x[i])
                y_vertices.append(y[i])
                inn.append(1)
            else:
                inn.append(0)
        elif i == len(x) - 1:
            if points_aligned([x[-2], y[-2]], [x[-1], y[-1]], [x[0], y[0]]) == False:
                x_vertices.append(x[i])
                y_vertices.append(y[i])
                inn.append(1)
            else:
                inn.append(0)
        elif points_aligned([x[i - 1], y[i - 1]], [x[i], y[i]], [x[i + 1], y[i + 1]]) == False:
            x_vertices.append(x[i])
            y_vertices.append(y[i])
            inn.append(1)
        else:
            inn.append(0)
    area = 0.0
    for i in range(-1, len(x_vertices) - 1):
        area += x_vertices[i] * (y_vertices[i + 1] - y_vertices[i - 1])
    return abs(area) / 2.0

def line_intersection(line1, line2):
    if False:
        while True:
            i = 10
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        if False:
            return 10
        return a[0] * b[1] - a[1] * b[0]
    div = det(xdiff, ydiff)
    if div == 0:
        x = 'no_intesection'
        y = 'no_intesection'
    else:
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
    return (x, y)

def calculate_initial_compass_bearing(pointA, pointB):
    if False:
        while True:
            i = 10
    '\n    Calculates the bearing between two points.\n\n    The formulae used is the following:\n        θ = atan2(sin(Δlong).cos(lat2),\n                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))\n\n    :Parameters:\n      - `pointA: The tuple representing the latitude/longitude for the\n        first point. Latitude and longitude must be in decimal degrees\n      - `pointB: The tuple representing the latitude/longitude for the\n        second point. Latitude and longitude must be in decimal degrees\n\n    :Returns:\n      The bearing in degrees\n\n    :Returns Type:\n      float\n    '
    if type(pointA) != tuple or type(pointB) != tuple:
        raise TypeError('Only tuples are supported as arguments')
    lat1 = radians(pointA[0])
    lat2 = radians(pointB[0])
    diffLong = radians(pointB[1] - pointA[1])
    x = sin(diffLong) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(diffLong)
    initial_bearing = atan2(x, y)
    initial_bearing = degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

def distance(lon1, lat1, lon2, lat2):
    if False:
        while True:
            i = 10
    '\n    Calculate the great circle distance between two points\n    on the earth (specified in decimal degrees)\n    '
    (lon1, lat1, lon2, lat2) = list(map(radians, [lon1, lat1, lon2, lat2]))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    return km