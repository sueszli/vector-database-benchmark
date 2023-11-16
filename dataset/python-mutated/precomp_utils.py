"""SHERIFS
Seismic Hazard and Earthquake Rates In Fault Systems

Version 1.2

@author: Thomas Chartier
"""
import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, radians, sin, asin, sqrt, atan2, degrees, acos, radians

def distance(lon1, lat1, lon2, lat2):
    if False:
        print('Hello World!')
    '\n    Calculate the great circle distance between two points\n    on the earth (specified in decimal degrees)\n    '
    (lon1, lat1, lon2, lat2) = list(map(radians, [lon1, lat1, lon2, lat2]))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    return km

def fault_length(lons, lats):
    if False:
        print('Hello World!')
    length = 0.0
    for i in range(len(lons) - 1):
        length += distance(lons[i], lats[i], lons[i + 1], lats[i + 1])
    return length

def find_bounding_box(faults):
    if False:
        for i in range(10):
            print('nop')
    nb_faults = len(faults)
    (maxmin_pt_lon, maxmin_pt_lat) = ([], [])
    for fi in range(nb_faults):
        maxmin_pt_lon.append([np.min([i[0] for i in faults[fi]['geometry']['coordinates']]), np.max([i[0] for i in faults[fi]['geometry']['coordinates']])])
        maxmin_pt_lat.append([np.min([i[1] for i in faults[fi]['geometry']['coordinates']]), np.max([i[1] for i in faults[fi]['geometry']['coordinates']])])
    return (maxmin_pt_lon, maxmin_pt_lat)

def find_possible_asso(maxmin_pt_lon, maxmin_pt_lat, plot_fig=False):
    if False:
        print('Hello World!')
    d = 0.5
    assso_fault = []
    for (lon_i, lat_i) in zip(maxmin_pt_lon, maxmin_pt_lat):
        assso_fault_i = []
        j_fault = 0
        for (lon_j, lat_j) in zip(maxmin_pt_lon, maxmin_pt_lat):
            if lon_j[0] > lon_i[0] - d and lon_j[0] < lon_i[1] + d:
                if lat_j[0] > lat_i[0] - d and lat_j[0] < lat_i[1] + d:
                    assso_fault_i.append(j_fault)
                if lat_j[1] > lat_i[0] - d and lat_j[1] < lat_i[1] + d:
                    assso_fault_i.append(j_fault)
            if lon_j[1] > lon_i[0] - d and lon_j[1] < lon_i[1] + d:
                if lat_j[0] > lat_i[0] - d and lat_j[0] < lat_i[1] + d:
                    assso_fault_i.append(j_fault)
                if lat_j[1] > lat_i[0] - d and lat_j[1] < lat_i[1] + d:
                    assso_fault_i.append(j_fault)
            j_fault += 1
        assso_fault_i = list(set(assso_fault_i))
        assso_fault.append(assso_fault_i)
    if plot_fig == True:
        x = []
        for i in assso_fault:
            x.append(len(i) - 1 - 0.5)
        plt.hist(x)
        plt.xlabel('number of close faults to be considered for rupture jump')
        plt.ylabel('number of faults in this situation')
        plt.show()
    return assso_fault

def calc_f_dims(faults, plt_fig=False):
    if False:
        print('Hello World!')
    nb_faults = len(faults)
    f_lengths = []
    f_areas = []
    for fi in range(nb_faults):
        lons_i = [i[0] for i in faults[fi]['geometry']['coordinates']]
        lats_i = [i[1] for i in faults[fi]['geometry']['coordinates']]
        length_i = fault_length(lons_i, lats_i)
        f_lengths.append(length_i)
        try:
            width_i = (faults[fi]['properties']['lsd'] - faults[fi]['properties']['usd']) / sin(radians(faults[fi]['properties']['dip']))
        except:
            width_i = (faults[fi]['properties']['lo_s_d'] - faults[fi]['properties']['up_s_d']) / sin(radians(faults[fi]['properties']['dip']))
        f_areas.append(length_i * width_i)
    if plt_fig == True:
        plt.hist(f_lengths)
        plt.xlabel('Lengths (km)')
        plt.ylabel('Nb faults')
        plt.show()
    print('In total, there are ', round(sum(f_lengths)), ' km of faults in the model.')
    return (f_lengths, f_areas)

def calculate_initial_compass_bearing(pointA, pointB):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculates the bearing between two points.\n\n    The formulae used is the following:\n        θ = atan2(sin(Δlong).cos(lat2),\n                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))\n\n    :Parameters:\n      - `pointA: The tuple representing the latitude/longitude for the\n        first point. Latitude and longitude must be in decimal degrees\n      - `pointB: The tuple representing the latitude/longitude for the\n        second point. Latitude and longitude must be in decimal degrees\n\n    :Returns:\n      The bearing in degrees\n\n    :Returns Type:\n      float\n    '
    lat1 = radians(pointA[0])
    lat2 = radians(pointB[0])
    diffLong = radians(pointB[1] - pointA[1])
    x = sin(diffLong) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(diffLong)
    initial_bearing = atan2(x, y)
    initial_bearing = degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

def wc1994_median_mag(area, rake):
    if False:
        while True:
            i = 10
    '\n    Return magnitude (Mw) given the area and rake.\n\n    Setting the rake to ``None`` causes their "All" rupture-types\n    to be applied.\n\n    :param area:\n        Area in square km.\n    :param rake:\n        Rake angle (the rupture propagation direction) in degrees,\n        from -180 to 180.\n    '
    assert rake is None or -180 <= rake <= 180
    if rake is None:
        return 4.07 + 0.98 * np.log10(area)
    elif -45 <= rake <= 45 or rake > 135 or rake < -135:
        return 3.98 + 1.02 * np.log10(area)
    elif rake > 0:
        return 4.33 + 0.9 * np.log10(area)
    else:
        return 3.93 + 1.02 * np.log10(area)

def mag_to_M0(mag):
    if False:
        while True:
            i = 10
    M0 = 10.0 ** (1.5 * mag + 9.1)
    return M0