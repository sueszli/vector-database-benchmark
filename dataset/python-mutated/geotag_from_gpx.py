import datetime
import math
import os
import shutil
import sys
import time
from typing import List, Union
import numpy as np
from opensfm import geo
try:
    import pyexiv2
    from pyexiv2.utils import make_fraction
except ImportError:
    print('ERROR: pyexiv2 module not available')
try:
    import gpxpy
except ImportError:
    print('ERROR: gpxpy module not available')
"\n(source: https://github.com/mapillary/mapillary_tools)\n\n\nScript for geotagging images using a gpx file from an external GPS.\nIntended as a lightweight tool.\n\n!!! This version needs testing, please report issues.!!!\n\nUses the capture time in EXIF and looks up an interpolated lat, lon, bearing\nfor each image, and writes the values to the EXIF of the image.\n\nYou can supply a time offset in seconds if the GPS clock and camera clocks are not in sync.\n\nRequires gpxpy, e.g. 'pip install gpxpy'\n\nRequires pyexiv2, see install instructions at http://tilloy.net/dev/pyexiv2/\n(or use your favorite installer, e.g. 'brew install pyexiv2').\n"

def utc_to_localtime(utc_time):
    if False:
        for i in range(10):
            print('nop')
    utc_offset_timedelta = datetime.datetime.utcnow() - datetime.datetime.now()
    return utc_time - utc_offset_timedelta

def get_lat_lon_time(gpx_file, gpx_time: str='utc'):
    if False:
        i = 10
        return i + 15
    '\n    Read location and time stamps from a track in a GPX file.\n\n    Returns a list of tuples (time, lat, lon, elevation).\n\n    GPX stores time in UTC, assume your camera used the local\n    timezone and convert accordingly.\n    '
    with open(gpx_file, 'r') as f:
        gpx = gpxpy.parse(f)
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                ptime = utc_to_localtime(point.time) if gpx_time == 'utc' else point.time
                points.append((ptime, point.latitude, point.longitude, point.elevation))
    points.sort()
    return points

def compute_bearing(start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> float:
    if False:
        i = 10
        return i + 15
    '\n    Get the compass bearing from start to end.\n\n    Formula from\n    http://www.movable-type.co.uk/scripts/latlong.html\n    '
    start_lat = math.radians(start_lat)
    start_lon = math.radians(start_lon)
    end_lat = math.radians(end_lat)
    end_lon = math.radians(end_lon)
    dLong = end_lon - start_lon
    if abs(dLong) > math.pi:
        if dLong > 0.0:
            dLong = -(2.0 * math.pi - dLong)
        else:
            dLong = 2.0 * math.pi + dLong
    y = math.sin(dLong) * math.cos(end_lat)
    x = math.cos(start_lat) * math.sin(end_lat) - math.sin(start_lat) * math.cos(end_lat) * math.cos(dLong)
    bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return bearing

def interpolate_lat_lon(points, t):
    if False:
        return 10
    '\n    Return interpolated lat, lon and compass bearing for time t.\n\n    Points is a list of tuples (time, lat, lon, elevation), t a datetime object.\n    '
    if t < points[0][0] or t >= points[-1][0]:
        raise ValueError('Time t not in scope of gpx file.')
    for (i, point) in enumerate(points):
        if t < point[0]:
            if i > 0:
                before = points[i - 1]
            else:
                before = points[i]
            after = points[i]
            break
    dt_before = (t - before[0]).total_seconds()
    dt_after = (after[0] - t).total_seconds()
    lat = (before[1] * dt_after + after[1] * dt_before) / (dt_before + dt_after)
    lon = (before[2] * dt_after + after[2] * dt_before) / (dt_before + dt_after)
    bearing = compute_bearing(before[1], before[2], after[1], after[2])
    if before[3] is not None:
        ele = (before[3] * dt_after + after[3] * dt_before) / (dt_before + dt_after)
    else:
        ele = None
    return (lat, lon, bearing, ele)

def to_deg(value, loc):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert decimal position to degrees.\n    '
    if value < 0:
        loc_value = loc[0]
    elif value > 0:
        loc_value = loc[1]
    else:
        loc_value = ''
    abs_value = abs(value)
    deg = int(abs_value)
    t1 = (abs_value - deg) * 60
    mint = int(t1)
    sec = round((t1 - mint) * 60, 6)
    return (deg, mint, sec, loc_value)

def gpx_lerp(alpha: int, a, b):
    if False:
        print('Hello World!')
    'Interpolate gpx point as (1 - alpha) * a + alpha * b'
    dt = alpha * (b[0] - a[0]).total_seconds()
    t = a[0] + datetime.timedelta(seconds=dt)
    lat = (1 - alpha) * a[1] + alpha * b[1]
    lon = (1 - alpha) * a[2] + alpha * b[2]
    alt = (1 - alpha) * a[3] + alpha * b[3]
    return (t, lat, lon, alt)

def segment_sphere_intersection(A, B, C, r):
    if False:
        i = 10
        return i + 15
    'Intersect the segment AB and the sphere (C,r).\n\n    Assumes A is inside the sphere and B is outside.\n    Return the ratio between the length of AI and the length\n    of AB, where I is the intersection.\n    '
    AB = np.array(B) - np.array(A)
    CA = np.array(A) - np.array(C)
    a = AB.dot(AB)
    b = 2 * AB.dot(CA)
    c = CA.dot(CA) - r ** 2
    d = max(0, b ** 2 - 4 * a * c)
    return (-b + np.sqrt(d)) / (2 * a)

def space_next_point(a, b, last, dx):
    if False:
        for i in range(10):
            print('nop')
    A = geo.ecef_from_lla(a[1], a[2], 0.0)
    B = geo.ecef_from_lla(b[1], b[2], 0.0)
    C = geo.ecef_from_lla(last[1], last[2], 0.0)
    alpha = segment_sphere_intersection(A, B, C, dx)
    return gpx_lerp(alpha, a, b)

def time_next_point(a, b, last, dt):
    if False:
        print('Hello World!')
    da = (a[0] - last[0]).total_seconds()
    db = (b[0] - last[0]).total_seconds()
    alpha = (dt - da) / (db - da)
    return gpx_lerp(alpha, a, b)

def time_distance(a, b) -> int:
    if False:
        while True:
            i = 10
    return (b[0] - a[0]).total_seconds()

def space_distance(a, b) -> float:
    if False:
        for i in range(10):
            print('nop')
    return geo.gps_distance(a[1:3], b[1:3])

def sample_gpx(points, dx: float, dt=None):
    if False:
        return 10
    if dt is not None:
        dx = float(dt)
        print('Sampling GPX file every {0} seconds'.format(dx))
        distance = time_distance
        next_point = time_next_point
    else:
        print('Sampling GPX file every {0} meters'.format(dx))
        distance = space_distance
        next_point = space_next_point
    key_points = [points[0]]
    a = points[0]
    for i in range(1, len(points)):
        (a, b) = (points[i - 1], points[i])
        dx_b = distance(key_points[-1], b)
        while dx and dx_b >= dx:
            a = next_point(a, b, key_points[-1], dx)
            key_points.append(a)
            assert np.fabs(dx - distance(key_points[-2], key_points[-1])) < 0.1
            dx_b = distance(key_points[-1], b)
    print('{} points sampled'.format(len(key_points)))
    return key_points

def add_gps_to_exif(filename: Union['os.PathLike[str]', str], lat, lon, bearing, elevation, updated_filename: Union[None, 'os.PathLike[str]', str]=None, remove_image_description: bool=True) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Given lat, lon, bearing, elevation, write to EXIF\n    '
    if updated_filename is not None:
        shutil.copy2(filename, updated_filename)
        filename = updated_filename
    metadata = pyexiv2.ImageMetadata(filename)
    metadata.read()
    lat_deg = to_deg(lat, ['S', 'N'])
    lon_deg = to_deg(lon, ['W', 'E'])
    exiv_lat = (make_fraction(lat_deg[0], 1), make_fraction(int(lat_deg[1]), 1), make_fraction(int(lat_deg[2] * 1000000), 1000000))
    exiv_lon = (make_fraction(lon_deg[0], 1), make_fraction(int(lon_deg[1]), 1), make_fraction(int(lon_deg[2] * 1000000), 1000000))
    exiv_bearing = make_fraction(int(bearing * 100), 100)
    metadata['Exif.GPSInfo.GPSLatitude'] = exiv_lat
    metadata['Exif.GPSInfo.GPSLatitudeRef'] = lat_deg[3]
    metadata['Exif.GPSInfo.GPSLongitude'] = exiv_lon
    metadata['Exif.GPSInfo.GPSLongitudeRef'] = lon_deg[3]
    metadata['Exif.Image.GPSTag'] = 654
    metadata['Exif.GPSInfo.GPSMapDatum'] = 'WGS-84'
    metadata['Exif.GPSInfo.GPSVersionID'] = '2 0 0 0'
    metadata['Exif.GPSInfo.GPSImgDirection'] = exiv_bearing
    metadata['Exif.GPSInfo.GPSImgDirectionRef'] = 'T'
    if remove_image_description:
        metadata['Exif.Image.ImageDescription'] = []
    if elevation is not None:
        exiv_elevation = make_fraction(int(abs(elevation) * 100), 100)
        metadata['Exif.GPSInfo.GPSAltitude'] = exiv_elevation
        metadata['Exif.GPSInfo.GPSAltitudeRef'] = '0' if elevation >= 0 else '1'
    metadata.write()

def add_exif_using_timestamp(filename, points, offset_time: int=0, timestamp=None, orientation: int=1, image_description=None) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Find lat, lon and bearing of filename and write to EXIF.\n    '
    metadata = pyexiv2.ImageMetadata(filename)
    metadata.read()
    if timestamp:
        metadata['Exif.Photo.DateTimeOriginal'] = timestamp
    t = metadata['Exif.Photo.DateTimeOriginal'].value
    t = t - datetime.timedelta(seconds=offset_time)
    try:
        (lat, lon, bearing, elevation) = interpolate_lat_lon(points, t)
        lat_deg = to_deg(lat, ['S', 'N'])
        lon_deg = to_deg(lon, ['W', 'E'])
        exiv_lat = (make_fraction(lat_deg[0], 1), make_fraction(int(lat_deg[1]), 1), make_fraction(int(lat_deg[2] * 1000000), 1000000))
        exiv_lon = (make_fraction(lon_deg[0], 1), make_fraction(int(lon_deg[1]), 1), make_fraction(int(lon_deg[2] * 1000000), 1000000))
        exiv_bearing = make_fraction(int(bearing * 1000), 1000)
        metadata['Exif.GPSInfo.GPSLatitude'] = exiv_lat
        metadata['Exif.GPSInfo.GPSLatitudeRef'] = lat_deg[3]
        metadata['Exif.GPSInfo.GPSLongitude'] = exiv_lon
        metadata['Exif.GPSInfo.GPSLongitudeRef'] = lon_deg[3]
        metadata['Exif.Image.GPSTag'] = 654
        metadata['Exif.GPSInfo.GPSMapDatum'] = 'WGS-84'
        metadata['Exif.GPSInfo.GPSVersionID'] = '2 0 0 0'
        metadata['Exif.GPSInfo.GPSImgDirection'] = exiv_bearing
        metadata['Exif.GPSInfo.GPSImgDirectionRef'] = 'T'
        metadata['Exif.Image.Orientation'] = orientation
        if image_description is not None:
            metadata['Exif.Image.ImageDescription'] = image_description
        if elevation is not None:
            exiv_elevation = make_fraction(int(abs(elevation) * 100), 100)
            metadata['Exif.GPSInfo.GPSAltitude'] = exiv_elevation
            metadata['Exif.GPSInfo.GPSAltitudeRef'] = '0' if elevation >= 0 else '1'
        metadata.write()
        print('Added geodata to: {0} ({1}, {2}, {3}), altitude {4}'.format(filename, lat, lon, bearing, elevation))
    except ValueError as e:
        print('Skipping {0}: {1}'.format(filename, e))
if __name__ == '__main__':
    "\n    Use from command line as: python geotag_from_gpx.py path gpx_file time_offset\n\n    The time_offset is optional and defaults to 0.\n    It is defined as 'exif time' - 'gpx time' in whole seconds,\n    so if your camera clock is ahead of the gpx clock by 2s,\n    then the offset is 2.\n    "
    if len(sys.argv) > 4:
        print('Usage: python geotag_from_gpx.py path gpx_file time_offset')
        raise IOError('Bad input parameters.')
    path: str = sys.argv[1]
    gpx_filename: str = sys.argv[2]
    if len(sys.argv) == 4:
        time_offset = int(sys.argv[3])
    else:
        time_offset = 0
    if path.lower().endswith('.jpg'):
        file_list: List[str] = [path]
    else:
        file_list: List[str] = []
        for (root, _, files) in os.walk(path):
            file_list += [os.path.join(root, filename) for filename in files if filename.lower().endswith('.jpg')]
    t: float = time.time()
    gpx = get_lat_lon_time(gpx_filename)
    print('===\nStarting geotagging of {0} images using {1}.\n==='.format(len(file_list), gpx_filename))
    for filepath in file_list:
        add_exif_using_timestamp(filepath, gpx, time_offset)
    print('Done geotagging {0} images in {1} seconds.'.format(len(file_list), time.time() - t))