"""Tools to work with the Overpass API."""
import datetime as dt
import logging as lg
import time
import numpy as np
import requests
from requests.exceptions import ConnectionError
from . import _downloader
from . import projection
from . import settings
from . import utils
from . import utils_geo

def _get_osm_filter(network_type):
    if False:
        i = 10
        return i + 15
    '\n    Create a filter to query OSM for the specified network type.\n\n    Parameters\n    ----------\n    network_type : string {"all_private", "all", "bike", "drive", "drive_service", "walk"}\n        what type of street network to get\n\n    Returns\n    -------\n    string\n    '
    filters = {}
    filters['drive'] = f'["highway"]["area"!~"yes"]{settings.default_access}["highway"!~"abandoned|bridleway|bus_guideway|construction|corridor|cycleway|elevator|escalator|footway|no|path|pedestrian|planned|platform|proposed|raceway|razed|service|steps|track"]["motor_vehicle"!~"no"]["motorcar"!~"no"]["service"!~"alley|driveway|emergency_access|parking|parking_aisle|private"]'
    filters['drive_service'] = f'["highway"]["area"!~"yes"]{settings.default_access}["highway"!~"abandoned|bridleway|bus_guideway|construction|corridor|cycleway|elevator|escalator|footway|no|path|pedestrian|planned|platform|proposed|raceway|razed|steps|track"]["motor_vehicle"!~"no"]["motorcar"!~"no"]["service"!~"emergency_access|parking|parking_aisle|private"]'
    filters['walk'] = f'["highway"]["area"!~"yes"]{settings.default_access}["highway"!~"abandoned|bus_guideway|construction|cycleway|motor|no|planned|platform|proposed|raceway|razed"]["foot"!~"no"]["service"!~"private"]'
    filters['bike'] = f'["highway"]["area"!~"yes"]{settings.default_access}["highway"!~"abandoned|bus_guideway|construction|corridor|elevator|escalator|footway|motor|no|planned|platform|proposed|raceway|razed|steps"]["bicycle"!~"no"]["service"!~"private"]'
    filters['all'] = f'["highway"]["area"!~"yes"]{settings.default_access}["highway"!~"abandoned|construction|no|planned|platform|proposed|raceway|razed"]["service"!~"private"]'
    filters['all_private'] = '["highway"]["area"!~"yes"]["highway"!~"abandoned|construction|no|planned|platform|proposed|raceway|razed"]'
    if network_type in filters:
        osm_filter = filters[network_type]
    else:
        msg = f'Unrecognized network_type {network_type!r}'
        raise ValueError(msg)
    return osm_filter

def _get_overpass_pause(base_endpoint, recursive_delay=5, default_duration=60):
    if False:
        return 10
    '\n    Retrieve a pause duration from the Overpass API status endpoint.\n\n    Check the Overpass API status endpoint to determine how long to wait until\n    the next slot is available. You can disable this via the `settings`\n    module\'s `overpass_rate_limit` setting.\n\n    Parameters\n    ----------\n    base_endpoint : string\n        base Overpass API url (without "/status" at the end)\n    recursive_delay : int\n        how long to wait between recursive calls if the server is currently\n        running a query\n    default_duration : int\n        if fatal error, fall back on returning this value\n\n    Returns\n    -------\n    pause : int\n    '
    if not settings.overpass_rate_limit:
        return 0
    try:
        url = base_endpoint.rstrip('/') + '/status'
        response = requests.get(url, headers=_downloader._get_http_headers(), timeout=settings.timeout, **settings.requests_kwargs)
        status = response.text.split('\n')[4]
        status_first_token = status.split(' ')[0]
    except ConnectionError:
        utils.log(f'Unable to query {url}, got status {response.status_code}', level=lg.ERROR)
        return default_duration
    except (AttributeError, IndexError, ValueError):
        utils.log(f'Unable to parse {url} response: {response.text}', level=lg.ERROR)
        return default_duration
    try:
        _ = int(status_first_token)
        pause = 0
    except ValueError:
        if status_first_token == 'Slot':
            utc_time_str = status.split(' ')[3]
            pattern = '%Y-%m-%dT%H:%M:%SZ,'
            utc_time = dt.datetime.strptime(utc_time_str, pattern).astimezone(dt.timezone.utc)
            utc_now = dt.datetime.now(tz=dt.timezone.utc)
            seconds = int(np.ceil((utc_time - utc_now).total_seconds()))
            pause = max(seconds, 1)
        elif status_first_token == 'Currently':
            time.sleep(recursive_delay)
            pause = _get_overpass_pause(base_endpoint)
        else:
            utils.log(f'Unrecognized server status: {status!r}', level=lg.ERROR)
            return default_duration
    return pause

def _make_overpass_settings():
    if False:
        print('Hello World!')
    '\n    Make settings string to send in Overpass query.\n\n    Returns\n    -------\n    string\n    '
    if settings.memory is None:
        maxsize = ''
    else:
        maxsize = f'[maxsize:{settings.memory}]'
    return settings.overpass_settings.format(timeout=settings.timeout, maxsize=maxsize)

def _make_overpass_polygon_coord_strs(polygon):
    if False:
        i = 10
        return i + 15
    '\n    Subdivide query polygon and return list of coordinate strings.\n\n    Project to utm, divide polygon up into sub-polygons if area exceeds a\n    max size (in meters), project back to lat-lon, then get a list of\n    polygon(s) exterior coordinates\n\n    Parameters\n    ----------\n    polygon : shapely.geometry.Polygon or shapely.geometry.MultiPolygon\n\n    Returns\n    -------\n    polygon_coord_strs : list\n        list of exterior coordinate strings for smaller sub-divided polygons\n    '
    (geometry_proj, crs_proj) = projection.project_geometry(polygon)
    gpcs = utils_geo._consolidate_subdivide_geometry(geometry_proj)
    (geometry, _) = projection.project_geometry(gpcs, crs=crs_proj, to_latlong=True)
    return utils_geo._get_polygons_coordinates(geometry)

def _create_overpass_query(polygon_coord_str, tags):
    if False:
        return 10
    '\n    Create an Overpass features query string based on passed tags.\n\n    Parameters\n    ----------\n    polygon_coord_str : list\n        list of lat lon coordinates\n    tags : dict\n        dict of tags used for finding elements in the search area\n\n    Returns\n    -------\n    query : string\n    '
    overpass_settings = _make_overpass_settings()
    err_msg = 'tags must be a dict with values of bool, str, or list of str'
    if not isinstance(tags, dict):
        raise TypeError(err_msg)
    tags_dict = {}
    for (key, value) in tags.items():
        if isinstance(value, bool):
            tags_dict[key] = value
        elif isinstance(value, str):
            tags_dict[key] = [value]
        elif isinstance(value, list):
            if not all((isinstance(s, str) for s in value)):
                raise TypeError(err_msg)
            tags_dict[key] = value
        else:
            raise TypeError(err_msg)
    tags_list = []
    for (key, value) in tags_dict.items():
        if isinstance(value, bool):
            tags_list.append({key: value})
        else:
            for value_item in value:
                tags_list.append({key: value_item})
    components = []
    for d in tags_list:
        for (key, value) in d.items():
            if isinstance(value, bool):
                tag_str = f'[{key!r}](poly:{polygon_coord_str!r});(._;>;);'
            else:
                tag_str = f'[{key!r}={value!r}](poly:{polygon_coord_str!r});(._;>;);'
            for kind in ('node', 'way', 'relation'):
                components.append(f'({kind}{tag_str});')
    components = ''.join(components)
    return f'{overpass_settings};({components});out;'

def _download_overpass_network(polygon, network_type, custom_filter):
    if False:
        print('Hello World!')
    '\n    Retrieve networked ways and nodes within boundary from the Overpass API.\n\n    Parameters\n    ----------\n    polygon : shapely.geometry.Polygon or shapely.geometry.MultiPolygon\n        boundary to fetch the network ways/nodes within\n    network_type : string\n        what type of street network to get if custom_filter is None\n    custom_filter : string\n        a custom "ways" filter to be used instead of the network_type presets\n\n    Yields\n    ------\n    response_json : dict\n        a generator of JSON responses from the Overpass server\n    '
    if custom_filter is not None:
        osm_filter = custom_filter
    else:
        osm_filter = _get_osm_filter(network_type)
    overpass_settings = _make_overpass_settings()
    polygon_coord_strs = _make_overpass_polygon_coord_strs(polygon)
    utils.log(f'Requesting data from API in {len(polygon_coord_strs)} request(s)')
    for polygon_coord_str in polygon_coord_strs:
        query_str = f'{overpass_settings};(way{osm_filter}(poly:{polygon_coord_str!r});>;);out;'
        yield _overpass_request(data={'data': query_str})

def _download_overpass_features(polygon, tags):
    if False:
        while True:
            i = 10
    '\n    Retrieve OSM features within boundary from the Overpass API.\n\n    Parameters\n    ----------\n    polygon : shapely.geometry.Polygon\n        boundaries to fetch elements within\n    tags : dict\n        dict of tags used for finding elements in the selected area\n\n    Yields\n    ------\n    response_json : dict\n        a generator of JSON responses from the Overpass server\n    '
    polygon_coord_strs = _make_overpass_polygon_coord_strs(polygon)
    utils.log(f'Requesting data from API in {len(polygon_coord_strs)} request(s)')
    for polygon_coord_str in polygon_coord_strs:
        query_str = _create_overpass_query(polygon_coord_str, tags)
        yield _overpass_request(data={'data': query_str})

def _overpass_request(data, pause=None, error_pause=60):
    if False:
        i = 10
        return i + 15
    '\n    Send a HTTP POST request to the Overpass API and return response.\n\n    Parameters\n    ----------\n    data : OrderedDict\n        key-value pairs of parameters\n    pause : float\n        how long to pause in seconds before request, if None, will query API\n        status endpoint to find when next slot is available\n    error_pause : float\n        how long to pause in seconds (in addition to `pause`) before re-trying\n        request if error\n\n    Returns\n    -------\n    response_json : dict\n    '
    _downloader._config_dns(settings.overpass_endpoint)
    url = settings.overpass_endpoint.rstrip('/') + '/interpreter'
    prepared_url = requests.Request('GET', url, params=data).prepare().url
    cached_response_json = _downloader._retrieve_from_cache(prepared_url)
    if cached_response_json is not None:
        return cached_response_json
    if pause is None:
        this_pause = _get_overpass_pause(settings.overpass_endpoint)
    domain = _downloader._hostname_from_url(url)
    utils.log(f'Pausing {this_pause} second(s) before making HTTP POST request to {domain!r}')
    time.sleep(this_pause)
    utils.log(f'Post {prepared_url} with timeout={settings.timeout}')
    response = requests.post(url, data=data, timeout=settings.timeout, headers=_downloader._get_http_headers(), **settings.requests_kwargs)
    if response.status_code in {429, 504}:
        this_pause = error_pause + _get_overpass_pause(settings.overpass_endpoint)
        msg = f"{domain!r} responded {response.status_code} {response.reason}: we'll retry in {this_pause} secs"
        utils.log(msg, level=lg.WARNING)
        time.sleep(this_pause)
        return _overpass_request(data, pause, error_pause)
    response_json = _downloader._parse_response(response)
    _downloader._save_to_cache(prepared_url, response_json, response.status_code)
    return response_json