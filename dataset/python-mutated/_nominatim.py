"""Tools to work with the Nominatim API."""
import logging as lg
import time
from collections import OrderedDict
import requests
from . import _downloader
from . import settings
from . import utils

def _download_nominatim_element(query, by_osmid=False, limit=1, polygon_geojson=1):
    if False:
        for i in range(10):
            print('nop')
    "\n    Retrieve an OSM element from the Nominatim API.\n\n    Parameters\n    ----------\n    query : string or dict\n        query string or structured query dict\n    by_osmid : bool\n        if True, treat query as an OSM ID lookup rather than text search\n    limit : int\n        max number of results to return\n    polygon_geojson : int\n        retrieve the place's geometry from the API, 0=no, 1=yes\n\n    Returns\n    -------\n    response_json : dict\n        JSON response from the Nominatim server\n    "
    params = OrderedDict()
    params['format'] = 'json'
    params['polygon_geojson'] = polygon_geojson
    if by_osmid:
        request_type = 'lookup'
        params['osm_ids'] = query
    else:
        request_type = 'search'
        params['dedupe'] = 0
        params['limit'] = limit
        if isinstance(query, str):
            params['q'] = query
        elif isinstance(query, dict):
            for key in sorted(query):
                params[key] = query[key]
        else:
            msg = 'query must be a dict or a string'
            raise TypeError(msg)
    return _nominatim_request(params=params, request_type=request_type)

def _nominatim_request(params, request_type='search', pause=1, error_pause=60):
    if False:
        for i in range(10):
            print('nop')
    '\n    Send a HTTP GET request to the Nominatim API and return response.\n\n    Parameters\n    ----------\n    params : OrderedDict\n        key-value pairs of parameters\n    request_type : string {"search", "reverse", "lookup"}\n        which Nominatim API endpoint to query\n    pause : float\n        how long to pause before request, in seconds. per the nominatim usage\n        policy: "an absolute maximum of 1 request per second" is allowed\n    error_pause : float\n        how long to pause in seconds before re-trying request if error\n\n    Returns\n    -------\n    response_json : dict\n    '
    if request_type not in {'search', 'reverse', 'lookup'}:
        msg = 'Nominatim request_type must be "search", "reverse", or "lookup"'
        raise ValueError(msg)
    url = settings.nominatim_endpoint.rstrip('/') + '/' + request_type
    params['key'] = settings.nominatim_key
    prepared_url = requests.Request('GET', url, params=params).prepare().url
    cached_response_json = _downloader._retrieve_from_cache(prepared_url)
    if cached_response_json is not None:
        return cached_response_json
    domain = _downloader._hostname_from_url(url)
    utils.log(f'Pausing {pause} second(s) before making HTTP GET request to {domain!r}')
    time.sleep(pause)
    utils.log(f'Get {prepared_url} with timeout={settings.timeout}')
    response = requests.get(url, params=params, timeout=settings.timeout, headers=_downloader._get_http_headers(), **settings.requests_kwargs)
    if response.status_code in {429, 504}:
        msg = f"{domain!r} responded {response.status_code} {response.reason}: we'll retry in {error_pause} secs"
        utils.log(msg, level=lg.WARNING)
        time.sleep(error_pause)
        return _nominatim_request(params, request_type, pause, error_pause)
    response_json = _downloader._parse_response(response)
    _downloader._save_to_cache(prepared_url, response_json, response.status_code)
    return response_json