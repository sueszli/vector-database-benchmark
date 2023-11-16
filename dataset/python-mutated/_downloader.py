"""Handle HTTP requests to web APIs."""
import json
import logging as lg
import socket
from hashlib import sha1
from pathlib import Path
from urllib.parse import urlparse
import requests
from requests.exceptions import JSONDecodeError
from . import settings
from . import utils
from ._errors import InsufficientResponseError
from ._errors import ResponseStatusCodeError
_original_getaddrinfo = socket.getaddrinfo

def _save_to_cache(url, response_json, ok):
    if False:
        return 10
    "\n    Save a HTTP response JSON object to a file in the cache folder.\n\n    Function calculates the checksum of url to generate the cache file's name.\n    If the request was sent to server via POST instead of GET, then URL should\n    be a GET-style representation of request. Response is only saved to a\n    cache file if settings.use_cache is True, response_json is not None, and\n    ok is True.\n\n    Users should always pass OrderedDicts instead of dicts of parameters into\n    request functions, so the parameters remain in the same order each time,\n    producing the same URL string, and thus the same hash. Otherwise the cache\n    will eventually contain multiple saved responses for the same request\n    because the URL's parameters appeared in a different order each time.\n\n    Parameters\n    ----------\n    url : string\n        the URL of the request\n    response_json : dict\n        the JSON response\n    ok : bool\n        requests response.ok value\n\n    Returns\n    -------\n    None\n    "
    if settings.use_cache:
        if not ok:
            utils.log('Did not save to cache because response status_code is not OK')
        elif response_json is None:
            utils.log('Did not save to cache because response_json is None')
        else:
            cache_folder = Path(settings.cache_folder)
            cache_folder.mkdir(parents=True, exist_ok=True)
            filename = sha1(url.encode('utf-8')).hexdigest() + '.json'
            cache_filepath = cache_folder / filename
            cache_filepath.write_text(json.dumps(response_json), encoding='utf-8')
            utils.log(f'Saved response to cache file {str(cache_filepath)!r}')

def _url_in_cache(url):
    if False:
        return 10
    "\n    Determine if a URL's response exists in the cache.\n\n    Calculates the checksum of url to determine the cache file's name.\n\n    Parameters\n    ----------\n    url : string\n        the URL to look for in the cache\n\n    Returns\n    -------\n    filepath : pathlib.Path\n        path to cached response for url if it exists, otherwise None\n    "
    filename = sha1(url.encode('utf-8')).hexdigest() + '.json'
    filepath = Path(settings.cache_folder) / filename
    return filepath if filepath.is_file() else None

def _retrieve_from_cache(url, check_remark=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Retrieve a HTTP response JSON object from the cache, if it exists.\n\n    Parameters\n    ----------\n    url : string\n        the URL of the request\n    check_remark : string\n        if True, only return filepath if cached response does not have a\n        remark key indicating a server warning\n\n    Returns\n    -------\n    response_json : dict\n        cached response for url if it exists in the cache, otherwise None\n    '
    if settings.use_cache:
        cache_filepath = _url_in_cache(url)
        if cache_filepath is not None:
            response_json = json.loads(cache_filepath.read_text(encoding='utf-8'))
            if check_remark and 'remark' in response_json:
                utils.log(f"Ignoring cache file {str(cache_filepath)!r} because it contains a remark: {response_json['remark']!r}")
                return None
            utils.log(f'Retrieved response from cache file {str(cache_filepath)!r}')
            return response_json
    return None

def _get_http_headers(user_agent=None, referer=None, accept_language=None):
    if False:
        i = 10
        return i + 15
    '\n    Update the default requests HTTP headers with OSMnx info.\n\n    Parameters\n    ----------\n    user_agent : string\n        the user agent string, if None will set with OSMnx default\n    referer : string\n        the referer string, if None will set with OSMnx default\n    accept_language : string\n        make accept-language explicit e.g. for consistent nominatim result\n        sorting\n\n    Returns\n    -------\n    headers : dict\n    '
    if user_agent is None:
        user_agent = settings.default_user_agent
    if referer is None:
        referer = settings.default_referer
    if accept_language is None:
        accept_language = settings.default_accept_language
    headers = requests.utils.default_headers()
    headers.update({'User-Agent': user_agent, 'referer': referer, 'Accept-Language': accept_language})
    return headers

def _resolve_host_via_doh(hostname):
    if False:
        while True:
            i = 10
    "\n    Resolve hostname to IP address via Google's public DNS-over-HTTPS API.\n\n    Necessary fallback as socket.gethostbyname will not always work when using\n    a proxy. See https://developers.google.com/speed/public-dns/docs/doh/json\n    If the user has set `settings.doh_url_template=None` or if resolution\n    fails (e.g., due to local network blocking DNS-over-HTTPS) the hostname\n    itself will be returned instead. Note that this means that server slot\n    management may be violated: see `_config_dns` documentation for details.\n\n    Parameters\n    ----------\n    hostname : string\n        the hostname to consistently resolve the IP address of\n\n    Returns\n    -------\n    ip_address : string\n        resolved IP address of host, or hostname itself if resolution failed\n    "
    if settings.doh_url_template is None:
        utils.log('User set `doh_url_template=None`, requesting host by name', level=lg.WARNING)
        return hostname
    err_msg = f'Failed to resolve {hostname!r} IP via DoH, requesting host by name'
    try:
        url = settings.doh_url_template.format(hostname=hostname)
        response = requests.get(url, timeout=settings.timeout)
        data = response.json()
    except requests.exceptions.RequestException:
        utils.log(err_msg, level=lg.ERROR)
        return hostname
    else:
        if response.ok and data['Status'] == 0:
            return data['Answer'][0]['data']
        utils.log(err_msg, level=lg.ERROR)
        return hostname

def _config_dns(url):
    if False:
        i = 10
        return i + 15
    "\n    Force socket.getaddrinfo to use IP address instead of hostname.\n\n    Resolves the URL's domain to an IP address so that we use the same server\n    for both 1) checking the necessary pause duration and 2) sending the query\n    itself even if there is round-robin redirecting among multiple server\n    machines on the server-side. Mutates the getaddrinfo function so it uses\n    the same IP address everytime it finds the hostname in the URL.\n\n    For example, the server overpass-api.de just redirects to one of the other\n    servers (currently gall.openstreetmap.de and lambert.openstreetmap.de). So\n    if we check the status endpoint of overpass-api.de, we may see results for\n    server gall, but when we submit the query itself it gets redirected to\n    server lambert. This could result in violating server lambert's slot\n    management timing.\n\n    Parameters\n    ----------\n    url : string\n        the URL to consistently resolve the IP address of\n\n    Returns\n    -------\n    None\n    "
    hostname = _hostname_from_url(url)
    try:
        ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        utils.log(f'Encountered gaierror while trying to resolve {hostname!r}, trying again via DoH...', level=lg.ERROR)
        ip = _resolve_host_via_doh(hostname)

    def _getaddrinfo(*args, **kwargs):
        if False:
            return 10
        if args[0] == hostname:
            utils.log(f'Resolved {hostname!r} to {ip!r}')
            return _original_getaddrinfo(ip, *args[1:], **kwargs)
        return _original_getaddrinfo(*args, **kwargs)
    socket.getaddrinfo = _getaddrinfo

def _hostname_from_url(url):
    if False:
        i = 10
        return i + 15
    '\n    Extract the hostname (domain) from a URL.\n\n    Parameters\n    ----------\n    url : string\n        the url from which to extract the hostname\n\n    Returns\n    -------\n    hostname : string\n        the extracted hostname (domain)\n    '
    return urlparse(url).netloc.split(':')[0]

def _parse_response(response):
    if False:
        return 10
    '\n    Parse JSON from a requests response and log the details.\n\n    Parameters\n    ----------\n    response : requests.response\n        the response object\n\n    Returns\n    -------\n    response_json : dict\n    '
    domain = _hostname_from_url(response.url)
    size_kb = len(response.content) / 1000
    utils.log(f'Downloaded {size_kb:,.1f}kB from {domain!r} with status {response.status_code}')
    try:
        response_json = response.json()
    except JSONDecodeError as e:
        msg = f'{domain!r} responded: {response.status_code} {response.reason} {response.text}'
        utils.log(msg, level=lg.ERROR)
        if response.ok:
            raise InsufficientResponseError(msg) from e
        raise ResponseStatusCodeError(msg) from e
    if 'remark' in response_json:
        utils.log(f"{domain!r} remarked: {response_json['remark']!r}", level=lg.WARNING)
    return response_json