import re
from http.client import HTTP_PORT, HTTPS_PORT
from json import dumps
from urllib.parse import ParseResult, parse_qsl, unquote, urlencode, urlparse
UDP = 'udp'
HTTP = 'http'
HTTPS = 'https'
SUPPORTED_SCHEMES = {UDP, HTTP, HTTPS}
DEFAULT_PORTS = {HTTP: HTTP_PORT, HTTPS: HTTPS_PORT}

class MalformedTrackerURLException(Exception):
    pass
delimiters_regex = re.compile('[\\r\\n\\x00\\s\\t;]+(%20)*')
url_regex = re.compile('^(?:http|udp|wss)s?://(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\\.)+(?:[A-Z]{2,6}\\.?|[A-Z0-9-]{2,}\\.?)|localhost|\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3})(?::\\d+)?(?:/?|[/?]\\S+)$', re.IGNORECASE)
remove_trailing_junk = re.compile('[,*.:]+\\Z')
truncated_url_detector = re.compile('\\.\\.\\.')

def get_uniformed_tracker_url(tracker_url: str):
    if False:
        return 10
    '\n    Parses the given tracker URL and returns in a uniform URL format.\n    It uses regex to sanitize the URL.\n\n    :param tracker_url: Tracker URL\n    :return: the tracker in a uniform format <type>://<host>:<port>/<page>\n    '
    assert isinstance(tracker_url, str), f'tracker_url is not a str: {type(tracker_url)}'
    for tracker_url in re.split(delimiters_regex, tracker_url):
        if not tracker_url:
            continue
        if re.search(truncated_url_detector, tracker_url):
            continue
        if not re.match(url_regex, tracker_url):
            continue
        tracker_url = re.sub(remove_trailing_junk, '', tracker_url)
        try:
            (scheme, (host, port), path) = _parse_tracker_url(tracker_url)
            if scheme == UDP:
                return f'{scheme}://{host}:{port}'
            if scheme in {HTTP, HTTPS}:
                path = path.rstrip('/')
                if not path:
                    continue
                uniformed_port = '' if port == DEFAULT_PORTS[scheme] else f':{port}'
                return f'{scheme}://{host}{uniformed_port}{path}'
        except MalformedTrackerURLException:
            continue
    return None

def parse_tracker_url(tracker_url):
    if False:
        print('Hello World!')
    '\n    Parses the tracker URL and checks whether it satisfies tracker URL constraints.\n    Additionally, it also checks if the tracker URL is a uniform and valid URL.\n\n    :param tracker_url the URL of the tracker\n    :returns: Tuple (scheme, (host, port), announce_path)\n    '
    http_prefix = f'{HTTP}://'
    http_port_suffix = f':{HTTP_PORT}/'
    https_prefix = f'{HTTPS}://'
    https_port_suffix = f':{HTTPS_PORT}/'
    url = tracker_url.lower()
    if url.startswith(http_prefix) and http_port_suffix in url:
        tracker_url = tracker_url.replace(http_port_suffix, '/', 1)
    if url.startswith(https_prefix) and https_port_suffix in url:
        tracker_url = tracker_url.replace(https_port_suffix, '/', 1)
    if tracker_url != get_uniformed_tracker_url(tracker_url):
        raise MalformedTrackerURLException(f'Tracker URL is not sanitized ({tracker_url}).')
    return _parse_tracker_url(tracker_url)

def _parse_tracker_url(tracker_url):
    if False:
        print('Hello World!')
    '\n    Parses the tracker URL and check whether it satisfies certain constraints:\n\n        - The tracker type must be one of the supported types (udp, http, https).\n        - UDP trackers requires a port.\n        - HTTP(s) trackers requires an announce path.\n        - HTTP(S) trackers default to HTTP(S)_PORT if port is not present on the URL.\n\n    :param tracker_url the URL of the tracker\n    :returns: Tuple (scheme, (host, port), announce_path)\n    '
    parsed_url = urlparse(tracker_url)
    host = parsed_url.hostname
    path = parsed_url.path
    scheme = parsed_url.scheme
    port = parsed_url.port
    if scheme not in SUPPORTED_SCHEMES:
        raise MalformedTrackerURLException(f'Unsupported tracker type ({scheme}).')
    if scheme == UDP and (not port):
        raise MalformedTrackerURLException(f'Missing port for UDP tracker URL ({tracker_url}).')
    if scheme in {HTTP, HTTPS}:
        if not path:
            raise MalformedTrackerURLException(f'Missing announce path for HTTP(S) tracker URL ({tracker_url}).')
        if not port:
            port = DEFAULT_PORTS[scheme]
    return (scheme, (host, port), path)

def add_url_params(url, params):
    if False:
        i = 10
        return i + 15
    " Add GET params to provided URL being aware of existing.\n    :param url: string of target URL\n    :param params: dict containing requested params to be added\n    :return: string with updated URL\n    >> url = 'http://stackoverflow.com/test?answers=true'\n    >> new_params = {'answers': False, 'data': ['some','values']}\n    >> add_url_params(url, new_params)\n    'http://stackoverflow.com/test?data=some&data=values&answers=false'\n    "
    url = unquote(url)
    parsed_url = urlparse(url)
    get_args = parsed_url.query
    parsed_get_args = dict(parse_qsl(get_args))
    parsed_get_args.update(params)
    parsed_get_args.update({k: dumps(v) for (k, v) in parsed_get_args.items() if isinstance(v, (bool, dict))})
    encoded_get_args = urlencode(parsed_get_args, doseq=True)
    new_url = ParseResult(parsed_url.scheme, parsed_url.netloc, parsed_url.path, parsed_url.params, encoded_get_args, parsed_url.fragment).geturl()
    return new_url