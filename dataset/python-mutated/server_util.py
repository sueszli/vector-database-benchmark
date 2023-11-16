"""Server related utility functions"""
from typing import Optional
from urllib.parse import urljoin
import tornado.web
from streamlit import config, net_util, url_util

def is_url_from_allowed_origins(url: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Return True if URL is from allowed origins (for CORS purpose).\n\n    Allowed origins:\n    1. localhost\n    2. The internal and external IP addresses of the machine where this\n       function was called from.\n\n    If `server.enableCORS` is False, this allows all origins.\n    '
    if not config.get_option('server.enableCORS'):
        return True
    hostname = url_util.get_hostname(url)
    allowed_domains = ['localhost', '0.0.0.0', '127.0.0.1', _get_server_address_if_manually_set, net_util.get_internal_ip, net_util.get_external_ip]
    for allowed_domain in allowed_domains:
        if callable(allowed_domain):
            allowed_domain = allowed_domain()
        if allowed_domain is None:
            continue
        if hostname == allowed_domain:
            return True
    return False

def _get_server_address_if_manually_set() -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    if config.is_manually_set('browser.serverAddress'):
        return url_util.get_hostname(config.get_option('browser.serverAddress'))
    return None

def make_url_path_regex(*path, **kwargs) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Get a regex of the form ^/foo/bar/baz/?$ for a path (foo, bar, baz).'
    path = [x.strip('/') for x in path if x]
    path_format = '^/%s/?$' if kwargs.get('trailing_slash', True) else '^/%s$'
    return path_format % '/'.join(path)

def get_url(host_ip: str) -> str:
    if False:
        print('Hello World!')
    'Get the URL for any app served at the given host_ip.\n\n    Parameters\n    ----------\n    host_ip : str\n        The IP address of the machine that is running the Streamlit Server.\n\n    Returns\n    -------\n    str\n        The URL.\n    '
    protocol = 'https' if config.get_option('server.sslCertFile') else 'http'
    port = _get_browser_address_bar_port()
    base_path = config.get_option('server.baseUrlPath').strip('/')
    if base_path:
        base_path = '/' + base_path
    host_ip = host_ip.strip('/')
    return f'{protocol}://{host_ip}:{port}{base_path}'

def _get_browser_address_bar_port() -> int:
    if False:
        while True:
            i = 10
    "Get the app URL that will be shown in the browser's address bar.\n\n    That is, this is the port where static assets will be served from. In dev,\n    this is different from the URL that will be used to connect to the\n    server-browser websocket.\n\n    "
    if config.get_option('global.developmentMode'):
        return 3000
    return int(config.get_option('browser.serverPort'))

def emit_endpoint_deprecation_notice(handler: tornado.web.RequestHandler, new_path: str) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Emits the warning about deprecation of HTTP endpoint in the HTTP header.\n    '
    handler.set_header('Deprecation', True)
    new_url = urljoin(f'{handler.request.protocol}://{handler.request.host}', new_path)
    handler.set_header('Link', f'<{new_url}>; rel="alternate"')