"""
This script is an example of how to manipulate cookies both outgoing (requests)
and ingoing (responses). In particular, this script inserts a cookie (specified
in a json file) into every request (overwriting any existing cookie of the same
name), and removes cookies from every response that have a certain set of names
specified in the variable (set) FILTER_COOKIES.

Usage:

    mitmproxy -s examples/contrib/http_manipulate_cookies.py

Note:
    this was created as a response to SO post:
    https://stackoverflow.com/questions/55358072/cookie-manipulation-in-mitmproxy-requests-and-responses

"""
import json
from mitmproxy import http
PATH_TO_COOKIES = './cookies.json'
FILTER_COOKIES = {'mycookie', '_ga'}

def load_json_cookies() -> list[dict[str, str | None]]:
    if False:
        while True:
            i = 10
    '\n    Load a particular json file containing a list of cookies.\n    '
    with open(PATH_TO_COOKIES) as f:
        return json.load(f)

def stringify_cookies(cookies: list[dict[str, str | None]]) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Creates a cookie string from a list of cookie dicts.\n    '
    return '; '.join([f"{c['name']}={c['value']}" if c.get('value', None) is not None else f"{c['name']}" for c in cookies])

def parse_cookies(cookie_string: str) -> list[dict[str, str | None]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Parses a cookie string into a list of cookie dicts.\n    '
    return [{'name': g[0], 'value': g[1]} if len(g) == 2 else {'name': g[0], 'value': None} for g in [k.split('=', 1) for k in [c.strip() for c in cookie_string.split(';')] if k]]

def request(flow: http.HTTPFlow) -> None:
    if False:
        return 10
    'Add a specific set of cookies to every request.'
    _req_cookies_str = flow.request.headers.get('cookie', '')
    req_cookies = parse_cookies(_req_cookies_str)
    all_cookies = req_cookies + load_json_cookies()
    flow.request.headers['cookie'] = stringify_cookies(all_cookies)

def response(flow: http.HTTPFlow) -> None:
    if False:
        return 10
    'Remove a specific cookie from every response.'
    set_cookies_str = flow.response.headers.get_all('set-cookie')
    set_cookies_str_modified: list[str] = []
    if set_cookies_str:
        for cookie in set_cookies_str:
            resp_cookies = parse_cookies(cookie)
            resp_cookies = [c for c in resp_cookies if c['name'] not in FILTER_COOKIES]
            set_cookies_str_modified.append(stringify_cookies(resp_cookies))
        flow.response.headers.set_all('set-cookie', set_cookies_str_modified)