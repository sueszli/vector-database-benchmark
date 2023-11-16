"""
IP utils wraps the 'ip' cli command (from iproute2) and creates a pythonic OO interface around
some of its functionality.
"""
import ipaddress
import json
import subprocess as sp
from typing import Any, Generator, TypedDict
from cachetools import TTLCache, cached

def ip_available() -> bool:
    if False:
        while True:
            i = 10
    try:
        output = sp.run(['ip'], capture_output=True)
        return 'Usage:' in output.stderr.decode('utf8')
    except FileNotFoundError:
        return False

class Route(TypedDict):
    """
    Represents an entry in the routing table.
    """
    dst: str | ipaddress.IPv4Network
    dev: str
    protocol: str
    prefsrc: ipaddress.IPv4Address
    gateway: ipaddress.IPv4Address | None
    metric: int | None
    flags: list[str]
IP_RESULTS_CACHE = TTLCache(maxsize=100, ttl=10)

def get_routes() -> Generator[Route, None, None]:
    if False:
        print('Hello World!')
    '\n    Return a generator over the routes.\n\n    :return: a generator over route descriptions\n    '
    yield from _run_ip_command('route', 'show')

def get_route(name: str) -> Route:
    if False:
        print('Hello World!')
    '\n    Get information about a single route.\n\n    :param name: name of the route to get details for\n    :return: the route definition\n    '
    return _run_ip_command('route', 'show', name)[0]

def get_default_route() -> Route:
    if False:
        i = 10
        return i + 15
    '\n    Get information about the default route.\n\n    :return: the definition of the default route\n    '
    return get_route('default')

def get_default_gateway() -> ipaddress.IPv4Address:
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the IPv4 address for the default gateway.\n\n    :return: the IPv4 address for the default gateway\n    '
    return ipaddress.IPv4Address(get_default_route()['gateway'])

@cached(cache=IP_RESULTS_CACHE)
def _run_ip_command(*args) -> Any:
    if False:
        for i in range(10):
            print('nop')
    cmd = ['ip', '--json'] + list(args)
    try:
        result = sp.check_output(cmd)
    except FileNotFoundError:
        raise RuntimeError('could not find ip binary on path')
    return json.loads(result.decode('utf8'))