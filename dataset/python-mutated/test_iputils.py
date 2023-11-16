from ipaddress import IPv4Address
import pytest
from localstack.utils import iputils
pytestmark = [pytest.mark.skipif(condition=not iputils.ip_available, reason='ip command must be available')]

def test_ip_route_show():
    if False:
        return 10
    for _ in list(iputils.get_routes()):
        pass

def test_default_gateway():
    if False:
        return 10
    gateway = iputils.get_default_gateway()
    assert isinstance(gateway, IPv4Address)