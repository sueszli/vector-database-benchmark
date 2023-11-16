import os
import pytest
import requests
here = os.path.dirname(__file__)
NETWORK_PORT = 9081
DOMAIN_PORT = 9082
HOST_IP = os.environ.get('HOST_IP', 'localhost')

@pytest.mark.frontend
def test_serves_domain_frontend() -> None:
    if False:
        i = 10
        return i + 15
    title_str = 'PyGrid'
    url = f'http://{HOST_IP}:{DOMAIN_PORT}'
    result = requests.get(url)
    assert result.status_code == 200
    assert title_str in result.text

@pytest.mark.frontend
def test_serves_network_frontend() -> None:
    if False:
        for i in range(10):
            print('nop')
    title_str = 'PyGrid'
    url = f'http://localhost:{NETWORK_PORT}'
    result = requests.get(url)
    assert result.status_code == 200
    assert title_str in result.text