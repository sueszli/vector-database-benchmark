import requests
from tests.conftest import TrackedContainer
from tests.pluto_check import check_pluto_proxy

def test_pluto_proxy(container: TrackedContainer, http_client: requests.Session) -> None:
    if False:
        print('Hello World!')
    'Pluto proxy starts Pluto correctly'
    check_pluto_proxy(container, http_client)