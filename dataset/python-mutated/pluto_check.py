import logging
import secrets
import time
import requests
from tests.conftest import TrackedContainer, find_free_port
LOGGER = logging.getLogger(__name__)

def check_pluto_proxy(container: TrackedContainer, http_client: requests.Session) -> None:
    if False:
        i = 10
        return i + 15
    host_port = find_free_port()
    token = secrets.token_hex()
    container.run_detached(command=['start-notebook.py', f'--IdentityProvider.token={token}'], ports={'8888/tcp': host_port})
    time.sleep(3)
    resp = http_client.get(f'http://localhost:{host_port}/pluto?token={token}')
    resp.raise_for_status()
    assert 'Pluto.jl notebooks' in resp.text, 'Pluto.jl text not found in /pluto page'