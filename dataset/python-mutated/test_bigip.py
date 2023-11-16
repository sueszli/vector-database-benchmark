"""
tests.unit.modules.test_bigip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for the bigip module
"""
import logging
import pytest
import salt.modules.bigip as bigip
from tests.support.mock import MagicMock, patch
log = logging.getLogger(__name__)

class RequestsSession:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.auth = None
        self.verify = None
        self.headers = {}

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {bigip: {}}

def test__build_session_verify_ssl():
    if False:
        print('Hello World!')
    requests_session = RequestsSession()
    with patch('salt.modules.bigip.requests.sessions.Session', MagicMock(return_value=requests_session)):
        bigip._build_session('username', 'password')
    assert requests_session.auth == ('username', 'password')
    assert requests_session.verify is True