"""
    Unit test for salt.grains.metadata_gce


    :codeauthor: :email" `Thomas Phipps <tphipps@vmware.com>

"""
import logging
import pytest
import salt.grains.metadata_gce as metadata
import salt.utils.http as http
from tests.support.mock import create_autospec, patch
log = logging.getLogger(__name__)

@pytest.fixture
def configure_loader_modules():
    if False:
        print('Hello World!')
    return {metadata: {'__opts__': {'metadata_server_grains': 'True'}}}

def test_metadata_gce_search():
    if False:
        while True:
            i = 10

    def mock_http(url='', headers=False, header_list=None):
        if False:
            i = 10
            return i + 15
        metadata_vals = {'http://169.254.169.254/computeMetadata/v1/?alt=json&recursive=true': {'body': '{"instance": {"test": "fulltest"}}', 'headers': {'Content-Type': 'application/octet-stream', 'Metadata-Flavor': 'Google'}}}
        return metadata_vals[url]
    with patch('salt.utils.http.query', create_autospec(http.query, autospec=True, side_effect=mock_http)):
        assert metadata.metadata() == {'instance': {'test': 'fulltest'}}

def test_metadata_virtual():
    if False:
        return 10
    with patch('salt.utils.http.query', create_autospec(http.query, autospec=True, return_value={'error': '[Errno -2] Name or service not known'})):
        assert metadata.__virtual__() is False
    with patch('salt.utils.http.query', create_autospec(http.query, autospec=True, return_value={'body': 'test', 'headers': {'Metadata-Flavor': 'Google'}, 'status': 200})):
        assert metadata.__virtual__() is True
    with patch('salt.utils.http.query', create_autospec(http.query, autospec=True, return_value={'body': 'test', 'headers': {'Metadata-Flavor': 'Google'}, 'status': 404})):
        assert metadata.__virtual__() is False
    with patch('salt.utils.http.query', create_autospec(http.query, autospec=True, return_value={'body': 'test', 'headers': {}, 'code': 200})):
        assert metadata.__virtual__() is False