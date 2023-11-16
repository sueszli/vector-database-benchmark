"""
    :codeauthor: `Andreas Thienemann <andreas@bawue.net>`

    tests.unit.cloud.clouds.xen_test
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import logging
import pytest
from salt.cloud.clouds import xen
from tests.support.mock import MagicMock, patch
log = logging.getLogger(__name__)

@pytest.fixture
def configure_loader_modules():
    if False:
        while True:
            i = 10
    return {xen: {'__active_provider_name__': '', '__opts__': {'providers': {'my-xen-cloud': {'xen': {'driver': 'xen', 'user': 'SantaClaus', 'password': 'TooManyElves', 'url': 'https://127.0.0.2'}}}}}}

def test_get_configured_provider_bad():
    if False:
        print('Hello World!')
    with patch.dict(xen.__opts__, {'providers': {}}):
        result = xen.get_configured_provider()
        assert not result

def test_get_configured_provider_auth():
    if False:
        i = 10
        return i + 15
    config = {'url': 'https://127.0.0.2'}
    with patch.dict(xen.__opts__, {'providers': {'my-xen-cloud': {'xen': config}}}):
        result = xen.get_configured_provider()
        assert config == result

def test_get_dependencies():
    if False:
        i = 10
        return i + 15
    with patch('salt.cloud.clouds.xen.HAS_XEN_API', True):
        result = xen._get_dependencies()
        assert result

def test_get_dependencies_no_xenapi():
    if False:
        return 10
    with patch('salt.cloud.clouds.xen.HAS_XEN_API', False):
        result = xen._get_dependencies()
        assert not result

def test_get_vm():
    if False:
        while True:
            i = 10
    XenAPI = MagicMock(name='mock_session')
    XenAPI.xenapi.VM.get_by_name_label = MagicMock(return_value=['0000'])
    XenAPI.xenapi.VM.get_is_a_template = MagicMock(return_value=False)
    with patch('salt.cloud.clouds.xen._get_session', MagicMock(return_value=XenAPI)):
        result = xen._get_vm(name='test')
        assert result == '0000'

def test_get_vm_multiple():
    if False:
        for i in range(10):
            print('nop')
    'Verify correct behavior if VM and template is returned'
    vms = {'0000': False, '0001': True}
    XenAPI = MagicMock(name='mock_session')
    XenAPI.xenapi.VM.get_by_name_label = MagicMock(return_value=vms.keys())
    XenAPI.xenapi.VM.get_is_a_template = MagicMock(side_effect=lambda x: vms[x])
    with patch('salt.cloud.clouds.xen._get_session', MagicMock(return_value=XenAPI)):
        result = xen._get_vm(name='test')
        assert result == '0000'