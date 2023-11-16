import pytest
import salt.beacons.smartos_vmadm as vmadm
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {vmadm: {'__context__': {}, '__salt__': {}}}

@pytest.fixture
def mock_clean_state():
    if False:
        for i in range(10):
            print('nop')
    return {'first_run': True, 'vms': []}

@pytest.fixture
def mock_vm_none():
    if False:
        i = 10
        return i + 15
    return {}

@pytest.fixture
def mock_vm_one():
    if False:
        for i in range(10):
            print('nop')
    return {'00000000-0000-0000-0000-000000000001': {'state': 'running', 'alias': 'vm1', 'hostname': 'vm1', 'dns_domain': 'example.org'}}

@pytest.fixture
def mock_vm_two_stopped():
    if False:
        while True:
            i = 10
    return {'00000000-0000-0000-0000-000000000001': {'state': 'running', 'alias': 'vm1', 'hostname': 'vm1', 'dns_domain': 'example.org'}, '00000000-0000-0000-0000-000000000002': {'state': 'stopped', 'alias': 'vm2', 'hostname': 'vm2', 'dns_domain': 'example.org'}}

@pytest.fixture
def mock_vm_two_started():
    if False:
        while True:
            i = 10
    return {'00000000-0000-0000-0000-000000000001': {'state': 'running', 'alias': 'vm1', 'hostname': 'vm1', 'dns_domain': 'example.org'}, '00000000-0000-0000-0000-000000000002': {'state': 'running', 'alias': 'vm2', 'hostname': 'vm2', 'dns_domain': 'example.org'}}

def test_non_list_config():
    if False:
        while True:
            i = 10
    '\n    We only have minimal validation so we test that here\n    '
    config = {}
    ret = vmadm.validate(config)
    assert ret == (False, 'Configuration for vmadm beacon must be a list!')

def test_created_startup(mock_clean_state, mock_vm_one):
    if False:
        i = 10
        return i + 15
    '\n    Test with one vm and startup_create_event\n    '
    with patch.dict(vmadm.VMADM_STATE, mock_clean_state), patch.dict(vmadm.__salt__, {'vmadm.list': MagicMock(return_value=mock_vm_one)}):
        config = [{'startup_create_event': True}]
        ret = vmadm.validate(config)
        assert ret == (True, 'Valid beacon configuration')
        ret = vmadm.beacon(config)
        res = [{'alias': 'vm1', 'tag': 'created/00000000-0000-0000-0000-000000000001', 'hostname': 'vm1', 'dns_domain': 'example.org'}, {'alias': 'vm1', 'tag': 'running/00000000-0000-0000-0000-000000000001', 'hostname': 'vm1', 'dns_domain': 'example.org'}]
        assert ret == res

def test_created_nostartup(mock_clean_state, mock_vm_one):
    if False:
        while True:
            i = 10
    '\n    Test with one image and startup_import_event unset/false\n    '
    with patch.dict(vmadm.VMADM_STATE, mock_clean_state), patch.dict(vmadm.__salt__, {'vmadm.list': MagicMock(return_value=mock_vm_one)}):
        config = []
        ret = vmadm.validate(config)
        assert ret == (True, 'Valid beacon configuration')
        ret = vmadm.beacon(config)
        res = [{'alias': 'vm1', 'tag': 'running/00000000-0000-0000-0000-000000000001', 'hostname': 'vm1', 'dns_domain': 'example.org'}]
        assert ret == res

def test_created(mock_clean_state, mock_vm_one, mock_vm_two_started):
    if False:
        while True:
            i = 10
    '\n    Test with one vm, create a 2nd one\n    '
    with patch.dict(vmadm.VMADM_STATE, mock_clean_state), patch.dict(vmadm.__salt__, {'vmadm.list': MagicMock(side_effect=[mock_vm_one, mock_vm_two_started])}):
        config = []
        ret = vmadm.validate(config)
        assert ret == (True, 'Valid beacon configuration')
        ret = vmadm.beacon(config)
        ret = vmadm.beacon(config)
        res = [{'alias': 'vm2', 'tag': 'created/00000000-0000-0000-0000-000000000002', 'hostname': 'vm2', 'dns_domain': 'example.org'}, {'alias': 'vm2', 'tag': 'running/00000000-0000-0000-0000-000000000002', 'hostname': 'vm2', 'dns_domain': 'example.org'}]
        assert ret == res

def test_deleted(mock_clean_state, mock_vm_two_stopped, mock_vm_one):
    if False:
        return 10
    '\n    Test with two vms and one gets destroyed\n    '
    with patch.dict(vmadm.VMADM_STATE, mock_clean_state), patch.dict(vmadm.__salt__, {'vmadm.list': MagicMock(side_effect=[mock_vm_two_stopped, mock_vm_one])}):
        config = []
        ret = vmadm.validate(config)
        assert ret == (True, 'Valid beacon configuration')
        ret = vmadm.beacon(config)
        ret = vmadm.beacon(config)
        res = [{'alias': 'vm2', 'tag': 'deleted/00000000-0000-0000-0000-000000000002', 'hostname': 'vm2', 'dns_domain': 'example.org'}]
        assert ret == res

def test_complex(mock_clean_state, mock_vm_one, mock_vm_two_started, mock_vm_two_stopped):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test with two vms, stop one, delete one\n    '
    with patch.dict(vmadm.VMADM_STATE, mock_clean_state), patch.dict(vmadm.__salt__, {'vmadm.list': MagicMock(side_effect=[mock_vm_two_started, mock_vm_two_stopped, mock_vm_one])}):
        config = []
        ret = vmadm.validate(config)
        assert ret == (True, 'Valid beacon configuration')
        ret = vmadm.beacon(config)
        ret = vmadm.beacon(config)
        res = [{'alias': 'vm2', 'tag': 'stopped/00000000-0000-0000-0000-000000000002', 'hostname': 'vm2', 'dns_domain': 'example.org'}]
        assert ret == res
        ret = vmadm.beacon(config)
        res = [{'alias': 'vm2', 'tag': 'deleted/00000000-0000-0000-0000-000000000002', 'hostname': 'vm2', 'dns_domain': 'example.org'}]
        assert ret == res