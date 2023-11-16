"""
Test cases for salt.states.openvswitch_bridge.
"""
import pytest
import salt.states.openvswitch_bridge as openvswitch_bridge
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        print('Hello World!')
    return {openvswitch_bridge: {'__opts__': {'test': False}}}

def test_present_no_parent_existing_no_parent():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test present function, not specifying a parent.\n\n    This tests the case where the bridge already exists and has no parent.\n    '
    create_mock = MagicMock()
    exists_mock = MagicMock(return_value=True)
    to_parent_mock = MagicMock(return_value='br0')
    to_vlan_mock = MagicMock(return_value=0)
    with patch.dict(openvswitch_bridge.__salt__, {'openvswitch.bridge_create': create_mock, 'openvswitch.bridge_exists': exists_mock, 'openvswitch.bridge_to_parent': to_parent_mock, 'openvswitch.bridge_to_vlan': to_vlan_mock}):
        ret = openvswitch_bridge.present(name='br0')
        create_mock.assert_not_called()
        assert ret['result'] is True

def test_present_no_parent_existing_with_parent():
    if False:
        return 10
    '\n    Test present function, not specifying a parent.\n\n    This tests the case where the bridge already exists and has a parent.\n    '
    create_mock = MagicMock()
    exists_mock = MagicMock(return_value=True)
    to_parent_mock = MagicMock(return_value='br0')
    to_vlan_mock = MagicMock(return_value=42)
    with patch.dict(openvswitch_bridge.__salt__, {'openvswitch.bridge_create': create_mock, 'openvswitch.bridge_exists': exists_mock, 'openvswitch.bridge_to_parent': to_parent_mock, 'openvswitch.bridge_to_vlan': to_vlan_mock}):
        ret = openvswitch_bridge.present(name='br1')
        create_mock.assert_not_called()
        assert ret['result'] is False

def test_present_no_parent_not_existing():
    if False:
        while True:
            i = 10
    '\n    Test present function, not specifying a parent.\n\n    This tests the case where the bridge does not exist yet.\n    '
    create_mock = MagicMock(return_value=True)
    exists_mock = MagicMock(return_value=False)
    with patch.dict(openvswitch_bridge.__salt__, {'openvswitch.bridge_create': create_mock, 'openvswitch.bridge_exists': exists_mock}):
        ret = openvswitch_bridge.present(name='br0')
        create_mock.assert_called_with('br0', parent=None, vlan=None)
        assert ret['result'] is True
        assert ret['changes'] == {'br0': {'new': 'Bridge br0 created', 'old': 'Bridge br0 does not exist.'}}

def test_present_with_parent_existing_with_parent():
    if False:
        print('Hello World!')
    '\n    Test present function, specifying a parent.\n\n    This tests the case where the bridge already exists and has a parent that\n    matches the specified one.\n    '
    create_mock = MagicMock()
    exists_mock = MagicMock(return_value=True)
    to_parent_mock = MagicMock(return_value='br0')
    to_vlan_mock = MagicMock(return_value=42)
    with patch.dict(openvswitch_bridge.__salt__, {'openvswitch.bridge_create': create_mock, 'openvswitch.bridge_exists': exists_mock, 'openvswitch.bridge_to_parent': to_parent_mock, 'openvswitch.bridge_to_vlan': to_vlan_mock}):
        ret = openvswitch_bridge.present(name='br1', parent='br0', vlan=42)
        create_mock.assert_not_called()
        assert ret['result'] is True

def test_present_with_parent_not_existing():
    if False:
        while True:
            i = 10
    '\n    Test present function, specifying a parent.\n\n    This tests the case where the bridge does not exist yet.\n    '
    create_mock = MagicMock(return_value=True)
    exists_mock = MagicMock(return_value=False)
    with patch.dict(openvswitch_bridge.__salt__, {'openvswitch.bridge_create': create_mock, 'openvswitch.bridge_exists': exists_mock}):
        ret = openvswitch_bridge.present(name='br1', parent='br0', vlan=42)
        create_mock.assert_called_with('br1', parent='br0', vlan=42)
        assert ret['result'] is True
        assert ret['changes'] == {'br1': {'new': 'Bridge br1 created', 'old': 'Bridge br1 does not exist.'}}