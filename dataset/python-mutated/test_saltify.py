"""
    :codeauthor: Alexander Schwartz <alexander.schwartz@gmx.net>
"""
import pytest
import salt.client
from salt.cloud.clouds import saltify
from tests.support.mock import ANY, MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        print('Hello World!')
    test_profiles = {'testprofile1': NotImplemented, 'testprofile2': {'ssh_username': 'fred', 'remove_config_on_destroy': False, 'shutdown_on_destroy': True}, 'testprofile3': {'wake_on_lan_mac': 'aa-bb-cc-dd-ee-ff', 'wol_sender_node': 'friend1', 'wol_boot_wait': 0.01}}
    return {saltify: {'__active_provider_name__': '', '__utils__': {'cloud.bootstrap': MagicMock(), 'cloud.fire_event': MagicMock()}, '__opts__': {'providers': {'sfy1': {'saltify': {'driver': 'saltify', 'profiles': test_profiles}}}, 'profiles': test_profiles, 'sock_dir': '/var/sockxxx', 'transport': 'tcp'}}}

def test_create_no_deploy():
    if False:
        while True:
            i = 10
    "\n    Test if deployment fails. This is the most basic test as saltify doesn't contain much logic\n    "
    with patch('salt.cloud.clouds.saltify._verify', MagicMock(return_value=True)):
        vm = {'deploy': False, 'driver': 'saltify', 'name': 'dummy'}
        assert saltify.create(vm)

def test_create_and_deploy():
    if False:
        print('Hello World!')
    '\n    Test if deployment can be done.\n    '
    mock_cmd = MagicMock(return_value=True)
    with patch.dict('salt.cloud.clouds.saltify.__utils__', {'cloud.bootstrap': mock_cmd}):
        vm_ = {'deploy': True, 'driver': 'saltify', 'name': 'new2', 'profile': 'testprofile2'}
        result = saltify.create(vm_)
        mock_cmd.assert_called_once_with(vm_, ANY)
        assert result

def test_create_no_ssh_host():
    if False:
        print('Hello World!')
    '\n    Test that ssh_host is set to the vm name if not defined\n    '
    mock_cmd = MagicMock(return_value=True)
    with patch.dict('salt.cloud.clouds.saltify.__utils__', {'cloud.bootstrap': mock_cmd}):
        vm_ = {'deploy': True, 'driver': 'saltify', 'name': 'new2', 'profile': 'testprofile2'}
        result = saltify.create(vm_)
        mock_cmd.assert_called_once_with(vm_, ANY)
        assert result
        assert 'ssh_host' in vm_
        assert vm_['ssh_host'] == 'new2'

def test_create_wake_on_lan():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if wake on lan works\n    '
    mock_sleep = MagicMock()
    mock_cmd = MagicMock(return_value=True)
    mm_cmd = MagicMock(return_value={'friend1': True})
    with salt.client.LocalClient() as lcl:
        lcl.cmd = mm_cmd
        with patch('time.sleep', mock_sleep):
            with patch('salt.client.LocalClient', return_value=lcl):
                with patch.dict('salt.cloud.clouds.saltify.__utils__', {'cloud.bootstrap': mock_cmd}):
                    vm_ = {'deploy': True, 'driver': 'saltify', 'name': 'new1', 'profile': 'testprofile3'}
                    result = saltify.create(vm_)
                    mock_cmd.assert_called_once_with(vm_, ANY)
                    mm_cmd.assert_called_with('friend1', 'network.wol', ['aa-bb-cc-dd-ee-ff'])
                    mock_sleep.assert_any_call(0.01)
                    assert result

def test_avail_locations():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the avail_locations will always return {}\n    '
    assert saltify.avail_locations() == {}

def test_avail_sizes():
    if False:
        i = 10
        return i + 15
    '\n    Test the avail_sizes will always return {}\n    '
    assert saltify.avail_sizes() == {}

def test_avail_images():
    if False:
        print('Hello World!')
    '\n    Test the avail_images will return profiles\n    '
    testlist = list(saltify.__opts__['profiles'].keys())
    assert saltify.avail_images()['Profiles'].sort() == testlist.sort()

def test_list_nodes():
    if False:
        print('Hello World!')
    '\n    Test list_nodes will return required fields only\n    '
    testgrains = {'nodeX1': {'id': 'nodeX1', 'ipv4': ['127.0.0.1', '192.1.2.22', '172.16.17.18'], 'ipv6': ['::1', 'fdef:bad:add::f00', '3001:DB8::F00D'], 'salt-cloud': {'driver': 'saltify', 'provider': 'saltyfy', 'profile': 'testprofile2'}, 'extra_stuff': 'does not belong'}}
    expected_result = {'nodeX1': {'id': 'nodeX1', 'image': 'testprofile2', 'private_ips': ['172.16.17.18', 'fdef:bad:add::f00'], 'public_ips': ['192.1.2.22', '3001:DB8::F00D'], 'size': '', 'state': 'running'}}
    mm_cmd = MagicMock(return_value=testgrains)
    with salt.client.LocalClient() as lcl:
        lcl.cmd = mm_cmd
        with patch('salt.client.LocalClient', return_value=lcl):
            assert saltify.list_nodes() == expected_result

def test_saltify_reboot():
    if False:
        i = 10
        return i + 15
    mm_cmd = MagicMock(return_value=True)
    with salt.client.LocalClient() as lcl:
        lcl.cmd = mm_cmd
        with patch('salt.client.LocalClient', return_value=lcl):
            result = saltify.reboot('nodeS1', 'action')
            mm_cmd.assert_called_with('nodeS1', 'system.reboot')
            assert result

def test_saltify_destroy():
    if False:
        i = 10
        return i + 15
    result_list = [{'nodeS1': {'driver': 'saltify', 'provider': 'saltify', 'profile': 'testprofile2'}}, {'nodeS1': 'a system.shutdown worked message'}]
    mm_cmd = MagicMock(side_effect=result_list)
    with salt.client.LocalClient() as lcl:
        lcl.cmd = mm_cmd
        with patch('salt.client.LocalClient', return_value=lcl):
            result = saltify.destroy('nodeS1', 'action')
            mm_cmd.assert_called_with('nodeS1', 'system.shutdown')
            assert result