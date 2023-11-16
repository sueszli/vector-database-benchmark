"""
    :codeauthor: Gareth J. Greenaway <gareth@saltstack.com>

    Test cases for salt.states.netconfig
"""
import pytest
import salt.modules.napalm_network as net_mod
import salt.states.netconfig as netconfig
import salt.utils.files
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    state_loader_globals = {'__env__': 'base', '__salt__': {'net.replace_pattern': net_mod.replace_pattern}}
    module_loader_globals = {'__env__': 'base', '__salt__': {'net.replace_pattern': net_mod.replace_pattern, 'net.load_config': net_mod.load_config}}
    return {netconfig: state_loader_globals, net_mod: module_loader_globals}

def test_replace_pattern_test_is_true():
    if False:
        return 10
    '\n    Test to replace_pattern to ensure that test=True\n    is being passed correctly.\n    '
    name = 'name'
    pattern = 'OLD-POLICY-NAME'
    repl = 'new-policy-name'
    mock = MagicMock()
    mock_net_replace_pattern = MagicMock()
    mock_loaded_ret = MagicMock()
    with patch.dict(netconfig.__salt__, {'config.merge': mock}):
        with patch.dict(netconfig.__salt__, {'net.replace_pattern': mock_net_replace_pattern}):
            with patch.object(salt.utils.napalm, 'loaded_ret', mock_loaded_ret):
                with patch.dict(netconfig.__opts__, {'test': True}):
                    netconfig.replace_pattern(name, pattern, repl)
                    (args, kwargs) = mock_net_replace_pattern.call_args_list[0]
                    assert kwargs['test']
                    (args, kwargs) = mock_loaded_ret.call_args_list[0]
                    assert args[2]
                netconfig.replace_pattern(name, pattern, repl, test=True)
                (args, kwargs) = mock_net_replace_pattern.call_args_list[0]
                assert kwargs['test']
                (args, kwargs) = mock_loaded_ret.call_args_list[0]
                assert args[2]

def test_managed_test_is_true():
    if False:
        return 10
    '\n    Test to managed to ensure that test=True\n    is being passed correctly.\n    '
    name = 'name'
    mock = MagicMock()
    mock_update_config = MagicMock()
    with patch.dict(netconfig.__salt__, {'config.merge': mock}):
        with patch.object(netconfig, '_update_config', mock_update_config):
            with patch.dict(netconfig.__opts__, {'test': True}):
                netconfig.managed(name)
                (args, kwargs) = mock_update_config.call_args_list[0]
                assert kwargs['test']
            netconfig.managed(name, test=True)
            (args, kwargs) = mock_update_config.call_args_list[0]
            assert kwargs['test']