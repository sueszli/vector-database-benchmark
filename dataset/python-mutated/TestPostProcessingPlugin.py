import os
import sys
from unittest.mock import patch, MagicMock
from UM.PluginRegistry import PluginRegistry
from UM.Resources import Resources
from UM.Trust import Trust
from ..PostProcessingPlugin import PostProcessingPlugin
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
' In this file, community refers to regular Cura for makers.'
mock_plugin_registry = MagicMock()
mock_plugin_registry.getPluginPath = MagicMock(return_value='mocked_plugin_path')

@patch('cura.ApplicationMetadata.IsEnterpriseVersion', False)
def test_community_user_script_allowed():
    if False:
        while True:
            i = 10
    assert PostProcessingPlugin._isScriptAllowed('blaat.py')

@patch('cura.ApplicationMetadata.IsEnterpriseVersion', False)
def test_community_bundled_script_allowed():
    if False:
        print('Hello World!')
    assert PostProcessingPlugin._isScriptAllowed(_bundled_file_path())

@patch('cura.ApplicationMetadata.IsEnterpriseVersion', True)
@patch.object(PluginRegistry, 'getInstance', return_value=mock_plugin_registry)
def test_enterprise_unsigned_user_script_not_allowed(plugin_registry):
    if False:
        for i in range(10):
            print('nop')
    assert not PostProcessingPlugin._isScriptAllowed('blaat.py')

@patch('cura.ApplicationMetadata.IsEnterpriseVersion', True)
@patch.object(PluginRegistry, 'getInstance', return_value=mock_plugin_registry)
def test_enterprise_signed_user_script_allowed(plugin_registry):
    if False:
        for i in range(10):
            print('nop')
    mocked_trust = MagicMock()
    mocked_trust.signedFileCheck = MagicMock(return_value=True)
    plugin_registry.getPluginPath = MagicMock(return_value='mocked_plugin_path')
    with patch.object(Trust, 'signatureFileExistsFor', return_value=True):
        with patch('UM.Trust.Trust.getInstanceOrNone', return_value=mocked_trust):
            assert PostProcessingPlugin._isScriptAllowed('mocked_plugin_path/scripts/blaat.py')

@patch('cura.ApplicationMetadata.IsEnterpriseVersion', False)
def test_enterprise_bundled_script_allowed():
    if False:
        for i in range(10):
            print('nop')
    assert PostProcessingPlugin._isScriptAllowed(_bundled_file_path())

def _bundled_file_path():
    if False:
        print('Hello World!')
    return os.path.join(Resources.getStoragePath(Resources.Resources) + 'scripts/blaat.py')