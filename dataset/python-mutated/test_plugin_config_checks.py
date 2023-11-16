import pytest
from ..apps import PluginConfig

def test_empty_plugin_path():
    if False:
        return 10
    plugin_path = ''
    with pytest.raises(ImportError):
        PluginConfig.load_and_check_plugin(None, plugin_path)

def test_invalid_plugin_path():
    if False:
        for i in range(10):
            print('nop')
    plugin_path = 'saleor.core.plugins.wrong_path.Plugin'
    with pytest.raises(ImportError):
        PluginConfig.load_and_check_plugin(None, plugin_path)