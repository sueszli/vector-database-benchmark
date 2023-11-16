from unittest.mock import patch, MagicMock
import pytest
from cura.Settings.SettingOverrideDecorator import SettingOverrideDecorator
extruder_manager = MagicMock(name='ExtruderManager')
application = MagicMock(name='application')
container_registry = MagicMock(name='container_registry')

@pytest.fixture()
def setting_override_decorator():
    if False:
        while True:
            i = 10
    container_registry.reset_mock()
    application.reset_mock()
    extruder_manager.reset_mock()
    with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
        with patch('UM.Application.Application.getInstance', MagicMock(return_value=application)):
            with patch('cura.Settings.ExtruderManager.ExtruderManager.getInstance', MagicMock(return_value=extruder_manager)):
                return SettingOverrideDecorator()

def test_onSettingValueChanged(setting_override_decorator):
    if False:
        while True:
            i = 10

    def mock_getRawProperty(key, property_name, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if property_name == 'limit_to_extruder':
            return '-1'
        return MagicMock(name='rawProperty')
    container_registry.findContainerStacks().__getitem__().getRawProperty = mock_getRawProperty
    assert application.getBackend().needsSlicing.call_count == 1
    with patch('UM.Application.Application.getInstance', MagicMock(return_value=application)):
        setting_override_decorator._onSettingChanged('blarg', 'value')
    assert application.getBackend().needsSlicing.call_count == 2

def test_onSettingEnableChanged(setting_override_decorator):
    if False:
        while True:
            i = 10
    assert application.getBackend().needsSlicing.call_count == 1
    with patch('UM.Application.Application.getInstance', MagicMock(return_value=application)):
        setting_override_decorator._onSettingChanged('blarg', 'enabled')
    assert application.getBackend().needsSlicing.call_count == 1

def test_setActiveExtruder(setting_override_decorator):
    if False:
        print('Hello World!')
    setting_override_decorator.activeExtruderChanged.emit = MagicMock()
    with patch('cura.Settings.ExtruderManager.ExtruderManager.getInstance', MagicMock(return_value=extruder_manager)):
        with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
            setting_override_decorator.setActiveExtruder('ZOMG')
    setting_override_decorator.activeExtruderChanged.emit.assert_called_once_with()
    assert setting_override_decorator.getActiveExtruder() == 'ZOMG'