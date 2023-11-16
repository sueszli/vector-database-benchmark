from unittest.mock import MagicMock, patch
import os.path
from UM.Preferences import Preferences
from UM.Resources import Resources
from cura.CuraApplication import CuraApplication
from cura.Machines.Models.SettingVisibilityPresetsModel import SettingVisibilityPresetsModel
from cura.Settings.SettingVisibilityPreset import SettingVisibilityPreset
setting_visibility_preset_test_settings = {'test', 'zomg', 'derp', 'yay', 'whoo'}
Resources.addSearchPath(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'resources')))
Resources.addStorageType(CuraApplication.ResourceTypes.SettingVisibilityPreset, 'setting_visibility')

def test_createVisibilityPresetFromLocalFile():
    if False:
        i = 10
        return i + 15
    visibility_preset = SettingVisibilityPreset()
    visibility_preset.loadFromFile(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'setting_visibility_preset_test.cfg'))
    assert setting_visibility_preset_test_settings == set(visibility_preset.settings)
    assert visibility_preset.name == 'test'
    assert visibility_preset.weight == 1
    assert visibility_preset.settings.count('yay') == 1

def test_visibilityFromPrevious():
    if False:
        i = 10
        return i + 15
    with patch('cura.CuraApplication.CuraApplication.getInstance'):
        visibility_model = SettingVisibilityPresetsModel(Preferences())
    basic_visibility = visibility_model.getVisibilityPresetById('basic')
    advanced_visibility = visibility_model.getVisibilityPresetById('advanced')
    expert_visibility = visibility_model.getVisibilityPresetById('expert')
    settings_not_in_advanced = set(basic_visibility.settings) - set(advanced_visibility.settings)
    assert len(settings_not_in_advanced) == 0
    settings_not_in_expert = set(advanced_visibility.settings) - set(expert_visibility.settings)
    assert len(settings_not_in_expert) == 0

def test_setActivePreset():
    if False:
        while True:
            i = 10
    preferences = Preferences()
    with patch('cura.CuraApplication.CuraApplication.getInstance'):
        visibility_model = SettingVisibilityPresetsModel(preferences)
    visibility_model.activePresetChanged = MagicMock()
    assert visibility_model.activePreset == 'basic'
    visibility_model.setActivePreset('basic')
    assert visibility_model.activePreset == 'basic'
    assert visibility_model.activePresetChanged.emit.call_count == 0
    visibility_model.setActivePreset('advanced')
    assert visibility_model.activePreset == 'advanced'
    assert visibility_model.activePresetChanged.emit.call_count == 1
    visibility_model.setActivePreset('OMGZOMGNOPE')
    assert visibility_model.activePreset == 'advanced'
    assert visibility_model.activePresetChanged.emit.call_count == 1

def test_preferenceChanged():
    if False:
        for i in range(10):
            print('nop')
    preferences = Preferences()
    preferences.addPreference('general/visible_settings', 'omgzomg')
    with patch('cura.CuraApplication.CuraApplication.getInstance'):
        visibility_model = SettingVisibilityPresetsModel(preferences)
    visibility_model.activePresetChanged = MagicMock()
    assert visibility_model.activePreset == 'custom'
    assert visibility_model.activePresetChanged.emit.call_count == 0
    basic_visibility = visibility_model.getVisibilityPresetById('basic')
    new_visibility_string = ';'.join(basic_visibility.settings)
    preferences.setValue('general/visible_settings', new_visibility_string)
    visibility_model._onPreferencesChanged('general/visible_settings')
    assert visibility_model.activePreset == 'basic'
    assert visibility_model.activePresetChanged.emit.call_count == 1