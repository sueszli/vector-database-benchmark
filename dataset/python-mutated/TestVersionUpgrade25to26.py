import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import configparser
import pytest
import VersionUpgrade25to26

@pytest.fixture
def upgrader():
    if False:
        print('Hello World!')
    return VersionUpgrade25to26.VersionUpgrade25to26()
test_cfg_version_good_data = [{'test_name': 'Simple', 'file_data': '[general]\nversion = 1\n', 'version': 1000000}, {'test_name': 'Other Data Around', 'file_data': '[nonsense]\nlife = good\n\n[general]\nversion = 3\n\n[values]\nlayer_height = 0.12\ninfill_sparse_density = 42\n', 'version': 3000000}, {'test_name': 'Negative Version', 'file_data': '[general]\nversion = -20\n', 'version': -20000000}, {'test_name': 'Setting Version', 'file_data': '[general]\nversion = 1\n[metadata]\nsetting_version = 1\n', 'version': 1000001}, {'test_name': 'Negative Setting Version', 'file_data': '[general]\nversion = 1\n[metadata]\nsetting_version = -3\n', 'version': 999997}]
test_upgrade_preferences_removed_settings_data = [{'test_name': 'Removed Setting', 'file_data': '[general]\nvisible_settings = baby;you;know;how;I;like;to;start_layers_at_same_position\n'}, {'test_name': 'No Removed Setting', 'file_data': '[general]\nvisible_settings = baby;you;now;how;I;like;to;eat;chocolate;muffins\n'}, {'test_name': 'No Visible Settings Key', 'file_data': '[general]\ncura = cool\n'}, {'test_name': 'No General Category', 'file_data': '[foos]\nfoo = bar\n'}]

@pytest.mark.parametrize('data', test_upgrade_preferences_removed_settings_data)
def test_upgradePreferencesRemovedSettings(data, upgrader):
    if False:
        while True:
            i = 10
    original_parser = configparser.ConfigParser(interpolation=None)
    original_parser.read_string(data['file_data'])
    settings = set()
    if original_parser.has_section('general') and 'visible_settings' in original_parser['general']:
        settings = set(original_parser['general']['visible_settings'].split(';'))
    (_, upgraded_preferences) = upgrader.upgradePreferences(data['file_data'], '<string>')
    upgraded_preferences = upgraded_preferences[0]
    settings -= VersionUpgrade25to26._removed_settings
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(upgraded_preferences)
    assert (parser.has_section('general') and 'visible_settings' in parser['general']) == (len(settings) > 0)
    if settings:
        assert settings == set(parser['general']['visible_settings'].split(';'))
test_upgrade_instance_container_removed_settings_data = [{'test_name': 'Removed Setting', 'file_data': '[values]\nlayer_height = 0.1337\nstart_layers_at_same_position = True\n'}, {'test_name': 'No Removed Setting', 'file_data': '[values]\noceans_number = 11\n'}, {'test_name': 'No Values Category', 'file_data': '[general]\ntype = instance_container\n'}]

@pytest.mark.parametrize('data', test_upgrade_instance_container_removed_settings_data)
def test_upgradeInstanceContainerRemovedSettings(data, upgrader):
    if False:
        print('Hello World!')
    original_parser = configparser.ConfigParser(interpolation=None)
    original_parser.read_string(data['file_data'])
    settings = set()
    if original_parser.has_section('values'):
        settings = set(original_parser['values'])
    (_, upgraded_container) = upgrader.upgradeInstanceContainer(data['file_data'], '<string>')
    upgraded_container = upgraded_container[0]
    settings -= VersionUpgrade25to26._removed_settings
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(upgraded_container)
    assert parser.has_section('values') == (len(settings) > 0)
    if settings:
        assert settings == set(parser['values'])