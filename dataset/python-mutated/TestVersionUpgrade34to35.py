import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import configparser
import pytest
import VersionUpgrade34to35

@pytest.fixture
def upgrader():
    if False:
        i = 10
        return i + 15
    return VersionUpgrade34to35.VersionUpgrade34to35()
test_upgrade_version_nr_data = [('Empty config file', '[general]\n    version = 5\n    [metadata]\n    setting_version = 4\n\n    [info]\n    asked_send_slice_info = True\n    send_slice_info = True\n')]

@pytest.mark.parametrize('test_name, file_data', test_upgrade_version_nr_data)
def test_upgradeVersionNr(test_name, file_data, upgrader):
    if False:
        return 10
    (_, upgraded_instances) = upgrader.upgradePreferences(file_data, '<string>')
    upgraded_instance = upgraded_instances[0]
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(upgraded_instance)
    assert parser['general']['version'] == '6'
    assert parser['metadata']['setting_version'] == '5'
    assert parser.get('info', 'asked_send_slice_info') == 'False'
    assert parser.get('info', 'send_slice_info') == 'True'