import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import configparser
import pytest
import VersionUpgrade27to30

@pytest.fixture
def upgrader():
    if False:
        for i in range(10):
            print('nop')
    return VersionUpgrade27to30.VersionUpgrade27to30()
test_cfg_version_good_data = [{'test_name': 'Simple', 'file_data': '[general]\nversion = 1\n', 'version': 1000000}, {'test_name': 'Other Data Around', 'file_data': '[nonsense]\nlife = good\n\n[general]\nversion = 3\n\n[values]\nlayer_height = 0.12\ninfill_sparse_density = 42\n', 'version': 3000000}, {'test_name': 'Negative Version', 'file_data': '[general]\nversion = -20\n', 'version': -20000000}, {'test_name': 'Setting Version', 'file_data': '[general]\nversion = 1\n[metadata]\nsetting_version = 1\n', 'version': 1000001}, {'test_name': 'Negative Setting Version', 'file_data': '[general]\nversion = 1\n[metadata]\nsetting_version = -3\n', 'version': 999997}]

@pytest.mark.parametrize('data', test_cfg_version_good_data)
def test_cfgVersionGood(data, upgrader):
    if False:
        print('Hello World!')
    version = upgrader.getCfgVersion(data['file_data'])
    assert version == data['version']
test_cfg_version_bad_data = [{'test_name': 'Empty', 'file_data': '', 'exception': configparser.Error}, {'test_name': 'No General', 'file_data': '[values]\nlayer_height = 0.1337\n', 'exception': configparser.Error}, {'test_name': 'No Version', 'file_data': '[general]\ntrue = false\n', 'exception': configparser.Error}, {'test_name': 'Not a Number', 'file_data': '[general]\nversion = not-a-text-version-number\n', 'exception': ValueError}, {'test_name': 'Setting Value NaN', 'file_data': '[general]\nversion = 4\n[metadata]\nsetting_version = latest_or_something\n', 'exception': ValueError}, {'test_name': 'Major-Minor', 'file_data': '[general]\nversion = 1.2\n', 'exception': ValueError}]

@pytest.mark.parametrize('data', test_cfg_version_bad_data)
def test_cfgVersionBad(data, upgrader):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(data['exception']):
        upgrader.getCfgVersion(data['file_data'])
test_translate_theme_data = [('Original Cura theme', '[general]\nversion = 4\ntheme = cura\n[metadata]\nsetting_version = 2\n', 'cura-light'), ('No theme', '[general]\nversion = 4\n[metadata]\nsetting_version = 2\n', None)]

@pytest.mark.parametrize('test_name, file_data, new_theme', test_translate_theme_data)
def test_translateTheme(test_name, file_data, new_theme, upgrader):
    if False:
        return 10
    original_parser = configparser.ConfigParser(interpolation=None)
    original_parser.read_string(file_data)
    (_, upgraded_stacks) = upgrader.upgradePreferences(file_data, '<string>')
    upgraded_stack = upgraded_stacks[0]
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(upgraded_stack)
    if not new_theme:
        assert 'theme' not in parser['general']
    else:
        assert 'theme' in parser['general']
        assert parser['general']['theme'] == new_theme