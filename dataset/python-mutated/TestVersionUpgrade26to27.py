import configparser
import pytest
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import VersionUpgrade26to27

@pytest.fixture
def upgrader():
    if False:
        print('Hello World!')
    return VersionUpgrade26to27.VersionUpgrade26to27()
test_cfg_version_good_data = [{'test_name': 'Simple', 'file_data': '[general]\nversion = 1\n', 'version': 1000000}, {'test_name': 'Other Data Around', 'file_data': '[nonsense]\nlife = good\n\n[general]\nversion = 3\n\n[values]\nlayer_height = 0.12\ninfill_sparse_density = 42\n', 'version': 3000000}, {'test_name': 'Setting Version', 'file_data': '[general]\nversion = 1\n[metadata]\nsetting_version = 1\n', 'version': 1000001}]

@pytest.mark.parametrize('data', test_cfg_version_good_data)
def test_cfgVersionGood(data, upgrader):
    if False:
        return 10
    version = upgrader.getCfgVersion(data['file_data'])
    assert version == data['version']
test_upgrade_stacks_with_not_supported_data = [{'test_name': 'Global stack with Not Supported quality profile', 'file_data': '[general]\nversion = 3\nname = Ultimaker 3\nid = Ultimaker 3\n\n[metadata]\ntype = machine\n\n[containers]\n0 = Ultimaker 3_user\n1 = empty\n2 = um3_global_Normal_Quality\n3 = empty\n4 = empty\n5 = empty\n6 = ultimaker3\n'}, {'test_name': 'Extruder stack left with Not Supported quality profile', 'file_data': '[general]\nversion = 3\nname = Extruder 1\nid = ultimaker3_extruder_left #2\n\n[metadata]\nposition = 0\nmachine = Ultimaker 3\ntype = extruder_train\n\n[containers]\n0 = ultimaker3_extruder_left #2_user\n1 = empty\n2 = um3_aa0.4_PVA_Not_Supported_Quality\n3 = generic_pva_ultimaker3_AA_0.4\n4 = ultimaker3_aa04\n5 = ultimaker3_extruder_left #2_settings\n6 = ultimaker3_extruder_left\n'}]

@pytest.mark.parametrize('data', test_upgrade_stacks_with_not_supported_data)
def test_upgradeStacksWithNotSupportedQuality(data, upgrader):
    if False:
        while True:
            i = 10
    original_parser = configparser.ConfigParser(interpolation=None)
    original_parser.read_string(data['file_data'])
    (_, upgraded_stacks) = upgrader.upgradeStack(data['file_data'], '<string>')
    upgraded_stack = upgraded_stacks[0]
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(upgraded_stack)
    assert 'Not_Supported' not in parser.get('containers', '2')