import configparser
import VersionUpgrade43to44
before_update = '[general]\nversion = 4\nname = Ultimaker 3\nid = Ultimaker 3\n\n[metadata]\ntype = machine\n\n[containers]\n0 = user_profile\n1 = quality_changes\n2 = quality\n3 = material\n4 = variant\n5 = definition_changes\n6 = definition\n'

def test_upgrade():
    if False:
        for i in range(10):
            print('nop')
    upgrader = VersionUpgrade43to44.VersionUpgrade43to44()
    (file_name, new_data) = upgrader.upgradeStack(before_update, 'whatever')
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(new_data[0])
    assert parser['containers']['0'] == 'user_profile'
    assert parser['containers']['1'] == 'quality_changes'
    assert parser['containers']['2'] == 'empty_intent'
    assert parser['containers']['3'] == 'quality'
    assert parser['containers']['4'] == 'material'
    assert parser['containers']['5'] == 'variant'
    assert parser['containers']['6'] == 'definition_changes'
    assert parser['containers']['7'] == 'definition'