import configparser
import VersionUpgrade44to45
import pytest
before_update = '[general]\nversion = 4\nname = Creality CR-10S_settings\ndefinition = creality_cr10s\n\n[metadata]\ntype = definition_changes\nsetting_version = 11\n\n[values]\n%s\n'
before_after_list = [('machine_head_with_fans_polygon = [[-99, 99], [-99, -44], [45, 99], [45, -44]]', '[[-99, 99], [-99, -44], [45, 99], [45, -44]]'), ('', None), ('machine_head_polygon = [[-98, 99], [-99, -44], [45, 99], [45, -44]]', '[[-98, 99], [-99, -44], [45, 99], [45, -44]]'), ('machine_head_polygon = [[-87, 99], [-99, -44], [45, 99], [45, -44]]\nmachine_head_with_fans_polygon = [[-99, 99], [-99, -44], [45, 99], [45, -44]]', '[[-99, 99], [-99, -44], [45, 99], [45, -44]]')]

class TestVersionUpgrade44to45:

    @pytest.mark.parametrize('after_string, after_value', before_after_list)
    def test_upgrade(self, after_string, after_value):
        if False:
            i = 10
            return i + 15
        upgrader = VersionUpgrade44to45.VersionUpgrade44to45()
        (file_name, new_data) = upgrader.upgradeInstanceContainer(before_update % after_string, 'whatever')
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(new_data[0])
        if after_value is None:
            assert 'machine_head_with_fans_polygon' not in parser['values']
        else:
            assert parser['values']['machine_head_with_fans_polygon'] == after_value
        assert 'machine_head_polygon' not in parser['values']