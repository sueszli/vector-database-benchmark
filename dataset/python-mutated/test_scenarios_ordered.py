import copy
import pytest
from molecule import config, scenarios

@pytest.fixture()
def _instance(config_instance: config.Config):
    if False:
        i = 10
        return i + 15
    config_instance_1 = copy.deepcopy(config_instance)
    config_instance_1.config['scenario']['name'] = 'two'
    config_instance_1.molecule_file = config_instance_1.molecule_file.replace('default', '02_foo')
    config_instance_2 = copy.deepcopy(config_instance)
    config_instance_2.config['scenario']['name'] = 'one'
    config_instance_2.molecule_file = config_instance_2.molecule_file.replace('default', '01_foo')
    config_instance_3 = copy.deepcopy(config_instance)
    config_instance_3.config['scenario']['name'] = 'three'
    config_instance_3.molecule_file = config_instance_3.molecule_file.replace('default', '03_foo')
    return scenarios.Scenarios([config_instance_1, config_instance_2, config_instance_3])

def test_all_ordered(_instance):
    if False:
        print('Hello World!')
    result = _instance.all
    assert len(result) == 3
    assert result[0].name == 'one'
    assert result[1].name == 'two'
    assert result[2].name == 'three'