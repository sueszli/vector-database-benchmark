import pytest
from molecule.model import schema_v3

@pytest.fixture()
def _model_scenario_section_data():
    if False:
        i = 10
        return i + 15
    return {'scenario': {'name': 'foo', 'check_sequence': ['check'], 'converge_sequence': ['converge'], 'create_sequence': ['create'], 'destroy_sequence': ['destroy'], 'test_sequence': ['test']}}

@pytest.mark.parametrize('_config', ['_model_scenario_section_data'], indirect=True)
def test_scenario(_config):
    if False:
        while True:
            i = 10
    assert not schema_v3.validate(_config)

@pytest.fixture()
def _model_scenario_errors_section_data():
    if False:
        print('Hello World!')
    return {'scenario': {'name': 0}}

@pytest.mark.parametrize('_config', ['_model_scenario_errors_section_data'], indirect=True)
def test_scenario_has_errors(_config):
    if False:
        i = 10
        return i + 15
    x = ["0 is not of type 'string'"]
    assert x == schema_v3.validate(_config)