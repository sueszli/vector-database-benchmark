import pytest
from molecule.model import schema_v3

@pytest.fixture()
def _model_verifier_section_data():
    if False:
        while True:
            i = 10
    return {'verifier': {'name': 'testinfra', 'enabled': True, 'directory': 'foo', 'options': {'foo': 'bar'}, 'env': {'FOO': 'foo', 'FOO_BAR': 'foo_bar'}, 'additional_files_or_dirs': ['foo']}}

@pytest.mark.parametrize('_config', ['_model_verifier_section_data'], indirect=True)
def test_verifier(_config):
    if False:
        while True:
            i = 10
    assert not schema_v3.validate(_config)

@pytest.fixture()
def _model_verifier_errors_section_data():
    if False:
        for i in range(10):
            print('nop')
    return {'verifier': {'name': 0}}

@pytest.mark.parametrize('_config', ['_model_verifier_errors_section_data'], indirect=True)
def test_verifier_has_errors(_config):
    if False:
        for i in range(10):
            print('nop')
    x = ["0 is not one of ['ansible', 'goss', 'inspec', 'testinfra']"]
    assert x == schema_v3.validate(_config)

@pytest.fixture()
def _model_verifier_allows_testinfra_section_data():
    if False:
        for i in range(10):
            print('nop')
    return {'verifier': {'name': 'testinfra'}}

@pytest.fixture()
def _model_verifier_allows_ansible_section_data():
    if False:
        for i in range(10):
            print('nop')
    return {'verifier': {'name': 'ansible'}}

@pytest.mark.parametrize('_config', ['_model_verifier_allows_testinfra_section_data', '_model_verifier_allows_ansible_section_data'], indirect=True)
def test_verifier_allows_name(_config):
    if False:
        return 10
    assert not schema_v3.validate(_config)