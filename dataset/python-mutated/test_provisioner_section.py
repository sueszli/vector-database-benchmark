import pytest
from molecule.model import schema_v3

@pytest.fixture()
def _model_provisioner_section_data():
    if False:
        for i in range(10):
            print('nop')
    return {'provisioner': {'name': 'ansible', 'log': True, 'config_options': {'foo': 'bar'}, 'connection_options': {'foo': 'bar'}, 'options': {'foo': 'bar'}, 'env': {'FOO': 'foo', 'FOO_BAR': 'foo_bar'}, 'inventory': {'hosts': {'foo': 'bar'}, 'host_vars': {'foo': 'bar'}, 'group_vars': {'foo': 'bar'}, 'links': {'foo': 'bar'}}, 'children': {'foo': 'bar'}, 'playbooks': {'create': 'foo.yml', 'converge': 'bar.yml', 'destroy': 'baz.yml', 'prepare': 'qux.yml', 'side_effect': 'quux.yml', 'foo': {'foo': 'bar'}}}}

@pytest.mark.parametrize('_config', ['_model_provisioner_section_data'], indirect=True)
def test_provisioner(_config):
    if False:
        while True:
            i = 10
    assert not schema_v3.validate(_config)

@pytest.fixture()
def _model_provisioner_errors_section_data():
    if False:
        print('Hello World!')
    return {'provisioner': {'name': 0}}

@pytest.mark.parametrize('_config', ['_model_provisioner_errors_section_data'], indirect=True)
def test_provisioner_has_errors(_config):
    if False:
        while True:
            i = 10
    x = ["0 is not one of ['ansible']"]
    assert x == schema_v3.validate(_config)

@pytest.fixture()
def _model_provisioner_allows_ansible_section_data():
    if False:
        return 10
    return {'provisioner': {'name': 'ansible'}}

@pytest.mark.parametrize('_config', ['_model_provisioner_allows_ansible_section_data'], indirect=True)
def test_provisioner_allows_name(_config):
    if False:
        print('Hello World!')
    assert not schema_v3.validate(_config)