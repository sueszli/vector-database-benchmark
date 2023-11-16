import pytest
from molecule.model import schema_v3

@pytest.fixture()
def _model_driver_section_data():
    if False:
        while True:
            i = 10
    return {'driver': {'name': 'default', 'provider': {'name': None}, 'options': {'managed': True}, 'ssh_connection_options': ['foo', 'bar'], 'safe_files': ['foo', 'bar']}}

@pytest.mark.parametrize('_config', ['_model_driver_section_data'], indirect=True)
def test_driver(_config):
    if False:
        i = 10
        return i + 15
    assert not schema_v3.validate(_config)

@pytest.fixture()
def _model_driver_errors_section_data():
    if False:
        i = 10
        return i + 15
    return {'driver': {'name': 0}}

@pytest.fixture()
def _model_driver_errors_section_data_no_prefix():
    if False:
        i = 10
        return i + 15
    return {'driver': {'name': 'random_name'}}

@pytest.mark.parametrize('_config', ['_model_driver_errors_section_data', '_model_driver_errors_section_data_no_prefix'], indirect=True)
def test_driver_has_errors(_config):
    if False:
        return 10
    base_error_msg = "is not one of ['azure', 'ec2', 'delegated', 'docker', 'containers', 'openstack', 'podman', 'vagrant', 'digitalocean', 'gce', 'libvirt', 'lxd', 'molecule-*', 'molecule_*', 'custom-*', 'custom_*']"
    driver_name = str(_config['driver']['name'])
    if isinstance(_config['driver']['name'], str):
        driver_name = f"'{_config['driver']['name']}'"
    error_msg = [f'{driver_name} {base_error_msg}']
    assert error_msg == schema_v3.validate(_config)

@pytest.fixture()
def _model_driver_provider_name_nullable_section_data():
    if False:
        i = 10
        return i + 15
    return {'driver': {'provider': {'name': None}}}

@pytest.mark.parametrize('_config', ['_model_driver_provider_name_nullable_section_data'], indirect=True)
def test_driver_provider_name_nullable(_config):
    if False:
        return 10
    assert not schema_v3.validate(_config)

@pytest.fixture()
def _model_driver_allows_delegated_section_data():
    if False:
        i = 10
        return i + 15
    return {'driver': {'name': 'default'}}

@pytest.fixture()
def _model_driver_allows_molecule_section_data1():
    if False:
        while True:
            i = 10
    return {'driver': {'name': 'molecule-test_driver.name'}}

@pytest.fixture()
def _model_driver_allows_molecule_section_data2():
    if False:
        for i in range(10):
            print('nop')
    return {'driver': {'name': 'molecule_test_driver.name'}}

@pytest.fixture()
def _model_driver_allows_custom_section_data1():
    if False:
        i = 10
        return i + 15
    return {'driver': {'name': 'custom-test_driver.name'}}

@pytest.fixture()
def _model_driver_allows_custom_section_data2():
    if False:
        i = 10
        return i + 15
    return {'driver': {'name': 'custom_test_driver.name'}}

@pytest.mark.parametrize('_config', ['_model_driver_allows_delegated_section_data', '_model_driver_allows_molecule_section_data1', '_model_driver_allows_molecule_section_data2', '_model_driver_allows_custom_section_data2', '_model_driver_allows_custom_section_data1'], indirect=True)
def test_driver_allows_name(_config):
    if False:
        print('Hello World!')
    assert not schema_v3.validate(_config)