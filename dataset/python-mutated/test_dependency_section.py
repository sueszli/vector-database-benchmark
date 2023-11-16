import pytest
from molecule.model import schema_v3

@pytest.fixture()
def _model_dependency_section_data():
    if False:
        return 10
    return {'dependency': {'name': 'galaxy', 'enabled': True, 'options': {'foo': 'bar'}, 'env': {'FOO': 'foo', 'FOO_BAR': 'foo_bar'}}}

@pytest.mark.parametrize('_config', ['_model_dependency_section_data'], indirect=True)
def test_dependency(_config):
    if False:
        while True:
            i = 10
    assert not schema_v3.validate(_config)

@pytest.fixture()
def _model_dependency_errors_section_data():
    if False:
        return 10
    return {'dependency': {'name': 0}}

@pytest.mark.parametrize('_config', ['_model_dependency_errors_section_data'], indirect=True)
def test_dependency_has_errors(_config):
    if False:
        for i in range(10):
            print('nop')
    x = ["0 is not one of ['galaxy', 'shell']"]
    assert x == schema_v3.validate(_config)

@pytest.fixture()
def _model_dependency_allows_galaxy_section_data():
    if False:
        while True:
            i = 10
    return {'dependency': {'name': 'galaxy'}}

@pytest.fixture()
def _model_dependency_allows_shell_section_data():
    if False:
        print('Hello World!')
    return {'dependency': {'name': 'shell'}}

@pytest.mark.parametrize('_config', ['_model_dependency_allows_galaxy_section_data', '_model_dependency_allows_shell_section_data'], indirect=True)
def test_dependency_allows_shell_name(_config):
    if False:
        print('Hello World!')
    assert not schema_v3.validate(_config)

@pytest.fixture()
def _model_dependency_shell_errors_section_data():
    if False:
        i = 10
        return i + 15
    return {'dependency': {'name': 'shell', 'command': None}}

@pytest.mark.parametrize('_config', ['_model_dependency_shell_errors_section_data'], indirect=True)
def test_dependency_shell_has_errors(_config):
    if False:
        i = 10
        return i + 15
    x = ["None is not of type 'string'"]
    assert x == schema_v3.validate(_config)