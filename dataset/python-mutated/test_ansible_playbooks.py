import os
import pytest
from molecule import config, util
from molecule.provisioner import ansible_playbooks
from molecule.test.a_unit.conftest import os_split

@pytest.fixture()
def _provisioner_section_data():
    if False:
        for i in range(10):
            print('nop')
    return {'provisioner': {'name': 'ansible', 'options': {}, 'config_options': {}}}

@pytest.fixture()
def _instance(_provisioner_section_data, config_instance: config.Config):
    if False:
        return 10
    return ansible_playbooks.AnsiblePlaybooks(config_instance)

def test_cleanup_property_is_optional(_instance):
    if False:
        return 10
    assert _instance._config.provisioner.playbooks.cleanup is None

@pytest.mark.skip(reason='create not running for delegated')
def test_create_property(_instance):
    if False:
        i = 10
        return i + 15
    x = os.path.join(_instance._get_playbook_directory(), 'default', 'create.yml')
    assert x == _instance._config.provisioner.playbooks.create

def test_converge_property(_instance):
    if False:
        i = 10
        return i + 15
    x = os.path.join(_instance._config.scenario.directory, 'converge.yml')
    assert x == _instance._config.provisioner.playbooks.converge

@pytest.mark.skip(reason='destroy not running for delegated')
def test_destroy_property(_instance):
    if False:
        i = 10
        return i + 15
    x = os.path.join(_instance._get_playbook_directory(), 'default', 'destroy.yml')
    assert x == _instance._config.provisioner.playbooks.destroy

def test_prepare_property(_instance):
    if False:
        i = 10
        return i + 15
    assert _instance._config.provisioner.playbooks.prepare is None

def test_side_effect_property(_instance):
    if False:
        for i in range(10):
            print('nop')
    assert _instance._config.provisioner.playbooks.side_effect is None

def test_verify_property(_instance):
    if False:
        print('Hello World!')
    assert _instance._config.provisioner.playbooks.verify is None

def test_get_playbook_directory(_instance):
    if False:
        for i in range(10):
            print('nop')
    result = _instance._get_playbook_directory()
    parts = os_split(result)
    x = ('molecule', 'provisioner', 'ansible', 'playbooks')
    assert x == parts[-4:]

def test_get_playbook(tmpdir, _instance):
    if False:
        while True:
            i = 10
    x = os.path.join(_instance._config.scenario.directory, 'create.yml')
    util.write_file(x, '')
    assert x == _instance._get_playbook('create')

@pytest.mark.skip(reason='create not running for delegated')
def test_get_playbook_returns_bundled_driver_playbook_when_local_not_found(tmpdir, _instance):
    if False:
        return 10
    x = os.path.join(_instance._get_playbook_directory(), 'default', 'create.yml')
    assert x == _instance._get_playbook('create')

@pytest.fixture()
def _provisioner_driver_section_data():
    if False:
        while True:
            i = 10
    return {'provisioner': {'name': 'ansible', 'playbooks': {'create': 'create.yml'}}}

@pytest.fixture()
def _provisioner_driver_playbook_key_missing_section_data():
    if False:
        i = 10
        return i + 15
    return {'provisioner': {'name': 'ansible', 'playbooks': {'side_effect': 'side_effect.yml'}}}

@pytest.mark.parametrize('config_instance', ['_provisioner_driver_playbook_key_missing_section_data'], indirect=True)
def test_get_ansible_playbook_with_driver_key_when_playbook_key_missing(tmpdir, _instance):
    if False:
        for i in range(10):
            print('nop')
    x = os.path.join(_instance._config.scenario.directory, 'side_effect.yml')
    util.write_file(x, '')
    assert x == _instance._get_playbook('side_effect')

def test_get_bundled_driver_playbook(_instance):
    if False:
        for i in range(10):
            print('nop')
    result = _instance._get_bundled_driver_playbook('create')
    parts = os_split(result)
    x = ('molecule', 'driver', 'playbooks', 'create.yml')
    assert x == parts[-4:]