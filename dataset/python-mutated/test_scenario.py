import os
import shutil
import pytest
from molecule import config, scenario, util

@pytest.fixture()
def _instance(patched_config_validate, config_instance: config.Config):
    if False:
        while True:
            i = 10
    return scenario.Scenario(config_instance)

def test_prune(_instance):
    if False:
        i = 10
        return i + 15
    e_dir = _instance.ephemeral_directory
    prune_data = {'safe_files': ['state.yml', 'ansible.cfg', 'inventory/ansible_inventory.yml'], 'safe_dirs': ['inventory'], 'pruned_files': ['foo', 'bar', 'inventory/foo', 'inventory/bar'], 'pruned_dirs': ['baz', 'roles', 'inventory/baz', 'roles/foo']}
    for directory in prune_data['safe_dirs'] + prune_data['pruned_dirs']:
        if directory == 'inventory':
            continue
        os.mkdir(os.path.join(e_dir, directory))
    for file in prune_data['safe_files'] + prune_data['pruned_files']:
        util.write_file(os.path.join(e_dir, file), '')
    _instance.prune()
    for safe_file in prune_data['safe_files']:
        assert os.path.isfile(os.path.join(e_dir, safe_file))
    for safe_dir in prune_data['safe_dirs']:
        assert os.path.isdir(os.path.join(e_dir, safe_dir))
    for pruned_file in prune_data['pruned_files']:
        assert not os.path.isfile(os.path.join(e_dir, pruned_file))
    for pruned_dir in prune_data['pruned_dirs']:
        assert not os.path.isdir(os.path.join(e_dir, pruned_dir))

def test_config_member(_instance):
    if False:
        print('Hello World!')
    assert isinstance(_instance.config, config.Config)

def test_scenario_init_calls_setup(patched_scenario_setup, _instance):
    if False:
        while True:
            i = 10
    patched_scenario_setup.assert_called_once_with()

def test_scenario_name_property(_instance):
    if False:
        i = 10
        return i + 15
    assert _instance.name == 'default'

def test_scenario_directory_property(molecule_scenario_directory_fixture, _instance):
    if False:
        i = 10
        return i + 15
    assert molecule_scenario_directory_fixture == _instance.directory

def test_ephemeral_directory_property(_instance):
    if False:
        return 10
    assert os.access(_instance.ephemeral_directory, os.W_OK)

def test_scenario_inventory_directory_property(_instance):
    if False:
        return 10
    ephemeral_directory = _instance.config.scenario.ephemeral_directory
    e_dir = os.path.join(ephemeral_directory, 'inventory')
    assert e_dir == _instance.inventory_directory

def test_check_sequence_property(_instance):
    if False:
        while True:
            i = 10
    sequence = ['dependency', 'cleanup', 'destroy', 'create', 'prepare', 'converge', 'check', 'cleanup', 'destroy']
    assert sequence == _instance.check_sequence

def test_converge_sequence_property(_instance):
    if False:
        while True:
            i = 10
    sequence = ['dependency', 'create', 'prepare', 'converge']
    assert sequence == _instance.converge_sequence

def test_create_sequence_property(_instance):
    if False:
        return 10
    sequence = ['dependency', 'create', 'prepare']
    assert sequence == _instance.create_sequence

def test_dependency_sequence_property(_instance):
    if False:
        return 10
    assert ['dependency'] == _instance.dependency_sequence

def test_destroy_sequence_property(_instance):
    if False:
        for i in range(10):
            print('nop')
    assert ['dependency', 'cleanup', 'destroy'] == _instance.destroy_sequence

def test_idempotence_sequence_property(_instance):
    if False:
        return 10
    assert ['idempotence'] == _instance.idempotence_sequence

def test_prepare_sequence_property(_instance):
    if False:
        i = 10
        return i + 15
    assert ['prepare'] == _instance.prepare_sequence

def test_side_effect_sequence_property(_instance):
    if False:
        i = 10
        return i + 15
    assert ['side_effect'] == _instance.side_effect_sequence

def test_syntax_sequence_property(_instance):
    if False:
        while True:
            i = 10
    assert ['syntax'] == _instance.syntax_sequence

def test_test_sequence_property(_instance):
    if False:
        print('Hello World!')
    sequence = ['dependency', 'cleanup', 'destroy', 'syntax', 'create', 'prepare', 'converge', 'idempotence', 'side_effect', 'verify', 'cleanup', 'destroy']
    assert sequence == _instance.test_sequence

def test_verify_sequence_property(_instance):
    if False:
        i = 10
        return i + 15
    assert ['verify'] == _instance.verify_sequence

def test_sequence_property_with_invalid_subcommand(_instance):
    if False:
        for i in range(10):
            print('nop')
    _instance.config.command_args = {'subcommand': 'invalid'}
    assert [] == _instance.sequence

def test_setup_creates_ephemeral_and_inventory_directories(_instance):
    if False:
        for i in range(10):
            print('nop')
    ephemeral_dir = _instance.config.scenario.ephemeral_directory
    inventory_dir = _instance.config.scenario.inventory_directory
    shutil.rmtree(ephemeral_dir)
    _instance._setup()
    assert os.path.isdir(ephemeral_dir)
    assert os.path.isdir(inventory_dir)

def test_ephemeral_directory():
    if False:
        while True:
            i = 10
    assert os.access(scenario.ephemeral_directory('foo/bar'), os.W_OK)

def test_ephemeral_directory_OVERRIDDEN_via_env_var(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setenv('MOLECULE_EPHEMERAL_DIRECTORY', 'foo/bar')
    assert os.access(scenario.ephemeral_directory('foo/bar'), os.W_OK)

def test_ephemeral_directory_OVERRIDDEN_via_env_var_uses_absolute_path(monkeypatch):
    if False:
        i = 10
        return i + 15
    monkeypatch.setenv('MOLECULE_EPHEMERAL_DIRECTORY', 'foo/bar')
    assert os.path.isabs(scenario.ephemeral_directory())