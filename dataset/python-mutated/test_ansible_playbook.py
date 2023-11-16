from subprocess import CompletedProcess
import pytest
from molecule import config
from molecule.provisioner import ansible_playbook

@pytest.fixture()
def _instance(config_instance: config.Config):
    if False:
        return 10
    _instance = ansible_playbook.AnsiblePlaybook('playbook', config_instance)
    return _instance

@pytest.fixture()
def _provisioner_section_data():
    if False:
        return 10
    return {'provisioner': {'name': 'ansible', 'env': {'FOO': 'bar'}}}

@pytest.fixture()
def _verifier_section_data():
    if False:
        print('Hello World!')
    return {'verifier': {'name': 'ansible', 'env': {'FOO': 'bar'}}}

@pytest.fixture()
def _provisioner_verifier_section_data():
    if False:
        i = 10
        return i + 15
    return {'provisioner': {'name': 'ansible', 'env': {'FOO': 'bar'}}, 'verifier': {'name': 'ansible', 'env': {'FOO': 'baz'}}}

@pytest.fixture()
def _instance_for_verifier_env(config_instance: config.Config):
    if False:
        return 10
    _instance = ansible_playbook.AnsiblePlaybook('playbook', config_instance, True)
    return _instance

@pytest.mark.parametrize('config_instance', ['_provisioner_section_data'], indirect=True)
def test_env_in_provision(_instance_for_verifier_env):
    if False:
        i = 10
        return i + 15
    assert _instance_for_verifier_env._env['FOO'] == 'bar'

@pytest.mark.parametrize('config_instance', ['_verifier_section_data'], indirect=True)
def test_env_in_verifier(_instance_for_verifier_env):
    if False:
        print('Hello World!')
    assert _instance_for_verifier_env._env['FOO'] == 'bar'

@pytest.mark.parametrize('config_instance', ['_provisioner_verifier_section_data'], indirect=True)
def test_env_in_verify_override_provision(_instance_for_verifier_env):
    if False:
        for i in range(10):
            print('nop')
    assert _instance_for_verifier_env._env['FOO'] == 'baz'

@pytest.fixture()
def _inventory_directory(_instance):
    if False:
        i = 10
        return i + 15
    return _instance._config.provisioner.inventory_directory

def test_ansible_command_private_member(_instance):
    if False:
        return 10
    assert _instance._ansible_command is None

def test_ansible_playbook_private_member(_instance):
    if False:
        return 10
    assert _instance._playbook == 'playbook'

def test_config_private_member(_instance):
    if False:
        i = 10
        return i + 15
    assert isinstance(_instance._config, config.Config)

def test_bake(_inventory_directory, _instance):
    if False:
        for i in range(10):
            print('nop')
    pb = _instance._config.provisioner.playbooks.converge
    _instance._playbook = pb
    _instance.bake()
    args = ['ansible-playbook', '--become', '--inventory', _inventory_directory, '--skip-tags', 'molecule-notest,notest', pb]
    assert _instance._ansible_command == args

def test_bake_removes_non_interactive_options_from_non_converge_playbooks(_inventory_directory, _instance):
    if False:
        return 10
    _instance.bake()
    args = ['ansible-playbook', '--inventory', _inventory_directory, '--skip-tags', 'molecule-notest,notest', 'playbook']
    assert _instance._ansible_command == args

def test_bake_has_ansible_args(_inventory_directory, _instance):
    if False:
        for i in range(10):
            print('nop')
    _instance._config.ansible_args = ('foo', 'bar')
    _instance._config.config['provisioner']['ansible_args'] = ('frob', 'nitz')
    _instance.bake()
    args = ['ansible-playbook', '--inventory', _inventory_directory, '--skip-tags', 'molecule-notest,notest', 'frob', 'nitz', 'foo', 'bar', 'playbook']
    assert _instance._ansible_command == args

def test_bake_does_not_have_ansible_args(_inventory_directory, _instance):
    if False:
        print('Hello World!')
    for action in ['create', 'destroy']:
        _instance._config.ansible_args = ('foo', 'bar')
        _instance._config.action = action
        _instance.bake()
        args = ['ansible-playbook', '--inventory', _inventory_directory, '--skip-tags', 'molecule-notest,notest', 'playbook']
        assert _instance._ansible_command == args

def test_bake_idem_does_have_skip_tag(_inventory_directory, _instance):
    if False:
        i = 10
        return i + 15
    _instance._config.action = 'idempotence'
    _instance.bake()
    args = ['ansible-playbook', '--inventory', _inventory_directory, '--skip-tags', 'molecule-notest,notest,molecule-idempotence-notest', 'playbook']
    assert _instance._ansible_command == args

def test_execute_playbook(patched_run_command, _instance):
    if False:
        i = 10
        return i + 15
    _instance._ansible_command = 'patched-command'
    result = _instance.execute()
    assert result == 'patched-run-command-stdout'

def test_ansible_execute_bakes(_inventory_directory, patched_run_command, _instance):
    if False:
        i = 10
        return i + 15
    _instance.execute()
    assert _instance._ansible_command is not None
    args = ['ansible-playbook', '--inventory', _inventory_directory, '--skip-tags', 'molecule-notest,notest', 'playbook']
    assert _instance._ansible_command == args

def test_execute_bakes_with_ansible_args(_inventory_directory, patched_run_command, _instance):
    if False:
        i = 10
        return i + 15
    _instance._config.ansible_args = ('-o', '--syntax-check')
    _instance.execute()
    assert _instance._ansible_command is not None
    args = ['ansible-playbook', '--inventory', _inventory_directory, '--skip-tags', 'molecule-notest,notest', '-o', '--syntax-check', 'playbook']
    assert _instance._ansible_command == args

def test_executes_catches_and_exits_return_code(patched_run_command, _instance):
    if False:
        while True:
            i = 10
    patched_run_command.side_effect = [CompletedProcess(args='ansible-playbook', returncode=1, stdout='out', stderr='err')]
    with pytest.raises(SystemExit) as e:
        _instance.execute()
    assert e.value.code == 1

def test_add_cli_arg(_instance):
    if False:
        return 10
    assert {} == _instance._cli
    _instance.add_cli_arg('foo', 'bar')
    assert {'foo': 'bar'} == _instance._cli

def test_add_env_arg(_instance):
    if False:
        while True:
            i = 10
    assert 'foo' not in _instance._env
    _instance.add_env_arg('foo', 'bar')
    assert _instance._env['foo'] == 'bar'