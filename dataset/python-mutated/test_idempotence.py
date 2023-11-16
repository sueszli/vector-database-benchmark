from unittest.mock import Mock
import pytest
from pytest_mock import MockerFixture
from molecule import config
from molecule.command import idempotence

@pytest.fixture()
def _patched_is_idempotent(mocker: MockerFixture) -> Mock:
    if False:
        for i in range(10):
            print('nop')
    return mocker.patch('molecule.command.idempotence.Idempotence._is_idempotent')

@pytest.fixture()
def _instance(patched_config_validate, config_instance: config.Config):
    if False:
        for i in range(10):
            print('nop')
    config_instance.state.change_state('converged', True)
    return idempotence.Idempotence(config_instance)

def test_idempotence_execute(mocker: MockerFixture, caplog: pytest.LogCaptureFixture, patched_ansible_converge, _patched_is_idempotent: Mock, _instance):
    if False:
        while True:
            i = 10
    _instance.execute()
    assert 'default' in caplog.text
    assert 'idempotence' in caplog.text
    patched_ansible_converge.assert_called_once_with()
    _patched_is_idempotent.assert_called_once_with('patched-ansible-converge-stdout')
    msg = 'Idempotence completed successfully.'
    assert msg in caplog.text

def test_execute_raises_when_not_converged(caplog: pytest.LogCaptureFixture, patched_ansible_converge, _instance):
    if False:
        print('Hello World!')
    _instance._config.state.change_state('converged', False)
    with pytest.raises(SystemExit) as e:
        _instance.execute()
    assert e.value.code == 1
    msg = 'Instances not converged.  Please converge instances first.'
    assert msg in caplog.text

def test_execute_raises_when_fails_idempotence(mocker: MockerFixture, caplog: pytest.LogCaptureFixture, patched_ansible_converge, _patched_is_idempotent: Mock, _instance):
    if False:
        i = 10
        return i + 15
    _patched_is_idempotent.return_value = False
    with pytest.raises(SystemExit) as e:
        _instance.execute()
    assert e.value.code == 1
    msg = 'Idempotence test failed because of the following tasks:\n'
    assert msg in caplog.text

def test_is_idempotent(_instance):
    if False:
        print('Hello World!')
    output = '\nPLAY RECAP ***********************************************************\ncheck-command-01: ok=3    changed=0    unreachable=0    failed=0\n    '
    assert _instance._is_idempotent(output)

def test_is_idempotent_not_idempotent(_instance):
    if False:
        while True:
            i = 10
    output = '\nPLAY RECAP ***********************************************************\ncheck-command-01: ok=2    changed=1    unreachable=0    failed=0\ncheck-command-02: ok=2    changed=1    unreachable=0    failed=0\n    '
    assert not _instance._is_idempotent(output)

def test_non_idempotent_tasks_idempotent(_instance):
    if False:
        return 10
    output = '\nPLAY [all] ***********************************************************\n\nGATHERING FACTS ******************************************************\nok: [check-command-01]\n\nTASK: [Idempotence test] *********************************************\nok: [check-command-01]\n\nPLAY RECAP ***********************************************************\ncheck-command-01: ok=3    changed=0    unreachable=0    failed=0\n'
    result = _instance._non_idempotent_tasks(output)
    assert result == []

def test_non_idempotent_tasks_not_idempotent(_instance):
    if False:
        print('Hello World!')
    output = '\nPLAY [all] ***********************************************************\n\nGATHERING FACTS ******************************************************\nok: [check-command-01]\nok: [check-command-02]\n\nTASK: [Idempotence test] *********************************************\nchanged: [check-command-01]\nchanged: [check-command-02]\n\nPLAY RECAP ***********************************************************\ncheck-command-01: ok=2    changed=1    unreachable=0    failed=0\ncheck-command-02: ok=2    changed=1    unreachable=0    failed=0\n'
    result = _instance._non_idempotent_tasks(output)
    assert result == ['* [check-command-01] => Idempotence test', '* [check-command-02] => Idempotence test']