import pytest
import salt.loader
import salt.modules.win_lgpo as win_lgpo_module
import salt.states.win_lgpo as win_lgpo_state
pytestmark = [pytest.mark.windows_whitelisted, pytest.mark.skip_unless_on_windows, pytest.mark.destructive_test, pytest.mark.slow_test]

@pytest.fixture
def configure_loader_modules(minion_opts, modules):
    if False:
        return 10
    utils = salt.loader.utils(minion_opts)
    return {win_lgpo_state: {'__opts__': minion_opts, '__salt__': modules, '__utils__': utils}, win_lgpo_module: {'__opts__': minion_opts, '__salt__': modules, '__utils__': utils}}

@pytest.fixture(scope='module')
def disable_legacy_auditing():
    if False:
        i = 10
        return i + 15
    from tests.support.sminion import create_sminion
    salt_minion = create_sminion()
    test_setting = 'Enabled'
    pre_security_setting = salt_minion.functions.lgpo.get_policy(policy_name='SceNoApplyLegacyAuditPolicy', policy_class='machine')
    pre_audit_setting = salt_minion.functions.lgpo.get_policy(policy_name='Audit User Account Management', policy_class='machine')
    try:
        if pre_security_setting != test_setting:
            salt_minion.functions.lgpo.set_computer_policy(name='SceNoApplyLegacyAuditPolicy', setting=test_setting)
            assert salt_minion.functions.lgpo.get_policy(policy_name='SceNoApplyLegacyAuditPolicy', policy_class='machine') == test_setting
        yield
    finally:
        salt_minion.functions.lgpo.set_computer_policy(name='SceNoApplyLegacyAuditPolicy', setting=pre_security_setting)
        salt_minion.functions.lgpo.set_computer_policy(name='Audit User Account Management', setting=pre_audit_setting)

@pytest.fixture
def clear_policy():
    if False:
        i = 10
        return i + 15
    test_setting = 'No Auditing'
    win_lgpo_module.set_computer_policy(name='Audit User Account Management', setting=test_setting)
    assert win_lgpo_module.get_policy(policy_name='Audit User Account Management', policy_class='machine') == test_setting

@pytest.fixture
def set_policy():
    if False:
        while True:
            i = 10
    test_setting = 'Success'
    win_lgpo_module.set_computer_policy(name='Audit User Account Management', setting=test_setting)
    assert win_lgpo_module.get_policy(policy_name='Audit User Account Management', policy_class='machine') == test_setting

def _test_adv_auditing(setting, expected):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function to set an audit setting and assert that it was successful\n    '
    win_lgpo_state.set_(name='Audit User Account Management', setting=setting, policy_class='machine')
    result = win_lgpo_module._get_advaudit_value('Audit User Account Management', refresh=True)
    assert result == expected

def test_no_auditing(disable_legacy_auditing, set_policy):
    if False:
        while True:
            i = 10
    _test_adv_auditing('No Auditing', '0')

def test_success(disable_legacy_auditing, clear_policy):
    if False:
        return 10
    _test_adv_auditing('Success', '1')

def test_failure(disable_legacy_auditing, clear_policy):
    if False:
        while True:
            i = 10
    _test_adv_auditing('Failure', '2')

def test_success_and_failure(disable_legacy_auditing, clear_policy):
    if False:
        return 10
    _test_adv_auditing('Success and Failure', '3')