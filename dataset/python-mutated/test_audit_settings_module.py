import os
import pytest
pytestmark = [pytest.mark.windows_whitelisted, pytest.mark.skip_unless_on_windows, pytest.mark.destructive_test, pytest.mark.slow_test]

@pytest.fixture(scope='module')
def lgpo(modules):
    if False:
        return 10
    return modules.lgpo

@pytest.fixture(scope='module')
def enable_legacy_auditing(lgpo):
    if False:
        while True:
            i = 10
    try:
        lgpo.set_computer_policy('SceNoApplyLegacyAuditPolicy', 'Disabled')
        lgpo.set_computer_policy('Audit account management', 'No auditing')
        check = lgpo.get_policy('SceNoApplyLegacyAuditPolicy', 'machine')
        assert check == 'Disabled'
        check = lgpo.get_policy('Audit account management', 'machine')
        assert check == 'No auditing'
        yield
    finally:
        lgpo.set_computer_policy('SceNoApplyLegacyAuditPolicy', 'Not Defined')
        lgpo.set_computer_policy('Audit account management', 'Not Defined')

@pytest.fixture(scope='module')
def clean_adv_audit():
    if False:
        return 10
    win_dir = os.environ.get('WINDIR')
    audit_csv_files = ['{}\\security\\audit\\audit.csv'.format(win_dir), '{}\\System32\\GroupPolicy\\Machine\\Microsoft\\Windows NT\\Audit\\audit.csv'.format(win_dir)]
    for audit_file in audit_csv_files:
        if os.path.exists(audit_file):
            os.remove(audit_file)
    yield

@pytest.fixture(scope='module')
def legacy_auditing_not_defined(lgpo):
    if False:
        print('Hello World!')
    try:
        lgpo.set_computer_policy('SceNoApplyLegacyAuditPolicy', 'Not Defined')
        check = lgpo.get_policy('SceNoApplyLegacyAuditPolicy', 'machine')
        assert check == 'Not Defined'
        yield
    finally:
        lgpo.set_computer_policy('SceNoApplyLegacyAuditPolicy', 'Not Defined')

@pytest.mark.parametrize('setting', ['No auditing', 'Success', 'Failure', 'Success, Failure'])
def test_auditing(lgpo, setting, enable_legacy_auditing, clean_adv_audit):
    if False:
        print('Hello World!')
    '\n    Helper function to set an audit setting and assert that it was successful\n    '
    lgpo.set_computer_policy('Audit account management', setting)
    result = lgpo.get_policy('Audit account management', 'machine')
    assert result == setting

@pytest.mark.parametrize('setting_name,setting', [('Audit account management', 'Success'), ('Audit Account Management', 'Failure')])
def test_auditing_case_names(lgpo, setting_name, setting, enable_legacy_auditing, clean_adv_audit):
    if False:
        print('Hello World!')
    '\n    Helper function to set an audit setting and assert that it was successful\n    '
    lgpo.set_computer_policy(setting_name, setting)
    result = lgpo.get_policy(setting_name, 'machine')
    assert result == setting

@pytest.mark.parametrize('setting', ['Enabled', 'Disabled'])
def test_enable_legacy_audit_policy(lgpo, setting, legacy_auditing_not_defined, clean_adv_audit):
    if False:
        for i in range(10):
            print('nop')
    lgpo.set_computer_policy('SceNoApplyLegacyAuditPolicy', setting)
    result = lgpo.get_policy('SceNoApplyLegacyAuditPolicy', 'machine')
    assert result == setting