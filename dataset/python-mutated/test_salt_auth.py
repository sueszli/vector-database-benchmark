import logging
import pytest
log = logging.getLogger(__name__)
pytestmark = [pytest.mark.core_test, pytest.mark.skip_if_not_root, pytest.mark.destructive_test, pytest.mark.skip_on_windows]

@pytest.fixture(scope='module')
def salt_auth_account_1(salt_auth_account_1_factory):
    if False:
        return 10
    with salt_auth_account_1_factory as account:
        yield account

@pytest.fixture(scope='module')
def salt_auth_account_2(salt_auth_account_2_factory):
    if False:
        print('Hello World!')
    with salt_auth_account_2_factory as account:
        yield account

def test_pam_auth_valid_user(salt_minion, salt_cli, salt_auth_account_1):
    if False:
        return 10
    '\n    test that pam auth mechanism works with a valid user\n    '
    ret = salt_cli.run('-a', 'pam', '--username', salt_auth_account_1.username, '--password', salt_auth_account_1.password, 'test.ping', minion_tgt=salt_minion.id)
    assert ret.returncode == 0
    assert ret.data is True

def test_pam_auth_invalid_user(salt_minion, salt_cli):
    if False:
        return 10
    '\n    test pam auth mechanism errors for an invalid user\n    '
    ret = salt_cli.run('-a', 'pam', '--username', 'nouser', '--password', '1234', 'test.ping', minion_tgt=salt_minion.id)
    assert ret.stdout == 'Authentication error occurred.'

def test_pam_auth_valid_group(salt_minion, salt_cli, salt_auth_account_2):
    if False:
        while True:
            i = 10
    '\n    test that pam auth mechanism works for a valid group\n    '
    ret = salt_cli.run('-a', 'pam', '--username', salt_auth_account_2.username, '--password', salt_auth_account_2.password, 'test.ping', minion_tgt=salt_minion.id)
    assert ret.returncode == 0
    assert ret.data is True