import logging
import pathlib
import pytest
import salt.states.ssh_auth as ssh_auth_state
import salt.utils.files
log = logging.getLogger(__name__)

@pytest.fixture
def configure_loader_modules(modules, minion_opts):
    if False:
        for i in range(10):
            print('nop')
    loader = {'__salt__': modules, '__opts__': minion_opts, '__env__': 'base'}
    return {ssh_auth_state: loader}

@pytest.fixture(scope='module')
def system_user():
    if False:
        i = 10
        return i + 15
    with pytest.helpers.create_account() as system_account:
        yield system_account

@pytest.mark.skip_if_not_root
@pytest.mark.destructive_test
@pytest.mark.slow_test
def test_ssh_auth_config(tmp_path, system_user, state_tree):
    if False:
        for i in range(10):
            print('nop')
    '\n    test running ssh_auth state when\n    different config is set. Ensure\n    it does not edit the default config.\n    '
    userdetails = system_user.info
    user_ssh_dir = pathlib.Path(userdetails.home, '.ssh')
    ret = ssh_auth_state.manage(name='test', user=system_user.username, ssh_keys=['ssh-dss AAAAB3NzaCL0sQ9fJ5bYTEyY== root@domain'])
    with salt.utils.files.fopen(user_ssh_dir / 'authorized_keys') as fp:
        pre_data = fp.read()
    file_contents = 'ssh-dss AAAAB3NzaCL0sQ9fJ5bYTEyY== root@domain'
    new_auth_file = tmp_path / 'authorized_keys3'
    with pytest.helpers.temp_file('authorized', file_contents, state_tree):
        ssh_auth_state.manage(name='test', user=system_user.username, source=f'salt://authorized', config=str(new_auth_file), ssh_keys=[''])
    with salt.utils.files.fopen(user_ssh_dir / 'authorized_keys') as fp:
        post_data = fp.read()
    assert pre_data == post_data
    with salt.utils.files.fopen(new_auth_file) as fp:
        data = fp.read().strip()
    assert data == file_contents