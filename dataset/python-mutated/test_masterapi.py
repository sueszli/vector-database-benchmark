"""
Test autosigning minions based on grain values.
"""
import os
import shutil
import stat
import subprocess
import pytest
import salt.utils.files
import salt.utils.stringutils
from tests.support.runtests import RUNTIME_VARS
pytestmark = [pytest.mark.slow_test]

@pytest.fixture
def autosign_file_permissions():
    if False:
        while True:
            i = 10
    return stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWUSR

@pytest.fixture
def autosign_file_path(tmp_path):
    if False:
        print('Hello World!')
    return str(tmp_path / 'autosign_file')

@pytest.fixture
def autosign_grains_dir(salt_master):
    if False:
        return 10
    return salt_master.config['autosign_grains_dir']

@pytest.fixture(autouse=True)
def setup_autosign_files(autosign_file_permissions, autosign_file_path, autosign_grains_dir, salt_key_cli, salt_call_cli, salt_minion):
    if False:
        return 10
    shutil.copyfile(os.path.join(RUNTIME_VARS.FILES, 'autosign_grains', 'autosign_file'), autosign_file_path)
    os.chmod(autosign_file_path, autosign_file_permissions)
    salt_key_cli.run('-d', salt_minion.id, '-y')
    salt_call_cli.run('test.ping', '-l', 'quiet')
    skip_msg = ''
    if salt_minion.id in salt_key_cli.run('-l', 'acc').data:
        skip_msg = 'Could not deauthorize minion'
    if salt_minion.id not in salt_key_cli.run('-l', 'un').data:
        skip_msg = 'minion did not try to reauthenticate itself'
    if not os.path.isdir(autosign_grains_dir):
        os.makedirs(autosign_grains_dir)
    if not skip_msg:
        yield
    shutil.copyfile(os.path.join(RUNTIME_VARS.FILES, 'autosign_file'), autosign_file_path)
    os.chmod(autosign_file_path, autosign_file_permissions)
    salt_call_cli.run('test.ping', '-l', 'quiet')
    try:
        if os.path.isdir(autosign_grains_dir):
            shutil.rmtree(autosign_grains_dir)
    except AttributeError:
        pass
    if skip_msg:
        pytest.skip(skip_msg)

@pytest.mark.slow_test
def test_autosign_grains_accept(autosign_grains_dir, autosign_file_permissions, salt_minion, salt_call_cli, salt_key_cli):
    if False:
        i = 10
        return i + 15
    grain_file_path = os.path.join(autosign_grains_dir, 'test_grain')
    with salt.utils.files.fopen(grain_file_path, 'w') as f:
        f.write(salt.utils.stringutils.to_str('#invalid_value\ncheese'))
    os.chmod(grain_file_path, autosign_file_permissions)
    salt_call_cli.run('test.ping', '-l', 'quiet')
    assert salt_minion.id in salt_key_cli.run('-l', 'acc')

@pytest.mark.slow_test
def test_autosign_grains_fail(autosign_grains_dir, autosign_file_permissions, salt_minion, salt_call_cli, salt_key_cli):
    if False:
        print('Hello World!')
    grain_file_path = os.path.join(autosign_grains_dir, 'test_grain')
    with salt.utils.files.fopen(grain_file_path, 'w') as f:
        f.write(salt.utils.stringutils.to_str('#cheese\ninvalid_value'))
    os.chmod(grain_file_path, autosign_file_permissions)
    salt_call_cli.run('test.ping', '-l', 'quiet')
    assert salt_minion.id not in salt_key_cli.run('-l', 'acc')
    assert salt_minion.id in salt_key_cli.run('-l', 'un')

@pytest.mark.skip_unless_on_linux
@pytest.mark.slow_test
def test_ufw_allow(salt_master, grains):
    if False:
        i = 10
        return i + 15
    if grains['os_family'] != 'Debian':
        pytest.skip('Only runs on Debian family.')
    expected_output = 'Skipping adding existing rule\nSkipping adding existing rule (v6)\n    '
    proc = subprocess.Popen('ufw allow salt\n', shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    (out, err) = proc.communicate()
    out_strg = out.decode()
    err_strg = err.decode()
    assert out_strg == expected_output
    assert err_strg != "ERROR: Could not find a profile matching 'salt'"