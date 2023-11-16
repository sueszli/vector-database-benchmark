"""
Test for ssh_pre_flight roster option
"""
try:
    import grp
    import pwd
except ImportError:
    pass
import os
import pathlib
import shutil
import subprocess
import pytest
import yaml
from saltfactories.utils import random_string
import salt.utils.files
pytestmark = pytest.mark.skip_on_windows(reason='Salt-ssh not available on Windows')

def _custom_roster(roster_file, roster_data):
    if False:
        return 10
    with salt.utils.files.fopen(roster_file, 'r') as fp:
        data = salt.utils.yaml.safe_load(fp)
    for (key, item) in roster_data.items():
        data['localhost'][key] = item
    with salt.utils.files.fopen(roster_file, 'w') as fp:
        yaml.safe_dump(data, fp)

@pytest.fixture
def _create_roster(salt_ssh_roster_file, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    ret = {}
    ret['roster'] = salt_ssh_roster_file
    ret['data'] = {'ssh_pre_flight': str(tmp_path / 'ssh_pre_flight.sh')}
    ret['test_script'] = str(tmp_path / 'test-pre-flight-script-worked.txt')
    ret['thin_dir'] = tmp_path / 'thin_dir'
    with salt.utils.files.fopen(salt_ssh_roster_file, 'r') as fp:
        data = salt.utils.yaml.safe_load(fp)
    pre_flight_script = ret['data']['ssh_pre_flight']
    data['localhost']['ssh_pre_flight'] = pre_flight_script
    data['localhost']['thin_dir'] = str(ret['thin_dir'])
    with salt.utils.files.fopen(salt_ssh_roster_file, 'w') as fp:
        yaml.safe_dump(data, fp)
    with salt.utils.files.fopen(pre_flight_script, 'w') as fp:
        fp.write('touch {}'.format(ret['test_script']))
    yield ret
    if ret['thin_dir'].exists():
        shutil.rmtree(ret['thin_dir'])

@pytest.mark.slow_test
def test_ssh_pre_flight(salt_ssh_cli, caplog, _create_roster):
    if False:
        return 10
    '\n    test ssh when ssh_pre_flight is set\n    ensure the script runs successfully\n    '
    ret = salt_ssh_cli.run('test.ping')
    assert ret.returncode == 0
    assert pathlib.Path(_create_roster['test_script']).exists()

@pytest.mark.slow_test
def test_ssh_run_pre_flight(salt_ssh_cli, _create_roster):
    if False:
        return 10
    '\n    test ssh when --pre-flight is passed to salt-ssh\n    to ensure the script runs successfully\n    '
    ret = salt_ssh_cli.run('test.ping')
    assert pathlib.Path(_create_roster['test_script']).exists()
    pathlib.Path(_create_roster['test_script']).unlink()
    assert salt_ssh_cli.run('test.ping').returncode == 0
    assert not pathlib.Path(_create_roster['test_script']).exists()
    ret = salt_ssh_cli.run('test.ping', '--pre-flight')
    assert ret.returncode == 0
    assert pathlib.Path(_create_roster['test_script']).exists()

@pytest.mark.slow_test
def test_ssh_run_pre_flight_args(salt_ssh_cli, _create_roster):
    if False:
        return 10
    '\n    test ssh when --pre-flight is passed to salt-ssh\n    to ensure the script runs successfully passing some args\n    '
    _custom_roster(salt_ssh_cli.roster_file, {'ssh_pre_flight_args': 'foobar test'})
    test_script = _create_roster['test_script']
    test_script_1 = pathlib.Path(test_script + '-foobar')
    test_script_2 = pathlib.Path(test_script + '-test')
    with salt.utils.files.fopen(_create_roster['data']['ssh_pre_flight'], 'w') as fp:
        fp.write(f'\n        touch {str(test_script)}-$1\n        touch {str(test_script)}-$2\n        ')
    ret = salt_ssh_cli.run('test.ping')
    assert ret.returncode == 0
    assert test_script_1.exists()
    assert test_script_2.exists()
    pathlib.Path(test_script_1).unlink()
    pathlib.Path(test_script_2).unlink()
    ret = salt_ssh_cli.run('test.ping')
    assert ret.returncode == 0
    assert not test_script_1.exists()
    assert not test_script_2.exists()
    ret = salt_ssh_cli.run('test.ping', '--pre-flight')
    assert ret.returncode == 0
    assert test_script_1.exists()
    assert test_script_2.exists()

@pytest.mark.slow_test
def test_ssh_run_pre_flight_args_prevent_injection(salt_ssh_cli, _create_roster, tmp_path):
    if False:
        print('Hello World!')
    '\n    test ssh when --pre-flight is passed to salt-ssh\n    and evil arguments are used in order to produce shell injection\n    '
    injected_file = tmp_path / 'injection'
    _custom_roster(salt_ssh_cli.roster_file, {'ssh_pre_flight_args': f'foobar; echo injected > {str(injected_file)}'})
    test_script = _create_roster['test_script']
    test_script_1 = pathlib.Path(test_script + '-echo')
    test_script_2 = pathlib.Path(test_script + '-foobar;')
    with salt.utils.files.fopen(_create_roster['data']['ssh_pre_flight'], 'w') as fp:
        fp.write(f'\n        touch {str(test_script)}-$1\n        touch {str(test_script)}-$2\n        ')
    ret = salt_ssh_cli.run('test.ping')
    assert ret.returncode == 0
    assert test_script_1.exists()
    assert test_script_2.exists()
    test_script_1.unlink()
    test_script_2.unlink()
    assert not injected_file.is_file()
    ret = salt_ssh_cli.run('test.ping', '--pre-flight')
    assert ret.returncode == 0
    assert test_script_1.exists()
    assert test_script_2.exists()
    assert not pathlib.Path(injected_file).is_file(), "File injection suceeded. This shouldn't happend"

@pytest.mark.flaky(max_runs=4)
@pytest.mark.slow_test
def test_ssh_run_pre_flight_failure(salt_ssh_cli, _create_roster):
    if False:
        i = 10
        return i + 15
    '\n    test ssh_pre_flight when there is a failure\n    in the script.\n    '
    with salt.utils.files.fopen(_create_roster['data']['ssh_pre_flight'], 'w') as fp_:
        fp_.write('exit 2')
    ret = salt_ssh_cli.run('test.ping', '--pre-flight')
    assert ret.data['retcode'] == 2

@pytest.fixture
def account():
    if False:
        print('Hello World!')
    username = random_string('test-account-', uppercase=False)
    with pytest.helpers.create_account(username=username) as account:
        yield account

@pytest.mark.slow_test
def test_ssh_pre_flight_script(salt_ssh_cli, caplog, _create_roster, tmp_path, account):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test to ensure user cannot create and run a script\n    with the expected pre_flight script path on target.\n    '
    try:
        script = pathlib.Path.home() / 'hacked'
        tmp_preflight = pathlib.Path('/tmp', 'ssh_pre_flight.sh')
        tmp_preflight.write_text(f'touch {script}')
        os.chown(tmp_preflight, account.info.uid, account.info.gid)
        ret = salt_ssh_cli.run('test.ping')
        assert not script.is_file()
        assert ret.returncode == 0
        assert ret.stdout == '{\n"localhost": true\n}\n'
    finally:
        for _file in [script, tmp_preflight]:
            if _file.is_file():
                _file.unlink()

def demote(user_uid, user_gid):
    if False:
        return 10

    def result():
        if False:
            i = 10
            return i + 15
        os.setgroups([])
        os.setgid(user_gid)
        os.setuid(user_uid)
    return result

@pytest.mark.slow_test
def test_ssh_pre_flight_perms(salt_ssh_cli, caplog, _create_roster, account):
    if False:
        i = 10
        return i + 15
    '\n    Test to ensure standard user cannot run pre flight script\n    on target when user sets wrong permissions (777) on\n    ssh_pre_flight script.\n    '
    try:
        script = pathlib.Path('/tmp', 'itworked')
        preflight = pathlib.Path('/ssh_pre_flight.sh')
        preflight.write_text(f'touch {str(script)}')
        tmp_preflight = pathlib.Path('/tmp', preflight.name)
        _custom_roster(salt_ssh_cli.roster_file, {'ssh_pre_flight': str(preflight)})
        preflight.chmod(511)
        run_script = pathlib.Path('/run_script')
        run_script.write_text(f'\n        x=1\n        while [ $x -le 200000 ]; do\n            SCRIPT=`bash {str(tmp_preflight)} 2> /dev/null; echo $?`\n            if [ ${{SCRIPT}} == 0 ]; then\n                break\n            fi\n            x=$(( $x + 1 ))\n        done\n        ')
        run_script.chmod(511)
        ret = subprocess.Popen(['sh', f'{run_script}'], preexec_fn=demote(account.info.uid, account.info.gid), stdout=None, stderr=None, stdin=None, universal_newlines=True)
        ret = salt_ssh_cli.run('test.ping')
        assert ret.returncode == 0
        assert os.stat(script).st_uid != account.info.uid
        assert script.is_file()
    finally:
        for _file in [script, preflight, tmp_preflight, run_script]:
            if _file.is_file():
                _file.unlink()

@pytest.mark.slow_test
def test_ssh_run_pre_flight_target_file_perms(salt_ssh_cli, _create_roster, tmp_path):
    if False:
        i = 10
        return i + 15
    '\n    test ssh_pre_flight to ensure the target pre flight script\n    has the correct perms\n    '
    perms_file = tmp_path / 'perms'
    with salt.utils.files.fopen(_create_roster['data']['ssh_pre_flight'], 'w') as fp_:
        fp_.write(f'\n        SCRIPT_NAME=$0\n        stat -L -c "%a %G %U" $SCRIPT_NAME > {perms_file}\n        ')
    ret = salt_ssh_cli.run('test.ping', '--pre-flight')
    assert ret.returncode == 0
    with salt.utils.files.fopen(perms_file) as fp:
        data = fp.read()
    assert data.split()[0] == '600'
    uid = os.getuid()
    gid = os.getgid()
    assert data.split()[1] == grp.getgrgid(gid).gr_name
    assert data.split()[2] == pwd.getpwuid(uid).pw_name