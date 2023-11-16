import os
import pathlib
import subprocess
import packaging.version
import psutil
import pytest
pytestmark = [pytest.mark.skip_unless_on_windows]

@pytest.fixture(autouse=True)
def _skip_on_less_than_3006_1(install_salt):
    if False:
        print('Hello World!')
    if packaging.version.parse(install_salt.version) <= packaging.version.parse('3006.1'):
        pytest.skip('Multi-minion script only available on versions greater than 3006.1')

@pytest.fixture
def mm_script(install_salt):
    if False:
        for i in range(10):
            print('nop')
    yield (install_salt.ssm_bin.parent / 'multi-minion.ps1')

@pytest.fixture(scope='function')
def mm_conf(mm_script):
    if False:
        while True:
            i = 10
    yield pathlib.Path(os.getenv('LocalAppData'), 'Salt Project', 'Salt', 'conf')
    subprocess.run(['powershell', str(mm_script).replace(' ', "' '"), '-d'], capture_output=True, check=False, text=True)

def test_script_present(mm_script):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensure the multi-minion.ps1 file is present in the root of the installation\n    '
    assert mm_script.exists()

def test_install(mm_script, mm_conf):
    if False:
        i = 10
        return i + 15
    '\n    Install a second minion with default settings. Should create a minion config\n    file in Local AppData\n    '
    ret = subprocess.run(['powershell', str(mm_script).replace(' ', "' '")], capture_output=True, check=False, text=True)
    assert ret.returncode == 0, ret.stderr
    conf_file = mm_conf / 'minion'
    assert conf_file.exists()
    assert conf_file.read_text().find('master: salt') > -1

def test_install_master(mm_script, mm_conf):
    if False:
        return 10
    '\n    Install a second minion and set the master to spongebob\n    '
    ret = subprocess.run(['powershell', str(mm_script).replace(' ', "' '"), '-m', 'spongebob'], capture_output=True, check=False, text=True)
    assert ret.returncode == 0, ret.stderr
    conf_file = mm_conf / 'minion'
    assert conf_file.exists()
    assert conf_file.read_text().find('master: spongebob') > -1

def test_install_prefix(mm_script, mm_conf):
    if False:
        return 10
    '\n    Install a second minion and add a prefix to the minion id\n    '
    ret = subprocess.run(['powershell', str(mm_script).replace(' ', "' '"), '-p', 'squarepants'], capture_output=True, check=False, text=True)
    assert ret.returncode == 0, ret.stderr
    conf_file = mm_conf / 'minion'
    assert conf_file.exists()
    assert conf_file.read_text().find('id: squarepants') > -1

def test_install_log_level(mm_script, mm_conf):
    if False:
        for i in range(10):
            print('nop')
    '\n    Install a second minion and set the log level in the log file to debug\n    '
    ret = subprocess.run(['powershell', str(mm_script).replace(' ', "' '"), '-l', 'debug'], capture_output=True, check=False, text=True)
    assert ret.returncode == 0, ret.stderr
    conf_file = mm_conf / 'minion'
    assert conf_file.exists()
    assert conf_file.read_text().find('log_level_logfile: debug') > -1

def test_install_start(mm_script, mm_conf):
    if False:
        print('Hello World!')
    '\n    Install a second minion and start that minion in a hidden process\n    '
    ret = subprocess.run(['powershell', str(mm_script).replace(' ', "' '"), '-s'], capture_output=True, check=False, text=True)
    assert ret.returncode == 0, ret.stderr
    conf_file = mm_conf / 'minion'
    assert conf_file.exists()
    assert conf_file.read_text().find('master: salt') > -1
    found = False
    for p in psutil.process_iter(['cmdline', 'name']):
        if p.info['name'] and p.info['name'] == 'salt-minion.exe':
            if f'{mm_conf}' in p.info['cmdline']:
                found = True
    assert found is True