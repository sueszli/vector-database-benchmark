"""
Functional tests for chocolatey state
"""
import os
import pathlib
import pytest
import salt.utils.path
import salt.utils.win_reg
pytestmark = [pytest.mark.windows_whitelisted, pytest.mark.skip_unless_on_windows, pytest.mark.slow_test, pytest.mark.destructive_test]

@pytest.fixture(scope='module')
def chocolatey(states):
    if False:
        for i in range(10):
            print('nop')
    yield states.chocolatey

@pytest.fixture(scope='module')
def chocolatey_mod(modules):
    if False:
        while True:
            i = 10
    current_path = salt.utils.win_reg.read_value(hive='HKLM', key='SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment', vname='PATH')['vdata']
    url = 'https://packages.chocolatey.org/chocolatey.1.2.1.nupkg'
    with pytest.helpers.temp_file(name='choco.nupkg') as nupkg:
        choco_pkg = pathlib.Path(str(nupkg))
    choco_dir = choco_pkg.parent / 'choco_dir'
    choco_script = choco_dir / 'tools' / 'chocolateyInstall.ps1'

    def install():
        if False:
            while True:
                i = 10
        modules.cp.get_url(path=url, dest=str(choco_pkg))
        modules.archive.unzip(zip_file=str(choco_pkg), dest=str(choco_dir), extract_perms=False)
        assert choco_script.exists()
        result = modules.cmd.script(source=str(choco_script), cwd=str(choco_script.parent), shell='powershell', python_shell=True)
        assert result['retcode'] == 0

    def uninstall():
        if False:
            print('Hello World!')
        choco_dir = os.environ.get('ChocolateyInstall', False)
        if choco_dir:
            modules.file.remove(path=choco_dir, force=True)
            for env_var in modules.environ.items():
                if env_var.lower().startswith('chocolatey'):
                    modules.environ.setval(key=env_var, val=False, false_unsets=True, permanent='HKLM')
                    modules.environ.setval(key=env_var, val=False, false_unsets=True, permanent='HKCU')
        salt.utils.win_reg.set_value(hive='HKLM', key='SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment', vname='PATH', vdata=current_path)
        modules.win_path.rehash()
    if salt.utils.path.which('choco.exe'):
        uninstall()
    install()
    yield modules.chocolatey
    uninstall()

@pytest.fixture(scope='function')
def clean(chocolatey_mod):
    if False:
        for i in range(10):
            print('nop')
    chocolatey_mod.uninstall(name='vim', force=True)
    yield
    chocolatey_mod.uninstall(name='vim', force=True)

@pytest.fixture(scope='function')
def vim(chocolatey_mod):
    if False:
        print('Hello World!')
    chocolatey_mod.install(name='vim', version='9.0.1672')
    yield
    chocolatey_mod.uninstall(name='vim', force=True)

def test_installed_latest(clean, chocolatey, chocolatey_mod):
    if False:
        while True:
            i = 10
    chocolatey.installed(name='vim')
    result = chocolatey_mod.version(name='vim')
    assert 'vim' in result

def test_installed_version(clean, chocolatey, chocolatey_mod):
    if False:
        for i in range(10):
            print('nop')
    chocolatey.installed(name='vim', version='9.0.1672')
    result = chocolatey_mod.version(name='vim')
    assert 'vim' in result
    assert result['vim']['installed'][0] == '9.0.1672'

def test_uninstalled(vim, chocolatey, chocolatey_mod):
    if False:
        for i in range(10):
            print('nop')
    chocolatey.uninstalled(name='vim')
    result = chocolatey_mod.version(name='vim')
    assert 'vim' not in result

def test_upgraded(vim, chocolatey, chocolatey_mod):
    if False:
        print('Hello World!')
    result = chocolatey_mod.version(name='vim')
    assert 'vim' in result
    assert result['vim']['installed'][0] == '9.0.1672'
    chocolatey.upgraded(name='vim', version='9.0.1677')
    result = chocolatey_mod.version(name='vim')
    assert 'vim' in result
    assert result['vim']['installed'][0] == '9.0.1677'