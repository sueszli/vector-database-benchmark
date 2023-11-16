import pytest
import testutils
sdvars = testutils.securedrop_test_vars
testinfra_hosts = [sdvars.app_hostname]

@pytest.mark.parametrize('exp_pip_pkg', sdvars.pip_deps)
def test_app_pip_deps(host, exp_pip_pkg):
    if False:
        for i in range(10):
            print('nop')
    'Ensure expected package versions are installed'
    cmd = '{}/bin/python3 -c "from importlib.metadata import version; print(version(\'{}\'))"'.format(sdvars.securedrop_venv, exp_pip_pkg['name'])
    result = host.run(cmd)
    assert result.stdout.strip() == exp_pip_pkg['version']

@pytest.mark.skip_in_prod()
def test_app_wsgi(host):
    if False:
        i = 10
        return i + 15
    'ensure logging is enabled for source interface in staging'
    f = host.file('/var/www/source.wsgi')
    with host.sudo():
        assert f.is_file
        assert f.mode == 420
        assert f.user == 'root'
        assert f.group == 'root'
        assert f.contains('^import logging$')
        assert f.contains('^logging\\.basicConfig(stream=sys\\.stderr)$')

def test_pidfile(host):
    if False:
        print('Hello World!')
    'ensure there are no pid files'
    assert not host.file('/tmp/journalist.pid').exists
    assert not host.file('/tmp/source.pid').exists

@pytest.mark.parametrize(('app_dir', 'owner'), [('/var/www/securedrop', 'root'), ('/var/lib/securedrop', 'www-data'), ('/var/lib/securedrop/store', 'www-data'), ('/var/lib/securedrop/keys', 'www-data'), ('/var/lib/securedrop/tmp', 'www-data')])
def test_app_directories(host, app_dir, owner):
    if False:
        while True:
            i = 10
    'ensure securedrop app directories exist with correct permissions'
    f = host.file(app_dir)
    mode = 493 if owner == 'root' else 448
    with host.sudo():
        assert f.is_directory
        assert f.user == owner
        assert f.group == owner
        assert f.mode == mode

def test_config_permissions(host):
    if False:
        for i in range(10):
            print('nop')
    'ensure config.py has correct permissions'
    f = host.file('/var/www/securedrop/config.py')
    with host.sudo():
        assert f.is_file
        assert f.user == 'root'
        assert f.group == 'www-data'
        assert f.mode == 416

def test_app_code_pkg(host):
    if False:
        i = 10
        return i + 15
    'ensure securedrop-app-code package is installed'
    assert host.package('securedrop-app-code').is_installed

def test_app_code_venv(host):
    if False:
        i = 10
        return i + 15
    '\n    Ensure the securedrop-app-code virtualenv is correct.\n    '
    cmd = 'test -z $VIRTUAL_ENV && . {}/bin/activate && test "$VIRTUAL_ENV" = "{}" '.format(sdvars.securedrop_venv, sdvars.securedrop_venv)
    result = host.run(cmd)
    assert result.rc == 0

def test_supervisor_not_installed(host):
    if False:
        i = 10
        return i + 15
    'ensure supervisor package is not installed'
    assert host.package('supervisor').is_installed is False

@pytest.mark.skip_in_prod()
def test_gpg_key_in_keyring(host):
    if False:
        i = 10
        return i + 15
    'ensure test gpg key is present in app keyring'
    with host.sudo(sdvars.securedrop_user):
        c = host.run('gpg --homedir /var/lib/securedrop/keys --list-keys 28271441')
        assert '2013-10-12' in c.stdout
        assert '28271441' in c.stdout

def test_ensure_logo(host):
    if False:
        i = 10
        return i + 15
    'ensure default logo header file exists'
    f = host.file(f'{sdvars.securedrop_code}/static/i/logo.png')
    with host.sudo():
        assert f.mode == 420
        assert f.user == 'root'
        assert f.group == 'root'

@pytest.mark.parametrize('user', ['root', 'www-data'])
def test_empty_crontabs(host, user):
    if False:
        return 10
    'Ensure root + www-data crontabs are empty'
    with host.sudo():
        host.run_expect([1], f'crontab -u {user} -l')