import pytest
import testutils
securedrop_test_vars = testutils.securedrop_test_vars
testinfra_hosts = [securedrop_test_vars.app_hostname]
python_version = securedrop_test_vars.python_version

def test_apache_default_docroot_is_absent(host):
    if False:
        while True:
            i = 10
    '\n    Ensure that the default docroot for Apache, containing static HTML\n    under Debian, has been removed. Leaving it in place can be a privacy\n    leak, as it displays version information by default.\n    '
    assert not host.file('/var/www/html').exists

@pytest.mark.parametrize('package', ['apache2', 'apparmor-utils', 'coreutils', 'gnupg2', 'libapache2-mod-xsendfile', f'libpython{python_version}', 'paxctld', 'python3', 'redis-server', 'securedrop-config', 'securedrop-keyring', 'sqlite3'])
def test_securedrop_application_apt_dependencies(host, package):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensure apt dependencies required to install `securedrop-app-code`\n    are present. These should be pulled in automatically via apt,\n    due to specification in Depends in package control file.\n    '
    assert host.package(package).is_installed

@pytest.mark.parametrize('package', ['cron-apt', 'haveged', 'libapache2-mod-wsgi', 'ntp', 'ntpdate', 'supervisor'])
def test_unwanted_packages_absent(host, package):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensure packages that conflict with `securedrop-app-code`\n    or are otherwise unwanted are not present.\n    '
    assert not host.package(package).is_installed

@pytest.mark.skip_in_prod()
def test_securedrop_application_test_locale(host):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensure both SecureDrop DEFAULT_LOCALE and SUPPORTED_LOCALES are present.\n    '
    securedrop_config = host.file(f'{securedrop_test_vars.securedrop_code}/config.py')
    with host.sudo():
        assert securedrop_config.is_file
        assert securedrop_config.contains('^DEFAULT_LOCALE')
        assert securedrop_config.content_string.count('DEFAULT_LOCALE') == 1
        assert securedrop_config.content_string.count('SUPPORTED_LOCALES') == 1
        assert "\nSUPPORTED_LOCALES = ['el', 'ar', 'en_US']\n" in securedrop_config.content_string

@pytest.mark.skip_in_prod()
def test_securedrop_application_test_journalist_key(host):
    if False:
        return 10
    '\n    Ensure the SecureDrop Application GPG public key file is present.\n    This is a test-only pubkey provided in the repository strictly for testing.\n    '
    pubkey_file = host.file(f'{securedrop_test_vars.securedrop_data}/journalist.pub')
    with host.sudo():
        assert pubkey_file.is_file
        assert pubkey_file.user == 'root'
        assert pubkey_file.group == 'www-data'
        assert pubkey_file.mode == 416
    securedrop_config = host.file(f'{securedrop_test_vars.securedrop_code}/config.py')
    with host.sudo():
        assert securedrop_config.is_file
        assert securedrop_config.user == securedrop_test_vars.securedrop_code_owner
        assert securedrop_config.group == securedrop_test_vars.securedrop_user
        assert securedrop_config.mode == 416
        assert securedrop_config.contains("^JOURNALIST_KEY = '65A1B5FF195B56353CC63DFFCC40EF1228271441'$")

def test_securedrop_application_sqlite_db(host):
    if False:
        i = 10
        return i + 15
    '\n    Ensure sqlite database exists for application. The database file should be\n    created by Ansible on first run.\n    '
    with host.sudo():
        f = host.file(f'{securedrop_test_vars.securedrop_data}/db.sqlite')
        assert f.is_file
        assert f.user == securedrop_test_vars.securedrop_user
        assert f.group == securedrop_test_vars.securedrop_user
        assert f.mode == 416