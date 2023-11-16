import re
import pytest
import testutils
securedrop_test_vars = testutils.securedrop_test_vars
testinfra_hosts = [securedrop_test_vars.app_hostname]

@pytest.mark.parametrize('package', ['libapache2-mod-xsendfile'])
def test_apache_apt_packages(host, package):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensure required Apache packages are installed.\n    '
    assert host.package(package).is_installed

def test_apache_security_config_deprecated(host):
    if False:
        i = 10
        return i + 15
    '\n    Ensure that /etc/apache2/security is absent, since it was setting\n    redundant options already present in /etc/apache2/apache2.conf.\n    See #643 for discussion.\n    '
    assert not host.file('/etc/apache2/security').exists

@pytest.mark.parametrize('apache_opt', ['Mutex file:${APACHE_LOCK_DIR} default', 'PidFile ${APACHE_PID_FILE}', 'Timeout 60', 'KeepAlive On', 'MaxKeepAliveRequests 100', 'KeepAliveTimeout 5', 'User www-data', 'Group www-data', 'AddDefaultCharset UTF-8', 'DefaultType None', 'HostnameLookups Off', 'ErrorLog /dev/null', 'LogLevel crit', 'IncludeOptional mods-enabled/*.load', 'IncludeOptional mods-enabled/*.conf', 'Include ports.conf', 'IncludeOptional sites-enabled/*.conf', 'ServerTokens Prod', 'ServerSignature Off', 'TraceEnable Off'])
def test_apache_config_settings(host, apache_opt):
    if False:
        while True:
            i = 10
    '\n    Check required Apache config settings for general server.\n    These checks do not target individual interfaces, e.g.\n    Source versus Document Interface, and instead apply to\n    Apache more generally.\n    '
    f = host.file('/etc/apache2/apache2.conf')
    assert f.is_file
    assert f.user == 'root'
    assert f.group == 'root'
    assert f.mode == 420
    assert re.search(f'^{re.escape(apache_opt)}$', f.content_string, re.M)

@pytest.mark.parametrize('port', ['80', '8080'])
def test_apache_ports_config(host, port):
    if False:
        return 10
    "\n    Ensure Apache ports config items, which specify how the\n    Source and Document Interfaces are configured to be served\n    over Tor. On staging hosts, they will listen on any interface,\n    to permit port forwarding for local testing, but in production,\n    they're restricted to localhost, for use over Tor.\n    "
    f = host.file('/etc/apache2/ports.conf')
    assert f.is_file
    assert f.user == 'root'
    assert f.group == 'root'
    assert f.mode == 420
    listening_regex = '^Listen {}:{}$'.format(re.escape(securedrop_test_vars.apache_listening_address), port)
    assert f.contains(listening_regex)

@pytest.mark.parametrize('apache_module', ['access_compat', 'authn_core', 'alias', 'authz_core', 'authz_host', 'authz_user', 'deflate', 'filter', 'dir', 'headers', 'mime', 'mpm_event', 'negotiation', 'reqtimeout', 'rewrite', 'wsgi', 'xsendfile'])
def test_apache_modules_present(host, apache_module):
    if False:
        return 10
    '\n    Ensure presence of required Apache modules. Application will not work\n    correctly if these are missing. A separate test will check for\n    disabled modules.\n    '
    with host.sudo():
        c = host.run(f'/usr/sbin/a2query -m {apache_module}')
        assert f'{apache_module} (enabled' in c.stdout
        assert c.rc == 0

@pytest.mark.parametrize('apache_module', ['auth_basic', 'authn_file', 'autoindex', 'env', 'status'])
def test_apache_modules_absent(host, apache_module):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensure absence of unwanted Apache modules. Application does not require\n    these modules, so they should be disabled to reduce attack surface.\n    A separate test will check for disabled modules.\n    '
    with host.sudo():
        c = host.run(f'/usr/sbin/a2query -m {apache_module}')
        assert f'No module matches {apache_module} (disabled' in c.stderr
        assert c.rc == 32

@pytest.mark.parametrize('logfile', securedrop_test_vars.allowed_apache_logfiles)
def test_apache_logfiles_present(host, logfile):
    if False:
        for i in range(10):
            print('nop')
    ' "\n    Ensure that whitelisted Apache log files for the Source and Journalist\n    Interfaces are present. In staging, we permit a "source-error" log,\n    but on prod even that is not allowed. A separate test will confirm\n    absence of unwanted logfiles by comparing the file count in the\n    Apache log directory.\n    '
    with host.sudo():
        f = host.file(logfile)
        assert f.is_file
        assert f.user == 'root'

def test_apache_logfiles_no_extras(host):
    if False:
        i = 10
        return i + 15
    "\n    Ensure that no unwanted Apache logfiles are present. Complements the\n    `test_apache_logfiles_present` config test. Here, we confirm that the\n    total number of Apache logfiles exactly matches the number permitted\n    on the Application Server, whether staging or prod.\n    Long-running instances may have rotated and gzipped logfiles, so this\n    test should only look for files ending in '.log'.\n    "
    with host.sudo():
        c = host.run("find /var/log/apache2 -mindepth 1 -name '*.log' | wc -l")
        assert int(c.stdout) == len(securedrop_test_vars.allowed_apache_logfiles)