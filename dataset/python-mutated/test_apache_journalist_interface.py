import re
import pytest
import testutils
securedrop_test_vars = testutils.securedrop_test_vars
testinfra_hosts = [securedrop_test_vars.app_hostname]

@pytest.mark.parametrize(('header', 'value'), securedrop_test_vars.wanted_apache_headers.items())
def test_apache_headers_journalist_interface(host, header, value):
    if False:
        print('Hello World!')
    '\n    Test for expected headers in Document Interface vhost config.\n    '
    f = host.file('/etc/apache2/sites-available/journalist.conf')
    assert f.is_file
    assert f.user == 'root'
    assert f.group == 'root'
    assert f.mode == 420
    header_unset = f'Header onsuccess unset {header}'
    assert f.contains(header_unset)
    header_set = f'Header always set {header} "{value}"'
    assert f.contains(header_set)

@pytest.mark.parametrize('apache_opt', [f'<VirtualHost {securedrop_test_vars.apache_listening_address}:8080>', 'WSGIDaemonProcess journalist processes=2 threads=30 display-name=%{{GROUP}} python-path={}'.format(securedrop_test_vars.securedrop_code), 'WSGIScriptAlias / /var/www/journalist.wsgi process-group=journalist application-group=journalist', 'WSGIPassAuthorization On', 'Header set Cache-Control "no-store"', f'Alias /static {securedrop_test_vars.securedrop_code}/static', 'XSendFile        On', 'LimitRequestBody 524288000', 'XSendFilePath    /var/lib/securedrop/store/', 'XSendFilePath    /var/lib/securedrop/tmp/', 'ErrorLog /var/log/apache2/journalist-error.log', 'CustomLog /var/log/apache2/journalist-access.log combined'])
def test_apache_config_journalist_interface(host, apache_opt):
    if False:
        i = 10
        return i + 15
    '\n    Ensure the necessary Apache settings for serving the application\n    are in place. Some values will change according to the host,\n    e.g. app-staging versus app-prod will have different listening\n    addresses, depending on whether Tor connections are forced.\n\n    These checks apply only to the Document Interface, used by Journalists.\n    '
    f = host.file('/etc/apache2/sites-available/journalist.conf')
    assert f.is_file
    assert f.user == 'root'
    assert f.group == 'root'
    assert f.mode == 420
    regex = f'^{re.escape(apache_opt)}$'
    assert re.search(regex, f.content_string, re.M)

def test_apache_config_journalist_interface_headers_per_distro(host):
    if False:
        i = 10
        return i + 15
    '\n    During migration to Focal, we updated the syntax for forcing HTTP headers.\n    '
    f = host.file('/etc/apache2/sites-available/journalist.conf')
    assert f.contains('Header onsuccess unset X-Frame-Options')
    assert f.contains('Header always set X-Frame-Options "DENY"')
    assert f.contains('Header onsuccess unset Referrer-Policy')
    assert f.contains('Header always set Referrer-Policy "no-referrer"')
    assert f.contains('Header edit Set-Cookie ^(.*)$ $1;HttpOnly')

def test_apache_logging_journalist_interface(host):
    if False:
        while True:
            i = 10
    '\n    Check that logging is configured correctly for the Journalist Interface.\n    The actions of Journalists are logged by the system, so that an Admin can\n    investigate incidents and track access.\n\n    Logs were broken for some period of time, logging only "combined" to\n    the logfile, rather than the combined LogFormat intended.\n    '
    with host.sudo():
        f = host.file('/var/log/apache2/journalist-access.log')
        assert f.is_file
        if f.size == 0:
            host.check_output('curl http://127.0.0.1:8080')
        assert f.size > 0
        assert not f.contains('^combined$')
        assert f.contains('GET')

@pytest.mark.parametrize('apache_opt', ['\n<Directory />\n  Options None\n  AllowOverride None\n  Require all denied\n</Directory>\n'.strip('\n'), '\n<Directory {}/static>\n  Require all granted\n  # Cache static resources for 1 hour\n  Header set Cache-Control "max-age=3600"\n</Directory>\n'.strip('\n').format(securedrop_test_vars.securedrop_code), '\n<Directory {}>\n  Options None\n  AllowOverride None\n  <Limit GET POST HEAD DELETE>\n    Require ip 127.0.0.1\n  </Limit>\n  <LimitExcept GET POST HEAD DELETE>\n    Require all denied\n  </LimitExcept>\n</Directory>\n'.strip('\n').format(securedrop_test_vars.securedrop_code)])
def test_apache_config_journalist_interface_access_control(host, apache_opt):
    if False:
        i = 10
        return i + 15
    '\n    Verifies the access control directives for the Journalist Interface.\n    '
    f = host.file('/etc/apache2/sites-available/journalist.conf')
    regex = f'^{re.escape(apache_opt)}$'
    assert re.search(regex, f.content_string, re.M)