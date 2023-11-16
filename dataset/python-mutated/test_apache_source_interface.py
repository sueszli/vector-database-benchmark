import re
import pytest
import testutils
securedrop_test_vars = testutils.securedrop_test_vars
testinfra_hosts = [securedrop_test_vars.app_hostname]

@pytest.mark.parametrize(('header', 'value'), securedrop_test_vars.wanted_apache_headers.items())
def test_apache_headers_source_interface(host, header, value):
    if False:
        return 10
    '\n    Test for expected headers in Source Interface vhost config.\n    '
    f = host.file('/etc/apache2/sites-available/source.conf')
    assert f.is_file
    assert f.user == 'root'
    assert f.group == 'root'
    assert f.mode == 420
    header_unset = f'Header onsuccess unset {header}'
    assert f.contains(header_unset)
    header_set = f'Header always set {header} "{value}"'
    assert f.contains(header_set)

@pytest.mark.parametrize('apache_opt', [f'<VirtualHost {securedrop_test_vars.apache_listening_address}:80>', 'WSGIDaemonProcess source  processes=2 threads=30 display-name=%{{GROUP}} python-path={}'.format(securedrop_test_vars.securedrop_code), 'WSGIProcessGroup source', 'WSGIScriptAlias / /var/www/source.wsgi', 'Header set Cache-Control "no-store"', 'Header unset Etag', f'Alias /static {securedrop_test_vars.securedrop_code}/static', 'XSendFile        Off', 'LimitRequestBody 524288000', f'ErrorLog {securedrop_test_vars.apache_source_log}'])
def test_apache_config_source_interface(host, apache_opt):
    if False:
        i = 10
        return i + 15
    '\n    Ensure the necessary Apache settings for serving the application\n    are in place. Some values will change according to the host,\n    e.g. app-staging versus app-prod will have different listening\n    addresses, depending on whether Tor connections are forced.\n\n    These checks apply only to the Source Interface, used by Sources.\n    '
    f = host.file('/etc/apache2/sites-available/source.conf')
    assert f.is_file
    assert f.user == 'root'
    assert f.group == 'root'
    assert f.mode == 420
    regex = f'^{re.escape(apache_opt)}$'
    assert re.search(regex, f.content_string, re.M)

def test_apache_config_source_interface_headers_per_distro(host):
    if False:
        while True:
            i = 10
    '\n    During migration to Focal, we updated the syntax for forcing HTTP headers.\n    '
    f = host.file('/etc/apache2/sites-available/source.conf')
    assert f.contains('Header onsuccess unset X-Frame-Options')
    assert f.contains('Header always set X-Frame-Options "DENY"')
    assert f.contains('Header onsuccess unset Referrer-Policy')
    assert f.contains('Header always set Referrer-Policy "same-origin"')
    assert f.contains('Header edit Set-Cookie ^(.*)$ $1;HttpOnly')

@pytest.mark.parametrize('apache_opt', ['\n<Directory />\n  Options None\n  AllowOverride None\n  Require all denied\n</Directory>\n'.strip('\n'), '\n<Directory {}/static>\n  Require all granted\n  # Cache static resources for 1 hour\n  Header set Cache-Control "max-age=3600"\n</Directory>\n'.strip('\n').format(securedrop_test_vars.securedrop_code), '\n<Directory {}>\n  Options None\n  AllowOverride None\n  <Limit GET POST HEAD>\n    Require ip 127.0.0.1\n  </Limit>\n  <LimitExcept GET POST HEAD>\n    Require all denied\n  </LimitExcept>\n</Directory>\n'.strip('\n').format(securedrop_test_vars.securedrop_code)])
def test_apache_config_source_interface_access_control(host, apache_opt):
    if False:
        while True:
            i = 10
    '\n    Verifies the access control directives for the Source Interface.\n    '
    f = host.file('/etc/apache2/sites-available/source.conf')
    regex = f'^{re.escape(apache_opt)}$'
    assert re.search(regex, f.content_string, re.M)