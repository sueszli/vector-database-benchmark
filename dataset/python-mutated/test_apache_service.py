import pytest
import testutils
securedrop_test_vars = testutils.securedrop_test_vars
testinfra_hosts = [securedrop_test_vars.app_hostname]

@pytest.mark.parametrize('apache_site', ['source', 'journalist'])
def test_apache_enabled_sites(host, apache_site):
    if False:
        print('Hello World!')
    '\n    Ensure the Source and Journalist interfaces are enabled.\n    '
    with host.sudo():
        c = host.run(f'/usr/sbin/a2query -s {apache_site}')
        assert f'{apache_site} (enabled' in c.stdout
        assert c.rc == 0

@pytest.mark.parametrize('apache_site', ['000-default'])
def test_apache_disabled_sites(host, apache_site):
    if False:
        while True:
            i = 10
    '\n    Ensure the default HTML document root is disabled.\n    '
    c = host.run(f'a2query -s {apache_site}')
    assert f'No site matches {apache_site} (disabled' in c.stderr
    assert c.rc == 32

def test_apache_service(host):
    if False:
        print('Hello World!')
    '\n    Ensure Apache service is running.\n    '
    with host.sudo():
        s = host.service('apache2')
        assert s.is_running
        assert s.is_enabled

def test_apache_user(host):
    if False:
        while True:
            i = 10
    '\n    Ensure user account for running application code is configured correctly.\n    '
    u = host.user('www-data')
    assert u.exists
    assert u.home == '/var/www'
    assert u.shell == '/usr/sbin/nologin'

@pytest.mark.parametrize('port', ['80', '8080'])
def test_apache_listening(host, port):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensure Apache is listening on proper ports and interfaces.\n    In staging, expect the service to be bound to 0.0.0.0,\n    but in prod, it should be restricted to 127.0.0.1.\n    '
    with host.sudo():
        s = host.socket(f'tcp://{securedrop_test_vars.apache_listening_address}:{port}')
        assert s.is_listening