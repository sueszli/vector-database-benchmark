import re
import pytest
import testutils
sdvars = testutils.securedrop_test_vars
testinfra_hosts = [sdvars.app_hostname]

@pytest.mark.skip_in_prod()
@pytest.mark.parametrize('tor_service', sdvars.tor_services)
def test_tor_service_directories(host, tor_service):
    if False:
        i = 10
        return i + 15
    '\n    Check mode and ownership on Tor service directories.\n    '
    with host.sudo():
        f = host.file('/var/lib/tor/services/{}'.format(tor_service['name']))
        assert f.is_directory
        assert f.mode == 448
        assert f.user == 'debian-tor'
        assert f.group == 'debian-tor'

@pytest.mark.skip_in_prod()
@pytest.mark.parametrize('tor_service', sdvars.tor_services)
def test_tor_service_hostnames(host, tor_service):
    if False:
        print('Hello World!')
    '\n    Check contents of Tor service hostname file. For v3 onion services,\n    the file should contain only hostname (.onion URL).\n    '
    ths_hostname_regex = '[a-z0-9]{16}\\.onion'
    ths_hostname_regex_v3 = '[a-z0-9]{56}\\.onion'
    with host.sudo():
        f = host.file('/var/lib/tor/services/{}/hostname'.format(tor_service['name']))
        assert f.is_file
        assert f.mode == 384
        assert f.user == 'debian-tor'
        assert f.group == 'debian-tor'
        assert re.search(ths_hostname_regex, f.content_string)
        if tor_service['authenticated'] and tor_service['version'] == 3:
            client_auth = host.file('/var/lib/tor/services/{}/authorized_clients/client.auth'.format(tor_service['name']))
            assert client_auth.is_file
        else:
            assert re.search(f'^{ths_hostname_regex_v3}$', f.content_string)

@pytest.mark.skip_in_prod()
@pytest.mark.parametrize('tor_service', sdvars.tor_services)
def test_tor_services_config(host, tor_service):
    if False:
        print('Hello World!')
    '\n    Ensure torrc file contains relevant lines for onion service declarations.\n    All onion services must include:\n\n      * HiddenServiceDir\n      * HiddenServicePort\n    '
    f = host.file('/etc/tor/torrc')
    dir_regex = 'HiddenServiceDir /var/lib/tor/services/{}'.format(tor_service['name'])
    remote_port = tor_service['ports'][0]
    try:
        local_port = tor_service['ports'][1]
    except IndexError:
        local_port = remote_port
    port_regex = f'HiddenServicePort {remote_port} 127.0.0.1:{local_port}'
    assert f.contains(f'^{dir_regex}$')
    assert f.contains(f'^{port_regex}$')
    service_regex = '\n'.join([dir_regex, port_regex])
    assert service_regex in f.content_string