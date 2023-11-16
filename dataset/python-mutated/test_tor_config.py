import re
import pytest
import testutils
sdvars = testutils.securedrop_test_vars
testinfra_hosts = [sdvars.app_hostname]

@pytest.mark.parametrize('package', ['tor'])
def test_tor_packages(host, package):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensure Tor packages are installed. Does not include the Tor keyring\n    package, since we want only the SecureDrop Release Signing Key\n    to be used even for Tor packages.\n    '
    assert host.package(package).is_installed

def test_tor_service_running(host):
    if False:
        i = 10
        return i + 15
    '\n    Ensure Tor is running and enabled. Tor is required for SSH access,\n    so it must be enabled to start on boot.\n    '
    s = host.service('tor')
    assert s.is_running
    assert s.is_enabled

@pytest.mark.parametrize('torrc_option', ['SocksPort 0', 'SafeLogging 1', 'RunAsDaemon 1'])
def test_tor_torrc_options(host, torrc_option):
    if False:
        while True:
            i = 10
    '\n    Check for required options in the system Tor config file.\n    These options should be present regardless of machine role,\n    meaning both Application and Monitor server will have them.\n\n    Separate tests will check for specific onion services.\n    '
    f = host.file('/etc/tor/torrc')
    assert f.is_file
    assert f.user == 'debian-tor'
    assert f.mode == 420
    assert f.contains(f'^{torrc_option}$')

def test_tor_torrc_sandbox(host):
    if False:
        i = 10
        return i + 15
    '\n    Check that the `Sandbox 1` declaration is not present in the torrc.\n    The torrc manpage states this option is experimental, and although we\n    use it already on Tails workstations, further testing is required\n    before we push it out to servers. See issues #944 and #1969.\n    '
    f = host.file('/etc/tor/torrc')
    assert not f.contains('^.*Sandbox.*$')

@pytest.mark.skip_in_prod()
def test_tor_v2_onion_url_file_absent(host):
    if False:
        i = 10
        return i + 15
    v2_url_filepath = '/var/lib/securedrop/source_v2_url'
    with host.sudo():
        f = host.file(v2_url_filepath)
        assert not f.exists

@pytest.mark.skip_in_prod()
def test_tor_v3_onion_url_readable_by_app(host):
    if False:
        while True:
            i = 10
    v3_url_filepath = '/var/lib/securedrop/source_v3_url'
    with host.sudo():
        f = host.file(v3_url_filepath)
        assert f.is_file
        assert f.user == 'www-data'
        assert f.mode == 420
        assert re.search('^[a-z0-9]{56}\\.onion$', f.content_string)