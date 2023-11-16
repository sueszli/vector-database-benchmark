import pytest
import testutils
sdvars = testutils.securedrop_test_vars
testinfra_hosts = [sdvars.app_hostname]

@pytest.mark.parametrize('pkg', ['apparmor', 'apparmor-utils'])
def test_apparmor_pkg(host, pkg):
    if False:
        for i in range(10):
            print('nop')
    'Apparmor package dependencies'
    assert host.package(pkg).is_installed

def test_apparmor_enabled(host):
    if False:
        return 10
    'Check that apparmor is enabled'
    with host.sudo():
        assert host.run('aa-status --enabled').rc == 0
apache2_capabilities = ['dac_override', 'kill', 'net_bind_service', 'sys_ptrace']

@pytest.mark.parametrize('cap', apache2_capabilities)
def test_apparmor_apache_capabilities(host, cap):
    if False:
        print('Hello World!')
    'check for exact list of expected app-armor capabilities for apache2'
    c = host.run("perl -nE '/^\\s+capability\\s+(\\w+),$/ && say $1' /etc/apparmor.d/usr.sbin.apache2")
    assert cap in c.stdout

def test_apparmor_apache_exact_capabilities(host):
    if False:
        i = 10
        return i + 15
    'ensure no extra capabilities are defined for apache2'
    c = host.check_output('grep -ic capability /etc/apparmor.d/usr.sbin.apache2')
    assert str(len(apache2_capabilities)) == c
tor_capabilities = ['setgid']

@pytest.mark.parametrize('cap', tor_capabilities)
def test_apparmor_tor_capabilities(host, cap):
    if False:
        while True:
            i = 10
    'check for exact list of expected app-armor capabilities for Tor'
    c = host.run("perl -nE '/^\\s+capability\\s+(\\w+),$/ && say $1' /etc/apparmor.d/usr.sbin.tor")
    assert cap in c.stdout

def test_apparmor_tor_exact_capabilities(host):
    if False:
        while True:
            i = 10
    'ensure no extra capabilities are defined for Tor'
    c = host.check_output('grep -ic capability /etc/apparmor.d/usr.sbin.tor')
    assert str(len(tor_capabilities)) == c

@pytest.mark.parametrize('profile', sdvars.apparmor_enforce)
def test_apparmor_ensure_not_disabled(host, profile):
    if False:
        print('Hello World!')
    '\n    Explicitly check that enforced profiles are NOT in /etc/apparmor.d/disable\n    Polling aa-status only checks the last config that was loaded,\n    this ensures it wont be disabled on reboot.\n    '
    f = host.file(f'/etc/apparmor.d/disabled/usr.sbin.{profile}')
    with host.sudo():
        assert not f.exists

@pytest.mark.parametrize('complain_pkg', sdvars.apparmor_complain)
def test_app_apparmor_complain(host, complain_pkg):
    if False:
        for i in range(10):
            print('nop')
    'Ensure app-armor profiles are in complain mode for staging'
    with host.sudo():
        awk = "awk '/[0-9]+ profiles.*complain./{flag=1;next}/^[0-9]+.*/{flag=0}flag'"
        c = host.check_output(f'aa-status | {awk}')
        assert complain_pkg in c

def test_app_apparmor_complain_count(host):
    if False:
        i = 10
        return i + 15
    'Ensure right number of app-armor profiles are in complain mode'
    with host.sudo():
        c = host.check_output('aa-status --complaining')
        assert c == str(len(sdvars.apparmor_complain))

@pytest.mark.parametrize('aa_enforced', sdvars.apparmor_enforce_actual)
def test_apparmor_enforced(host, aa_enforced):
    if False:
        print('Hello World!')
    awk = "awk '/[0-9]+ profiles.*enforce./{flag=1;next}/^[0-9]+.*/{flag=0}flag'"
    with host.sudo():
        c = host.check_output(f'aa-status | {awk}')
        assert aa_enforced in c

def test_apparmor_total_profiles(host):
    if False:
        print('Hello World!')
    'Ensure number of total profiles is sum of enforced and\n    complaining profiles'
    with host.sudo():
        total_expected = len(sdvars.apparmor_enforce) + len(sdvars.apparmor_complain)
        assert int(host.check_output('aa-status --profiled')) >= total_expected

def test_aastatus_unconfined(host):
    if False:
        print('Hello World!')
    'Ensure that there are no processes that are unconfined but have\n    a profile'
    expected_unconfined = 0
    unconfined_chk = str('{} processes are unconfined but have a profile defined'.format(expected_unconfined))
    with host.sudo():
        aa_status_output = host.check_output('aa-status')
        assert unconfined_chk in aa_status_output

def test_aa_no_denies_in_syslog(host):
    if False:
        for i in range(10):
            print('nop')
    'Ensure that there are no apparmor denials in syslog'
    with host.sudo():
        f = host.file('/var/log/syslog')
        lines = f.content_string.splitlines()
    found = []
    for line in lines:
        if 'apparmor="DENIED"' in line:
            found.append(line)
    assert found == []