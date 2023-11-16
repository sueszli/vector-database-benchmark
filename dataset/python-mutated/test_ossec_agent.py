import os
import re
import pytest
import testutils
sdvars = testutils.securedrop_test_vars
testinfra_hosts = [sdvars.app_hostname]

def test_hosts_files(host):
    if False:
        return 10
    'Ensure host files mapping are in place'
    f = host.file('/etc/hosts')
    mon_ip = os.environ.get('MON_IP', sdvars.mon_ip)
    mon_host = sdvars.monitor_hostname
    assert f.contains('^127.0.0.1\\s*localhost')
    assert f.contains(f'^{mon_ip}\\s*{mon_host}\\s*securedrop-monitor-server-alias$')

def test_ossec_service_start_style(host):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensure that the OSSEC services are managed by systemd.\n    '
    with host.sudo():
        c = host.check_output('systemctl status ossec')
        assert '/etc/systemd/system/ossec.service' in c

def test_hosts_duplicate(host):
    if False:
        for i in range(10):
            print('nop')
    'Regression test for duplicate entries'
    assert host.check_output('uniq --repeated /etc/hosts') == ''

def test_ossec_agent_installed(host):
    if False:
        print('Hello World!')
    'Check that ossec-agent package is present'
    assert host.package('securedrop-ossec-agent').is_installed

@pytest.mark.xfail()
def test_ossec_keyfile_present(host):
    if False:
        print('Hello World!')
    'ensure client keyfile for ossec-agent is present'
    pattern = '^1024 {} {} [0-9a-f]{{64}}$'.format(sdvars.app_hostname, os.environ.get('APP_IP', sdvars.app_ip))
    regex = re.compile(pattern)
    with host.sudo():
        f = host.file('/var/ossec/etc/client.keys')
        assert f.exists
        assert f.mode == 420
        assert f.user == 'root'
        assert f.group == 'ossec'
        assert f.content_string
        assert bool(re.search(regex, f.content))