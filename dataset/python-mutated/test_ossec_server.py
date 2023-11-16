import os
import pytest
import testutils
securedrop_test_vars = testutils.securedrop_test_vars
testinfra_hosts = [securedrop_test_vars.monitor_hostname]
python_version = securedrop_test_vars.python_version

def test_ossec_connectivity(host):
    if False:
        for i in range(10):
            print('nop')
    "\n    Ensure ossec-server machine has active connection to the ossec-agent.\n    The ossec service will report all available agents, and we can inspect\n    that list to make sure it's the host we expect.\n    "
    desired_output = '{}-{} is available.'.format(securedrop_test_vars.app_hostname, os.environ.get('APP_IP', securedrop_test_vars.app_ip))
    with host.sudo():
        c = host.check_output('/var/ossec/bin/list_agents -a')
        assert c == desired_output

def test_ossec_service_start_style(host):
    if False:
        while True:
            i = 10
    '\n    Ensure that the OSSEC services are managed by systemd.\n    '
    with host.sudo():
        c = host.check_output('systemctl status ossec')
        assert '/etc/systemd/system/ossec.service' in c

@pytest.mark.xfail()
@pytest.mark.parametrize('keyfile', ['/var/ossec/etc/sslmanager.key', '/var/ossec/etc/sslmanager.cert'])
def test_ossec_keyfiles(host, keyfile):
    if False:
        while True:
            i = 10
    "\n    Ensure that the OSSEC transport key pair exists. These keys are used\n    to protect the connection between the ossec-server and ossec-agent.\n\n    All this check does in confirm they're present, it doesn't perform any\n    matching checks to validate the configuration.\n    "
    with host.sudo():
        f = host.file(keyfile)
        assert f.is_file
        assert f.mode == 288
        assert f.user == 'root'
        assert f.group == 'ossec'

@pytest.mark.xfail()
def test_procmail_log(host):
    if False:
        while True:
            i = 10
    '\n    Ensure procmail log file exist with proper ownership.\n    Only the ossec user should have read/write permissions.\n    '
    with host.sudo():
        f = host.file('/var/log/procmail.log')
        assert f.is_file
        assert f.user == 'ossec'
        assert f.group == 'root'
        assert f.mode == 432

def test_ossec_authd(host):
    if False:
        print('Hello World!')
    'Ensure that authd is not running'
    with host.sudo():
        c = host.run('pgrep ossec-authd')
        assert c.stdout == ''
        assert c.rc != 0

def test_hosts_files(host):
    if False:
        while True:
            i = 10
    'Ensure host files mapping are in place'
    f = host.file('/etc/hosts')
    app_ip = os.environ.get('APP_IP', securedrop_test_vars.app_ip)
    app_host = securedrop_test_vars.app_hostname
    assert f.contains('^127.0.0.1.*localhost')
    assert f.contains(f'^{app_ip}\\s*{app_host}$')

def test_ossec_log_contains_no_malformed_events(host):
    if False:
        print('Hello World!')
    "\n    Ensure the OSSEC log reports no errors for incorrectly formatted\n    messages. These events indicate that the OSSEC server failed to decrypt\n    the event sent by the OSSEC agent, which implies a misconfiguration,\n    likely the IPv4 address or keypair differing from what's declared.\n\n    Documentation regarding this error message can be found at:\n    http://ossec-docs.readthedocs.io/en/latest/faq/unexpected.html#id4\n    "
    with host.sudo():
        f = host.file('/var/ossec/logs/ossec.log')
        assert not f.contains('ERROR: Incorrectly formated message from')

def test_regression_hosts(host):
    if False:
        print('Hello World!')
    'Regression test to check for duplicate entries.'
    assert host.check_output('uniq --repeated /etc/hosts') == ''