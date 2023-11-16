import difflib
import os
import pytest
import testutils
from jinja2 import Template
securedrop_test_vars = testutils.securedrop_test_vars
testinfra_hosts = [securedrop_test_vars.monitor_hostname]

@pytest.mark.skip_in_prod()
def test_mon_iptables_rules(host):
    if False:
        while True:
            i = 10
    local = host.get_host('local://')
    kwargs = dict(app_ip=os.environ.get('APP_IP', securedrop_test_vars.app_ip), default_interface=host.check_output("ip r | head -n 1 | awk '{ print $5 }'"), tor_user_id=host.check_output('id -u debian-tor'), time_service_user=host.check_output('id -u systemd-timesync'), ssh_group_gid=host.check_output('getent group ssh | cut -d: -f3'), postfix_user_id=host.check_output('id -u postfix'), dns_server=securedrop_test_vars.dns_server)
    if local.interface('eth0').exists:
        kwargs['ssh_ip'] = local.interface('eth0').addresses[0]
    iptables = "iptables-save | sed 's/ \\[[0-9]*\\:[0-9]*\\]//g' | egrep -v '^#'"
    environment = os.environ.get('SECUREDROP_TESTINFRA_TARGET_HOST', 'staging')
    iptables_file = '{}/iptables-mon-{}.j2'.format(os.path.dirname(os.path.abspath(__file__)), environment)
    jinja_iptables = Template(open(iptables_file).read())
    iptables_expected = jinja_iptables.render(**kwargs)
    with host.sudo():
        iptables = host.check_output(iptables)
        for iptablesdiff in difflib.context_diff(iptables_expected.split('\n'), iptables.split('\n')):
            print(iptablesdiff)
        assert iptables_expected == iptables

@pytest.mark.skip_in_prod()
@pytest.mark.parametrize('ossec_service', [dict(host='0.0.0.0', proto='tcp', port=22, listening=True), dict(host='0.0.0.0', proto='udp', port=1514, listening=True), dict(host='0.0.0.0', proto='tcp', port=1515, listening=False)])
def test_listening_ports(host, ossec_service):
    if False:
        i = 10
        return i + 15
    '\n    Ensure the OSSEC-related services are listening on the\n    expected sockets. Services to check include ossec-remoted\n    and ossec-authd. Helper services such as postfix are checked\n    separately.\n\n    Note that the SSH check will fail if run against a prod host, due\n    to the SSH-over-Tor strategy. We can port the parametrized values\n    to config test YAML vars at that point.\n    '
    socket = '{proto}://{host}:{port}'.format(**ossec_service)
    with host.sudo():
        assert host.socket(socket).is_listening == ossec_service['listening']