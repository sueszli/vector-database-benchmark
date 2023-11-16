import difflib
import os
import pytest
import testutils
from jinja2 import Template
securedrop_test_vars = testutils.securedrop_test_vars
testinfra_hosts = [securedrop_test_vars.app_hostname]

@pytest.mark.skip_in_prod()
def test_app_iptables_rules(host):
    if False:
        return 10
    local = host.get_host('local://')
    kwargs = dict(mon_ip=os.environ.get('MON_IP', securedrop_test_vars.mon_ip), default_interface=host.check_output("ip r | head -n 1 | awk '{ print $5 }'"), tor_user_id=host.check_output('id -u debian-tor'), time_service_user=host.check_output('id -u systemd-timesync'), securedrop_user_id=host.check_output('id -u www-data'), ssh_group_gid=host.check_output('getent group ssh | cut -d: -f3'), dns_server=securedrop_test_vars.dns_server)
    if local.interface('eth0').exists:
        kwargs['ssh_ip'] = local.interface('eth0').addresses[0]
    iptables = "iptables-save | sed 's/ \\[[0-9]*\\:[0-9]*\\]//g' | egrep -v '^#'"
    environment = os.environ.get('SECUREDROP_TESTINFRA_TARGET_HOST', 'staging')
    iptables_file = '{}/iptables-app-{}.j2'.format(os.path.dirname(os.path.abspath(__file__)), environment)
    jinja_iptables = Template(open(iptables_file).read())
    iptables_expected = jinja_iptables.render(**kwargs)
    with host.sudo():
        iptables = host.check_output(iptables)
        for iptablesdiff in difflib.context_diff(iptables_expected.split('\n'), iptables.split('\n')):
            print(iptablesdiff)
        assert iptables_expected == iptables