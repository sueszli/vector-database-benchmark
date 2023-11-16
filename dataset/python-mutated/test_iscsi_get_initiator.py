from __future__ import annotations
from ansible.module_utils.facts.network import iscsi
from unittest.mock import Mock
LSATTR_OUTPUT = '\ndisc_filename  /etc/iscsi/targets            Configuration file                            False\ndisc_policy    file                          Discovery Policy                              True\ninitiator_name iqn.localhost.hostid.7f000002 iSCSI Initiator Name                          True\nisns_srvnames  auto                          iSNS Servers IP Addresses                     True\nisns_srvports                                iSNS Servers Port Numbers                     True\nmax_targets    16                            Maximum Targets Allowed                       True\nnum_cmd_elems  200                           Maximum number of commands to queue to driver True\n'
ISCSIUTIL_OUTPUT = '\nInitiator Name             : iqn.2001-04.com.hp.stor:svcio\nInitiator Alias            :\nAuthentication Method      : None\nCHAP Method                : CHAP_UNI\nInitiator CHAP Name        :\nCHAP Secret                :\nNAS Hostname               :\nNAS Secret                 :\nRadius Server Hostname     :\nHeader Digest              : None,CRC32C (default)\nData Digest                : None,CRC32C (default)\nSLP Scope list for iSLPD   :\n'

def test_get_iscsi_info(mocker):
    if False:
        while True:
            i = 10
    module = Mock()
    inst = iscsi.IscsiInitiatorNetworkCollector()
    mocker.patch('sys.platform', 'aix6')
    mocker.patch('ansible.module_utils.facts.network.iscsi.get_bin_path', return_value='/usr/sbin/lsattr')
    mocker.patch.object(module, 'run_command', return_value=(0, LSATTR_OUTPUT, ''))
    aix_iscsi_expected = {'iscsi_iqn': 'iqn.localhost.hostid.7f000002'}
    assert aix_iscsi_expected == inst.collect(module=module)
    mocker.patch('sys.platform', 'hp-ux')
    mocker.patch('ansible.module_utils.facts.network.iscsi.get_bin_path', return_value='/opt/iscsi/bin/iscsiutil')
    mocker.patch.object(module, 'run_command', return_value=(0, ISCSIUTIL_OUTPUT, ''))
    hpux_iscsi_expected = {'iscsi_iqn': ' iqn.2001-04.com.hp.stor:svcio'}
    assert hpux_iscsi_expected == inst.collect(module=module)