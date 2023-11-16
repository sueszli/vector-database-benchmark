from __future__ import annotations
import unittest
from unittest.mock import patch
from ansible.module_utils import basic
from ansible.modules.service_facts import AIXScanService
LSSRC_OUTPUT = '\nSubsystem         Group            PID          Status\n sendmail         mail             5243302      active\n syslogd          ras              5636528      active\n portmap          portmap          5177768      active\n snmpd            tcpip            5308844      active\n hostmibd         tcpip            5374380      active\n snmpmibd         tcpip            5439918      active\n aixmibd          tcpip            5505456      active\n nimsh            nimclient        5571004      active\n aso                               6029758      active\n biod             nfs              6357464      active\n nfsd             nfs              5701906      active\n rpc.mountd       nfs              6488534      active\n rpc.statd        nfs              7209216      active\n rpc.lockd        nfs              7274988      active\n qdaemon          spooler          6816222      active\n writesrv         spooler          6685150      active\n clcomd           caa              7471600      active\n sshd             ssh              7602674      active\n pfcdaemon                         7012860      active\n ctrmc            rsct             6947312      active\n IBM.HostRM       rsct_rm          14418376     active\n IBM.ConfigRM     rsct_rm          6160674      active\n IBM.DRM          rsct_rm          14680550     active\n IBM.MgmtDomainRM rsct_rm          14090676     active\n IBM.ServiceRM    rsct_rm          13828542     active\n cthats           cthats           13959668     active\n cthags           cthags           14025054     active\n IBM.StorageRM    rsct_rm          12255706     active\n inetd            tcpip            12517828     active\n lpd              spooler                       inoperative\n keyserv          keyserv                       inoperative\n ypbind           yp                            inoperative\n gsclvmd                                        inoperative\n cdromd                                         inoperative\n ndpd-host        tcpip                         inoperative\n ndpd-router      tcpip                         inoperative\n netcd            netcd                         inoperative\n tftpd            tcpip                         inoperative\n routed           tcpip                         inoperative\n mrouted          tcpip                         inoperative\n rsvpd            qos                           inoperative\n policyd          qos                           inoperative\n timed            tcpip                         inoperative\n iptrace          tcpip                         inoperative\n dpid2            tcpip                         inoperative\n rwhod            tcpip                         inoperative\n pxed             tcpip                         inoperative\n binld            tcpip                         inoperative\n xntpd            tcpip                         inoperative\n gated            tcpip                         inoperative\n dhcpcd           tcpip                         inoperative\n dhcpcd6          tcpip                         inoperative\n dhcpsd           tcpip                         inoperative\n dhcpsdv6         tcpip                         inoperative\n dhcprd           tcpip                         inoperative\n dfpd             tcpip                         inoperative\n named            tcpip                         inoperative\n automountd       autofs                        inoperative\n nfsrgyd          nfs                           inoperative\n gssd             nfs                           inoperative\n cpsd             ike                           inoperative\n tmd              ike                           inoperative\n isakmpd                                        inoperative\n ikev2d                                         inoperative\n iked             ike                           inoperative\n clconfd          caa                           inoperative\n ksys_vmmon                                     inoperative\n nimhttp                                        inoperative\n IBM.SRVPROXY     ibmsrv                        inoperative\n ctcas            rsct                          inoperative\n IBM.ERRM         rsct_rm                       inoperative\n IBM.AuditRM      rsct_rm                       inoperative\n isnstgtd         isnstgtd                      inoperative\n IBM.LPRM         rsct_rm                       inoperative\n cthagsglsm       cthags                        inoperative\n'

class TestAIXScanService(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.mock1 = patch.object(basic.AnsibleModule, 'get_bin_path', return_value='/usr/sbin/lssrc')
        self.mock1.start()
        self.addCleanup(self.mock1.stop)
        self.mock2 = patch.object(basic.AnsibleModule, 'run_command', return_value=(0, LSSRC_OUTPUT, ''))
        self.mock2.start()
        self.addCleanup(self.mock2.stop)
        self.mock3 = patch('platform.system', return_value='AIX')
        self.mock3.start()
        self.addCleanup(self.mock3.stop)

    def test_gather_services(self):
        if False:
            print('Hello World!')
        svcmod = AIXScanService(basic.AnsibleModule)
        result = svcmod.gather_services()
        self.assertIsInstance(result, dict)
        self.assertIn('IBM.HostRM', result)
        self.assertEqual(result['IBM.HostRM'], {'name': 'IBM.HostRM', 'source': 'src', 'state': 'running'})
        self.assertIn('IBM.AuditRM', result)
        self.assertEqual(result['IBM.AuditRM'], {'name': 'IBM.AuditRM', 'source': 'src', 'state': 'stopped'})