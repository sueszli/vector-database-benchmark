"""Package: mininet
   Test creation and pings for topologies with link and/or CPU options."""
import unittest
import sys
from functools import partial
from mininet.net import Mininet
from mininet.node import OVSSwitch, UserSwitch, IVSSwitch
from mininet.node import CPULimitedHost
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.log import setLogLevel
from mininet.util import quietRun
from mininet.clean import cleanup
N = 2

class SingleSwitchOptionsTopo(Topo):
    """Single switch connected to n hosts."""

    def __init__(self, n=2, hopts=None, lopts=None):
        if False:
            while True:
                i = 10
        if not hopts:
            hopts = {}
        if not lopts:
            lopts = {}
        Topo.__init__(self, hopts=hopts, lopts=lopts)
        switch = self.addSwitch('s1')
        for h in range(n):
            host = self.addHost('h%s' % (h + 1))
            self.addLink(host, switch)

class testOptionsTopoCommon(object):
    """Verify ability to create networks with host and link options
       (common code)."""
    switchClass = None

    @staticmethod
    def tearDown():
        if False:
            while True:
                i = 10
        'Clean up if necessary'
        if sys.exc_info() != (None, None, None):
            cleanup()

    def runOptionsTopoTest(self, n, msg, hopts=None, lopts=None):
        if False:
            return 10
        'Generic topology-with-options test runner.'
        mn = Mininet(topo=SingleSwitchOptionsTopo(n=n, hopts=hopts, lopts=lopts), host=CPULimitedHost, link=TCLink, switch=self.switchClass, waitConnected=True)
        dropped = mn.run(mn.ping)
        hoptsStr = ', '.join(('%s: %s' % (opt, value) for (opt, value) in hopts.items()))
        loptsStr = ', '.join(('%s: %s' % (opt, value) for (opt, value) in lopts.items()))
        msg += '%s%% of pings were dropped during mininet.ping().\nTopo = SingleSwitchTopo, %s hosts\nhopts = %s\nlopts = %s\nhost = CPULimitedHost\nlink = TCLink\nSwitch = %s\n' % (dropped, n, hoptsStr, loptsStr, self.switchClass)
        self.assertEqual(dropped, 0, msg=msg)

    def assertWithinTolerance(self, measured, expected, tolerance_frac, msg):
        if False:
            while True:
                i = 10
        'Check that a given value is within a tolerance of expected\n        tolerance_frac: less-than-1.0 value; 0.8 would yield 20% tolerance.\n        '
        upperBound = float(expected) + (1 - tolerance_frac) * float(expected)
        lowerBound = float(expected) * tolerance_frac
        info = 'measured value is out of bounds\nexpected value: %s\nmeasured value: %s\nfailure tolerance: %s\nupper bound: %s\nlower bound: %s\n' % (expected, measured, tolerance_frac, upperBound, lowerBound)
        msg += info
        self.assertGreaterEqual(float(measured), lowerBound, msg=msg)
        self.assertLessEqual(float(measured), upperBound, msg=msg)

    def testCPULimits(self):
        if False:
            for i in range(10):
                print('nop')
        'Verify topology creation with CPU limits set for both schedulers.'
        CPU_FRACTION = 0.1
        CPU_TOLERANCE = 0.8
        hopts = {'cpu': CPU_FRACTION}
        mn = Mininet(SingleSwitchOptionsTopo(n=N, hopts=hopts), host=CPULimitedHost, switch=self.switchClass, waitConnected=True)
        mn.start()
        results = mn.runCpuLimitTest(cpu=CPU_FRACTION)
        mn.stop()
        hostUsage = '\n'.join(('h%s: %s' % (n + 1, results[(n - 1) * 5:n * 5 - 1]) for n in range(N)))
        hoptsStr = ', '.join(('%s: %s' % (opt, value) for (opt, value) in hopts.items()))
        msg = '\nTesting cpu limited to %d%% of cpu per host\ncpu usage percent per host:\n%s\nTopo = SingleSwitchTopo, %s hosts\nhopts = %s\nhost = CPULimitedHost\nSwitch = %s\n' % (CPU_FRACTION * 100, hostUsage, N, hoptsStr, self.switchClass)
        for pct in results:
            self.assertWithinTolerance(pct / 100, CPU_FRACTION, CPU_TOLERANCE, msg)

    def testLinkBandwidth(self):
        if False:
            return 10
        'Verify that link bandwidths are accurate within a bound.'
        if self.switchClass is UserSwitch:
            self.skipTest('UserSwitch has very poor performance - skipping for now')
        BW = 5
        BW_TOLERANCE = 0.8
        lopts = {'bw': BW, 'use_htb': True}
        mn = Mininet(SingleSwitchOptionsTopo(n=N, lopts=lopts), link=TCLink, switch=self.switchClass, waitConnected=True)
        bw_strs = mn.run(mn.iperf, fmt='m')
        loptsStr = ', '.join(('%s: %s' % (opt, value) for (opt, value) in lopts.items()))
        msg = '\nTesting link bandwidth limited to %d Mbps per link\niperf results[ client, server ]: %s\nTopo = SingleSwitchTopo, %s hosts\nLink = TCLink\nlopts = %s\nhost = default\nswitch = %s\n' % (BW, bw_strs, N, loptsStr, self.switchClass)
        (serverRate, _clientRate) = bw_strs
        bw = float(serverRate.split(' ')[0])
        self.assertWithinTolerance(bw, BW, BW_TOLERANCE, msg)

    def testLinkDelay(self):
        if False:
            while True:
                i = 10
        'Verify that link delays are accurate within a bound.'
        DELAY_MS = 15
        DELAY_TOLERANCE = 0.8
        REPS = 3
        lopts = {'delay': '%sms' % DELAY_MS, 'use_htb': True}
        mn = Mininet(SingleSwitchOptionsTopo(n=N, lopts=lopts), link=TCLink, switch=self.switchClass, autoStaticArp=True, waitConnected=True)
        mn.start()
        for _ in range(REPS):
            ping_delays = mn.pingFull()
        mn.stop()
        test_outputs = ping_delays[0]
        (node, dest, ping_outputs) = test_outputs
        (sent, received, rttmin, rttavg, rttmax, rttdev) = ping_outputs
        pingFailMsg = 'sent %s pings, only received %s' % (sent, received)
        self.assertEqual(sent, received, msg=pingFailMsg)
        loptsStr = ', '.join(('%s: %s' % (opt, value) for (opt, value) in lopts.items()))
        msg = '\nTesting Link Delay of %s ms\nping results across 4 links:\n(Sent, Received, rttmin, rttavg, rttmax, rttdev)\n%s\nTopo = SingleSwitchTopo, %s hosts\nLink = TCLink\nlopts = %s\nhost = defaultswitch = %s\n' % (DELAY_MS, ping_outputs, N, loptsStr, self.switchClass)
        for rttval in [rttmin, rttavg, rttmax]:
            self.assertWithinTolerance(rttval, DELAY_MS * 4.0, DELAY_TOLERANCE, msg)

    def testLinkLoss(self):
        if False:
            for i in range(10):
                print('nop')
        'Verify that we see packet drops with a high configured loss rate.'
        LOSS_PERCENT = 99
        REPS = 1
        lopts = {'loss': LOSS_PERCENT, 'use_htb': True}
        mn = Mininet(topo=SingleSwitchOptionsTopo(n=N, lopts=lopts), host=CPULimitedHost, link=TCLink, switch=self.switchClass, waitConnected=True)
        dropped_total = 0
        mn.start()
        for _ in range(REPS):
            dropped_total += mn.ping(timeout='1')
        mn.stop()
        loptsStr = ', '.join(('%s: %s' % (opt, value) for (opt, value) in lopts.items()))
        msg = '\nTesting packet loss with %d%% loss rate\nnumber of dropped pings during mininet.ping(): %s\nexpected number of dropped packets: 1\nTopo = SingleSwitchTopo, %s hosts\nLink = TCLink\nlopts = %s\nhost = default\nswitch = %s\n' % (LOSS_PERCENT, dropped_total, N, loptsStr, self.switchClass)
        self.assertGreater(dropped_total, 0, msg)

    def testMostOptions(self):
        if False:
            i = 10
            return i + 15
        'Verify topology creation with most link options and CPU limits.'
        lopts = {'bw': 10, 'delay': '5ms', 'use_htb': True}
        hopts = {'cpu': 0.5 / N}
        msg = '\nTesting many cpu and link options\n'
        self.runOptionsTopoTest(N, msg, hopts=hopts, lopts=lopts)

class testOptionsTopoOVSKernel(testOptionsTopoCommon, unittest.TestCase):
    """Verify ability to create networks with host and link options
       (OVS kernel switch)."""
    longMessage = True
    switchClass = OVSSwitch

@unittest.skip('Skipping OVS user switch test for now')
class testOptionsTopoOVSUser(testOptionsTopoCommon, unittest.TestCase):
    """Verify ability to create networks with host and link options
       (OVS user switch)."""
    longMessage = True
    switchClass = partial(OVSSwitch, datapath='user')

@unittest.skipUnless(quietRun('which ivs-ctl'), 'IVS is not installed')
class testOptionsTopoIVS(testOptionsTopoCommon, unittest.TestCase):
    """Verify ability to create networks with host and link options (IVS)."""
    longMessage = True
    switchClass = IVSSwitch

@unittest.skipUnless(quietRun('which ofprotocol'), 'Reference user switch is not installed')
class testOptionsTopoUserspace(testOptionsTopoCommon, unittest.TestCase):
    """Verify ability to create networks with host and link options
     (UserSwitch)."""
    longMessage = True
    switchClass = UserSwitch
if __name__ == '__main__':
    setLogLevel('warning')
    unittest.main()