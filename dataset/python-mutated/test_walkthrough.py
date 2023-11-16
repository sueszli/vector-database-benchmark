"""
Tests for the Mininet Walkthrough

TODO: missing xterm test
"""
import os
import re
import unittest
from sys import stdout
from mininet.util import quietRun, pexpect, StrictVersion
from mininet.clean import cleanup

def tsharkVersion():
    if False:
        i = 10
        return i + 15
    'Return tshark version'
    versionStr = quietRun('tshark -v')
    versionMatch = re.findall('TShark[^\\d]*(\\d+.\\d+.\\d+)', versionStr)
    return versionMatch[0]

class testWalkthrough(unittest.TestCase):
    """Test Mininet walkthrough"""
    prompt = 'mininet>'

    @staticmethod
    def setup():
        if False:
            i = 10
            return i + 15
        'Be paranoid and run cleanup() before each test'
        cleanup()

    def testHelp(self):
        if False:
            for i in range(10):
                print('nop')
        'Check the usage message'
        p = pexpect.spawn('mn -h')
        index = p.expect(['Usage: mn', pexpect.EOF])
        self.assertEqual(index, 0)

    def testWireshark(self):
        if False:
            print('Hello World!')
        'Use tshark to test the of dissector'
        assert self
        if StrictVersion(tsharkVersion()) < StrictVersion('1.12.0'):
            tshark = pexpect.spawn('tshark -i lo -R of')
        else:
            tshark = pexpect.spawn('tshark -i lo -Y openflow_v1')
        tshark.expect(['Capturing on lo', "Capturing on 'Loopback"])
        mn = pexpect.spawn('mn --test pingall')
        mn.expect('0% dropped')
        tshark.expect(['74 Hello', '74 of_hello', '74 Type: OFPT_HELLO'])
        tshark.sendintr()
        mn.expect(pexpect.EOF)
        tshark.expect('aptured')
        tshark.expect(pexpect.EOF)

    def testBasic(self):
        if False:
            print('Hello World!')
        'Test basic CLI commands (help, nodes, net, dump)'
        p = pexpect.spawn('mn -w')
        p.expect(self.prompt)
        p.sendline('help')
        index = p.expect(['commands', self.prompt])
        self.assertEqual(index, 0, 'No output for "help" command')
        p.sendline('nodes')
        p.expect('([chs]\\d ?){4}')
        nodes = p.match.group(0).split()
        self.assertEqual(len(nodes), 4, 'No nodes in "nodes" command')
        p.expect(self.prompt)
        p.sendline('net')
        expected = list(nodes)
        while len(expected) > 0:
            index = p.expect(expected)
            node = p.match.group(0)
            expected.remove(node)
            p.expect('\n')
        self.assertEqual(len(expected), 0, '"nodes" and "net" differ')
        p.expect(self.prompt)
        p.sendline('dump')
        expected = ['<\\w+ (%s)' % n for n in nodes]
        actual = []
        for _ in nodes:
            index = p.expect(expected)
            node = p.match.group(1)
            actual.append(node)
            p.expect('\n')
        self.assertEqual(actual.sort(), nodes.sort(), '"nodes" and "dump" differ')
        p.expect(self.prompt)
        p.sendline('exit')
        p.wait()

    def testHostCommands(self):
        if False:
            return 10
        'Test ifconfig and ps on h1 and s1'
        p = pexpect.spawn('mn -w')
        p.expect(self.prompt)
        interfaces = ['h1-eth0[:\\s]', 's1-eth1[:\\s]', '[^-](eth|en)\\w*\\d[:\\s]', 'lo[:\\s]', self.prompt]
        p.sendline('h1 ifconfig -a')
        ifcount = 0
        while True:
            index = p.expect(interfaces)
            if index in (0, 3):
                ifcount += 1
            elif index == 1:
                self.fail('s1 interface displayed in "h1 ifconfig"')
            elif index == 2:
                self.fail('eth0 displayed in "h1 ifconfig"')
            else:
                break
        self.assertEqual(ifcount, 2, 'Missing interfaces on h1')
        p.sendline('s1 ifconfig -a')
        ifcount = 0
        while True:
            index = p.expect(interfaces)
            if index == 0:
                self.fail('h1 interface displayed in "s1 ifconfig"')
            elif index in (1, 2, 3):
                ifcount += 1
            else:
                break
        self.assertTrue(ifcount >= 3, 'Missing interfaces on s1')
        p.sendline("h1 ps -a | egrep -v 'ps|grep'")
        p.expect(self.prompt)
        h1Output = p.before
        p.sendline("s1 ps -a | egrep -v 'ps|grep'")
        p.expect(self.prompt)
        s1Output = p.before
        h1Output = h1Output.split('\n')[1:]
        s1Output = s1Output.split('\n')[1:]
        diffs = set(h1Output).difference(set(s1Output))
        self.assertTrue(len(diffs) <= 2, 'h1 and s1 "ps" output differ too much: %s' % diffs)
        p.sendline('exit')
        p.wait()

    def testConnectivity(self):
        if False:
            return 10
        'Test ping and pingall'
        p = pexpect.spawn('mn -w')
        p.expect(self.prompt)
        p.sendline('h1 ping -c 1 h2')
        p.expect('1 packets transmitted, 1 received')
        p.expect(self.prompt)
        p.sendline('pingall')
        p.expect('0% dropped')
        p.expect(self.prompt)
        p.sendline('exit')
        p.wait()

    def testSimpleHTTP(self):
        if False:
            return 10
        'Start an HTTP server on h1 and wget from h2'
        if 'Python 2' in quietRun('python --version'):
            httpserver = 'SimpleHTTPServer'
        else:
            httpserver = 'http.server'
        p = pexpect.spawn('mn -w', logfile=stdout)
        p.expect(self.prompt)
        p.sendline('h1 python -m %s 80 >& /dev/null &' % httpserver)
        p.expect(self.prompt)
        p.sendline('px from mininet.util import waitListening;waitListening(h1, port=80, timeout=30)')
        p.expect(self.prompt)
        p.sendline(' h2 wget -O - h1')
        p.expect('200 OK')
        p.expect(self.prompt)
        p.sendline('h1 kill %python')
        p.expect(self.prompt)
        p.sendline('exit')
        p.wait()

    def testRegressionRun(self):
        if False:
            while True:
                i = 10
        'Test pingpair (0% drop) and iperf (bw > 0) regression tests'
        p = pexpect.spawn('mn --test pingpair')
        p.expect('0% dropped')
        p.expect(pexpect.EOF)
        p = pexpect.spawn('mn --test iperf')
        p.expect("Results: \\['([\\d\\.]+) .bits/sec',")
        bw = float(p.match.group(1))
        self.assertTrue(bw > 0)
        p.expect(pexpect.EOF)

    def testTopoChange(self):
        if False:
            return 10
        'Test pingall on single,3 and linear,4 topos'
        p = pexpect.spawn('mn --test pingall --topo single,3')
        p.expect('(\\d+)/(\\d+) received')
        received = int(p.match.group(1))
        sent = int(p.match.group(2))
        self.assertEqual(sent, 6, 'Wrong number of pings sent in single,3')
        self.assertEqual(sent, received, 'Dropped packets in single,3')
        p.expect(pexpect.EOF)
        p = pexpect.spawn('mn --test pingall --topo linear,4')
        p.expect('(\\d+)/(\\d+) received')
        received = int(p.match.group(1))
        sent = int(p.match.group(2))
        self.assertEqual(sent, 12, 'Wrong number of pings sent in linear,4')
        self.assertEqual(sent, received, 'Dropped packets in linear,4')
        p.expect(pexpect.EOF)

    def testLinkChange(self):
        if False:
            while True:
                i = 10
        'Test TCLink bw and delay'
        p = pexpect.spawn('mn -w --link tc,bw=10,delay=10ms')
        p.expect(self.prompt)
        p.sendline('h1 route && ping -c1 h2')
        p.expect(self.prompt)
        p.sendline('iperf')
        p.expect("Results: \\['([\\d\\.]+) Mbits/sec',")
        bw = float(p.match.group(1))
        self.assertTrue(bw < 10.1, 'Bandwidth %.2f >= 10.1 Mb/s' % bw)
        self.assertTrue(bw > 9.0, 'Bandwidth %.2f <= 9 Mb/s' % bw)
        p.expect(self.prompt)
        p.sendline('h1 ping -c 4 h2')
        p.expect('rtt min/avg/max/mdev = ([\\d\\.]+)/([\\d\\.]+)/([\\d\\.]+)/([\\d\\.]+) ms')
        delay = float(p.match.group(2))
        self.assertTrue(delay >= 40, 'Delay < 40ms')
        self.assertTrue(delay <= 50, 'Delay > 50ms')
        p.expect(self.prompt)
        p.sendline('exit')
        p.wait()

    def testVerbosity(self):
        if False:
            for i in range(10):
                print('nop')
        'Test debug and output verbosity'
        p = pexpect.spawn('mn -v output')
        p.expect(self.prompt)
        self.assertEqual(len(p.before), 0, 'Too much output for "output"')
        p.sendline('exit')
        p.wait()
        p = pexpect.spawn('mn -v debug --test none')
        p.expect(pexpect.EOF)
        lines = p.before.split('\n')
        self.assertTrue(len(lines) > 70, 'Debug output is too short')

    def testCustomTopo(self):
        if False:
            while True:
                i = 10
        'Start Mininet using a custom topo, then run pingall'
        assert self
        custom = os.path.dirname(os.path.realpath(__file__))
        custom = os.path.join(custom, '../../custom/topo-2sw-2host.py')
        custom = os.path.normpath(custom)
        p = pexpect.spawn('mn --custom %s --topo mytopo --test pingall' % custom)
        p.expect('0% dropped')
        p.expect(pexpect.EOF)

    def testStaticMAC(self):
        if False:
            i = 10
            return i + 15
        'Verify that MACs are set to easy to read numbers'
        p = pexpect.spawn('mn --mac')
        p.expect(self.prompt)
        for i in range(1, 3):
            p.sendline('h%d ifconfig' % i)
            p.expect('\\s00:00:00:00:00:0%d\\s' % i)
            p.expect(self.prompt)
        p.sendline('exit')
        p.expect(pexpect.EOF)

    def testSwitches(self):
        if False:
            while True:
                i = 10
        'Run iperf test using user and ovsk switches'
        switches = ['user', 'ovsk']
        for sw in switches:
            p = pexpect.spawn('mn --switch %s --test iperf' % sw)
            p.expect("Results: \\['([\\d\\.]+) .bits/sec',")
            bw = float(p.match.group(1))
            self.assertTrue(bw > 0)
            p.expect(pexpect.EOF)

    def testBenchmark(self):
        if False:
            return 10
        'Run benchmark and verify that it takes less than 2 seconds'
        p = pexpect.spawn('mn --test none')
        p.expect('completed in ([\\d\\.]+) seconds')
        time = float(p.match.group(1))
        self.assertTrue(time < 2, 'Benchmark takes more than 2 seconds')

    def testOwnNamespace(self):
        if False:
            while True:
                i = 10
        'Test running user switch in its own namespace'
        p = pexpect.spawn('mn --innamespace --switch user')
        p.expect(self.prompt)
        interfaces = ['h1-eth0[:\\s]', 's1-eth1[:\\s]', '[^-](eth|en)\\w*\\d[:\\s]', 'lo[:\\s]', self.prompt]
        p.sendline('s1 ifconfig -a')
        ifcount = 0
        while True:
            index = p.expect(interfaces)
            if index in (1, 3):
                ifcount += 1
            elif index == 0:
                self.fail('h1 interface displayed in "s1 ifconfig"')
            elif index == 2:
                self.fail('eth0 displayed in "s1 ifconfig"')
            else:
                break
        self.assertEqual(ifcount, 2, 'Missing interfaces on s1')
        p.sendline('pingall')
        p.expect('(\\d+)% dropped')
        dropped = int(p.match.group(1))
        self.assertEqual(dropped, 0, 'pingall failed')
        p.expect(self.prompt)
        p.sendline('exit')
        p.wait()

    def testPythonInterpreter(self):
        if False:
            return 10
        'Test py and px by checking IP for h1 and adding h3'
        p = pexpect.spawn('mn -w')
        p.expect(self.prompt)
        p.sendline('py h1.IP()')
        p.expect('10.0.0.1')
        p.expect(self.prompt)
        p.sendline("px net.addHost('h3')")
        p.expect(self.prompt)
        p.sendline('px net.addLink(s1, h3)')
        p.expect(self.prompt)
        p.sendline('net')
        p.expect('h3')
        p.expect(self.prompt)
        p.sendline('py h3.MAC()')
        p.expect('([a-f0-9]{2}:?){6}')
        p.expect(self.prompt)
        p.sendline('exit')
        p.wait()

    def testLink(self):
        if False:
            print('Hello World!')
        'Test link CLI command using ping'
        p = pexpect.spawn('mn -w')
        p.expect(self.prompt)
        p.sendline('link s1 h1 down')
        p.expect(self.prompt)
        p.sendline('h1 ping -c 1 h2')
        p.expect('unreachable')
        p.expect(self.prompt)
        p.sendline('link s1 h1 up')
        p.expect(self.prompt)
        p.sendline('h1 ping -c 1 h2')
        p.expect('0% packet loss')
        p.expect(self.prompt)
        p.sendline('exit')
        p.wait()

    @unittest.skipUnless(os.path.exists('/tmp/pox') or '1 received' in quietRun('ping -c 1 github.com'), 'Github is not reachable; cannot download Pox')
    def testRemoteController(self):
        if False:
            return 10
        'Test Mininet using Pox controller'
        assert self
        if not os.path.exists('/tmp/pox'):
            p = pexpect.spawn('git clone https://github.com/noxrepo/pox.git /tmp/pox')
            p.expect(pexpect.EOF)
        pox = pexpect.spawn('/tmp/pox/pox.py forwarding.l2_learning')
        net = pexpect.spawn('mn --controller=remote,ip=127.0.0.1,port=6633 --test pingall')
        net.expect('0% dropped')
        net.expect(pexpect.EOF)
        pox.sendintr()
        pox.wait()
if __name__ == '__main__':
    unittest.main()