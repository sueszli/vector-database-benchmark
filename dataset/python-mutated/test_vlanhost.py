"""
Test for vlanhost.py
"""
import unittest
from mininet.util import pexpect
import sys
from mininet.util import quietRun

class testVLANHost(unittest.TestCase):
    prompt = 'mininet>'

    @unittest.skipIf('-quick' in sys.argv, 'long test')
    def testVLANTopo(self):
        if False:
            i = 10
            return i + 15
        'Test connectivity (or lack thereof) between hosts in VLANTopo'
        p = pexpect.spawn('python -m mininet.examples.vlanhost')
        p.expect(self.prompt)
        p.sendline('pingall 1')
        p.expect('(\\d+)% dropped', timeout=30)
        percent = int(p.match.group(1)) if p.match else -1
        p.expect(self.prompt)
        p.sendline('exit')
        p.wait()
        self.assertEqual(percent, 80)

    def testSpecificVLAN(self):
        if False:
            i = 10
            return i + 15
        'Test connectivity between hosts on a specific VLAN'
        vlan = 1001
        p = pexpect.spawn('python -m mininet.examples.vlanhost %d' % vlan)
        p.expect(self.prompt)
        p.sendline('h1 ping -c 1 h2')
        p.expect('(\\d+)% packet loss')
        percent = int(p.match.group(1)) if p.match else -1
        p.expect(self.prompt)
        p.sendline('h1 ifconfig')
        i = p.expect(['h1-eth0.%d' % vlan, pexpect.TIMEOUT], timeout=2)
        p.expect(self.prompt)
        p.sendline('exit')
        p.wait()
        self.assertEqual(percent, 0)
        self.assertEqual(i, 0)
if __name__ == '__main__':
    unittest.main()