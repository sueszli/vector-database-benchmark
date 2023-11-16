"""
Test for natnet.py
"""
import unittest
from mininet.util import pexpect
from mininet.util import quietRun

class testNATNet(unittest.TestCase):
    prompt = 'mininet>'

    def setUp(self):
        if False:
            return 10
        self.net = pexpect.spawn('python -m mininet.examples.natnet')
        self.net.expect(self.prompt)

    def testPublicPing(self):
        if False:
            i = 10
            return i + 15
        'Attempt to ping the public server (h0) from h1 and h2'
        self.net.sendline('h1 ping -c 1 h0')
        self.net.expect('(\\d+)% packet loss')
        percent = int(self.net.match.group(1)) if self.net.match else -1
        self.assertEqual(percent, 0)
        self.net.expect(self.prompt)
        self.net.sendline('h2 ping -c 1 h0')
        self.net.expect('(\\d+)% packet loss')
        percent = int(self.net.match.group(1)) if self.net.match else -1
        self.assertEqual(percent, 0)
        self.net.expect(self.prompt)

    def testPrivatePing(self):
        if False:
            while True:
                i = 10
        'Attempt to ping h1 and h2 from public server'
        self.net.sendline('h0 ping -c 1 -t 1 h1')
        result = self.net.expect(['unreachable', 'loss'])
        self.assertEqual(result, 0)
        self.net.expect(self.prompt)
        self.net.sendline('h0 ping -c 1 -t 1 h2')
        result = self.net.expect(['unreachable', 'loss'])
        self.assertEqual(result, 0)
        self.net.expect(self.prompt)

    def testPrivateToPrivatePing(self):
        if False:
            while True:
                i = 10
        "Attempt to ping from NAT'ed host h1 to NAT'ed host h2"
        self.net.sendline('h1 ping -c 1 -t 1 h2')
        result = self.net.expect(['[Uu]nreachable', 'loss'])
        self.assertEqual(result, 0)
        self.net.expect(self.prompt)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.net.sendline('exit')
        self.net.wait()
if __name__ == '__main__':
    unittest.main()