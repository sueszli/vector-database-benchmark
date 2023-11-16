"""
Test for scratchnet.py
"""
import unittest
from mininet.util import pexpect

class testScratchNet(unittest.TestCase):
    opts = ['1 packets transmitted, 1 received, 0% packet loss', pexpect.EOF]

    def pingTest(self, name):
        if False:
            i = 10
            return i + 15
        'Verify that no ping packets were dropped'
        p = pexpect.spawn('python -m %s' % name)
        index = p.expect(self.opts, timeout=120)
        self.assertEqual(index, 0)
        p.wait()

    def testPingKernel(self):
        if False:
            while True:
                i = 10
        self.pingTest('mininet.examples.scratchnet')

    def testPingUser(self):
        if False:
            return 10
        self.pingTest('mininet.examples.scratchnetuser')
if __name__ == '__main__':
    unittest.main()