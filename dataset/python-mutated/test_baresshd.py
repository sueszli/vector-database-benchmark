"""
Tests for baresshd.py
"""
import unittest
from mininet.util import pexpect
from mininet.clean import cleanup, sh
from sys import stdout

class testBareSSHD(unittest.TestCase):
    opts = ['Welcome to h1', pexpect.EOF, pexpect.TIMEOUT]

    def connected(self):
        if False:
            while True:
                i = 10
        'Log into ssh server, check banner, then exit'
        p = pexpect.spawn('ssh 10.0.0.1 -o ConnectTimeout=1 -o StrictHostKeyChecking=no -i /tmp/ssh/test_rsa exit')
        while True:
            index = p.expect(self.opts)
            if index == 0:
                return True
            else:
                return False

    def setUp(self):
        if False:
            return 10
        self.assertFalse(self.connected())
        sh('rm -rf /tmp/ssh')
        sh('mkdir /tmp/ssh')
        sh("ssh-keygen -t rsa -P '' -f /tmp/ssh/test_rsa")
        sh('cat /tmp/ssh/test_rsa.pub >> /tmp/ssh/authorized_keys')
        cmd = 'python -m mininet.examples.baresshd -o AuthorizedKeysFile=/tmp/ssh/authorized_keys -o StrictModes=no'
        p = pexpect.spawn(cmd)
        runOpts = ['You may now ssh into h1 at 10.0.0.1', 'after 5 seconds, h1 is not listening on port 22', pexpect.EOF, pexpect.TIMEOUT]
        while True:
            index = p.expect(runOpts)
            if index == 0:
                break
            else:
                self.tearDown()
                self.fail('sshd failed to start in host h1')

    def testSSH(self):
        if False:
            for i in range(10):
                print('nop')
        'Simple test to verify that we can ssh into h1'
        result = False
        result = self.connected()
        self.assertTrue(result)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        sh("ps aux | grep ssh |grep Banner| awk '{ print $2 }' | xargs kill")
        cleanup()
        sh('rm -rf /tmp/ssh')
if __name__ == '__main__':
    unittest.main()