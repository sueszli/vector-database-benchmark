"""
Test for sshd.py
"""
import unittest
from mininet.util import pexpect
from mininet.clean import sh

class testSSHD(unittest.TestCase):
    opts = ['\\(yes/no\\)\\?', 'refused', 'Welcome|\\$|#', pexpect.EOF, pexpect.TIMEOUT]

    def connected(self, ip):
        if False:
            for i in range(10):
                print('nop')
        'Log into ssh server, check banner, then exit'
        ssh = 'ssh -o StrictHostKeyChecking=no -i /tmp/ssh/test_rsa ' + ip
        p = pexpect.spawn(ssh, timeout=5)
        while True:
            index = p.expect(self.opts)
            if index == 0:
                print(p.match.group(0))
                p.sendline('yes')
            elif index == 1:
                return False
            elif index == 2:
                p.sendline('exit')
                p.wait()
                return True
            else:
                return False

    def setUp(self):
        if False:
            return 10
        sh('rm -rf /tmp/ssh')
        sh('mkdir /tmp/ssh')
        sh("ssh-keygen -t rsa -P '' -f /tmp/ssh/test_rsa")
        sh('cat /tmp/ssh/test_rsa.pub >> /tmp/ssh/authorized_keys')
        cmd = 'python -m mininet.examples.sshd -D -o AuthorizedKeysFile=/tmp/ssh/authorized_keys -o StrictModes=no -o UseDNS=no -u0'
        self.net = pexpect.spawn(cmd)
        self.net.expect('mininet>')

    def testSSH(self):
        if False:
            return 10
        'Verify that we can ssh into all hosts (h1 to h4)'
        for h in range(1, 5):
            self.assertTrue(self.connected('10.0.0.%d' % h))

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.net.sendline('exit')
        self.net.wait()
        sh('rm -rf /tmp/ssh')
if __name__ == '__main__':
    unittest.main()