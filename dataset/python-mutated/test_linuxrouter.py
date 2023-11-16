"""
Test for linuxrouter.py
"""
import unittest
from mininet.util import pexpect
from mininet.util import quietRun

class testLinuxRouter(unittest.TestCase):
    prompt = 'mininet>'

    def testPingall(self):
        if False:
            print('Hello World!')
        'Test connectivity between hosts'
        p = pexpect.spawn('python -m mininet.examples.linuxrouter')
        p.expect(self.prompt)
        p.sendline('pingall')
        p.expect('(\\d+)% dropped')
        percent = int(p.match.group(1)) if p.match else -1
        p.expect(self.prompt)
        p.sendline('exit')
        p.wait()
        self.assertEqual(percent, 0)

    def testRouterPing(self):
        if False:
            print('Hello World!')
        'Test connectivity from h1 to router'
        p = pexpect.spawn('python -m mininet.examples.linuxrouter')
        p.expect(self.prompt)
        p.sendline('h1 ping -c 1 r0')
        p.expect('(\\d+)% packet loss')
        percent = int(p.match.group(1)) if p.match else -1
        p.expect(self.prompt)
        p.sendline('exit')
        p.wait()
        self.assertEqual(percent, 0)

    def testTTL(self):
        if False:
            return 10
        'Verify that the TTL is decremented'
        p = pexpect.spawn('python -m mininet.examples.linuxrouter')
        p.expect(self.prompt)
        p.sendline('h1 ping -c 1 h2')
        p.expect('ttl=(\\d+)')
        ttl = int(p.match.group(1)) if p.match else -1
        p.expect(self.prompt)
        p.sendline('exit')
        p.wait()
        self.assertEqual(ttl, 63)
if __name__ == '__main__':
    unittest.main()