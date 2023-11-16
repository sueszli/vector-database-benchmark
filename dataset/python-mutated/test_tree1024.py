"""
Test for tree1024.py
"""
import unittest
from mininet.util import pexpect
import sys

class testTree1024(unittest.TestCase):
    prompt = 'mininet>'

    @unittest.skipIf('-quick' in sys.argv, 'long test')
    def testTree1024(self):
        if False:
            i = 10
            return i + 15
        'Run the example and do a simple ping test from h1 to h1024'
        p = pexpect.spawn('python -m mininet.examples.tree1024')
        p.expect(self.prompt, timeout=6000)
        p.sendline('h1 ping -c 20 h1024')
        p.expect('(\\d+)% packet loss')
        packetLossPercent = int(p.match.group(1)) if p.match else -1
        p.expect(self.prompt)
        p.sendline('exit')
        p.wait()
        self.assertLess(packetLossPercent, 60)
if __name__ == '__main__':
    unittest.main()