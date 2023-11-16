"""
Tests for controllers.py and controllers2.py
"""
import unittest
from mininet.util import pexpect

class testControllers(unittest.TestCase):
    prompt = 'mininet>'

    def connectedTest(self, name, cmap):
        if False:
            return 10
        'Verify that switches are connected to the controller specified by cmap'
        p = pexpect.spawn('python -m %s' % name)
        p.expect(self.prompt)
        p.sendline('pingall')
        p.expect('(\\d+)% dropped')
        percent = int(p.match.group(1)) if p.match else -1
        self.assertEqual(percent, 0)
        p.expect(self.prompt)
        for switch in cmap:
            p.sendline('sh ovs-vsctl get-controller %s' % switch)
            p.expect('tcp:([\\d.:]+)')
            actual = p.match.group(1)
            expected = cmap[switch]
            self.assertEqual(actual, expected)
        p.expect(self.prompt)
        p.sendline('exit')
        p.wait()

    def testControllers(self):
        if False:
            i = 10
            return i + 15
        c0 = '127.0.0.1:6633'
        c1 = '127.0.0.1:6634'
        cmap = {'s1': c0, 's2': c1, 's3': c0}
        self.connectedTest('mininet.examples.controllers', cmap)

    def testControllers2(self):
        if False:
            print('Hello World!')
        c0 = '127.0.0.1:6633'
        c1 = '127.0.0.1:6634'
        cmap = {'s1': c0, 's2': c1}
        self.connectedTest('mininet.examples.controllers2', cmap)
if __name__ == '__main__':
    unittest.main()