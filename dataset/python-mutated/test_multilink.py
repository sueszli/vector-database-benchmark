"""
Test for multiple links between nodes
validates mininet interfaces against systems interfaces
"""
import unittest
from mininet.util import pexpect

class testMultiLink(unittest.TestCase):
    prompt = 'mininet>'

    def testMultiLink(self):
        if False:
            return 10
        p = pexpect.spawn('python -m mininet.examples.multilink')
        p.expect(self.prompt)
        p.sendline('intfs')
        p.expect('s(\\d): lo')
        intfsOutput = p.before
        hostToIntfs = intfsOutput.split('\r\n')[1:3]
        intfList = []
        for hostToIntf in hostToIntfs:
            intfList += [intf for intf in hostToIntf.split()[1].split(',')]
        sysIntfList = []
        opts = ['h(\\d)-eth(\\d)', self.prompt]
        p.expect(self.prompt)
        p.sendline('h1 ifconfig')
        while True:
            p.expect(opts)
            if p.after == self.prompt:
                break
            sysIntfList.append(p.after)
        p.sendline('h2 ifconfig')
        while True:
            p.expect(opts)
            if p.after == self.prompt:
                break
            sysIntfList.append(p.after)
        failMsg = 'The systems interfaces and mininet interfaces\nare not the same'
        self.assertEqual(sysIntfList, intfList, msg=failMsg)
        p.sendline('exit')
        p.wait()
if __name__ == '__main__':
    unittest.main()