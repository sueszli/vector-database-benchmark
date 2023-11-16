"""This test checks for correct wait4() behavior.
"""
import os
import time
import sys
import unittest
from test.fork_wait import ForkWait
from test import support
support.get_attribute(os, 'fork')
support.get_attribute(os, 'wait4')

class Wait4Test(ForkWait):

    def wait_impl(self, cpid, *, exitcode):
        if False:
            print('Hello World!')
        option = os.WNOHANG
        if sys.platform.startswith('aix'):
            option = 0
        deadline = time.monotonic() + support.SHORT_TIMEOUT
        while time.monotonic() <= deadline:
            (spid, status, rusage) = os.wait4(cpid, option)
            if spid == cpid:
                break
            time.sleep(0.1)
        self.assertEqual(spid, cpid)
        self.assertEqual(os.waitstatus_to_exitcode(status), exitcode)
        self.assertTrue(rusage)

def tearDownModule():
    if False:
        for i in range(10):
            print('nop')
    support.reap_children()
if __name__ == '__main__':
    unittest.main()