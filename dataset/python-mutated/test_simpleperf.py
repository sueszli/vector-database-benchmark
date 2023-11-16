"""
Test for simpleperf.py
"""
import unittest
from mininet.util import pexpect
import sys
from mininet.log import setLogLevel
from mininet.examples.simpleperf import SingleSwitchTopo

class testSimplePerf(unittest.TestCase):

    @unittest.skipIf('-quick' in sys.argv, 'long test')
    def testE2E(self):
        if False:
            for i in range(10):
                print('nop')
        'Run the example and verify iperf results'
        BW = 10
        TOLERANCE = 0.2
        p = pexpect.spawn('python -m mininet.examples.simpleperf testmode')
        p.logfile = sys.stdout
        p.expect("Results: \\['10M', '([\\d\\.]+) .bits/sec", timeout=90)
        measuredBw = float(p.match.group(1))
        lowerBound = BW * (1 - TOLERANCE)
        upperBound = BW + (1 + TOLERANCE)
        self.assertGreaterEqual(measuredBw, lowerBound)
        self.assertLessEqual(measuredBw, upperBound)
        p.wait()
if __name__ == '__main__':
    setLogLevel('debug')
    unittest.main()