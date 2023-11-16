"""Test for Blast records."""
import unittest
from Bio.Blast.Record import HSP

class TestHsp(unittest.TestCase):

    def test_str(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(str(HSP()), 'Score <unknown> (<unknown> bits), expectation <unknown>, alignment length <unknown>')
        hsp = HSP()
        hsp.score = 1.0
        hsp.bits = 2.0
        hsp.expect = 3.0
        hsp.align_length = 4
        self.assertEqual('\n'.join((line.strip() for line in str(hsp).split('\n'))), 'Score 1 (2 bits), expectation 3.0e+00, alignment length 4\nQuery:    None  None\n\nSbjct:    None  None')
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)