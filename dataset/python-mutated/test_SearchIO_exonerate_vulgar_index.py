"""Tests for SearchIO exonerate-vulgar indexing."""
import os
import unittest
from search_tests_common import CheckIndex

class ExonerateVulgarIndexCases(CheckIndex):
    fmt = 'exonerate-vulgar'

    def test_exn_22_m_est2genome(self):
        if False:
            for i in range(10):
                print('nop')
        'Test exonerate-vulgar indexing, single.'
        filename = os.path.join('Exonerate', 'exn_22_o_vulgar.exn')
        self.check_index(filename, self.fmt)

    def test_exn_22_q_multiple(self):
        if False:
            print('Hello World!')
        'Test exonerate-vulgar indexing, single.'
        filename = os.path.join('Exonerate', 'exn_22_q_multiple_vulgar.exn')
        self.check_index(filename, self.fmt)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)