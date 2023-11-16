"""Testing online code in Bio.SCOP module."""
import unittest
from Bio import SCOP
import requires_internet
requires_internet.check()

class ScopSearch(unittest.TestCase):
    """SCOP search tests."""

    def test_search(self):
        if False:
            for i in range(10):
                print('nop')
        'Test search.'
        handle = SCOP.search('1JOY')
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)