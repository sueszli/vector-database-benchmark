from .context import sample
import unittest

class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_thoughts(self):
        if False:
            while True:
                i = 10
        self.assertIsNone(sample.hmm())
if __name__ == '__main__':
    unittest.main()