import unittest
from coalib.results.HiddenResult import HiddenResult

class HiddenResultTest(unittest.TestCase):

    def test_hidden_result(self):
        if False:
            print('Hello World!')
        uut = HiddenResult('any', 'anything')
        self.assertEqual(uut.contents, 'anything')