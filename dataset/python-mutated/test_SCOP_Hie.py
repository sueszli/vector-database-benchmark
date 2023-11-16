"""Unit test for Hie."""
import unittest
from Bio.SCOP import Hie

class HieTests(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.filename = './SCOP/dir.hie.scop.txt_test'

    def testParse(self):
        if False:
            return 10
        'Test if all records in a HIE file are being read.'
        count = 0
        with open(self.filename) as f:
            for record in Hie.parse(f):
                count += 1
        self.assertEqual(count, 21)

    def testStr(self):
        if False:
            print('Hello World!')
        'Test if we can convert each record to a string correctly.'
        with open(self.filename) as f:
            for line in f:
                record = Hie.Record(line)
                self.assertEqual(str(record).rstrip(), line.rstrip())

    def testError(self):
        if False:
            i = 10
            return i + 15
        'Test if a corrupt record raises the appropriate exception.'
        corruptRec = '4926sdfhjhfgyjdfyg'
        self.assertRaises(ValueError, Hie.Record, corruptRec)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)