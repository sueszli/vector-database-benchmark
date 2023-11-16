"""Unit test for Des."""
import unittest
from Bio.SCOP import Des

class DesTests(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.filename = './SCOP/dir.des.scop.txt_test'

    def testParse(self):
        if False:
            return 10
        'Test if all records in a DES file are being read.'
        count = 0
        with open(self.filename) as f:
            records = Des.parse(f)
            for record in records:
                count += 1
        self.assertEqual(count, 20)

    def testStr(self):
        if False:
            i = 10
            return i + 15
        'Test if we can convert each record to a string correctly.'
        with open(self.filename) as f:
            for line in f:
                record = Des.Record(line)
                self.assertEqual(str(record).rstrip(), line.rstrip())

    def testError(self):
        if False:
            print('Hello World!')
        'Test if a corrupt record raises the appropriate exception.'
        corruptRec = '49268\tsp\tb.1.2.1\t-\n'
        self.assertRaises(ValueError, Des.Record, corruptRec)

    def testRecord(self):
        if False:
            while True:
                i = 10
        'Test one record in detail.'
        recLine = '49268\tsp\tb.1.2.1\t-\tHuman (Homo sapiens)    \n'
        recFields = (49268, 'sp', 'b.1.2.1', '', 'Human (Homo sapiens)')
        record = Des.Record(recLine)
        self.assertEqual(record.sunid, recFields[0])
        self.assertEqual(record.nodetype, recFields[1])
        self.assertEqual(record.sccs, recFields[2])
        self.assertEqual(record.name, recFields[3])
        self.assertEqual(record.description, recFields[4])
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)