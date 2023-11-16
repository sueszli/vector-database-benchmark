"""Unit test for Dom.

This test requires the mini DOM file 'testDom.txt'
"""
import unittest
from Bio.SCOP import Dom

class DomTests(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.filename = './SCOP/testDom.txt'

    def testParse(self):
        if False:
            i = 10
            return i + 15
        'Test if all records in a DOM file are being read.'
        count = 0
        with open(self.filename) as f:
            for record in Dom.parse(f):
                count += 1
        self.assertEqual(count, 10)

    def testStr(self):
        if False:
            print('Hello World!')
        'Test if we can convert each record to a string correctly.'
        with open(self.filename) as f:
            for line in f:
                record = Dom.Record(line)
                self.assertEqual(str(record).rstrip(), line.rstrip())

    def testError(self):
        if False:
            print('Hello World!')
        'Test if a corrupt record raises the appropriate exception.'
        corruptDom = '49xxx268\tsp\tb.1.2.1\t-\n'
        self.assertRaises(ValueError, Dom.Record, corruptDom)

    def testRecord(self):
        if False:
            return 10
        'Test one record in detail.'
        recLine = 'd7hbib_\t7hbi\tb:\t1.001.001.001.001.001'
        rec = Dom.Record(recLine)
        self.assertEqual(rec.sid, 'd7hbib_')
        self.assertEqual(rec.residues.pdbid, '7hbi')
        self.assertEqual(rec.residues.fragments, (('b', '', ''),))
        self.assertEqual(rec.hierarchy, '1.001.001.001.001.001')
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)