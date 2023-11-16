"""Unit test for Cla."""
import unittest
from Bio.SCOP import Cla

class ClaTests(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.filename = './SCOP/dir.cla.scop.txt_test'

    def testParse(self):
        if False:
            while True:
                i = 10
        'Test if all records in a CLA file are being read.'
        count = 0
        with open(self.filename) as f:
            records = Cla.parse(f)
            for record in records:
                count += 1
        self.assertEqual(count, 14)

    def testStr(self):
        if False:
            while True:
                i = 10
        'Test if we can convert each record to a string correctly.'
        with open(self.filename) as f:
            for line in f:
                record = Cla.Record(line)
                expected_hierarchy = line.rstrip().split('\t')[5].split(',')
                expected_hierarchy = dict((pair.split('=') for pair in expected_hierarchy))
                actual_hierarchy = str(record).rstrip().split('\t')[5].split(',')
                actual_hierarchy = dict((pair.split('=') for pair in actual_hierarchy))
                self.assertEqual(len(actual_hierarchy), len(expected_hierarchy))
                for (key, actual_value) in actual_hierarchy.items():
                    self.assertEqual(actual_value, expected_hierarchy[key])

    def testError(self):
        if False:
            i = 10
            return i + 15
        'Test if a corrupt record raises the appropriate exception.'
        corruptRec = '49268\tsp\tb.1.2.1\t-\n'
        self.assertRaises(ValueError, Cla.Record, corruptRec)

    def testRecord(self):
        if False:
            while True:
                i = 10
        'Test one record in detail.'
        recLine = 'd1dan.1\t1dan\tT:,U:91-106\tb.1.2.1\t21953\tcl=48724,cf=48725,sf=49265,fa=49266,dm=49267,sp=49268,px=21953'
        record = Cla.Record(recLine)
        self.assertEqual(record.sid, 'd1dan.1')
        self.assertEqual(record.residues.pdbid, '1dan')
        self.assertEqual(record.residues.fragments, (('T', '', ''), ('U', '91', '106')))
        self.assertEqual(record.sccs, 'b.1.2.1')
        self.assertEqual(record.sunid, 21953)
        self.assertEqual(record.hierarchy, {'cl': 48724, 'cf': 48725, 'sf': 49265, 'fa': 49266, 'dm': 49267, 'sp': 49268, 'px': 21953})

    def testIndex(self):
        if False:
            i = 10
            return i + 15
        'Test CLA file indexing.'
        index = Cla.Index(self.filename)
        self.assertEqual(len(index), 14)
        self.assertIn('d4hbia_', index)
        rec = index['d1hbia_']
        self.assertEqual(rec.sunid, 14996)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)