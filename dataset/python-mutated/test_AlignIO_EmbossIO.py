"""Tests for Bio.AlignIO.EmbossIO module."""
import unittest
from io import StringIO
from Bio.AlignIO.EmbossIO import EmbossIterator
with open('Emboss/alignret.txt') as handle:
    simple_example = handle.read()
with open('Emboss/water.txt') as handle:
    pair_example = handle.read()
with open('Emboss/needle.txt') as handle:
    pair_example2 = handle.read()
with open('Emboss/needle_overhang.txt') as handle:
    pair_example3 = handle.read()

class TestEmbossIO(unittest.TestCase):

    def test_pair_example(self):
        if False:
            while True:
                i = 10
        alignments = list(EmbossIterator(StringIO(pair_example)))
        self.assertEqual(len(alignments), 1)
        self.assertEqual(len(alignments[0]), 2)
        self.assertEqual([r.id for r in alignments[0]], ['IXI_234', 'IXI_235'])

    def test_simple_example(self):
        if False:
            print('Hello World!')
        alignments = list(EmbossIterator(StringIO(simple_example)))
        self.assertEqual(len(alignments), 1)
        self.assertEqual(len(alignments[0]), 4)
        self.assertEqual([r.id for r in alignments[0]], ['IXI_234', 'IXI_235', 'IXI_236', 'IXI_237'])

    def test_pair_plus_simple(self):
        if False:
            print('Hello World!')
        alignments = list(EmbossIterator(StringIO(pair_example + simple_example)))
        self.assertEqual(len(alignments), 2)
        self.assertEqual(len(alignments[0]), 2)
        self.assertEqual(len(alignments[1]), 4)
        self.assertEqual([r.id for r in alignments[0]], ['IXI_234', 'IXI_235'])
        self.assertEqual([r.id for r in alignments[1]], ['IXI_234', 'IXI_235', 'IXI_236', 'IXI_237'])

    def test_pair_example2(self):
        if False:
            for i in range(10):
                print('nop')
        alignments = list(EmbossIterator(StringIO(pair_example2)))
        self.assertEqual(len(alignments), 5)
        self.assertEqual(len(alignments[0]), 2)
        self.assertEqual([r.id for r in alignments[0]], ['ref_rec', 'gi|94968718|receiver'])
        self.assertEqual([r.id for r in alignments[4]], ['ref_rec', 'gi|94970041|receiver'])

    def test_pair_example3(self):
        if False:
            for i in range(10):
                print('nop')
        alignments = list(EmbossIterator(StringIO(pair_example3)))
        self.assertEqual(len(alignments), 1)
        self.assertEqual(len(alignments[0]), 2)
        self.assertEqual([r.id for r in alignments[0]], ['asis', 'asis'])
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)