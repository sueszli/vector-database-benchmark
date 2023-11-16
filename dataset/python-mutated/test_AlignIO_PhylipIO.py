"""Tests for Bio.AlignIO.PhylipIO module."""
import unittest
from io import StringIO
from Bio.AlignIO.PhylipIO import PhylipIterator
from Bio.AlignIO.PhylipIO import PhylipWriter

class TestPhylipIO(unittest.TestCase):

    def test_one(self):
        if False:
            for i in range(10):
                print('nop')
        input_file = 'Phylip/one.dat'
        with open(input_file) as handle:
            ids = []
            for alignment in PhylipIterator(handle):
                for record in alignment:
                    ids.append(record.id)
        self.assertEqual(ids, ['V_Harveyi_', 'B_subtilis', 'B_subtilis', 'YA80_HAEIN', 'FLIY_ECOLI', 'E_coli_Gln', 'Deinococcu', 'HISJ_E_COL'])
        expected = 'mkklvlslsl vlafssataa faaipqniri gtdptyapfe sknsqgelvg\n        fdidlakelc krintqctfv enpldalips lkakkidaim sslsitekrq qeiaftdkly\n        aadsrlvvak nsdiqptves lkgkrvgvlq gttqetfgne hwapkgieiv syqgqdniys\n        dltagridaafqdevaaseg flkqpvgkdy kfggpsvkde klfgvgtgmg lrkednelre\n        alnkafaemradgtyeklak kyfdfdvygg'.replace(' ', '').replace('\n', '').upper()
        self.assertEqual(str(record.seq).replace('-', ''), expected)

    def test_two_and_three(self):
        if False:
            for i in range(10):
                print('nop')
        path = 'Phylip/two.dat'
        with open(path) as handle:
            list2 = list(PhylipIterator(handle))
        self.assertEqual(len(list2), 1)
        self.assertEqual(len(list2[0]), 5)
        path = 'Phylip/three.dat'
        with open(path) as handle:
            list3 = list(PhylipIterator(handle))
        self.assertEqual(len(list3), 1)
        self.assertEqual(len(list3[0]), 5)
        for i in range(5):
            self.assertEqual(list2[0][i].id, list3[0][i].id)
            self.assertEqual(list2[0][i].seq, list3[0][i].seq)

    def test_four(self):
        if False:
            while True:
                i = 10
        path = 'Phylip/four.dat'
        with open(path) as handle:
            list4 = list(PhylipIterator(handle))
        self.assertEqual(len(list4), 1)
        self.assertEqual(len(list4[0]), 5)

    def test_five(self):
        if False:
            for i in range(10):
                print('nop')
        path = 'Phylip/five.dat'
        with open(path) as handle:
            self.assertRaises(ValueError, list, PhylipIterator(handle))

    def test_six(self):
        if False:
            print('Hello World!')
        path = 'Phylip/six.dat'
        with open(path) as handle:
            list5 = list(PhylipIterator(handle))
        self.assertEqual(len(list5), 1)

    def test_concatenation(self):
        if False:
            while True:
                i = 10
        path = 'Phylip/one.dat'
        with open(path) as handle:
            phylip_text = handle.read()
        path = 'Phylip/three.dat'
        with open(path) as handle:
            phylip_text3 = handle.read()
        path = 'Phylip/four.dat'
        with open(path) as handle:
            phylip_text4 = handle.read()
        handle = StringIO(phylip_text4 + '\n' + phylip_text4)
        self.assertEqual(len(list(PhylipIterator(handle))), 2)
        handle = StringIO(phylip_text3 + '\n' + phylip_text4 + '\n\n\n' + phylip_text)
        self.assertEqual(len(list(PhylipIterator(handle))), 3)

    def test_write_read(self):
        if False:
            for i in range(10):
                print('nop')
        path = 'Phylip/six.dat'
        with open(path) as handle:
            list5 = list(PhylipIterator(handle))
        handle = StringIO()
        PhylipWriter(handle).write_file(list5)
        handle.seek(0)
        list6 = list(PhylipIterator(handle))
        self.assertEqual(len(list5), len(list6))
        for (a1, a2) in zip(list5, list6):
            self.assertEqual(len(a1), len(a2))
            for (r1, r2) in zip(a1, a2):
                self.assertEqual(r1.id, r2.id)
                self.assertEqual(r1.seq, r2.seq)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)