"""Tests for GenePop easy-controller."""
import os
import unittest
from Bio import MissingExternalDependencyError
from Bio.PopGen.GenePop.EasyController import EasyController
found = False
for path in os.environ['PATH'].split(os.pathsep):
    try:
        for filename in os.listdir(path):
            if filename.startswith('Genepop'):
                found = True
    except OSError:
        pass
if not found:
    raise MissingExternalDependencyError('Install GenePop if you want to use Bio.PopGen.GenePop.')
cur_dir = os.path.abspath('.')

class AppTest(unittest.TestCase):
    """Tests genepop execution via biopython using EasyController."""

    def setUp(self):
        if False:
            while True:
                i = 10
        'Change working directory.'
        os.chdir('PopGen')
        self.ctrl = EasyController('big.gen')

    def tearDown(self):
        if False:
            return 10
        'Restore working directory.'
        os.chdir(cur_dir)

    def test_basic_info(self):
        if False:
            for i in range(10):
                print('nop')
        'Test basic info.'
        (pops, loci) = self.ctrl.get_basic_info()
        self.assertEqual(len(pops), 10)
        self.assertEqual(len(loci), 37)

    def test_get_heterozygosity_info(self):
        if False:
            for i in range(10):
                print('nop')
        'Test heterozygosity info.'
        hz_info = self.ctrl.get_heterozygosity_info(0, 'Locus2')
        self.assertEqual(hz_info[1], 24)
        self.assertEqual(hz_info[3], 7)

    def test_get_alleles(self):
        if False:
            for i in range(10):
                print('nop')
        'Test get alleles.'
        self.assertCountEqual(self.ctrl.get_alleles(0, 'Locus3'), [3, 20])

    def test_get_alleles_all_pops(self):
        if False:
            print('Hello World!')
        'Test get alleles for all populations.'
        self.assertEqual(self.ctrl.get_alleles_all_pops('Locus4'), [1, 3])

    def test_get_fis(self):
        if False:
            i = 10
            return i + 15
        'Test get Fis.'
        (alleles, overall) = self.ctrl.get_fis(0, 'Locus2')
        self.assertEqual(alleles[3][0], 55)
        self.assertEqual(overall[0], 62)

    def test_get_allele_frequency(self):
        if False:
            for i in range(10):
                print('nop')
        'Test allele frequency.'
        (tot_genes, alleles) = self.ctrl.get_allele_frequency(0, 'Locus2')
        self.assertEqual(tot_genes, 62)
        self.assertLess(abs(alleles[20] - 0.113), 0.05)

    def test_get_genotype_count(self):
        if False:
            print('Hello World!')
        'Test genotype count.'
        self.assertEqual(len(self.ctrl.get_genotype_count(0, 'Locus2')), 3)

    def test_estimate_nm(self):
        if False:
            return 10
        'Test Nm estimation.'
        nms = self.ctrl.estimate_nm()
        self.assertEqual(nms[0], 28.0)

    def test_hwe_excess(self):
        if False:
            i = 10
            return i + 15
        'Test Hardy-Weinberg Equilibrium.'
        hwe_excess = self.ctrl.test_hw_pop(0, 'excess')
        self.assertEqual(hwe_excess['Locus1'], (0.4955, None, -0.16, -0.1623, 5))

    def test_get_avg_fis(self):
        if False:
            print('Hello World!')
        'Test average Fis.'
        self.ctrl.get_avg_fis()

    def test_get_multilocus_f_stats(self):
        if False:
            return 10
        'Test multilocus F stats.'
        mf = self.ctrl.get_multilocus_f_stats()
        self.assertEqual(len(mf), 3)
        self.assertLess(mf[0], 0.1)

    def test_get_f_stats(self):
        if False:
            i = 10
            return i + 15
        'Test F stats.'
        fs = self.ctrl.get_f_stats('Locus2')
        self.assertEqual(len(fs), 5)
        self.assertLess(fs[0], 0)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)