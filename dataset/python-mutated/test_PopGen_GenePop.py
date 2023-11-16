"""Test GenePop."""
import os
import unittest
from Bio import MissingExternalDependencyError
from Bio.PopGen.GenePop.Controller import GenePopController
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

class AppTest(unittest.TestCase):
    """Tests genepop execution via biopython."""

    def test_allele_genotype_frequencies(self):
        if False:
            return 10
        'Test genepop execution on basic allele and genotype frequencies.'
        ctrl = GenePopController()
        path = os.path.join('PopGen', 'big.gen')
        (pop_iter, locus_iter) = ctrl.calc_allele_genotype_freqs(path)

    def test_calc_diversities_fis_with_identity(self):
        if False:
            for i in range(10):
                print('nop')
        'Test calculations of diversities.'
        ctrl = GenePopController()
        path = os.path.join('PopGen', 'big.gen')
        (iter, avg_fis, avg_Qintra) = ctrl.calc_diversities_fis_with_identity(path)
        liter = list(iter)
        self.assertEqual(len(liter), 37)
        self.assertEqual(liter[0][0], 'Locus1')
        self.assertEqual(len(avg_fis), 10)
        self.assertEqual(len(avg_Qintra), 10)

    def test_estimate_nm(self):
        if False:
            return 10
        'Test Nm estimation.'
        ctrl = GenePopController()
        path = os.path.join('PopGen', 'big.gen')
        (mean_sample_size, mean_priv_alleles, mig10, mig25, mig50, mig_corrected) = ctrl.estimate_nm(path)
        self.assertAlmostEqual(mean_sample_size, 28.0)
        self.assertAlmostEqual(mean_priv_alleles, 0.016129)
        self.assertAlmostEqual(mig10, 52.5578)
        self.assertAlmostEqual(mig25, 15.3006)
        self.assertAlmostEqual(mig50, 8.94583)
        self.assertAlmostEqual(mig_corrected, 13.6612)

    def test_fst_all(self):
        if False:
            for i in range(10):
                print('nop')
        'Test genepop execution on all fst.'
        ctrl = GenePopController()
        path = os.path.join('PopGen', 'c2line.gen')
        ((allFis, allFst, allFit), itr) = ctrl.calc_fst_all(path)
        results = list(itr)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], '136255903')
        self.assertAlmostEqual(results[1][3], 0.335846)

    def test_haploidy(self):
        if False:
            return 10
        'Test haploidy.'
        ctrl = GenePopController()
        path = os.path.join('PopGen', 'haplo.gen')
        ((allFis, allFst, allFit), itr) = ctrl.calc_fst_all(path)
        litr = list(itr)
        self.assertNotIsInstance(allFst, int)
        self.assertEqual(len(litr), 37)
        self.assertEqual(litr[36][0], 'Locus37')
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)