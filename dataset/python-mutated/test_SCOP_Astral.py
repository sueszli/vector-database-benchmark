"""Unit test for Astral."""
import unittest
from Bio.SCOP import Astral, Scop

class AstralTests(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.scop = Scop(dir_path='SCOP', version='test')
        self.astral = Astral(scop=self.scop, dir_path='SCOP', version='test')

    def testGetSeq(self):
        if False:
            return 10
        self.assertEqual(self.astral.getSeqBySid('d3sdha_'), 'AAAAA')
        self.assertEqual(self.astral.getSeqBySid('d4hbib_'), 'KKKKK')
        dom = self.scop.getDomainBySid('d3sdha_')
        self.assertEqual(self.astral.getSeq(dom), 'AAAAA')

    def testConstructWithCustomFile(self):
        if False:
            return 10
        scop = Scop(dir_path='SCOP', version='test')
        astral = Astral(scop=scop, astral_file='SCOP/scopseq-test/astral-scopdom-seqres-all-test.fa')
        self.assertEqual(astral.getSeqBySid('d3sdha_'), 'AAAAA')
        self.assertEqual(astral.getSeqBySid('d4hbib_'), 'KKKKK')

    def testGetDomainsFromFile(self):
        if False:
            for i in range(10):
                print('nop')
        filename = 'SCOP/scopseq-test/astral-scopdom-seqres-sel-gs-bib-20-test.id'
        domains = self.astral.getAstralDomainsFromFile(filename)
        self.assertEqual(len(domains), 3)
        self.assertEqual(domains[0].sid, 'd3sdha_')
        self.assertEqual(domains[1].sid, 'd4hbib_')
        self.assertEqual(domains[2].sid, 'd5hbia_')

    def testGetDomainsClustered(self):
        if False:
            for i in range(10):
                print('nop')
        domains1 = self.astral.domainsClusteredById(20)
        self.assertEqual(len(domains1), 3)
        self.assertEqual(domains1[0].sid, 'd3sdha_')
        self.assertEqual(domains1[1].sid, 'd4hbib_')
        self.assertEqual(domains1[2].sid, 'd5hbia_')
        domains2 = self.astral.domainsClusteredByEv(1e-15)
        self.assertEqual(len(domains2), 1)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)