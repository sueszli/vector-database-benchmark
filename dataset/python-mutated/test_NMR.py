"""Unit tests for the Bio.NMR Module."""
import unittest
import tempfile
import os
from Bio.NMR import xpktools
from Bio.NMR import NOEtools

class NmrTests(unittest.TestCase):
    """Tests for NMR module."""

    def test_xpktools(self):
        if False:
            return 10
        'Self test for NMR.xpktools.'
        self.xpk_file = 'NMR/noed.xpk'
        self.peaklist = xpktools.Peaklist(self.xpk_file)
        self.assertEqual(self.peaklist.firstline, 'label dataset sw sf ')
        self.assertEqual(self.peaklist.axislabels, 'H1 15N2 N15 ')
        self.assertEqual(self.peaklist.dataset, 'test.nv')
        self.assertEqual(self.peaklist.sw, '{1571.86 } {1460.01 } {1460.00 }')
        self.assertEqual(self.peaklist.sf, '{599.8230 } { 60.7860 } { 60.7860 }')
        self.assertEqual(self.peaklist.datalabels, ' H1.L  H1.P  H1.W  H1.B  H1.E  H1.J  15N2.L  15N2.P  15N2.W  15N2.B  15N2.E  15N2.J  N15.L  N15.P  N15.W  N15.B  N15.E  N15.J  vol  int  stat ')
        self.assertEqual(len(self.peaklist.data), 8)
        self.assertEqual(self.peaklist.data[0], '0  3.hn   8.853   0.021   0.010   ++   0.000   3.n   120.104   0.344   0.010   PP   0.000   3.n   120.117   0.344   0.010   PP   0.000  1.18200 1.18200 0')
        self.assertEqual(self.peaklist.data[7], '8  10.hn   7.663   0.021   0.010   ++   0.000   10.n   118.341   0.324   0.010   +E   0.000   10.n   118.476   0.324   0.010   +E   0.000  0.49840 0.49840 0')
        self.assertEqual(len(self.peaklist.residue_dict('H1')['10']), 1)
        self.assertEqual(self.peaklist.residue_dict('H1')['10'][0], '8  10.hn   7.663   0.021   0.010   ++   0.000   10.n   118.341   0.324   0.010   +E   0.000   10.n   118.476   0.324   0.010   +E   0.000  0.49840 0.49840 0')

    def test_noetools(self):
        if False:
            return 10
        'Self test for NMR.NOEtools.\n\n        Calculate and compare crosspeak peaklist files\n        Adapted from Doc/examples/nmr/simplepredict.py by Robert Bussell, Jr.\n        '
        self.xpk_i_file = os.path.join('NMR', 'noed.xpk')
        self.xpk_expected = os.path.join('NMR', 'out_example.xpk')
        (self.f_number, self.f_predicted) = tempfile.mkstemp()
        os.close(self.f_number)
        try:
            self.peaklist = xpktools.Peaklist(self.xpk_i_file)
            self.res_dict = self.peaklist.residue_dict('H1')
            max_res = self.res_dict['maxres']
            min_res = self.res_dict['minres']
            self.peaklist.write_header(self.f_predicted)
            inc = 1
            count = 0
            res = min_res
            out_list = []
            while res <= max_res:
                noe1 = NOEtools.predictNOE(self.peaklist, '15N2', 'H1', res, res + inc)
                noe2 = NOEtools.predictNOE(self.peaklist, '15N2', 'H1', res, res - inc)
                if noe1 != '':
                    noe1 = noe1 + '\n'
                    noe1 = xpktools.replace_entry(noe1, 1, count)
                    out_list.append(noe1)
                    count += 1
                    if noe2 != '':
                        noe2 = noe2 + '\n'
                        noe2 = xpktools.replace_entry(noe2, 1, count)
                        out_list.append(noe2)
                        count += 1
                res += 1
            with open(self.f_predicted, 'a') as outfile:
                outfile.writelines(out_list)
            pre_content = open(self.f_predicted).read()
            exp_content = open(self.xpk_expected).read()
            self.assertEqual(pre_content, exp_content)
        finally:
            os.remove(self.f_predicted)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)