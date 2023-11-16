"""Unit tests for the Bio.PDB.Polypeptide module."""
import unittest
from Bio.PDB import PDBParser, PPBuilder, CaPPBuilder
from Bio.Seq import Seq

class PolypeptideTests(unittest.TestCase):
    """Test Polypeptide module."""

    @classmethod
    def setUpClass(self):
        if False:
            return 10
        pdb1 = 'PDB/1A8O.pdb'
        self.parser = PDBParser(PERMISSIVE=True)
        self.structure = self.parser.get_structure('scr', pdb1)

    def test_ppbuilder_real(self):
        if False:
            while True:
                i = 10
        'Test PPBuilder on real PDB file.'
        ppb = PPBuilder()
        pp = ppb.build_peptides(self.structure)
        self.assertEqual(len(pp), 3)
        self.assertEqual(pp[0][0].get_id()[1], 152)
        self.assertEqual(pp[0][-1].get_id()[1], 184)
        self.assertEqual(pp[1][0].get_id()[1], 186)
        self.assertEqual(pp[1][-1].get_id()[1], 213)
        self.assertEqual(pp[2][0].get_id()[1], 216)
        self.assertEqual(pp[2][-1].get_id()[1], 220)
        pp0_seq = pp[0].get_sequence()
        pp1_seq = pp[1].get_sequence()
        pp2_seq = pp[2].get_sequence()
        self.assertIsInstance(pp0_seq, Seq)
        self.assertEqual(pp0_seq, 'DIRQGPKEPFRDYVDRFYKTLRAEQASQEVKNW')
        self.assertEqual(pp1_seq, 'TETLLVQNANPDCKTILKALGPGATLEE')
        self.assertEqual(pp2_seq, 'TACQG')

    def test_ppbuilder_real_nonstd(self):
        if False:
            while True:
                i = 10
        'Test PPBuilder on real PDB file allowing non-standard amino acids.'
        ppb = PPBuilder()
        pp = ppb.build_peptides(self.structure, False)
        self.assertEqual(len(pp), 1)
        self.assertEqual(pp[0][0].get_id()[1], 151)
        self.assertEqual(pp[0][-1].get_id()[1], 220)
        s = pp[0].get_sequence()
        self.assertIsInstance(s, Seq)
        self.assertEqual('MDIRQGPKEPFRDYVDRFYKTLRAEQASQEVKNWMTETLLVQNANPDCKTILKALGPGATLEEMMTACQG', s)

    def test_ppbuilder_torsion(self):
        if False:
            i = 10
            return i + 15
        'Test phi/psi angles calculated with PPBuilder.'
        ppb = PPBuilder()
        pp = ppb.build_peptides(self.structure)
        phi_psi = pp[0].get_phi_psi_list()
        self.assertIsNone(phi_psi[0][0])
        self.assertAlmostEqual(phi_psi[0][1], -0.46297171497725553, places=3)
        self.assertAlmostEqual(phi_psi[1][0], -1.0873937604007962, places=3)
        self.assertAlmostEqual(phi_psi[1][1], 2.133770783263711, places=3)
        self.assertAlmostEqual(phi_psi[2][0], -2.405223274365188, places=3)
        self.assertAlmostEqual(phi_psi[2][1], 2.3807316946081554, places=3)
        phi_psi = pp[1].get_phi_psi_list()
        self.assertIsNone(phi_psi[0][0])
        self.assertAlmostEqual(phi_psi[0][1], -0.6810077089092923, places=3)
        self.assertAlmostEqual(phi_psi[1][0], -1.2654003477656888, places=3)
        self.assertAlmostEqual(phi_psi[1][1], -0.5868998704275631, places=3)
        self.assertAlmostEqual(phi_psi[2][0], -1.7467679151684763, places=3)
        self.assertAlmostEqual(phi_psi[2][1], -1.5655066256698336, places=3)
        phi_psi = pp[2].get_phi_psi_list()
        self.assertIsNone(phi_psi[0][0])
        self.assertAlmostEqual(phi_psi[0][1], -0.7322288421088972, places=3)
        self.assertAlmostEqual(phi_psi[1][0], -1.104474023456626, places=3)
        self.assertAlmostEqual(phi_psi[1][1], -0.6968133459278288, places=3)
        self.assertAlmostEqual(phi_psi[2][0], -1.8497413300164958, places=3)
        self.assertAlmostEqual(phi_psi[2][1], 0.3476288983480906, places=3)

    def test_cappbuilder_real(self):
        if False:
            return 10
        'Test CaPPBuilder on real PDB file.'
        ppb = CaPPBuilder()
        pp = ppb.build_peptides(self.structure)
        pp0_seq = pp[0].get_sequence()
        pp1_seq = pp[1].get_sequence()
        pp2_seq = pp[2].get_sequence()
        self.assertEqual(pp0_seq, 'DIRQGPKEPFRDYVDRFYKTLRAEQASQEVKNW')
        self.assertEqual(pp1_seq, 'TETLLVQNANPDCKTILKALGPGATLEE')
        self.assertEqual(pp2_seq, 'TACQG')
        self.assertEqual([ca.serial_number for ca in pp[0].get_ca_list()], [10, 18, 26, 37, 46, 50, 57, 66, 75, 82, 93, 104, 112, 124, 131, 139, 150, 161, 173, 182, 189, 197, 208, 213, 222, 231, 236, 242, 251, 260, 267, 276, 284])

    def test_cappbuilder_real_nonstd(self):
        if False:
            i = 10
            return i + 15
        'Test CaPPBuilder on real PDB file allowing non-standard amino acids.'
        ppb = CaPPBuilder()
        pp = ppb.build_peptides(self.structure, False)
        self.assertEqual(len(pp), 1)
        self.assertEqual(pp[0][0].get_id()[1], 151)
        self.assertEqual(pp[0][-1].get_id()[1], 220)
        s = pp[0].get_sequence()
        self.assertIsInstance(s, Seq)
        self.assertEqual('MDIRQGPKEPFRDYVDRFYKTLRAEQASQEVKNWMTETLLVQNANPDCKTILKALGPGATLEEMMTACQG', s)

    def test_cappbuilder_tau(self):
        if False:
            i = 10
            return i + 15
        'Test tau angles calculated with CaPPBuilder.'
        ppb = CaPPBuilder()
        pp = ppb.build_peptides(self.structure)
        taus = pp[1].get_tau_list()
        self.assertAlmostEqual(taus[0], 0.3597907225123525, places=3)
        self.assertAlmostEqual(taus[1], 0.43239284636769254, places=3)
        self.assertAlmostEqual(taus[2], 0.9982015749271211, places=3)
        thetas = pp[2].get_theta_list()
        self.assertAlmostEqual(thetas[0], 1.6610069445335354, places=3)
        self.assertAlmostEqual(thetas[1], 1.7491703334817772, places=3)
        self.assertAlmostEqual(thetas[2], 2.0702447422720143, places=3)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)