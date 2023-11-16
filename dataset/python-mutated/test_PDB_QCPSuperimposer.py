"""Tests for QCPSuperimposer module."""
import unittest
try:
    import numpy as np
except ImportError:
    from Bio import MissingPythonDependencyError
    raise MissingPythonDependencyError('Install NumPy if you want to use Bio.QCPSuperimposer.') from None
from Bio.PDB import PDBParser, Selection
from Bio.PDB.qcprot import QCPSuperimposer
from Bio.SVDSuperimposer import SVDSuperimposer

class QCPSuperimposerTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.x = np.array([[51.65, -1.9, 50.07], [50.4, -1.23, 50.65], [50.68, -0.04, 51.54], [50.22, -0.02, 52.85]])
        self.y = np.array([[51.3, -2.99, 46.54], [51.09, -1.88, 47.58], [52.36, -1.2, 48.03], [52.71, -1.18, 49.38]])

    def test_set(self):
        if False:
            while True:
                i = 10
        'Test setting of initial parameters.'
        sup = QCPSuperimposer()
        sup.set(self.x, self.y)
        self.assertTrue(np.allclose(sup.reference_coords, self.x, atol=1e-06))
        self.assertTrue(np.allclose(sup.coords, self.y, atol=1e-06))
        self.assertIsNone(sup.transformed_coords)
        self.assertIsNone(sup.rot)
        self.assertIsNone(sup.tran)
        self.assertIsNone(sup.rms)
        self.assertIsNone(sup.init_rms)

    def test_run(self):
        if False:
            for i in range(10):
                print('nop')
        'Test QCP on dummy data.'
        sup = QCPSuperimposer()
        sup.set(self.x, self.y)
        sup.run()
        self.assertTrue(np.allclose(sup.reference_coords, self.x, atol=1e-06))
        self.assertTrue(np.allclose(sup.coords, self.y, atol=1e-06))
        self.assertIsNone(sup.transformed_coords)
        calc_rot = [[0.683, 0.537, 0.495], [-0.523, 0.833, -0.181], [-0.51, -0.135, 0.849]]
        calc_tran = [38.786, -20.655, -15.422]
        self.assertTrue(np.allclose(np.array(calc_rot), sup.rot, atol=0.001))
        self.assertTrue(np.allclose(np.array(calc_tran), sup.tran, atol=0.001))
        self.assertAlmostEqual(sup.rms, 0.003, places=3)
        self.assertIsNone(sup.init_rms)

    def test_compare_to_svd(self):
        if False:
            i = 10
            return i + 15
        'Compare results of QCP to SVD.'
        sup = QCPSuperimposer()
        sup.set(self.x, self.y)
        sup.run()
        svd_sup = SVDSuperimposer()
        svd_sup.set(self.x, self.y)
        svd_sup.run()
        self.assertAlmostEqual(svd_sup.get_rms(), sup.rms, places=3)
        self.assertTrue(np.allclose(svd_sup.rot, sup.rot, atol=0.001))
        self.assertTrue(np.allclose(svd_sup.tran, sup.tran, atol=0.001))
        self.assertTrue(np.allclose(svd_sup.get_transformed(), sup.get_transformed(), atol=0.001))

    def test_get_transformed(self):
        if False:
            i = 10
            return i + 15
        'Test transformation of coordinates after QCP.'
        sup = QCPSuperimposer()
        sup.set(self.x, self.y)
        sup.run()
        transformed_coords = [[51.652, -1.9, 50.071], [50.398, -1.229, 50.649], [50.68, -0.042, 51.537], [50.22, -0.019, 52.853]]
        self.assertTrue(np.allclose(sup.get_transformed(), np.array(transformed_coords), atol=0.001))

    def test_get_init_rms(self):
        if False:
            i = 10
            return i + 15
        'Test initial RMS calculation.'
        x = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        y = np.array([[2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        sup = QCPSuperimposer()
        sup.set(x, y)
        self.assertIsNone(sup.init_rms)
        expected_init_rms = 1.0
        self.assertAlmostEqual(sup.get_init_rms(), expected_init_rms, places=6)

    def test_on_pdb(self):
        if False:
            return 10
        'Align a PDB to itself.'
        pdb1 = 'PDB/1A8O.pdb'
        p = PDBParser()
        s1 = p.get_structure('FIXED', pdb1)
        fixed = Selection.unfold_entities(s1, 'A')
        s2 = p.get_structure('MOVING', pdb1)
        moving = Selection.unfold_entities(s2, 'A')
        rot = np.eye(3, dtype=np.float64)
        tran = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        for atom in moving:
            atom.transform(rot, tran)
        sup = QCPSuperimposer()
        sup.set_atoms(fixed, moving)
        self.assertTrue(np.allclose(sup.rotran[0], rot, atol=0.001))
        self.assertTrue(np.allclose(sup.rotran[1], -tran, atol=0.001))
        self.assertAlmostEqual(sup.rms, 0.0, places=6)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)