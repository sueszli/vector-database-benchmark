"""Tests for SVDSuperimposer module."""
import unittest
try:
    from numpy import array
    from numpy import dot
    from numpy import array_equal
    from numpy import around
    from numpy.linalg import svd, det
except ImportError:
    from Bio import MissingPythonDependencyError
    raise MissingPythonDependencyError('Install NumPy if you want to use Bio.SVDSuperimposer.') from None
from Bio.SVDSuperimposer import SVDSuperimposer

class SVDSuperimposerTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = array([[51.65, -1.9, 50.07], [50.4, -1.23, 50.65], [50.68, -0.04, 51.54], [50.22, -0.02, 52.85]])
        self.y = array([[51.3, -2.99, 46.54], [51.09, -1.88, 47.58], [52.36, -1.2, 48.03], [52.71, -1.18, 49.38]])
        self.sup = SVDSuperimposer()
        self.sup.set(self.x, self.y)

    def test_get_init_rms(self):
        if False:
            print('Hello World!')
        x = array([[1.19, 1.28, 1.37], [1.46, 1.55, 1.64], [1.73, 1.82, 1.91]])
        y = array([[1.91, 1.82, 1.73], [1.64, 1.55, 1.46], [1.37, 1.28, 1.19]])
        self.sup.set(x, y)
        self.assertIsNone(self.sup.init_rms)
        self.assertAlmostEqual(self.sup.get_init_rms(), 0.8049844719)

    def test_oldTest(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(array_equal(around(self.sup.reference_coords, decimals=3), around(self.x, decimals=3)))
        self.assertTrue(array_equal(around(self.sup.coords, decimals=3), around(self.y, decimals=3)))
        self.assertIsNone(self.sup.rot)
        self.assertIsNone(self.sup.tran)
        self.assertIsNone(self.sup.rms)
        self.assertIsNone(self.sup.init_rms)
        self.sup.run()
        self.assertTrue(array_equal(around(self.sup.reference_coords, decimals=3), around(self.x, decimals=3)))
        self.assertTrue(array_equal(around(self.sup.coords, decimals=3), around(self.y, decimals=3)))
        rot = array([[0.68304983, 0.53664371, 0.49543563], [-0.52277295, 0.83293229, -0.18147242], [-0.51005037, -0.13504564, 0.84947707]])
        tran = array([38.78608157, -20.65451334, -15.42227366])
        self.assertTrue(array_equal(around(self.sup.rot, decimals=3), around(rot, decimals=3)))
        self.assertTrue(array_equal(around(self.sup.tran, decimals=3), around(tran, decimals=3)))
        self.assertIsNone(self.sup.rms)
        self.assertIsNone(self.sup.init_rms)
        self.assertAlmostEqual(self.sup.get_rms(), 0.00304266526014)
        (rot_get, tran_get) = self.sup.get_rotran()
        self.assertTrue(array_equal(around(rot_get, decimals=3), around(rot, decimals=3)))
        self.assertTrue(array_equal(around(tran_get, decimals=3), around(tran, decimals=3)))
        y_on_x1 = dot(self.y, rot) + tran
        y_x_solution = array([[51.6518846, -1.9001827, 50.0708397], [50.3977138, -1.2287705, 50.64882], [50.6801788, -0.0416095666, 51.5368866], [50.2202228, -0.0194372374, 52.8534537]])
        self.assertTrue(array_equal(around(y_on_x1, decimals=3), around(y_x_solution, decimals=3)))
        y_on_x2 = self.sup.get_transformed()
        self.assertTrue(array_equal(around(y_on_x2, decimals=3), around(y_x_solution, decimals=3)))
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)