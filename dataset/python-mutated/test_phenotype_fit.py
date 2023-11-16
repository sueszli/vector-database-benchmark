"""Tests for the Bio.phenotype module's fitting functionality."""
try:
    import numpy
    del numpy
except ImportError:
    from Bio import MissingExternalDependencyError
    raise MissingExternalDependencyError('Install NumPy if you want to use Bio.phenotype.') from None
try:
    import scipy
    del scipy
    from scipy.optimize import OptimizeWarning
except ImportError:
    from Bio import MissingExternalDependencyError
    raise MissingExternalDependencyError('Install SciPy if you want to use Bio.phenotype fit functionality.') from None
import json
import unittest
from Bio import BiopythonExperimentalWarning
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', BiopythonExperimentalWarning)
    from Bio import phenotype
JSON_PLATE = 'phenotype/Plate.json'

class TestPhenoMicro(unittest.TestCase):
    """Tests for phenotype module."""

    def test_WellRecord(self):
        if False:
            print('Hello World!')
        'Test basic functionalities of WellRecord objects.'
        with open(JSON_PLATE) as handle:
            p = json.load(handle)
        times = p['measurements']['Hour']
        w = phenotype.phen_micro.WellRecord('A10', signals={times[i]: p['measurements']['A10'][i] for i in range(len(times))})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', OptimizeWarning)
            w.fit()
        self.assertAlmostEqual(w.area, 20879.5)
        self.assertEqual(w.model, 'gompertz')
        self.assertAlmostEqual(w.lag, 6.042586872509036, places=5)
        self.assertAlmostEqual(w.plateau, 188.51404344898586, places=4)
        self.assertAlmostEqual(w.slope, 48.19061828483113, places=4)
        self.assertAlmostEqual(w.v, 0.1, places=5)
        self.assertAlmostEqual(w.y0, 45.87977006980799, places=4)
        self.assertEqual(w.max, 313.0)
        self.assertEqual(w.min, 29.0)
        self.assertEqual(w.average_height, 217.82552083333334)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)