import unittest
import numpy as np
from coremltools._deps import _HAS_SKLEARN
if _HAS_SKLEARN:
    from coremltools.converters import sklearn as converter
    from sklearn.preprocessing import Imputer

@unittest.skipIf(not _HAS_SKLEARN, 'Missing sklearn. Skipping tests.')
class ImputerTestCase(unittest.TestCase):
    """
    Unit test class for testing scikit-learn converter.
    """

    @classmethod
    def setUpClass(self):
        if False:
            return 10
        '\n        Set up the unit test by loading the dataset and training a model.\n        '
        from sklearn.datasets import load_boston
        scikit_data = load_boston()
        scikit_model = Imputer(strategy='most_frequent', axis=0)
        scikit_data['data'][1, 8] = np.NaN
        input_data = scikit_data['data'][:, 8].reshape(-1, 1)
        scikit_model.fit(input_data, scikit_data['target'])
        self.scikit_data = scikit_data
        self.scikit_model = scikit_model

    def test_conversion(self):
        if False:
            return 10
        spec = converter.convert(self.scikit_model, 'data', 'out').get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.pipeline.models[-1].HasField('imputer'))

    def test_conversion_bad_inputs(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(Exception):
            model = Imputer()
            spec = converter.convert(model, 'data', 'out')
        with self.assertRaises(Exception):
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            spec = converter.convert(model, 'data', 'out')