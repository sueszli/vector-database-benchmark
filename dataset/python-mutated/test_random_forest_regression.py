import unittest
from sklearn.ensemble import RandomForestRegressor
from coremltools._deps import _HAS_SKLEARN
from coremltools.proto import Model_pb2
from coremltools.proto import FeatureTypes_pb2
if _HAS_SKLEARN:
    from sklearn.ensemble import RandomForestRegressor
    from coremltools.converters import sklearn as skl_converter

@unittest.skipIf(not _HAS_SKLEARN, 'Missing scikit-learn. Skipping tests.')
class RandomForestRegressorScikitTest(unittest.TestCase):
    """
    Unit test class for testing scikit-learn converter.
    """

    @classmethod
    def setUpClass(self):
        if False:
            return 10
        '\n        Set up the unit test by loading the dataset and training a model.\n        '
        from sklearn.datasets import load_boston
        from sklearn.ensemble import RandomForestRegressor
        scikit_data = load_boston()
        scikit_model = RandomForestRegressor(random_state=1)
        scikit_model.fit(scikit_data['data'], scikit_data['target'])
        self.scikit_data = scikit_data
        self.scikit_model = scikit_model

    def test_conversion(self):
        if False:
            i = 10
            return i + 15
        input_names = self.scikit_data.feature_names
        output_name = 'target'
        spec = skl_converter.convert(self.scikit_model, input_names, 'target').get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertEquals(spec.description.predictedFeatureName, 'target')
        self.assertEquals(len(spec.description.output), 1)
        self.assertEquals(spec.description.output[0].name, 'target')
        self.assertEquals(spec.description.output[0].type.WhichOneof('Type'), 'doubleType')
        for input_type in spec.description.input:
            self.assertEquals(input_type.type.WhichOneof('Type'), 'doubleType')
        self.assertEqual(sorted(input_names), sorted(map(lambda x: x.name, spec.description.input)))
        self.assertEquals(len(spec.pipelineRegressor.pipeline.models), 2)
        tr = spec.pipelineRegressor.pipeline.models[-1].treeEnsembleRegressor.treeEnsemble
        self.assertIsNotNone(tr)
        self.assertEquals(len(tr.nodes), 5996)

    def test_conversion_bad_inputs(self):
        if False:
            return 10
        with self.assertRaises(Exception):
            model = RandomForestRegressor()
            spec = skl_converter.convert(model, 'data', 'out')
        from sklearn.preprocessing import OneHotEncoder
        with self.assertRaises(Exception):
            model = OneHotEncoder()
            spec = skl_converter.convert(model, 'data', 'out')