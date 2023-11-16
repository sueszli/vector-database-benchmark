import unittest
import tempfile
import json
from coremltools._deps import _HAS_SKLEARN, _HAS_XGBOOST
if _HAS_XGBOOST:
    import xgboost
    from coremltools.converters import xgboost as xgb_converter
if _HAS_SKLEARN:
    from sklearn.datasets import load_boston
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import OneHotEncoder
    from coremltools.converters import sklearn as skl_converter

@unittest.skipIf(not _HAS_SKLEARN, 'Missing scikit-learn. Skipping tests.')
class GradientBoostingRegressorScikitTest(unittest.TestCase):
    """
    Unit test class for testing scikit-learn converter.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set up the unit test by loading the dataset and training a model.\n        '
        if not _HAS_SKLEARN:
            return
        scikit_data = load_boston()
        scikit_model = GradientBoostingRegressor(random_state=1)
        scikit_model.fit(scikit_data['data'], scikit_data['target'])
        cls.scikit_data = scikit_data
        cls.scikit_model = scikit_model

    def test_conversion(self):
        if False:
            i = 10
            return i + 15
        input_names = self.scikit_data.feature_names
        output_name = 'target'
        spec = skl_converter.convert(self.scikit_model, input_names, 'target').get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertEqual(spec.description.predictedFeatureName, 'target')
        self.assertEqual(len(spec.description.output), 1)
        self.assertEqual(spec.description.output[0].name, 'target')
        self.assertEqual(spec.description.output[0].type.WhichOneof('Type'), 'doubleType')
        for input_type in spec.description.input:
            self.assertEqual(input_type.type.WhichOneof('Type'), 'doubleType')
        self.assertEqual(sorted(input_names), sorted(map(lambda x: x.name, spec.description.input)))
        tr = spec.pipelineRegressor.pipeline.models[-1].treeEnsembleRegressor.treeEnsemble
        self.assertIsNotNone(tr)
        self.assertEqual(len(tr.nodes), 1426)

    def test_conversion_bad_inputs(self):
        if False:
            print('Hello World!')
        with self.assertRaises(Exception):
            model = GradientBoostingRegressor()
            spec = skl_converter.convert(model, 'data', 'out')
        with self.assertRaises(Exception):
            model = OneHotEncoder()
            spec = skl_converter.convert(model, 'data', 'out')

@unittest.skipIf(not _HAS_SKLEARN, 'Missing scikit-learn. Skipping tests.')
@unittest.skipIf(not _HAS_XGBOOST, 'Skipping, no xgboost')
class BoostedTreeRegressorXGboostTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set up the unit test by loading the dataset and training a model.\n        '
        if not _HAS_XGBOOST:
            return
        if not _HAS_SKLEARN:
            return
        scikit_data = load_boston()
        dtrain = xgboost.DMatrix(scikit_data.data, label=scikit_data.target, feature_names=scikit_data.feature_names)
        xgb_model = xgboost.train({}, dtrain, 1)
        self.scikit_data = scikit_data
        self.xgb_model = xgb_model
        self.feature_names = self.scikit_data.feature_names

    def test_conversion(self):
        if False:
            return 10
        feature_names = self.scikit_data.feature_names
        output_name = 'target'
        spec = xgb_converter.convert(self.xgb_model, feature_names, 'target').get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertIsNotNone(spec.treeEnsembleRegressor)
        self.assertEqual(spec.description.predictedFeatureName, 'target')
        self.assertEqual(len(spec.description.output), 1)
        self.assertEqual(spec.description.output[0].name, 'target')
        self.assertEqual(spec.description.output[0].type.WhichOneof('Type'), 'doubleType')
        for input_type in spec.description.input:
            self.assertEqual(input_type.type.WhichOneof('Type'), 'doubleType')
        self.assertEqual(sorted(self.feature_names), sorted(map(lambda x: x.name, spec.description.input)))
        tr = spec.treeEnsembleRegressor.treeEnsemble
        self.assertIsNotNone(tr)
        self.assertEqual(len(tr.nodes), 23)

    def test_conversion_from_file(self):
        if False:
            print('Hello World!')
        output_name = 'target'
        feature_names = self.feature_names
        xgb_model_json = tempfile.mktemp('tree_model.json')
        xgb_json_out = self.xgb_model.get_dump(dump_format='json')
        with open(xgb_model_json, 'w') as f:
            json.dump(xgb_json_out, f)
        spec = xgb_converter.convert(xgb_model_json, feature_names, 'target').get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertIsNotNone(spec.treeEnsembleRegressor)
        self.assertEqual(spec.description.predictedFeatureName, 'target')
        self.assertEqual(len(spec.description.output), 1)
        self.assertEqual(spec.description.output[0].name, 'target')
        self.assertEqual(spec.description.output[0].type.WhichOneof('Type'), 'doubleType')
        for input_type in spec.description.input:
            self.assertEqual(input_type.type.WhichOneof('Type'), 'doubleType')
        self.assertEqual(sorted(self.feature_names), sorted(map(lambda x: x.name, spec.description.input)))
        tr = spec.treeEnsembleRegressor.treeEnsemble
        self.assertIsNotNone(tr)
        self.assertEqual(len(tr.nodes), 23)

    def test_unsupported_conversion(self):
        if False:
            i = 10
            return i + 15
        feature_names = self.scikit_data.feature_names
        output_name = 'target'
        xgb_model = xgboost.XGBRegressor(objective='reg:gamma')
        xgb_model.fit(self.scikit_data.data, self.scikit_data.target)
        with self.assertRaises(ValueError):
            spec = xgb_converter.convert(xgb_model, feature_names, 'target')
        xgb_model = xgboost.XGBRegressor(objective='reg:tweedie')
        xgb_model.fit(self.scikit_data.data, self.scikit_data.target)
        with self.assertRaises(ValueError):
            spec = xgb_converter.convert(xgb_model, feature_names, 'target')

    def test_conversion_bad_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypeError):
            model = GradientBoostingRegressor()
            spec = xgb_converter.convert(model, 'data', 'out')
        with self.assertRaises(TypeError):
            model = OneHotEncoder()
            spec = xgb_converter.convert(model, 'data', 'out')