import unittest
import tempfile
import json
from sklearn.ensemble import GradientBoostingClassifier
from coremltools.converters import sklearn as skl_converter
from coremltools.proto import Model_pb2
from coremltools.proto import FeatureTypes_pb2
from coremltools._deps import _HAS_XGBOOST
from coremltools._deps import _HAS_SKLEARN
if _HAS_XGBOOST:
    import xgboost
    from coremltools.converters import xgboost as xgb_converter

@unittest.skipIf(not _HAS_SKLEARN, 'Missing sklearn. Skipping tests.')
class GradientBoostingBinaryClassifierScikitTest(unittest.TestCase):
    """
    Unit test class for testing scikit-learn converter.
    """

    @classmethod
    def setUpClass(self):
        if False:
            i = 10
            return i + 15
        '\n        Set up the unit test by loading the dataset and training a model.\n        '
        from sklearn.datasets import load_boston
        scikit_data = load_boston()
        scikit_model = GradientBoostingClassifier(random_state=1)
        target = scikit_data['target'] > scikit_data['target'].mean()
        scikit_model.fit(scikit_data['data'], target)
        self.scikit_data = scikit_data
        self.scikit_model = scikit_model

    def test_conversion(self):
        if False:
            while True:
                i = 10
        input_names = self.scikit_data.feature_names
        output_name = 'target'
        spec = skl_converter.convert(self.scikit_model, input_names, 'target').get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertIsNotNone(spec.treeEnsembleClassifier)
        self.assertEqual(spec.description.predictedFeatureName, 'target')
        self.assertEqual(len(spec.description.output), 2)
        self.assertEqual(spec.description.output[0].name, 'target')
        self.assertEqual(spec.description.output[0].type.WhichOneof('Type'), 'int64Type')
        for input_type in spec.description.input:
            self.assertEqual(input_type.type.WhichOneof('Type'), 'doubleType')
        self.assertEqual(sorted(input_names), sorted(map(lambda x: x.name, spec.description.input)))
        tr = spec.pipelineClassifier.pipeline.models[1].treeEnsembleClassifier.treeEnsemble
        self.assertIsNotNone(tr)
        self.assertEqual(len(tr.nodes), 1416)

    def test_conversion_bad_inputs(self):
        if False:
            return 10
        with self.assertRaises(Exception):
            model = GradientBoostingClassifier()
            spec = skl_converter.convert(model, 'data', 'out')
        from sklearn.preprocessing import OneHotEncoder
        with self.assertRaises(Exception):
            model = OneHotEncoder()
            spec = skl_converter.convert(model, 'data', 'out')

@unittest.skipIf(not _HAS_SKLEARN, 'Missing sklearn. Skipping tests.')
class GradientBoostingMulticlassClassifierScikitTest(unittest.TestCase):
    """
    Unit test class for testing scikit-learn converter.
    """

    @classmethod
    def setUpClass(self):
        if False:
            return 10
        '\n        Set up the unit test by loading the dataset and training a model.\n        '
        from sklearn.datasets import load_boston
        import numpy as np
        scikit_data = load_boston()
        scikit_model = GradientBoostingClassifier(random_state=1)
        t = scikit_data.target
        target = np.digitize(t, np.histogram(t)[1]) - 1
        scikit_model.fit(scikit_data.data, target)
        self.target = target
        self.scikit_data = scikit_data
        self.scikit_model = scikit_model

    def test_conversion(self):
        if False:
            while True:
                i = 10
        input_names = self.scikit_data.feature_names
        output_name = 'target'
        spec = skl_converter.convert(self.scikit_model, input_names, 'target').get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertEqual(spec.description.predictedFeatureName, 'target')
        self.assertEqual(len(spec.description.output), 2)
        self.assertEqual(spec.description.output[0].name, 'target')
        self.assertEqual(spec.description.output[0].type.WhichOneof('Type'), 'int64Type')
        for input_type in spec.description.input:
            self.assertEqual(input_type.type.WhichOneof('Type'), 'doubleType')
        self.assertEqual(sorted(input_names), sorted(map(lambda x: x.name, spec.description.input)))
        self.assertEqual(len(spec.pipelineClassifier.pipeline.models), 2)
        tr = spec.pipelineClassifier.pipeline.models[-1].treeEnsembleClassifier.treeEnsemble
        self.assertIsNotNone(tr)
        self.assertEqual(len(tr.nodes), 15056)

    def test_conversion_bad_inputs(self):
        if False:
            return 10
        with self.assertRaises(Exception):
            model = GradientBoostingClassifier()
            spec = skl_converter.convert(model, 'data', 'out')
        from sklearn.preprocessing import OneHotEncoder
        with self.assertRaises(Exception):
            model = OneHotEncoder()
            spec = skl_converter.convert(model, 'data', 'out')

@unittest.skipIf(not _HAS_SKLEARN, 'Missing sklearn. Skipping tests.')
@unittest.skipIf(not _HAS_XGBOOST, 'Skipping, no xgboost')
class GradientBoostingBinaryClassifierXGboostTest(unittest.TestCase):
    """
    Unit test class for testing xgboost converter.
    """

    @classmethod
    def setUpClass(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set up the unit test by loading the dataset and training a model.\n        '
        from sklearn.datasets import load_boston
        scikit_data = load_boston()
        self.xgb_model = xgboost.XGBClassifier()
        target = scikit_data['target'] > scikit_data['target'].mean()
        self.xgb_model.fit(scikit_data['data'], target)
        self.scikit_data = scikit_data

    def test_conversion(self):
        if False:
            print('Hello World!')
        input_names = self.scikit_data.feature_names
        output_name = 'target'
        spec = xgb_converter.convert(self.xgb_model, input_names, output_name, mode='classifier').get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertIsNotNone(spec.treeEnsembleClassifier)
        self.assertEqual(spec.description.predictedFeatureName, output_name)
        self.assertEqual(len(spec.description.output), 2)
        self.assertEqual(spec.description.output[0].name, output_name)
        self.assertEqual(spec.description.output[0].type.WhichOneof('Type'), 'int64Type')
        for input_type in spec.description.input:
            self.assertEqual(input_type.type.WhichOneof('Type'), 'doubleType')
        self.assertEqual(sorted(input_names), sorted(map(lambda x: x.name, spec.description.input)))
        tr = spec.treeEnsembleClassifier.treeEnsemble
        self.assertIsNotNone(tr)

    def test_conversion_bad_inputs(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(Exception):
            model = xgboost.XGBClassifier()
            spec = xgb_converter.convert(model, 'data', 'out', mode='classifier')
        with self.assertRaises(Exception):
            model = xgboost.XGBRegressor()
            spec = xgb_converter.convert(model, 'data', 'out', mode='classifier')

@unittest.skipIf(not _HAS_SKLEARN, 'Missing sklearn. Skipping tests.')
@unittest.skipIf(not _HAS_XGBOOST, 'Skipping, no xgboost')
class GradientBoostingMulticlassClassifierXGboostTest(unittest.TestCase):
    """
    Unit test class for testing xgboost converter.
    """

    @classmethod
    def setUpClass(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set up the unit test by loading the dataset and training a model.\n        '
        from sklearn.datasets import load_boston
        import numpy as np
        scikit_data = load_boston()
        t = scikit_data.target
        target = np.digitize(t, np.histogram(t)[1]) - 1
        dtrain = xgboost.DMatrix(scikit_data.data, label=target, feature_names=scikit_data.feature_names)
        self.xgb_model = xgboost.train({}, dtrain)
        self.target = target
        self.scikit_data = scikit_data
        self.n_classes = len(np.unique(self.target))

    def test_conversion(self):
        if False:
            print('Hello World!')
        input_names = self.scikit_data.feature_names
        output_name = 'target'
        spec = xgb_converter.convert(self.xgb_model, input_names, output_name, mode='classifier', n_classes=self.n_classes).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertEqual(spec.description.predictedFeatureName, output_name)
        self.assertEqual(len(spec.description.output), 2)
        self.assertEqual(spec.description.output[0].name, output_name)
        self.assertEqual(spec.description.output[0].type.WhichOneof('Type'), 'int64Type')
        for input_type in spec.description.input:
            self.assertEqual(input_type.type.WhichOneof('Type'), 'doubleType')
        self.assertEqual(sorted(input_names), sorted(map(lambda x: x.name, spec.description.input)))
        tr = spec.treeEnsembleClassifier.treeEnsemble
        self.assertIsNotNone(tr)

    def test_conversion_from_file(self):
        if False:
            print('Hello World!')
        import numpy as np
        output_name = 'target'
        feature_names = self.scikit_data.feature_names
        xgb_model_json = tempfile.mktemp('xgb_tree_model_classifier.json')
        xgb_json_out = self.xgb_model.get_dump(with_stats=True, dump_format='json')
        with open(xgb_model_json, 'w') as f:
            json.dump(xgb_json_out, f)
        spec = xgb_converter.convert(xgb_model_json, feature_names, output_name, mode='classifier', n_classes=self.n_classes).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertIsNotNone(spec.treeEnsembleRegressor)
        self.assertEqual(spec.description.predictedFeatureName, output_name)
        self.assertEqual(len(spec.description.output), 2)
        self.assertEqual(spec.description.output[0].name, output_name)
        self.assertEqual(spec.description.output[0].type.WhichOneof('Type'), 'int64Type')
        for input_type in spec.description.input:
            self.assertEqual(input_type.type.WhichOneof('Type'), 'doubleType')
        self.assertEqual(sorted(self.scikit_data.feature_names), sorted(map(lambda x: x.name, spec.description.input)))
        tr = spec.treeEnsembleClassifier.treeEnsemble
        self.assertIsNotNone(tr)