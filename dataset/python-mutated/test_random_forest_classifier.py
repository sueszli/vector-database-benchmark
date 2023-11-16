import unittest
from coremltools._deps import _HAS_SKLEARN
from coremltools.proto import Model_pb2
from coremltools.proto import FeatureTypes_pb2
if _HAS_SKLEARN:
    from sklearn.ensemble import RandomForestClassifier
    from coremltools.converters import sklearn as skl_converter

@unittest.skipIf(not _HAS_SKLEARN, 'Missing sklearn. Skipping tests.')
class RandomForestBinaryClassifierScikitTest(unittest.TestCase):
    """
    Unit test class for testing scikit-learn converter.
    """

    @classmethod
    def setUpClass(self):
        if False:
            while True:
                i = 10
        '\n        Set up the unit test by loading the dataset and training a model.\n        '
        from sklearn.datasets import load_boston
        from sklearn.ensemble import RandomForestClassifier
        scikit_data = load_boston()
        scikit_model = RandomForestClassifier(random_state=1)
        target = 1 * (scikit_data['target'] > scikit_data['target'].mean())
        scikit_model.fit(scikit_data['data'], target)
        self.scikit_data = scikit_data
        self.scikit_model = scikit_model

    def test_conversion(self):
        if False:
            print('Hello World!')
        input_names = self.scikit_data.feature_names
        output_name = 'target'
        spec = skl_converter.convert(self.scikit_model, input_names, 'target').get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertEquals(spec.description.predictedFeatureName, 'target')
        self.assertEquals(len(spec.description.output), 2)
        self.assertEquals(spec.description.output[0].name, 'target')
        self.assertEquals(spec.description.output[0].type.WhichOneof('Type'), 'int64Type')
        for input_type in spec.description.input:
            self.assertEquals(input_type.type.WhichOneof('Type'), 'doubleType')
        self.assertEqual(sorted(input_names), sorted(map(lambda x: x.name, spec.description.input)))
        self.assertEquals(len(spec.pipelineClassifier.pipeline.models), 2)
        tr = spec.pipelineClassifier.pipeline.models[-1].treeEnsembleClassifier.treeEnsemble
        self.assertIsNotNone(tr)
        self.assertEquals(len(tr.nodes), 1048)

    def test_conversion_bad_inputs(self):
        if False:
            return 10
        with self.assertRaises(Exception):
            model = RandomForestClassifier()
            spec = skl_converter.convert(model, 'data', 'out')
        from sklearn.preprocessing import OneHotEncoder
        with self.assertRaises(Exception):
            model = OneHotEncoder()
            spec = skl_converter.convert(model, 'data', 'out')

@unittest.skipIf(not _HAS_SKLEARN, 'Missing sklearn. Skipping tests.')
class RandomForestMultiClassClassifierScikitTest(unittest.TestCase):
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
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        scikit_data = load_boston()
        scikit_model = RandomForestClassifier(random_state=1)
        t = scikit_data.target
        target = np.digitize(t, np.histogram(t)[1]) - 1
        scikit_model.fit(scikit_data.data, target)
        self.scikit_data = scikit_data
        self.target = target
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
        self.assertIsNotNone(spec.treeEnsembleClassifier)
        self.assertEquals(spec.description.predictedFeatureName, 'target')
        self.assertEquals(len(spec.description.output), 2)
        self.assertEquals(spec.description.output[0].name, 'target')
        self.assertEquals(spec.description.output[0].type.WhichOneof('Type'), 'int64Type')
        for input_type in spec.description.input:
            self.assertEquals(input_type.type.WhichOneof('Type'), 'doubleType')
        self.assertEqual(sorted(input_names), sorted(map(lambda x: x.name, spec.description.input)))
        self.assertEquals(len(spec.pipelineClassifier.pipeline.models), 2)
        tr = spec.pipelineClassifier.pipeline.models[-1].treeEnsembleClassifier.treeEnsemble
        self.assertIsNotNone(tr)
        self.assertEquals(len(tr.nodes), 2970)

    def test_conversion_bad_inputs(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(Exception):
            model = RandomForestClassifier()
            spec = skl_converter.convert(model, 'data', 'out')
        with self.assertRaises(Exception):
            from sklearn.preprocessing import OneHotEncoder
            model = OneHotEncoder()
            spec = skl_converter.convert(model, 'data', 'out')