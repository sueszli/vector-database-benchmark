from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import unittest
import tempfile
import turicreate as tc
from turicreate.toolkits._main import ToolkitError
from turicreate.toolkits._internal_utils import _mac_ver
from . import util as test_util
import sys
if sys.version_info.major == 3:
    unittest.TestCase.assertItemsEqual = unittest.TestCase.assertCountEqual

class TextClassifierTest(unittest.TestCase):
    """
    Unit test class for an already trained model.
    """

    @classmethod
    def setUpClass(self):
        if False:
            return 10
        text = ['hello friend', 'how exciting', 'mostly exciting', 'hello again']
        score = [0, 1, 1, 0]
        self.docs = tc.SFrame({'text': text, 'score': score})
        self.features = ['text']
        self.num_features = 1
        self.target = 'score'
        self.method = 'bow-logistic'
        self.model = tc.text_classifier.create(self.docs, target=self.target, features=self.features, method='auto')
        self.num_examples = 4

    def test__list_fields(self):
        if False:
            while True:
                i = 10
        '\n        Check the model list fields method.\n        '
        correct_fields = ['classifier', 'features', 'num_features', 'method', 'num_examples', 'target']
        self.assertItemsEqual(self.model._list_fields(), correct_fields)

    def test_get(self):
        if False:
            while True:
                i = 10
        "\n        Check the various 'get' methods against known answers for each field.\n        "
        correct_fields = {'features': self.features, 'num_features': self.num_features, 'target': self.target, 'method': self.method, 'num_examples': self.num_examples}
        print(self.model)
        for (field, ans) in correct_fields.items():
            self.assertEqual(self.model._get(field), ans, '{} failed'.format(field))

    def test_model_access(self):
        if False:
            i = 10
            return i + 15
        m = self.model.classifier
        self.assertTrue(isinstance(m, tc.classifier.logistic_classifier.LogisticClassifier))

    def test_summaries(self):
        if False:
            print('Hello World!')
        '\n        Unit test for __repr__, __str__, and model summary methods; should fail\n        if they raise an Exception.\n        '
        ans = str(self.model)
        print(self.model)
        self.model.summary()

    def test_evaluate(self):
        if False:
            while True:
                i = 10
        '\n        Tests for evaluating the model.\n        '
        self.model.evaluate(self.docs)

    def test_export_coreml(self):
        if False:
            while True:
                i = 10
        import platform
        import coremltools
        filename = tempfile.NamedTemporaryFile(suffix='.mlmodel').name
        self.model.export_coreml(filename)
        coreml_model = coremltools.models.MLModel(filename)
        metadata = coreml_model.user_defined_metadata
        self.assertEqual(metadata['com.github.apple.turicreate.version'], tc.__version__)
        self.assertEqual(metadata['com.github.apple.os.platform'], platform.platform())
        self.assertEqual(metadata['type'], self.model.__class__.__name__)
        expected_result = 'Text classifier created by Turi Create (version %s)' % tc.__version__
        self.assertEquals(expected_result, coreml_model.short_description)

    @unittest.skipIf(_mac_ver() < (10, 13), 'Only supported on macOS 10.13+')
    def test_export_coreml_with_predict(self):
        if False:
            while True:
                i = 10
        filename = tempfile.NamedTemporaryFile(suffix='.mlmodel').name
        self.model.export_coreml(filename)
        preds = self.model.predict(self.docs, output_type='probability_vector')
        import coremltools
        coreml_model = coremltools.models.MLModel(filename)
        coreml_preds = coreml_model.predict({'text': {'hello': 1, 'friend': 1}})
        self.assertAlmostEqual(preds[0][0], coreml_preds['scoreProbability'][0])
        self.assertAlmostEqual(preds[0][1], coreml_preds['scoreProbability'][1])

    def test_save_and_load(self):
        if False:
            i = 10
            return i + 15
        '\n        Ensure that model saving and loading retains all model information.\n        '
        with test_util.TempDirectory() as f:
            self.model.save(f)
            self.model = tc.load_model(f)
            loaded_model = tc.load_model(f)
            self.test__list_fields()
            print('Saved model list fields passed')
            self.test_get()
            print('Saved model get passed')
            self.test_summaries()
            print('Saved model summaries passed')

class TextClassifierCreateTests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if False:
            for i in range(10):
                print('nop')
        self.data = tc.SFrame({'rating': [1, 5, 2, 3, 3, 5], 'place': ['a', 'a', 'b', 'b', 'b', 'c'], 'text': ['The burrito was terrible and awful and I hated it', 'I will come here every day of my life because the burrito is awesome and delicious', 'Meh, the waiter died while serving us. Other than that the experience was OK, but the burrito was not great.', 'Mediocre burrito. Nothing much else to report.', 'My dad works here, so I guess I have to kinda like it. Hate the burrito, though.', 'Love it! Mexican restaurant of my dreams and a burrito from the gods.']})
        self.rating_column = 'rating'
        self.features = ['text']
        self.keywords = ['burrito', 'dad']
        self.model = tc.text_classifier.create(self.data, target='rating', features=self.features)

    def test_sentiment_create_no_features(self):
        if False:
            return 10
        model = tc.text_classifier.create(self.data, target='rating')
        self.assertTrue(isinstance(model, tc.text_classifier.TextClassifier))

    def test_sentiment_create_string_target(self):
        if False:
            for i in range(10):
                print('nop')
        data_str = self.data[:]
        data_str['rating'] = data_str['rating'].astype(str)
        model = tc.text_classifier.create(data_str, target='rating')
        self.assertTrue(isinstance(model, tc.text_classifier.TextClassifier))

    def test_invalid_data_set(self):
        if False:
            print('Hello World!')
        a = tc.SArray(['str', None])
        b = tc.SArray(['str', 'str'])
        sf = tc.SFrame({'a': a, 'b': b})
        with self.assertRaises(ToolkitError):
            tc.text_classifier.create(sf, target='a', features=['b'], word_count_threshold=1)
        sf = tc.SFrame({'b': a, 'a': b})
        with self.assertRaises(ToolkitError):
            tc.text_classifier.create(sf, target='b', features=['a'], word_count_threshold=1)

    def test_validation_set(self):
        if False:
            print('Hello World!')
        train = self.data
        valid = self.data
        model = tc.text_classifier.create(train, target='rating', validation_set=valid)
        self.assertTrue('Validation Accuracy' in model.classifier.progress.column_names())
        model = tc.text_classifier.create(train, target='rating', validation_set=None)
        self.assertTrue('Validation Accuracy' not in model.classifier.progress.column_names())
        big_data = train.append(tc.SFrame({'rating': [5] * 100, 'place': ['d'] * 100, 'text': ['large enough data for %5 percent validation split to activate'] * 100}))
        model = tc.text_classifier.create(big_data, target='rating', validation_set='auto')
        self.assertTrue('Validation Accuracy' in model.classifier.progress.column_names())
        with self.assertRaises(TypeError):
            tc.text_classifier.create(train, target='rating', validation_set='wrong')
        with self.assertRaises(TypeError):
            tc.text_classifier.create(train, target='rating', validation_set=5)

    def test_sentiment_classifier(self):
        if False:
            for i in range(10):
                print('nop')
        m = self.model
        self.assertEqual(m.classifier.classes, [1, 2, 3, 5])

    def test_predict(self):
        if False:
            while True:
                i = 10
        m = self.model
        preds = m.predict(self.data)
        self.assertTrue(isinstance(preds, tc.SArray))
        self.assertEqual(preds.dtype, int)

    def test_classify(self):
        if False:
            print('Hello World!')
        m = self.model
        preds = m.classify(self.data)
        self.assertTrue(isinstance(preds, tc.SFrame))
        self.assertEqual(preds.column_names(), ['class', 'probability'])

    def test_not_sframe_create_error(self):
        if False:
            print('Hello World!')
        dataset = {'rating': [1, 5], 'text': ['this is bad', 'this is good']}
        try:
            tc.text_classifier.create(dataset, 'rating', features=['text'])
        except ToolkitError as t:
            exception_msg = t.args[0]
            self.assertTrue(exception_msg.startswith('Input dataset is not an SFrame. '))
        else:
            self.fail('This should have thrown an exception')

class TextClassifierCreateBadValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if False:
            print('Hello World!')
        self.data = tc.SFrame({'rating': [1, 5, 2, 3], 'place': ['a', 'a', 'b', 'b'], 'text': ['The burrito was terrible and awful and I hated it', 'I will come here a lot', '......', '']})
        self.rating_column = 'rating'
        self.features = ['text']
        self.keywords = ['burrito', 'dad']

    def test_create(self):
        if False:
            for i in range(10):
                print('nop')
        model = tc.text_classifier.create(self.data, target=self.rating_column, features=self.features)
        self.assertTrue(model is not None)