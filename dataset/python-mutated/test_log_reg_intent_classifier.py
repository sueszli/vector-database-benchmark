from __future__ import unicode_literals
import io
import sys
from builtins import str
from unittest import skipIf
from checksumdir import dirhash
from mock import patch
from snips_nlu.common.io_utils import temp_dir
from snips_nlu.constants import INTENTS, LANGUAGE_EN, RES_INTENT_NAME, RES_PROBA, UTTERANCES
from snips_nlu.dataset import Dataset
from snips_nlu.exceptions import NotTrained
from snips_nlu.intent_classifier import LogRegIntentClassifier
from snips_nlu.intent_classifier.featurizer import Featurizer
from snips_nlu.intent_classifier.log_reg_classifier_utils import text_to_utterance
from snips_nlu.pipeline.configs import LogRegIntentClassifierConfig
from snips_nlu.result import intent_classification_result
from snips_nlu.tests.utils import FixtureTest, get_empty_dataset

def get_mocked_augment_utterances(dataset, intent_name, language, min_utterances, capitalization_ratio, add_builtin_entities_examples, resources, random_state):
    if False:
        while True:
            i = 10
    return dataset[INTENTS][intent_name][UTTERANCES]

class TestLogRegIntentClassifier(FixtureTest):

    def test_should_get_intent(self):
        if False:
            return 10
        dataset_stream = io.StringIO("\n---\ntype: intent\nname: my_first_intent\nutterances:\n- how are you\n- hello how are you?\n- what's up\n\n---\ntype: intent\nname: my_second_intent\nutterances:\n- what is the weather today ?\n- does it rain\n- will it rain tomorrow")
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        classifier = LogRegIntentClassifier(random_state=42).fit(dataset)
        text = 'hey how are you doing ?'
        res = classifier.get_intent(text)
        intent = res[RES_INTENT_NAME]
        self.assertEqual('my_first_intent', intent)

    def test_should_get_none_intent_when_empty_input(self):
        if False:
            while True:
                i = 10
        dataset_stream = io.StringIO("\n---\ntype: intent\nname: my_first_intent\nutterances:\n- how are you\n- hello how are you?\n- what's up\n\n---\ntype: intent\nname: my_second_intent\nutterances:\n- what is the weather today ?\n- does it rain\n- will it rain tomorrow")
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        classifier = LogRegIntentClassifier().fit(dataset)
        text = ''
        result = classifier.get_intent(text)
        self.assertEqual(intent_classification_result(None, 1.0), result)

    def test_should_get_intent_when_filter(self):
        if False:
            return 10
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: MakeTea\nutterances:\n- make me a cup of tea\n- i want two cups of tea please\n- can you prepare one cup of tea ?\n\n---\ntype: intent\nname: MakeCoffee\nutterances:\n- make me a cup of coffee please\n- brew two cups of coffee\n- can you prepare one cup of coffee')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        classifier = LogRegIntentClassifier(random_state=42).fit(dataset)
        text1 = 'Make me two cups of tea'
        res1 = classifier.get_intent(text1, ['MakeCoffee', 'MakeTea'])
        text2 = 'Make me two cups of tea'
        res2 = classifier.get_intent(text2, ['MakeCoffee'])
        text3 = 'bla bla bla'
        res3 = classifier.get_intent(text3, ['MakeCoffee'])
        self.assertEqual('MakeTea', res1[RES_INTENT_NAME])
        self.assertEqual('MakeCoffee', res2[RES_INTENT_NAME])
        self.assertEqual(None, res3[RES_INTENT_NAME])

    def test_should_raise_when_not_fitted(self):
        if False:
            i = 10
            return i + 15
        intent_classifier = LogRegIntentClassifier()
        self.assertFalse(intent_classifier.fitted)
        with self.assertRaises(NotTrained):
            intent_classifier.get_intent('foobar')

    def test_should_get_none_intent_when_empty_dataset(self):
        if False:
            print('Hello World!')
        dataset = get_empty_dataset(LANGUAGE_EN)
        classifier = LogRegIntentClassifier().fit(dataset)
        text = 'this is a dummy query'
        intent = classifier.get_intent(text)
        expected_intent = intent_classification_result(None, 1.0)
        self.assertEqual(intent, expected_intent)

    def test_should_get_intents(self):
        if False:
            return 10
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: intent1\nutterances:\n  - yala yili\n\n---\ntype: intent\nname: intent2\nutterances:\n  - yala yili yulu\n\n---\ntype: intent\nname: intent3\nutterances:\n  - yili yulu yele')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        classifier = LogRegIntentClassifier(random_state=42).fit(dataset)
        text = 'yala yili yulu'
        results = classifier.get_intents(text)
        intents = [res[RES_INTENT_NAME] for res in results]
        expected_intents = ['intent2', 'intent1', 'intent3', None]
        self.assertEqual(expected_intents, intents)

    def test_should_get_intents_when_empty_dataset(self):
        if False:
            while True:
                i = 10
        dataset = get_empty_dataset(LANGUAGE_EN)
        classifier = LogRegIntentClassifier().fit(dataset)
        text = 'this is a dummy query'
        results = classifier.get_intents(text)
        expected_results = [{RES_INTENT_NAME: None, RES_PROBA: 1.0}]
        self.assertEqual(expected_results, results)

    def test_should_get_intents_when_empty_input(self):
        if False:
            for i in range(10):
                print('nop')
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: intent1\nutterances:\n  - foo bar\n\n---\ntype: intent\nname: intent2\nutterances:\n  - lorem ipsum')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        classifier = LogRegIntentClassifier().fit(dataset)
        text = ''
        results = classifier.get_intents(text)
        expected_results = [{RES_INTENT_NAME: None, RES_PROBA: 1.0}, {RES_INTENT_NAME: 'intent1', RES_PROBA: 0.0}, {RES_INTENT_NAME: 'intent2', RES_PROBA: 0.0}]
        self.assertEqual(expected_results, results)

    def test_should_be_serializable(self):
        if False:
            i = 10
            return i + 15
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: intent1\nutterances:\n  - foo bar\n\n---\ntype: intent\nname: intent2\nutterances:\n  - lorem ipsum')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        intent_classifier = LogRegIntentClassifier(random_state=42).fit(dataset)
        coeffs = intent_classifier.classifier.coef_.tolist()
        intercept = intent_classifier.classifier.intercept_.tolist()
        t_ = intent_classifier.classifier.t_
        intent_classifier.persist(self.tmp_file_path)
        intent_list = ['intent1', 'intent2', None]
        expected_dict = {'config': LogRegIntentClassifierConfig().to_dict(), 'coeffs': coeffs, 'intercept': intercept, 't_': t_, 'intent_list': intent_list, 'featurizer': 'featurizer'}
        metadata = {'unit_name': 'log_reg_intent_classifier'}
        self.assertJsonContent(self.tmp_file_path / 'metadata.json', metadata)
        self.assertJsonContent(self.tmp_file_path / 'intent_classifier.json', expected_dict)
        featurizer_path = self.tmp_file_path / 'featurizer'
        self.assertTrue(featurizer_path.exists())
        self.assertTrue(featurizer_path.is_dir())

    def test_should_be_deserializable(self):
        if False:
            return 10
        featurizer = Featurizer()
        featurizer_path = self.tmp_file_path / 'featurizer'
        self.tmp_file_path.mkdir()
        featurizer.persist(featurizer_path)
        intent_list = ['MakeCoffee', 'MakeTea', None]
        coeffs = [[1.23, 4.5], [6.7, 8.9], [1.01, 2.345]]
        intercept = [0.34, 0.41, -0.98]
        t_ = 701.0
        config = LogRegIntentClassifierConfig().to_dict()
        classifier_dict = {'coeffs': coeffs, 'intercept': intercept, 't_': t_, 'intent_list': intent_list, 'config': config, 'featurizer': 'featurizer'}
        metadata = {'unit_name': 'log_reg_intent_classifier'}
        self.writeJsonContent(self.tmp_file_path / 'metadata.json', metadata)
        self.writeJsonContent(self.tmp_file_path / 'intent_classifier.json', classifier_dict)
        classifier = LogRegIntentClassifier.from_path(self.tmp_file_path)
        self.assertEqual(classifier.intent_list, intent_list)
        self.assertIsNotNone(classifier.featurizer)
        self.assertListEqual(classifier.classifier.coef_.tolist(), coeffs)
        self.assertListEqual(classifier.classifier.intercept_.tolist(), intercept)
        self.assertDictEqual(classifier.config.to_dict(), config)

    def test_should_get_intent_after_deserialization(self):
        if False:
            i = 10
            return i + 15
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: MakeTea\nutterances:\n- make me a cup of tea\n- i want two cups of tea please\n- can you prepare one cup of tea ?\n\n---\ntype: intent\nname: MakeCoffee\nutterances:\n- make me a cup of coffee please\n- brew two cups of coffee\n- can you prepare one cup of coffee')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        shared = self.get_shared_data(dataset)
        classifier = LogRegIntentClassifier(**shared).fit(dataset)
        classifier.persist(self.tmp_file_path)
        loaded_classifier = LogRegIntentClassifier.from_path(self.tmp_file_path, **shared)
        result = loaded_classifier.get_intent('Make me two cups of tea')
        expected_intent = 'MakeTea'
        self.assertEqual(expected_intent, result[RES_INTENT_NAME])

    def test_should_be_serializable_into_bytearray(self):
        if False:
            i = 10
            return i + 15
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: MakeTea\nutterances:\n- make me a cup of tea\n- i want two cups of tea please\n- can you prepare one cup of tea ?\n\n---\ntype: intent\nname: MakeCoffee\nutterances:\n- make me a cup of coffee please\n- brew two cups of coffee\n- can you prepare one cup of coffee')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        shared = self.get_shared_data(dataset)
        intent_classifier = LogRegIntentClassifier(**shared).fit(dataset)
        intent_classifier_bytes = intent_classifier.to_byte_array()
        loaded_classifier = LogRegIntentClassifier.from_byte_array(intent_classifier_bytes, **shared)
        result = loaded_classifier.get_intent('make me two cups of tea')
        expected_intent = 'MakeTea'
        self.assertEqual(expected_intent, result[RES_INTENT_NAME])

    @patch('snips_nlu.intent_classifier.log_reg_classifier.build_training_data')
    def test_empty_vocabulary_should_fit_and_return_none_intent(self, mocked_build_training):
        if False:
            while True:
                i = 10
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: dummy_intent_1\nutterances:\n  - "[dummy_slot_name:dummy_entity_1](...)"\n  \n---\ntype: entity\nname: dummy_entity_1\nautomatically_extensible: true\nuse_synonyms: false\nmatching_strictness: 1.0\nvalues:\n  - ...\n')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        text = ' '
        noise_size = 6
        utterances = [text] + [text] * noise_size
        utterances = [text_to_utterance(t) for t in utterances]
        labels = [0] + [1] * noise_size
        intent_list = ['dummy_intent_1', None]
        mocked_build_training.return_value = (utterances, labels, intent_list)
        intent_classifier = LogRegIntentClassifier().fit(dataset)
        intent = intent_classifier.get_intent('no intent there')
        self.assertEqual(intent_classification_result(None, 1.0), intent)

    def test_log_activation_weights(self):
        if False:
            i = 10
            return i + 15
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: intent1\nutterances:\n  - foo bar\n\n---\ntype: intent\nname: intent2\nutterances:\n  - lorem ipsum')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        shared = self.get_shared_data(dataset)
        intent_classifier = LogRegIntentClassifier(**shared)
        text = 'yo'
        utterances = [text_to_utterance(text)]
        self.assertIsNone(intent_classifier.log_activation_weights(text, None))
        intent_classifier.fit(dataset)
        x = intent_classifier.featurizer.transform(utterances)[0]
        log = intent_classifier.log_activation_weights(text, x, top_n=42)
        self.assertIsInstance(log, str)
        self.assertIn('Top 42', log)

    def test_log_best_features(self):
        if False:
            for i in range(10):
                print('nop')
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: intent1\nutterances:\n  - foo bar\n\n---\ntype: intent\nname: intent2\nutterances:\n  - lorem ipsum')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        shared = self.get_shared_data(dataset)
        intent_classifier = LogRegIntentClassifier(**shared)
        self.assertIsNone(intent_classifier.log_best_features(20))
        intent_classifier.fit(dataset)
        log = intent_classifier.log_best_features(20)
        self.assertIsInstance(log, str)
        self.assertIn('Top 20', log)

    @skipIf(sys.version_info[0:2] < (3, 5), 'The bug fixed here https://github.com/scikit-learn/scikit-learn/pull/13422 is available for scikit-learn>=0.21.0 in which the support for Python<=3.4 has been dropped')
    def test_training_should_be_reproducible(self):
        if False:
            for i in range(10):
                print('nop')
        random_state = 40
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: MakeTea\nutterances:\n- make me a [beverage_temperature:Temperature](hot) cup of tea\n- make me [number_of_cups:snips/number](five) tea cups\n\n---\ntype: intent\nname: MakeCoffee\nutterances:\n- make me [number_of_cups:snips/number](one) cup of coffee please\n- brew [number_of_cups] cups of coffee')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        classifier1 = LogRegIntentClassifier(random_state=random_state)
        classifier1.fit(dataset)
        classifier2 = LogRegIntentClassifier(random_state=random_state)
        classifier2.fit(dataset)
        with temp_dir() as tmp_dir:
            dir_classifier1 = tmp_dir / 'classifier1'
            dir_classifier2 = tmp_dir / 'classifier2'
            classifier1.persist(dir_classifier1)
            classifier2.persist(dir_classifier2)
            hash1 = dirhash(str(dir_classifier1), 'sha256')
            hash2 = dirhash(str(dir_classifier2), 'sha256')
            self.assertEqual(hash1, hash2)