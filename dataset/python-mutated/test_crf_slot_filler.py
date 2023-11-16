from __future__ import unicode_literals
import io
import os
from builtins import range
from pathlib import Path
from unittest import skipIf
from mock import MagicMock, PropertyMock
from sklearn_crfsuite import CRF
from snips_nlu.constants import DATA, END, ENTITY, LANGUAGE_EN, SLOT_NAME, START, TEXT, RANDOM_STATE
from snips_nlu.dataset import Dataset
from snips_nlu.entity_parser import CustomEntityParserUsage
from snips_nlu.exceptions import NotTrained
from snips_nlu.pipeline.configs import CRFSlotFillerConfig
from snips_nlu.preprocessing import tokenize, Token
from snips_nlu.result import unresolved_slot
from snips_nlu.slot_filler.crf_slot_filler import CRFSlotFiller, _ensure_safe, _encode_tag, CRF_MODEL_FILENAME
from snips_nlu.slot_filler.crf_utils import TaggingScheme
from snips_nlu.slot_filler.feature_factory import IsDigitFactory, NgramFactory, ShapeNgramFactory
from snips_nlu.tests.utils import FixtureTest, TEST_PATH

class TestCRFSlotFiller(FixtureTest):

    def test_should_get_slots(self):
        if False:
            for i in range(10):
                print('nop')
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: MakeTea\nutterances:\n- make me [number_of_cups:snips/number](five) cups of tea\n- please I want [number_of_cups](two) cups of tea')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        shared = self.get_shared_data(dataset)
        shared[RANDOM_STATE] = 42
        slot_filler = CRFSlotFiller(**shared)
        intent = 'MakeTea'
        slot_filler.fit(dataset, intent)
        slots = slot_filler.get_slots('make me two cups of tea')
        expected_slots = [unresolved_slot(match_range={START: 8, END: 11}, value='two', entity='snips/number', slot_name='number_of_cups')]
        self.assertListEqual(slots, expected_slots)

    def test_should_get_builtin_slots(self):
        if False:
            return 10
        dataset_stream = io.StringIO("\n---\ntype: intent\nname: GetWeather\nutterances:\n- what is the weather [datetime:snips/datetime](at 9pm)\n- what's the weather in [location:weather_location](berlin)\n- What's the weather in [location](tokyo) [datetime](this weekend)?\n- Can you tell me the weather [datetime] please ?\n- what is the weather forecast [datetime] in [location](paris)")
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        intent = 'GetWeather'
        shared = self.get_shared_data(dataset)
        shared[RANDOM_STATE] = 42
        slot_filler = CRFSlotFiller(**shared)
        slot_filler.fit(dataset, intent)
        slots = slot_filler.get_slots('Give me the weather at 9pm in Paris')
        expected_slots = [unresolved_slot(match_range={START: 20, END: 26}, value='at 9pm', entity='snips/datetime', slot_name='datetime'), unresolved_slot(match_range={START: 30, END: 35}, value='Paris', entity='weather_location', slot_name='location')]
        self.assertListEqual(expected_slots, slots)

    def test_should_get_sub_builtin_slots(self):
        if False:
            for i in range(10):
                print('nop')
        dataset_stream = io.StringIO("\n---\ntype: intent\nname: PlanBreak\nutterances:\n- 'I want to leave from [start:snips/datetime](tomorrow) until \n  [end:snips/datetime](next thursday)'\n- find me something from [start](9am) to [end](12pm)\n- I need a break from [start](2pm) until [end](4pm)\n- Can you suggest something from [start](april 4th) until [end](april 6th) ?\n- find an activity from [start](6pm) to [end](8pm)\n- Book me a trip from [start](this friday) to [end](next tuesday)")
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        intent = 'PlanBreak'
        shared = self.get_shared_data(dataset)
        shared[RANDOM_STATE] = 42
        slot_filler = CRFSlotFiller(**shared)
        slot_filler.fit(dataset, intent)
        slots = slot_filler.get_slots('Find me a plan from 5pm to 6pm')
        expected_slots = [unresolved_slot(match_range={START: 20, END: 23}, value='5pm', entity='snips/datetime', slot_name='start'), unresolved_slot(match_range={START: 27, END: 30}, value='6pm', entity='snips/datetime', slot_name='end')]
        self.assertListEqual(expected_slots, slots)

    def test_should_not_use_crf_when_dataset_with_no_slots(self):
        if False:
            print('Hello World!')
        dataset = {'language': 'en', 'intents': {'intent1': {'utterances': [{'data': [{'text': 'This is an utterance without slots'}]}]}}, 'entities': {}}
        slot_filler = CRFSlotFiller(**self.get_shared_data(dataset))
        mock_compute_features = MagicMock()
        slot_filler.compute_features = mock_compute_features
        slot_filler.fit(dataset, 'intent1')
        slots = slot_filler.get_slots('This is an utterance without slots')
        mock_compute_features.assert_not_called()
        self.assertListEqual([], slots)

    def test_should_compute_sequence_probability_when_no_slots(self):
        if False:
            return 10
        dataset = {'language': 'en', 'intents': {'intent1': {'utterances': [{'data': [{'text': 'This is an utterance without slots'}]}]}}, 'entities': {}}
        shared = self.get_shared_data(dataset)
        slot_filler = CRFSlotFiller(**shared).fit(dataset, 'intent1')
        tokens = tokenize('hello world foo bar', 'en')
        res1 = slot_filler.get_sequence_probability(tokens, ['O', 'O', 'O', 'O'])
        res2 = slot_filler.get_sequence_probability(tokens, ['O', 'O', 'B-location', 'O'])
        self.assertEqual(1.0, res1)
        self.assertEqual(0.0, res2)

    def test_should_parse_naughty_strings(self):
        if False:
            print('Hello World!')
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: my_intent\nutterances:\n- this is [entity1](my first entity)')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        naughty_strings_path = TEST_PATH / 'resources' / 'naughty_strings.txt'
        with naughty_strings_path.open(encoding='utf8') as f:
            naughty_strings = [line.strip('\n') for line in f.readlines()]
        shared = self.get_shared_data(dataset)
        slot_filler = CRFSlotFiller(**shared).fit(dataset, 'my_intent')
        for s in naughty_strings:
            with self.fail_if_exception('Naughty string crashes'):
                slot_filler.get_slots(s)

    def test_should_not_get_slots_when_not_fitted(self):
        if False:
            return 10
        slot_filler = CRFSlotFiller()
        self.assertFalse(slot_filler.fitted)
        with self.assertRaises(NotTrained):
            slot_filler.get_slots('foobar')

    def test_should_not_get_sequence_probability_when_not_fitted(self):
        if False:
            print('Hello World!')
        slot_filler = CRFSlotFiller()
        with self.assertRaises(NotTrained):
            slot_filler.get_sequence_probability(tokens=[], labels=[])

    def test_should_not_log_weights_when_not_fitted(self):
        if False:
            i = 10
            return i + 15
        slot_filler = CRFSlotFiller()
        with self.assertRaises(NotTrained):
            slot_filler.log_weights()

    def test_refit(self):
        if False:
            print('Hello World!')
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: my_intent\nutterances:\n- this is [entity1](my first entity)')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        updated_dataset_stream = io.StringIO('\n---\ntype: intent\nname: my_intent\nutterances:\n- this is [entity1](my first entity)\n- this is [entity1](my first entity) again')
        updated_dataset = Dataset.from_yaml_files('en', [updated_dataset_stream]).json
        config = CRFSlotFillerConfig(feature_factory_configs=[{'args': {'common_words_gazetteer_name': 'top_10000_words_stemmed', 'use_stemming': True, 'n': 1}, 'factory_name': 'ngram', 'offsets': [-2, -1, 0, 1, 2]}])
        slot_filler = CRFSlotFiller(config).fit(dataset, 'my_intent')
        slot_filler.fit(updated_dataset, 'my_intent')

    def test_should_fit_with_naughty_strings_no_tags(self):
        if False:
            while True:
                i = 10
        naughty_strings_path = TEST_PATH / 'resources' / 'naughty_strings.txt'
        with naughty_strings_path.open(encoding='utf8') as f:
            naughty_strings = [line.strip('\n') for line in f.readlines()]
        utterances = [{DATA: [{TEXT: naughty_string}]} for naughty_string in naughty_strings]
        naughty_dataset = {'intents': {'naughty_intent': {'utterances': utterances}}, 'entities': dict(), 'language': 'en'}
        with self.fail_if_exception('Naughty string crashes'):
            shared = self.get_shared_data(naughty_dataset)
            CRFSlotFiller(**shared).fit(naughty_dataset, 'naughty_intent')

    def test_should_fit_and_parse_with_non_ascii_tags(self):
        if False:
            for i in range(10):
                print('nop')
        inputs = ('string%s' % i for i in range(10))
        utterances = [{DATA: [{TEXT: string, ENTITY: 'non_ascìi_entïty', SLOT_NAME: 'non_ascìi_slöt'}]} for string in inputs]
        naughty_dataset = {'intents': {'naughty_intent': {'utterances': utterances}}, 'entities': {'non_ascìi_entïty': {'use_synonyms': False, 'automatically_extensible': True, 'data': [], 'matching_strictness': 1.0}}, 'language': 'en'}
        with self.fail_if_exception('Naughty string make NLU crash'):
            shared = self.get_shared_data(naughty_dataset)
            slot_filler = CRFSlotFiller(**shared)
            slot_filler.fit(naughty_dataset, 'naughty_intent')
            slots = slot_filler.get_slots('string0')
            expected_slot = {'entity': 'non_ascìi_entïty', 'range': {'start': 0, 'end': 7}, 'slotName': u'non_ascìi_slöt', 'value': u'string0'}
            self.assertListEqual([expected_slot], slots)

    def test_should_get_slots_after_deserialization(self):
        if False:
            for i in range(10):
                print('nop')
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: MakeTea\nutterances:\n- make me [number_of_cups:snips/number](one) cup of tea\n- i want [number_of_cups] cups of tea please\n- can you prepare [number_of_cups] cups of tea ?')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        intent = 'MakeTea'
        shared = self.get_shared_data(dataset)
        shared[RANDOM_STATE] = 42
        slot_filler = CRFSlotFiller(**shared)
        slot_filler.fit(dataset, intent)
        slot_filler.persist(self.tmp_file_path)
        deserialized_slot_filler = CRFSlotFiller.from_path(self.tmp_file_path, **shared)
        slots = deserialized_slot_filler.get_slots('make me two cups of tea')
        expected_slots = [unresolved_slot(match_range={START: 8, END: 11}, value='two', entity='snips/number', slot_name='number_of_cups')]
        self.assertListEqual(expected_slots, slots)

    def test_should_be_serializable_before_fit(self):
        if False:
            while True:
                i = 10
        features_factories = [{'factory_name': ShapeNgramFactory.name, 'args': {'n': 1}, 'offsets': [0]}, {'factory_name': IsDigitFactory.name, 'args': {}, 'offsets': [-1, 0]}]
        config = CRFSlotFillerConfig(tagging_scheme=TaggingScheme.BILOU, feature_factory_configs=features_factories)
        slot_filler = CRFSlotFiller(config)
        slot_filler.persist(self.tmp_file_path)
        metadata_path = self.tmp_file_path / 'metadata.json'
        self.assertJsonContent(metadata_path, {'unit_name': 'crf_slot_filler'})
        expected_slot_filler_dict = {'crf_model_file': None, 'language_code': None, 'config': config.to_dict(), 'intent': None, 'slot_name_mapping': None}
        slot_filler_path = self.tmp_file_path / 'slot_filler.json'
        self.assertJsonContent(slot_filler_path, expected_slot_filler_dict)

    def test_should_be_deserializable_before_fit(self):
        if False:
            print('Hello World!')
        features_factories = [{'factory_name': ShapeNgramFactory.name, 'args': {'n': 1}, 'offsets': [0]}, {'factory_name': IsDigitFactory.name, 'args': {}, 'offsets': [-1, 0]}]
        slot_filler_config = CRFSlotFillerConfig(feature_factory_configs=features_factories)
        slot_filler_dict = {'unit_name': 'crf_slot_filler', 'crf_model_file': None, 'language_code': None, 'intent': None, 'slot_name_mapping': None, 'config': slot_filler_config.to_dict()}
        metadata = {'unit_name': 'crf_slot_filler'}
        self.tmp_file_path.mkdir()
        self.writeJsonContent(self.tmp_file_path / 'metadata.json', metadata)
        self.writeJsonContent(self.tmp_file_path / 'slot_filler.json', slot_filler_dict)
        slot_filler = CRFSlotFiller.from_path(self.tmp_file_path)
        expected_features_factories = [{'factory_name': ShapeNgramFactory.name, 'args': {'n': 1}, 'offsets': [0]}, {'factory_name': IsDigitFactory.name, 'args': {}, 'offsets': [-1, 0]}]
        expected_language = None
        expected_config = CRFSlotFillerConfig(feature_factory_configs=expected_features_factories)
        expected_intent = None
        expected_slot_name_mapping = None
        expected_crf_model = None
        self.assertEqual(slot_filler.crf_model, expected_crf_model)
        self.assertEqual(slot_filler.language, expected_language)
        self.assertEqual(slot_filler.intent, expected_intent)
        self.assertEqual(slot_filler.slot_name_mapping, expected_slot_name_mapping)
        self.assertDictEqual(expected_config.to_dict(), slot_filler.config.to_dict())

    def test_should_be_serializable(self):
        if False:
            for i in range(10):
                print('nop')
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: my_intent\nutterances:\n- this is [slot1:entity1](my first entity)\n- this is [slot2:entity2](second_entity)')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        features_factories = [{'factory_name': ShapeNgramFactory.name, 'args': {'n': 1}, 'offsets': [0]}, {'factory_name': IsDigitFactory.name, 'args': {}, 'offsets': [-1, 0]}]
        config = CRFSlotFillerConfig(tagging_scheme=TaggingScheme.BILOU, feature_factory_configs=features_factories)
        shared = self.get_shared_data(dataset)
        slot_filler = CRFSlotFiller(config, **shared)
        intent = 'my_intent'
        slot_filler.fit(dataset, intent=intent)
        slot_filler.persist(self.tmp_file_path)
        metadata_path = self.tmp_file_path / 'metadata.json'
        self.assertJsonContent(metadata_path, {'unit_name': 'crf_slot_filler'})
        self.assertTrue((self.tmp_file_path / CRF_MODEL_FILENAME).exists())
        expected_feature_factories = [{'factory_name': ShapeNgramFactory.name, 'args': {'n': 1, 'language_code': 'en'}, 'offsets': [0]}, {'factory_name': IsDigitFactory.name, 'args': {}, 'offsets': [-1, 0]}]
        expected_config = CRFSlotFillerConfig(tagging_scheme=TaggingScheme.BILOU, feature_factory_configs=expected_feature_factories)
        expected_slot_filler_dict = {'crf_model_file': CRF_MODEL_FILENAME, 'language_code': 'en', 'config': expected_config.to_dict(), 'intent': intent, 'slot_name_mapping': {'slot1': 'entity1', 'slot2': 'entity2'}}
        slot_filler_path = self.tmp_file_path / 'slot_filler.json'
        self.assertJsonContent(slot_filler_path, expected_slot_filler_dict)

    def test_should_be_deserializable(self):
        if False:
            while True:
                i = 10
        language = LANGUAGE_EN
        feature_factories = [{'factory_name': ShapeNgramFactory.name, 'args': {'n': 1, 'language_code': language}, 'offsets': [0]}, {'factory_name': IsDigitFactory.name, 'args': {}, 'offsets': [-1, 0]}]
        slot_filler_config = CRFSlotFillerConfig(feature_factory_configs=feature_factories)
        slot_filler_dict = {'unit_name': 'crf_slot_filler', 'crf_model_file': 'foobar.crfsuite', 'language_code': 'en', 'intent': 'dummy_intent_1', 'slot_name_mapping': {'dummy_intent_1': {'dummy_slot_name': 'dummy_entity_1'}}, 'config': slot_filler_config.to_dict()}
        metadata = {'unit_name': 'crf_slot_filler'}
        self.tmp_file_path.mkdir()
        self.writeJsonContent(self.tmp_file_path / 'metadata.json', metadata)
        self.writeJsonContent(self.tmp_file_path / 'slot_filler.json', slot_filler_dict)
        self.writeFileContent(self.tmp_file_path / 'foobar.crfsuite', 'foo bar')
        slot_filler = CRFSlotFiller.from_path(self.tmp_file_path)
        expected_language = LANGUAGE_EN
        expected_feature_factories = [{'factory_name': ShapeNgramFactory.name, 'args': {'n': 1, 'language_code': language}, 'offsets': [0]}, {'factory_name': IsDigitFactory.name, 'args': {}, 'offsets': [-1, 0]}]
        expected_config = CRFSlotFillerConfig(feature_factory_configs=expected_feature_factories)
        expected_intent = 'dummy_intent_1'
        expected_slot_name_mapping = {'dummy_intent_1': {'dummy_slot_name': 'dummy_entity_1'}}
        self.assertEqual(slot_filler.language, expected_language)
        self.assertEqual(slot_filler.intent, expected_intent)
        self.assertEqual(slot_filler.slot_name_mapping, expected_slot_name_mapping)
        self.assertDictEqual(expected_config.to_dict(), slot_filler.config.to_dict())
        crf_path = Path(slot_filler.crf_model.modelfile.name)
        self.assertFileContent(crf_path, 'foo bar')

    def test_should_be_serializable_when_fitted_without_slots(self):
        if False:
            return 10
        features_factories = [{'factory_name': ShapeNgramFactory.name, 'args': {'n': 1}, 'offsets': [0]}, {'factory_name': IsDigitFactory.name, 'args': {}, 'offsets': [-1, 0]}]
        config = CRFSlotFillerConfig(tagging_scheme=TaggingScheme.BILOU, feature_factory_configs=features_factories)
        dataset = {'language': 'en', 'intents': {'intent1': {'utterances': [{'data': [{'text': 'This is an utterance without slots'}]}]}}, 'entities': {}}
        slot_filler = CRFSlotFiller(config, **self.get_shared_data(dataset))
        slot_filler.fit(dataset, intent='intent1')
        slot_filler.persist(self.tmp_file_path)
        metadata_path = self.tmp_file_path / 'metadata.json'
        self.assertJsonContent(metadata_path, {'unit_name': 'crf_slot_filler'})
        self.assertIsNone(slot_filler.crf_model)

    def test_should_be_deserializable_when_fitted_without_slots(self):
        if False:
            print('Hello World!')
        dataset = {'language': 'en', 'intents': {'intent1': {'utterances': [{'data': [{'text': 'This is an utterance without slots'}]}]}}, 'entities': {}}
        shared = self.get_shared_data(dataset)
        slot_filler = CRFSlotFiller(**shared)
        slot_filler.fit(dataset, intent='intent1')
        slot_filler.persist(self.tmp_file_path)
        loaded_slot_filler = CRFSlotFiller.from_path(self.tmp_file_path, **shared)
        slots = loaded_slot_filler.get_slots('This is an utterance without slots')
        self.assertListEqual([], slots)

    def test_should_be_serializable_into_bytearray(self):
        if False:
            i = 10
            return i + 15
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: MakeTea\nutterances:\n- make me [number_of_cups:snips/number](one) cup of tea\n- i want [number_of_cups] cups of tea please\n- can you prepare [number_of_cups] cups of tea ?')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        shared = self.get_shared_data(dataset)
        slot_filler = CRFSlotFiller(**shared).fit(dataset, 'MakeTea')
        slot_filler_bytes = slot_filler.to_byte_array()
        loaded_slot_filler = CRFSlotFiller.from_byte_array(slot_filler_bytes, **shared)
        slots = loaded_slot_filler.get_slots('make me two cups of tea')
        expected_slots = [unresolved_slot(match_range={START: 8, END: 11}, value='two', entity='snips/number', slot_name='number_of_cups')]
        self.assertListEqual(expected_slots, slots)

    def test_should_compute_features(self):
        if False:
            return 10
        features_factories = [{'factory_name': NgramFactory.name, 'args': {'n': 1, 'use_stemming': False, 'common_words_gazetteer_name': None}, 'offsets': [0], 'drop_out': 0.3}]
        slot_filler_config = CRFSlotFillerConfig(feature_factory_configs=features_factories)
        tokens = tokenize('foo hello world bar', LANGUAGE_EN)
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: my_intent\nutterances:\n- this is [slot1:entity1](my first entity)\n- this is [slot2:entity2](second_entity)')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        shared = self.get_shared_data(dataset, CustomEntityParserUsage.WITHOUT_STEMS)
        slot_filler = CRFSlotFiller(slot_filler_config, **shared)
        slot_filler.fit(dataset, intent='my_intent')
        features_with_drop_out = slot_filler.compute_features(tokens, True)
        expected_features = [{}, {'ngram_1': 'hello'}, {'ngram_1': 'world'}, {'ngram_1': 'bar'}]
        self.assertListEqual(expected_features, features_with_drop_out)

    def test_should_fit_and_parse_empty_intent(self):
        if False:
            print('Hello World!')
        dataset = {'intents': {'dummy_intent': {'utterances': [{'data': [{'text': ' '}]}]}}, 'language': 'en', 'entities': dict()}
        slot_filler = CRFSlotFiller(**self.get_shared_data(dataset))
        slot_filler.fit(dataset, 'dummy_intent')
        slot_filler.get_slots('ya')

    def test_ensure_safe(self):
        if False:
            print('Hello World!')
        unsafe_examples = [([[]], [[]]), ([[], []], [[], []])]
        for (x, y) in unsafe_examples:
            (x, y) = _ensure_safe(x, y)
            model = CRF().fit(x, y)
            model.predict_single([''])

    def test_log_inference_weights(self):
        if False:
            for i in range(10):
                print('nop')
        self.maxDiff = None
        text = 'this is a slot in a text'
        tokens = [Token('this', 0, 0), Token('is', 0, 0), Token('a', 0, 0), Token('slot', 0, 0), Token('in', 0, 0), Token('a', 0, 0), Token('text', 0, 0)]
        features = [{'ngram_1': 'this', 'is_first': '1'}, {'ngram_1': 'is', 'common': '1'}, {'ngram_1': 'a'}, {'ngram_1': 'slot'}, {'ngram_1': 'in'}, {'ngram_1': 'a'}, {'ngram_1': 'text'}]
        tags = ['O', 'O', 'B-slot', 'I-slot', 'O', 'O', 'O']
        tags = [_encode_tag(t) for t in tags]
        transitions_weights = {(_encode_tag('O'), _encode_tag('O')): 2, (_encode_tag('O'), _encode_tag('B-slot')): 1, (_encode_tag('B-slot'), _encode_tag('I-slot')): 2, (_encode_tag('B-slot'), _encode_tag('O')): 1.5}
        states_weights = {('ngram_1:this', _encode_tag('O')): 5, ('ngram_1:this', _encode_tag('B-slot')): -2, ('ngram_1:slot', _encode_tag('B-slot')): 5, ('ngram_1:slot', _encode_tag('I-slot')): -3, ('ngram_1:slot', _encode_tag('O')): -1}

        class MockedSlotFiller(CRFSlotFiller):

            def __init__(self, transition_features, state_features):
                if False:
                    return 10
                mocked_model = MagicMock()
                type(mocked_model).transition_features_ = PropertyMock(return_value=transition_features)
                type(mocked_model).state_features_ = PropertyMock(return_value=state_features)
                self.crf_model = mocked_model
                self.slot_name_mapping = 1

            def __del__(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        slot_filler = MockedSlotFiller(transitions_weights, states_weights)
        log = slot_filler.log_inference_weights(text=text, tokens=tokens, features=features, tags=tags)
        expected_log = 'Feature weights for "this is a slot in a text":\n\n# Token "this" (tagged as O):\n\nTransition weights to next tag:\n- (O, O) -> 2\n- (B-slot, O) -> 1.5\n\nFeature weights:\n- (ngram_1:this, O) -> 5\n- (ngram_1:this, B-slot) -> -2\n\nFeatures not seen at train time:\n- is_first:1\n\n\n# Token "is" (tagged as O):\n\nTransition weights from previous tag:\n- (O, O) -> 2\n- (O, B-slot) -> 1\n\nTransition weights to next tag:\n- (O, B-slot) -> 1\n\nNo feature weights !\n\nFeatures not seen at train time:\n- common:1\n- ngram_1:is\n\n\n# Token "a" (tagged as B-slot):\n\nTransition weights from previous tag:\n- (O, O) -> 2\n- (O, B-slot) -> 1\n\nTransition weights to next tag:\n- (B-slot, I-slot) -> 2\n\nNo feature weights !\n\nFeatures not seen at train time:\n- ngram_1:a\n\n\n# Token "slot" (tagged as I-slot):\n\nTransition weights from previous tag:\n- (B-slot, I-slot) -> 2\n- (B-slot, O) -> 1.5\n\nTransition weights to next tag:\n- (O, O) -> 2\n- (B-slot, O) -> 1.5\n\nFeature weights:\n- (ngram_1:slot, B-slot) -> 5\n- (ngram_1:slot, I-slot) -> -3\n- (ngram_1:slot, O) -> -1\n\n\n# Token "in" (tagged as O):\n\nNo transition from previous tag seen at train time !\n\nTransition weights to next tag:\n- (O, O) -> 2\n- (B-slot, O) -> 1.5\n\nNo feature weights !\n\nFeatures not seen at train time:\n- ngram_1:in\n\n\n# Token "a" (tagged as O):\n\nTransition weights from previous tag:\n- (O, O) -> 2\n- (O, B-slot) -> 1\n\nTransition weights to next tag:\n- (O, O) -> 2\n- (B-slot, O) -> 1.5\n\nNo feature weights !\n\nFeatures not seen at train time:\n- ngram_1:a\n\n\n# Token "text" (tagged as O):\n\nTransition weights from previous tag:\n- (O, O) -> 2\n- (O, B-slot) -> 1\n\nNo feature weights !\n\nFeatures not seen at train time:\n- ngram_1:text'
        self.assertEqual(expected_log, log)

    def test_training_should_be_reproducible(self):
        if False:
            return 10
        random_state = 42
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: MakeTea\nutterances:\n- make me a [beverage_temperature:Temperature](hot) cup of tea\n- make me [number_of_cups:snips/number](five) tea cups')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        slot_filler1 = CRFSlotFiller(random_state=random_state)
        slot_filler1.fit(dataset, 'MakeTea')
        slot_filler2 = CRFSlotFiller(random_state=random_state)
        slot_filler2.fit(dataset, 'MakeTea')
        self.assertDictEqual(slot_filler1.crf_model.state_features_, slot_filler2.crf_model.state_features_)
        self.assertDictEqual(slot_filler1.crf_model.transition_features_, slot_filler2.crf_model.transition_features_)

    def test_should_cleanup(self):
        if False:
            print('Hello World!')
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: MakeTea\nutterances:\n- make me a [beverage_temperature:Temperature](hot) cup of tea\n- make me [number_of_cups:snips/number](five) tea cups')
        dataset = Dataset.from_yaml_files('en', [dataset_stream])
        slot_filler = CRFSlotFiller().fit(dataset, 'MakeTea')
        crf_file = Path(slot_filler.crf_model.modelfile.name)
        self.assertTrue(crf_file.exists())
        slot_filler._cleanup()
        self.assertFalse(crf_file.exists())

    @skipIf(os.name != 'posix', 'files permissions are different on windows')
    def test_crfsuite_files_modes_should_be_644(self):
        if False:
            while True:
                i = 10
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: MakeTea\nutterances:\n- make me a [beverage_temperature:Temperature](hot) cup of tea\n- make me [number_of_cups:snips/number](five) tea cups')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        slot_filler = CRFSlotFiller().fit(dataset, 'MakeTea')
        slot_filler.persist(self.tmp_file_path)
        crfmodel_file = str(self.tmp_file_path / CRF_MODEL_FILENAME)
        filemode = oct(os.stat(crfmodel_file).st_mode & 511)
        self.assertEqual(oct(420), filemode)