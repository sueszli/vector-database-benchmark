from __future__ import unicode_literals
import io
from builtins import range, str
from future.utils import iteritems
from mock import mock, patch
from snips_nlu.constants import ENTITIES, SNIPS_DATETIME, VALIDATED
from snips_nlu.dataset import Dataset
from snips_nlu.dataset.validation import validate_and_format_dataset, _validate_and_format_custom_entity
from snips_nlu.exceptions import DatasetFormatError
from snips_nlu.tests.utils import SnipsTest, EntityParserMock

class TestDatasetValidation(SnipsTest):

    def test_missing_intent_key_should_raise_exception(self):
        if False:
            while True:
                i = 10
        dataset = {'intents': {'intent1': {'utterances': [{'data': [{'text': 'unknown entity', 'entity': 'unknown_entity'}]}]}}, 'entities': {}, 'language': 'en'}
        with self.assertRaises(DatasetFormatError) as ctx:
            validate_and_format_dataset(dataset)
        self.assertEqual("Expected chunk to have key: 'slot_name'", str(ctx.exception.args[0]))

    def test_unknown_entity_should_raise_exception(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = {'intents': {'intent1': {'utterances': [{'data': [{'text': 'unknown entity', 'entity': 'unknown_entity', 'slot_name': 'unknown_entity_slot'}]}]}}, 'entities': {'entity1': {'data': [], 'use_synonyms': True, 'automatically_extensible': False}}, 'language': 'en'}
        with self.assertRaises(DatasetFormatError) as ctx:
            validate_and_format_dataset(dataset)
        self.assertEqual("Expected entities to have key: 'unknown_entity'", str(ctx.exception.args[0]))

    def test_missing_entity_key_should_raise_exception(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = {'intents': {}, 'entities': {'entity1': {'data': [], 'automatically_extensible': False, 'matching_strictness': 1.0}}, 'language': 'en'}
        with self.assertRaises(DatasetFormatError) as ctx:
            validate_and_format_dataset(dataset)
        self.assertEqual("Expected custom entity to have key: 'use_synonyms'", str(ctx.exception.args[0]))

    def test_should_support_int_or_float_for_matching_strictness(self):
        if False:
            while True:
                i = 10
        dataset = {'intents': {}, 'entities': {'entity1': {'data': [], 'automatically_extensible': False, 'use_synonyms': True, 'matching_strictness': 0.5}, 'entity2': {'data': [], 'automatically_extensible': False, 'use_synonyms': True, 'matching_strictness': 1}}, 'language': 'en'}
        dataset = validate_and_format_dataset(dataset)
        self.assertEqual(0.5, dataset['entities']['entity1'].get('matching_strictness'))
        self.assertEqual(1, dataset['entities']['entity2'].get('matching_strictness'))

    def test_missing_matching_strictness_should_be_handled(self):
        if False:
            while True:
                i = 10
        dataset = {'intents': {}, 'entities': {'entity1': {'data': [], 'automatically_extensible': False, 'use_synonyms': True}}, 'language': 'en'}
        dataset = validate_and_format_dataset(dataset)
        self.assertEqual(1.0, dataset['entities']['entity1'].get('matching_strictness'))

    def test_deprecated_parser_threshold_should_be_handled(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = {'intents': {}, 'entities': {'entity1': {'data': [], 'automatically_extensible': False, 'use_synonyms': True, 'parser_threshold': 0.5}}, 'language': 'en'}
        dataset = validate_and_format_dataset(dataset)
        self.assertEqual(0.5, dataset['entities']['entity1'].get('matching_strictness'))

    def test_invalid_language_should_raise_exception(self):
        if False:
            while True:
                i = 10
        dataset = {'intents': {}, 'entities': {}, 'language': 'eng'}
        with self.assertRaises(DatasetFormatError) as ctx:
            validate_and_format_dataset(dataset)
        self.assertEqual("Unknown language: 'eng'", str(ctx.exception.args[0]))

    @mock.patch('snips_nlu.dataset.validation.get_string_variations')
    def test_should_format_dataset_by_adding_synonyms(self, mocked_get_string_variations):
        if False:
            while True:
                i = 10

        def mock_get_string_variations(string, language, builtin_entity_parser, numbers=True, case=True, and_=True, punctuation=True):
            if False:
                for i in range(10):
                    print('nop')
            return {string.lower(), string.title()}
        mocked_get_string_variations.side_effect = mock_get_string_variations
        dataset = {'intents': {}, 'entities': {'entity1': {'data': [{'value': 'Entity_1', 'synonyms': ['entity 2']}], 'use_synonyms': True, 'automatically_extensible': False, 'matching_strictness': 1.0}}, 'language': 'en'}
        expected_dataset = {'intents': {}, 'entities': {'entity1': {'utterances': {'Entity_1': 'Entity_1', 'entity_1': 'Entity_1', 'entity 2': 'Entity_1', 'Entity 2': 'Entity_1'}, 'automatically_extensible': False, 'capitalize': False, 'matching_strictness': 1.0}}, 'language': 'en', 'validated': True}
        dataset = validate_and_format_dataset(dataset)
        self.assertDictEqual(expected_dataset, dataset)

    @mock.patch('snips_nlu.dataset.validation.get_string_variations')
    def test_should_format_dataset_by_adding_entity_values(self, mocked_get_string_variations):
        if False:
            return 10

        def mock_get_string_variations(string, language, builtin_entity_parser, numbers=True, case=True, and_=True, punctuation=True):
            if False:
                for i in range(10):
                    print('nop')
            return {string, string.title()}
        mocked_get_string_variations.side_effect = mock_get_string_variations
        dataset = {'intents': {'intent1': {'utterances': [{'data': [{'text': 'this is '}, {'text': 'alternative entity 1', 'entity': 'entity1', 'slot_name': 'slot1'}]}, {'data': [{'text': 'this is '}, {'text': 'entity 1', 'entity': 'entity1', 'slot_name': 'slot1'}]}]}}, 'entities': {'entity1': {'data': [{'value': 'entity 1', 'synonyms': ['entity 1', 'entity 1 bis']}], 'use_synonyms': True, 'automatically_extensible': False, 'matching_strictness': 1.0}}, 'language': 'en'}
        expected_dataset = {'intents': {'intent1': {'utterances': [{'data': [{'text': 'this is '}, {'text': 'alternative entity 1', 'entity': 'entity1', 'slot_name': 'slot1'}]}, {'data': [{'text': 'this is '}, {'text': 'entity 1', 'entity': 'entity1', 'slot_name': 'slot1'}]}]}}, 'entities': {'entity1': {'utterances': {'entity 1 bis': 'entity 1', 'Entity 1 Bis': 'entity 1', 'entity 1': 'entity 1', 'Entity 1': 'entity 1', 'alternative entity 1': 'alternative entity 1', 'Alternative Entity 1': 'alternative entity 1'}, 'automatically_extensible': False, 'capitalize': False, 'matching_strictness': 1.0}}, 'language': 'en', 'validated': True}
        dataset = validate_and_format_dataset(dataset)
        self.assertEqual(expected_dataset, dataset)

    @mock.patch('snips_nlu.dataset.validation.get_string_variations')
    def test_should_add_missing_reference_entity_values_when_not_use_synonyms(self, mocked_get_string_variations):
        if False:
            print('Hello World!')

        def mock_get_string_variations(string, language, builtin_entity_parser, numbers=True, case=True, and_=True, punctuation=True):
            if False:
                print('Hello World!')
            return {string}
        mocked_get_string_variations.side_effect = mock_get_string_variations
        dataset = {'intents': {'intent1': {'utterances': [{'data': [{'text': 'this is '}, {'text': 'alternative entity 1', 'entity': 'entity1', 'slot_name': 'slot1'}]}, {'data': [{'text': 'this is '}, {'text': 'entity 1', 'entity': 'entity1', 'slot_name': 'slot1'}]}]}}, 'entities': {'entity1': {'data': [{'value': 'entity 1', 'synonyms': ['entity 1', 'alternative entity 1']}], 'use_synonyms': False, 'automatically_extensible': False, 'matching_strictness': 1.0}}, 'language': 'en'}
        expected_dataset = {'intents': {'intent1': {'utterances': [{'data': [{'text': 'this is '}, {'text': 'alternative entity 1', 'entity': 'entity1', 'slot_name': 'slot1'}]}, {'data': [{'text': 'this is '}, {'text': 'entity 1', 'entity': 'entity1', 'slot_name': 'slot1'}]}]}}, 'entities': {'entity1': {'utterances': {'alternative entity 1': 'alternative entity 1', 'entity 1': 'entity 1'}, 'automatically_extensible': False, 'capitalize': False, 'matching_strictness': 1.0}}, 'language': 'en', 'validated': True}
        dataset = validate_and_format_dataset(dataset)
        self.assertEqual(expected_dataset, dataset)

    def test_should_not_require_data_for_builtin_entities(self):
        if False:
            return 10
        dataset = {'intents': {'intent1': {'utterances': [{'data': [{'text': 'this is '}, {'text': '10p.m', 'entity': SNIPS_DATETIME, 'slot_name': 'startTime'}]}]}}, 'entities': {SNIPS_DATETIME: {}}, 'language': 'en'}
        with self.fail_if_exception('Could not validate dataset'):
            validate_and_format_dataset(dataset)

    @mock.patch('snips_nlu.dataset.validation.get_string_variations')
    def test_should_remove_empty_entities_value_and_empty_synonyms(self, mocked_get_string_variations):
        if False:
            return 10

        def mock_get_string_variations(string, language, builtin_entity_parser, numbers=True, case=True, and_=True, punctuation=True):
            if False:
                for i in range(10):
                    print('nop')
            return {string, string.title()}
        mocked_get_string_variations.side_effect = mock_get_string_variations
        dataset = {'intents': {'intent1': {'utterances': [{'data': [{'text': 'this is '}, {'text': '', 'entity': 'entity1', 'slot_name': 'slot1'}]}, {'data': [{'text': 'this is '}, {'text': 'entity 1', 'entity': 'entity1', 'slot_name': 'slot1'}]}]}}, 'entities': {'entity1': {'data': [{'value': 'entity 1', 'synonyms': ['']}, {'value': '', 'synonyms': []}], 'use_synonyms': False, 'automatically_extensible': False, 'matching_strictness': 1.0}}, 'language': 'en'}
        expected_dataset = {'intents': {'intent1': {'utterances': [{'data': [{'text': 'this is '}, {'text': '', 'entity': 'entity1', 'slot_name': 'slot1'}]}, {'data': [{'text': 'this is '}, {'text': 'entity 1', 'entity': 'entity1', 'slot_name': 'slot1'}]}]}}, 'entities': {'entity1': {'utterances': {'entity 1': 'entity 1', 'Entity 1': 'entity 1'}, 'capitalize': False, 'automatically_extensible': False, 'matching_strictness': 1.0}}, 'language': 'en', 'validated': True}
        dataset = validate_and_format_dataset(dataset)
        self.assertEqual(expected_dataset, dataset)

    @mock.patch('snips_nlu.dataset.validation.get_string_variations')
    def test_should_add_capitalize_field(self, mocked_get_string_variations):
        if False:
            return 10

        def mock_get_string_variations(string, language, builtin_entity_parser, numbers=True, case=True, and_=True, punctuation=True):
            if False:
                i = 10
                return i + 15
            return {string, string.title()}
        mocked_get_string_variations.side_effect = mock_get_string_variations
        dataset = {'intents': {'intent1': {'utterances': [{'data': [{'text': 'My entity1', 'entity': 'entity1', 'slot_name': 'slot0'}, {'text': 'entity1', 'entity': 'entity1', 'slot_name': 'slot2'}, {'text': 'entity1', 'entity': 'entity1', 'slot_name': 'slot2'}, {'text': 'entity1', 'entity': 'entity1', 'slot_name': 'slot3'}, {'text': 'My entity2', 'entity': 'entity2', 'slot_name': 'slot1'}, {'text': 'myentity2', 'entity': 'entity2', 'slot_name': 'slot1'}, {'text': 'm_entity3', 'entity': 'entity3', 'slot_name': 'slot1'}]}]}}, 'entities': {'entity1': {'data': [], 'use_synonyms': False, 'automatically_extensible': True, 'matching_strictness': 1.0}, 'entity2': {'data': [], 'use_synonyms': False, 'automatically_extensible': True, 'matching_strictness': 1.0}, 'entity3': {'data': [{'value': 'Entity3', 'synonyms': ['entity3']}], 'use_synonyms': False, 'automatically_extensible': True, 'matching_strictness': 1.0}}, 'language': 'en'}
        expected_dataset = {'intents': {'intent1': {'utterances': [{'data': [{'text': 'My entity1', 'entity': 'entity1', 'slot_name': 'slot0'}, {'text': 'entity1', 'entity': 'entity1', 'slot_name': 'slot2'}, {'text': 'entity1', 'entity': 'entity1', 'slot_name': 'slot2'}, {'text': 'entity1', 'entity': 'entity1', 'slot_name': 'slot3'}, {'text': 'My entity2', 'entity': 'entity2', 'slot_name': 'slot1'}, {'text': 'myentity2', 'entity': 'entity2', 'slot_name': 'slot1'}, {'text': 'm_entity3', 'entity': 'entity3', 'slot_name': 'slot1'}]}]}}, 'entities': {'entity1': {'utterances': {'My entity1': 'My entity1', 'My Entity1': 'My entity1', 'entity1': 'entity1', 'Entity1': 'entity1'}, 'automatically_extensible': True, 'capitalize': True, 'matching_strictness': 1.0}, 'entity2': {'utterances': {'My entity2': 'My entity2', 'My Entity2': 'My entity2', 'myentity2': 'myentity2', 'Myentity2': 'myentity2'}, 'automatically_extensible': True, 'capitalize': True, 'matching_strictness': 1.0}, 'entity3': {'utterances': {'Entity3': 'Entity3', 'm_entity3': 'm_entity3', 'M_Entity3': 'm_entity3'}, 'automatically_extensible': True, 'capitalize': False, 'matching_strictness': 1.0}}, 'language': 'en', 'validated': True}
        dataset = validate_and_format_dataset(dataset)
        self.assertDictEqual(expected_dataset, dataset)

    @mock.patch('snips_nlu.dataset.validation.get_string_variations')
    def test_should_normalize_synonyms(self, mocked_get_string_variations):
        if False:
            print('Hello World!')

        def mock_get_string_variations(string, language, builtin_entity_parser, numbers=True, case=True, and_=True, punctuation=True):
            if False:
                print('Hello World!')
            return {string.lower(), string.title()}
        mocked_get_string_variations.side_effect = mock_get_string_variations
        dataset = {'intents': {'intent1': {'utterances': [{'data': [{'text': 'ëNtity', 'entity': 'entity1', 'slot_name': 'startTime'}]}]}}, 'entities': {'entity1': {'data': [], 'use_synonyms': True, 'automatically_extensible': True, 'matching_strictness': 1.0}}, 'language': 'en'}
        expected_dataset = {'intents': {'intent1': {'utterances': [{'data': [{'text': 'ëNtity', 'entity': 'entity1', 'slot_name': 'startTime'}]}]}}, 'entities': {'entity1': {'utterances': {'ëntity': 'ëNtity', 'Ëntity': 'ëNtity', 'ëNtity': 'ëNtity'}, 'automatically_extensible': True, 'capitalize': False, 'matching_strictness': 1.0}}, 'language': 'en', 'validated': True}
        dataset = validate_and_format_dataset(dataset)
        self.assertDictEqual(expected_dataset, dataset)

    @mock.patch('snips_nlu.dataset.validation.get_string_variations')
    def test_dataset_should_handle_synonyms(self, mocked_get_string_variations):
        if False:
            for i in range(10):
                print('nop')

        def mock_get_string_variations(string, language, builtin_entity_parser, numbers=True, case=True, and_=True, punctuation=True):
            if False:
                return 10
            return {string.lower(), string.title()}
        mocked_get_string_variations.side_effect = mock_get_string_variations
        dataset = {'intents': {}, 'entities': {'entity1': {'data': [{'value': 'Ëntity 1', 'synonyms': ['entity 2']}], 'use_synonyms': True, 'automatically_extensible': True, 'matching_strictness': 1.0}}, 'language': 'en'}
        dataset = validate_and_format_dataset(dataset)
        expected_entities = {'entity1': {'automatically_extensible': True, 'utterances': {'Ëntity 1': 'Ëntity 1', 'ëntity 1': 'Ëntity 1', 'entity 2': 'Ëntity 1', 'Entity 2': 'Ëntity 1'}, 'capitalize': False, 'matching_strictness': 1.0}}
        self.assertDictEqual(dataset[ENTITIES], expected_entities)

    def test_should_not_avoid_synomyms_variations_collision(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = {'intents': {'dummy_but_tricky_intent': {'utterances': [{'data': [{'text': 'dummy_value', 'entity': 'dummy_but_tricky_entity', 'slot_name': 'dummy_but_tricky_slot'}]}]}}, 'entities': {'dummy_but_tricky_entity': {'data': [{'value': 'a', 'synonyms': ['favorïte']}, {'value': 'b', 'synonyms': ['favorite']}], 'use_synonyms': True, 'automatically_extensible': False, 'matching_strictness': 1.0}}, 'language': 'en'}
        dataset = validate_and_format_dataset(dataset)
        entity = dataset['entities']['dummy_but_tricky_entity']
        expected_utterances = {'A': 'a', 'B': 'b', 'DummyValue': 'dummy_value', 'Dummy_Value': 'dummy_value', 'Favorïte': 'a', 'a': 'a', 'b': 'b', 'dummy_value': 'dummy_value', 'dummyvalue': 'dummy_value', 'favorite': 'b', 'favorïte': 'a'}
        self.assertDictEqual(expected_utterances, entity['utterances'])

    def test_should_create_number_variation(self):
        if False:
            for i in range(10):
                print('nop')
        args = {1: {'numbers': True, 'and_': True, 'case': True, 'punctuation': True}, 1001: {'numbers': False, 'and_': True, 'case': True, 'punctuation': True}, 10001: {'numbers': False, 'and_': False, 'case': False, 'punctuation': False}}
        for (num_ents, expected_args) in iteritems(args):
            entity = {'matching_strictness': 1.0, 'use_synonyms': False, 'automatically_extensible': False, 'data': [{'value': str(i), 'synonyms': []} for i in range(num_ents)]}
            builtin_entity_parser = EntityParserMock(dict())
            with patch('snips_nlu.dataset.validation.get_string_variations') as mocked_string_variations:
                mocked_string_variations.return_value = []
                _validate_and_format_custom_entity(entity, [], 'en', builtin_entity_parser)
                for call in mocked_string_variations.mock_calls:
                    kwargs = call[2]
                    for k in expected_args:
                        self.assertEqual(expected_args[k], kwargs[k])

    def test_should_not_collapse_utterance_entity_variations(self):
        if False:
            return 10
        dataset = {'language': 'en', 'intents': {'verify_length': {'utterances': [{'data': [{'text': 'hello '}, {'text': '9', 'slot_name': 'expected', 'entity': 'expected'}]}, {'data': [{'text': 'hello '}, {'text': 'nine', 'slot_name': 'expected', 'entity': 'expected'}]}]}}, 'entities': {'expected': {'automatically_extensible': True, 'use_synonyms': True, 'data': [], 'matching_strictness': 1.0}}}
        validated_dataset = validate_and_format_dataset(dataset)
        expected_dataset = {'language': 'en', 'intents': {'verify_length': {'utterances': [{'data': [{'text': 'hello '}, {'text': '9', 'slot_name': 'expected', 'entity': 'expected'}]}, {'data': [{'text': 'hello '}, {'text': 'nine', 'slot_name': 'expected', 'entity': 'expected'}]}]}}, 'entities': {'expected': {'automatically_extensible': True, 'matching_strictness': 1.0, 'capitalize': False, 'utterances': {'nine': 'nine', 'Nine': 'nine', '9': '9'}}}, 'validated': True}
        self.assertDictEqual(expected_dataset, validated_dataset)

    def test_should_keep_license_info(self):
        if False:
            while True:
                i = 10
        dataset = {'intents': {}, 'entities': {'my_entity': {'data': [{'value': 'foo', 'synonyms': []}], 'use_synonyms': True, 'automatically_extensible': True, 'matching_strictness': 1.0, 'license_info': {'filename': 'LICENSE', 'content': 'some license content here'}}}, 'language': 'en'}
        validated_dataset = validate_and_format_dataset(dataset)
        expected_dataset = {'entities': {'my_entity': {'automatically_extensible': True, 'capitalize': False, 'matching_strictness': 1.0, 'utterances': {'Foo': 'foo', 'foo': 'foo'}, 'license_info': {'filename': 'LICENSE', 'content': 'some license content here'}}}, 'intents': {}, 'language': 'en', 'validated': True}
        self.assertDictEqual(expected_dataset, validated_dataset)

    def test_validate_should_be_idempotent(self):
        if False:
            print('Hello World!')
        dataset_stream = io.StringIO('\n# getWeather Intent\n---\ntype: intent\nname: getWeather\nutterances:\n  - what is the weather in [weatherLocation:location](Paris)?\n  - is it raining in [weatherLocation] [weatherDate:snips/datetime]\n\n# Location Entity\n---\ntype: entity\nname: location\nautomatically_extensible: true\nvalues:\n- [new york, big apple]\n- london\n        ')
        dataset = Dataset.from_yaml_files('en', [dataset_stream])
        validated_dataset = validate_and_format_dataset(dataset)
        validated_dataset_2 = validate_and_format_dataset(validated_dataset)
        self.assertDictEqual(validated_dataset, validated_dataset_2)
        self.assertTrue(validated_dataset.get(VALIDATED, False))

    def test_validate_should_accept_dataset_object(self):
        if False:
            while True:
                i = 10
        dataset_stream = io.StringIO('\n# getWeather Intent\n---\ntype: intent\nname: getWeather\nutterances:\n  - what is the weather in [weatherLocation:location](Paris)?\n  - is it raining in [weatherLocation] [weatherDate:snips/datetime]\n\n# Location Entity\n---\ntype: entity\nname: location\nautomatically_extensible: true\nvalues:\n- [new york, big apple]\n- london\n        ')
        dataset = Dataset.from_yaml_files('en', [dataset_stream])
        validated_dataset = validate_and_format_dataset(dataset)
        self.assertTrue(validated_dataset.get(VALIDATED, False))