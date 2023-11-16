from __future__ import unicode_literals
import io
from future.utils import itervalues
from mock import patch
from snips_nlu.constants import RES_ENTITY, RES_INTENT, RES_INTENT_NAME, RES_SLOTS, RES_VALUE, RANDOM_STATE
from snips_nlu.dataset import Dataset
from snips_nlu.exceptions import IntentNotFoundError, NotTrained
from snips_nlu.intent_classifier import IntentClassifier, LogRegIntentClassifier
from snips_nlu.intent_parser import ProbabilisticIntentParser
from snips_nlu.pipeline.configs import CRFSlotFillerConfig, LogRegIntentClassifierConfig, ProbabilisticIntentParserConfig
from snips_nlu.result import unresolved_slot, intent_classification_result
from snips_nlu.slot_filler import SlotFiller
from snips_nlu.tests.utils import FixtureTest, MockIntentClassifier, MockSlotFiller

class TestProbabilisticIntentParser(FixtureTest):

    def test_should_parse(self):
        if False:
            print('Hello World!')
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: intent1\nutterances:\n  - "[slot1:entity1](foo) bar"\n\n---\ntype: intent\nname: intent2\nutterances:\n  - foo bar [slot2:entity2](baz)\n\n---\ntype: intent\nname: intent3\nutterances:\n  - foz for [slot3:entity3](baz)')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        shared = self.get_shared_data(dataset)
        shared[RANDOM_STATE] = 42
        parser = ProbabilisticIntentParser(**shared)
        parser.fit(dataset)
        text = 'foo bar baz'
        result = parser.parse(text)
        expected_slots = [unresolved_slot((8, 11), 'baz', 'entity2', 'slot2')]
        self.assertEqual('intent2', result[RES_INTENT][RES_INTENT_NAME])
        self.assertEqual(expected_slots, result[RES_SLOTS])

    def test_should_parse_with_filter(self):
        if False:
            return 10
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: intent1\nutterances:\n  - "[slot1:entity1](foo) bar"\n\n---\ntype: intent\nname: intent2\nutterances:\n  - foo bar [slot2:entity2](baz)\n\n---\ntype: intent\nname: intent3\nutterances:\n  - foz for [slot3:entity3](baz)')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        shared = self.get_shared_data(dataset)
        shared[RANDOM_STATE] = 42
        parser = ProbabilisticIntentParser(**shared)
        parser.fit(dataset)
        text = 'foo bar baz'
        result = parser.parse(text, intents=['intent1', 'intent3'])
        expected_slots = [unresolved_slot((0, 3), 'foo', 'entity1', 'slot1')]
        self.assertEqual('intent1', result[RES_INTENT][RES_INTENT_NAME])
        self.assertEqual(expected_slots, result[RES_SLOTS])

    def test_should_parse_top_intents(self):
        if False:
            i = 10
            return i + 15
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: intent1\nutterances:\n  - "[entity1](foo) bar"\n\n---\ntype: intent\nname: intent2\nutterances:\n  - foo bar [entity2](baz)\n\n---\ntype: intent\nname: intent3\nutterances:\n  - foz for [entity3](baz)')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        shared = self.get_shared_data(dataset)
        shared[RANDOM_STATE] = 42
        parser = ProbabilisticIntentParser(**shared)
        parser.fit(dataset)
        text = 'foo bar baz'
        results = parser.parse(text, top_n=2)
        intents = [res[RES_INTENT][RES_INTENT_NAME] for res in results]
        entities = [[s[RES_VALUE] for s in res[RES_SLOTS]] for res in results]
        expected_intents = ['intent2', 'intent1']
        expected_entities = [['baz'], ['foo']]
        self.assertListEqual(expected_intents, intents)
        self.assertListEqual(expected_entities, entities)

    def test_should_get_intents(self):
        if False:
            for i in range(10):
                print('nop')
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: intent1\nutterances:\n  - yala yili\n\n---\ntype: intent\nname: intent2\nutterances:\n  - yala yili yulu\n\n---\ntype: intent\nname: intent3\nutterances:\n  - yili yulu yele')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        shared = self.get_shared_data(dataset)
        shared[RANDOM_STATE] = 42
        parser = ProbabilisticIntentParser(**shared).fit(dataset)
        text = 'yala yili yulu'
        results = parser.get_intents(text)
        intents = [res[RES_INTENT_NAME] for res in results]
        expected_intents = ['intent2', 'intent1', 'intent3', None]
        self.assertEqual(expected_intents, intents)

    def test_should_get_slots(self):
        if False:
            i = 10
            return i + 15
        slots_dataset_stream = io.StringIO('\n---\ntype: intent\nname: greeting1\nutterances:\n  - Hello [name1](John)\n\n---\ntype: intent\nname: greeting2\nutterances:\n  - Hello [name2](John)\n\n---\ntype: intent\nname: greeting3\nutterances:\n  - Hello John')
        dataset = Dataset.from_yaml_files('en', [slots_dataset_stream]).json
        parser = ProbabilisticIntentParser().fit(dataset)
        slots_greeting1 = parser.get_slots('Hello John', 'greeting1')
        slots_greeting2 = parser.get_slots('Hello John', 'greeting2')
        slots_goodbye = parser.get_slots('Hello John', 'greeting3')
        self.assertEqual(1, len(slots_greeting1))
        self.assertEqual(1, len(slots_greeting2))
        self.assertEqual(0, len(slots_goodbye))
        self.assertEqual('John', slots_greeting1[0][RES_VALUE])
        self.assertEqual('name1', slots_greeting1[0][RES_ENTITY])
        self.assertEqual('John', slots_greeting2[0][RES_VALUE])
        self.assertEqual('name2', slots_greeting2[0][RES_ENTITY])

    def test_should_get_no_slots_with_none_intent(self):
        if False:
            for i in range(10):
                print('nop')
        slots_dataset_stream = io.StringIO('\n---\ntype: intent\nname: greeting\nutterances:\n  - Hello [name](John)')
        dataset = Dataset.from_yaml_files('en', [slots_dataset_stream]).json
        parser = ProbabilisticIntentParser().fit(dataset)
        slots = parser.get_slots('Hello John', None)
        self.assertListEqual([], slots)

    def test_get_slots_should_raise_with_unknown_intent(self):
        if False:
            for i in range(10):
                print('nop')
        slots_dataset_stream = io.StringIO('\n---\ntype: intent\nname: greeting1\nutterances:\n  - Hello [name1](John)\n\n---\ntype: intent\nname: goodbye\nutterances:\n  - Goodbye [name](Eric)')
        dataset = Dataset.from_yaml_files('en', [slots_dataset_stream]).json

        @IntentClassifier.register('my_intent_classifier', True)
        class MyIntentClassifier(MockIntentClassifier):
            pass

        @SlotFiller.register('my_slot_filler', True)
        class MySlotFiller(MockSlotFiller):
            pass
        config = ProbabilisticIntentParserConfig(intent_classifier_config='my_intent_classifier', slot_filler_config='my_slot_filler')
        parser = ProbabilisticIntentParser(config).fit(dataset)
        with self.assertRaises(IntentNotFoundError):
            parser.get_slots('Hello John', 'greeting3')

    def test_should_retrain_intent_classifier_when_force_retrain(self):
        if False:
            return 10
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: MakeTea\nutterances:\n- make me a [beverage_temperature:Temperature](hot) cup of tea\n- make me [number_of_cups:snips/number](five) tea cups\n\n---\ntype: intent\nname: MakeCoffee\nutterances:\n- make me [number_of_cups:snips/number](one) cup of coffee please\n- brew [number_of_cups] cups of coffee')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        parser = ProbabilisticIntentParser()
        intent_classifier = LogRegIntentClassifier()
        intent_classifier.fit(dataset)
        parser.intent_classifier = intent_classifier
        with patch('snips_nlu.intent_classifier.log_reg_classifier.LogRegIntentClassifier.fit') as mock_fit:
            parser.fit(dataset, force_retrain=True)
            mock_fit.assert_called_once()

    def test_should_not_retrain_intent_classifier_when_no_force_retrain(self):
        if False:
            while True:
                i = 10
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: MakeTea\nutterances:\n- make me a [beverage_temperature:Temperature](hot) cup of tea\n- make me [number_of_cups:snips/number](five) tea cups\n\n---\ntype: intent\nname: MakeCoffee\nutterances:\n- make me [number_of_cups:snips/number](one) cup of coffee please\n- brew [number_of_cups] cups of coffee')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        parser = ProbabilisticIntentParser()
        intent_classifier = LogRegIntentClassifier()
        intent_classifier.fit(dataset)
        parser.intent_classifier = intent_classifier
        with patch('snips_nlu.intent_classifier.log_reg_classifier.LogRegIntentClassifier.fit') as mock_fit:
            parser.fit(dataset, force_retrain=False)
            mock_fit.assert_not_called()

    def test_should_retrain_slot_filler_when_force_retrain(self):
        if False:
            i = 10
            return i + 15
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: MakeTea\nutterances:\n- make me a [beverage_temperature:Temperature](hot) cup of tea\n- make me [number_of_cups:snips/number](five) tea cups\n\n---\ntype: intent\nname: MakeCoffee\nutterances:\n- make me [number_of_cups:snips/number](one) cup of coffee please\n- brew [number_of_cups] cups of coffee')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json

        @IntentClassifier.register('my_intent_classifier', True)
        class MyIntentClassifier(MockIntentClassifier):
            pass

        @SlotFiller.register('my_slot_filler', True)
        class MySlotFiller(MockSlotFiller):
            fit_call_count = 0

            def fit(self, dataset, intent):
                if False:
                    print('Hello World!')
                MySlotFiller.fit_call_count += 1
                return super(MySlotFiller, self).fit(dataset, intent)
        parser_config = ProbabilisticIntentParserConfig(intent_classifier_config='my_intent_classifier', slot_filler_config='my_slot_filler')
        parser = ProbabilisticIntentParser(parser_config)
        slot_filler = MySlotFiller(None)
        slot_filler.fit(dataset, 'MakeCoffee')
        parser.slot_fillers['MakeCoffee'] = slot_filler
        parser.fit(dataset, force_retrain=True)
        self.assertEqual(3, MySlotFiller.fit_call_count)

    def test_should_not_retrain_slot_filler_when_no_force_retrain(self):
        if False:
            for i in range(10):
                print('nop')
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: MakeTea\nutterances:\n- make me a [beverage_temperature:Temperature](hot) cup of tea\n- make me [number_of_cups:snips/number](five) tea cups\n\n---\ntype: intent\nname: MakeCoffee\nutterances:\n- make me [number_of_cups:snips/number](one) cup of coffee please\n- brew [number_of_cups] cups of coffee')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json

        @IntentClassifier.register('my_intent_classifier', True)
        class MyIntentClassifier(MockIntentClassifier):
            pass

        @SlotFiller.register('my_slot_filler', True)
        class MySlotFiller(MockSlotFiller):
            fit_call_count = 0

            def fit(self, dataset, intent):
                if False:
                    while True:
                        i = 10
                MySlotFiller.fit_call_count += 1
                return super(MySlotFiller, self).fit(dataset, intent)
        parser_config = ProbabilisticIntentParserConfig(intent_classifier_config='my_intent_classifier', slot_filler_config='my_slot_filler')
        parser = ProbabilisticIntentParser(parser_config)
        slot_filler = MySlotFiller(None)
        slot_filler.fit(dataset, 'MakeCoffee')
        parser.slot_fillers['MakeCoffee'] = slot_filler
        parser.fit(dataset, force_retrain=False)
        self.assertEqual(2, MySlotFiller.fit_call_count)

    def test_should_not_parse_when_not_fitted(self):
        if False:
            return 10
        parser = ProbabilisticIntentParser()
        self.assertFalse(parser.fitted)
        with self.assertRaises(NotTrained):
            parser.parse('foobar')

    def test_should_be_serializable_before_fitting(self):
        if False:
            print('Hello World!')
        parser = ProbabilisticIntentParser()
        parser.persist(self.tmp_file_path)
        expected_parser_dict = {'config': {'unit_name': 'probabilistic_intent_parser', 'slot_filler_config': CRFSlotFillerConfig().to_dict(), 'intent_classifier_config': LogRegIntentClassifierConfig().to_dict()}, 'slot_fillers': []}
        metadata = {'unit_name': 'probabilistic_intent_parser'}
        self.assertJsonContent(self.tmp_file_path / 'metadata.json', metadata)
        self.assertJsonContent(self.tmp_file_path / 'intent_parser.json', expected_parser_dict)

    def test_should_be_deserializable_before_fitting(self):
        if False:
            return 10
        config = ProbabilisticIntentParserConfig().to_dict()
        parser_dict = {'unit_name': 'probabilistic_intent_parser', 'config': config, 'intent_classifier': None, 'slot_fillers': dict()}
        self.tmp_file_path.mkdir()
        metadata = {'unit_name': 'probabilistic_intent_parser'}
        self.writeJsonContent(self.tmp_file_path / 'metadata.json', metadata)
        self.writeJsonContent(self.tmp_file_path / 'intent_parser.json', parser_dict)
        parser = ProbabilisticIntentParser.from_path(self.tmp_file_path)
        self.assertEqual(parser.config.to_dict(), config)
        self.assertIsNone(parser.intent_classifier)
        self.assertDictEqual(dict(), parser.slot_fillers)

    def test_should_be_serializable(self):
        if False:
            return 10
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: MakeTea\nutterances:\n- make me a [beverage_temperature:Temperature](hot) cup of tea\n- make me [number_of_cups:snips/number](five) tea cups\n\n---\ntype: intent\nname: MakeCoffee\nutterances:\n- make me [number_of_cups:snips/number](one) cup of coffee please\n- brew [number_of_cups] cups of coffee')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json

        @IntentClassifier.register('my_intent_classifier', True)
        class MyIntentClassifier(MockIntentClassifier):
            pass

        @SlotFiller.register('my_slot_filler', True)
        class MySlotFiller(MockSlotFiller):
            pass
        parser_config = ProbabilisticIntentParserConfig(intent_classifier_config='my_intent_classifier', slot_filler_config='my_slot_filler')
        parser = ProbabilisticIntentParser(parser_config).fit(dataset)
        parser.persist(self.tmp_file_path)
        expected_parser_config = {'unit_name': 'probabilistic_intent_parser', 'slot_filler_config': {'unit_name': 'my_slot_filler'}, 'intent_classifier_config': {'unit_name': 'my_intent_classifier'}}
        expected_parser_dict = {'config': expected_parser_config, 'slot_fillers': [{'intent': 'MakeCoffee', 'slot_filler_name': 'slot_filler_0'}, {'intent': 'MakeTea', 'slot_filler_name': 'slot_filler_1'}]}
        metadata = {'unit_name': 'probabilistic_intent_parser'}
        metadata_slot_filler = {'unit_name': 'my_slot_filler', 'fitted': True}
        metadata_intent_classifier = {'unit_name': 'my_intent_classifier', 'fitted': True}
        self.assertJsonContent(self.tmp_file_path / 'metadata.json', metadata)
        self.assertJsonContent(self.tmp_file_path / 'intent_parser.json', expected_parser_dict)
        self.assertJsonContent(self.tmp_file_path / 'intent_classifier' / 'metadata.json', metadata_intent_classifier)
        self.assertJsonContent(self.tmp_file_path / 'slot_filler_0' / 'metadata.json', metadata_slot_filler)
        self.assertJsonContent(self.tmp_file_path / 'slot_filler_1' / 'metadata.json', metadata_slot_filler)

    def test_should_be_deserializable(self):
        if False:
            while True:
                i = 10

        @IntentClassifier.register('my_intent_classifier', True)
        class MyIntentClassifier(MockIntentClassifier):
            pass

        @SlotFiller.register('my_slot_filler', True)
        class MySlotFiller(MockSlotFiller):
            pass
        parser_config = {'unit_name': 'probabilistic_intent_parser', 'intent_classifier_config': {'unit_name': 'my_intent_classifier'}, 'slot_filler_config': {'unit_name': 'my_slot_filler'}}
        parser_dict = {'unit_name': 'probabilistic_intent_parser', 'slot_fillers': [{'intent': 'MakeCoffee', 'slot_filler_name': 'slot_filler_MakeCoffee'}, {'intent': 'MakeTea', 'slot_filler_name': 'slot_filler_MakeTea'}], 'config': parser_config}
        self.tmp_file_path.mkdir()
        (self.tmp_file_path / 'intent_classifier').mkdir()
        (self.tmp_file_path / 'slot_filler_MakeCoffee').mkdir()
        (self.tmp_file_path / 'slot_filler_MakeTea').mkdir()
        self.writeJsonContent(self.tmp_file_path / 'intent_parser.json', parser_dict)
        self.writeJsonContent(self.tmp_file_path / 'intent_classifier' / 'metadata.json', {'unit_name': 'my_intent_classifier', 'fitted': True})
        self.writeJsonContent(self.tmp_file_path / 'slot_filler_MakeCoffee' / 'metadata.json', {'unit_name': 'my_slot_filler', 'fitted': True})
        self.writeJsonContent(self.tmp_file_path / 'slot_filler_MakeTea' / 'metadata.json', {'unit_name': 'my_slot_filler', 'fitted': True})
        parser = ProbabilisticIntentParser.from_path(self.tmp_file_path)
        self.assertDictEqual(parser.config.to_dict(), parser_config)
        self.assertIsInstance(parser.intent_classifier, MyIntentClassifier)
        self.assertListEqual(sorted(parser.slot_fillers), ['MakeCoffee', 'MakeTea'])
        for slot_filler in itervalues(parser.slot_fillers):
            self.assertIsInstance(slot_filler, MySlotFiller)

    def test_should_be_serializable_into_bytearray(self):
        if False:
            for i in range(10):
                print('nop')
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: MakeTea\nutterances:\n- make me a [beverage_temperature:Temperature](hot) cup of tea\n- make me [number_of_cups:snips/number](five) tea cups\n\n---\ntype: intent\nname: MakeCoffee\nutterances:\n- make me [number_of_cups:snips/number](one) cup of coffee please\n- brew [number_of_cups] cups of coffee')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json

        @IntentClassifier.register('my_intent_classifier', True)
        class MyIntentClassifier(MockIntentClassifier):

            def get_intent(self, text, intents_filter):
                if False:
                    for i in range(10):
                        print('nop')
                if 'tea' in text:
                    return intent_classification_result('MakeTea', 1.0)
                elif 'coffee' in text:
                    return intent_classification_result('MakeCoffee', 1.0)
                return intent_classification_result(None, 1.0)

        @SlotFiller.register('my_slot_filler', True)
        class MySlotFiller(MockSlotFiller):
            pass
        parser_config = ProbabilisticIntentParserConfig(intent_classifier_config='my_intent_classifier', slot_filler_config='my_slot_filler')
        parser = ProbabilisticIntentParser(parser_config).fit(dataset)
        intent_parser_bytes = parser.to_byte_array()
        loaded_intent_parser = ProbabilisticIntentParser.from_byte_array(intent_parser_bytes)
        result = loaded_intent_parser.parse('make me two cups of tea')
        self.assertEqual('MakeTea', result[RES_INTENT][RES_INTENT_NAME])

    def test_fitting_should_be_reproducible_after_serialization(self):
        if False:
            return 10
        dataset_stream = io.StringIO('\n---\ntype: intent\nname: MakeTea\nutterances:\n- make me a [beverage_temperature:Temperature](hot) cup of tea\n- make me [number_of_cups:snips/number](five) tea cups\n\n---\ntype: intent\nname: MakeCoffee\nutterances:\n- make me [number_of_cups:snips/number](one) cup of coffee please\n- brew [number_of_cups] cups of coffee')
        dataset = Dataset.from_yaml_files('en', [dataset_stream]).json
        seed = 666
        shared = self.get_shared_data(dataset)
        shared[RANDOM_STATE] = seed
        parser = ProbabilisticIntentParser(**shared)
        parser.persist(self.tmp_file_path)
        fitted_parser_1 = ProbabilisticIntentParser.from_path(self.tmp_file_path, **shared).fit(dataset)
        fitted_parser_2 = ProbabilisticIntentParser.from_path(self.tmp_file_path, **shared).fit(dataset)
        feature_weights_1 = fitted_parser_1.slot_fillers['MakeTea'].crf_model.state_features_
        feature_weights_2 = fitted_parser_2.slot_fillers['MakeTea'].crf_model.state_features_
        self.assertEqual(feature_weights_1, feature_weights_2)