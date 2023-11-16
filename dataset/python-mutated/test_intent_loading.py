from __future__ import unicode_literals
import io
from unittest import TestCase
from snips_nlu.dataset import Intent
from snips_nlu.exceptions import IntentFormatError

class TestIntentLoading(TestCase):

    def test_should_load_from_yaml_file(self):
        if False:
            i = 10
            return i + 15
        intent = Intent.from_yaml(io.StringIO('\n# getWeather Intent\n---\ntype: intent\nname: getWeather\nutterances:\n  - "what is the weather in [weatherLocation:location](paris) \n    [date:snips/datetime](today) ?"\n  - "Will it rain [date:snips/datetime](tomorrow) in\n    [weatherLocation:location](london)?"\n        '))
        intent_dict = intent.json
        expected_intent_dict = {'utterances': [{'data': [{'text': 'what is the weather in '}, {'text': 'paris', 'entity': 'location', 'slot_name': 'weatherLocation'}, {'text': ' '}, {'text': 'today', 'entity': 'snips/datetime', 'slot_name': 'date'}, {'text': ' ?'}]}, {'data': [{'text': 'Will it rain '}, {'text': 'tomorrow', 'entity': 'snips/datetime', 'slot_name': 'date'}, {'text': ' in '}, {'text': 'london', 'entity': 'location', 'slot_name': 'weatherLocation'}, {'text': '?'}]}]}
        self.assertDictEqual(expected_intent_dict, intent_dict)

    def test_should_load_from_yaml_file_using_slot_mapping(self):
        if False:
            while True:
                i = 10
        intent = Intent.from_yaml(io.StringIO('\n# getWeather Intent\n---\ntype: intent\nname: getWeather\nslots:\n  - name: date\n    entity: snips/datetime\n  - name: weatherLocation\n    entity: location\nutterances:\n  - what is the weather in [weatherLocation](paris) ?\n  - Will it rain [date] in [weatherLocation](london)?\n        '))
        intent_dict = intent.json
        expected_intent_dict = {'utterances': [{'data': [{'text': 'what is the weather in '}, {'text': 'paris', 'entity': 'location', 'slot_name': 'weatherLocation'}, {'text': ' ?'}]}, {'data': [{'text': 'Will it rain '}, {'text': None, 'entity': 'snips/datetime', 'slot_name': 'date'}, {'text': ' in '}, {'text': 'london', 'entity': 'location', 'slot_name': 'weatherLocation'}, {'text': '?'}]}]}
        self.assertDictEqual(expected_intent_dict, intent_dict)

    def test_should_load_from_yaml_file_using_implicit_values(self):
        if False:
            return 10
        intent = Intent.from_yaml(io.StringIO('\n# getWeather Intent\n---\ntype: intent\nname: getWeather\nutterances:\n  - what is the weather in [location] ?\n        '))
        intent_dict = intent.json
        expected_intent_dict = {'utterances': [{'data': [{'text': 'what is the weather in '}, {'text': None, 'entity': 'location', 'slot_name': 'location'}, {'text': ' ?'}]}]}
        self.assertDictEqual(expected_intent_dict, intent_dict)

    def test_should_raise_when_missing_bracket_in_utterance(self):
        if False:
            i = 10
            return i + 15
        intent_io = io.StringIO("\n# getWeather Intent\n---\ntype: intent\nname: getWeather\nutterances:\n  - what is the weather in [location] ?\n  - give me the weather forecast in [location tomorrow please\n  - what's the weather in [location] this weekend ?\n        ")
        with self.assertRaises(IntentFormatError) as cm:
            Intent.from_yaml(intent_io)
        faulty_utterance = 'give me the weather forecast in [location tomorrow please'
        self.assertTrue(faulty_utterance in str(cm.exception))

    def test_should_raise_when_missing_parenthesis_in_utterance(self):
        if False:
            return 10
        intent_io = io.StringIO("\n# getWeather Intent\n---\ntype: intent\nname: getWeather\nutterances:\n  - what is the weather in [location] ?\n  - give me the weather forecast in [location] tomorrow please\n  - what's the weather in [location](Paris this weekend ?\n        ")
        with self.assertRaises(IntentFormatError) as cm:
            Intent.from_yaml(intent_io)
        faulty_utterance = "what's the weather in [location](Paris this weekend ?"
        self.assertTrue(faulty_utterance in str(cm.exception))