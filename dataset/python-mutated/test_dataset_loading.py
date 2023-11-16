from __future__ import unicode_literals
import io
from unittest import TestCase
import mock
from snips_nlu.dataset import Dataset, validate_and_format_dataset
EXPECTED_DATASET_DICT = {'entities': {'company': {'automatically_extensible': True, 'data': [], 'use_synonyms': True, 'matching_strictness': 1.0}, 'country': {'automatically_extensible': True, 'data': [], 'use_synonyms': True, 'matching_strictness': 1.0}, 'location': {'automatically_extensible': True, 'data': [{'synonyms': ['big apple'], 'value': 'new york'}, {'synonyms': [], 'value': 'london'}], 'use_synonyms': True, 'matching_strictness': 1.0}, 'role': {'automatically_extensible': True, 'data': [], 'use_synonyms': True, 'matching_strictness': 1.0}, 'snips/datetime': {}}, 'intents': {'getWeather': {'utterances': [{'data': [{'text': 'what is the weather in '}, {'entity': 'location', 'slot_name': 'weatherLocation', 'text': 'Paris'}, {'text': '?'}]}, {'data': [{'text': 'is it raining in '}, {'entity': 'location', 'slot_name': 'weatherLocation', 'text': 'new york'}, {'text': ' '}, {'entity': 'snips/datetime', 'slot_name': 'weatherDate', 'text': 'Today'}]}]}, 'whoIsGame': {'utterances': [{'data': [{'text': 'who is the '}, {'entity': 'role', 'slot_name': 'role', 'text': 'president'}, {'text': ' of '}, {'entity': 'country', 'slot_name': 'country', 'text': 'France'}]}, {'data': [{'text': 'who is the '}, {'entity': 'role', 'slot_name': 'role', 'text': 'CEO'}, {'text': ' of '}, {'entity': 'company', 'slot_name': 'company', 'text': 'Google'}, {'text': ' please'}]}]}}, 'language': 'en'}

class TestDatasetLoading(TestCase):

    def test_should_generate_dataset_from_yaml_files(self):
        if False:
            while True:
                i = 10
        who_is_game_yaml = io.StringIO('\n# whoIsGame Intent\n---\ntype: intent\nname: whoIsGame\nutterances:\n  - who is the [role](president) of [country](France)\n  - who is the [role](CEO) of [company](Google) please\n        ')
        get_weather_yaml = io.StringIO('\n# getWeather Intent\n---\ntype: intent\nname: getWeather\nutterances:\n  - what is the weather in [weatherLocation:location](Paris)?\n  - is it raining in [weatherLocation] [weatherDate:snips/datetime]\n        ')
        location_yaml = io.StringIO('\n# Location Entity\n---\ntype: entity\nname: location\nautomatically_extensible: true\nvalues:\n- [new york, big apple]\n- london\n        ')
        dataset_files = [who_is_game_yaml, get_weather_yaml, location_yaml]
        with mock.patch('snips_nlu_parsers.get_builtin_entity_examples', return_value=['Today']):
            dataset = Dataset.from_yaml_files('en', dataset_files)
        validate_and_format_dataset(dataset)
        self.assertDictEqual(EXPECTED_DATASET_DICT, dataset.json)

    def test_should_generate_dataset_from_merged_yaml_file(self):
        if False:
            while True:
                i = 10
        dataset_stream = io.StringIO('\n# whoIsGame Intent\n---\ntype: intent\nname: whoIsGame\nutterances:\n  - who is the [role](president) of [country](France)\n  - who is the [role](CEO) of [company](Google) please\n\n# getWeather Intent\n---\ntype: intent\nname: getWeather\nutterances:\n  - what is the weather in [weatherLocation:location](Paris)?\n  - is it raining in [weatherLocation] [weatherDate:snips/datetime]\n  \n# Location Entity\n---\ntype: entity\nname: location\nautomatically_extensible: true\nvalues:\n- [new york, big apple]\n- london\n        ')
        with mock.patch('snips_nlu_parsers.get_builtin_entity_examples', return_value=['Today']):
            dataset = Dataset.from_yaml_files('en', [dataset_stream])
        validate_and_format_dataset(dataset)
        self.assertDictEqual(EXPECTED_DATASET_DICT, dataset.json)