from __future__ import unicode_literals
import io
from unittest import TestCase
from snips_nlu.dataset import Entity
from snips_nlu.exceptions import EntityFormatError

class TestEntityLoading(TestCase):

    def test_from_yaml_file(self):
        if False:
            for i in range(10):
                print('nop')
        entity = Entity.from_yaml(io.StringIO('\n# Location Entity\n---\ntype: entity\nname: location\nautomatically_extensible: no\nuse_synonyms: yes\nmatching_strictness: 0.5\nvalues:\n- [new york, big apple]\n- [paris, city of lights]\n- london\n        '))
        entity_dict = entity.json
        expected_entity_dict = {'automatically_extensible': False, 'data': [{'synonyms': ['big apple'], 'value': 'new york'}, {'synonyms': ['city of lights'], 'value': 'paris'}, {'synonyms': [], 'value': 'london'}], 'use_synonyms': True, 'matching_strictness': 0.5}
        self.assertDictEqual(expected_entity_dict, entity_dict)

    def test_from_yaml_file_with_defaults(self):
        if False:
            print('Hello World!')
        entity = Entity.from_yaml(io.StringIO('\n# Location Entity\n---\nname: location\nvalues:\n- [new york, big apple]\n- [paris, city of lights]\n- london\n        '))
        entity_dict = entity.json
        expected_entity_dict = {'automatically_extensible': True, 'data': [{'synonyms': ['big apple'], 'value': 'new york'}, {'synonyms': ['city of lights'], 'value': 'paris'}, {'synonyms': [], 'value': 'london'}], 'use_synonyms': True, 'matching_strictness': 1.0}
        self.assertDictEqual(expected_entity_dict, entity_dict)

    def test_fail_from_yaml_file_when_wrong_type(self):
        if False:
            return 10
        yaml_stream = io.StringIO('\n# Location Entity\n---\ntype: intent\nname: location\nvalues:\n- [new york, big apple]\n- [paris, city of lights]\n- london\n        ')
        with self.assertRaises(EntityFormatError):
            Entity.from_yaml(yaml_stream)

    def test_fail_from_yaml_file_when_no_name(self):
        if False:
            print('Hello World!')
        entity_io = io.StringIO('\n# Location Entity\n---\nvalues:\n- [new york, big apple]\n- [paris, city of lights]\n- london\n        ')
        with self.assertRaises(EntityFormatError):
            Entity.from_yaml(entity_io)