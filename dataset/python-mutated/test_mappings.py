import unittest
from troposphere import Template
single_mapping = '{\n "Mappings": {\n  "map": {\n   "n": "v"\n  }\n },\n "Resources": {}\n}'
multiple_mappings = '{\n "Mappings": {\n  "map": {\n   "k1": {\n    "n1": "v1"\n   },\n   "k2": {\n    "n2": "v2"\n   }\n  }\n },\n "Resources": {}\n}'

class TestMappings(unittest.TestCase):

    def test_single_mapping(self):
        if False:
            print('Hello World!')
        template = Template()
        template.add_mapping('map', {'n': 'v'})
        json = template.to_json()
        self.assertEqual(single_mapping, json)

    def test_multiple_mappings(self):
        if False:
            i = 10
            return i + 15
        template = Template()
        template.add_mapping('map', {'k1': {'n1': 'v1'}})
        template.add_mapping('map', {'k2': {'n2': 'v2'}})
        json = template.to_json()
        self.assertEqual(multiple_mappings, json)
if __name__ == '__main__':
    unittest.main()