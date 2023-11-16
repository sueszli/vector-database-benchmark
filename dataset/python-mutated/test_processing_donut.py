import unittest
from transformers import DonutProcessor
DONUT_PRETRAINED_MODEL_NAME = 'naver-clova-ix/donut-base'

class DonutProcessorTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.processor = DonutProcessor.from_pretrained(DONUT_PRETRAINED_MODEL_NAME)

    def test_token2json(self):
        if False:
            print('Hello World!')
        expected_json = {'name': 'John Doe', 'age': '99', 'city': 'Atlanta', 'state': 'GA', 'zip': '30301', 'phone': '123-4567', 'nicknames': [{'nickname': 'Johnny'}, {'nickname': 'JD'}]}
        sequence = '<s_name>John Doe</s_name><s_age>99</s_age><s_city>Atlanta</s_city><s_state>GA</s_state><s_zip>30301</s_zip><s_phone>123-4567</s_phone><s_nicknames><s_nickname>Johnny</s_nickname><sep/><s_nickname>JD</s_nickname></s_nicknames>'
        actual_json = self.processor.token2json(sequence)
        self.assertDictEqual(actual_json, expected_json)