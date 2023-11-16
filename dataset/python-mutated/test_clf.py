import os
import unittest
import json
from typing import Dict
from jc.parsers.clf import parse
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        fixtures = {'clf': ('fixtures/generic/common-log-format.log', 'fixtures/generic/common-log-format.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_clf_nodata(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'clf' with no data\n        "
        self.assertEqual(parse('', quiet=True), [])

    def test_clf(self):
        if False:
            while True:
                i = 10
        "\n        Test 'clf' with various log lines\n        "
        self.assertEqual(parse(self.f_in['clf'], quiet=True), self.f_json['clf'])
if __name__ == '__main__':
    unittest.main()