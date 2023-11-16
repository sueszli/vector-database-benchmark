import os
import unittest
import json
from typing import Dict
from jc.parsers.lsb_release import parse
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        fixtures = {'lsb_release_a': ('fixtures/generic/lsb_release-a.out', 'fixtures/generic/lsb_release-a.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_lsb_release_nodata(self):
        if False:
            print('Hello World!')
        "\n        Test 'lsb_release' with no data\n        "
        self.assertEqual(parse('', quiet=True), {})

    def test_lsb_release_a(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'lsb_release -a'\n        "
        self.assertEqual(parse(self.f_in['lsb_release_a'], quiet=True), self.f_json['lsb_release_a'])
if __name__ == '__main__':
    unittest.main()