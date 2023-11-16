import os
import unittest
import json
from typing import Dict
from jc.parsers.zpool_iostat import parse
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        fixtures = {'zpool_iostat': ('fixtures/generic/zpool-iostat.out', 'fixtures/generic/zpool-iostat.json'), 'zpool_iostat_v': ('fixtures/generic/zpool-iostat-v.out', 'fixtures/generic/zpool-iostat-v.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_zpool_iostat_nodata(self):
        if False:
            while True:
                i = 10
        "\n        Test 'zpool iostat' with no data\n        "
        self.assertEqual(parse('', quiet=True), [])

    def test_zpool_iostat(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'zpool iostat'\n        "
        self.assertEqual(parse(self.f_in['zpool_iostat'], quiet=True), self.f_json['zpool_iostat'])

    def test_zpool_iostat_v(self):
        if False:
            print('Hello World!')
        "\n        Test 'zpool iostat -v'\n        "
        self.assertEqual(parse(self.f_in['zpool_iostat_v'], quiet=True), self.f_json['zpool_iostat_v'])
if __name__ == '__main__':
    unittest.main()