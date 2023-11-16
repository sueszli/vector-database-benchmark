import os
import unittest
import json
from typing import Dict
from jc.parsers.openvpn import parse
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        fixtures = {'openvpn': ('fixtures/generic/openvpn-status.log', 'fixtures/generic/openvpn-status.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_openvpn_nodata(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'openvpn' with no data\n        "
        self.assertEqual(parse('', quiet=True), {})

    def test_openvpn(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'openvpn'\n        "
        self.assertEqual(parse(self.f_in['openvpn'], quiet=True), self.f_json['openvpn'])
if __name__ == '__main__':
    unittest.main()