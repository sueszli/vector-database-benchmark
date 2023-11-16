import os
import unittest
import json
from typing import Dict
from jc.parsers.certbot import parse
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        fixtures = {'account': ('fixtures/generic/certbot-account.out', 'fixtures/generic/certbot-account.json'), 'certificates': ('fixtures/generic/certbot-certs.out', 'fixtures/generic/certbot-certs.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_certbot_nodata(self):
        if False:
            return 10
        "\n        Test 'certbot' with no data\n        "
        self.assertEqual(parse('', quiet=True), {})

    def test_certbot_certificates(self):
        if False:
            while True:
                i = 10
        "\n        Test 'certbot certificates'\n        "
        self.assertEqual(parse(self.f_in['certificates'], quiet=True), self.f_json['certificates'])

    def test_certbot_account(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'certbot account'\n        "
        self.assertEqual(parse(self.f_in['account'], quiet=True), self.f_json['account'])
if __name__ == '__main__':
    unittest.main()