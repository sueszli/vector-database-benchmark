import os
import unittest
import json
import jc.parsers.chage
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/chage.out'), 'r', encoding='utf-8') as f:
        centos_7_7_chage = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/chage.json'), 'r', encoding='utf-8') as f:
        centos_7_7_chage_json = json.loads(f.read())

    def test_chage_nodata(self):
        if False:
            print('Hello World!')
        "\n        Test 'chage' with no data\n        "
        self.assertEqual(jc.parsers.chage.parse('', quiet=True), {})

    def test_chage_centos_7_7(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'chage' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.chage.parse(self.centos_7_7_chage, quiet=True), self.centos_7_7_chage_json)
if __name__ == '__main__':
    unittest.main()