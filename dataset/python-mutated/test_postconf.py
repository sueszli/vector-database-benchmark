import os
import unittest
import json
import jc.parsers.postconf
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/postconf-M.out'), 'r', encoding='utf-8') as f:
        generic_postconf_m = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/postconf-M.json'), 'r', encoding='utf-8') as f:
        generic_postconf_m_json = json.loads(f.read())

    def test_postconf_nodata(self):
        if False:
            while True:
                i = 10
        "\n        Test 'postconf' with no data\n        "
        self.assertEqual(jc.parsers.postconf.parse('', quiet=True), [])

    def test_postconf(self):
        if False:
            print('Hello World!')
        "\n        Test 'postconf -M'\n        "
        self.assertEqual(jc.parsers.postconf.parse(self.generic_postconf_m, quiet=True), self.generic_postconf_m_json)
if __name__ == '__main__':
    unittest.main()