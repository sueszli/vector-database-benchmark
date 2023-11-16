import os
import unittest
import json
import jc.parsers.cef
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/cef.out'), 'r', encoding='utf-8') as f:
        cef = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/cef.json'), 'r', encoding='utf-8') as f:
        cef_json = json.loads(f.read())

    def test_cef_nodata(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'cef' with no data\n        "
        self.assertEqual(jc.parsers.cef.parse('', quiet=True), [])

    def test_cef_sample(self):
        if False:
            while True:
                i = 10
        '\n        Test with sample cef log\n        '
        self.assertEqual(jc.parsers.cef.parse(self.cef, quiet=True), self.cef_json)
if __name__ == '__main__':
    unittest.main()