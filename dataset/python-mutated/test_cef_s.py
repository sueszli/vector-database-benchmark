import os
import json
import unittest
import jc.parsers.cef_s
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/cef.out'), 'r', encoding='utf-8') as f:
        cef = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/cef-streaming.json'), 'r', encoding='utf-8') as f:
        cef_streaming_json = json.loads(f.read())

    def test_cef_s_nodata(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'cef' with no data\n        "
        self.assertEqual(list(jc.parsers.cef_s.parse([], quiet=True)), [])

    def test_cef_s_sample(self):
        if False:
            return 10
        '\n        Test with sample cef log\n        '
        self.assertEqual(list(jc.parsers.cef_s.parse(self.cef.splitlines(), quiet=True)), self.cef_streaming_json)
if __name__ == '__main__':
    unittest.main()