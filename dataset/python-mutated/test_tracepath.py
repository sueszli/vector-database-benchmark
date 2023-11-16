import os
import unittest
import json
import jc.parsers.tracepath
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/tracepath.out'), 'r', encoding='utf-8') as f:
        centos_7_7_tracepath = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/tracepath6.out'), 'r', encoding='utf-8') as f:
        centos_7_7_tracepath6 = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/tracepath.json'), 'r', encoding='utf-8') as f:
        centos_7_7_tracepath_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/tracepath6.json'), 'r', encoding='utf-8') as f:
        centos_7_7_tracepath6_json = json.loads(f.read())

    def test_tracepath_nodata(self):
        if False:
            while True:
                i = 10
        "\n        Test 'tracepath' with no data\n        "
        self.assertEqual(jc.parsers.tracepath.parse('', quiet=True), {})

    def test_tracepath_centos_7_7(self):
        if False:
            while True:
                i = 10
        "\n        Test 'tracepath' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.tracepath.parse(self.centos_7_7_tracepath, quiet=True), self.centos_7_7_tracepath_json)

    def test_tracepath6_centos_7_7(self):
        if False:
            return 10
        "\n        Test 'tracepath6' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.tracepath.parse(self.centos_7_7_tracepath6, quiet=True), self.centos_7_7_tracepath6_json)
if __name__ == '__main__':
    unittest.main()