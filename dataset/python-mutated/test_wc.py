import os
import unittest
import json
import jc.parsers.wc
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/wc.out'), 'r', encoding='utf-8') as f:
        centos_7_7_wc = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/wc.out'), 'r', encoding='utf-8') as f:
        osx_10_14_6_wc = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/wc-stdin.out'), 'r', encoding='utf-8') as f:
        osx_10_14_6_wc_stdin = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/wc.json'), 'r', encoding='utf-8') as f:
        centos_7_7_wc_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/wc.json'), 'r', encoding='utf-8') as f:
        osx_10_14_6_wc_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/wc-stdin.json'), 'r', encoding='utf-8') as f:
        osx_10_14_6_wc_stdin_json = json.loads(f.read())

    def test_wc_nodata(self):
        if False:
            return 10
        "\n        Test 'wc' parser with no data\n        "
        self.assertEqual(jc.parsers.wc.parse('', quiet=True), [])

    def test_wc_centos_7_7(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'wc' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.wc.parse(self.centos_7_7_wc, quiet=True), self.centos_7_7_wc_json)

    def test_wc_osx_10_14_6(self):
        if False:
            return 10
        "\n        Test 'wc' on OSX 10.14.6\n        "
        self.assertEqual(jc.parsers.wc.parse(self.osx_10_14_6_wc, quiet=True), self.osx_10_14_6_wc_json)

    def test_wc_stdin_osx_10_14_6(self):
        if False:
            print('Hello World!')
        "\n        Test 'wc' from `STDIN` on OSX 10.14.6\n        "
        self.assertEqual(jc.parsers.wc.parse(self.osx_10_14_6_wc_stdin, quiet=True), self.osx_10_14_6_wc_stdin_json)
if __name__ == '__main__':
    unittest.main()