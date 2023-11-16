import os
import unittest
import json
import jc.parsers.iw_scan
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/iw-scan0.out'), 'r', encoding='utf-8') as f:
        centos_7_7_iw_scan0 = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/iw-scan1.out'), 'r', encoding='utf-8') as f:
        centos_7_7_iw_scan1 = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/iw-scan0.json'), 'r', encoding='utf-8') as f:
        centos_7_7_iw_scan0_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/iw-scan1.json'), 'r', encoding='utf-8') as f:
        centos_7_7_iw_scan1_json = json.loads(f.read())

    def test_iw_scan_nodata(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'iw_scan' parser with no data\n        "
        self.assertEqual(jc.parsers.iw_scan.parse('', quiet=True), [])

    def test_iw_scan0_centos_7_7(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'iw_scan' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.iw_scan.parse(self.centos_7_7_iw_scan0, quiet=True), self.centos_7_7_iw_scan0_json)

    def test_iw_scan1_centos_7_7(self):
        if False:
            print('Hello World!')
        "\n        Test 'iw_scan' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.iw_scan.parse(self.centos_7_7_iw_scan1, quiet=True), self.centos_7_7_iw_scan1_json)
if __name__ == '__main__':
    unittest.main()