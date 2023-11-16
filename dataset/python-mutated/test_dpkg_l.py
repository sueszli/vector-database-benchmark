import os
import json
import unittest
import jc.parsers.dpkg_l
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/dpkg-l.out'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_dpkg_l = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/dpkg-l-columns500.out'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_dpkg_l_columns500 = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/dpkg-l-codes.out'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_dpkg_l_codes = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/dpkg-l.json'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_dpkg_l_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/dpkg-l-columns500.json'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_dpkg_l_columns500_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/dpkg-l-codes.json'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_dpkg_l_codes_json = json.loads(f.read())

    def test_dpkg_l_nodata(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test plain 'dpkg_l' with no data\n        "
        self.assertEqual(jc.parsers.dpkg_l.parse('', quiet=True), [])

    def test_dpkg_l_ubuntu_18_4(self):
        if False:
            print('Hello World!')
        "\n        Test plain 'dpkg -l' on Ubuntu 18.4\n        "
        self.assertEqual(jc.parsers.dpkg_l.parse(self.ubuntu_18_4_dpkg_l, quiet=True), self.ubuntu_18_4_dpkg_l_json)

    def test_dpkg_l_columns500_ubuntu_18_4(self):
        if False:
            return 10
        "\n        Test 'dpkg -l' on Ubuntu 18.4 with COLUMNS=500 set\n        "
        self.assertEqual(jc.parsers.dpkg_l.parse(self.ubuntu_18_4_dpkg_l_columns500, quiet=True), self.ubuntu_18_4_dpkg_l_columns500_json)

    def test_dpkg_l_codes_ubuntu_18_4(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'dpkg -l' on Ubuntu 18.4 with multiple codes set\n        "
        self.assertEqual(jc.parsers.dpkg_l.parse(self.ubuntu_18_4_dpkg_l_codes, quiet=True), self.ubuntu_18_4_dpkg_l_codes_json)
if __name__ == '__main__':
    unittest.main()