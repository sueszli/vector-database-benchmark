import os
import json
import unittest
import jc.parsers.hciconfig
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/hciconfig.out'), 'r', encoding='utf-8') as f:
        centos_7_7_hciconfig = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/hciconfig-a.out'), 'r', encoding='utf-8') as f:
        centos_7_7_hciconfig_a = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-20.04/hciconfig.out'), 'r', encoding='utf-8') as f:
        ubuntu_20_4_hciconfig = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-20.04/hciconfig-a.out'), 'r', encoding='utf-8') as f:
        ubuntu_20_4_hciconfig_a = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/hciconfig.json'), 'r', encoding='utf-8') as f:
        centos_7_7_hciconfig_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/hciconfig-a.json'), 'r', encoding='utf-8') as f:
        centos_7_7_hciconfig_a_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-20.04/hciconfig.json'), 'r', encoding='utf-8') as f:
        ubuntu_20_4_hciconfig_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-20.04/hciconfig-a.json'), 'r', encoding='utf-8') as f:
        ubuntu_20_4_hciconfig_a_json = json.loads(f.read())

    def test_hciconfig_nodata(self):
        if False:
            i = 10
            return i + 15
        "\n        Test plain 'hciconfig' with no data\n        "
        self.assertEqual(jc.parsers.hciconfig.parse('', quiet=True), [])

    def test_hciconfig_centos_7_7(self):
        if False:
            return 10
        "\n        Test plain 'hciconfig' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.hciconfig.parse(self.centos_7_7_hciconfig, quiet=True), self.centos_7_7_hciconfig_json)

    def test_hciconfig_a_centos_7_7(self):
        if False:
            while True:
                i = 10
        "\n        Test 'hciconfig -a' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.hciconfig.parse(self.centos_7_7_hciconfig_a, quiet=True), self.centos_7_7_hciconfig_a_json)

    def test_hciconfig_ubuntu_18_4(self):
        if False:
            i = 10
            return i + 15
        "\n        Test plain 'hciconfig' on Ubuntu 20.4\n        "
        self.assertEqual(jc.parsers.hciconfig.parse(self.ubuntu_20_4_hciconfig, quiet=True), self.ubuntu_20_4_hciconfig_json)

    def test_hciconfig_a_ubuntu_18_4(self):
        if False:
            return 10
        "\n        Test 'hciconfig -a' on Ubuntu 20.4\n        "
        self.assertEqual(jc.parsers.hciconfig.parse(self.ubuntu_20_4_hciconfig_a, quiet=True), self.ubuntu_20_4_hciconfig_a_json)
if __name__ == '__main__':
    unittest.main()