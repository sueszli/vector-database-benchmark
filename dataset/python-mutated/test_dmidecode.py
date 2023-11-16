import os
import json
import unittest
import jc.parsers.dmidecode
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/dmidecode.out'), 'r', encoding='utf-8') as f:
        centos_7_7_dmidecode = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/dmidecode.out'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_dmidecode = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/fedora32/dmidecode.out'), 'r', encoding='utf-8') as f:
        fedora32_dmidecode = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/dmidecode.json'), 'r', encoding='utf-8') as f:
        centos_7_7_dmidecode_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/dmidecode.json'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_dmidecode_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/fedora32/dmidecode.json'), 'r', encoding='utf-8') as f:
        fedora32_dmidecode_json = json.loads(f.read())

    def test_dmidecode_nodata(self):
        if False:
            return 10
        "\n        Test 'dmidecode' with no data\n        "
        self.assertEqual(jc.parsers.dmidecode.parse('', quiet=True), [])

    def test_dmidecode_centos_7_7(self):
        if False:
            while True:
                i = 10
        "\n        Test 'dmidecode' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.dmidecode.parse(self.centos_7_7_dmidecode, quiet=True), self.centos_7_7_dmidecode_json)

    def test_dmidecode_ubuntu_18_4(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'dmidecode' on Ubuntu 18.4\n        "
        self.assertEqual(jc.parsers.dmidecode.parse(self.ubuntu_18_4_dmidecode, quiet=True), self.ubuntu_18_4_dmidecode_json)

    def test_dmidecode_fedora32(self):
        if False:
            while True:
                i = 10
        "\n        Test 'dmidecode' on Fedora 32\n        "
        self.assertEqual(jc.parsers.dmidecode.parse(self.fedora32_dmidecode, quiet=True), self.fedora32_dmidecode_json)
if __name__ == '__main__':
    unittest.main()