import os
import json
import unittest
import jc.parsers.lsmod
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/lsmod.out'), 'r', encoding='utf-8') as f:
        centos_7_7_lsmod = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/lsmod.out'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_lsmod = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/lsmod.json'), 'r', encoding='utf-8') as f:
        centos_7_7_lsmod_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/lsmod.json'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_lsmod_json = json.loads(f.read())

    def test_lsmod_nodata(self):
        if False:
            return 10
        "\n        Test 'lsmod' with no data\n        "
        self.assertEqual(jc.parsers.lsmod.parse('', quiet=True), [])

    def test_lsmod_centos_7_7(self):
        if False:
            return 10
        "\n        Test 'lsmod' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.lsmod.parse(self.centos_7_7_lsmod, quiet=True), self.centos_7_7_lsmod_json)

    def test_lsmod_ubuntu_18_4(self):
        if False:
            print('Hello World!')
        "\n        Test 'lsmod' on Ubuntu 18.4\n        "
        self.assertEqual(jc.parsers.lsmod.parse(self.ubuntu_18_4_lsmod, quiet=True), self.ubuntu_18_4_lsmod_json)
if __name__ == '__main__':
    unittest.main()