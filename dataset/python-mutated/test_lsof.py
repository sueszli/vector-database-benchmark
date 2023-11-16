import os
import json
import unittest
import jc.parsers.lsof
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/lsof.out'), 'r', encoding='utf-8') as f:
        centos_7_7_lsof = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/lsof.out'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_lsof = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/lsof-sudo.out'), 'r', encoding='utf-8') as f:
        centos_7_7_lsof_sudo = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/lsof-sudo.out'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_lsof_sudo = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/lsof.json'), 'r', encoding='utf-8') as f:
        centos_7_7_lsof_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/lsof.json'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_lsof_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/lsof-sudo.json'), 'r', encoding='utf-8') as f:
        centos_7_7_lsof_sudo_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/lsof-sudo.json'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_lsof_sudo_json = json.loads(f.read())

    def test_lsof_nodata(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'lsof' with no data\n        "
        self.assertEqual(jc.parsers.lsof.parse('', quiet=True), [])

    def test_lsof_centos_7_7(self):
        if False:
            return 10
        "\n        Test 'lsof' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.lsof.parse(self.centos_7_7_lsof, quiet=True), self.centos_7_7_lsof_json)

    def test_lsof_ubuntu_18_4(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'lsof' on Ubuntu 18.4\n        "
        self.assertEqual(jc.parsers.lsof.parse(self.ubuntu_18_4_lsof, quiet=True), self.ubuntu_18_4_lsof_json)

    def test_lsof_sudo_centos_7_7(self):
        if False:
            return 10
        "\n        Test 'sudo lsof' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.lsof.parse(self.centos_7_7_lsof_sudo, quiet=True), self.centos_7_7_lsof_sudo_json)

    def test_lsof_sudo_ubuntu_18_4(self):
        if False:
            print('Hello World!')
        "\n        Test 'sudo lsof' on Ubuntu 18.4\n        "
        self.assertEqual(jc.parsers.lsof.parse(self.ubuntu_18_4_lsof_sudo, quiet=True), self.ubuntu_18_4_lsof_sudo_json)
if __name__ == '__main__':
    unittest.main()