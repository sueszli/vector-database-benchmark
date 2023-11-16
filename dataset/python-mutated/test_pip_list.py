import os
import json
import unittest
import jc.parsers.pip_list
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/pip-list.out'), 'r', encoding='utf-8') as f:
        centos_7_7_pip_list = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/pip-list.out'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_pip_list = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/pip-list-legacy.out'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_pip_list_legacy = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.11.6/pip-list.out'), 'r', encoding='utf-8') as f:
        osx_10_11_6_pip_list = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/pip-list.out'), 'r', encoding='utf-8') as f:
        osx_10_14_6_pip_list = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/pip-list.json'), 'r', encoding='utf-8') as f:
        centos_7_7_pip_list_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/pip-list.json'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_pip_list_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/pip-list-legacy.json'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_pip_list_legacy_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.11.6/pip-list.json'), 'r', encoding='utf-8') as f:
        osx_10_11_6_pip_list_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/pip-list.json'), 'r', encoding='utf-8') as f:
        osx_10_14_6_pip_list_json = json.loads(f.read())

    def test_pip_list_nodata(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'pip_list' with no data\n        "
        self.assertEqual(jc.parsers.pip_list.parse('', quiet=True), [])

    def test_pip_list_centos_7_7(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'pip_list' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.pip_list.parse(self.centos_7_7_pip_list, quiet=True), self.centos_7_7_pip_list_json)

    def test_pip_list_ubuntu_18_4(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'pip_list' on Ubuntu 18.4\n        "
        self.assertEqual(jc.parsers.pip_list.parse(self.ubuntu_18_4_pip_list, quiet=True), self.ubuntu_18_4_pip_list_json)

    def test_pip_list_legacy_ubuntu_18_4(self):
        if False:
            print('Hello World!')
        "\n        Test 'pip_list' with legacy output on Ubuntu 18.4\n        "
        self.assertEqual(jc.parsers.pip_list.parse(self.ubuntu_18_4_pip_list_legacy, quiet=True), self.ubuntu_18_4_pip_list_legacy_json)

    def test_pip_list_osx_10_11_6(self):
        if False:
            while True:
                i = 10
        "\n        Test 'pip_list' on OSX 10.11.6\n        "
        self.assertEqual(jc.parsers.pip_list.parse(self.osx_10_11_6_pip_list, quiet=True), self.osx_10_11_6_pip_list_json)

    def test_pip_list_osx_10_14_6(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'pip_list' on OSX 10.14.6\n        "
        self.assertEqual(jc.parsers.pip_list.parse(self.osx_10_14_6_pip_list, quiet=True), self.osx_10_14_6_pip_list_json)
if __name__ == '__main__':
    unittest.main()