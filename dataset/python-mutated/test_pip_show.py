import os
import json
import unittest
import jc.parsers.pip_show
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/pip-show.out'), 'r', encoding='utf-8') as f:
        centos_7_7_pip_show = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/pip-show.out'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_pip_show = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.11.6/pip-show.out'), 'r', encoding='utf-8') as f:
        osx_10_11_6_pip_show = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/pip-show.out'), 'r', encoding='utf-8') as f:
        osx_10_14_6_pip_show = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/pip-show-multiline-license.out'), 'r', encoding='utf-8') as f:
        generic_pip_show_multiline_license = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/pip-show.json'), 'r', encoding='utf-8') as f:
        centos_7_7_pip_show_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/pip-show.json'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_pip_show_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.11.6/pip-show.json'), 'r', encoding='utf-8') as f:
        osx_10_11_6_pip_show_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/pip-show.json'), 'r', encoding='utf-8') as f:
        osx_10_14_6_pip_show_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/pip-show-multiline-license.json'), 'r', encoding='utf-8') as f:
        generic_pip_show_multiline_license_json = json.loads(f.read())

    def test_pip_show_nodata(self):
        if False:
            while True:
                i = 10
        "\n        Test 'pip show' with no data\n        "
        self.assertEqual(jc.parsers.pip_show.parse('', quiet=True), [])

    def test_pip_show_centos_7_7(self):
        if False:
            return 10
        "\n        Test 'pip show' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.pip_show.parse(self.centos_7_7_pip_show, quiet=True), self.centos_7_7_pip_show_json)

    def test_pip_show_ubuntu_18_4(self):
        if False:
            while True:
                i = 10
        "\n        Test 'pip show' on Ubuntu 18.4\n        "
        self.assertEqual(jc.parsers.pip_show.parse(self.ubuntu_18_4_pip_show, quiet=True), self.ubuntu_18_4_pip_show_json)

    def test_pip_show_osx_10_11_6(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'pip show' on OSX 10.11.6\n        "
        self.assertEqual(jc.parsers.pip_show.parse(self.osx_10_11_6_pip_show, quiet=True), self.osx_10_11_6_pip_show_json)

    def test_pip_show_osx_10_14_6(self):
        if False:
            return 10
        "\n        Test 'pip show' on OSX 10.14.6\n        "
        self.assertEqual(jc.parsers.pip_show.parse(self.osx_10_14_6_pip_show, quiet=True), self.osx_10_14_6_pip_show_json)

    def test_pip_show_multiline_license(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'pip show' with a multiline license\n        "
        self.assertEqual(jc.parsers.pip_show.parse(self.generic_pip_show_multiline_license, quiet=True), self.generic_pip_show_multiline_license_json)
if __name__ == '__main__':
    unittest.main()