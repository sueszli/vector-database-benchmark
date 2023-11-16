import os
import json
import unittest
import jc.parsers.ifconfig
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/ifconfig.out'), 'r', encoding='utf-8') as f:
        centos_7_7_ifconfig = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-16.04/ifconfig.out'), 'r', encoding='utf-8') as f:
        ubuntu_16_4_ifconfig = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/ifconfig.out'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_ifconfig = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.11.6/ifconfig.out'), 'r', encoding='utf-8') as f:
        osx_10_11_6_ifconfig = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.11.6/ifconfig2.out'), 'r', encoding='utf-8') as f:
        osx_10_11_6_ifconfig2 = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/ifconfig.out'), 'r', encoding='utf-8') as f:
        osx_10_14_6_ifconfig = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/ifconfig2.out'), 'r', encoding='utf-8') as f:
        osx_10_14_6_ifconfig2 = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/freebsd12/ifconfig-extra-fields.out'), 'r', encoding='utf-8') as f:
        osx_freebsd12_ifconfig_extra_fields = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/freebsd12/ifconfig-extra-fields2.out'), 'r', encoding='utf-8') as f:
        osx_freebsd12_ifconfig_extra_fields2 = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/freebsd12/ifconfig-extra-fields3.out'), 'r', encoding='utf-8') as f:
        osx_freebsd12_ifconfig_extra_fields3 = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/freebsd12/ifconfig-extra-fields4.out'), 'r', encoding='utf-8') as f:
        osx_freebsd12_ifconfig_extra_fields4 = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/ifconfig.json'), 'r', encoding='utf-8') as f:
        centos_7_7_ifconfig_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-16.04/ifconfig.json'), 'r', encoding='utf-8') as f:
        ubuntu_16_4_ifconfig_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/ifconfig.json'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_ifconfig_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.11.6/ifconfig.json'), 'r', encoding='utf-8') as f:
        osx_10_11_6_ifconfig_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.11.6/ifconfig2.json'), 'r', encoding='utf-8') as f:
        osx_10_11_6_ifconfig2_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/ifconfig.json'), 'r', encoding='utf-8') as f:
        osx_10_14_6_ifconfig_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/ifconfig2.json'), 'r', encoding='utf-8') as f:
        osx_10_14_6_ifconfig2_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/freebsd12/ifconfig-extra-fields.json'), 'r', encoding='utf-8') as f:
        freebsd12_ifconfig_extra_fields_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/freebsd12/ifconfig-extra-fields2.json'), 'r', encoding='utf-8') as f:
        freebsd12_ifconfig_extra_fields2_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/freebsd12/ifconfig-extra-fields3.json'), 'r', encoding='utf-8') as f:
        freebsd12_ifconfig_extra_fields3_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/freebsd12/ifconfig-extra-fields4.json'), 'r', encoding='utf-8') as f:
        freebsd12_ifconfig_extra_fields4_json = json.loads(f.read())

    def test_ifconfig_nodata(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'ifconfig' with no data\n        "
        self.assertEqual(jc.parsers.ifconfig.parse('', quiet=True), [])

    def test_ifconfig_centos_7_7(self):
        if False:
            return 10
        "\n        Test 'ifconfig' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.ifconfig.parse(self.centos_7_7_ifconfig, quiet=True), self.centos_7_7_ifconfig_json)

    def test_ifconfig_ubuntu_16_4(self):
        if False:
            while True:
                i = 10
        "\n        Test 'ifconfig' on Ubuntu 16.4\n        "
        self.assertEqual(jc.parsers.ifconfig.parse(self.ubuntu_16_4_ifconfig, quiet=True), self.ubuntu_16_4_ifconfig_json)

    def test_ifconfig_ubuntu_18_4(self):
        if False:
            while True:
                i = 10
        "\n        Test 'ifconfig' on Ubuntu 18.4\n        "
        self.assertEqual(jc.parsers.ifconfig.parse(self.ubuntu_18_4_ifconfig, quiet=True), self.ubuntu_18_4_ifconfig_json)

    def test_ifconfig_osx_10_11_6(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'ifconfig' on OSX 10.11.6\n        "
        self.assertEqual(jc.parsers.ifconfig.parse(self.osx_10_11_6_ifconfig, quiet=True), self.osx_10_11_6_ifconfig_json)

    def test_ifconfig_osx_10_11_6_2(self):
        if False:
            print('Hello World!')
        "\n        Test 'ifconfig' on OSX 10.11.6\n        "
        self.assertEqual(jc.parsers.ifconfig.parse(self.osx_10_11_6_ifconfig2, quiet=True), self.osx_10_11_6_ifconfig2_json)

    def test_ifconfig_osx_10_14_6(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'ifconfig' on OSX 10.14.6\n        "
        self.assertEqual(jc.parsers.ifconfig.parse(self.osx_10_14_6_ifconfig, quiet=True), self.osx_10_14_6_ifconfig_json)

    def test_ifconfig_osx_10_14_6_2(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'ifconfig' on OSX 10.14.6\n        "
        self.assertEqual(jc.parsers.ifconfig.parse(self.osx_10_14_6_ifconfig2, quiet=True), self.osx_10_14_6_ifconfig2_json)

    def test_ifconfig_freebsd_extra_fields(self):
        if False:
            return 10
        "\n        Test 'ifconfig' on freebsd12\n        "
        self.assertEqual(jc.parsers.ifconfig.parse(self.osx_freebsd12_ifconfig_extra_fields, quiet=True), self.freebsd12_ifconfig_extra_fields_json)

    def test_ifconfig_freebsd_extra_fields2(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'ifconfig' on freebsd12 with other fields\n        "
        self.assertEqual(jc.parsers.ifconfig.parse(self.osx_freebsd12_ifconfig_extra_fields2, quiet=True), self.freebsd12_ifconfig_extra_fields2_json)

    def test_ifconfig_freebsd_extra_fields3(self):
        if False:
            while True:
                i = 10
        "\n        Test 'ifconfig' on freebsd12 with other extra fields\n        "
        self.assertEqual(jc.parsers.ifconfig.parse(self.osx_freebsd12_ifconfig_extra_fields3, quiet=True), self.freebsd12_ifconfig_extra_fields3_json)

    def test_ifconfig_freebsd_extra_fields4(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'ifconfig' on freebsd12 with lane fields\n        "
        self.assertEqual(jc.parsers.ifconfig.parse(self.osx_freebsd12_ifconfig_extra_fields4, quiet=True), self.freebsd12_ifconfig_extra_fields4_json)
if __name__ == '__main__':
    unittest.main()