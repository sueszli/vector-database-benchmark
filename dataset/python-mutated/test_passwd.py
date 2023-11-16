import os
import json
import unittest
import jc.parsers.passwd
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/passwd.out'), 'r', encoding='utf-8') as f:
        centos_7_7_passwd = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/passwd.out'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_passwd = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/passwd.out'), 'r', encoding='utf-8') as f:
        osx_10_14_6_passwd = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/passwd.json'), 'r', encoding='utf-8') as f:
        centos_7_7_passwd_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/passwd.json'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_passwd_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/passwd.json'), 'r', encoding='utf-8') as f:
        osx_10_14_6_passwd_json = json.loads(f.read())

    def test_passwd_nodata(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'cat /etc/passwd' with no data\n        "
        self.assertEqual(jc.parsers.passwd.parse('', quiet=True), [])

    def test_passwd_centos_7_7(self):
        if False:
            while True:
                i = 10
        "\n        Test 'cat /etc/passwd' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.passwd.parse(self.centos_7_7_passwd, quiet=True), self.centos_7_7_passwd_json)

    def test_passwd_ubuntu_18_4(self):
        if False:
            return 10
        "\n        Test 'cat /etc/passwd' on Ubuntu 18.4\n        "
        self.assertEqual(jc.parsers.passwd.parse(self.ubuntu_18_4_passwd, quiet=True), self.ubuntu_18_4_passwd_json)

    def test_passwd_osx_10_14_6(self):
        if False:
            return 10
        "\n        Test 'cat /etc/passwd' on OSX 10.14.6\n        "
        self.assertEqual(jc.parsers.passwd.parse(self.osx_10_14_6_passwd, quiet=True), self.osx_10_14_6_passwd_json)
if __name__ == '__main__':
    unittest.main()